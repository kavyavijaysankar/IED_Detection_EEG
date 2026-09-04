"""Run a saved IED detection model on an EDF recordings. 
Prints detected IED events above the score threshold: time (s), representative channel, score, and how many channels the event spanned.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from detect_data import CH19_ELECTRODES, load_recording_path
from detection import DetectionPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', help='saved pipeline, e.g. detect_model.joblib')
    ap.add_argument('edf', help='EDF recording to scan')
    ap.add_argument('--score', type=float, default=0.766,
                    help='min IED score to report (default 0.766, the pre-registered <=10 FP/min point)')
    args = ap.parse_args()

    pipe = DetectionPipeline.load(args.model)
    X, bads = load_recording_path(args.edf, pipe.cfg)
    events = sorted((e for e in pipe.predict(X, bads['hard']) if e['score'] >= args.score),
                    key=lambda e: e['time'])

    name = lambda idx: ', '.join(CH19_ELECTRODES[i] for i in idx)
    if bads['soft']:
        print(f"  note: {name(bads['soft'])} excluded from the average reference (noisy, kept in analysis)")
    if bads['hard']:
        print(f"  note: {name(bads['hard'])} interpolated and excluded from detection")
    if bads['over_cap']:
        print(f"  WARNING: more than {pipe.cfg.max_interpolate} bad channels — none interpolated, all "
              f"excluded from the average only. Treat this recording's numbers with caution.")
    print(f"{args.edf}: {len(events)} detection(s) at score >= {args.score}")
    for e in events:
        print(f"  t={e['time'] / pipe.cfg.sfreq:6.2f}s  ch={CH19_ELECTRODES[e['channel']]:5s}  "
              f"score={e['score']:.3f}  ({e['n_channels']} channels)")


if __name__ == '__main__':
    main()
