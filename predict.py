"""Run a saved IED detection model on an EDF recording (supervisor / external-data entrypoint).

    python predict.py <model.joblib> <recording.edf> [--score 0.5]

Prints detected IED events above the score threshold: time (s), representative channel, score, and how
many channels the event spanned. The EDF must contain the 19 standard 10-20 channels named as in the
Kural data (`E Fp1-Ref`, ...) at 500 Hz — see detect_data.load_recording_path.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from detect_data import CH19, SF, load_recording_path
from detection import DetectionPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', help='saved pipeline, e.g. detect_model.joblib')
    ap.add_argument('edf', help='EDF recording to scan')
    ap.add_argument('--score', type=float, default=0.5, help='min IED score to report (default 0.5)')
    args = ap.parse_args()

    pipe = DetectionPipeline.load(args.model)
    X = load_recording_path(args.edf)
    events = sorted((e for e in pipe.predict(X) if e['score'] >= args.score), key=lambda e: e['time'])

    print(f"{args.edf}: {len(events)} detection(s) at score >= {args.score}")
    for e in events:
        print(f"  t={e['time'] / SF:6.2f}s  ch={CH19[e['channel']]:12s}  "
              f"score={e['score']:.3f}  ({e['n_channels']} channels)")


if __name__ == '__main__':
    main()
