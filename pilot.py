import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np

warnings.filterwarnings('ignore', message='Invalid measurement date')

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from detect_data import CH19_ELECTRODES, electrode_positions, load_recording_path, match_channels
from detection import DetectionPipeline, _smooth, member_sign, member_ptp, spatial_features

THRESHOLD = 0.766   # pre-registered <=10 FP/min operating point from grouped CV

REC_COLS = ['file', 'folder', 'duration_s', 'input_sfreq', 'channels_matched', 'error',
            'n_candidates', 'n_events', 'n_above_thr', 'threshold',
            'q90', 'q95', 'q99', 'n_above_0.5', 'n_above_0.7', 'n_above_0.9',
            'background_uv', 'p99_abs_uv', 'bad_soft', 'bad_hard', 'over_cap', 'mad_ratio']
EVENT_COLS = ['file', 'folder', 'time_s', 'channel', 'members', 'n_channels', 'score', 'l3_score',
              'dipole', 'polarity', 'prominence', 'l1_sharpness']


def header(path):
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    return raw.info['sfreq'], raw.n_times / raw.info['sfreq'], raw.ch_names


def edfs(root):
    """Every .edf under `root`, recursively, as (path, sub-folder relative to root)."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() == '.edf':
            rel = p.parent.relative_to(root).as_posix()
            out.append((p, '' if rel == '.' else rel))
    return out


def check(files):
    print(f"{'file':<28} {'sfreq':>7} {'dur_s':>8}  montage")
    ok = 0
    for path, _ in files:
        try:
            sfreq, dur, names = header(path)
        except Exception as ex:
            print(f"{path.name:<28} {'-':>7} {'-':>8}  UNREADABLE: {ex}")
            continue
        try:
            match_channels(names)
            note, good = 'OK (19 matched)', True
        except ValueError as ex:
            note, good = f"FAIL: {ex}", False
        ok += good
        print(f"{path.name:<28} {sfreq:>7.0f} {dur:>8.1f}  {note}")
    print(f"\n{ok}/{len(files)} files ready. Sample rate is resampled automatically; only the montage "
          f"has to match.")


def scan(pipe, path, folder, threshold):
    """One recording -> (recording row, list of event rows)."""
    t0 = time.time()
    sfreq_in, dur_in, _ = header(path)
    X, bads = load_recording_path(path, pipe.cfg)
    cands, stat = pipe.stage1.detect(X, bads['hard'])
    events = pipe.predict(X, bads['hard'])

    Xs, pos = _smooth(X, pipe.cfg), electrode_positions()
    rows = []
    for e in events:
        # electrode@time:peak-to-peak:sign per member. The sign is what makes the field's dipole
        # structure recoverable offline — L2 groups on |correlation| and would otherwise discard it.
        ms = sorted(zip(e['members'], member_ptp(e['members'], Xs, pipe.cfg),
                        member_sign(e['members'], Xs, pipe.cfg)), key=lambda m: m[0][1])
        members = [f"{CH19_ELECTRODES[c]}@{t / pipe.cfg.sfreq:.3f}:{p:.1f}:{s:+.0f}"
                   for (c, t), p, s in ms]
        rows.append({'file': path.name, 'folder': folder,
                     'time_s': round(e['time'] / pipe.cfg.sfreq, 4),
                     'channel': CH19_ELECTRODES[e['channel']], 'members': ';'.join(members),
                     'n_channels': e['n_channels'], 'score': round(e['score'], 6),
                     'l3_score': round(e['l3_score'], 6),
                     'dipole': round(spatial_features(e, Xs, pipe.cfg, pos)['dipole'], 4),
                     'polarity': e['polarity'], 'prominence': round(e['prominence'], 4),
                     'l1_sharpness': round(float(stat[e['channel'], e['time']]), 4)})

    s = np.array([e['score'] for e in events])
    q = lambda p: round(float(np.quantile(s, p)), 6) if len(s) else ''
    name = lambda idx: ';'.join(CH19_ELECTRODES[i] for i in idx)
    rec = {'file': path.name, 'folder': folder, 'duration_s': round(X.shape[1] / pipe.cfg.sfreq, 3),
           'input_sfreq': sfreq_in, 'channels_matched': 19, 'error': '',
           'n_candidates': len(cands), 'n_events': len(events),
           'n_above_thr': int((s >= threshold).sum()), 'threshold': threshold,
           'q90': q(0.90), 'q95': q(0.95), 'q99': q(0.99),
           'n_above_0.5': int((s >= 0.5).sum()), 'n_above_0.7': int((s >= 0.7).sum()),
           'n_above_0.9': int((s >= 0.9).sum()),
           'background_uv': round(DetectionPipeline.background(X), 4),
           'p99_abs_uv': round(float(np.percentile(np.abs(X), 99)), 4),
           'bad_soft': name(bads['soft']), 'bad_hard': name(bads['hard']),
           'over_cap': int(bads['over_cap']),
           'mad_ratio': ';'.join(f"{r:.3f}" for r in bads['mad_ratio'])}
    print(f"  {path.name:<28} {rec['duration_s']:>7.1f}s  {len(cands):>6} cand  {len(events):>5} ev  "
          f"{rec['n_above_thr']:>4} >={threshold}  {time.time() - t0:>5.1f}s")
    return rec, rows


def ask_folder():
    """Prompt for a folder, tolerating pasted quotes (Windows 'Copy as path') and escaped spaces (macOS)."""
    while True:
        raw = input("Folder with the EDF recordings: ").strip()
        if not raw:
            continue
        p = Path(raw.strip('"').strip("'").replace('\\ ', ' ')).expanduser()
        if p.is_dir():
            return p
        print(f"  not a folder: {p}")


def run(pipe, files, threshold, out, resume=False):
    out.mkdir(parents=True, exist_ok=True)
    rec_p, ev_p = out / 'pilot_recordings.csv', out / 'pilot_events.csv'
    done = set()
    if resume and rec_p.exists():
        done = {(r['folder'], r['file']) for r in csv.DictReader(open(rec_p))}
        files = [(p, f) for p, f in files if (f, p.name) not in done]
        print(f"resuming: {len(done)} recordings already written, {len(files)} to go\n")
    mode = 'a' if done else 'w'
    t0, n_fail = time.time(), 0
    with open(rec_p, mode, newline='') as rf, open(ev_p, mode, newline='') as ef:
        rw, ew = csv.DictWriter(rf, REC_COLS), csv.DictWriter(ef, EVENT_COLS)
        if mode == 'w':
            rw.writeheader(); ew.writeheader()
        for path, folder in files:
            try:
                rec, rows = scan(pipe, path, folder, threshold)
            except Exception as ex:
                rec, rows, n_fail = {c: '' for c in REC_COLS}, [], n_fail + 1
                rec.update(file=path.name, folder=folder, channels_matched=0,
                           error=f"{type(ex).__name__}: {ex}")
                print(f"  {path.name:<28} FAILED: {rec['error']}")
            rw.writerow(rec)
            ew.writerows(rows)
            rf.flush(); ef.flush()
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min, {n_fail} failed. Wrote:"
          f"\n  {rec_p.resolve()}\n  {ev_p.resolve()}")


def selfcheck(model):
    """End-to-end check on Kural recordings in sub-folders, plus one deliberately corrupt file."""
    import shutil
    import tempfile
    src = [p for p, _ in edfs('Kural_Dataset/Recordings')][:3]
    assert src, "self-check needs Kural_Dataset/Recordings"
    tmp = Path(tempfile.mkdtemp())
    data = tmp / 'data'
    (data / 'a').mkdir(parents=True); (data / 'b').mkdir()
    for p in src[:2]:
        shutil.copy(p, data / 'a' / p.name)
    shutil.copy(src[2], data / 'b' / src[2].name)
    (data / 'b' / 'corrupt.edf').write_text('not an EDF at all')

    files, thr = edfs(data), 0.8
    assert [f for _, f in files] == ['a', 'a', 'b', 'b'], f"recursion/folder labels wrong: {files}"
    run(DetectionPipeline.load(model), files, thr, tmp)

    n_before = len(list(csv.DictReader(open(tmp / 'pilot_recordings.csv'))))
    run(DetectionPipeline.load(model), files, thr, tmp, resume=True)   # must be a no-op
    recs = list(csv.DictReader(open(tmp / 'pilot_recordings.csv')))
    assert len(recs) == n_before, f'resume duplicated rows: {n_before} -> {len(recs)}'
    evs = list(csv.DictReader(open(tmp / 'pilot_events.csv')))
    assert [r['file'] for r in recs] == [p.name for p, _ in files], "one row per file, in order"
    bad = [r for r in recs if r['error']]
    assert len(bad) == 1 and bad[0]['file'] == 'corrupt.edf' and bad[0]['channels_matched'] == '0', \
        "a corrupt file must be recorded as an error row, not kill the run"
    good = [r for r in recs if not r['error']]
    assert sum(int(r['n_events']) for r in good) == len(evs), "n_events must match the event rows"
    for r in good:
        s = np.array([float(e['score']) for e in evs if e['file'] == r['file']])
        assert int(r['n_above_thr']) == int((s >= thr).sum()), f"{r['file']}: n_above_thr disagrees"
        assert abs(float(r['q99']) - np.quantile(s, 0.99)) < 1e-5, f"{r['file']}: q99 disagrees"
        assert len(r['mad_ratio'].split(';')) == 19, "mad_ratio must carry all 19 channels"
        assert float(r['duration_s']) > 0 and float(r['background_uv']) > 0
    for e in evs:
        assert e['folder'] in ('a', 'b'), f"bad folder label {e['folder']}"
        assert e['channel'] in CH19_ELECTRODES, f"bad electrode label {e['channel']}"
        assert e['polarity'] in ('-1', '1', '0'), f"bad polarity {e['polarity']}"
        assert -1.0 <= float(e['dipole']) <= 1.0, f"dipole out of range: {e['dipole']}"
        parts = e['members'].split(';')
        assert len(parts) >= int(e['n_channels']), "members must cover at least n_channels"
        for m in parts:
            el, _, rest = m.partition('@')
            assert el in CH19_ELECTRODES and len(rest.split(':')) == 3, f"malformed member {m}"
    shutil.rmtree(tmp)
    print(f"\npilot selfcheck OK  |  {len(good)} scanned, 1 error row, {len(evs)} events")


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', help='folder of EDF recordings, searched recursively')
    ap.add_argument('--model', help='saved model (default: detect_model.joblib beside this script)')
    ap.add_argument('--threshold', type=float, default=THRESHOLD, help=f'score threshold ({THRESHOLD})')
    ap.add_argument('--out', help='output folder (default: results beside this script)')
    ap.add_argument('--check', action='store_true', help='validate headers/montage only, then stop')
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted run, skipping recordings already in the CSV')
    ap.add_argument('--selfcheck', action='store_true', help='end-to-end test on the Kural recordings')
    args = ap.parse_args()

    model = Path(args.model) if args.model else here / 'detect_model.joblib'
    if args.selfcheck:
        return selfcheck(model)

    prompted = not args.dir
    if prompted:
        print("IED detection pilot.\n")
    files = edfs(args.dir if args.dir else ask_folder())
    if not files:
        return print("no .edf files found in that folder.")

    pipe = DetectionPipeline.load(model)     # before --check too, so the check covers the unpickle
    print(f"\nmodel {model.name} loaded OK ({pipe.cfg.sfreq:.0f} Hz, {pipe.cfg.reference} reference)\n")
    check(files)
    if args.check:
        return
    print(f"\n{len(files)} recordings, threshold {args.threshold}\n")
    run(pipe, files, args.threshold, Path(args.out) if args.out else here / 'results', args.resume)
    if prompted:
        try:
            input("\nPress Enter to close.")
        except EOFError:
            pass


if __name__ == '__main__':
    main()
