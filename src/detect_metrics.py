"""Evaluation metrics for the IED detection cascade (kept separate from the model).

Per-layer diagnostics + the system FROC (sensitivity vs false-positives/min). A false positive is any
predicted event not within +/-hit_tol of an IED marker — valid because epileptic recordings are verified
single-IED, so all non-marker EEG is IED-free.

Note: the local test set has only ~5 IEDs, so FROC sensitivity moves in coarse steps — these are
indicative; the real evaluation is on external (Neuronostics) data.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from detection import Classifier, event_centre, event_window, _smooth


def _tol(cfg):
    return cfg.samp(cfg.hit_tol_ms)


def layer1_recall(pipe, recs):
    """Fraction of IEDs with an L1 candidate within +/-hit_tol of the marker (any channel)."""
    tol, hit, n = _tol(pipe.cfg), 0, 0
    for r in recs:
        if not r['epi']:
            continue
        n += 1
        cands, _ = pipe.stage1.detect(r['X'])
        hit += any(abs(t - r['mk']) <= tol for _, t in cands)
    return hit / n if n else float('nan')


def consolidation_recall(pipe, recs):
    """Fraction of IEDs that survive L2 as an event (>=2 ch) within +/-hit_tol of the marker."""
    tol, hit, n = _tol(pipe.cfg), 0, 0
    for r in recs:
        if not r['epi']:
            continue
        n += 1
        hit += any(abs(e['time'] - r['mk']) <= tol for e in pipe._events(r['X'])[0])
    return hit / n if n else float('nan')


def _labelled_windows(pipe, recs):
    """(windows, labels, recording index per window) over every consolidated event."""
    tol = _tol(pipe.cfg)
    W, Y, G = [], [], []
    for i, r in enumerate(recs):
        events, wins = pipe._event_windows(r['X'])
        for e, w in zip(events, wins):
            W.append(w)
            Y.append(int(bool(r['epi']) and abs(e['time'] - r['mk']) <= tol))
            G.append(i)
    return np.array(W), np.array(Y), np.array(G)


def classifier_auc(pipe, recs):
    """AUC of L3 scores over the test candidate windows (IED vs everything else)."""
    W, Y, _ = _labelled_windows(pipe, recs)
    if len(set(Y.tolist())) < 2:
        return float('nan')
    return roc_auc_score(Y, pipe.classifier.predict_proba(W))


def grouped_cv_auc(pipe, recs, n_splits=5):
    """Out-of-fold L3 AUC over the TRAIN candidates, folds grouped by recording. (pooled, per-fold).

    The 10-recording test set holds only ~5 IEDs, so its AUC swings wildly and must not be tuned
    against; this pools out-of-fold scores over all TRAIN candidates instead. Grouping by recording
    keeps the ~3 events one IED spawns out of both sides of a fold. SLOW — one L3 fit per fold (~4 min
    each on the 90), so it is a deliberate one-off, not part of `report`.
    """
    W, Y, G = _labelled_windows(pipe, recs)
    scores, folds = np.zeros(len(Y)), []
    for tr, te in GroupKFold(n_splits=n_splits).split(W, Y, G):
        scores[te] = Classifier(pipe.cfg).fit(W[tr], Y[tr]).predict_proba(W[te])
        folds.append(roc_auc_score(Y[te], scores[te]))
    return roc_auc_score(Y, scores), folds


def froc(pipe, recs):
    """List of (sensitivity, fp_per_min, score_threshold), from strict to permissive."""
    tol = _tol(pipe.cfg)
    total_min = sum(r['dur'] for r in recs) / 60.0
    n_epi = sum(r['epi'] for r in recs)
    per = []
    for r in recs:
        ev = pipe.predict(r['X'])
        per.append((r, ev))
    thresholds = sorted({e['score'] for _, ev in per for e in ev}, reverse=True)
    out = []
    for s in [1.01] + thresholds:                      # 1.01 = nothing passes -> (0 sens, 0 fp)
        tp_rec, fp = 0, 0
        for r, ev in per:
            above = [e for e in ev if e['score'] >= s]
            if r['epi'] and any(abs(e['time'] - r['mk']) <= tol for e in above):
                tp_rec += 1
            for e in above:
                if not (r['epi'] and abs(e['time'] - r['mk']) <= tol):
                    fp += 1
        out.append((tp_rec / n_epi, fp / total_min, s))
    return out


def localisation_error_ms(pipe, recs):
    """Median |detected event time - marker| in ms, over IEDs where an event landed within +/-hit_tol."""
    tol, errs = _tol(pipe.cfg), []
    for r in recs:
        if not r['epi']:
            continue
        near = [e for e in pipe.predict(r['X']) if abs(e['time'] - r['mk']) <= tol]
        if near:
            best = max(near, key=lambda e: e['score'])
            errs.append(abs(best['time'] - r['mk']) / pipe.cfg.sfreq * 1000)
    return float(np.median(errs)) if errs else float('nan')


def centre_baseline(recs, cfg):
    """'Always predict the recording centre' — the number to beat. Returns (sensitivity, fp_per_min).

    Markers sit mid-recording here, so this scores high WITHOUT any modelling; that is exactly why it must
    be reported (and why real eval needs random cropping so the marker isn't centred).
    """
    tol = _tol(cfg)
    total_min = sum(r['dur'] for r in recs) / 60.0
    n_epi = sum(r['epi'] for r in recs)
    tp = fp = 0
    for r in recs:
        centre = r['X'].shape[1] // 2
        if r['epi'] and abs(centre - r['mk']) <= tol:
            tp += 1
        else:
            fp += 1                                     # a centre prediction that isn't a true hit
    return tp / n_epi, fp / total_min


def froc_summary(points, fp_targets=(1, 5, 10, 25, 50, 100)):
    """Best sensitivity achievable at or below each FP/min budget (compact view of the FROC)."""
    return {t: max([s for s, fp, _ in points if fp <= t], default=0.0) for t in fp_targets}


def centring_offsets(pipe, recs):
    """Window-centring diagnostic on the IED events: (windows (n, L), offsets in ms, recording ids).

    offset = window centre - marker, so 0 means the classifier window sits exactly on the annotated
    spike peak. Registration is alignment-sensitive: if `cfg.centre_on` re-centres on the wrong extremum
    (a slow wave, a drift, the biphasic down-stroke) the offsets scatter and the FDA features degrade.
    Only near-marker (positive) events are measured — a mimic has no "correct" centre.
    """
    cfg, tol = pipe.cfg, _tol(pipe.cfg)
    W, offs, fids = [], [], []
    for r in recs:
        if not r['epi']:
            continue
        events, stat = pipe._events(r['X'])
        Xs = _smooth(r['X'], cfg)
        for e in events:
            if abs(e['time'] - r['mk']) > tol:
                continue
            w = event_window(r['X'], e, cfg, Xs, stat)
            if w is None:
                continue
            W.append(w)
            offs.append((event_centre(r['X'], e, cfg, Xs, stat) - r['mk']) / cfg.sfreq * 1000)
            fids.append(r['fid'])
    return np.array(W), np.array(offs), fids


def report(pipe, recs):
    print(f"  L1 IED recall            : {layer1_recall(pipe, recs):.2f}")
    print(f"  L2 consolidation recall  : {consolidation_recall(pipe, recs):.2f}")
    print(f"  L3 classifier AUC (test) : {classifier_auc(pipe, recs):.3f}")
    print(f"  localisation error (ms)  : {localisation_error_ms(pipe, recs):.0f}")
    b_sens, b_fp = centre_baseline(recs, pipe.cfg)
    print(f"  centre baseline          : sens {b_sens:.2f}, {b_fp:.1f} FP/min  (to beat)")
    print("  FROC — best sensitivity at FP/min budget:")
    for fp, sens in froc_summary(froc(pipe, recs)).items():
        print(f"    <= {fp:>3} FP/min : sens {sens:.2f}")
