"""Evaluation metrics for the IED detection cascade (kept separate from the model).

Per-layer diagnostics + the system FROC (sensitivity vs false-positives/min). A false positive is any
predicted event not within +/-hit_tol of an IED marker — valid because epileptic recordings are verified
single-IED, so all non-marker EEG is IED-free.

Note: the local test set has only ~5 IEDs, so FROC sensitivity moves in coarse steps — these are
indicative; the real evaluation is on external (Neuronostics) data.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from detection import Classifier, event_centre, event_window, _smooth


FP_BUDGET = 10.0    # FP/min operating point the confusion matrix is reported at (see threshold_at)


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


def labelled_events(pipe, recs):
    """Every consolidated event with a valid window: (events, windows, labels, recording index).

    The single L1+L2 pass everything else is built on. `events[k]` belongs to `recs[G[k]]`, so any score
    vector aligned to it can be turned into a FROC — the fitted pipeline's scores on the test set, or
    out-of-fold scores from grouped CV.
    """
    tol = _tol(pipe.cfg)
    E, W, Y, G = [], [], [], []
    for i, r in enumerate(recs):
        events, wins = pipe._event_windows(r['X'])
        for e, w in zip(events, wins):
            E.append(e); W.append(w)
            Y.append(int(bool(r['epi']) and abs(e['time'] - r['mk']) <= tol))
            G.append(i)
    return E, np.array(W), np.array(Y), np.array(G)


def classifier_auc(pipe, recs):
    """AUC of L3 scores over the test candidate windows (IED vs everything else)."""
    _, W, Y, _ = labelled_events(pipe, recs)
    if len(set(Y.tolist())) < 2:
        return float('nan')
    return roc_auc_score(Y, pipe.classifier.predict_proba(W))


def grouped_cv(pipe, recs, n_splits=5):
    """Out-of-fold L3 scores over the TRAIN candidates, folds grouped by recording.

    The 10-recording test set holds only ~5 IEDs, so its AUC swings wildly and must not be tuned
    against; this pools out-of-fold scores over all TRAIN candidates instead. Grouping by recording
    keeps the several events one IED spawns out of both sides of a fold. SLOW — one L3 fit per fold
    (~3-4 min each on the 90), so it is a deliberate one-off, not part of `report`.

    Returns a dict with the pooled `auc` and average precision `ap`, the per-fold `folds` AUCs, and the
    raw `scores` / `Y` / `G` / `events` so the caller can build a cross-validated FROC and the plots.
    """
    E, W, Y, G = labelled_events(pipe, recs)
    scores, fold, folds = np.zeros(len(Y)), np.zeros(len(Y), int), []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(W, Y, G)):
        scores[te] = Classifier(pipe.cfg).fit(W[tr], Y[tr]).predict_proba(W[te])
        fold[te] = k
        folds.append(roc_auc_score(Y[te], scores[te]))
    return {'auc': roc_auc_score(Y, scores), 'ap': average_precision_score(Y, scores),
            'folds': folds, 'fold': fold, 'scores': scores, 'Y': Y, 'G': G, 'events': E}


def bootstrap_auc_ci(Y, G, scores, n_boot=2000, seed=0):
    """95% CI for an AUC, resampling RECORDINGS — the independent unit — not windows.

    Windows within a recording are not independent (one IED spawns several events), so a window-level
    bootstrap would give an interval that is far too narrow.
    """
    rng = np.random.default_rng(seed)
    groups = np.unique(G)
    idx = {g: np.where(G == g)[0] for g in groups}
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([idx[g] for g in rng.choice(groups, len(groups), replace=True)])
        if len(set(Y[take].tolist())) > 1:
            vals.append(roc_auc_score(Y[take], scores[take]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def froc_points(recs, events, G, scores, cfg):
    """(sensitivity, fp_per_min, threshold) sweeping the score threshold, strict to permissive.

    Sensitivity counts a recording once (any near-marker event above threshold = found); every other
    surviving event is a false positive. Works with any score vector aligned to `events`/`G`.
    """
    tol = cfg.samp(cfg.hit_tol_ms)
    total_min = sum(r['dur'] for r in recs) / 60.0
    n_epi = sum(r['epi'] for r in recs)
    hit = np.array([bool(recs[g]['epi']) and abs(e['time'] - recs[g]['mk']) <= tol
                    for e, g in zip(events, G)])
    scores = np.asarray(scores, float)
    order = np.argsort(-scores)
    s, h, g = scores[order], hit[order], np.asarray(G)[order]
    fp_cum = np.cumsum(~h)
    seen, tp_cum, n = set(), [], 0
    for hk, gk in zip(h, g):
        if hk and gk not in seen:
            seen.add(gk); n += 1
        tp_cum.append(n)
    pts = [(0.0, 0.0, float('inf'))]                  # nothing passes -> (0 sens, 0 FP)
    for k in range(len(s)):
        if k + 1 < len(s) and s[k + 1] == s[k]:
            continue                                   # one point per distinct threshold
        pts.append((tp_cum[k] / n_epi, fp_cum[k] / total_min, float(s[k])))
    return pts


def froc(pipe, recs):
    """FROC of the fitted pipeline on `recs`. See froc_points."""
    E, W, Y, G = labelled_events(pipe, recs)
    return froc_points(recs, E, G, pipe.classifier.predict_proba(W), pipe.cfg)


def localisation_errors(recs, events, G, scores, cfg):
    """|event time - marker| in ms per IED, using that IED's highest-scoring near-marker event."""
    tol, best = cfg.samp(cfg.hit_tol_ms), {}
    for e, gk, s in zip(events, G, scores):
        r = recs[gk]
        if r['epi'] and abs(e['time'] - r['mk']) <= tol and s > best.get(gk, (-np.inf, 0))[0]:
            best[gk] = (s, abs(e['time'] - r['mk']) / cfg.sfreq * 1000)
    return np.array([v[1] for v in best.values()])


def localisation_error_ms(pipe, recs):
    """Median localisation error (ms) over the IEDs that were hit. See localisation_errors."""
    E, W, Y, G = labelled_events(pipe, recs)
    errs = localisation_errors(recs, E, G, pipe.classifier.predict_proba(W), pipe.cfg)
    return float(np.median(errs)) if len(errs) else float('nan')


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


def threshold_at(points, fp_per_min):
    """The score threshold that gives the best sensitivity within an FP/min budget — an operating point.

    Ties break toward the higher threshold (same sensitivity, fewer false positives).
    """
    ok = [(s, t) for s, fp, t in points if fp <= fp_per_min]
    return max(ok)[1] if ok else float('inf')


def detection_counts(recs, events, G, scores, cfg, thr):
    """Hits / misses / false positives at one score threshold — the FROC at a single operating point.

      hit   an IED with at least one above-threshold event within +/-hit_tol of its marker
      miss  an IED with none
      fp    every other above-threshold event

    **Correct rejections are deliberately not counted.** Over continuous EEG there is no meaningful number
    of "non-events", and counting rejected candidate windows instead would flatter the model with easy
    background: ~98.5% of candidates are negatives, most of them trivially so. Anything derived from that
    count (specificity, accuracy, a full 2x2) is inflated for the same reason — report hit/miss/fp.
    """
    tol = cfg.samp(cfg.hit_tol_ms)
    hit_recs, fp = set(), 0
    for e, g, above in zip(events, G, np.asarray(scores) >= thr):
        if not above:
            continue
        r = recs[g]
        if r['epi'] and abs(e['time'] - r['mk']) <= tol:
            hit_recs.add(g)
        else:
            fp += 1
    n_epi = sum(r['epi'] for r in recs)
    return {'hit': len(hit_recs), 'miss': n_epi - len(hit_recs), 'fp': fp, 'threshold': float(thr),
            'sensitivity': len(hit_recs) / max(n_epi, 1),
            'fp_per_min': fp / (sum(r['dur'] for r in recs) / 60.0)}


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
    """Compact per-layer + system report. Returns (events, scores, labels, G) so the plots in the run
    notebook reuse this single L1+L2+L3 pass instead of repeating it."""
    cfg = pipe.cfg
    E, W, Y, G = labelled_events(pipe, recs)
    scores = pipe.classifier.predict_proba(W)
    n_epi = sum(r['epi'] for r in recs)
    mins = sum(r['dur'] for r in recs) / 60.0
    print(f"  L1 IED recall            : {layer1_recall(pipe, recs):.2f}")
    print(f"  L2 consolidation recall  : {consolidation_recall(pipe, recs):.2f}")
    print(f"  L2 events                : {len(E) / mins:.0f}/min, "
          f"{Y.sum() / n_epi:.2f} per IED, {np.mean([e['n_channels'] for e in E]):.2f} channels each")
    if len(set(Y.tolist())) > 1:
        print(f"  L3 ROC-AUC / PR-AUC      : {roc_auc_score(Y, scores):.3f} / "
              f"{average_precision_score(Y, scores):.3f}   (PR chance = {Y.mean():.3f})")
    errs = localisation_errors(recs, E, G, scores, cfg)
    print(f"  localisation error (ms)  : {np.median(errs) if len(errs) else float('nan'):.0f}")
    b_sens, b_fp = centre_baseline(recs, cfg)
    print(f"  centre baseline          : sens {b_sens:.2f}, {b_fp:.1f} FP/min  (to beat)")
    pts = froc_points(recs, E, G, scores, cfg)
    print("  FROC — best sensitivity at FP/min budget:")
    for fp, sens in froc_summary(pts).items():
        print(f"    <= {fp:>3} FP/min : sens {sens:.2f}")
    c = detection_counts(recs, E, G, scores, cfg, threshold_at(pts, FP_BUDGET))
    print(f"  operating point @ <= {FP_BUDGET:.0f} FP/min (score >= {c['threshold']:.2f}):")
    print(f"    hits {c['hit']}/{c['hit'] + c['miss']}   misses {c['miss']}   "
          f"false positives {c['fp']} ({c['fp_per_min']:.1f}/min)")
    return E, scores, Y, G
