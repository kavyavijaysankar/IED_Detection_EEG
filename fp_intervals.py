"""Inter-event interval analysis for false positives: is a temporal merge stage justified?

ANALYSIS ONLY — nothing here touches the pipeline. It reads the out-of-fold CV output, which no cell
saved to disk, so `export_cv` regenerates it once into results/cv_events.csv (~20 min) and everything
after that is fast and re-runnable.

    python fp_intervals.py            # analyse, exporting the CV first if the CSV is missing
    python fp_intervals.py --refresh  # re-run the CV even if the CSV exists
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'src'))

from classify_loader import load_CSV
from detect_config import Config
from detect_data import electrode_positions, load_dataset, stratified_split
from detection import DetectionPipeline, _smooth, spatial_features
import detect_metrics as M

CV_CSV = HERE / 'results' / 'cv_events.csv'
OUT = HERE / 'results'
FIG = HERE / 'figures' / 'fp_intervals.png'


def export_cv(path):
    """Run the grouped CV and write one row per candidate event: the four fields the analysis needs
    (recording, time, score, matched) plus what is needed to re-derive the operating point.

    Mirrors the run notebook's CV cell exactly — L3 out-of-fold, then L4 refitted inside the same folds,
    so `score` is what the SHIPPED pipeline would give. The pipeline need not be fitted first: L1, L2 and
    the window cutting are unfitted, and grouped_cv fits its own Classifier per fold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cfg = Config()
    man = load_CSV(HERE / 'Kural_Dataset' / 'eeg_summary.csv')
    recs_df, _ = stratified_split(man, n_test=cfg.n_test, seed=cfg.split_seed)
    recs = load_dataset(recs_df, HERE / 'Kural_Dataset' / 'Recordings', cfg)
    print(f"{len(recs)} recordings, {sum(r['epi'] for r in recs)} IEDs — running grouped CV")

    pipe = DetectionPipeline(cfg)
    cv = M.grouped_cv(pipe, recs)
    print(f"L3 out-of-fold AUC {cv['auc']:.3f}  PR-AUC {cv['ap']:.3f}  ({len(cv['Y'])} windows)")

    s_final = cv['scores']
    S = np.zeros((len(cv['Y']), 2))
    if cfg.spatial_feature:
        pos = electrode_positions()
        Xsm = {g: _smooth(recs[g]['X'], cfg) for g in set(cv['G'].tolist())}
        S = np.array([[spatial_features(e, Xsm[g], cfg, pos)[n] for n in ('n_channels', 'dipole')]
                      for e, g in zip(cv['events'], cv['G'])], float)
        F4 = np.column_stack([cv['scores'], S])
        s_final = np.zeros(len(cv['Y']))
        for kf in np.unique(cv['fold']):
            tr, te = cv['fold'] != kf, cv['fold'] == kf
            sk = StandardScaler().fit(F4[tr])
            lr = LogisticRegression(C=cfg.lr_C, max_iter=1000, class_weight='balanced')
            lr.fit(sk.transform(F4[tr]), cv['Y'][tr])
            s_final[te] = lr.predict_proba(sk.transform(F4[te]))[:, 1]
        print("L4 refitted in-fold (score + n_channels + dipole)")

    tol = cfg.samp(cfg.hit_tol_ms)
    rows = []
    for e, g, s3, s4, y, k, sp in zip(cv['events'], cv['G'], cv['scores'], s_final, cv['Y'],
                                      cv['fold'], S):
        r = recs[g]
        rows.append({'recording': r['fid'], 'time_s': e['time'] / cfg.sfreq, 'score': float(s4),
                     'l3_score': float(s3), 'matched': int(y), 'fold': int(k),
                     'epileptic': int(bool(r['epi'])), 'certainty': r['cert'],
                     'marker_s': r['mk'] / cfg.sfreq, 'duration_s': r['dur'],
                     'n_channels': int(e['n_channels']), 'dipole': float(sp[1]),
                     'offset_ms': (e['time'] - r['mk']) / cfg.sfreq * 1000.0})
    df = pd.DataFrame(rows).sort_values(['recording', 'time_s'])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    pts = M.froc_points(recs, cv['events'], cv['G'], s_final, cfg)
    thr = M.threshold_at(pts, cfg.fp_budget)
    cm = M.detection_counts(recs, cv['events'], cv['G'], s_final, cfg, thr)
    print(f"operating point @ <={cfg.fp_budget:.0f} FP/min: score >= {thr:.3f}  "
          f"hits {cm['hit']} misses {cm['miss']} FPs {cm['fp']} ({cm['fp_per_min']:.1f}/min)")
    print(f"wrote {path} ({len(df)} events, tol +/-{cfg.hit_tol_ms:.0f} ms)")
    return df, thr


def gaps_within(df, key='recording'):
    """Consecutive-event gaps in seconds, per group, as a long frame.

    Zero-length gaps are dropped: two events sharing a timestamp are simultaneous detections on
    different channels, not an interval, and log10(0) would take the density estimate with it.
    """
    out = []
    for name, g in df.groupby(key, sort=True):
        t = np.sort(g['time_s'].to_numpy())
        for a, b in zip(t[:-1], t[1:]):
            if b > a:
                out.append({'group': name, 't_prev': a, 't_next': b, 'gap_s': b - a})
    return pd.DataFrame(out, columns=['group', 't_prev', 't_next', 'gap_s'])


def pct(x, ps=(5, 10, 25, 50, 75, 95)):
    return {f'p{p}': float(np.percentile(x, p)) for p in ps} if len(x) else {}


def describe(name, g):
    if not len(g):
        print(f"{name}: no gaps"); return
    q = pct(g['gap_s'].to_numpy())
    print(f"{name}: {len(g)} gaps from {g.groupby('group').ngroups} groups")
    print(f"    median {q['p50']:.3f} s   IQR {q['p25']:.3f}-{q['p75']:.3f} s "
          f"(width {q['p75'] - q['p25']:.3f} s)")
    print("    " + "  ".join(f"p{p} {v:.3f}s" for p, v in
                             ((5, q['p5']), (10, q['p10']), (25, q['p25']),
                              (75, q['p75']), (95, q['p95']))))


def troughs(logx, grid, bw=None, depth=0.10, min_mode=0.15):
    """Interior local minima of a KDE over log10(gap), as (position, relative depth), deepest first.

    Two criteria, and the second is not optional:
      depth     1 - density(min) / min(flanking peaks): how far the density actually falls, so a
                shoulder wiggle is not called a trough.
      min_mode  both flanking peaks must reach this fraction of the global maximum. Without it a single
                outlying gap forms its own tiny bump, and the empty stretch between it and the bulk
                scores as a deep trough. Measured on a unimodal control: the bootstrap reported a
                spurious trough in 87% of resamples, all of them out in that tail.
    """
    if len(logx) < 10:
        return [], np.zeros_like(grid)
    d = gaussian_kde(logx, bw_method=bw)(grid)
    floor = min_mode * d.max()
    found = []
    for i in range(1, len(d) - 1):
        if not (d[i] <= d[i - 1] and d[i] < d[i + 1]):
            continue
        flank = min(d[:i].max(initial=0.0), d[i + 1:].max(initial=0.0))
        if flank < floor:
            continue
        rel = 1.0 - d[i] / flank
        if rel >= depth:
            found.append((float(10 ** grid[i]), float(rel)))
    return sorted(found, key=lambda t: -t[1]), d


def bootstrap_trough(df_gaps, grid, n_boot=2000, seed=0):
    """How often a trough survives resampling RECORDINGS — the independent unit, as in bootstrap_auc_ci.

    Windows inside a recording are not independent, so resampling gaps would understate the uncertainty.
    """
    rng = np.random.default_rng(seed)
    groups = df_gaps['group'].unique()
    by = {g: np.log10(df_gaps.loc[df_gaps['group'] == g, 'gap_s'].to_numpy()) for g in groups}
    hits = []
    for _ in range(n_boot):
        x = np.concatenate([by[g] for g in rng.choice(groups, len(groups), replace=True)])
        t, _ = troughs(x, grid)
        hits.append(t[0][0] if t else np.nan)
    hits = np.array(hits)
    frac = float(np.mean(~np.isnan(hits)))
    loc = hits[~np.isnan(hits)]
    return frac, (float(np.percentile(loc, 2.5)), float(np.percentile(loc, 97.5))) if len(loc) else None


def plots(fp, ied, cfg, thr, path):
    lo = max(min(fp['gap_s'].min(), ied['gap_s'].min() if len(ied) else np.inf), 1e-3)
    hi = fp['gap_s'].max()
    edges = np.logspace(np.log10(lo * 0.9), np.log10(hi * 1.1), 25)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))

    for ax in axs:
        ax.axvline(cfg.nms_ms / 1000, color='0.5', ls=':', lw=1.0)
        ax.axvline(2 * cfg.hit_tol_ms / 1000, color='C1', ls=':', lw=1.0)
        ax.set_xscale('log')
        ax.set_xlabel('gap between consecutive events (s)')
        ax.grid(alpha=0.25)

    axs[0].hist(fp['gap_s'], bins=edges, color='C0', alpha=0.75, label=f"false positives (n={len(fp)})")
    if len(ied):
        axs[0].hist(ied['gap_s'], bins=edges, color='C3', alpha=0.6,
                    label=f"within one IED (n={len(ied)})")
    axs[0].set_ylabel('gaps')
    axs[0].set_title(f"FP inter-event intervals, score >= {thr:.3f}", fontsize=9)
    axs[0].legend(fontsize=7)

    for g, c, lab in ((fp['gap_s'], 'C0', 'false positives'),
                      (ied['gap_s'], 'C3', 'within one IED')):
        if len(g):
            s = np.sort(np.asarray(g))
            axs[1].step(s, np.arange(1, len(s) + 1) / len(s), where='post', color=c, lw=1.4, label=lab)
    axs[1].set_ylabel('cumulative fraction')
    axs[1].set_ylim(-0.03, 1.03)
    axs[1].set_title(f"ECDF (grey = L1 NMS {cfg.nms_ms:.0f} ms, orange = 2x hit tolerance)", fontsize=9)
    axs[1].legend(fontsize=7)

    plt.suptitle('Do false positives cluster in time? Kural is 11-14 s per recording, so the right '
                 'tail is epoch length, not detector behaviour', y=1.02, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=800, bbox_inches='tight')
    plt.close(fig)
    return edges


def channel_overlap(df, thr, cfg):
    """Jaccard overlap of member ELECTRODES for consecutive above-threshold FP pairs, by gap.

    The trough question is only interesting if short-gap false positives are one transient counted
    twice. Two events of the same discharge share channels; two independent transients need not. L1+L2
    are deterministic and unfitted, so re-deriving members costs a few seconds and needs no model.
    """
    man = load_CSV(HERE / 'Kural_Dataset' / 'eeg_summary.csv')
    recs_df, _ = stratified_split(man, n_test=cfg.n_test, seed=cfg.split_seed)
    recs = load_dataset(recs_df, HERE / 'Kural_Dataset' / 'Recordings', cfg)
    pipe = DetectionPipeline(cfg)
    mem = {}
    for r in recs:
        for e in pipe._event_windows(r['X'], r['bad_hard'])[0]:
            mem[(r['fid'], round(e['time'] / cfg.sfreq, 4))] = {c for c, _ in e['members']}
    rows = []
    for fid, g in df[(df['score'] >= thr) & (df['matched'] == 0)].groupby('recording'):
        g = g.sort_values('time_s')
        t = g['time_s'].to_numpy()
        for a, b in zip(t[:-1], t[1:]):
            A, B = mem.get((fid, round(a, 4))), mem.get((fid, round(b, 4)))
            if A and B:
                rows.append({'gap_s': b - a, 'jaccard': len(A & B) / len(A | B)})
    return pd.DataFrame(rows)


def merge_sweep(df, thr, cfg, xs=(0.0, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0)):
    """What a post-L4 temporal merge at X would actually do, at the FIXED operating threshold.

    Collapses above-threshold detections whose consecutive gap is <= X, keeping the highest-scoring
    member's TIME as the merged detection's time — which is what a merge stage would do, and is why a
    hit can be lost: if an FP outscores the true event, the merged time moves off the marker.
    """
    tol = cfg.hit_tol_ms / 1000.0
    hi = df[df['score'] >= thr]
    mins = df.groupby('recording')['duration_s'].first().sum() / 60
    n_epi = int(df.groupby('recording')['epileptic'].first().sum())
    out = []
    for X in xs:
        hits, fp = set(), 0
        for fid, g in hi.groupby('recording'):
            g = g.sort_values('time_s')
            t, s = g['time_s'].to_numpy(), g['score'].to_numpy()
            mk, epi = g['marker_s'].iloc[0], bool(g['epileptic'].iloc[0])
            cut = np.where(np.diff(t) > X)[0] + 1
            for blk in np.split(np.arange(len(t)), cut):
                rep = blk[np.argmax(s[blk])]
                if epi and abs(t[rep] - mk) <= tol:
                    hits.add(fid)
                else:
                    fp += 1
        out.append({'X_s': X, 'hits': len(hits), 'misses': n_epi - len(hits), 'fp': fp,
                    'fp_per_min': fp / mins, 'sensitivity': len(hits) / n_epi})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='re-run the grouped CV even if the CSV exists')
    args = ap.parse_args()

    cfg = Config()
    if args.refresh or not CV_CSV.exists():
        df, thr = export_cv(CV_CSV)
    else:
        df = pd.read_csv(CV_CSV)
        recs = [{'epi': bool(e), 'dur': d, 'mk': m * cfg.sfreq}
                for e, d, m in df.groupby('recording')[['epileptic', 'duration_s', 'marker_s']]
                .first().itertuples(index=False)]
        ev = [{'time': t * cfg.sfreq} for t in df['time_s']]
        gi = {f: i for i, f in enumerate(df.groupby('recording').groups)}
        G = np.array([gi[f] for f in df['recording']])
        thr = M.threshold_at(M.froc_points(recs, ev, G, df['score'].to_numpy(), cfg), cfg.fp_budget)
        print(f"read {CV_CSV} ({len(df)} events); operating point score >= {thr:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    n_rec, mins = df['recording'].nunique(), df.groupby('recording')['duration_s'].first().sum() / 60
    print(f"\n{n_rec} recordings, {mins:.1f} min, {len(df)} candidate events, "
          f"{int(df['matched'].sum())} matched an IED marker (+/-{cfg.hit_tol_ms:.0f} ms)\n")

    # 1. false positives at the operating point = above threshold and not matching a marker
    fp_ev = df[(df['score'] >= thr) & (df['matched'] == 0)]
    fp = gaps_within(fp_ev)
    describe('FALSE-POSITIVE gaps', fp)
    n_fp_rec = fp_ev['recording'].nunique()
    print(f"    {n_fp_rec} of {n_rec} recordings hold >=1 FP, "
          f"{fp.groupby('group').ngroups} hold >=2 and so contribute a gap")

    # 3. comparison group: events matching the SAME IED. Threshold-independent, because fragmenting one
    #    discharge into several events is an L2 property, not a scoring one.
    ied_ev = df[df['matched'] == 1]
    ied = gaps_within(ied_ev)
    print()
    if ied.groupby('group').ngroups < 10:
        print(f"WITHIN-IED gaps: only {ied.groupby('group').ngroups} IEDs split into >1 event "
              f"(<10, so raw values not percentiles): "
              f"{', '.join(f'{v:.3f}s' for v in sorted(ied['gap_s']))}")
    else:
        describe('WITHIN-IED gaps', ied)
    above = ied_ev[ied_ev['score'] >= thr]
    print(f"    of those, {gaps_within(above).groupby('group').ngroups} IEDs contribute >1 event "
          f"ABOVE threshold")
    print(f"    CEILING BY CONSTRUCTION: an event counts as matching only within +/-"
          f"{cfg.hit_tol_ms:.0f} ms of the marker, so these gaps cannot exceed "
          f"{2 * cfg.hit_tol_ms / 1000:.1f} s whatever the discharge does.")

    # 4. bimodality on the log scale
    print()
    if len(fp) >= 10:
        lg = np.log10(fp['gap_s'].to_numpy())
        grid = np.linspace(lg.min() - 0.2, lg.max() + 0.2, 400)
        t, dens = troughs(lg, grid)
        if t:
            print(f"KDE on log10(gap): trough at {t[0][0]:.3f} s (relative depth {t[0][1]:.0%})"
                  + (f"; others at {', '.join(f'{p:.2f}s' for p, _ in t[1:])}" if len(t) > 1 else ""))
        else:
            print("KDE on log10(gap): NO interior local minimum deeper than 10% — unimodal")
        frac, ci = bootstrap_trough(fp, grid)
        print(f"    recording-level bootstrap (2000x): a trough appears in {frac:.0%} of resamples"
              + (f", 95% of them between {ci[0]:.3f} and {ci[1]:.3f} s" if ci else ""))
        print("    (Hartigan's dip test would need a new pinned dependency; a KDE trough plus a "
              "bootstrap over recordings answers the same question and reports its own stability.)")
        pd.DataFrame({'log10_gap_s': grid, 'gap_s': 10 ** grid, 'density': dens}) \
            .to_csv(OUT / 'fp_intervals_kde.csv', index=False)
    else:
        print(f"only {len(fp)} FP gaps — too few for a density estimate")

    # 5. is a short gap actually one transient counted twice, and what would merging cost?
    print()
    ov = channel_overlap(df, thr, cfg)
    if len(ov):
        short, long = ov[ov['gap_s'] <= 0.5], ov[ov['gap_s'] > 0.5]
        print(f"channel overlap of consecutive FP pairs (Jaccard over member electrodes):")
        for nm, s in (('gap <= 0.5 s', short), ('gap  > 0.5 s', long)):
            if len(s):
                print(f"    {nm}: n={len(s):3d}  median {s['jaccard'].median():.3f}  "
                      f"mean {s['jaccard'].mean():.3f}  share with NO shared channel "
                      f"{(s['jaccard'] == 0).mean():.0%}")
        ov.to_csv(OUT / 'fp_intervals_overlap.csv', index=False)

    print()
    sw = merge_sweep(df, thr, cfg)
    print(f"post-L4 temporal merge at X, at the fixed operating threshold {thr:.3f}:")
    print(f"    {'X (s)':>7}{'hits':>7}{'misses':>8}{'FPs':>7}{'FP/min':>9}{'sens':>7}")
    for _, r in sw.iterrows():
        print(f"    {r['X_s']:>7.2f}{int(r['hits']):>7}{int(r['misses']):>8}{int(r['fp']):>7}"
              f"{r['fp_per_min']:>9.2f}{r['sensitivity']:>7.2f}")
    sw.to_csv(OUT / 'fp_intervals_merge_sweep.csv', index=False)

    edges = plots(fp, ied, cfg, thr, FIG)
    fp.assign(kind='false_positive').pipe(
        lambda a: pd.concat([a, ied.assign(kind='within_ied')])).to_csv(
        OUT / 'fp_intervals_gaps.csv', index=False)
    cnt, _ = np.histogram(fp['gap_s'], bins=edges)
    cnt_i, _ = np.histogram(ied['gap_s'], bins=edges) if len(ied) else (np.zeros(len(edges) - 1), None)
    pd.DataFrame({'bin_lo': edges[:-1], 'bin_hi': edges[1:], 'n_fp': cnt, 'n_within_ied': cnt_i}) \
        .to_csv(OUT / 'fp_intervals_hist.csv', index=False)
    for nm, g in (('fp', fp), ('within_ied', ied)):
        if len(g):
            s = np.sort(g['gap_s'].to_numpy())
            pd.DataFrame({'gap_s': s, 'cdf': np.arange(1, len(s) + 1) / len(s)}) \
                .to_csv(OUT / f'fp_intervals_ecdf_{nm}.csv', index=False)

    print(f"\nwrote {FIG}")
    print(f"      {OUT}/fp_intervals_gaps.csv, _hist.csv, _ecdf_fp.csv, _ecdf_within_ied.csv, _kde.csv")
    print(f"\nCAUTION: recordings are {df['duration_s'].min():.0f}-{df['duration_s'].max():.0f} s, so no "
          f"gap can exceed ~{df['duration_s'].max():.0f} s. The right tail is epoch length.")


if __name__ == '__main__':
    main()
