"""Evaluation plots for the detection cascade — shared by the test-set and grouped-CV notebook cells.

Every function draws into an `ax` you pass in, so the notebook only lays out figures. Metrics come from
detect_metrics; nothing here computes a number that isn't already defined there.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def froc(ax, points, label=None, baseline=None, color='C0'):
    """FROC: sensitivity vs false positives per minute, log x (FP rates span orders of magnitude).

    `baseline` = the (sens, fp_per_min) of the always-predict-the-centre null model, drawn as the bar
    the detector has to clear.
    """
    sens = [s for s, _, _ in points]
    fp = [max(f, 1e-2) for _, f, _ in points]           # 1e-2 so the zero-FP point survives a log axis
    ax.step(fp, sens, where='post', color=color, lw=1.4, label=label)
    if baseline is not None:
        ax.plot(max(baseline[1], 1e-2), baseline[0], '*', ms=13, color='C3', label='centre baseline')
    ax.set_xscale('log')
    ax.set_xlabel('false positives / min')
    ax.set_ylabel('sensitivity')
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')


def roc(ax, y, scores, label='pooled', folds=None):
    """ROC. `folds` = [(y, scores), ...] draws the per-fold curves behind the pooled one."""
    for k, (yf, sf) in enumerate(folds or []):
        fpr, tpr, _ = roc_curve(yf, sf)
        ax.plot(fpr, tpr, lw=0.8, alpha=0.45, color='0.5',
                label='per fold' if k == 0 else None)
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(fpr, tpr, lw=1.6, color='C0', label=f'{label} (AUC {roc_auc_score(y, scores):.3f})')
    ax.plot([0, 1], [0, 1], ls=':', lw=0.8, color='k')
    ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc='lower right')


def pr(ax, y, scores):
    """Precision-recall — the imbalance-aware view (positives are ~1-3% of candidate windows)."""
    prec, rec, _ = precision_recall_curve(y, scores)
    ax.plot(rec, prec, lw=1.5, color='C0',
            label=f'AP {average_precision_score(y, scores):.3f}')
    ax.axhline(y.mean(), ls=':', lw=0.8, color='k', label=f'chance ({y.mean():.3f})')
    ax.set_xlabel('recall'); ax.set_ylabel('precision')
    ax.set_yscale('log'); ax.grid(alpha=0.25); ax.legend(fontsize=8)


def score_hist(ax, y, scores, groups=None, recs=None):
    """Score distribution, IED vs the rest. With `groups`+`recs`, positives split clear vs unclear."""
    bins = np.linspace(0, 1, 31)
    ax.hist(scores[y == 0], bins=bins, color='0.6', label=f'non-IED ({(y == 0).sum()})',
            density=True, alpha=0.7)
    if groups is not None and recs is not None:
        cert = np.array([recs[g]['cert'] for g in groups])
        for c, col in (('clear_positive', 'C0'), ('unclear_positive', 'C1')):
            m = (y == 1) & (cert == c)
            if m.any():
                ax.hist(scores[m], bins=bins, histtype='step', lw=1.6, color=col, density=True,
                        label=f'{c.split("_")[0]} IED ({m.sum()})')
    else:
        ax.hist(scores[y == 1], bins=bins, histtype='step', lw=1.6, color='C0', density=True,
                label=f'IED ({(y == 1).sum()})')
    ax.set_xlabel('L3 score'); ax.set_ylabel('density'); ax.legend(fontsize=8)


def fold_aucs(ax, folds, pooled):
    """Per-fold AUC against the pooled value — shows how noisy a single fold's estimate is."""
    ax.plot(range(1, len(folds) + 1), folds, 'o', ms=7, color='C0', label='fold (out-of-fold)')
    ax.axhline(pooled, color='C3', lw=1.2, label=f'pooled {pooled:.3f}')
    ax.axhline(0.5, ls=':', lw=0.8, color='k', label='chance')
    ax.set_xticks(range(1, len(folds) + 1))
    ax.set_xlabel('fold'); ax.set_ylabel('AUC'); ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc='upper right')


def timeline(ax, recs, events, groups, scores, cfg, top_n=None):
    """One row per recording: every proposed event in time, shaded by score, marker in red.

    The qualitative figure — it shows what the detector actually does to a recording, which the summary
    metrics cannot. `top_n` keeps only the highest-scoring events per recording.
    """
    for g, r in enumerate(recs):
        idx = [k for k, gk in enumerate(groups) if gk == g]
        if top_n:
            idx = sorted(idx, key=lambda k: -scores[k])[:top_n]
        t = [events[k]['time'] / cfg.sfreq for k in idx]
        ax.scatter(t, [g] * len(t), c=[scores[k] for k in idx], cmap='viridis', vmin=0, vmax=1,
                   s=18, edgecolors='none')
        if r['epi']:
            ax.plot(r['mk'] / cfg.sfreq, g, '|', ms=14, color='C3', mew=2)
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels([f"{r['fid']}{'*' if r['epi'] else ''}" for r in recs], fontsize=7)
    ax.set_xlabel('time (s)')
    ax.set_title('proposed events (colour = L3 score); red bar = IED marker, * = epileptic', fontsize=9)


def counts(ax, c, title='operating point'):
    """Hits / misses / false positives at one threshold (detect_metrics.detection_counts).

    No correct-rejection bar: over continuous EEG that number is not defined, and the candidate-window
    version of it would be dominated by easy background.
    """
    bars = [('hit', c['hit'], 'C2'), ('miss', c['miss'], 'C1'), ('false pos', c['fp'], 'C3')]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v, f' {v}', ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, max(c['hit'], c['miss'], c['fp']) * 1.18 + 1)
    ax.set_ylabel('count')
    ax.set_title(f"{title}\nscore >= {c['threshold']:.2f} · sens {c['sensitivity']:.2f} · "
                 f"{c['fp_per_min']:.1f} FP/min", fontsize=8)


def loc_error(ax, errs, cfg):
    """Localisation error distribution — capped at hit_tol by construction, so state the cap."""
    ax.hist(errs, bins=np.arange(0, cfg.hit_tol_ms + 5, 5), color='C0', alpha=0.8)
    ax.axvline(np.median(errs), color='C3', lw=1.2, label=f'median {np.median(errs):.0f} ms')
    ax.set_xlabel(f'|event - marker| (ms), capped at hit_tol={cfg.hit_tol_ms:.0f}')
    ax.set_ylabel('IEDs'); ax.legend(fontsize=8)
