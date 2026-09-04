from dataclasses import fields, replace

import joblib
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from skfda import FDataGrid
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import FisherRaoElasticRegistration, LeastSquaresShiftRegistration
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from detect_config import Config
from detect_data import HOMOLOGOUS, electrode_positions, window_around
from detect_stage1 import channel_stat, candidates


def _smooth(X, cfg):
    """SG-smoothed signal (deriv=0) for morphology correlation and peak-to-peak."""
    return savgol_filter(X, cfg.sg_samples(), cfg.sg_poly, axis=-1)


def _norm_lag_corr(a, b, max_lag):
    """Max over lags of |normalised cross-correlation| between two equal-length windows.

    Normalised -> amplitude-invariant (a diminished copy still matches); lag search -> a delayed copy still matches; absolute value -> polarity-invariant (opposite side of a dipole still matches).
    """
    a = a - a.mean()
    b = b - b.mean()
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return 0.0
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:len(a) + lag], b[-lag:])
        na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
        if na < 1e-9 or nb < 1e-9:
            continue
        best = max(best, abs(float(np.dot(aa, bb)) / (na * nb)))
    return best


class Stage1:
    """L1: multichannel SG 2nd-derivative candidate generation (recorded reference)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, X, bad=()):
        """X (n_ch, T) -> (candidates [(ch, sample)], sharpness stat (n_ch, T)).
        bad = interpolated channel indices, which are dropped here. An interpolated channel is a smooth blend of its neighbours and would otherwise correlate with them by construction, manufacturing exactly the agreement L2 looks for.
        """
        stat = channel_stat(X, self.cfg)
        cands = candidates(stat, self.cfg.l1_threshold, self.cfg.samp(self.cfg.nms_ms))
        if len(bad):
            drop = set(bad)
            cands = [(c, t) for c, t in cands if c not in drop]
        return cands, stat


class Consolidator:
    """L2: group per-channel candidates of the same spike into events via cross-correlation.
    Two grouping rules (cfg.grouping), both using the same normalised/lag-searched/absolute correlation: greedy & components.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def consolidate(self, X, cands, stat):
        if self.cfg.grouping == 'components':
            return self._components(X, cands)
        if self.cfg.grouping != 'greedy':
            raise ValueError(f"unknown grouping: {self.cfg.grouping}")
        return self._greedy(X, cands, stat)

    def _components(self, X, cands):
        """Single-linkage grouping: events are connected components of the 'same spike' graph.
        """
        cfg = self.cfg
        ch_half = cfg.samp(cfg.corr_halfwin_ms)
        coin = cfg.samp(cfg.coincidence_ms)
        max_lag = cfg.samp(cfg.corr_max_lag_ms)
        Xs = _smooth(X, cfg)
        wins = [window_around(Xs[c], t, ch_half) for c, t in cands]
        keep = [k for k, w in enumerate(wins) if w is not None]   # drop candidates at a recording edge
        order = sorted(keep, key=lambda k: cands[k][1])

        parent = list(range(len(cands)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for pos, i in enumerate(order):
            ci, ti = cands[i]
            for j in order[pos + 1:]:
                cj, tj = cands[j]
                if tj - ti > coin:                                # time-sorted -> nothing further matches
                    break
                ri, rj = find(i), find(j)
                if cj == ci or ri == rj:
                    continue
                if _norm_lag_corr(wins[i], wins[j], max_lag) >= cfg.corr_threshold:
                    parent[rj] = ri

        groups = {}
        for k in keep:
            groups.setdefault(find(k), []).append(k)
        events = []
        for ks in groups.values():
            members = [cands[k] for k in ks]
            chans = {c for c, _ in members}
            if len(chans) < cfg.min_channels:
                continue
            rep = max(ks, key=lambda k: np.ptp(wins[k]))
            events.append({'time': int(cands[rep][1]), 'channel': int(cands[rep][0]),
                           'n_channels': len(chans), 'members': members})
        return sorted(events, key=lambda e: e['time'])

    def _greedy(self, X, cands, stat):
        cfg = self.cfg
        ch_half = cfg.samp(cfg.corr_halfwin_ms)
        coin = cfg.samp(cfg.coincidence_ms)
        max_lag = cfg.samp(cfg.corr_max_lag_ms)
        Xs = _smooth(X, cfg)
        # process candidates strongest-first  # ponytail: O(n^2) per recording; fine at these rates
        order = sorted(range(len(cands)), key=lambda k: stat[cands[k][0], cands[k][1]], reverse=True)
        used = [False] * len(cands)
        events = []
        for idx in order:
            if used[idx]:
                continue
            ci, ti = cands[idx]
            used[idx] = True
            wi = window_around(Xs[ci], ti, ch_half)
            if wi is None:                      # too near a recording edge
                continue
            members = [(ci, ti)]
            for jdx in order:
                if used[jdx]:
                    continue
                cj, tj = cands[jdx]
                if cj == ci or abs(tj - ti) > coin:
                    continue
                wj = window_around(Xs[cj], tj, ch_half)
                if wj is None:
                    continue
                if _norm_lag_corr(wi, wj, max_lag) >= cfg.corr_threshold:
                    members.append((cj, tj))
                    used[jdx] = True
            chans = {c for c, _ in members}
            if len(chans) < cfg.min_channels:
                continue
            rep = max(members, key=lambda m: np.ptp(window_around(Xs[m[0]], m[1], ch_half)))
            events.append({'time': int(rep[1]), 'channel': int(rep[0]),
                           'n_channels': len(chans), 'members': members})
        return events


def event_centre(X, event, cfg, Xs=None, stat=None):
    """Sample the classifier window is centred on: the local extremum within +/-corr_halfwin_ms.
    """
    ch = event['channel']
    rc = cfg.samp(cfg.corr_halfwin_ms)
    lo, hi = max(0, event['time'] - rc), min(X.shape[1], event['time'] + rc)
    if cfg.centre_on == 'sharpness':
        s = stat if stat is not None else channel_stat(X, cfg)
        return lo + int(np.argmax(s[ch, lo:hi]))
    if cfg.centre_on != 'amplitude':
        raise ValueError(f"unknown centre_on: {cfg.centre_on}")
    seg = (Xs[ch] if Xs is not None else X[ch])[lo:hi]
    return lo + int(np.argmax(np.abs(seg - np.median(seg))))


def event_window(X, event, cfg, Xs=None, stat=None):
    """Representative-channel 2 s window for an event, re-centred per cfg.centre_on (event_centre).

    Returns a (2*half,) array, or None if the window falls off a recording edge.
    """
    half = int(round(cfg.classifier_halfwin_s * cfg.sfreq))
    return window_around(X[event['channel']], event_centre(X, event, cfg, Xs, stat), half)


def member_ptp(members, Xs, cfg):
    """Peak-to-peak of each member's +/-corr_halfwin_ms smoothed window; NaN at a recording edge."""
    half = cfg.samp(cfg.corr_halfwin_ms)
    out = []
    for c, t in members:
        w = window_around(Xs[c], t, half)
        out.append(float(np.ptp(w)) if w is not None else np.nan)
    return np.array(out)


def _rank_corr(a, b):
    """Spearman correlation, 0 where it would be degenerate (<3 points, or no spread in either)."""
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    r = spearmanr(a, b).statistic
    return 0.0 if np.isnan(r) else float(r)


def member_sign(members, Xs, cfg):
    """Sign of each member's central deflection (-1 down, +1 up); 0 at a recording edge.

    Same rule as Classifier._polarity but per member channel, against the member's own +/-corr_halfwin
    window. L2 groups on |correlation| and throws this away, so it is information the cascade discards.
    """
    half, h = cfg.samp(cfg.corr_halfwin_ms), cfg.samp(cfg.polarity_ms)
    out = []
    for c, t in members:
        w = window_around(Xs[c], t, half)
        if w is None:
            out.append(0.0)
            continue
        mid = len(w) // 2
        out.append(float(np.sign(w[mid - h:mid + h].mean() - np.median(w))))
    return np.array(out)


def spatial_features(event, Xs, cfg, pos):
    """L4 field geometry for one event, in mm: n_channels, compactness, extent, gradient.

    compactness  mean distance from the member electrodes to their centroid — is the field contiguous
    extent       max distance between any two members — separates an adjacent pair from opposite sides
    nn_distance  mean distance from each member to its closest other member. Size-free: a contiguous
                 field scores about one inter-electrode spacing whether it spans 2 channels or 15.
    gradient     rank correlation of member peak-to-peak against distance from the representative
                 channel. A real dipolar field falls off smoothly, so this is negative; 0 when fewer
                 than 3 channels, where a correlation is degenerate.
    dipole       distance between the positive- and negative-member centroids, over the extent. An
                 average reference forces both signs to appear, so what matters is whether they are
                 spatially ORGANISED (two poles) or scattered. 0 when either sign is absent.
    propagation  rank correlation of member time against distance from the earliest member — a field
                 travelling outward scores positive. 0 below 3 channels or with no time spread.
    symmetry     mean over the 8 homologous L/R pairs of 1 - |L-R|/(L+R) on member peak-to-peak, with
                 non-members counted as 0. 1 = mirrored, 0 = one-sided or midline-only.
    """
    ms = member_ptp(event['members'], Xs, cfg)
    sg = member_sign(event['members'], Xs, cfg)
    best = {}
    for (c, t), p, s in zip(event['members'], ms, sg):
        if not np.isnan(p) and p > best.get(c, (-np.inf,))[0]:
            best[c] = (p, t, s)
    chans = sorted(best)
    P = pos[chans]
    amp = np.array([best[c][0] for c in chans])
    tim = np.array([best[c][1] for c in chans], float)
    sgn = np.array([best[c][2] for c in chans])

    compact = float(np.linalg.norm(P - P.mean(0), axis=1).mean())
    extent = float(max((np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P))),
                       default=0.0))
    nn = float(np.mean([min(np.linalg.norm(P[i] - P[j]) for j in range(len(P)) if j != i)
                        for i in range(len(P))])) if len(P) > 1 else 0.0

    grad = _rank_corr(np.linalg.norm(P - pos[event['channel']], axis=1), amp)

    origin = int(np.argmin(tim))
    prop = _rank_corr(np.linalg.norm(P - P[origin], axis=1), tim - tim[origin])

    dipole = 0.0
    if extent > 0 and (sgn > 0).any() and (sgn < 0).any():
        dipole = float(np.linalg.norm(P[sgn > 0].mean(0) - P[sgn < 0].mean(0)) / extent)

    a19 = np.zeros(len(pos))
    a19[chans] = amp
    L, R = a19[[i for i, _ in HOMOLOGOUS]], a19[[j for _, j in HOMOLOGOUS]]
    tot = L + R
    sym = float(np.mean(1 - np.abs(L - R)[tot > 0] / tot[tot > 0])) if (tot > 0).any() else 0.0

    return {'n_channels': len(chans), 'compactness': compact, 'extent': extent,
            'nn_distance': nn, 'gradient': grad, 'dipole': dipole,
            'propagation': prop, 'symmetry': sym}


class Classifier:
    """L3: FDA classifier on the representative-channel 2 s window (amplitude condition).
    Smooth (B-spline) -> Fisher-Rao register to a template learned from TRAIN -> FPCA -> StandardScaler
    LogisticRegression. Fixed hyperparameters from the classification phase (in Config). Fitted on consolidated TRAIN candidates only; the registration template + FPCA basis + scaler are then frozen.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _resample(self, windows):
        """(n, L) raw windows -> B-spline-smoothed FDataGrid on an n_reg-point grid."""
        L = windows.shape[1]
        t = (np.arange(L) - L // 2) / self.cfg.sfreq
        fd = FDataGrid(windows, t)
        basis = BSplineBasis(domain_range=(t[0], t[-1]), n_basis=self.cfg.n_basis, order=4)
        sm = BasisSmoother(basis, return_basis=False).fit_transform(fd)
        tg = np.linspace(t[0], t[-1], self.cfg.n_reg_points)
        return FDataGrid(sm(tg).squeeze(-1), tg)

    def _polarity(self, windows):
        """Sign of each window's central deflection: -1 downward, +1 upward.
        """
        W = np.asarray(windows, float)
        h, c = self.cfg.samp(self.cfg.polarity_ms), W.shape[1] // 2
        return np.sign(W[:, c - h:c + h].mean(axis=1) - np.median(W, axis=1))

    def _features(self, aligned, windows):
        """FPCA scores, with the polarity column appended when cfg.polarity_feature."""
        F = self._fpca.transform(aligned)
        return np.c_[F, self._polarity(windows)] if self.cfg.polarity_feature else F

    def _registration(self):
        if self.cfg.registration == 'shift':
            return LeastSquaresShiftRegistration()
        if self.cfg.registration != 'elastic':
            raise ValueError(f"unknown registration: {self.cfg.registration}")
        return FisherRaoElasticRegistration(penalty=self.cfg.penalty)

    def fit(self, windows, y):
        fds = self._resample(windows)
        self._reg = self._registration().fit(fds)
        aligned = self._reg.transform(fds)
        self._fpca = FPCA(n_components=self.cfg.n_components).fit(aligned)
        F = self._features(aligned, windows)
        self._scaler = StandardScaler().fit(F)
        Z = self._scaler.transform(F)
        self._lr = self._fit_lr(Z, y)
        if self.cfg.hard_neg_ratio:
            self._lr = self._fit_lr(*self._mine(Z, y))
        return self

    def _fit_lr(self, Z, y):
        return LogisticRegression(C=self.cfg.lr_C, max_iter=1000,
                                  class_weight='balanced').fit(Z, y)

    def _mine(self, Z, y):
        """Hard-negative mining: keep every positive but only the negatives the first pass ranks highest.
        """
        s = self._lr.predict_proba(Z)[:, 1]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        k = min(len(neg), int(round(self.cfg.hard_neg_ratio * max(len(pos), 1))))
        keep = np.concatenate([pos, neg[np.argsort(-s[neg])[:k]]])
        return Z[keep], y[keep]

    def predict_proba(self, windows):
        aligned = self._reg.transform(self._resample(windows))
        F = self._features(aligned, windows)
        return self._lr.predict_proba(self._scaler.transform(F))[:, 1]

    def aligned_curves(self, windows):
        fds = self._resample(windows)
        return fds.data_matrix.squeeze(-1), self._reg.transform(fds).data_matrix.squeeze(-1)


class DetectionPipeline:

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.stage1 = Stage1(self.cfg)
        self.consolidator = Consolidator(self.cfg)
        self.classifier = Classifier(self.cfg)
        self.l4 = None

    def _events(self, X, bad=()):
        """(consolidated events, L1 sharpness stat) — the stat is reused for 'sharpness' centring.
        `bad` = interpolated channels, excluded from candidate generation (see Stage1.detect).
        """
        cands, stat = self.stage1.detect(X, bad)
        return self.consolidator.consolidate(X, cands, stat), stat

    @staticmethod
    def background(X):
        """Recording background amplitude: median over channels of each channel's MAD.
        """
        return max(float(np.median(np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1))),
                   1e-9)

    def _event_windows(self, X, bad=()):
        """Events with a valid 2 s window, paired with those windows (drops edge events).
        """
        Xs = _smooth(X, self.cfg)
        events, stat = self._events(X, bad)
        scale = self.background(X) if self.cfg.normalise_amplitude else 1.0
        kept, wins = [], []
        for e in events:
            w = event_window(X, e, self.cfg, Xs, stat)
            if w is not None:
                kept.append(e); wins.append(w / scale)
        return kept, wins

    def fit(self, recs):
        """recs: list of load_dataset dicts (fid, X, mk, epi, cert, ...) — TRAIN split only."""
        tol = self.cfg.samp(self.cfg.hit_tol_ms)
        pos = electrode_positions()
        W, Y, S, G = [], [], [], []
        for i, r in enumerate(recs):
            events, wins = self._event_windows(r['X'], r.get('bad_hard', ()))
            Xs = _smooth(r['X'], self.cfg)
            for e, w in zip(events, wins):
                W.append(w); S.append(self._l4_row(e, Xs, pos)); G.append(i)
                Y.append(int(bool(r['epi']) and abs(e['time'] - r['mk']) <= tol))
        W, Y = np.array(W), np.array(Y)
        self.classifier.fit(W, Y)
        self.l4 = self._fit_l4(W, Y, np.array(S, float), np.array(G)) \
            if self.cfg.spatial_feature else None
        self.train_stats_ = {'windows': len(Y), 'positives': int(Y.sum())}
        return self

    def _l4_row(self, event, Xs, pos):
        """The L4 spatial inputs for one event: (n_channels, dipole)."""
        f = spatial_features(event, Xs, self.cfg, pos)
        return [f['n_channels'], f['dipole']]

    def _fit_l4(self, W, Y, S, G):
        """Event decision: LR over (L3 score, n_channels, dipole). Returns (scaler, lr).

        The L3 scores it trains on are cross-validated over the already-fitted representation. In-sample
        scores are inflated for positives, which would make L4 lean on the score and under-use the field.
        ponytail: only the L3 LR is cross-validated, not the registration/FPCA — those are unsupervised
        and refitting them per fold here would add ~15 min to every model fit for a second-order effect.
        """
        C = self.classifier
        Z = C._scaler.transform(C._features(C._reg.transform(C._resample(W)), W))
        lr = LogisticRegression(C=self.cfg.lr_C, max_iter=1000, class_weight='balanced')
        s = cross_val_predict(lr, Z, Y, groups=G, cv=GroupKFold(5), method='predict_proba')[:, 1]
        F = np.c_[s, S]
        scaler = StandardScaler().fit(F)
        return scaler, lr.fit(scaler.transform(F), Y)

    def predict(self, X, bad=()):
        """X (19, T) -> event dicts with 'score', 'l3_score', 'polarity' and 'prominence' added.

        'score' is L4's when cfg.spatial_feature, otherwise L3's; 'l3_score' is always L3's alone.
        """
        events, wins = self._event_windows(X, bad)
        if not wins:
            return []
        W = np.array(wins)
        s3 = self.classifier.predict_proba(W)
        s = s3
        if self.l4 is not None:
            scaler, lr = self.l4
            Xs, pos = _smooth(X, self.cfg), electrode_positions()
            F = np.c_[s3, [self._l4_row(e, Xs, pos) for e in events]]
            s = lr.predict_proba(scaler.transform(F))[:, 1]
        for e, sc, s0, p, w in zip(events, s, s3, self.classifier._polarity(W), W):
            e['score'], e['l3_score'] = float(sc), float(s0)
            e['polarity'], e['prominence'] = int(p), float(np.ptp(w))
        return events

    def save(self, path):
        joblib.dump({'cfg': self.cfg, 'classifier': self.classifier, 'l4': self.l4}, path)

    @classmethod
    def load(cls, path):
        """Load a saved pipeline, refusing one whose stored Config predates a field the class now has."""
        d = joblib.load(path)
        missing = sorted({f.name for f in fields(Config)} - set(d['cfg'].__dict__))
        if missing:
            raise ValueError(
                f"{path} was saved before Config gained {missing}, which would now silently take today's "
                f"class defaults and change how this model preprocesses its input. Re-save the model from "
                f"the run notebook, or check out the code it was trained with.")
        obj = cls(d['cfg'])
        obj.classifier = d['classifier']
        obj.l4 = d.get('l4')
        return obj


def _demo():
    cfg = Config(corr_threshold=0.8)
    rng = np.random.default_rng(0)
    T = 4000
    X = rng.normal(0, 1, (3, T)) * 0.02
    bump = lambda amp, c: amp * np.exp(-0.5 * ((np.arange(T) - c) / 5.0) ** 2)
    X[0] += bump(8, 2000)          # same IED on ch0 & ch1 (scaled), ch2 flat
    X[1] += bump(5, 2000)
    X[0] += bump(8, 3400)          # a lone single-channel spike -> must be rejected

    cands, stat = Stage1(cfg).detect(X)
    for grouping in ('greedy', 'components'):
        g = replace(cfg, grouping=grouping)
        events = Consolidator(g).consolidate(X, cands, stat)
        assert len(events) == 1, f"{grouping}: expected 1 event, got {len(events)}"
        e = events[0]
        assert e['n_channels'] == 2, f"{grouping}: IED should group 2 channels, got {e['n_channels']}"
        assert abs(e['time'] - 2000) <= 15, f"{grouping}: event mislocated"
        assert e['channel'] == 0, f"{grouping}: representative should be the largest-ptp channel (ch0)"
        for mode in ('amplitude', 'sharpness'):
            c = event_centre(X, e, replace(g, centre_on=mode), _smooth(X, cfg), stat)
            assert abs(c - 2000) <= 15, f"{mode} centring mislocated ({c})"
        print(f"detection demo OK ({grouping}): {len(cands)} candidates -> {len(events)} event "
              f"(ch{e['channel']}, {e['n_channels']} channels, t={e['time']})")

    # hard-negative mining keeps every positive and only the ratio*n_pos highest-scoring negatives
    c = Classifier(replace(cfg, hard_neg_ratio=2.0))
    c._lr = type('L', (), {'predict_proba': staticmethod(lambda Z: np.c_[1 - Z[:, 0], Z[:, 0]])})()
    Zt, yt = np.array([[0.9], [0.8], [0.7], [0.1], [0.95]]), np.array([0, 0, 0, 0, 1])
    Zk, yk = c._mine(Zt, yt)
    assert yk.sum() == 1 and len(yk) == 3, f"mining kept {len(yk)} rows, expected 1 pos + 2 neg"
    assert sorted(Zk[yk == 0].ravel()) == [0.8, 0.9], "mining should keep the hardest negatives"
    print("hard-negative mining demo OK")

    # an interpolated channel must contribute no candidates, so it can never satisfy min_channels or inflate channel count
    c_all, _ = Stage1(cfg).detect(X)
    c_excl, _ = Stage1(cfg).detect(X, bad=[0])
    assert any(c == 0 for c, _ in c_all), "demo signal should give channel 0 candidates to exclude"
    assert not any(c == 0 for c, _ in c_excl), "excluded channel still produced candidates"
    assert [k for k in c_excl] == [k for k in c_all if k[0] != 0], "exclusion changed other channels"
    ev_excl = Consolidator(cfg).consolidate(X, c_excl, stat)
    assert not ev_excl, "with ch0 excluded the 2-channel IED must drop below min_channels"
    print("interpolated-channel exclusion demo OK")

    # polarity feature: sign of the central deflection, and it must not depend on scale
    g = np.exp(-0.5 * ((np.arange(1000) - 500) / 5.0) ** 2)
    pc = Classifier(replace(cfg, polarity_feature=True))
    for k in (1.0, 7.0):
        pol = pc._polarity(np.array([-g, 3 * g]) * k)
        assert list(pol) == [-1.0, 1.0], f"polarity at scale {k} should be (down, up), got {pol}"
    print("polarity-feature demo OK")

    L = 200
    bump = lambda c: np.exp(-0.5 * ((np.arange(L) - c) / 18.0) ** 2)
    pair = np.array([bump(92), bump(108)])
    sc = Classifier(replace(cfg, registration='shift', n_basis=40, n_reg_points=L))
    fds = sc._resample(pair)
    al = sc._registration().fit(fds).transform(fds).data_matrix.squeeze(-1)
    raw_peaks = [int(np.argmax(c)) for c in fds.data_matrix.squeeze(-1)]
    peaks = [int(np.argmax(c)) for c in al]
    assert min(np.ptp(al, axis=1)) > 0.5, "shift registration flattened the bumps"
    assert abs(peaks[0] - peaks[1]) < abs(raw_peaks[0] - raw_peaks[1]), \
        f"shift registration should reduce the offset: {raw_peaks} -> {peaks}"
    for mode in ('elastic', 'shift'):
        assert Classifier(replace(cfg, registration=mode))._registration() is not None
    try:
        Classifier(replace(cfg, registration='nope'))._registration()
        raise AssertionError("unknown registration should raise")
    except ValueError:
        pass
    print("shift-registration demo OK")

    # amplitude normalisation must make the classifier's windows invariant to recording scale
    p = DetectionPipeline(replace(cfg, normalise_amplitude=True))
    _, w1 = p._event_windows(X)
    _, w2 = p._event_windows(X * 10)
    assert w1 and len(w1) == len(w2), "scaling the recording changed the event set"
    assert np.allclose(w1[0], w2[0], rtol=1e-6), "normalised windows should be scale-invariant"
    raw = DetectionPipeline(replace(cfg, normalise_amplitude=False))._event_windows(X)[1][0]
    assert not np.allclose(w1[0], raw), "normalisation should actually change the windows"
    print("amplitude-normalisation demo OK")

    # L4 spatial features: geometry from real electrode positions, on events with a known layout
    from detect_data import CH19_ELECTRODES, electrode_positions
    pos = electrode_positions()
    ix = {e: i for i, e in enumerate(CH19_ELECTRODES)}
    Xf = np.zeros((19, 2000))
    for e, a in {'F7': 10.0, 'T7': 8.0, 'P7': 6.0, 'O2': 9.0}.items():
        Xf[ix[e]] += a * np.exp(-0.5 * ((np.arange(2000) - 1000) / 5.0) ** 2)
    ev = lambda els: {'members': [(ix[e], 1000) for e in els], 'channel': ix[els[0]]}
    near = spatial_features(ev(['F7', 'T7']), Xf, cfg, pos)
    far = spatial_features(ev(['F7', 'O2']), Xf, cfg, pos)
    assert near['extent'] < far['extent'], "an adjacent pair must be tighter than opposite sides of the head"
    assert near['n_channels'] == 2 and near['gradient'] == 0.0, "gradient is undefined below 3 channels"
    assert abs(near['compactness'] - near['extent'] / 2) < 1e-6, "at k=2 compactness is half the extent"
    chain = spatial_features(ev(['F7', 'T7', 'P7']), Xf, cfg, pos)
    assert chain['gradient'] < -0.9, f"amplitude falling with distance should be negative: {chain}"
    assert abs(near['nn_distance'] - chain['nn_distance']) < 20, \
        "nn_distance must be size-free: a tight pair and a tight chain should score alike"
    assert far['nn_distance'] > 2 * near['nn_distance'], "a scattered pair must score far higher"

    # dipole / propagation / symmetry, each against the field that should and should not trigger it
    def field(spec, times=None):
        """spec: {electrode: signed amplitude}; times: {electrode: sample} (default all coincident)."""
        Z = np.zeros((19, 2000))
        for e, a in spec.items():
            t = (times or {}).get(e, 1000)
            Z[ix[e]] += a * np.exp(-0.5 * ((np.arange(2000) - t) / 5.0) ** 2)
        ev2 = {'members': [(ix[e], (times or {}).get(e, 1000)) for e in spec],
               'channel': ix[max(spec, key=lambda e: abs(spec[e]))]}
        return spatial_features(ev2, Z, cfg, pos)

    two_pole = field({'F7': -10, 'T7': -8, 'F8': 9, 'T8': 7})
    one_pole = field({'F7': -10, 'T7': -8, 'F8': -9, 'T8': -7})
    assert two_pole['dipole'] > 0.5, f"two opposite poles should separate: {two_pole['dipole']:.2f}"
    assert one_pole['dipole'] == 0.0, "a single-sign field has no dipole"

    moving = field({'F7': -10, 'T7': -8, 'P7': -6},
                   times={'F7': 1000, 'T7': 1004, 'P7': 1008})
    still = field({'F7': -10, 'T7': -8, 'P7': -6})
    assert moving['propagation'] > 0.9, f"a travelling field should score high: {moving}"
    assert still['propagation'] == 0.0, "a simultaneous field has no propagation"

    mirrored = field({'F7': -10, 'F8': -10})
    onesided = field({'F7': -10, 'T7': -10})
    assert mirrored['symmetry'] > 0.95, f"equal L/R amplitudes should be symmetric: {mirrored}"
    assert onesided['symmetry'] < 0.05, f"a one-sided field should not be: {onesided}"
    print("spatial-feature demo OK")

    # L4 must re-score on top of L3 and keep the L3 score; with the flag off it must be a no-op
    p4 = DetectionPipeline(replace(cfg, spatial_feature=True))
    p4.classifier.predict_proba = lambda W: np.full(len(W), 0.5)
    p4.classifier._polarity = lambda W: np.ones(len(W))
    tr = np.array([[0.0, 2.0, 0.0], [1.0, 19.0, 0.9]])   # (L3 score, n_channels, dipole)
    ssc = StandardScaler().fit(tr)
    p4.l4 = (ssc, LogisticRegression().fit(ssc.transform(tr), [0, 1]))
    evs = p4.predict(X)
    assert evs, "demo signal should still produce an event"
    assert all(e['l3_score'] == 0.5 for e in evs), "the raw L3 score must be preserved"
    assert all(e['score'] != 0.5 for e in evs), "L4 should have re-scored the event"
    p3 = DetectionPipeline(replace(cfg, spatial_feature=False))
    p3.classifier.predict_proba = lambda W: np.full(len(W), 0.5)
    p3.classifier._polarity = lambda W: np.ones(len(W))
    assert all(e['score'] == e['l3_score'] == 0.5 for e in p3.predict(X)), "flag off must be a no-op"
    print("L4 event-decision demo OK")

    import os, tempfile
    path = os.path.join(tempfile.mkdtemp(), 'stale.joblib')
    stale = replace(cfg)
    del stale.__dict__['bandpass']
    joblib.dump({'cfg': stale, 'classifier': None}, path)
    try:
        DetectionPipeline.load(path)
        raise AssertionError("a model whose cfg predates a Config field should be refused")
    except ValueError as ex:
        assert 'bandpass' in str(ex), f"the error must name the missing field: {ex}"
    joblib.dump({'cfg': cfg, 'classifier': None}, path)
    assert DetectionPipeline.load(path).cfg == cfg, "a complete cfg must still load"
    print("stale-model guard demo OK")


if __name__ == '__main__':
    _demo()
