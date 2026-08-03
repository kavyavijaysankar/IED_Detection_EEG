"""IED detection cascade (v1): candidate generation -> consolidation -> classification.

One module, one class per stage, plus a `Config` dataclass holding every tunable in one place, and
(to come) a `DetectionPipeline` that fits/predicts/saves the whole cascade. Data loading/splitting
lives in `detect_data`; evaluation metrics in `detect_metrics`.

Layers:
  L1 Stage1        multichannel SG 2nd-derivative candidate generation      (recorded reference)
  L2 Consolidator  cross-correlation grouping -> events, single-channel reject, representative channel
  L3 Classifier    FDA classifier on the representative 2 s window           (to come)
  L4 event decision + spatial features                                       (v2, not in v1)
"""
from dataclasses import replace

import joblib
import numpy as np
from scipy.signal import savgol_filter
from skfda import FDataGrid
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import FisherRaoElasticRegistration
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from detect_config import Config
from detect_data import window_around
from detect_stage1 import channel_stat, candidates


def _smooth(X, cfg):
    """SG-smoothed signal (deriv=0) for morphology correlation and peak-to-peak."""
    return savgol_filter(X, cfg.sg_win, cfg.sg_poly, axis=-1)


def _norm_lag_corr(a, b, max_lag):
    """Max over lags of |normalised cross-correlation| between two equal-length windows.

    Normalised -> amplitude-invariant (a diminished copy still matches); lag search -> a delayed copy
    still matches; absolute value -> polarity-invariant (opposite side of a dipole still matches).
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

    def detect(self, X):
        """X (n_ch, T) -> (candidates [(ch, sample)], sharpness stat (n_ch, T))."""
        stat = channel_stat(X, self.cfg.sfreq)
        cands = candidates(stat, self.cfg.l1_threshold, self.cfg.samp(self.cfg.nms_ms))
        return cands, stat


class Consolidator:
    """L2: group per-channel candidates of the same spike into events via cross-correlation.

    Greedy by sharpness: the sharpest unused candidate seeds an event; any nearby-in-time candidate on
    another channel whose window correlates >= corr_threshold joins it. Events with < min_channels are
    rejected (single-channel = artefact). Representative channel = max peak-to-peak over the window.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def consolidate(self, X, cands, stat):
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

    Registration is alignment-sensitive, so this choice matters (see cfg.centre_on):
      'amplitude'  max |deviation from the local median| of the SMOOTHED signal (Xs, if given, so a
                   noise sample can't grab the centre). Matches the annotation convention (marker at the
                   spike peak) and how the classification-phase training windows were cut — but a spike
                   riding a slow wave or a baseline shift can hand the centre to the drift instead.
      'sharpness'  max of the L1 statistic (|SG 2nd deriv| / MAD, passed in as `stat` or recomputed).
                   Curvature peaks at the apex of a sharp transient, so drift can't win; effectively
                   keeps the L1 candidate time, since event['time'] is already a peak of that statistic.
    """
    ch = event['channel']
    rc = cfg.samp(cfg.corr_halfwin_ms)
    lo, hi = max(0, event['time'] - rc), min(X.shape[1], event['time'] + rc)
    if cfg.centre_on == 'sharpness':
        s = stat if stat is not None else channel_stat(X, cfg.sfreq)
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


class Classifier:
    """L3: FDA classifier on the representative-channel 2 s window (amplitude condition).

    Smooth (B-spline) -> Fisher-Rao register to a template learned from TRAIN -> FPCA -> StandardScaler
    -> LogisticRegression. Fixed hyperparameters from the classification phase (in Config). Fitted on
    consolidated TRAIN candidates only; the registration template + FPCA basis + scaler are then frozen.
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

    def fit(self, windows, y):
        fds = self._resample(windows)
        self._reg = FisherRaoElasticRegistration(penalty=self.cfg.penalty).fit(fds)
        aligned = self._reg.transform(fds)
        self._fpca = FPCA(n_components=self.cfg.n_components).fit(aligned)
        F = self._fpca.transform(aligned)
        self._scaler = StandardScaler().fit(F)
        self._lr = LogisticRegression(C=self.cfg.lr_C, max_iter=1000,
                                      class_weight='balanced').fit(self._scaler.transform(F), y)
        return self

    def predict_proba(self, windows):
        aligned = self._reg.transform(self._resample(windows))
        F = self._fpca.transform(aligned)
        return self._lr.predict_proba(self._scaler.transform(F))[:, 1]

    def aligned_curves(self, windows):
        """(smoothed, registered) curve matrices (n, n_reg_points) — the registration diagnostic.

        Overlaying the registered curves shows whether the windows the LR actually sees are aligned;
        badly centred windows warp to the template instead of stacking on the spike.
        """
        fds = self._resample(windows)
        return fds.data_matrix.squeeze(-1), self._reg.transform(fds).data_matrix.squeeze(-1)


class DetectionPipeline:
    """The v1 cascade: L1 Stage1 -> L2 Consolidator -> L3 Classifier. Fit on TRAIN recordings only.

    predict(X) returns the recording's events, each with an 'score' (IED probability). save/load persist
    Config + the fitted classifier so the model runs identically on external EDFs (load them with
    detect_data.load_recording so the preprocessing contract — 500 Hz, CH19, µV — matches).
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.stage1 = Stage1(self.cfg)
        self.consolidator = Consolidator(self.cfg)
        self.classifier = Classifier(self.cfg)

    def _events(self, X):
        """(consolidated events, L1 sharpness stat) — the stat is reused for 'sharpness' centring."""
        cands, stat = self.stage1.detect(X)
        return self.consolidator.consolidate(X, cands, stat), stat

    def _event_windows(self, X):
        """Events with a valid 2 s window, paired with those windows (drops edge events)."""
        Xs = _smooth(X, self.cfg)
        events, stat = self._events(X)
        kept, wins = [], []
        for e in events:
            w = event_window(X, e, self.cfg, Xs, stat)
            if w is not None:
                kept.append(e); wins.append(w)
        return kept, wins

    def fit(self, recs):
        """recs: list of load_dataset dicts (fid, X, mk, epi, cert, ...) — TRAIN split only."""
        tol = self.cfg.samp(self.cfg.hit_tol_ms)
        W, Y = [], []
        for r in recs:
            events, wins = self._event_windows(r['X'])
            for e, w in zip(events, wins):
                W.append(w)
                Y.append(int(bool(r['epi']) and abs(e['time'] - r['mk']) <= tol))
        W, Y = np.array(W), np.array(Y)
        self.classifier.fit(W, Y)
        self.train_stats_ = {'windows': len(Y), 'positives': int(Y.sum())}
        return self

    def predict(self, X):
        """X (19, T) -> list of event dicts with an added 'score' (IED probability)."""
        events, wins = self._event_windows(X)
        if not wins:
            return []
        for e, s in zip(events, self.classifier.predict_proba(np.array(wins))):
            e['score'] = float(s)
        return events

    def save(self, path):
        joblib.dump({'cfg': self.cfg, 'classifier': self.classifier}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        obj = cls(d['cfg'])
        obj.classifier = d['classifier']
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
    events = Consolidator(cfg).consolidate(X, cands, stat)

    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    e = events[0]
    assert e['n_channels'] == 2, f"IED should group 2 channels, got {e['n_channels']}"
    assert abs(e['time'] - 2000) <= 15, "event mislocated"
    assert e['channel'] == 0, "representative should be the largest-ptp channel (ch0)"
    for mode in ('amplitude', 'sharpness'):
        c = event_centre(X, e, replace(cfg, centre_on=mode), _smooth(X, cfg), stat)
        assert abs(c - 2000) <= 15, f"{mode} centring mislocated ({c})"
    print(f"detection demo OK: {len(cands)} candidates -> {len(events)} event "
          f"(ch{e['channel']}, {e['n_channels']} channels, t={e['time']})")


if __name__ == '__main__':
    _demo()
