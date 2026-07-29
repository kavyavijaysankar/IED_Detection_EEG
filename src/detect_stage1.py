"""Stage-1 IED candidate detector: Savitzky-Golay 2nd-derivative sharpness, MAD-normalised.

The front end for the detection cascade: flags where a signal is sharp (high curvature), per channel.
Not a classifier - it thresholds a signal-processing statistic. Shared by the feasibility and
threshold-diagnostic notebooks so the detector is defined once.
"""
import numpy as np
from scipy.signal import savgol_filter, find_peaks

SG_WIN, SG_POLY = 21, 3   # 42 ms window @ 500 Hz, cubic; near spike width, preserves peaks


def sharpness(x, sf=500.0):
    """Per-channel MAD-normalised |SG 2nd derivative| (channel-adaptive sharpness units)."""
    d2 = savgol_filter(x, SG_WIN, SG_POLY, deriv=2, delta=1.0 / sf)
    s = np.abs(d2)
    return s / (np.median(np.abs(s - np.median(s))) + 1e-12)


def channel_stat(X, sf=500.0):
    """(n_ch, T) sharpness array for a multichannel recording X (n_ch, T)."""
    return np.stack([sharpness(X[c], sf) for c in range(X.shape[0])])


def candidates(stat, thresh, distance):
    """Peak-picked candidates [(channel, sample)] from a sharpness array; distance = NMS in samples."""
    out = []
    for c in range(stat.shape[0]):
        pk, _ = find_peaks(stat[c], height=thresh, distance=distance)
        out += [(c, int(p)) for p in pk]
    return out


def n_channels_crossing(stat, thresh, t0, tol):
    """How many channels have a supra-threshold sample within +/-tol of time t0 (sample index)."""
    lo, hi = max(0, t0 - tol), min(stat.shape[1], t0 + tol)
    return int(np.sum(stat[:, lo:hi].max(axis=1) >= thresh))


def _demo():
    sf = 500.0
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000) * 0.02
    x += 8 * np.exp(-0.5 * ((np.arange(2000) - 1000) / 5.0) ** 2)   # sharp bump (~20 ms) at sample 1000
    s = sharpness(x, sf)
    assert abs(int(np.argmax(s)) - 1000) <= 10, "sharpness should peak at the spike"
    st = channel_stat(x[None, :], sf)
    assert any(abs(p - 1000) <= 10 for _, p in candidates(st, thresh=6, distance=75)), "spike not detected"
    assert n_channels_crossing(st, 6, 1000, 50) == 1, "crossing count wrong"
    print("detect_stage1 demo OK")


if __name__ == '__main__':
    _demo()
