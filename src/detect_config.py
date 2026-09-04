"""All parameters for the IED detection. Times are in ms unless the name says otherwise.
"""
from dataclasses import dataclass

@dataclass
class Config:
    sfreq: float = 250.0  # Recordings are resampled

    bandpass: tuple = (0.5, 45.0) #(high-pass, low-pass) Hz, applied before resampling

    reference: str = 'average' # 'recorded' (as stored) or 'average'

    bad_soft_factor: float = 3.0 # MAD > this x the median across channels -> dropped from the average

    bad_hard_factor: float = 10.0 # MAD > this x the median -> removed from the analysis and interpolated.

    bad_flat_uv: float = 0.5 # peak-to-peak below this (µV) = dead/disconnected

    max_interpolate: int = 2 # more hard-bad channels than this -> interpolate NONE of them, demote

    # --- L1 candidate generation ---
    sg_ms: float = 42.0 # SG smoothing window in MS (was 21 samples @ 500 Hz).

    sg_poly: int = 3

    l1_threshold: float = 6.0 # MAD units; permissive for recall (selectivity is L2's job)

    nms_ms: float = 150.0 # within-channel merge; spans a biphasic complex

    # --- L2 consolidation ---
    corr_halfwin_ms: float = 150.0 # +/-150 ms window for cross-correlation + peak-to-peak

    coincidence_ms: float = 50.0 # max cross-channel time gap for "same event"

    corr_max_lag_ms: float = 30.0 # lag search (focal propagation)

    corr_threshold: float = 0.7 # |corr| for "same spike"; TBD on the 90-train

    min_channels: int = 2 # single-channel reject (IEDs are >=2 channels; Q3 confirmed)

    grouping: str = 'components' # 'greedy' (seed-relative: every member must match the seed) | 'components' (single-linkage: A~B and B~C -> one event)

    # --- L3 FDA classifier ---
    classifier_halfwin_s: float = 1.0   # +/-1 s -> 2 s window

    normalise_amplitude: bool = True   # divide each window by its RECORDING's background MAD - judges relative prominence rather than absolute microvolts

    centre_on: str = 'amplitude' # 'amplitude' (max |dev| of the smoothed signal) or 'sharpness' (max L1 statistic)
    
    polarity_feature: bool = True # append the sign of the central deflection to the FPCA scores as a feature.

    polarity_ms: float = 20.0 # half-width of the central mean the sign is taken from
    
    registration: str = 'elastic' # 'elastic' (Fisher-Rao nonlinear warping) or 'shift' (rigid time shift)

    spatial_feature: bool = True # L4: re-score events with an LR over (L3 score, n_channels, dipole)

    n_basis: int = 70

    penalty: float = 0.1 # elastic only; the shift registration has no warping penalty

    n_components: int = 24

    lr_C: float = 1.0

    hard_neg_ratio: float = 0.0   # hard-negative mining: keep this many hardest negatives per positive when refitting the LR (0 = off, use all negatives)

    n_reg_points: int = 100

    # --- evaluation ---
    hit_tol_ms: float = 100.0

    fp_budget: float = 10.0 # FP/min the operating point is quoted at.

    # --- split (recording-level, in detect_data) ---
    n_test: int = 0 # 0 = train on all 100; the 10-recording holdout held only 5 IEDs
    split_seed: int = 0

    def samp(self, ms):
        """ms -> samples at this sfreq."""
        return int(round(ms / 1000.0 * self.sfreq))

    def sg_samples(self):
        """Savitzky-Golay window in samples, forced ODD (savgol_filter requires an odd length).

        42 ms gives 21 samples at 500 Hz (the original hard-coded value) and 11 at 250 Hz. Without the
        odd-forcing, 250 Hz would yield 10 and savgol_filter would raise.
        """
        n = self.samp(self.sg_ms)
        return n if n % 2 else n + 1
