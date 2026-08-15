"""All tunables for the IED detection cascade, in one place.

Change thresholds/windows HERE, nowhere else. Imported by detection.py, detect_metrics.py, the run
notebook and predict.py. Times are in ms unless the name says otherwise.
"""
from dataclasses import dataclass

SF = 500.0


@dataclass
class Config:
    sfreq: float = SF
    # --- L1 candidate generation ---
    sg_win: int = 21           # SG smoothing window (samples); ~42 ms
    sg_poly: int = 3
    l1_threshold: float = 6.0  # MAD units; permissive for recall (selectivity is L2's job)
    nms_ms: float = 150.0      # within-channel merge; spans a biphasic complex
    # --- L2 consolidation ---
    corr_halfwin_ms: float = 150.0   # +/-150 ms window for cross-correlation + peak-to-peak
    coincidence_ms: float = 50.0     # max cross-channel time gap for "same event"
    corr_max_lag_ms: float = 30.0    # lag search (focal propagation)
    corr_threshold: float = 0.7      # |corr| for "same spike"; TBD on the 90-train
    min_channels: int = 2            # single-channel reject (IEDs are >=2 channels; Q3 confirmed)
    grouping: str = 'components'         # 'greedy' (seed-relative: every member must match the seed) |
                                     # 'components' (single-linkage: A~B and B~C -> one event)
    # --- L3 classifier (FDA; fixed hyperparams from the classification phase) ---
    classifier_halfwin_s: float = 1.0   # +/-1 s -> 2 s window
    centre_on: str = 'amplitude'        # window centring: 'amplitude' (max |dev| of the smoothed
                                        # signal, matches the spike-peak marker) | 'sharpness' (max of
                                        # the L1 statistic; immune to slow-wave/drift capture)
    n_basis: int = 70
    penalty: float = 0.1
    n_components: int = 24
    lr_C: float = 1.0
    hard_neg_ratio: float = 0.0   # hard-negative mining: keep this many hardest negatives per positive
                                  # when refitting the LR (0 = off, use all negatives)
    n_reg_points: int = 100
    # --- evaluation ---
    hit_tol_ms: float = 100.0
    # --- split (recording-level, in detect_data) ---
    n_test: int = 10
    split_seed: int = 0

    def samp(self, ms):
        """ms -> samples at this sfreq."""
        return int(round(ms / 1000.0 * self.sfreq))
