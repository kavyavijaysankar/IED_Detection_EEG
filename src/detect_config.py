"""All tunables for the IED detection cascade, in one place.

Change thresholds/windows HERE, nowhere else. Imported by detection.py, detect_metrics.py, the run
notebook and predict.py. Times are in ms unless the name says otherwise.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # --- preprocessing (applied in detect_data.load_recording_path, before L1) ---
    sfreq: float = 250.0       # THE single source of truth for the sample rate. Recordings are resampled
                               # to this on load, so nothing downstream may hard-code a rate: everything
                               # is specified in ms and converted with samp() / sg_samples().
                               # 250 Hz agreed with the supervisor 2026-08-17 (Kural is 500 Hz native).
                               # Note resampling ALWAYS low-passes near the new Nyquist (~125 Hz) — that is
                               # the anti-aliasing filter, applied by mne's FFT-based resample.
    bandpass: tuple = (0.5, 45.0)  # (high-pass, low-pass) Hz, applied BEFORE resampling. 0.5 Hz chosen by
                               # the supervisor to preserve the after-going slow wave (0.5-3 Hz), which is
                               # part of the discharge and sits inside the 2 s window L3 sees. NO mains
                               # notch is needed: 45 Hz already excludes 50 and 60 Hz. If the upper edge
                               # ever goes above 50 (e.g. 0.5-70 for IED review), a notch must come back.
                               # None disables filtering.
    reference: str = 'average'    # 'recorded' (as stored) | 'average' (common-average over the GOOD
                                  # channels). Average reference moved ahead of L1 on 2026-08-17, reversing
                                  # the earlier "L4 only" decision: a recorded reference differs between
                                  # labs, an average one does not, so it is the stronger transfer choice.
    # --- bad channels (two thresholds, matched to the cost of being wrong) ---
    bad_soft_factor: float = 3.0   # MAD > this x the median across channels -> dropped from the average
                                   # reference computation but KEPT in the analysis. Cheap if wrong (the
                                   # average is over 18 channels instead of 19), so it can be sensitive:
                                   # clean Kural has p99 = 1.88 and only 1 of 1710 pairs above 3x.
    bad_hard_factor: float = 10.0  # MAD > this x the median -> removed from the analysis and interpolated.
                                   # EXPENSIVE if wrong (real data replaced by a synthetic estimate), so it
                                   # is deliberately far above clean variation: Kural's max is 8.28x and
                                   # nothing at all trips 10x. Real electrode failures are catastrophic,
                                   # not marginal, so they clear this easily.
    bad_flat_uv: float = 0.5       # peak-to-peak below this (µV) = dead/disconnected -> hard. NOT a MAD
                                   # test: low MAD means a quiet baseline, not a dead channel. Kural's
                                   # quietest real channel has MAD 0.012 µV but ptp 2.3 µV and thousands of
                                   # distinct values — a MAD-based floor would have interpolated away good
                                   # reference-adjacent midline channels (Pz/Cz sit near Kural's reference).
    max_interpolate: int = 2       # more hard-bad channels than this -> interpolate NONE of them, demote
                                   # them all to soft, and flag the recording. Spherical splines over 19
                                   # electrodes reconstruct 1-2 routinely; beyond that it is fiction.
    # --- L1 candidate generation ---
    sg_ms: float = 42.0        # SG smoothing window in MS (was 21 samples @ 500 Hz). In ms so it means the
                               # same thing at any sample rate — see sg_samples().
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
    normalise_amplitude: bool = True   # divide each window by its RECORDING's background MAD, so L3
                                        # judges relative prominence rather than absolute microvolts
    centre_on: str = 'amplitude'        # window centring: 'amplitude' (max |dev| of the smoothed
                                        # signal, matches the spike-peak marker) | 'sharpness' (max of
                                        # the L1 statistic; immune to slow-wave/drift capture)
    polarity_feature: bool = True      # append the sign of the central deflection to the FPCA scores as
                                        # one extra feature. L1 (|2nd deriv|) and L2 (|corr|) are
                                        # polarity-blind by design; Phase 2 measured that L3 isn't using
                                        # sign either (IEDs ~75% downward, top FPs 40%), so hand it over.
    polarity_ms: float = 20.0           # half-width of the central mean the sign is taken from (+/-10
                                        # samples at 500 Hz). Pre-specified from the Phase-2 measurement,
                                        # then confirmed MID-PLATEAU by a robustness sweep: 4-20 ms score
                                        # identically, while 80 ms mixes in the opposite-polarity slow wave
                                        # and reverts to baseline. SETTLED — do not sweep it again.
    registration: str = 'elastic'       # 'elastic' (Fisher-Rao: nonlinear warping, aligns tightly but
                                        # RESHAPES curves) | 'shift' (rigid time shift only, shape-
                                        # preserving). Measured defect driving this switch: elastic warping
                                        # moves IEDs by ~1.25x their own signal but mimics only 0.85x,
                                        # because the template is a Karcher mean over a 98.5%-negative
                                        # pool, i.e. a mimic shape. See report.md.
    n_basis: int = 70
    penalty: float = 0.1                # elastic only; the shift registration has no warping penalty
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

    def sg_samples(self):
        """Savitzky-Golay window in samples, forced ODD (savgol_filter requires an odd length).

        42 ms gives 21 samples at 500 Hz (the original hard-coded value) and 11 at 250 Hz. Without the
        odd-forcing, 250 Hz would yield 10 and savgol_filter would raise.
        """
        n = self.samp(self.sg_ms)
        return n if n % 2 else n + 1
