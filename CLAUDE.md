# CLAUDE.md — IED Detection pipeline (handover)

Handover for the **detection** phase of this MSc dissertation. The classification phase is done (see
`IED classification/`); detection is the main deliverable. Read this whole file before writing code —
many decisions below were argued out carefully and have non-obvious rationale.

**`report.md` (repo root) is the running log** — every result, every decision, and every idea that was
considered and rejected, with reasons. **It must be updated on every major addition.** This file is the
operating handover: what the task is, where things live, what not to relitigate. Detail lives in report.md.

**Standing instructions from the user:**
- **Optimise for the dissertation, not for the industrial partner.** When a better number and a more
  defensible method conflict, argue for the defensible method. Honest estimates, pre-specified choices,
  reported uncertainty and negative results beat favourable headlines.
- **Report hits / misses / false positives. Never correct rejections**, nor anything derived from them
  (specificity, accuracy, a 2×2 confusion matrix) — they are inflated by the huge pool of easy background.
  ROC-AUC is a *supporting* number for the same reason; lead with the FROC and PR-AUC.

---

## 1. The task

Detect **interictal epileptiform discharges (IEDs)** in EEG: given a recording, find *where in time* an
IED occurs (if any). On epileptic recordings the detector should fire at the IED; on non-epileptic ones
it should not mistake ordinary sharp transients (mimics) for IEDs.

**Dataset (Kural 2020):** 100 recordings, ~11–14 s each, 500 Hz, `.EDF`. **Exactly one annotated IED per
epileptic recording**, marker at the spike peak (`transient_onset_s` in `Kural_Dataset/eeg_summary.csv`).
54 epileptic (22 `clear_positive`, 32 `unclear_positive`), 46 `negative`. **Verified by neurologists:
epileptic recordings contain exactly one IED and nothing else** — so *all* non-marker EEG is trustworthy
IED-free ground truth. Total ~21 minutes. The 90-train holds **49 IEDs**, the test set **5**.

**Hard constraints from the data (state these in the writeup):**
- Ground truth is **time-only**, not per-channel. "Which channels have the IED" is always *derived*, never known.
- The marker sits mid-recording in ~99% of clips → a naive "predict centre" baseline scores well;
  any detection result must beat it (random cropping at eval is still owed).
- ~54 independent positive events is the real statistical ceiling. Augmentation (random crops only — the
  user declined background splicing) doesn't raise it.
- Cohort is **mixed focal + generalized** IEDs (generalized = bilateral synchronous, many channels;
  focal = a lobar cluster) and **the type is NOT labelled per recording**. So everything must be
  **type-agnostic**. "Number of channels" is therefore a *bimodal, weak* spatial feature — do not treat
  "more channels = more IED-like".
- **Neuronostics** will supply external data for **test only** (never training). Keep the 10 test
  recordings here untouched too.

---

## 2. Architecture — the 4-layer cascade (signed off)

| Layer | Does | Reference |
|---|---|---|
| **L1 `Stage1`** | multichannel SG 2nd-deriv sharpness, MAD-normalised → `(channel, time)` candidates; tuned for **recall** | recorded |
| **L2 `Consolidator`** | single-channel reject; cross-correlate to group the same spike into **events**; representative channel by **peak-to-peak** | recorded |
| **L3 `Classifier`** | FDA classifier (smooth→register→FPCA→LR) retrained on consolidated candidates, single channel → IED score | recorded |
| **L4 event decision** | combine L3 score + spatial features → detection | **re-referenced** (avg/bipolar) |

**v1 = L1→L2→L3, recorded reference throughout. Built and running.** v2 = L4 (spatial features,
re-referencing) — not started, and now the main remaining FP lever.

**System metric = the cross-validated FROC** (sensitivity vs false-positives/min), computed from
out-of-fold scores over the **90-train (49 IEDs)**. The 10-recording test FROC exists but moves in steps
of 0.2 and settles nothing. Plus localisation error (ms) and the centre baseline.

Why a cascade and not "slide the classifier": every classification-phase training window was centred on a
transient, so L3 has never seen background — sliding it puts it out-of-distribution. The cascade keeps L3
in-distribution (it only ever sees consolidated candidate windows).

---

## 3. Current state — what's built & tested

- **`src/detect_config.py`** — `Config` dataclass: every tunable in one place (§6). Imported everywhere.
- **`src/detect_data.py`** — `CH19` (19 common 10-20 channels), `load_recording`/`load_recording_path`/
  `load_dataset` (full recordings, µV, `reference='recorded'|'average'`), `stratified_split`
  (recording-level on `certainty`, `n_test=10, seed=0`), `window_around` (edge-safe). `__main__` self-check.
- **`src/detect_stage1.py`** — L1 primitives: `sharpness`, `channel_stat`, `candidates` (find_peaks + NMS),
  `n_channels_crossing`. `__main__` self-check.
- **`src/detection.py`** — the cascade. `Stage1`; `Consolidator` with **two grouping rules** (§5);
  `event_centre`/`event_window` (window cutting, `cfg.centre_on`); `Classifier` (FDA; `_mine`/`_fit_lr` for
  hard-negative mining, `aligned_curves` for diagnostics); `DetectionPipeline`
  (`.fit`/`.predict`/`.save`/`.load`). `__main__` self-check covers both grouping rules and mining.
- **`src/detect_metrics.py`** — evaluation only. `labelled_events` is the single L1+L2 pass everything is
  built on; `froc_points`, `localisation_errors`, `detection_counts` and `bootstrap_auc_ci` then work off
  **any** score vector, so the same code serves the test set and out-of-fold CV scores. Plus
  `layer1_recall`, `consolidation_recall`, `classifier_auc`, `grouped_cv`, `froc`, `froc_summary`,
  `threshold_at`, `centre_baseline`, `centring_offsets`, `report`. `FP_BUDGET = 10` sets the operating
  point hits/misses/FPs are quoted at.
- **`src/detect_plots.py`** — one function per figure panel, each drawing into an `ax`: `froc`, `roc`, `pr`,
  `counts`, `score_hist`, `fold_aucs`, `timeline`, `loc_error`. Shared by the test and CV cells. Computes
  no metric of its own.
- **`predict.py`** + **`requirements.txt`** (pinned) — supervisor entrypoint:
  `python predict.py <model.joblib> <recording.edf> [--score]`.
- **`IED Detection/detection_run.ipynb`** — split → fit(90) → window-centring check (train only) → test
  report + figures → save model → **grouped CV** → three sweeps. ~5 min for the main path.
  - The grouped CV and all three sweeps are **flag-guarded** (`RUN_GROUPED_CV`, `n_components_sweep`,
    `n_basis_sweep`, `hard_neg_sweep`) so Run All stays fast. Keep that pattern for anything slow.
- Experimental notebooks (reference only, don't extend): `detection_stage1.ipynb` (L1 feasibility),
  `detection_threshold_diagnostic.ipynb` (per-channel selectivity / reference study).

---

## 4. Where the numbers stand

**Grouped 5-fold CV on the 90-train** (`components`/0.7, `n_basis=70`, `n_components=24`; 4,330 windows,
64 positives, 49 IEDs):

| | |
|---|---|
| ROC-AUC | **0.828**, 95% CI [0.763, 0.889] (recording-level bootstrap) |
| PR-AUC | **0.117** against a chance rate of 0.0148 (~8× chance) |
| per fold | 0.750 · 0.749 · 0.872 · 0.939 · 0.831 |
| localisation | 16 ms median over 49 IEDs |
| FROC sens @1/5/10/25/50/100 FP/min | 0.08 / **0.41** / 0.59 / 0.71 / 0.84 / 0.92 |
| centre baseline | 0.29 @ 4.0 FP/min |

**The detector now clears the do-nothing baseline** (0.41 vs 0.29 at ≤5 FP/min). It did not before —
at `n_components=4` it scored 0.04 there, seven times *worse* than guessing the recording's midpoint.

**Held-out test (5 IEDs, the only selection-free estimate left):** ROC-AUC 0.847 — agreeing with the CV,
which is the best evidence the CV number isn't badly inflated by the sweeps. Its FROC left end (0.20 vs
baseline 0.40) is one IED versus two; don't read it either way.

**Front end is not the problem:** L1 recall 1.00, L2 recall 1.00, **1.00 events per IED**, 5.28 channels
per event. **L3 discrimination is the bottleneck** — at ≤10 FP/min the CV finds 29 of 49 IEDs and raises
~190 false positives (precision ≈ 0.13).

**Remaining work, in order** (full detail in report.md §8):
1. Hard-negative mining sweep (`hard_neg_ratio`) — built, not yet run on the full 90.
2. One notebook re-run at the final config to refresh the model, figures and test report.
3. Pilot script for the Neuronostics data (spec in report.md §9) — staged 5-per-group validation run first.
4. Random-crop evaluation, so the centre baseline is a fair bar.
5. One confirmatory greedy-vs-components run at the tuned k (that comparison was made at k=4).
6. **v2 = L4** — spatial features on re-referenced data, combined with the L3 score. The real FP lever,
   and now viable: components grouping captures 15 of 17 involved channels instead of 6.

---

## 5. Key decisions & findings (do not relitigate — evidence in report.md)

**L1 threshold = 6 (MAD), permissive on purpose.** 100% IED recall. Not lowered by mistake — selectivity is
L2's and L3's job. Raising it to get few channels destroys recall (th14–16 → 2–4 channels but 57–63%
recall). At threshold 6 a **median 17 of 19 channels cross at the marker**, largely real IED spread
(background is gone by th8 while 12 channels still cross), amplified by per-channel MAD normalisation.

**L2 grouping — `cfg.grouping='components'`, `corr_threshold=0.7`.** Two rules exist:
- `'greedy'` (original): sharpest candidate seeds an event, every member must correlate with **that seed**.
  Star-shaped — and a discharge's field is a *chain* (A~B, B~C, A≁C across a dipole), so it fragmented one
  IED into **2.86 events** and captured only 6 of the 17 involved channels.
- `'components'` (now default): single-linkage — events are connected components of the "same spike" graph.
  **1.31 events per IED, 15 of 17 channels, L2 recall still 1.00.**
- **The threshold direction reverses under single-linkage**: *lowering* `corr_threshold` de-fragments.
  The old plan to raise it to 0.85–0.9 is abandoned. 0.6 is the floor — recall drops to 0.98 there as
  chaining drags an event's representative >100 ms off the marker.
- Honest caveat: de-fragmentation removed an *artificial* sensitivity advantage (the FROC counts a hit if
  *any* near-marker event clears threshold, so 2.86 events per IED meant ~3 attempts). At k=4 the matched
  FROC comparison was a wash; components was kept for field capture, defensibility and runtime.

**Cross-correlation design (unchanged, applies to both rules):** must be **normalised** (amplitude-invariant),
**lag-searched** (±30 ms, propagation) and **absolute-valued** (polarity-invariant — a dipole's far side is
reversed and raw zero-lag correlation would drop exactly those channels). Correlate the **SG-smoothed**
signal over **±150 ms** (must span the whole biphasic complex).

**Single-channel reject is the only hard filter.** IEDs cannot be single-channel; focal ones may be exactly
2 adjacent channels, so the floor is **≥2, never 3–4**. Filtering philosophy: hard-filter only *unambiguous*
non-IEDs; everything ambiguous (amplitude, channel count, field) is a **soft feature for L4**. Every hard
filter is a permanent recall cap — measure IED recall after each.

**`n_components` was the single biggest defect** (4 → 24 took CV ROC-AUC 0.657 → 0.828 and sensitivity at
≤5 FP/min 0.04 → 0.41). FPCA orders components by *variance* and the pool is 98.5% negatives, so the
IED-discriminative direction is not in the leading few — truncating at 4 threw it away. This is the class
imbalance showing up in the **representation**, but the fix needed no class-informed fitting at all.

**`n_basis` makes no consistent difference — the spike-bandwidth argument is REFUTED.** The prediction
(30 B-splines over 2 s ≈ 7 Hz ceiling, too coarse for a 20–70 ms spike) failed: across a 19-cell grid,
`n_basis` 30 and 100 perform the same at matched k. What matters is the *interaction* — good cells sit near
**k ≈ 0.3–0.4 × n_basis**. Kept at 70 only because it leaves headroom for larger k.

**AUC and the FROC's left end can disagree, and the FROC wins.** AUC scores the whole ranking; a low FP
budget depends only on its very top. Extra components sharpened the top while adding noise to the middle.

**Window centring — settled, keep `'amplitude'`.** `'sharpness'` aligns far better to the marker
(median |offset| 20 vs 54 ms; 1% vs 22% beyond 100 ms) but **costs nothing in AUC** — recording-level
bootstrap of the difference = +0.051 favouring `'amplitude'`, 95% CI [0.000, +0.105]. FPCA is
amplitude-sensitive, so pinning the largest deflection at t=0 is the stronger regularity, and Fisher-Rao
warping absorbs the residual shift. Centring is **ruled out** as the bottleneck. (Measured pre-components;
under the new grouping the misalignment largely fixed itself anyway, 54 → 26 ms.)
- **Don't rank centring rules by how tightly the windows stack** — an "alignment spread" statistic was built
  and deleted as circular: every rule maximises self-similarity at its own centre. The offset to the
  neurologist marker is the only measure using information from outside the rule being tested.

**Reference contamination is real but only matters for L4.** A large IED near the shared reference copies
onto all channels (S65 diffuse under recorded, focal under average). **Crucial nuance: contamination
spreads the spike in space, not time**, so the IED is still detected at the correct *time* — it corrupts
only the secondary spatial question. **Do NOT re-reference L1/L2/L3**; re-referencing (average, and try
bipolar for focal) enters at L4 only.

**Metric semantics:** recall is measured over **IEDs only** — missing a *mimic* is harmless (a mimic never
proposed can't become a false positive). Mimics are counted where they matter: as false positives in the
FROC.

**Selection optimism is now live.** `n_components` and `n_basis` were both chosen on the 90-train CV over
~19 configurations. At 49 IEDs the binomial SE on a sensitivity near 0.4 is ~0.07, so the quoted maxima are
optimistic by perhaps 0.05–0.10. Declare this in the writeup; do not present a tuned maximum as clean.

---

## 6. Config — every tunable (in `src/detect_config.py`, class `Config`)

```
sfreq=500; sg_win=21; sg_poly=3; l1_threshold=6.0; nms_ms=150
corr_halfwin_ms=150; coincidence_ms=50; corr_max_lag_ms=30; corr_threshold=0.7; min_channels=2
grouping='components'
classifier_halfwin_s=1.0; centre_on='amplitude'; n_basis=70; penalty=0.1; n_components=24; lr_C=1.0
hard_neg_ratio=0.0; n_reg_points=100
hit_tol_ms=100; n_test=10; split_seed=0
```
`Config.samp(ms)` converts ms→samples. **All thresholds/windows live here — change them here, nowhere else.**

---

## 7. Leakage discipline (the user is strict about this)

- Recording-level split (`stratified_split`, seed 0), whole recordings in/out.
- **Everything fitted on the 90 only:** L1 threshold, `corr_threshold`, L3 (registration template, FPCA
  basis, scaler, LR weights), and hard-negative selection. L2/L3 training candidates come only from the 90.
- Anything label-derived and *fitted* (registration template, FPCA basis, mined negatives) must be fitted
  **inside each CV fold**, never on the whole train set before splitting.
- The **10 test recordings** are the only selection-free estimate left. Touch them once, at the end.
- Freeze the split seed and all fitted params into the `.joblib` so external runs are reproducible.

---

## 8. Gotchas

- **Stale imports are the #1 time-waster here.** A live Jupyter kernel keeps the old `detect_config` /
  `detect_metrics`, so editing a value silently does nothing — it made an `n_components` change look like
  it had "no effect", and later produced `module 'detect_metrics' has no attribute 'grouped_cv'`.
  Cell 1 of the run notebook now starts with `%autoreload 2`; for a final model run, **Restart & Run All**.
- Figures convention: **`dpi=800`**, `bbox_inches='tight'`, into `figures/`.
- Channel names are `E FP1-Ref` etc.; EEG channels start with `"E "`, EKG is `"P EKG"`. For mne
  montage/topomaps map `'E FP1-Ref'→'Fp1'` then `standard_1020`.
- Runtime: L1+L2 over all 90 recordings is ~8 s; **the L3 Fisher-Rao registration dominates everything**
  (~3–4 min per fit, so a grouped CV is ~13 min). Sweeps that only change what sits *downstream* of
  registration (`n_components`, `lr_C`, `hard_neg_ratio`) share one registration per fold and are then
  nearly free — use that pattern. `n_basis` is upstream and needs a full refit per value.
- `Consolidator._greedy` is O(seeds × candidates); on a 20-minute recording that is ~tens of seconds
  (an earlier note claiming it "would not finish" was wrong). `_components` is roughly linear.
- Build via generator scripts in the scratchpad + headless smoke tests (patch `plt.savefig`/`plt.show` to
  no-ops) — the pattern used throughout; keeps notebook JSON valid and avoids writing bogus figures.
- Expect a **high raw event rate** (~262/min on train before scoring). Normal — L3 and later L4 cut it.

---

## 9. Parking lot — rough ideas, not decisions (delete freely)

**Multi-view event scoring (raised 2026-08-05).** The FROC counts sensitivity as "any near-marker event
above threshold" but counts *every* off-marker event as its own FP. So fragmentation was asymmetric: it
inflated sensitivity (several attempts per IED) and inflated FP/min (several FPs per mimic). Fixing the
grouping removed both. Idea: get the multiple views back deliberately, without the artifact —
- score **several member channels per event** (top-k by peak-to-peak) instead of only the representative,
  then aggregate to one event score (mean or top-k mean; **not** max — max gives mimics k lucky attempts too);
- an event now spans ~5 channels (15 for the largest), so the views are real signals, not clustering accidents;
- side effect: training positives go 64 → ~330, and unlike fragmentation duplicates these are genuinely
  different channels rather than near-copies;
- this *is* L4 — aggregation stats sit next to the spatial features in the event decision.
- Costs ~3-5x L3 registration time (fit ~20 min, grouped CV >1 h). Adds no information: still 49 events.
- Cheap pre-test before building any of it: take greedy's out-of-fold scores, merge events within ~150 ms
  post-hoc, aggregate by max and by mean, recompute the CV FROC. No refitting. If merged-greedy doesn't
  beat `components`, drop the whole idea.

**Class-informed registration templates / weighted FPCA — parked by the supervisor**, who doubts they would
generalise to other EEG recordings. Note the `n_components` fix achieved the same goal (recovering
IED-discriminative directions from a negative-dominated pool) with no class-informed fitting at all.
