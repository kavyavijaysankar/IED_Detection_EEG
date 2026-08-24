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
- **It must generalise, not just score well here.** The goal is not the best number on Kural or on the
  Neuronostics data — it is a detector that transfers. A change that is neutral on the CV but makes the
  model scale-, site- or acquisition-invariant is worth adopting *on those grounds*, and should be argued
  that way in the writeup rather than dressed up as a performance gain (see amplitude normalisation, §5).

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
- **`src/detect_data.py`** — `CH19` (19 common 10-20 channels), `electrode`/`match_channels` (the
  external-data naming trust boundary: strips reference suffix + modality prefix, exact-matches, aliases
  the old `T3/T4/T5/T6`, and **refuses** missing / ambiguous / bipolar montages — added 2026-08-18, see
  report.md §9 pre-flight), `load_recording`/`load_recording_path`/
  `load_dataset` — all three take **`cfg`** (not a `reference` string) since Phase 0. `load_recording_path`
  is **the single preprocessing entry point**: everything applied to the signal before L1 goes there and
  nowhere else, driven entirely by `cfg`, so a run can't preprocess differently from training. Also
  `stratified_split` (recording-level on `certainty`, `n_test=10, seed=0`), `window_around` (edge-safe).
  `__main__` self-check.
- **`src/detect_stage1.py`** — L1 primitives: `sharpness(x, cfg)`, `channel_stat(X, cfg)`, `candidates`
  (find_peaks + NMS), `n_channels_crossing`. `__main__` self-check (includes the odd-SG-window invariant).
- **`src/detection.py`** — the cascade. `Stage1`; `Consolidator` with **two grouping rules** (§5);
  `event_centre`/`event_window` (window cutting, `cfg.centre_on`); `Classifier` (FDA; `_registration` for the
  elastic/shift switch, `_polarity`/`_features` for the polarity column, `_mine`/`_fit_lr` for hard-negative
  mining, `aligned_curves` for diagnostics); `DetectionPipeline` (`.fit`/`.predict`/`.save`/`.load`).
  `__main__` self-check covers both grouping rules, mining, polarity, shift registration and the
  scale-invariance of normalisation.
- **`src/detect_metrics.py`** — evaluation only. `labelled_events` is the single L1+L2 pass everything is
  built on; `froc_points`, `localisation_errors`, `detection_counts` and `bootstrap_auc_ci` then work off
  **any** score vector, so the same code serves the test set and out-of-fold CV scores. Plus
  `layer1_recall`, `consolidation_recall`, `classifier_auc`, `grouped_cv`, `froc`, `froc_summary`,
  `threshold_at`, `centre_baseline`, `centring_offsets`, `report`. `FP_BUDGET = 10` sets the operating
  point hits/misses/FPs are quoted at.
- **`src/detect_plots.py`** — one function per figure panel, each drawing into an `ax`: `froc`, `roc`, `pr`,
  `counts`, `score_hist`, `fold_aucs`, `timeline`, `loc_error`, `event_panel` (single-event waveform for
  eyeballing failures). Shared by the test and CV cells. Computes no metric of its own.
- **Notebook sections after the CV**, all reusing its out-of-fold scores (no refitting): three
  flag-guarded sweeps (`n_components_sweep`, `n_basis_sweep`, `hard_neg_sweep`), **error analysis Phase 1**
  (hits/misses by certainty, miss cost in extra FPs, FP concentration, epileptic vs non-epileptic FP rate),
  **error analysis Phase 2** (grids of the top FPs and every missed IED), the **pilot pre-test**
  (recording-level statistics + the marked-IED-removed control), and the **amplitude confound check**.
  All need `RUN_GROUPED_CV = True` — they reuse its out-of-fold scores. Phase 3 (spatial-feature
  separation, the L4 go/no-go) is specified in report.md §7 but not built.
- **`predict.py`** + **`requirements.txt`** (pinned) — supervisor entrypoint:
  `python predict.py <model.joblib> <recording.edf> [--score]`. `--score` defaults to **0.80**, the
  pre-registered ≤10 FP/min CV operating point (was an uncalibrated 0.5 until 2026-08-18).
  **`requirements.txt` also pins `multimethod==2.0.2`** — without it a clean install dies at `import skfda`
  with a metaclass conflict, because `scikit-fda==0.9.1` does not pin its own dependencies. **Verified by
  building a fresh venv**, which is the only way that class of bug shows up. Python 3.11.
- **`IED Detection/detection_run.ipynb`** — split → fit(90) → window-centring check (train only) → test
  report + figures → save model → **grouped CV** → three sweeps. ~5 min for the main path.
  - The grouped CV and all four sweeps are **flag-guarded** (`RUN_GROUPED_CV`, `n_components_sweep`,
    `n_basis_sweep`, `hard_neg_sweep`, `polarity_sweep`) so Run All stays fast. Keep that pattern for
    anything slow.
  - The **polarity test** is the last cell (`polarity_sweep`). Both arms share one registration per fold,
    so it is a single ~11 min CV run *and* a paired comparison — the pattern to copy for any L3 feature.
- Experimental notebooks (reference only, don't extend): `detection_stage1.ipynb` (L1 feasibility),
  `detection_threshold_diagnostic.ipynb` (per-channel selectivity / reference study).

---

## 4. Where the numbers stand

**FINAL PIPELINE (2026-08-18):** 250 Hz · 0.5–45 Hz zero-phase bandpass · two-threshold bad channels ·
average reference over good channels · `components`/0.7 · `normalise_amplitude` · polarity ON · elastic ·
`n_basis=70`, `k=24`. Grouped 5-fold CV over 4,157 windows / 60 positives / 49 IEDs.

| | grouped CV (49 IEDs) | held-out test (5 IEDs) |
|---|---|---|
| ROC-AUC | **0.865** [0.797, 0.929] | **0.874** |
| PR-AUC (chance) | **0.170** (0.0144) | 0.143 (0.011) |
| sens @1 / 5 / 10 / 25 / 50 / 100 FP/min | 0.14 / 0.51 / 0.67 / 0.82 / 0.92 / 0.94 | 0.20 / 0.20 / 0.20 / 0.60 / 0.80 / 1.00 |
| hits / misses / FPs @ ≤10 FP/min | **33 / 16 / 181** | 1 / 4 / 1 |
| localisation | **12 ms** | 4 ms |
| L1 / L2 recall | 1.00 / 0.9796 | 1.00 / 1.00 |

**Test 0.874 vs CV 0.865 — the test corroborates the CV**, and that is the best evidence the CV isn't badly
selection-inflated. **The test FROC is NOT interpretable**: 1 of 5 at ≤10 FP/min is a ~4% draw under the
CV's own rate, on 5 IEDs, while the test AUC simultaneously improved. **Do not iterate on it** — that would
convert the last held-out estimate into a tuning signal. Also stop calling it "selection-free": it has been
read several times and the config was chosen against CV numbers. "Held out from fitting" is the true claim.

CV operating point across the session: 28/21/187 → (polarity) 30/19/176 → (250 Hz + bandpass) 28/21/182 →
(average reference) **33/16/181**. Misses are the lowest they have been.

**Front end is not the problem:** L1 recall 1.00, L2 recall 1.00, **1.00 events per IED**, 5.28 channels
per event. **L3 discrimination is the bottleneck** — at ≤10 FP/min the CV finds 29 of 49 IEDs and raises
~190 false positives (precision ≈ 0.13).

**Two baselines, both must be beaten:** centre baseline 0.29 @ 4.0 FP/min, and the **prominence baseline**
(rank candidates by peak-to-peak of the normalised window — no model, no fitting): ROC-AUC 0.801, PR-AUC
0.054, 0.12 @5 FP/min, 17/32/185 at ≤10. **The prominence baseline is the cleanest demonstration of why
ROC-AUC is not the headline here** — it sits 0.03 below L3 on ROC-AUC while having 2.5× less PR-AUC and a
quarter the sensitivity at ≤5 FP/min.

**Levers already tested and closed** — do not redo these: window centring (no effect), `n_basis` (no
effect, hypothesis refuted), hard-negative mining (gain not distinguishable from noise), amplitude
normalisation (neutral on CV, adopted for transfer), polarity (small honest gain, adopted),
**shift registration (clearly worse — elastic warping earns its place)**. `n_components` was the one large
win and it is spent.

**The misses are no longer unexplained — it is relative prominence** (report.md §7 Phase 2b). Hits are
19.5× the recording background, misses 11.9×, top FPs 27.9×; corr(score, prominence) is +0.59 among the 64
positives but only +0.14 overall. **L3 has no mechanism to promote a discharge that is small but
well-formed.** Four hypotheses tested: polarity split, scale, warping — all refuted; prominence supported.

**Convergent finding worth stating in the writeup:** `n_basis` is null, prominence explains the misses, and
the registration gain comes from aligning the ~97.5% of the window that is *not* the spike. Together,
**L3 is largely judging window context and prominence, not fine spike morphology** — which is why every
morphology lever tried so far has produced a small or null effect.

**PREPROCESSING — ALL PHASES COMPLETE (0–4, 2026-08-17/18). Nothing here is outstanding.** Kept as the
record of what was done and why. Full detail in report.md §8; current numbers in the table above.
Rationale to reuse in the writeup: real EEG is 15–20 min with patients being poked, sneezing, falling
asleep, while Kural's 11–14 s clips are far cleaner — so Kural **cannot show preprocessing helps**, only
that it does no harm. Argue it as a transfer property, exactly like amplitude normalisation.

**Working style the user asked for and it worked well: do ONE phase at a time, report back, and wait for
the go-ahead before starting the next.** Keep doing that.

**What is actually next: nothing is queued.** Pick from the deferred list at the end of this section
(**A**–**I**), or the two open questions — whether to refit the shipped model on all 100 once decisions are
frozen (report.md §7), and the supervisor's expected IEDs-per-recording for the pilot's `mean top-N`.

- **Phase 0 — DONE 2026-08-17.** `Config.sfreq` is now the only sample rate; `sg_win`→`sg_ms=42.0` with
  `Config.sg_samples()` (forces ODD — savgol requires it and 42 ms @ 250 Hz rounds to 10); duplicate
  `SG_WIN` deleted; marker + `dur` from `cfg.sfreq`; `reference` moved into Config and
  `load_recording_path`/`load_recording`/`load_dataset` take `cfg`; `predict.py` passes `pipe.cfg`; the
  500 Hz assert is a guarded resample. **Verified bit-identical** at 500 Hz (window matrix, Y, G, markers,
  durations, every event field; L1/L2 recall 1.00/1.00; 227.5 events/min), so L3's input is unchanged and no
  CV re-run was needed.
- **Phase 1 — DONE 2026-08-17.** `Config.sfreq=250.0`; `sg_samples()` 21 → 11; everything else is in ms so
  unchanged in time. **L1 recall 1.0000 (gate passed).** L2 recall **0.9796** — one IED (S26), diagnosed as
  NOT a resampling regression: the discharge is captured (16 of 23 members within ±100 ms) but the event's
  reported time is +200 ms at **both** rates, and the 500 Hz hit was carried by a separate well-timed
  fragment. Real cause is representative selection landing on the after-going slow wave → deferred item I.
  Pool 4,330/64 → 4,133/62; events/min 227.5 → 217.2; channels/event 5.18 unchanged.
  **CV deliberately skipped** — no decision hung on it; deterministic, so runnable retrospectively.
  Anti-aliasing verified by PSD (all retained bands preserved; only 0.0105% of power was above 125 Hz).
- **Phase 2 — DONE 2026-08-17, ADOPTED.** `Config.bandpass=(0.5, 45.0)`, filtered BEFORE resampling,
  zero-phase. **No notch** — 45 Hz already excludes 50 and 60 Hz (it must come back if the upper edge ever
  goes above 50). 0.5 Hz (not 1) was the supervisor's call, to keep the after-going slow wave.
  **L1 recall stayed 1.0000** — the predicted risk (L1 is a second derivative, so a 45 Hz low-pass removes
  what it amplifies) was REFUTED; MAD normalisation absorbs it. L2 recall unchanged at 0.9796.
  Passband preserved to 99.7–100.1%, >60 Hz gone; asserted in `detect_data._selfcheck`.
  **CV: ROC-AUC 0.843, PR-AUC 0.146 → 0.169, @1 0.10 → 0.14, @5 0.47 → 0.49, @25 0.76 → 0.82**
  (dips at @10 and @50); 28/21/182 @ ≤10 FP/min. **ΔPR-AUC +0.0268, 95% CI [−0.0362, +0.0962], P(>0)=0.80 —
  suggestive, NOT established.** Report as a transfer measure validated as no-harm, never as a gain.
  Of the 21 misses, **20 are L3 ranking failures and 1 is S26** (deferred item I), so the reachable ceiling
  is 48/49 and the honest comparison on hits is 28/48 vs 30/49.
- **Phase 3 — DONE 2026-08-17, ADOPTED.** `reference='average'` + two-threshold bad-channel handling:
  **soft (MAD > 3x median)** = dropped from the average only, kept in the analysis; **hard (MAD > 10x, or
  ptp < 0.5 uV)** = removed from the analysis and interpolated. Thresholds are matched to the cost of being
  wrong — excluding from the average is free if wrong, interpolating destroys real data. Over the cap of 2:
  interpolate none, demote all to soft, flag the recording (degrade, don't refuse).
  - **Interpolated channels generate no L1 candidates** — the single enforcement point, so a reconstructed
    channel can never satisfy `min_channels`, inflate a channel count, or enter an L4 spatial feature.
  - **Dead is tested on peak-to-peak, NEVER on MAD.** A MAD floor was proposed and REFUTED by the
    diagnostic: Kural's reference sits near Cz/Pz (median MAD by electrode rises monotonically from Pz 0.36
    to FP1 1.29), so midline channels legitimately have tiny MAD — S19/Cz has MAD 0.012 uV but ptp 2.3 uV.
    A MAD floor would have interpolated away good reference-adjacent channels.
  - 10x is calibrated from clean data: p99 = 1.88, max = 8.28, **nothing trips 10x**.
  - **Inert on Kural** — 1 soft flag (S49/Pz), 0 hard. Ships untested on this data; say so.
  - **L1 recall 1.0000 (gate passed)**, L2 unchanged 0.9796. Channels/event 5.15 -> 4.83 (the documented
    reference-contamination effect, measured).
  - **POLARITY SURVIVES: 51% / 75% downward, identical to the recorded reference — the column stays.** The
    montage-dependence risk recorded in §5 did not materialise, and is now measured, not assumed.
  - **CV: ROC-AUC 0.843 -> 0.865, @10 0.57 -> 0.67, @50 0.84 -> 0.92, hits/misses/FPs 28/21/182 ->
    33/16/181, localisation 16 -> 12 ms.** +5 IEDs at the same FP cost. **But ΔPR-AUC +0.0013
    [-0.083, +0.089] P(>0)=0.50 and +5 hits is ~1.5 SE — encouraging, NOT established.** PR-AUC must be read
    relative to chance here (pool changed): 10.6x -> 11.8x.
- **Phase 4 — DONE 2026-08-18.** Restart & Run All; model re-saved WITH the full config (pickle hazard
  resolved, model safe to share); notebook reproduced the scratch CV exactly. Notebook restructured so it
  stops going stale: **results prose removed, method/rationale kept** — the notebook says what is measured
  and why, report.md says what came out. **Confound-cell verdict bug found and fixed** (see §4 below).
- **The reference reversal is deliberate**: §5's "do NOT re-reference L1/L2/L3" is overruled, because
  average reference is reproducible at any lab and a recorded reference isn't. Record it as a reversal.

**Coming back to later** (report.md §8): **A** window length (`classifier_halfwin_s`; the spike is ~2.5% of
a 2 s window so registration and FPCA are dominated by background) · **B** shape-versus-size, the main
sensitivity lever · ~~**C** L4 spatial features~~ **DONE 2026-08-20, report.md §7** · ~~**D** the
Neuronostics pilot~~ **DONE — `pilot.py`, validation run returned** · **E** random-crop eval · **F** report
the prominence baseline as a second null model · ~~**G** re-measure window centring~~ **DONE 2026-08-20,
amplitude confirmed, mechanism refuted** · **I** decouple an event's TIME
(sharpest member) from its representative CHANNEL (max peak-to-peak) — the S26 lesson; needs its own CV.

**L4 (2026-08-20): gate passed, biggest gain since `n_components`, but read the caveats.** Four arms over
out-of-fold CV scores, combining LR fitted inside the same folds. PR-AUC 0.170 → **0.358** (arm D), @1
FP/min 0.14 → 0.41, hits 33 → 39 (arm B), FPs 181 → 154 (arm D). ΔPR-AUC vs arm A: B +0.179
[+0.079, +0.291], D +0.207 [+0.102, +0.324], both P(>0)=1.00. **But:** the gain is `n_channels`, not
geometry (arm C alone is not established, +0.027 [−0.017, +0.071]); `gradient` failed its gate (0.480)
and its dipolar-falloff hypothesis is refuted; and **the entire gain is in wide-field IEDs** — narrow
(≤9 ch) stays 6/13 in every arm while wide goes 27/35 → 33/35. The pre-registered ≤2-channel subgroup is
EMPTY (Kural's narrowest IED spans 4 channels), so **Kural cannot test the focal case**.
**Nine arms tested in total (A–D, P1, P2, E, F, E+F) and B is still the best** — all in the run
notebook's last three sections; `src/detection.py` has `spatial_features`/`member_ptp`,
`detect_data.electrode_positions` has the montage coordinates. Nothing adopted yet.
- **"It is just size re-entering" was pre-specified and REFUTED.** Prominence scores 0.805 against the
  full negative pool (matching the prominence baseline's 0.801) but **0.455 — below chance — against the
  top FPs**, where `n_channels` holds 0.703. Adding prominence to channel count *costs* PR-AUC (P2 vs B
  −0.075) and its weight collapses to +0.06. The two correlate only +0.327. **The gain is genuinely
  spatial.**
- **Binning `n_channels` and a size-free contiguity measure both LOSE to plain B** (E vs B −0.123
  [−0.219, −0.024]; F vs B −0.155 [−0.279, −0.046]). E's coefficients confirm the relationship is
  non-monotone (`k=2 +0.22, k≥8 +0.50` vs the 3–7 reference) but collapsing 8–19 throws away where 40 of
  60 positives live. F failed because **top FPs are contiguous too** (60.8 vs 64.6 mm) — L2's ≥0.7
  correlation grouping already enforces contiguity, so the feature is redundant two layers downstream.
- **Lead the L4 writeup with this: narrow-field IEDs are 6/13 in all nine arms.** Not size, not channel
  count, not binning, not contiguity, not prominence. A well-evidenced negative, and the honest
  counterweight to a doubled PR-AUC.

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
only the secondary spatial question. ~~**Do NOT re-reference L1/L2/L3**; re-referencing (average, and try
bipolar for focal) enters at L4 only.~~
- **REVERSED 2026-08-17 by the supervisor (§4 Phase 3).** Average reference now happens in **preprocessing,
  before L1**. The nuance above stands for *localisation* but loses to the transfer argument: a recorded
  reference differs between labs, an average reference doesn't. Same reasoning as amplitude normalisation.
  Keep the contamination nuance in the writeup — it explains what average referencing costs and why the
  spatial question, not the timing one, is the part that was ever at risk.

**Metric semantics:** recall is measured over **IEDs only** — missing a *mimic* is harmless (a mimic never
proposed can't become a false positive). Mimics are counted where they matter: as false positives in the
FROC.

**AMPLITUDE CONFOUND — found, explained, fixed. `normalise_amplitude=True` is now the default.**
L1 divides its statistic by each channel's MAD and is therefore **scale-blind**; L3 worked on raw µV and
was not. So L3's scores tracked how *loud* a recording was, independently of anything being IED-like.
- Evidence: epileptic recordings average 9 µV background vs 7 µV for non-epileptic; background amplitude
  **alone** separates the two groups at AUC 0.690; it correlates **+0.708** with the per-recording L3
  statistic; and in the amplitude-vs-statistic scatter the groups do **not** separate vertically.
- **Three consequences were predicted in advance; two confirmed, one REFUTED.** Confirmed after
  normalisation: FP concentration fell (worst 9 recordings 53% → 39% of FPs, worst single 22 → 11, FPs
  now spread over 58 of 90 recordings instead of 45), and the epileptic/non-epileptic FP asymmetry
  collapsed **5.9× → 1.6×** (11.9 vs 7.3 FP/min). Note the *total* FP count cannot fall — the operating
  point fixes FP/min — so what changed is where they come from: they redistributed from loud recordings
  to quiet ones, which is what "one global threshold now means the same thing everywhere" looks like.
- **Refuted: the expensive misses are NOT a scale artefact.** The predicted drop in miss cost did not
  happen — median extra FPs to catch a missed IED went 776 → 771, unchanged, with 14 of 21 still costing
  >500. So the missed IEDs are outranked by hundreds of events *even when scale is fair*. That is a
  genuine L3 **morphology** failure, not a calibration one, and it means better thresholds or
  recalibration cannot reach them — only better features can. Makes error-analysis Phase 2 (plot the ~21
  missed IEDs and see what they look like) the highest-value next step.
- A residual **1.6×** FP asymmetry survives normalisation and can no longer be scale. Either epileptic
  background is genuinely more IED-like, or there are unmarked discharges — the §7 Q4 question, now
  asked cleanly. Phase 2 answers it.
- Fix: divide each window by `DetectionPipeline.background(X)` = median over channels of per-channel MAD.
  L3 then judges **relative prominence**, and the model **self-calibrates on unseen recordings**.
- **Measured: detection effect is within noise** (CV ROC-AUC 0.828→0.831, PR-AUC 0.117→0.133, sens@5
  0.41→0.47 = 3 IEDs of 49 against SE ~0.07). **Adopted for generalisation, not for the numbers** — say so.
- **It cut the recording-level pilot statistic from 0.805 to 0.687** (marker-removed control 0.747→0.603).
  That drop is the *point*: the old number was mostly confound. **0.687 is the honest pilot expectation.**

**THE AMPLITUDE CONFOUND IS BROKEN — and the notebook's automated verdict said the opposite.** The check
printed "CONFOUND IS LIVE" because its condition was `auc_bg > 0.65 or abs(rho) > 0.5` and only the first
was true. **ρ — the correlation between recording amplitude and the pilot statistic, which is what the
confound actually IS — collapsed +0.708 → +0.060.** Normalisation worked. Condition now keys on ρ. Lesson:
an automated verdict keyed on the wrong quantity is worse than none, because it gets believed.
- **New finding it surfaced: background amplitude ALONE separates epileptic from non-epileptic recordings
  at AUC 0.759 — beating the pre-registered pilot statistic (0.666) — and is now uncorrelated with it.**
  Report it as a pilot baseline. Treat cautiously: epileptic recordings being louder (6.26 vs 4.46 µV) may
  be a Kural cohort/acquisition artefact that won't transfer. Test it in the pilot by exporting per-recording
  background amplitude.
- Pilot pre-test now: `mean top-5` **0.666** [0.543, 0.779], marker-removed control 0.582. Give 0.666.

**Selection optimism is now live.** `n_components` and `n_basis` were both chosen on the 90-train CV over
~19 configurations. At 49 IEDs the binomial SE on a sensitivity near 0.4 is ~0.07, so the quoted maxima are
optimistic by perhaps 0.05–0.10. Declare this in the writeup; do not present a tuned maximum as clean.

---

## 6. Config — every tunable (in `src/detect_config.py`, class `Config`)

```
sfreq=250; bandpass=(0.5, 45.0); sg_poly=3; l1_threshold=6.0; nms_ms=150
corr_halfwin_ms=150; coincidence_ms=50; corr_max_lag_ms=30; corr_threshold=0.7; min_channels=2
grouping='components'
sg_ms=42.0 (Config.sg_samples() -> odd, 11 @ 250 Hz); reference='average'
bad_soft_factor=3.0; bad_hard_factor=10.0; bad_flat_uv=0.5; max_interpolate=2
classifier_halfwin_s=1.0; normalise_amplitude=True; centre_on='amplitude'
polarity_feature=True; polarity_ms=20.0
registration='elastic'; n_basis=70; penalty=0.1; n_components=24; lr_C=1.0; hard_neg_ratio=0.0
n_reg_points=100
hit_tol_ms=100; n_test=10; split_seed=0
```
**`detect_model.joblib` is current and shippable — re-verified 2026-08-18.** All 32 Config fields present
(`sfreq=250`, `bandpass=(0.5,45)`, `reference='average'`, `grouping='components'`, `registration='elastic'`,
`normalise_amplitude=True`, `polarity_feature=True`), 25 features in both scaler and LR, FPCA k=24. The
Phase 4 Restart & Run All is done; nothing is owed here.
**`Config.sg_samples()`, not `samp(sg_ms)`, is what the SG filter must use** — `savgol_filter` requires an
odd window and 42 ms at 250 Hz rounds to 10.
**`polarity_ms=20` is settled — do not sweep it again.** Pre-specified from the Phase-2 measurement, then
confirmed mid-plateau by a robustness sweep (4–20 ms identical; 80 ms collapses the LR coefficient
−0.68 → −0.016 as the opposite-polarity slow wave contaminates the sign). Sweeping it *for* a value would
add selection optimism to an effect that is already marginal.
**`registration='elastic'` is settled** — `'shift'` was measured and is clearly worse (PR-AUC 0.133 → 0.082,
@5 0.47 → 0.31, paired ΔPR-AUC −0.067 [−0.131, −0.013] at k=24, and elastic wins at every k from 12 up).
The `'shift'` branch is kept as a re-runnable ablation; don't delete it, and don't re-litigate the choice.
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
- **A new `Config` field silently takes the class default inside an existing `.joblib` — and this HAS now
  bitten, silently.** The pickle stores the dataclass instance's `__dict__`, so a field added *after* a model
  was saved is absent from the instance and falls back to the current class default on load. Verified on
  2026-08-17: `detect_model.joblib` stores `sfreq=500.0` but has no `bandpass` or `reference`, so it
  inherited `(0.5, 45.0)` and `'average'` from the new defaults — meaning `predict.py` ran a model trained on
  500 Hz / unfiltered / recorded-reference data over 500 Hz / bandpassed / average-referenced input. **It did
  not crash and the output looked entirely plausible.** (An earlier version of this note said the mismatch
  fails "loudly". It does not. That is worse.)
  - **Rule: give a new Config field the default that reproduces existing saved models, or re-save the model
    in the same change.** ~~Until the Phase 4 re-run, `detect_model.joblib` must not be used or shared.~~
  - **RESOLVED 2026-08-18 and now GUARDED, not just documented.** `DetectionPipeline.load` compares the
    stored cfg's `__dict__` with the current `Config` fields and raises, naming what is missing. The
    shipped `detect_model.joblib` was verified to hold all 32 fields (250 Hz, average, 25 features), so it
    is safe to share. Keep the rule anyway — the guard stops a stale model, it does not fix one.
  - The polarity part of the earlier note was also wrong and is corrected: the saved model holds 25 features
    and `polarity_feature=True`, so it IS polarity-fitted.
- **Amplitude statistics cannot measure warping — they are invariant to it by construction.** Time warping
  is a reparametrisation (`registered(t) = original(gamma(t))`, gamma monotonic), so the set of values is
  preserved and peak-to-peak comes back as exactly 1.00. Use timing quantities (gamma, gamma', peak
  position) or whole-curve distances. Cost an hour on 2026-08-17.
- **Selecting a group by score and then comparing it on something score-correlated is circular.** The
  IED/mimic warping asymmetry was first quoted as ~47% from the top-30 FPs; against the full negative pool
  it is **11%**. Same trap inflates any "top FPs look like X" claim — always re-check against all negatives.
- Figures convention: **`dpi=800`**, `bbox_inches='tight'`, into `figures/`.
- Channel names are `E FP1-Ref` etc.; EEG channels start with `"E "`, EKG is `"P EKG"`. For mne
  montage/topomaps map `'E FP1-Ref'→'Fp1'` then `standard_1020`.
- Runtime: L1+L2 over all 90 recordings is ~8 s; **the L3 Fisher-Rao registration dominates everything**
  (~3–4 min per fit, so a grouped CV is ~13 min). Sweeps that only change what sits *downstream* of
  registration (`n_components`, `lr_C`, `hard_neg_ratio`) share one registration per fold and are then
  nearly free — use that pattern. `n_basis` is upstream and needs a full refit per value.
  - **That is FIT. Inference is ~25× cheaper and this was measured, not assumed (2026-08-18).** `predict`
    only calls `transform` against an already-fitted template: **21.1 min of EEG scores in 42.6 s end to
    end** (L1 0.2 s, L2 6.3 s, L3 36.1 s, peak RSS 0.61 GB). So a 20-min clinical recording is ~45 s and
    the whole Neuronostics pilot is ~1.2 h, not overnight. Don't quote the fit cost for a prediction run.
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
