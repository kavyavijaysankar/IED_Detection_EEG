# Project report — IED detection (running log)

What was built, what was decided, and what was deliberately *not* done. Updated on every major addition.
Terminology matches the code. Last updated: 2026-08-15.

---

## 0. Task and data

- **Goal:** given an EEG recording, find *where in time* an interictal epileptiform discharge (IED) occurs.
  On non-epileptic recordings, do not fire on ordinary sharp transients ("mimics").
- **Data:** Kural 2020. 100 recordings, ~11–14 s each, 500 Hz, `.EDF`. 54 epileptic (22 `clear_positive`,
  32 `unclear_positive`), 46 `negative`. ~21 minutes total.
- **Labels:** exactly **one** annotated IED per epileptic recording, marker at the spike peak
  (`transient_onset_s` in `Kural_Dataset/eeg_summary.csv`). Neurologist-verified as single-IED.
  - Consequence: all non-marker EEG is trustworthy IED-free ground truth → false positives are countable.
  - Ground truth is **time-only**. "Which channel has the IED" is always derived, never known.
- **Known constraints (state these in the writeup):**
  - The marker sits mid-recording in ~99% of clips → a "predict the centre" baseline scores well without
    any modelling. Every result must be compared against it.
  - ~54 independent positive events is the statistical ceiling. Augmentation does not raise it.
  - Cohort mixes focal and generalized IEDs and the **type is not labelled**, so everything must be
    type-agnostic. Channel count is a weak, bimodal feature — not "more channels = more IED-like".

---

## 1. Files

### Detection code (`src/`)
- **`detect_config.py`** — `Config` dataclass. Every tunable in one place; imported everywhere. Nothing
  else in the codebase hard-codes a threshold or window.
- **`detect_data.py`** — data layer. `CH19` (19 common 10-20 channels), `load_recording` /
  `load_recording_path` / `load_dataset` (full recordings, µV, reference `'recorded'|'average'`),
  `stratified_split` (recording-level, stratified on `certainty`), `window_around` (edge-safe fixed window).
- **`detect_stage1.py`** — L1 primitives: `sharpness` (Savitzky-Golay 2nd derivative, MAD-normalised),
  `channel_stat`, `candidates` (peak-pick + non-maximum suppression), `n_channels_crossing`.
- **`detection.py`** — the cascade. `Stage1` (L1), `Consolidator` (L2), `event_centre` / `event_window`
  (window cutting), `Classifier` (L3 FDA model, plus `aligned_curves` for diagnostics),
  `DetectionPipeline` (`fit` / `predict` / `save` / `load`).
- **`detect_metrics.py`** — evaluation only, kept out of the model. `labelled_events` is the single
  L1+L2 pass everything is built on (events + windows + labels + recording index); `froc_points`,
  `localisation_errors` and `bootstrap_auc_ci` then work off *any* score vector, so the same code serves
  the test set and the out-of-fold CV scores. Plus `layer1_recall`, `consolidation_recall`,
  `classifier_auc`, `grouped_cv`, `froc`, `froc_summary`, `threshold_at`, `detection_counts`,
  `centre_baseline`, `centring_offsets`, `report`. Module constant `FP_BUDGET = 10` fixes the operating
  point hits/misses/FPs are reported at.
- **`detect_plots.py`** — evaluation figures, one function per panel, each drawing into an `ax` passed in:
  `froc`, `roc`, `pr`, `counts`, `score_hist`, `fold_aucs`, `timeline`, `loc_error`. Shared by the
  test-set and grouped-CV notebook cells so both are plotted identically. Computes no metric of its own.

### Classification-phase code (`src/`, earlier phase, still used)
- **`classify_loader.py`** — `load_CSV` (reads + filters the manifest; still used by the detection
  notebook) and the tight marker-centred windowing used by the classification phase.
- **`EEG_loader.py`** — original bulk EDF + demographics loader from the EDA phase.

### Notebooks
- **`IED Detection/detection_run.ipynb`** — the main runner: split → fit on 90 → window-centring
  diagnostic (train only) → test report + diagnostic figures → save model. ~5 min. The final section is
  the **grouped CV, gated behind `RUN_GROUPED_CV = False`** — it is deliberately independent (no other
  cell reads it) so the default run stays fast, and it is where tuning decisions are read from.
- **`IED Detection/detection_stage1.ipynb`** — L1 feasibility study (threshold sweep, recall vs candidate
  rate). Reference only, do not extend.
- **`IED Detection/detection_threshold_diagnostic.ipynb`** — per-channel selectivity and reference
  (recorded vs average) study, including topomaps. Reference only.
- **`IED classification/`** — the completed classification phase: `EDA`, `classification_registered`,
  `fourier_comparison`, `nested_cv`, `nested_cv_stabilised`, `residual_experiment`.

### Root
- **`predict.py`** — supervisor/external entrypoint: `python predict.py <model.joblib> <rec.edf> [--score]`.
- **`requirements.txt`** — pinned environment.
- **`detect_model.joblib`** — saved pipeline (Config + fitted L3), written by the run notebook.
- **`CLAUDE.md`** — handover/instructions for AI assistance.
- **`report.md`** — this file.
- **`figures/`** — all figures, saved at `dpi=800`, `bbox_inches='tight'`.

---

## 2. Architecture — the cascade

Four layers; **v1 = L1→L2→L3** (built), **v2 = + L4** (not started).

| Layer | Does | Reference |
|---|---|---|
| L1 `Stage1` | per-channel sharpness → candidate `(channel, time)`; tuned for recall | recorded |
| L2 `Consolidator` | group candidates of the same spike into multi-channel events | recorded |
| L3 `Classifier` | FDA model on the representative channel's 2 s window → IED score | recorded |
| L4 (v2) | event decision combining L3 score + spatial features | re-referenced |

- **Decision: cascade, not a sliding classifier.** Every classification-phase training window was centred
  on a transient, so the classifier has never seen background; sliding it across a recording puts it
  out of distribution. The cascade keeps L3 in-distribution — it only ever sees candidate windows.
- **Decision: v1 stays on the recorded reference throughout.** Reference contamination spreads a spike in
  *space*, not *time*, so it corrupts the spatial question only. Re-referencing enters at L4.

---

## 3. Data handling and splitting

- **Split:** recording-level, stratified on `certainty`, `n_test=10`, `seed=0` → 90 train / 10 test.
  Deterministic and re-derivable.
  - Stratifying on `certainty` covers both epileptic-vs-not and clear-vs-unclear in one variable.
- **Channels:** fixed 19-channel 10-20 montage (`CH19`) present in all 100 recordings.
  - 3 epileptic recordings lack the inferior-temporal chain (F9/F10, T9/T10, P9/P10), so those channels
    are excluded to keep one consistent montage. Revisit only if temporal IEDs prove to be missed.
- **Leakage discipline:** everything fitted (L1 threshold, `corr_threshold`, L3 template/basis/scaler/
  weights) comes from the 90 only. The 10 test recordings are for one final evaluation.
  - **Closed:** `n_basis` was originally raised 30 → 70 off the *test* AUC (0.683 → 0.702), which was not
    a valid signal. It has since been re-derived on the 90-train grouped CV (§6) — the sweep shows
    `n_basis` makes no consistent difference, so the value stands but for a different reason.
  - **Live caveat:** `n_components` and `n_basis` are now selected on the 90-train CV, so the CV numbers
    quoted for the chosen cell are selection-optimistic. Declare this; the 10 test recordings are the only
    untouched estimate left.

---

## 4. L1 — candidate generation

- **Statistic:** |Savitzky-Golay 2nd derivative| (curvature = "sharpness"), normalised per channel by MAD.
  - SG window 21 samples (42 ms), cubic — near spike width, preserves peaks.
  - MAD normalisation makes the threshold channel-adaptive.
- **Threshold = 6 MAD, deliberately permissive.** 100% IED recall at this value.
  - **Not** raised for selectivity: at threshold 6 a median 17 of 19 channels cross at the marker, and
    raising it to 14–16 gives 2–4 channels but drops recall to 57–63%. No single threshold gives both
    recall and selectivity → **selectivity is L2's job**.
  - The many crossings are largely genuine IED spread (background is gone by threshold 8 while 12
    channels still cross), amplified by per-channel MAD normalisation sensitising quiet channels.
- **Peak picking:** `find_peaks` with non-maximum suppression at 150 ms, so a spike's up-stroke and
  down-stroke (two curvature peaks) collapse to one candidate.
- **Consequence accepted:** ~380 candidates/recording. High rate is intentional; L1 recall caps the
  whole system, so it is tuned for recall, not rate.

---

## 5. L2 — consolidation

- **Purpose:** group per-channel candidates of the same physical spike into one *event*, and reject
  things that cannot be IEDs.
- **Grouping rule:** greedy by sharpness — the sharpest unused candidate seeds an event; any candidate on
  another channel within `coincidence_ms` whose window correlates ≥ `corr_threshold` joins it.
- **Cross-correlation design (all three properties are required):**
  - **normalised** → amplitude-invariant, so a diminished copy on a distant channel still matches;
  - **lag-searched (±30 ms)** → tolerates propagation delay;
  - **absolute-valued** → polarity-invariant, because the far side of a dipole is polarity-reversed and
    raw zero-lag correlation would drop exactly those channels.
- **Correlate the SG-smoothed signal**, not the raw — morphology, not noise.
- **Correlation window ±150 ms** — must span the whole biphasic complex.
- **Representative channel = maximum peak-to-peak** over that window on the smoothed signal.
  - Chosen to match how the classification-phase training windows were selected. Consistency with the
    existing classifier mattered more than the specific statistic.
- **Single-channel reject is the only hard filter.** IEDs cannot be single-channel (confirmed).
  - Floor is **≥2 channels, never 3–4** — focal IEDs may be exactly 2 adjacent channels.
- **Filtering philosophy:** hard-filter only *unambiguous* non-IEDs. Anything ambiguous (amplitude,
  channel count, field shape) is a **soft feature for L4**, never a hard drop. Every hard filter is a
  permanent recall cap, so recall is re-measured after each one.
- **Known behaviour:** one IED produces **2.86 events** on average within ±100 ms of its marker
  (49 train IEDs → 140 positive windows). Per-IED counts: 8 IEDs give 1 event, 11 give 2, 14 give 3,
  13 give 4, 2 give 5, 1 gives 6.
  - Accepted for training: more positives is mild augmentation, but they are **not independent** —
    effective count stays 49.
  - **Not accepted for evaluation:** mimics fragment the same way, so the FROC's FP/min axis is inflated
    by roughly the same factor (~3×). This is a real, one-sided cost — near-marker fragments are counted
    as the single TP, never as FPs, so fixing fragmentation cannot reduce sensitivity.

### Fragmentation sweep (measured on the 90-train, L1+L2 only)
- **Earlier hypothesis — REFUTED.** It was assumed the fragments were the biphasic complex spread over
  time, escaping the 50 ms `coincidence_ms` gate. Both parts are wrong:
  - Sweeping `coincidence_ms` 50 → 250 ms changes nothing: events/min 337 → 333, events/IED 2.86 → 2.86.
  - The 140 near-marker events have median |offset| **19 ms** and p90 **82 ms** from the marker — they are
    **simultaneous**, not spread across the complex.
- **Actual cause — spatial, not temporal.** A median of **17 of 19 channels cross the L1 threshold at the
  marker**, but the near-marker events together cover only **12** channels, and the **largest single event
  holds only 6**. One spike's field is being split into ~3 correlation clusters, with ~5 channels lost
  entirely into rejected (<2-channel) events.
  - Root cause is the grouping *algorithm*, not the threshold: grouping is **greedy and seed-relative** —
    every member must correlate with the seed candidate, not with the cluster. A spike's field is a chain
    (A~B, B~C, but A≁C, especially across a dipole), so a star-shaped test fragments it. Candidates are
    also consumed permanently (`used`), so a candidate absorbed by a weak event cannot join a better one.
- **Sweep results (L2 recall = 1.00 in all 38 configurations tested — no setting risks recall):**

  | knob | range | events/min | events/IED | channels/event |
  |---|---|---|---|---|
  | `coincidence_ms` | 50 → 250 | 337 → 333 | 2.86 → 2.86 | 3.22 → 3.46 |
  | `corr_threshold` | 0.70 → 0.90 | 351 → 263 | 2.67 → 2.69 | 3.61 → 2.80 |
  | `nms_ms` (L1) | 150 → 350 | 336 → 232 | 2.88 → 2.61 | 3.38 → 3.23 |

  - `nms_ms` is the only effective knob (−31% events/min) but it does not fix fragmentation either.
  - **No parameter setting removes fragmentation** → this is an algorithm change, not a config change.
- **Options identified:**
  - **(A) Event-level non-maximum suppression** after L2: merge events whose times fall within ~150–200 ms,
    keeping the strongest. Cheap and cannot reduce recall. **Not implemented** — (B) alone fixed it.
  - **(B) Connected-component clustering** — **IMPLEMENTED, see below.**

### Grouping rule: `cfg.grouping` — `'greedy'` vs `'components'` (option B)
- **`'components'` (new):** candidates are graph nodes; an edge joins two candidates on *different*
  channels within `coincidence_ms` whose windows correlate ≥ `corr_threshold`; **events are connected
  components** (single-linkage). A~B and B~C put all three in one event even if A and C do not match.
  - Implemented with union-find over candidates swept in time order; an edge is skipped when the pair is
    already connected, so it builds a spanning forest rather than the full graph (~11 s for 90 recordings
    vs ~7 s for greedy — negligible).
  - Rejection (<`min_channels` distinct channels) and representative selection (max peak-to-peak) are
    unchanged, so only the grouping topology differs.
- **Measured on the 90-train (L1+L2 only):**

  | | greedy @0.8 (old) | components @0.8 | **components @0.7** | components @0.6 |
  |---|---|---|---|---|
  | L2 recall | 1.00 | 1.00 | **1.00** | 0.98 ✗ |
  | events/min | 337 | 267 | **262** | 254 |
  | FP/min (pre-scoring) | 329 | 263 | **259** | 251 |
  | events per IED | 2.86 | 1.67 | **1.31** | 1.29 |
  | channels/event | 3.22 | 4.37 | **5.16** | 5.71 |
  | channels in IED's largest event | 6 of 17 | 10 of 17 | **15 of 17** | 16 of 17 |
  | max member time span | 92 ms | 240 ms | 342 ms | 520 ms |

  - **36 of 49 IEDs now produce exactly one event** (was 8 of 49); the worst case is 3 (was 6).
  - **The IED field is now essentially intact** — 15 of the 17 channels that cross L1 at the marker end up
    in one event, versus 6 before. This is the result that matters for L4 spatial features.
  - **Chaining, the single-linkage failure mode, was checked and is not occurring** at ≥0.7: no event's
    members span more than 342 ms, and 0% exceed 500 ms.
- **`corr_threshold` under the new rule: 0.7 is the operating point.**
  - Under `'components'` the threshold behaves *opposite* to the old plan: **lowering** it de-fragments
    (0.8 → 0.7 takes events/IED 1.67 → 1.31 and field capture 10 → 15 channels).
  - **0.6 is rejected: L2 recall drops to 0.98.** One IED's event chains into a neighbour, and since the
    event's reported time is its max-peak-to-peak member, the representative moves >100 ms off the marker
    and the IED is no longer counted as hit. Max member span jumps to 520 ms — the chaining boundary.
  - The earlier plan to *raise* `corr_threshold` to 0.85–0.9 is therefore abandoned; it was the right
    instinct for a seed-relative rule and the wrong one for single-linkage.
- **Now the default** (`grouping='components'`, `corr_threshold=0.7`). Confirmed end-to-end: the fit went
  from 5,575 windows / 140 positives to **4,330 / 64**, and the test set shows exactly **1.00 events per
  IED** with 5.28 channels each.
- **Side benefit:** window centring improved without being touched — median |offset| 54 → 26 ms, and the
  fraction >100 ms off fell 22% → 17%. The representative channel is now chosen from the full member set
  rather than a fragment, so it lands on the true dominant channel more often.
- **A caveat that must go in the writeup: de-fragmentation removes an artificial sensitivity advantage.**
  The FROC counts a recording as found if *any* near-marker event clears the threshold. At 2.86 events per
  IED the old pipeline effectively got ~3 attempts per IED; at 1.00 it gets one. So fragmentation was
  inflating **both** the false-alarm rate (bad) and the sensitivity (flattering).
- **Matched grouped-CV comparison** (identical L3 hyperparameters, `n_basis=70`, `n_components=4`, so
  grouping is the only difference):

  | | greedy @0.8 | components @0.7 |
  |---|---|---|
  | windows / positives | 5575 / 140 | 4330 / 64 |
  | grouped-CV AUC | 0.543 [0.462, 0.628] | **0.657 [0.582, 0.730]** |
  | per fold | 0.43 · 0.63 · 0.53 · 0.47 · 0.63 | 0.59 · 0.71 · 0.70 · 0.51 · 0.71 |
  | average precision (chance) | 0.035 (0.025) | 0.028 (0.015) |
  | CV FROC sens @ 5 / 10 / 25 / 50 / 100 FP/min | 0.04 / 0.14 / 0.39 / 0.57 / 0.71 | 0.04 / 0.14 / 0.35 / 0.53 / 0.69 |

  - **AUC improves substantially (+0.114)** — the classifier discriminates much better when each IED is one
    well-supported event instead of three fragments. Relative to chance, average precision also improves
    (1.9× vs 1.4×).
  - **The FROC does not follow**: identical at 5 and 10 FP/min, and 2–3 IEDs *worse* out of 49 at higher
    budgets. This is the predicted trade-off — greedy's sensitivity was partly bought by proposing each
    IED three times, and losing that redundancy cancels the lower false-alarm rate.
  - **Decision: keep `'components'`.** The FROC differences are ~2 IEDs out of 49 (inside noise), while the
    AUC gain is large, the grouping is physically correct, and L4 needs the field capture (15 of 17
    channels vs 6). A sensitivity advantage that comes from emitting duplicate detections is not
    defensible in a viva even when it flatters the curve.
- **Downstream consequence to note for L4:** because the largest event captures 6 of 17 involved channels,
  the current `n_channels` figure **understates the IED field by ~3×**. Any spatial feature built on it
  today would be measuring the clustering artifact, not the discharge.
- **Open:** `corr_threshold` is still at the default 0.8. The plan to raise it to ~0.85–0.9 is now
  questionable: it *reduces* the event rate (351 → 263/min) but shrinks channels/event (3.61 → 2.80),
  i.e. it fragments the field further. Decide it after the grouping algorithm is fixed, not before.

---

## 6. L3 — the FDA classifier

Pipeline: cut a 2 s window on the representative channel → B-spline smooth → Fisher-Rao elastic
registration to a learned template → FPCA → StandardScaler → logistic regression (`class_weight='balanced'`).

### Window cutting and centring
- The window is **±1 s (2 s total)** on the representative channel.
- Before cutting, the centre is moved to a local extremum within **±150 ms** of the event time
  (`event_centre`). Two rules implemented:
  - `'amplitude'` — max |deviation from the local median| of the smoothed signal. **Current default.**
  - `'sharpness'` — max of the L1 statistic.
- **Tested and settled (90-train, 140 IED windows, 49 recordings):**

  | rule | median \|offset\| | p90 | max | >100 ms off | grouped-CV AUC |
  |---|---|---|---|---|---|
  | `'amplitude'` | 54 ms | 154 ms | 238 ms | 22% | **0.669** |
  | `'sharpness'` | 20 ms | 86 ms | 152 ms | 1% | 0.617 |

  - The misalignment is **real**: under `'amplitude'` the centre often jumps to the after-going slow wave
    (e.g. S65 +126 ms, S19 +160 ms) instead of the spike. `'sharpness'` essentially removes it.
  - It **costs nothing in AUC**: recording-level bootstrap of the difference = **+0.051 favouring
    `'amplitude'`, 95% CI [0.000, +0.105]**.
  - Interpretation: marker fidelity is not what the FDA features want. FPCA is amplitude-sensitive, so
    pinning the largest deflection at t=0 is the stronger regularity, and Fisher-Rao warping absorbs the
    residual shift.
  - **Decision: keep `'amplitude'`. Centring is ruled out as the AUC bottleneck.**
  - Offsets are measured against the neurologist marker — the only alignment measure that uses
    information from outside the rule being tested.
- **Rejected — ranking centring rules by how tightly the windows stack.** An "alignment spread" statistic
  was built and deleted: every rule maximises self-similarity at its own centre, so the comparison is
  circular and it favoured whichever rule centred on the biggest deflection.
- **Rejected (for now) — sliding sub-windows inside a larger window and taking the highest-scoring chunk.**
  Reasons: reported localisation uses the L1 event time and is already ~4 ms, so it fixes nothing in the
  output; it puts L3 out of distribution (it never trained on offset views); and taking a max over *k*
  chunks gives mimics *k* attempts too, which biases scores upward for everything.
  - The salvageable version, not yet tried: **jitter the window offset during training** for shift
    robustness, and **average** (not max) over jittered windows at test time.

### Hyperparameters
- Inherited from the classification phase's nested CV: `n_basis=30`, `penalty=0.1`, `n_components=5`,
  `lr_C=1.0`, `n_reg_points=100`.
- **Current values: `n_basis=70`, `n_components=4`.** Provisional — see the open issue in §3.
  - Rationale for raising `n_basis`: 30 cubic B-splines over 2 s is a knot every ~70 ms, an implicit
    ~7 Hz ceiling, while an IED spike is 20–70 ms wide. The basis could not resolve the spike.
- **L3 is deterministic** — two back-to-back fits of the identical config gave identical results
  (test AUC 0.657 both times), so run-to-run randomness is ruled out as a confounder.
- **`n_components` was the single biggest defect in the pipeline.** Grouped-CV sweep at `n_basis=70`
  (registration fitted once per fold and reused, so all k share one CV run — ~13 min total):

  | k (FPCA components) | CV AUC | 95% CI | sens @1 | @5 | @10 | @25 | @50 | @100 FP/min |
  |---|---|---|---|---|---|---|---|---|
  | 2 | 0.578 | [0.477, 0.653] | — | — | 0.12 | 0.31 | 0.53 | — |
  | 4 *(was default)* | 0.657 | [0.575, 0.725] | 0.00 | 0.04 | 0.14 | 0.35 | 0.53 | 0.69 |
  | 6 | 0.756 | [0.689, 0.811] | — | — | 0.24 | 0.53 | 0.63 | — |
  | 8 | 0.770 | [0.703, 0.828] | — | — | 0.22 | 0.61 | 0.80 | — |
  | 12 | 0.811 | [0.757, 0.860] | 0.04 | 0.22 | 0.37 | 0.73 | 0.88 | 0.94 |
  | 16 | 0.786 | [0.728, 0.844] | 0.08 | 0.24 | 0.51 | 0.69 | 0.82 | 0.88 |
  | **24 (chosen)** | **0.828** | [0.760, 0.880] | 0.08 | **0.41** | 0.59 | 0.71 | 0.84 | 0.92 |
  | 32 | 0.799 | [0.718, 0.861] | 0.06 | 0.45 | 0.59 | 0.71 | 0.86 | 0.88 |

  - **Why it happened:** FPCA orders components by *variance*, and the candidate pool is 98.5% negatives,
    so the IED-discriminative direction is not among the leading few. Truncating at 4 threw it away.
    This is the imbalance problem showing up in the representation — but the fix needed no class-informed
    fitting at all, only keeping more components.
  - **The detector now beats the centre baseline** (0.29 @ 4.0 FP/min): 0.41 at ≤5 FP/min versus 0.04
    before. That threshold had never been cleared. Note the baseline is itself inflated by the marker
    sitting mid-recording in ~99% of clips, so random cropping will widen the margin further.
  - **Choice of k = 24 is on the low-FP end of the CV FROC, not on best AUC.** 12–32 are statistically
    indistinguishable on AUC (CIs overlap almost entirely; the 12↑16↓24↑32↓ wobble is noise at 49 IEDs).
    k=12 is better mid-curve (0.73 @25) but nearly half as sensitive at ≤5 FP/min. A one-standard-error
    rule on AUC would pick k=12; state in the writeup that the argmax over 8 swept values is optimistic
    by perhaps 0.02–0.04.
  - **This invalidates comparisons made in the k≤5 regime.** greedy-vs-components (0.543 vs 0.657) and the
    centring result were both measured with the discriminative components truncated away. The grouping
    decision stands on other grounds (runtime on long recordings, field capture for L4) but is worth one
    confirmatory run at k=24.
- Earlier test-set readings at `n_basis=70` (3 → 0.597, 4 → 0.657, 5 → 0.648) are superseded; they were
  measured on 5 IEDs.

### `n_basis` × `n_components` grid (grouped CV, 90-train, ~50 min)
Registration is refitted once per `n_basis` per fold; all k inside a row share it.

| n_basis | k | CV AUC | sens @5 | @10 | @25 FP/min |
|---|---|---|---|---|---|
| 30 | 8 | 0.817 | 0.18 | 0.37 | 0.69 |
| 30 | 12 | **0.834** | 0.31 | 0.51 | 0.76 |
| 30 | 16 | 0.827 | 0.33 | 0.55 | 0.76 |
| 30 | 24 | 0.815 | 0.35 | 0.55 | 0.71 |
| 50 | 8 | 0.823 | 0.27 | 0.41 | 0.69 |
| 50 | 12 | 0.825 | 0.22 | 0.45 | 0.80 |
| 50 | 16 | 0.805 | 0.24 | 0.59 | 0.78 |
| 50 | 24 | 0.807 | 0.33 | **0.63** | **0.82** |
| 50 | 32 | 0.789 | 0.39 | 0.59 | 0.73 |
| 70 | 8 | 0.770 | 0.12 | 0.22 | 0.61 |
| 70 | 12 | 0.811 | 0.22 | 0.37 | 0.73 |
| 70 | 16 | 0.786 | 0.24 | 0.51 | 0.69 |
| 70 | 24 | 0.828 | 0.41 | 0.59 | 0.71 |
| 70 | 32 | 0.799 | **0.45** | 0.59 | 0.71 |
| 100 | 8 | 0.774 | 0.16 | 0.29 | 0.63 |
| 100 | 12 | 0.809 | 0.24 | 0.37 | 0.69 |
| 100 | 16 | 0.784 | 0.24 | 0.45 | 0.63 |
| 100 | 24 | 0.819 | 0.33 | 0.51 | 0.65 |
| 100 | 32 | 0.819 | 0.33 | 0.43 | 0.63 |

**Implications:**
- **`n_basis` has no consistent effect — the spike-bandwidth argument is REFUTED.** The earlier reasoning
  (30 cubic B-splines over 2 s ≈ a knot every 70 ms ≈ a ~7 Hz ceiling, while an IED spike is 20–70 ms wide,
  therefore 30 cannot resolve the spike) predicted that larger bases would win. They do not: at every k the
  four bases lie within ~0.05 AUC with no direction, and `n_basis=30` is as good as 100. A plausible
  physical argument that the measurement contradicts — record it as such in the writeup.
- **`n_basis` and `k` interact; neither matters alone.** At `n_basis=30`, k=8 already reaches 0.817; at
  `n_basis=70`, k=8 gives only 0.770 and needs k=24 to catch up. A larger basis spreads the same
  information over more components, so k must rise with it. The strong cells all sit near
  **k ≈ 0.3–0.4 × n_basis**.
- **k is the real lever, and sensitivity at ≤5 FP/min rises monotonically with k in all four rows**
  (30: 0.18→0.35, 50: 0.27→0.39, 70: 0.12→0.45, 100: 0.16→0.33). Four independent replications of one
  trend is a signal, not noise — and it had not turned over at the edge of the grid, so k is being extended
  to 40/48/64 at `n_basis=70`.
- **AUC and the FROC's left end disagree, and the disagreement is informative.** AUC peaks around k=12–24
  then falls, while @5 keeps climbing. AUC scores the *whole* ranking; the FROC at a low FP budget depends
  only on the very top of it. Extra components appear to sharpen the top of the ranking while adding noise
  to the middle. **Since only the top of the ranking is clinically usable, follow the FROC, not the AUC.**
- **Selection optimism — state this in the writeup.** 19 configurations have now been scored on the same
  49 IEDs. The binomial standard error on a sensitivity near 0.4 with 49 IEDs is ~0.07, so most of this
  grid is one standard error wide and both maxima (AUC 0.834, @5 0.45) are inflated by perhaps 0.05–0.10.
  The honest reporting is: *the sweep shows a broad plateau; performance depends on k and not on n_basis;
  k was chosen on the low-FP end of the CV FROC on this data* — then quote the chosen cell with its
  interval and declare the selection. Do not present a tuned maximum as a clean estimate.
- **`n_basis` stays at 70** — not because it wins (nothing does) but because it leaves headroom for larger
  k, which `n_basis=30` would cap at 28.
  - The earlier reading of *exactly* 0.702 for both 3 and 5 is the signature of an edit that never took
    effect — a notebook kernel that had already imported `Config` will keep the old value until it is
    restarted. **Restart the kernel after editing `detect_config.py`.**
  - No setting reproduces the 0.702 stored in the notebook, so that run's configuration is unknown.
- **Position taken: hyperparameters selected on the classification distribution do not transfer.** That
  phase used tightly-cropped, roughly balanced transient windows; detection L3 sees 5,575 windows at
  ~1:40 imbalance dominated by mimics and background.
- **Rejected — repeating a full nested CV on the detection candidates.** With 49 independent events the
  per-fold AUC standard error is ~0.09, so the inner loop would select among configurations separated by
  less than noise, at a cost of ~6 h per small grid. Preferred instead:
  - pre-specify `n_basis` from spike bandwidth (a signal-processing argument, not a data-driven one);
  - report a **sensitivity analysis** of grouped-CV AUC vs `n_basis` rather than a single tuned value;
  - optionally one nested run purely to quantify selection optimism.
- **Cost structure for any sweep:** registration dominates, so `n_basis` (upstream) needs a full refit
  (~4 min/fold), while `n_components` (truncate a fitted FPCA) and `lr_C` (refit the final LR only) are
  nearly free.

### Hard-negative mining (`cfg.hard_neg_ratio`, default 0 = off)
- **The problem it targets:** at the ≤10 FP/min operating point the CV finds 29 of 49 IEDs but raises
  ~190 false positives — **precision ≈ 0.13**, seven of eight flags wrong. Training is 64 positives against
  4,266 negatives, nearly all easy, so the LR spends its capacity separating IEDs from background that was
  never in contention.
- **Method:** fit the LR on everything (first pass) → score every training negative → keep all positives
  plus the `hard_neg_ratio × n_positives` highest-scoring negatives → refit the LR on that subset.
- **Where it sits, and why that is the whole trick:** only the LR is supervised. Registration and FPCA are
  unsupervised and stay fitted on **all** the data, so mining changes only which rows the LR trains on.
  - the representation is untouched and stays label-free — it does not attract the generalisation
    objection raised against class-conditional registration templates;
  - it costs **one logistic fit** (the expensive registration is already done), so a ratio sweep shares a
    single CV run, like the `n_components` sweep.
- First-pass scores are **in-sample** — standard practice, and not a leakage risk because the whole
  procedure happens inside whatever training set `fit` receives, so CV folds stay clean.
- Sweep lives at the end of `detection_run.ipynb` behind `hard_neg_sweep = False`, reporting ROC-AUC,
  PR-AUC, FROC at 5/10/25 FP/min and hits/misses/FPs per ratio.

**Result (grouped CV, 90-train, `n_basis=70`, `n_components=24`):**

| ratio | ROC-AUC | PR-AUC | sens @5 | @10 | @25 | hits / misses / FP @ ≤10 FP/min |
|---|---|---|---|---|---|---|
| **0 (all negatives)** | 0.828 | 0.117 | 0.41 | 0.59 | 0.71 | 29 / 20 / **176** |
| 2 | 0.270 | 0.010 | 0.02 | 0.04 | 0.08 | 2 / 47 / 156 |
| 5 | 0.431 | 0.014 | 0.02 | 0.04 | 0.18 | 2 / 47 / 185 |
| 10 | 0.740 | 0.069 | 0.24 | 0.35 | 0.59 | 17 / 32 / 160 |
| 20 | 0.823 | **0.145** | 0.43 | 0.59 | 0.76 | 29 / 20 / **145** |
| 50 | 0.829 | 0.126 | 0.39 | 0.59 | 0.73 | 29 / 20 / 174 |

- **DECISION: `hard_neg_ratio` stays 0 (mining off).** A negative result, kept for the writeup.
  - Best cell (ratio 20) gives the *same* 29 hits with 31 fewer false positives and PR-AUC 0.117 → 0.145.
  - But a paired recording-level bootstrap of ΔPR-AUC gives **+0.027, 95% CI [−0.008, +0.069], P(>0)=0.94**
    — suggestive, not established.
  - **And that interval is optimistic**, because ratio 20 was *selected* as the best of six before being
    tested against the baseline. Post-selection, the true effect is smaller than +0.027. Testing the
    winner of a sweep against the incumbent is a biased comparison; say so rather than quoting +0.027.
  - Same standard as the centring decision: the incumbent stays unless the challenger clears the bar.
    Adding a fitted step, a config knob and a selection procedure for an unproven gain is not worth it.
- **ROC-AUC is completely flat across ratios 0/20/50 (0.828/0.823/0.829) while PR-AUC moves 0.117→0.145.**
  Good illustration for the writeup of why ROC-AUC is the wrong headline here: mining acts on the top of
  the ranking, which is exactly what ROC-AUC is insensitive to and what the FROC and PR-AUC measure.
- **Low ratios don't merely underperform, they invert** (ratio 2 → ROC-AUC 0.270, far *below* chance).
  Mechanism: trained on positives against only the most IED-like negatives, the LR learns "resembles a hard
  negative ⇒ negative". Easy background resembles nothing of the sort, so it lands on the positive side of
  that boundary. Enough negatives must be kept to preserve the easy-to-hard gradient — with 64 positives
  the floor is ~20 per positive.
- Prior expectation was right for the wrong reason: mining pays off most for high-capacity models, and this
  LR (24 features, regularised, already `class_weight='balanced'`) had little capacity to be distracted.

### Class imbalance
- Training is ~140 positive vs ~5,435 negative windows.
- `class_weight='balanced'` in the logistic regression is already weighted cross-entropy.
- **Position taken: reweighting the classifier can't help much.** AUC and FROC are rank-based, so
  changing the decision boundary's location barely moves them.
- **The imbalance bites upstream, where nothing is weighted:** the registration template is a Karcher mean
  over all 5,575 curves (i.e. a *mimic* shape), and FPCA describes the variance of a negative-dominated
  pool. IEDs are warped toward a non-IED template and projected onto a basis not built to represent them.
- **Not yet done, ranked:**
  - class-informed registration template (fit on positives / balanced subsample) — cheap probe;
  - two templates + Fisher-Rao distance to each as features — highest-value remaining L3 experiment;
  - keep the **warping functions** as features (amplitude–phase separation) — most novel, highest risk;
  - weighted or balanced FPCA, or supervised dimension reduction (functional PLS) instead of FPCA;
  - cluster-aware weights: weight each positive by 1/(events from that IED) so all 49 IEDs count equally
    rather than letting a fragmented IED count 5 times;
  - **hard-negative mining** — the resampling analogue of cost-sensitive learning, and the planned next step.
- All of these are *fitted* objects, so they must be fitted inside each CV fold or labels leak into the
  representation.

---

## 7. Evaluation

### Metric definitions
- **L1 recall** — fraction of IEDs with any L1 candidate within ±100 ms of the marker.
- **L2 consolidation recall** — fraction of IEDs surviving as a ≥2-channel event within ±100 ms.
- **L3 AUC** — probability an IED window is scored above a non-IED candidate window.
- **Grouped-CV AUC** — out-of-fold AUC, 5 folds **grouped by recording**. Grouping is essential because
  one IED spawns ~3 near-duplicate events; ungrouped folds would put copies on both sides and inflate it.
  This is the trustworthy L3 number.
- **FROC** — sensitivity vs **false positives per minute**, obtained by sweeping the score threshold.
  Used instead of ROC because detection is free-response: a recording can produce any number of
  detections, so there is no count of true negatives and specificity is undefined.
  - "`<= 25 FP/min : sens 0.60`" = there is a threshold producing at most 25 false alarms per minute, at
    which 60% of IEDs are detected within ±100 ms.
- **Localisation error** — median |detected event time − marker| in ms, over IEDs that were hit. Capped at
  the ±100 ms hit tolerance by construction, and conditioned on a hit, so always read it with sensitivity.
- **PR-AUC / average precision** — reported alongside ROC-AUC everywhere, with the chance rate
  (= positive fraction ≈ 0.015) stated next to it. **ROC-AUC is inflated here** and is a supporting number
  only: it asks whether a random IED outranks a random non-IED, and ~98.5% of the non-IEDs are easy
  background the model gets credit for beating. PR-AUC and the FROC cannot be gamed that way.
- **Operating point: hits / misses / false positives** (`M.detection_counts`) at the threshold from
  `M.threshold_at(points, M.FP_BUDGET)` (`FP_BUDGET = 10` FP/min).
  - **hit** = an IED with ≥1 above-threshold event within ±hit_tol of its marker; **miss** = an IED with
    none; **fp** = every other above-threshold event.
  - **Correct rejections are deliberately never reported**, and neither is anything derived from them
    (specificity, accuracy, a full 2×2). Over continuous EEG there is no meaningful count of "non-events",
    and using rejected candidate windows instead flatters the model with easy background.
  - This is the FROC at one operating point — the point a clinician would actually run at.
- **Centre baseline** — a null model that ignores the EEG and always predicts the recording's midpoint.
  A true positive if the recording is epileptic and the midpoint is within ±100 ms of the marker; every
  other centre prediction is a false positive. Included because the marker is near-centred in ~99% of
  clips, so this scores well with no modelling at all.
- **Metric semantics:** recall is measured over **IEDs only**. Missing a mimic is harmless — a mimic that
  is never proposed cannot become a false positive. Mimics are counted where they matter: as false
  positives in the FROC.

### Cross-validated results — the trustworthy numbers (90-train, `components`/0.7, `n_basis=70`)
Grouped 5-fold CV over 4,330 candidate windows / **64 positives** / 49 IEDs. Effect of `n_components`:

| | k=4 (old default) | **k=24 (current)** |
|---|---|---|
| CV AUC (bootstrap CI) | 0.657 [0.582, 0.730] | **0.828 [0.763, 0.889]** |
| per fold | 0.589 · 0.710 · 0.697 · 0.505 · 0.705 | 0.750 · 0.749 · 0.872 · 0.939 · 0.831 |
| average precision (chance 0.0148) | 0.028 (~2× chance) | **0.117 (~8× chance)** |
| localisation, median over 49 IEDs | — | **16 ms** |
| sens @1 / 5 / 10 / 25 / 50 / 100 FP/min | 0.00 / 0.04 / 0.14 / 0.35 / 0.53 / 0.69 | **0.08 / 0.41 / 0.59 / 0.71 / 0.84 / 0.92** |

- **Centre baseline on the same 90: sensitivity 0.29 at 4.0 FP/min.**
- **The detector now clears the do-nothing bar** — 0.41 at ≤5 FP/min vs 0.29 for "always guess the middle
  of the recording". At k=4 it was 0.04, i.e. seven times *worse* than the trivial baseline. This is the
  first configuration of this pipeline that beats it. (The baseline is itself inflated: the marker sits
  mid-recording in ~99% of clips, so random cropping will widen the margin.)
- **The fold spread collapsed too**: worst fold 0.749, better than the *best* fold at k=4 (0.710). Part of
  what looked like small-sample instability was the model sitting near chance — one k=4 fold was 0.505.
- Localisation 16 ms is measured over all 49 CV IEDs; the 4 ms figure quoted earlier came from 5 test IEDs.
- Still short of clinical usability (published detectors: 0.7–0.9 sensitivity at 0.5–5 FP/min), but no
  longer in a different league. **L3 discrimination remains the bottleneck.**

### Held-out test set (10 recordings, 5 IEDs) — the only selection-free estimate left
Run at the final config (`components`/0.7, `n_basis=70`, `n_components=24`):
- L1 recall 1.00 · L2 recall 1.00 · **1.00 events per IED**, 5.28 channels each · localisation 4 ms.
- **Test AUC 0.847**, against the grouped-CV 0.828. Since `n_basis` and `n_components` were both selected
  on the 90-train CV, this is the one estimate the sweep could not inflate — **and it agrees**. That is
  the best available evidence that the CV number is not badly selection-optimistic.
- FROC: 0.00 @1 · 0.20 @5 · 0.60 @10 · 0.60 @25 · 0.80 @50 · 1.00 @100 FP/min.
  Centre baseline 0.40 @ 4.0 FP/min.
- **Do not read the test FROC's left end in either direction.** 0.20 vs the baseline's 0.40 is one IED
  versus two, out of five. The CV FROC (49 IEDs) is the only version of that comparison worth quoting.
- For reference, the same test set at `n_components=4` gave AUC 0.571 and 0.00 @5 FP/min.

### Evaluation issues still open
- **Random cropping not yet implemented.** The marker is centred in ~99% of clips, which inflates the
  centre baseline. Varying the marker position at eval (min 2.33 s before / 5.08 s after) is needed
  before the FROC is quoted in the writeup.
- **The test AUC is not a usable tuning signal — demonstrated, not just argued.** Across four fits that
  differed only in `n_components`, the test AUC ranged **0.597–0.657** while the FROC at ≤10 FP/min was
  **0.80 in every single one**. The two metrics disagree in direction, and with ~14 positive windows from
  5 IEDs a handful of rank swaps moves the AUC by 0.05. Every hyperparameter decision must come from
  `grouped_cv_auc` on the 90-train.
- **The FROC is computed on 5 IEDs**, so sensitivity moves in steps of 0.2. Proposed fix: compute a
  **cross-validated FROC** on the 90 (fit per fold, predict on held-out recordings, pool out-of-fold
  scores) for a curve over 49 IEDs instead of 5.
- **Considered and rejected — dissolving the test split to gain 10 training recordings.** The per-fold
  AUC spread (0.57–0.83) matches the Hanley-McNeil standard error for ~10 positives per fold almost
  exactly, so it is *measurement noise*, not evidence of a data-starved model; +5 IEDs would not change
  it. The holdout matters more, not less, now that hyperparameters are being tuned on the 90.
  - Legitimate variant, not yet done: once all decisions are frozen, refit the *shipped* model on all
    100 while still reporting the cross-validated 90 as the performance estimate.

---

## 8. Next steps (priority order)

1. **Pilot script for the Neuronostics data** (see §9). Send a 5-recording-per-group validation run first
   to shake out montage/sampling-rate/runtime issues, then the full run.
2. **v2 / L4: spatial features** on re-referenced data (average, then bipolar) combined with the L3 score.
   **Now the only untested FP lever left.** Centring, hard-negative mining and `n_basis` have all been
   measured and found not to help; `n_components` was the one large win and it is spent. L4 is also the
   one that only became viable after the grouping fix (15 of 17 channels captured, versus 6).
3. Random-crop evaluation, so the centre baseline is a fair bar.
4. **Optional, cheap:** finish the k extension (k = 40/48/64 at `n_basis=70`) — @5 sensitivity was still
   rising at the grid edge. Low expected value given how flat that plateau is, and every extra cell swept
   adds to the selection optimism already declared.
5. One confirmatory greedy-vs-components run at the tuned k — that comparison was made at k=4.

**Parked:** class-informed / two-template registration features — the supervisor doubts they would
generalise to other EEG recordings. Weighted or balanced FPCA is parked with them for now, since the k
sweep achieved the same goal (recovering IED-discriminative directions from a negative-dominated pool)
with no class-informed fitting at all.

---

## 9. Neuronostics pilot script (designed, not yet built)

**Task as set by the supervisor:** two groups of recordings — *confirmatory* (contain IEDs, count unknown,
not localised) and *non-confirmatory* (no IEDs). Run L1→L2→L3 on each, output numbers only (we never see
the recordings). If the groups separate, a neurologist knows which EEG to review. **Not a localisation
task** — no FROC, no localisation error, since there are no markers.

- **It is `predict.py` in a loop.** That script already loads the model, loads an arbitrary EDF over CH19
  and runs all three layers. The delta is: walk two folders, write CSV instead of printing. One new file,
  ~40 lines. Nothing else in the repo changes.
- **Recordings are 15–20 min** (vs 11–14 s for Kural) → ~5,000 candidate events each, ~3–4 min per
  recording. 40 recordings ≈ 2 h; 100 ≈ overnight.
  - **`grouping='components'` is strongly preferred on runtime, but is not a hard blocker** (an earlier
    note in this report claimed greedy "would not finish" — that was wrong). Greedy's inner loop runs over
    every candidate for each *seed*, so it is O(seeds x candidates): at ~34,000 candidates and ~10,000
    seeds in a 20-min file, ~3e8 cheap Python iterations, i.e. **tens of seconds to ~1 min per recording**
    on top of L3's 3-4 min. The components path sweeps in time order and breaks past `coincidence_ms`, so
    it is roughly linear and costs ~1-2 s. Real, worth having, not decisive on its own.
- **Send every event's score, not counts.** Any threshold, count or summary statistic is then recomputable
  here without another round trip — and the 0.5 threshold was never calibrated on anything but Kural.
- **CSV 1 — `pilot_recordings.csv`**, one row per file: file, group, duration_s, input sfreq, channels
  matched, error; **L1 candidates, L2 events, L3 counts above threshold**; score distribution **q90, q95,
  q99**; raw counts above 0.5/0.7/0.9. Duration is a separate column so counts can be normalised (or not)
  later.
- **CSV 2 — `pilot_events.csv`**, one row per event: file, group, time_s, channel, **member channel list**,
  n_channels, score.
  - The member list is the L4 hook: spatial features (field spread, focal vs bilateral, which electrodes)
    can be computed here from the returned CSV, so **L4 can be developed later and applied retrospectively
    without him re-running anything**. Only per-channel L3 *scores* are unrecoverable, and scoring every
    member channel would multiply runtime 3–5× (30 h for 100 recordings) — rejected for a pilot.
- **Normalised 0–1 "IED-ness" per recording: use a high quantile, not a count.**
  - the mean is swamped by background events; the max saturates near 1.0 because ~5,000 draws always throw
    up one high score, and it grows with recording length;
  - a quantile is a property of the score *distribution*, so it is duration-invariant in expectation;
  - but the right quantile depends on IED density (5–20 IEDs among ~5,000 events is the top 0.1–0.4%, so
    q99 may be too low and q999 too noisy) → **export q90/q95/q99 and decide from the returned data**.
  - **Per-minute rates were considered and dropped** as the headline: IEDs are bursty, not a steady rate,
    and with all recordings at 15–20 min duration barely varies. Raw counts + duration are exported instead.
- **Robustness at the trust boundary (do not simplify these):** channel-name normalisation (clinical EDFs
  use `EEG Fp1-REF`, `Fp1-A1`, …, not `E FP1-Ref`), resample to 500 Hz (250/256/512 are common and the
  whole cascade is calibrated in ms at 500 Hz), per-file try/except recording the failure in the CSV so one
  bad file does not kill a 2-hour run, and a validation pass printed up front so he can stop in 10 s if the
  montage does not map.
- **Ship with `requirements.txt` and a fresh venv.** The `.joblib` holds fitted scikit-learn / scikit-fda
  objects, which do not reliably unpickle across library versions — the most likely way this dies at step 1.
- **Their event rate is the transfer diagnostic.** If events/min differs sharply from our ~262, something
  upstream differs (reference montage, filtering, noise floor) and every downstream number is
  uninterpretable. That is why L1 and L2 counts are in the CSV, not just L3 scores.
- **Pre-register the threshold.** With all scores in hand it is tempting to pick whichever separates the
  groups best; that is test-set tuning. Fix one threshold from our CV (~10 FP/min operating point) as the
  primary result and mark any sweep as exploratory.
- **Statistics when the CSV returns:** AUC of the chosen quantile between groups with a bootstrap CI
  (directly "does this flag the right EEG"), plus Mann-Whitney U on counts above the pre-registered
  threshold. Lead with effect size and interval, not a p-value — group sizes are unknown and probably small.
- **Expectation to set with him:** this asks an easier question than the FROC. Pooling thousands of events
  into one per-recording number can separate groups even when per-event detection is weak. A positive
  result here is real but is **not** evidence that the detector localises IEDs.
- **Plan:** send a 5-recording-per-group validation run first (10 min of his time, shakes out montage /
  sampling rate / runtime), then the full run with the tuned model.
