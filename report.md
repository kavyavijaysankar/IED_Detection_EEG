# Project report — IED detection (running log)

What was built, what was decided, and what was deliberately *not* done. Updated on every major addition.
Terminology matches the code. Last updated: 2026-08-18.

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
  (window cutting), `Classifier` (L3 FDA model: `_registration` for the elastic/shift switch,
  `_polarity`/`_features` for the polarity column, `_mine`/`_fit_lr` for hard-negative mining,
  `aligned_curves` for diagnostics), `DetectionPipeline` (`fit` / `predict` / `save` / `load`).
  `__main__` self-check covers both grouping rules, mining, polarity, shift registration and the
  scale-invariance of amplitude normalisation.
- **`detect_metrics.py`** — evaluation only, kept out of the model. `labelled_events` is the single
  L1+L2 pass everything is built on (events + windows + labels + recording index); `froc_points`,
  `localisation_errors` and `bootstrap_auc_ci` then work off *any* score vector, so the same code serves
  the test set and the out-of-fold CV scores. Plus `layer1_recall`, `consolidation_recall`,
  `classifier_auc`, `grouped_cv`, `froc`, `froc_summary`, `threshold_at`, `detection_counts`,
  `centre_baseline`, `centring_offsets`, `report`. Module constant `FP_BUDGET = 10` fixes the operating
  point hits/misses/FPs are reported at.
- **`detect_plots.py`** — evaluation figures, one function per panel, each drawing into an `ax` passed in:
  `froc`, `roc`, `pr`, `counts`, `score_hist`, `fold_aucs`, `timeline`, `loc_error`, `event_panel`. Shared by the
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

### A gotcha to know about `.joblib` + `Config` (2026-08-17)
`detect_model.joblib` stores the `Config` **dataclass instance**, and unpickling restores only that
instance's `__dict__`. Any field added to the class *after* the model was saved is therefore absent from the
instance and silently falls back to the **current class default**.

Benign in the case checked (the saved cfg predates `sg_ms`/`reference`/`registration`, and their defaults
reproduce the old behaviour exactly — 42 ms still resolves to 21 samples at 500 Hz), but the mechanism is
real: a field whose new default *changes* behaviour would alter how an old model preprocesses or predicts,
with nothing to signal it. **Rule: give a new Config field the default that reproduces existing saved
models' behaviour, or re-save the model immediately.**

**Correction to an earlier draft of this note:** it claimed the current `detect_model.joblib` was broken by
the `polarity_feature` addition (25 features against a 24-feature classifier) and must not be shared. That
was wrong — checked directly, the saved model has `polarity_feature=True` in its own `__dict__` and its
scaler and LR both hold **25** features, i.e. it is already a polarity-fitted model. The file was evidently
re-saved by a notebook run after the feature was added.

**But the hazard is real and it bit during Phase 3, silently.** The saved cfg stores `sfreq=500.0` and has
no `bandpass` or `reference` field, so on load it inherited the *new* class defaults `(0.5, 45.0)` and
`'average'`. `predict.py` therefore ran a model trained on 500 Hz / unfiltered / recorded-reference data
over 500 Hz / bandpassed / average-referenced input. **It did not error, and the detections it printed
looked entirely plausible** — the scores were meaningless. An earlier draft of this note asserted such a
mismatch would "fail loudly"; it does not, which makes it considerably more dangerous. **`detect_model.joblib`
must not be used or shared until the Phase 4 re-run re-saves it.**

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
  - **REVERSED 2026-08-17 (§8):** average reference now happens in preprocessing, before L1. The reasoning
    above is not wrong about *localisation*, but it answers a narrower question than transfer does — a
    recorded reference differs between labs and an average reference does not.

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
    - **This last clause is now doubtful and is flagged for revisit (2026-08-17).** Shift-only registration
      is precisely the thing that absorbs translation, and the §6 ablation shows it gains **nothing**; and
      after `event_centre` there is only ~0.9 ms of residual shift left for it to find. So warping is not
      earning its keep by absorbing a residual shift. The centring *decision* may still be right, but this
      explanation of why is probably wrong, and the comparison predates both the `n_components` fix and
      amplitude normalisation — so it deserves re-measuring rather than re-interpreting.
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

### Polarity as an explicit feature (`cfg.polarity_feature`) — MEASURED, and ADOPTED (2026-08-17)
Built exactly as specified in §7 Phase 2: `Classifier._polarity` takes the sign of the central deflection
**from the window itself** (mean over ±`cfg.polarity_ms` = ±20 ms, minus the window median) and
`Classifier._features` appends it to the FPCA scores before the scaler. One column, no signature changes
anywhere, no new fitted object, and **scale-free**, so `normalise_amplitude` cannot touch it. The windows
were *not* flipped to a common polarity — that would have destroyed the cue (§7 Phase 2).

**Status: `polarity_feature=True` is now the default (set 2026-08-17), on the balance of the argument
below. The saved model, the figures and the test report still predate it — a Restart & Run All is owed.**

**Base rates first, and they revise the expected effect size.** Over the 4,330 candidate windows 51% are
downward; over the 64 positives 75% are — reproducing the Phase-2 measurement. But the likelihood ratio
against the *whole* negative pool is therefore ≈**1.47** downward / **0.51** upward, not the ~1.9 / 0.42
estimated in Phase 2 from the top-30 FPs. The cue is sharpest at the very top of the ranking and much
weaker against ordinary background — which is where the FROC's low-FP end lives, so a modest effect was the
right expectation for a slightly different reason than the one recorded.

**Paired grouped CV** (registration and FPCA fitted once per fold and shared by both arms, so the folds,
the representation and the scaler are identical and only the LR's feature matrix differs; ~11 min). The
`base` arm reproduces the current normalised CV exactly, which validates the harness:

| | base (current default) | + polarity |
|---|---|---|
| ROC-AUC | 0.831 [0.759, 0.894] | **0.840** [0.766, 0.903] |
| PR-AUC (chance 0.0148) | 0.133 | **0.146** |
| sens @1 / 5 / 10 / 25 / 50 / 100 FP/min | 0.10 / 0.47 / 0.57 / 0.71 / 0.82 / 0.90 | 0.10 / 0.47 / **0.61** / **0.76** / **0.88** / 0.90 |
| hits / misses / FPs @ ≤10 FP/min | 28 / 21 / 187 | **30 / 19 / 176** |

- Paired recording-level bootstrap: **ΔPR-AUC +0.0135, 95% CI [−0.0006, +0.0323], P(>0)=0.97**;
  ΔROC-AUC +0.0093, 95% CI [−0.0137, +0.0320], P(>0)=0.79.
- **The LR uses the column, in the predicted direction, in every fold.** Coefficients (standardised
  features) −0.519 · −0.758 · −0.637 · −0.594 · −0.873 — negative throughout, i.e. downward windows score
  higher, as predicted before the run. Its |coefficient| ranks 8 / 1 / 3 / 4 / 2 among the 25 features and
  is ~2× the mean |coefficient| of the 24 FPCA columns.
- **Mechanism confirmed directly.** Mean score on *negatives*, downward vs upward: base **0.247 vs 0.242**
  (i.e. L3 was genuinely blind to sign, as Phase 2 inferred); with the column, **0.309 vs 0.155**. The
  feature does what it was built to do — it is not a proxy for something already in the FPCA scores.
- **What actually changed at the operating point:** 27 IEDs hit by both arms, **3 newly hit, 1 newly
  missed**; of the false positives, 151 shared, **36 removed and 25 new**. So the net +2 hits / −11 FPs is
  a real re-ranking, not a threshold shift — but it is a churn of tens of events, not a transformation.
- **The surviving false positives are now more IED-like in polarity** (45% → 60% downward overall; top-30
  FPs 40% → 50%). Expected, and worth saying in the writeup: the feature culls upward mimics, so what
  remains is the subset polarity cannot discriminate. It is a cue that is spent once used.

#### `polarity_ms` — robustness check, explicitly not a selection
`polarity_ms` is the half-width of the central stretch the sign is read from (20 ms = ±10 samples at
500 Hz, so a 40 ms average on the spike peak, compared against the window median).
**Pre-committed before the run: 20 ms stays whatever the sweep returns**; the question was whether the
polarity result depends on the choice, not which choice scores best. Sweeping *for* a value would tune a
knob on an effect whose interval already touches zero, over 49 IEDs. Widths share one registration per
fold, so all five cost one CV run (~11 min).

**Pre-registered prediction: wide widths should degrade**, because the after-going slow wave has the
opposite polarity to the spike, so a wide central average mixes the two and mismeasures the sign.

The sign measurement itself, before any model is involved:

| `polarity_ms` | samples | downward, all windows | downward, positives | same sign as 20 ms |
|---|---|---|---|---|
| 4 | 4 | 51% | 75% | 99% |
| 10 | 10 | 51% | 75% | 99% |
| **20** | 20 | 51% | 75% | 100% |
| 40 | 40 | 51% | 75% | 94% |
| 80 | 80 | 50% | **69%** | **79%** |

Then the paired CV (`base` = no polarity column, for reference):

| arm | ROC-AUC | PR-AUC | @1 | @5 | @10 | @25 | @50 | @100 | hits/miss/FP @≤10 | polarity coef |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.831 | 0.133 | 0.10 | 0.47 | 0.57 | 0.71 | 0.82 | 0.90 | 28 / 21 / 187 | — |
| 4 ms | 0.840 | 0.146 | 0.10 | 0.47 | 0.61 | 0.76 | 0.88 | 0.90 | 30 / 19 / 175 | −0.672 |
| 10 ms | 0.840 | 0.146 | 0.10 | 0.47 | 0.61 | 0.76 | 0.88 | 0.90 | 30 / 19 / 176 | −0.671 |
| **20 ms** | **0.840** | **0.146** | 0.10 | 0.47 | 0.61 | 0.76 | 0.88 | 0.90 | **30 / 19 / 176** | **−0.676** |
| 40 ms | 0.839 | 0.143 | 0.10 | 0.45 | 0.57 | 0.73 | 0.86 | 0.92 | 28 / 21 / 178 | −0.664 |
| 80 ms | 0.824 | 0.132 | 0.08 | 0.47 | 0.59 | 0.71 | 0.82 | 0.92 | 29 / 20 / 189 | **−0.016** |

Paired ΔPR-AUC vs base: 4 ms +0.0135 [−0.0006, +0.0323] · 10 ms +0.0138 [−0.0004, +0.0327] ·
**20 ms +0.0135 [−0.0006, +0.0323]** · 40 ms +0.0106 [−0.0025, +0.0287] ·
80 ms **−0.0012 [−0.0069, +0.0049], P(>0)=0.32**.

- **4–20 ms is a flat plateau** — identical to three decimals on both AUCs and on the operating point. The
  polarity result does **not** hinge on the exact half-width, which is what the check was for.
- **The prediction is confirmed, and it shows up in three independent places at 80 ms**: the sign
  measurement degrades (positives 75% → 69% downward, only 79% of windows keep their sign), the LR's
  coefficient collapses **−0.68 → −0.016**, and detection falls back to baseline. The model correctly stops
  trusting a feature that has become noise — a useful sanity property to be able to report.
- **20 ms is now kept for a measured reason as well as a pre-specified one: it sits mid-plateau**, with
  margin on both sides. 4 ms is two samples from the noise-dominated edge (a risk on data with a different
  noise floor) and 40 ms is already at the degradation boundary. Mid-plateau is the robust place to stand.
- Report this as a **sensitivity analysis, not a selection** — the value was fixed in advance and the table
  is evidence of insensitivity. Had one width wildly beaten the rest, the correct reading would have been
  that the effect is noise, not that the winner should be adopted.

**The decision is finely balanced and both readings are defensible — recorded here in full because the
writeup needs the argument either way.**
- *For adopting:* it is a **single pre-specified change** with the direction of effect predicted in advance
  and confirmed in all five folds, so — unlike the mining sweep — its interval carries **no post-selection
  bias**. It improves both axes of the quoted operating point simultaneously (+2 hits *and* −11 FPs), which
  most levers here do not. It adds no fitted object, no selection procedure and no runtime. And it is
  physiologically motivated (IEDs are predominantly surface-negative), which is a transfer argument of the
  same kind as amplitude normalisation.
- *Against:* **the FROC's low-FP end did not move at all** — 0.10 @1 and 0.47 @5 in both arms — and that
  end is this project's stated primary criterion ("only the top of the ranking is clinically usable, follow
  the FROC, not the AUC"). The gains are mid-curve (@10–@50) plus PR-AUC, whose 95% CI still touches zero.
  By the letter of the standard applied to centring and mining — *the incumbent stays unless the challenger
  clears the bar* — that is not a clear pass.
- **Transfer risk, and it cuts the opposite way to amplitude normalisation — this must be declared.** The
  sign of a discharge on the representative channel is a property of the **reference montage**, not of the
  discharge alone. Surface-negativity is physiological, but it is only read as a negative deflection under
  a referential montage like the recorded reference used here; under a different reference convention, or
  when the representative channel lands on the far side of a dipole, the sign can flip systematically.
  Normalisation made the model *less* dependent on the acquisition setup; polarity makes it *more* so.
  - Cheap mitigation, and it costs nothing to add: have the pilot export the **downward fraction per
    recording** alongside the event rate. Our values (51% of all events, ~75% of the top-scoring ones) are
    the reference; if the Neuronostics data departs sharply from them, the montage convention differs and
    the polarity column is mis-signed there — diagnosable *before* the scores are interpreted.

### Registration — `cfg.registration`, and what the warping actually does (2026-08-17)
Raised by the supervisor: the missed IEDs look like textbook discharges when plotted, so is the **warping**
damaging them? If so, try shift registration instead. Two things were run: a read-only diagnostic on the
saved model, then a full elastic-vs-shift ablation.

#### Diagnostic — the hypothesis is NOT supported
Out-of-fold hit/miss split at ≤10 FP/min, curves from `Classifier.aligned_curves` (in-sample: the saved
registration was fitted on all 90, so this asks what warping does to these curves, not what the scoring
model saw).

| | misses (21) | hits (28) | p |
|---|---|---|---|
| warp magnitude, mean \|gamma(t) − t\| | 56.1 ms | 62.4 ms | 0.58 |
| curve movement, ‖smoothed − registered‖ / ‖smoothed − median‖ | 1.148 | 1.246 | 0.89 |
| local time-stretch at the spike, mean gamma' over ±40 ms | 1.099 | 1.154 | 0.93 |

- **All null, and all in the wrong direction** — the misses are warped slightly *less* than the hits.
  Warping is not doing anything selectively to the IEDs L3 misses.
- **A proposed mechanism was pre-registered and REFUTED.** The idea: elastic warping can locally rescale
  time, and duration is part of what makes an IED an IED (spike 20–70 ms vs a broader mimic), so warping
  might normalise the duration difference away. Direct test — correlation between a curve's width before
  registration and how much it gets stretched — **+0.030, p=0.79.** Broad curves are not being compressed,
  narrow ones not stretched. It does not happen.
- **Measurement trap worth recording: amplitude statistics are useless here.** The first pass compared
  peak-to-peak before and after and got a ratio of exactly 1.00 for every group. Time warping is a
  reparametrisation — `registered(t) = original(gamma(t))` with gamma monotonic — so the *set* of values is
  preserved and any amplitude statistic is invariant **by construction**. Only timing quantities
  (gamma, gamma', peak position) and whole-curve distances carry information.

#### What the diagnostic did establish
- **Registration is aggressive.** Curve movement is 0.85–1.25, i.e. the change it makes is as large as the
  curve's own signal, with a mean point displacement of ~55 ms — about one spike width.
- **It moves the spike off centre**, by 200–300 ms in several cases (see
  `figures/detection_registration_misses.png`). `event_centre` places the spike at t=0 and the warping
  walks it away again — in hits and misses alike.
- **CORRECTION to an earlier figure in this log.** The IED-vs-mimic distortion asymmetry was first quoted
  as 1.25 vs 0.85 (≈47%) from the top-30 FPs. Against the **full** negative pool, out-of-fold, it is
  **1.175 vs 1.058 — a ratio of 1.11, i.e. 11%.** The original number was inflated by selecting the FPs on
  score, the same circularity flagged elsewhere in §7. The defect is real but small.
- **The spike's shape largely survives; it is mainly shifted** (user's read of the plots, and consistent
  with the numbers). The 1.15 movement figure is a *whole-curve* measure, so it is dominated by background
  being rearranged, not by the discharge being deformed. Together with the 11% correction, very little of
  the original "warping damages IEDs" concern survives.
- **Why the warping is so large: the spike is ~2.5% of the window.** The window is 2 s and a spike is
  20–70 ms, so alignment is overwhelmingly driven by the other ~97.5% — background. This is the origin of
  the deferred `classifier_halfwin_s` experiment (§8).

#### Ablation — elastic vs shift, swept across k
`cfg.registration = 'elastic' | 'shift'` (`Classifier._registration`); `'shift'` is
`LeastSquaresShiftRegistration`, a rigid time shift that cannot reshape a curve.

- **First, a measurement that reframed the experiment: shift registration is a near no-op on real windows**
  — median shift **0.9 ms**, curve movement **0.025** (elastic: 1.08). `event_centre` has already centred
  every window, so a shift-only method has nothing left to remove. **So this ablation is effectively
  "with warping" vs "no registration at all".**
- k was swept for **both** arms, because shift leaves different variation in the curves and FPCA may need a
  different number of components. Comparing at one k is the trap that invalidated the greedy-vs-components
  comparison at k=4 (§5). Both arms share folds and B-spline smoothing; `polarity_feature` held OFF so
  registration is the only difference.

| k | elastic PR-AUC | elastic @5 | shift PR-AUC | shift @5 |
|---|---|---|---|---|
| 8 | 0.054 | 0.18 | 0.059 | 0.24 |
| 12 | 0.097 | 0.35 | 0.065 | 0.20 |
| 16 | 0.112 | 0.33 | 0.058 | 0.16 |
| **24** | **0.133** | **0.47** | 0.068 | 0.22 |
| 32 | 0.131 | 0.49 | **0.082** | **0.31** |
| 40 | 0.123 | 0.49 | 0.070 | 0.29 |
| 48 | 0.108 | 0.41 | 0.080 | 0.24 |

- At each arm's own best k: elastic (k=24) **ROC-AUC 0.831 / PR-AUC 0.133 / @5 0.47 / 28-21-187** versus
  shift (k=32) **0.731 / 0.082 / 0.31 / 23-26-188**. Elastic has 62% more PR-AUC and finds 7 more IEDs of
  49 at ≤5 FP/min.
- **From k=12 upward, every elastic cell beats shift's single best cell** on both PR-AUC and @5, so this is
  not an artefact of the k choice — the sweep confirms rather than overturns.
- Paired recording-level bootstrap of ΔPR-AUC (shift − elastic): negative at every k but k=8; interval
  excludes zero at k=16, 24 and 40. At k=24: **−0.0665, 95% CI [−0.1310, −0.0134], P(>0)=0.01.**
- For scale, shift's best (0.082) is barely above the prominence baseline (0.054, §7), while elastic
  (0.133) is 2.5× it.
- **DECISION: keep `registration='elastic'`, k=24 unchanged (so no new selection).** The decision rule was
  **pre-committed before the numbers were seen** — adopt shift if better or within noise, keep elastic only
  if it was clearly worse on PR-AUC or lost more than ~3 IEDs of 49 at @5. It was clearly worse on both.
- **This is a genuine positive result, not just a rejected idea: the FDA registration step earns its place
  by a wide margin.** Fisher-Rao registration is the methodological centrepiece and its contribution had
  been assumed, never tested. It has now been ablated and it is worth PR-AUC 0.133 → 0.082.
- **And it identifies the mechanism.** Shift-only registration is exactly the thing that absorbs
  *translation*, and it gains nothing; elastic, which reshapes, gains a lot. So the warping earns its keep
  by **normalising shape and aligning context — not by fixing alignment.** Note the corollary: since the
  spike itself is mostly just shifted, the gain appears to come from aligning the ~97.5% of the window that
  is *not* the spike.
- The `'shift'` branch is kept (six lines) as a re-runnable ablation, since the preprocessing work will
  change the curves and the comparison may be worth repeating.
- **Consequence for a parked idea: warp magnitude as an L3 feature is DOWNGRADED, not promoted.** With the
  asymmetry at 1.175 vs 1.058 it would be a weak univariate feature, and it measures distance from *this*
  dataset's mimic-shaped template — a worse transfer property than polarity. It is also mutually exclusive
  with fixing the root cause: fix the template or drop the warping and the feature evaporates.

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

### CURRENT RESULTS — final pipeline, full Restart & Run All (2026-08-18)
Config: 250 Hz · 0.5–45 Hz zero-phase bandpass · two-threshold bad channels · average reference over good
channels · `components`/0.7 · `centre_on='amplitude'` · `normalise_amplitude=True` · polarity ON ·
elastic registration · `n_basis=70`, `k=24`. Grouped 5-fold CV over **4,157 windows / 60 positives /
49 IEDs**.

| | grouped CV (49 IEDs) | held-out test (5 IEDs) |
|---|---|---|
| ROC-AUC | **0.865** [0.797, 0.929] | **0.874** |
| PR-AUC (chance) | **0.170** (0.0144) | 0.143 (0.011) |
| per fold | 0.840 · 0.865 · 0.817 · 0.919 · 0.894 | — |
| sens @1 / 5 / 10 / 25 / 50 / 100 FP/min | 0.14 / 0.51 / 0.67 / 0.82 / 0.92 / 0.94 | 0.20 / 0.20 / 0.20 / 0.60 / 0.80 / 1.00 |
| hits / misses / FPs @ ≤10 FP/min | **33 / 16 / 181** (9.5/min) | 1 / 4 / 1 (0.5/min) |
| localisation, median | **12 ms** | 4 ms |
| L1 / L2 recall | 1.00 / 0.9796 | 1.00 / 1.00 |
| centre baseline | 0.29 @ 4.0 FP/min | 0.40 @ 4.0 FP/min |

- **The test set corroborates the CV: 0.874 against 0.865.** Four things have been adopted since the test
  set was last read (amplitude normalisation, polarity, 250 Hz, bandpass + average reference), all chosen
  with reference to the 90-train CV — and the held-out estimate lands on top of it. **That is the single
  best piece of evidence that the CV is not badly selection-inflated.**
- **The test FROC is not interpretable and must be quoted with that caveat.** At ≤10 FP/min it gives 1 of 5
  against the centre baseline's 2 of 5. Under the CV's own 67% rate, drawing ≤1 of 5 has probability ~4% —
  a low draw, worth stating honestly rather than explaining away, but it is **one realisation of five
  IEDs** while the test ROC-AUC simultaneously *improved* (0.847 → 0.874). This is the documented
  AUC-versus-FROC divergence: AUC scores the whole ranking, the operating point only its very top.
- **A methodological wrinkle found here:** the test operating point landed at threshold **≥ 1.00**,
  admitting one event and one FP when 21 FPs were within budget. 20 of 4,157 out-of-fold CV scores are
  ≥0.999 and one is exactly 1.0, so the LR mildly saturates — and on a small set the ≤10 FP/min point can
  land on that ceiling, where it is decided by ties at 1.0 rather than by a meaningful threshold. Record
  this as a limitation of quoting an operating point on a 5-IED set.
- **Do not iterate on the test result.** Investigating why those four were missed, and adjusting anything
  in response, converts the last held-out estimate into a tuning signal and destroys the corroboration
  above. The legitimate route to a firmer operating point is more IEDs (the CV's 49), not more scrutiny of
  these five — which is the argument for putting bootstrap intervals on the CV's FROC sensitivities.
- **Honesty note on wording:** this log has called the test set "the only selection-free estimate left".
  That is now too strong. It has been read several times across the project (at k=4, at k=24, after
  normalisation), and the configuration was chosen with reference to CV numbers. It is **held out from
  fitting**, which is the accurate claim, and each further read erodes its independence.
- Progression of the CV operating point across the session, for the writeup:
  500 Hz polarity-off **28/21/187** → 500 Hz polarity-on **30/19/176** → 250 Hz + bandpass **28/21/182**
  → + average reference **33/16/181**. Misses are the lowest they have been.

#### Superseded — the 500 Hz results these replace
Kept because the `n_components` comparison is the single largest effect in the project's history.
Grouped CV over 4,330 windows / 64 positives, recorded reference, 500 Hz, polarity off:

| | k=4 (old default) | k=24 |
|---|---|---|
| CV AUC | 0.657 [0.582, 0.730] | 0.828 [0.763, 0.889] |
| average precision (chance 0.0148) | 0.028 (~2× chance) | 0.117 (~8× chance) |
| sens @1 / 5 / 10 / 25 / 50 / 100 | 0.00 / 0.04 / 0.14 / 0.35 / 0.53 / 0.69 | 0.08 / 0.41 / 0.59 / 0.71 / 0.84 / 0.92 |

- At k=4 the detector scored 0.04 at ≤5 FP/min, seven times *worse* than "always guess the middle". k=24
  was the first configuration to clear that bar. Test set at the same config: AUC 0.847 (0.571 at k=4).
- Still short of clinical usability (published detectors: 0.7–0.9 at 0.5–5 FP/min), though the current
  0.51 @5 is no longer in a different league. **L3 discrimination remains the bottleneck.**

### Error analysis, Phase 1 (out-of-fold CV scores, ≤10 FP/min operating point: score ≥ 0.855)
29 hits / 20 misses / 176 false positives over the 90-train. All 20 misses are **L3 ranking failures** —
L1 and L2 recall are both 1.00, so every IED already has a near-marker event in the pool.

- **Q1 — certainty does not explain the misses.** clear_positive 11/20 found (55%), unclear_positive
  18/29 (62%). If anything *inverted*, and well within noise at these counts.
  - **The hoped-for reframing is dead:** we cannot say "it finds the clear ones and fails on the ambiguous
    ones". The model's failures do not track neurologist confidence, which means they are not "hard cases"
    in the clinical sense — they are something else, and only looking at them will say what.
- **Q2 — there are no cheap sensitivity gains. This is the most important operational finding.**
  - Median **+776 false positives** to catch one more missed IED. **0 of 20 misses cost ≤50 FPs**, and
    **14 of 20 cost >500** — effectively unreachable at any usable threshold.
  - The cheapest miss (S69, best near-marker score 0.728) still sits below **272** non-hit events.
  - **Score gaps are small but rank gaps are enormous**: the misses score 0.49–0.73 against a threshold of
    0.855, yet hundreds of mimics are packed into that same band. **Threshold tuning is exhausted** —
    improving sensitivity requires the model to re-rank, not to move an operating point.
- **Q3 — false positives are concentrated.** 176 FPs over 45 of 90 recordings; half the recordings produce
  none, the **worst 9 (10%) hold 53%**, worst single recording 22.
  - Points at a **recording-level nuisance factor**, so per-recording amplitude/background normalisation is
    now a live and cheap candidate — the FDA features are amplitude-sensitive, so a noisy-background
    recording produces uniformly high-scoring negatives.
- **Q4 — epileptic recordings produce 6× the false-positive rate of non-epileptic ones**:
  155 FPs over 10.5 min (**14.8 FP/min**) versus 21 FPs over 8.6 min (**2.5 FP/min**).
  - Two readings, and they have very different consequences:
    (a) the background EEG of epileptic patients genuinely contains more sharp transients (clinically
    plausible — irritative zone, subclinical discharges); or
    (b) **the "exactly one IED per recording" label is incomplete**, and some of those 155 are real
    unmarked discharges. The FROC's FP count depends entirely on that assumption, so if (b) holds, the
    quoted FP/min is overstated and the ceiling is the label set rather than the model.
  - Distinguishing them is a **Phase 2 question**: plot the top-scoring FPs from epileptic recordings and
    ask whether they look like IEDs.
  - **Either way it is good news for the Neuronostics pilot** (§9), whose whole premise is that recordings
    containing IEDs produce more detections. This is that experiment run on Kural, and it separates 6:1.
    Worth quantifying directly as a free pre-test: per-recording detection count (or score quantile) as a
    discriminator between epileptic and non-epileptic *training* recordings, reported as an AUC.

### Error analysis, Phase 2 — read by eye (user, domain judgement)
- **The top false positives clearly are NOT IEDs to a trained eye.** So the ceiling is the **model, not the
  label set** — the detector is making errors a human would not. This also kills the "unmarked discharges"
  reading of the residual 1.6× epileptic/non-epileptic FP asymmetry (§7 Q4), and means the headroom is real.
- **Most of the missed IEDs show a downward dip** that the false positives lack. User's read: the negative
  deflection is important to what makes an IED an IED.

**Hypothesis this raises — POLARITY, and it is the same shape of bug as the amplitude confound:**
- L1 uses **|2nd derivative|** and L2 uses **|correlation|** — both deliberately polarity-invariant, the
  latter specifically so a dipole's reversed far side still groups (§5). **L3 is the only layer that is
  not**: FPCA runs on raw *signed* µV, so an upward and a downward spike sit in different regions of
  feature space.
- Mechanism: the representative channel is chosen by peak-to-peak, so which side of the dipole it lands on
  varies. With only 64 positives, a polarity split halves the effective training signal for the rarer sign,
  and IEDs presenting with that sign become out-of-distribution → missed.
- **Do not "fix" this by flipping windows to a common polarity without measuring first.** IEDs are
  predominantly surface-negative clinically, so polarity is plausibly *informative*; flipping would make L3
  consistent with L1/L2 but could destroy a real cue. Two live possibilities: (a) polarity splits the
  positive class and hurts, (b) polarity is informative and rep-channel selection is scrambling it.
**MEASURED** (sign of the central deflection, out-of-fold, ≤10 FP/min operating point):

| group | n | downward |
|---|---|---|
| hits | 28 | 71% |
| misses | 21 | 81% |
| **all IEDs** | **49** | **~75%** |
| top false positives | 30 | **40%** |

- **Polarity is a real IED cue, and L3 is not using it.** IEDs are ~75% downward; the top-scoring false
  positives are 40%, indistinguishable from a coin flip (SE ~0.09). If L3 were exploiting sign, its
  highest-scoring errors would be predominantly downward, mimicking discharges. They are not.
- **The polarity-split hypothesis is NOT supported.** Hits 71% vs misses 81% is well inside noise
  (SE ~0.08/0.09), so the rarer sign is *not* going out-of-distribution and being missed. The user's read
  that missed IEDs show the dip is correct — but the hits show it just as often, so **the dip does not
  explain the misses**. The misses remain an unexplained morphology failure.
  - **SUPERSEDED 2026-08-17 — the misses now have an explanation: relative prominence.** See §7
    "Error analysis, Phase 2b". They are the low-prominence discharges, not a morphology mystery.
- **Do NOT flip windows to a common polarity.** That was the obvious "make L3 consistent with L1/L2" move
  and it would destroy the cue just identified.
- **Next lever instead — give L3 polarity explicitly.** Compute the sign of the central deflection *from
  the window itself* inside `Classifier.fit` and concatenate it to the FPCA score matrix before the scaler:
  one extra column, one extra line in `predict_proba`, **no signature changes anywhere** (polarity needs no
  data the classifier does not already hold). Rationale: with 64 positives and 24 components the LR may
  lack the data to isolate a direction encoding sign, so hand it the feature directly.
  - Judge it the same way as everything else: grouped CV, FROC's low-FP end and PR-AUC, keep only if it
    clears noise. Expect a modest effect — as a lone binary feature the likelihood ratio is ~1.9 for
    downward, ~0.42 for upward.
  - **BUILT AND MEASURED — see §6 "Polarity as an explicit feature".** The mechanism check there confirms
    this section's inference directly: without the column, L3's mean score on negatives is 0.247 downward
    vs 0.242 upward, i.e. it really was blind to sign. The effect on detection is small and the decision is
    finely balanced; the ~1.9 / 0.42 likelihood ratio quoted above holds against the *top* FPs but is
    ~1.47 / 0.51 against the whole negative pool.

### Error analysis, Phase 2b — the misses EXPLAINED: relative prominence (2026-08-17)
Came out of the registration diagnostic. Measured on the out-of-fold scores, ≤10 FP/min operating point.
**Relative prominence** = peak-to-peak of the *normalised* window, i.e. how far the biggest deflection
stands out in multiples of that recording's own background MAD.

| group | n | median prominence | IQR |
|---|---|---|---|
| hits | 28 | 19.5× | [16.3, 23.7] |
| **misses** | **21** | **11.9×** | [9.9, 15.2] |
| top false positives | 30 | 27.9× | [19.2, 33.2] |

- 86% of the misses sit below the median hit; 100% of the top FPs sit above the median miss.
- **The missed IEDs are the small ones** — small relative to their own background. Visible by eye in
  `figures/detection_registration_misses.png`: the misses run ±2 to ±6 while the reference hits run ±10
  to ±15.
- **This comparison is partly circular and must be quoted as such**: hits and misses are *defined* by
  score, and the top FPs were selected on score, so "the low-scoring ones are smaller" is close to
  restating that score tracks size. The non-circular versions are below.
- **Non-circular measurements:** Spearman corr(score, prominence) = **+0.141** over all 4,330 windows,
  +0.122 over negatives, and **+0.592 over the 64 positives**. So L3's score is *not* globally driven by
  size — but *among true IEDs* it strongly is. Which IEDs get found: the prominent ones.
- **Interpretation: L3 has no mechanism for promoting a discharge that is modest in size but well-formed in
  shape.** That is what the misses are. This **retires "unexplained morphology failure"** (§7 Phase 2) and
  replaces it with a specific, quantified deficiency.
- Four hypotheses for the misses have now been tested: polarity split (refuted, §7 Phase 2), scale artefact
  (refuted, §9), warping (refuted, §6), **low relative prominence (supported)**.
- **The remaining lever on sensitivity is therefore a feature that lets shape compete with size** — for
  example normalising each window by its *own* peak-to-peak so only shape remains. Risk to measure, not
  assume: absolute prominence is genuine evidence of a discharge, so removing it may cost more than it
  recovers. Same discipline as recording-level normalisation (§9).
- **One case that contradicts the pattern and is worth a look:** S53, the lowest score of all 21 (0.00), has
  the *largest* amplitude of the 21 (±20) and a clear sharp trough. Whatever is wrong there is not size.

### A second null model — the prominence baseline, and a lesson about ROC-AUC
Ranking every candidate by relative prominence alone, with **no model and no fitting at all** (so no CV is
needed — it cannot overfit):

| | ROC-AUC | PR-AUC | @1 | @5 | @10 | hits/miss/FP @≤10 |
|---|---|---|---|---|---|---|
| L3 (out-of-fold) | 0.831 | 0.133 | 0.10 | 0.47 | 0.57 | 28 / 21 / 187 |
| L3 + polarity | 0.840 | 0.146 | 0.10 | 0.47 | 0.61 | 30 / 19 / 176 |
| **prominence alone** | **0.801** | **0.054** | 0.02 | **0.12** | 0.35 | **17 / 32 / 185** |
| centre baseline | — | — | \- | \- | \- | sens 0.29 @ 4.0 FP/min |

- **Adopt this as a second null model alongside the centre baseline** and report it: at ≤10 FP/min it is a
  *stronger* bar (0.35 vs 0.29), and unlike the centre baseline it is a real ranking, so it appears on the
  FROC at every budget.
- **It is also the cleanest available demonstration of why ROC-AUC is not the headline here.** Prominence
  alone reaches ROC-AUC 0.801 against L3's 0.831 — a gap of 0.03 that makes the whole FDA pipeline look
  nearly pointless. On the metrics that matter it has **2.5× less PR-AUC and one quarter the sensitivity at
  ≤5 FP/min** (0.12 vs 0.47). A 0.03 ROC-AUC gap concealing a 4× difference where it counts. This log has
  argued that point since §7; it is now demonstrated with a concrete case.
- Note L3 must be read out-of-fold while prominence needs no fitting, so if anything the comparison is
  generous to prominence.

### A convergent finding worth stating in the writeup
Three independent results now point the same way:
- `n_basis` makes no consistent difference (§6) — 30 B-splines resolve a spike as well as 100, so fine
  spike structure is not being used;
- relative prominence explains the misses (above) — size matters more than shape;
- the registration gain survives only in the *elastic* arm, and the spike is mostly just shifted (§6) — so
  the benefit comes from aligning the ~97.5% of the window that is **not** the spike.

**Together: L3 is largely judging the window's context and the discharge's prominence, not the fine
morphology of the spike itself.** That is an uncomfortable but defensible characterisation of what the model
does, it is far more informative than "L3 discrimination is the bottleneck", and it explains why the
promising-looking morphology levers (centring, `n_basis`, polarity, warping) have all produced small or null
effects. It also predicts where the remaining headroom is: the window (§8 A) and shape-versus-size (§8 B).

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

**0. DONE — re-run of `detection_run.ipynb` with `normalise_amplitude=True`.** The saved model, the test
report, all figures, the error analysis and the pilot pre-test are now the normalised ones (2026-08-16).
L1/L2 outputs (recall, 1.00 events per IED, 5.28 channels, centring offsets) are computed on the raw
signal and were unaffected.

**0b. Re-run `detection_run.ipynb` for the polarity feature — OWED.** `polarity_feature=True` was adopted
2026-08-17 (§6) but the saved model, all figures and the test report are still the polarity-off versions.
Restart & Run All with `RUN_GROUPED_CV=True`, ~18 min; leave `polarity_sweep` and `polarity_ms_sweep` off,
both have already been run and recorded. Note that cell 1 overwrites `detect_model.joblib`.
Expected after the re-run: CV ROC-AUC 0.840, PR-AUC 0.146, 30/19/176 at ≤10 FP/min. **The 10-recording
test set will be touched by this run** — it is the only selection-free estimate left, and polarity is now
the second thing (after `normalise_amplitude`) adopted since it was last read, so quote it once and do not
iterate against it.

### THE CURRENT PLAN — preprocessing first (agreed with the supervisor, 2026-08-17)
Rationale: real EEG is 15–20 min with patients being poked, sneezing and falling asleep, so the recordings
the detector must eventually run on are far more artefact-laden than Kural's 11–14 s hand-picked clips.
**Kural cannot demonstrate that preprocessing helps** — it can only show it does no harm here. That is the
same argument structure as amplitude normalisation (§9): adopted for transfer, validated as no-harm, and it
must be *stated that way* rather than dressed up as a performance gain.

**One phase at a time, each reported back before the next starts.** Every phase invalidates the model,
figures and CV, so they are separated to keep attribution clean rather than bundled into one run.

- **Phase 0 — tidy-up, no behaviour change. DONE 2026-08-17.** One source of truth for `sfreq`
  (`Config.sfreq`; removed `SF` from `detect_data`, the `sf=500` defaults in `detect_stage1`, and the import
  in `predict.py`); `sg_win` (samples) → **`sg_ms=42.0`** plus `Config.sg_samples()`, and the duplicate
  `SG_WIN` in `detect_stage1` deleted; marker sample and `dur` derived from `cfg.sfreq`;
  `load_recording_path`/`load_recording`/`load_dataset` take `cfg` instead of a `reference` string, with
  `reference` now a Config field; `predict.py` passes `pipe.cfg`; the 500 Hz assert replaced by a
  resample-if-needed.
  - **`Config.sg_samples()` forces the window ODD** — `savgol_filter` requires it, and 42 ms at 250 Hz
    rounds to 10, which would raise. It returns 21 at 500 Hz (the original value) and 11 at 250 Hz. Asserted
    odd across 200/250/256/500/512/1000 Hz in the `detect_stage1` self-check.
  - **Resampling is guarded** (`if raw.info['sfreq'] != cfg.sfreq`), so a file already at the target rate is
    provably untouched rather than round-tripped through a filter.
  - **VERIFIED as a no-op at 500 Hz, bit-identically:** the 4,330×1,000 window matrix, `Y`, `G`, all markers,
    all durations, and every event's time/channel/n_channels are `array_equal` to a pre-refactor snapshot;
    L1 recall 1.0000, L2 recall 1.0000 and 227.5 events/min unchanged. Since L3's input is bit-identical, no
    CV re-run is needed to prove the refactor is clean — which is why the snapshot-and-compare test was worth
    more than re-running the CV.
  - Three of these are not cosmetic. `sg_win` is currently hard-coded **twice** and L1 reads its own module
    constant, so changing the config would silently fix only half the pipeline. The marker conversion and
    `dur` are hard-wired to 500, so resampling without fixing them would move every ground-truth marker and
    look exactly like "the filters destroyed the signal". And `predict.py` currently calls
    `load_recording_path` with no reference, so it would preprocess *differently from training* — silently.
- **Phase 1 — resample to 250 Hz. DONE 2026-08-17** (250 Hz confirmed by the supervisor). One line:
  `Config.sfreq = 250.0`. `sg_samples()` → 11; every other threshold is in ms so it is unchanged in *time*
  (hit tolerance still ±100 ms, classifier window still 2 s = 500 samples). `n_reg_points=100` is untouched,
  so **L3's input dimensionality does not change at all** — resampling bites only on L1 (a cubic fitted over
  11 samples instead of 21) and L2 (correlation over ~38 instead of 75).

  | 90-train | 500 Hz | 250 Hz |
  |---|---|---|
  | **L1 IED recall** | 1.0000 | **1.0000** |
  | L2 consolidation recall | 1.0000 | **0.9796** |
  | candidates per recording | — | 343.7 |
  | events / min | 227.5 | 217.2 |
  | positive windows per IED | 1.31 | 1.27 |
  | channels per event | 5.18 | 5.18 |
  | window pool | 4,330 / 64 pos | 4,133 / 62 pos |

  - **The CV was deliberately skipped.** No decision hangs on it — 250 Hz is agreed and would not be
    reverted on a CV result — so it would have been ~18 min of information we would not act on. Everything
    is deterministic, so it can be run retrospectively if a later phase needs bisecting.
  - **Anti-aliasing verified, not assumed.** It is applied inside mne's FFT-based `resample`; writing a
    separate low-pass would filter twice. Demonstrated by PSD (µV²/Hz — a *density*, so comparable across
    rates): every retained band 0.5–124 Hz is preserved at ratio 0.89–1.24, the spread being Welch
    resolution rather than signal change. **Only 0.0105% of the original power sits above 125 Hz**, so Kural
    is already effectively band-limited and resampling is close to lossless here.
    - Measurement trap: a first attempt used raw FFT power, which scales with N², so halving the sample count
      gave a spurious *uniform* 0.25 ratio in every band. Use a density, or renormalise.
  - **L2 recall 0.9796 — diagnosed, and it is NOT a resampling regression.** The lost IED is **S26**. L1
    finds it 8 ms from the marker and a 17-channel event captures it with **16 of 23 members within ±100 ms**
    — but the event's *reported* time (its max-peak-to-peak member, channel P8) sits at **+200 ms**, outside
    tolerance. **At 500 Hz that same event is also reported at +194 ms, also outside tolerance**; the 500 Hz
    hit was carried by a *separate, well-timed fragment* that no longer exists at 250 Hz (fragmentation fell
    1.31 → 1.27 positives per IED). So the discharge is captured identically at both rates.
    - This is the §5 caveat made concrete: *"at 2.86 events per IED the old pipeline effectively got ~3
      attempts per IED"*. S26's hit was one of those extra attempts.
    - **The real weakness it exposes is representative selection**, not the sample rate: max peak-to-peak
      lands on the after-going slow wave, so this event's timestamp is ~200 ms late at *both* rates and
      whether it scores a hit is incidental. Deferred as item **I** below — the user's call, taken
      2026-08-17: address later, don't hold up preprocessing.
- **Phase 2 — bandpass 0.5–45 Hz. DONE 2026-08-17, ADOPTED.** `Config.bandpass = (0.5, 45.0)`, one
  `raw.filter(0.5, 45)` in `load_recording_path`, placed **before** the resample (mne's recommended order —
  the anti-aliasing step then has nothing left to remove). Zero-phase, so spike timing and shape survive.
  **No notch**: 45 Hz already excludes mains at 50 *and* 60 Hz, so a notch would delete something that is no
  longer there. It must come back if the upper edge ever goes above 50 (e.g. 0.5–70 for IED review).
  - 0.5 Hz over 1 Hz was the supervisor's call, to preserve the after-going slow wave (0.5–3 Hz), which is
    part of the discharge and sits inside the 2 s window L3 sees. Cost: a lower high-pass settles more
    slowly, so more of each 11–14 s clip has unreliable edges — partly mitigated because events within 1 s
    of an edge are already rejected (the 2 s window cannot fit).
  - **The filter does what it claims** (PSD, 250 Hz both arms so only the filter differs): passband
    0.5–45 Hz preserved to **99.7–100.1%**, 45–60 Hz transition 37.4%, 60–124 Hz **0.0%**. Asserted in
    `detect_data._selfcheck` (stopband < 2% retained, passband > 70% retained) so it cannot silently break.
  - **A PREDICTION OF THIS LOG WAS WRONG, and it is the reassuring kind.** It was predicted that this would
    be "the riskiest phase for recall", because L1's statistic is a second derivative and therefore
    amplifies exactly the high frequencies a 45 Hz low-pass removes. **L1 IED recall stayed at 1.0000.**
    MAD normalisation absorbs it — signal and noise shrink together, so the ratio is preserved. Record the
    prediction and its refutation; it is evidence that L1's channel-adaptive normalisation is doing real
    work for transfer, not just for convenience.
  - **L2 recall unchanged at 0.9796** — still only S26, so the bandpass introduced no new loss.
  - The candidate pool got slightly healthier: 336 candidates/recording (from 344), 4,080 windows (from
    4,133), but **65 positives (from 62)** — so the positive fraction rose 1.50% → 1.59%.

  **Grouped CV (250 Hz + bandpass, polarity ON, elastic, k=24) vs the 500 Hz polarity-on baseline:**

  | | 500 Hz baseline | 250 Hz + bandpass |
  |---|---|---|
  | ROC-AUC | 0.840 | 0.843 [0.786, 0.894] |
  | **PR-AUC** (chance 0.0159) | 0.146 | **0.169** |
  | per fold | — | 0.808 · 0.911 · 0.847 · 0.946 · 0.762 |
  | sens @1 / 5 / 10 / 25 / 50 / 100 FP/min | 0.10 / 0.47 / 0.61 / 0.76 / 0.88 / 0.90 | **0.14** / **0.49** / 0.57 / **0.82** / 0.84 / **0.92** |
  | hits / misses / FPs @ ≤10 FP/min | 30 / 19 / 176 | 28 / 21 / 182 |
  | localisation, median | 16 ms (49 IEDs) | 16 ms (48 IEDs) |

  - **The FROC's low-FP end — the stated primary criterion — improved at both @1 (+2 IEDs) and @5 (+1).**
    PR-AUC is up 16% relative. The dips are mid-curve, at @10 and @50.
  - **But the gain is NOT established.** Paired-by-recording bootstrap (the same 90 recordings resampled,
    each arm scoring its *own* candidate pool for them): **ΔPR-AUC +0.0268, 95% CI [−0.0362, +0.0962],
    P(>0)=0.80.** The interval is wide because the two arms have different pools, which adds variance a
    matched comparison would not. **This is a "no harm, plausibly a small help" result — which is exactly
    what preprocessing was justified on. Report it as a transfer measure, not as a performance gain.**
  - **Of the 21 misses, 20 are L3 ranking failures and 1 is S26**, the deferred L2 timing loss (item I). So
    the reachable ceiling is 48 of 49, and the honest like-for-like on hits is 28/48 vs 30/49 — the apparent
    2-hit drop is partly bookkeeping, not L3.
  - **Attribution caveat:** Phase 1 already low-passed at ~125 Hz as part of resampling, so this is the
    *incremental* effect of narrowing 125 → 45 Hz. Since only 0.0105% of the original power sat above
    125 Hz, essentially all of the filtering effect belongs to this phase — but the two are not cleanly
    separable in principle, because resampling cannot be done without an anti-aliasing low-pass.
- **Phase 3 — bad channels + average reference. DONE 2026-08-17, ADOPTED.**
  `reference='average'`, plus `bad_soft_factor=3.0`, `bad_hard_factor=10.0`, `bad_flat_uv=0.5`,
  `max_interpolate=2`. Average referencing subtracts the mean over channels, so **one broken electrode
  contaminates all 19** — hence the bad-channel handling had to come with it.

  **Two thresholds, matched to the cost of being wrong** (the user's design, and it is better than the
  single threshold originally planned):

  | | trigger | action |
  |---|---|---|
  | **soft** | MAD > 3 × median across channels | dropped from the average-reference computation, **kept in the analysis** |
  | **hard** | MAD > 10 × median, **or** ptp < 0.5 µV | removed from the analysis **and interpolated** |

  - Excluding a channel from the average is nearly free if wrong (18 channels instead of 19), so that
    threshold can be sensitive. Interpolating **destroys real data** and replaces it with a synthetic
    estimate, so that one must be conservative. One threshold cannot serve both.
  - **Over the cap → graceful degradation, not refusal.** More than 2 hard channels means interpolating
    would be fiction, so none are interpolated, all are demoted to soft, and the recording is flagged. It
    returns a caveated number rather than nothing — each pilot run costs the supervisor hours of compute, so
    maximum information per run matters. (An earlier draft justified this by "we only get one round trip";
    that was wrong — the user can send scripts to Neuronostics whenever. The per-run cost argument stands,
    and *pre-registration* matters MORE with free access, not less, since the temptation to tune rises.)
  - **The single enforcement point: interpolated channels generate no L1 candidates.** No candidates means
    no event membership, so a reconstructed channel can never satisfy `min_channels` (L2's only hard
    filter), never inflate a channel count, and never enter an L4 spatial feature. That is two lines in
    `Stage1.detect` instead of threading a bad-list through L2, and it is the literal reading of the
    supervisor's *"once a channel is identified as bad you probably do want to remove it completely from
    the analysis"*. Consequence worth stating plainly: once an interpolated channel is out of the average,
    out of detection and out of spatial features, interpolation only keeps the array 19 wide and gives the
    amplitude normalisation a sane value. That is standard practice and harmless, but it is not carrying
    weight.

  **Calibration — and the diagnostic prevented a bad design.** Run on the 90-train only (choosing a
  threshold from data is a fitting step, so the 10 test recordings were deliberately excluded):

  - **High side:** ratio of channel MAD to the median across channels has p95 1.50, p99 **1.88**, max
    **8.28**; only 1 of 1710 channel-recordings exceeds 3×, and **nothing trips 10×**. So 10× sits far above
    clean variation and will only fire on catastrophic failures — which is what real electrode failures
    are. Empirical justification for the factor, from the cleanest data available.
  - **Low side — the MAD floor originally proposed was WRONG and was dropped.** A relative floor would have
    flagged 9 channel-recordings; inspection showed **none of them are dead**: e.g. S19/Cz has MAD 0.012 µV
    but peak-to-peak 2.32 µV across 3,184 distinct values. And the median MAD by electrode runs
    **Pz 0.36, Cz 0.40, C3/C4/P3/P4 0.62 … F7 1.22, FP1 1.29** — a monotonic rise with distance from the
    centro-parietal midline, i.e. **Kural's recording reference sits at or near Cz/Pz**, so midline channels
    legitimately have tiny MAD. A MAD-based floor would have interpolated away good reference-adjacent
    channels. **Dead is tested on peak-to-peak, not MAD**: low MAD means a quiet baseline, not a dead
    electrode.
  - **Known limitation, documented rather than engineered around:** the global 10× rule is unevenly
    sensitive across electrodes (Pz must reach 28× its own norm to trip, FP1 only 7.8×). Per-electrode
    norms were rejected because that gradient is a property of *this* reference montage and the check runs
    *before* re-referencing, so it would not transfer. Neighbour-correlation (as PREP uses) is the
    principled upgrade if the pilot shows the rule missing real failures.
  - Also unhandled: a channel that goes bad **intermittently** (an electrode popping at minute 12) may not
    move a whole-recording MAD enough to trip. The realistic failure mode is a miss, not a false alarm.

  **On Kural the machinery is essentially inert — 1 soft flag (S49/Pz), 0 hard, 0 over-cap.** So it ships
  **untested on this data**, and the CV below is driven almost entirely by the average re-referencing. Say
  so in the writeup rather than implying it was validated.

  | 90-train | Phase 2 (recorded ref) | Phase 3 (average ref) |
  |---|---|---|
  | **L1 IED recall** | 1.0000 | **1.0000** |
  | L2 consolidation recall | 0.9796 | 0.9796 |
  | events / min | 214.4 | 218.4 |
  | positive windows per IED | 1.33 | 1.22 |
  | channels per event | 5.15 | **4.83** |
  | window pool | 4,080 / 65 pos | 4,157 / 60 pos |

  - **Channels per event fell 5.15 → 4.83** — the reference-contamination effect this log describes
    (*"contamination spreads the spike in space, not time"*) showing up as measured spatial narrowing once
    the shared reference is removed. Mildly encouraging for L4, whose channel counts were inflated.

  **THE DECISION OF THIS PHASE — polarity survives, and the column stays.**

  | | recorded reference | average reference |
  |---|---|---|
  | downward, all windows | 51% | **51%** |
  | downward, positive windows | 75% | **75%** |

  - Identical. The risk flagged when polarity was adopted — that sign is a montage property and
    re-referencing could destroy the cue — **did not materialise, and is now measured rather than assumed.**
  - Mechanism, clear in hindsight: polarity is read on the *representative* channel, the event's largest
    peak-to-peak. Average referencing subtracts the mean over 19 channels, and for a discharge dominant on
    a handful of them that mean is a small fraction of the dominant channel's amplitude, so the sign of the
    biggest deflection is robust to it.
  - This is also evidence the feature should survive a montage change on the Neuronostics data — which was
    the transfer worry recorded in §6.

  **Grouped CV (250 Hz + bandpass + average reference + bad-channel handling, polarity ON, elastic, k=24):**

  | | Phase 2 | Phase 3 |
  |---|---|---|
  | ROC-AUC | 0.843 | **0.865** [0.797, 0.929] |
  | PR-AUC | 0.169 | 0.170 |
  | **PR-AUC relative to chance** | 10.6× | **11.8×** |
  | per fold | 0.808 · 0.911 · 0.847 · 0.946 · 0.762 | 0.840 · 0.865 · 0.817 · 0.919 · 0.894 |
  | sens @1 / 5 / 10 / 25 / 50 / 100 | 0.14 / 0.49 / 0.57 / 0.82 / 0.84 / 0.92 | 0.14 / **0.51** / **0.67** / 0.82 / **0.92** / **0.94** |
  | **hits / misses / FPs @ ≤10 FP/min** | 28 / 21 / 182 | **33 / 16 / 181** |
  | localisation, median | 16 ms | **12 ms** |

  - **+5 IEDs found at the same false-positive cost** (181 vs 182), localisation 16 → 12 ms, and the
    per-fold spread tightened (0.817–0.919 vs 0.808–0.946).
  - **But the paired bootstrap does NOT establish it.** Paired by recording, each arm scoring its own pool:
    **ΔPR-AUC +0.0013, 95% CI [−0.0831, +0.0891], P(>0)=0.50** — dead flat; ΔROC-AUC +0.0229,
    95% CI [−0.0465, +0.0895], P(>0)=0.75. And **+5 hits of 49 is ~1.5 SE** (binomial SE at sensitivity
    0.67 is 3.3 IEDs). **Encouraging, not established** — report it that way.
  - **Read PR-AUC relative to chance here, because the pool changed** (60 positives in 4,157 windows vs 65
    in 4,080). Flat in absolute terms is 10.6× → 11.8× against chance. Quoting the raw PR-AUC alone would
    understate it; quoting the ratio alone would overstate the certainty. Give both.
  - Gains are concentrated mid-FROC and at the operating point rather than at @1, which is where PR-AUC and
    the far tail live — consistent with PR-AUC moving less than the FROC did.

- **Phase 4 — consolidate. DONE 2026-08-18.** Restart & Run All: 20 code cells, all executed in order,
  no errors, sweeps off, `RUN_GROUPED_CV=True`. Results in §7 "CURRENT RESULTS".
  - **`detect_model.joblib` re-saved and now stores the full config** (250 Hz, bandpass, average reference,
    bad-channel thresholds, polarity, elastic, k=24). The pickle hazard is resolved and the model is safe
    to use and share again.
  - The notebook's CV reproduced the Phase 3 scratch run **exactly** (0.865 / 0.170 / 33-16-181), so the
    notebook and the standalone scripts agree.
  - **Error analysis Phase 1** (operating point score ≥ 0.800): Q1 certainty still does not explain the
    misses — clear 13/20 found (65%), unclear 20/29 (69%). Q2 the misses are still expensive — median
    **+567 FPs** to catch one more, **0 of 16 cost ≤50**, 8 cost >500. Q3 181 FPs spread over **67 of 90**
    recordings, worst 9 holding 35% (was 45/90 and 53% before normalisation — the spreading continues).
    Q4 the epileptic/non-epileptic FP asymmetry has collapsed further: **10.7 vs 8.1 FP/min = 1.3×**
    (5.9× → 1.6× → 1.3× across the project).
  - **Polarity composition at the operating point:** hits **82%** downward, misses 73%, top FPs **60%**.
    The FP figure has risen from 40% because the model now culls upward mimics — the "cue spent once used"
    effect predicted in §6 — while the hit-vs-FP gap persists.
  - **Deferred item G is strengthened.** Re-measured window centring on the current pipeline:
    `'sharpness'` gives median |offset| **16 ms with 0% beyond 100 ms**, `'amplitude'` **28 ms with 22%
    beyond 100 ms**. That gap is wider than when the centring decision was made, and the decision's stated
    mechanism is already in doubt (§6). Worth re-running properly.
  - **Notebook restructured so it stops going stale.** All results-interpretation prose was removed and the
    method/rationale kept: the notebook now documents *what is being measured and why*, and report.md holds
    *what came out*. Three pure-results markdown cells deleted (their content is in §6/§7 already) and five
    mixed cells stripped of their numbers, leaving 29 cells with **no volatile figures in prose**.
  - **A verdict bug in the amplitude-confound cell was found and fixed** — see below; it is the most
    important single finding of this run.
  - Figures: 11 regenerated by the run. The 4 stale ones belong to the two reference-only experimental
    notebooks. `detection_registration_misses.png` was regenerated separately against the current pipeline,
    since report.md cites it as evidence and it was still showing 500 Hz recorded-reference curves.

### The amplitude confound is BROKEN — and the notebook said the opposite
The confound cell printed **"CONFOUND IS LIVE"**. Its condition was `auc_bg > 0.65 or abs(rho) > 0.5`, and
only the first disjunct was true. The two numbers:

| | before (500 Hz, pre-normalisation) | now |
|---|---|---|
| background amplitude alone, AUC | 0.690 | **0.759** |
| Spearman corr(amplitude, `mean top-5`) | **+0.708** | **+0.060** |

- **ρ is what "the statistic is measuring scale" means, and it has collapsed 0.708 → 0.060.** Amplitude
  normalisation did exactly what it was adopted to do. The cell announced a confound at the precise moment
  it had been removed. **Condition fixed to key on ρ.** Worth recording as a lesson: an automated verdict
  keyed on the wrong quantity is worse than no verdict, because it is believed.
- **But it surfaced a genuine new finding: background amplitude ALONE separates epileptic from
  non-epileptic recordings at AUC 0.759 — better than the pre-registered pilot statistic at 0.666 — and
  the two are now uncorrelated.** A one-line measure with nothing to do with IEDs beats the detector at
  the pilot's own question, using independent information.
  - It must be reported as a **pilot baseline**, exactly as the prominence baseline is for detection (§7).
  - Read it cautiously: epileptic recordings being louder (6.26 vs 4.46 µV) could easily be a cohort or
    acquisition artefact of Kural that would not transfer. That is testable in the pilot — export
    background amplitude per recording and check whether it separates their groups too.
- **Pilot pre-test on the current pipeline:** `mean top-5` **0.666 [0.543, 0.779]**, marker-removed control
  **0.582**. So roughly half the recording-level separation is still not the annotated discharges
  (was ~20% surviving at 0.805 → ~50% now). The honest expectation to give the supervisor is **0.666**.
- **The reference reversal is deliberate and must be recorded as such.** §2 and §5 previously said "do NOT
  re-reference L1/L2/L3; re-referencing enters at L4 only", on the grounds that contamination spreads a
  spike in space rather than time. That is overruled: average reference is reproducible at any lab while a
  recorded reference is not, so it is the stronger *transfer* choice — the same reasoning that justified
  amplitude normalisation. The old rationale was not wrong about localisation; it was answering a narrower
  question.

### Coming back to later (deliberately deferred, in rough priority order)
- **A. Window length (`classifier_halfwin_s`, 1.0 → 0.5, maybe 0.25).** The spike is ~2.5% of a 2 s window,
  so both the registration and the FPCA components are dominated by background (§6). Deferred because the
  bandpass changes how much background there is. Do not go below the slow wave that the 0.5 Hz high-pass
  was chosen to keep — try ±500 ms first. Note it slightly changes the event pool, since a shorter window
  is rejected at fewer recording edges, so counts move off 4,330.
- **B. Shape versus size — the main remaining lever on sensitivity.** §7 Phase 2b: the misses are the
  low-prominence discharges and L3 cannot promote "small but well-formed". Candidate: normalise each window
  by its own peak-to-peak so only shape remains. Measure, don't assume — absolute prominence is real
  evidence of a discharge.
- **C. L4 / spatial features — the main remaining lever on false positives.** Go/no-go is the Phase 3
  error-analysis test: if no spatial feature reaches univariate AUC ~0.65 separating hits from FPs, report
  L4 as tested-and-rejected. Now also has the bad-channel flag available.
- **D. Neuronostics pilot script** (§9). Now with **two extra transfer diagnostics** to export: the
  downward fraction per recording (ours 51% of events, ~75% of the top) and the bad-channel count.
- **E. Random-crop evaluation**, so the centre baseline is a fair bar.
- **F. Report the prominence baseline** as a second null model (§7) — stronger than the centre baseline at
  ≤10 FP/min and it appears on the FROC at every budget.
- **G. Re-measure the window-centring comparison** — its stated mechanism is now doubtful (§6) and the
  original measurement predates both the `n_components` fix and normalisation.
- **H. Optional, low value:** extend k to 64; one confirmatory greedy-vs-components run at the tuned k.
- **I. Decouple an event's TIME from its representative CHANNEL** (raised by the S26 diagnosis, Phase 1).
  Both currently come from the max-peak-to-peak member, but they answer different questions: the time feeds
  hit/miss and localisation, while the channel selects the window L3 classifies. On a slow-wave-dominated
  discharge peak-to-peak lands on the after-going wave, putting the timestamp ~200 ms late (S26, at both
  500 and 250 Hz) — so whether such an IED counts as found depends on incidental fragmentation.
  - Proposal: keep the representative **channel** by peak-to-peak (unchanged, and consistent with how the
    classification-phase windows were cut), but take the event **time** from its *sharpest* member (the L1
    statistic). report.md §6 already measured sharpness aligning far better to the marker — median \|offset\|
    20 ms vs 54 ms, 1% vs 22% beyond 100 ms.
  - Not a preprocessing change and not free: it would move hits/misses across all 49 IEDs, so it needs its
    own grouped CV and its own pre-specified decision rule. Do it deliberately, after the preprocessing
    phases, not folded into one of them.
  - Would likely recover S26 and may recover other near-tolerance cases; could also lose some. Measure.

**Parked:** class-informed / two-template registration features — the supervisor doubts they would
generalise to other EEG recordings. Weighted or balanced FPCA is parked with them for now, since the k
sweep achieved the same goal (recovering IED-discriminative directions from a negative-dominated pool)
with no class-informed fitting at all. **Warp magnitude as a feature is parked here too** — §6 downgraded
its evidence, and it is mutually exclusive with fixing the template.

### Open questions for the supervisor
- The exact target sample rate (250? 256?). Does not block — it is a config value.
- Whether 0.5–45 Hz is the final band, or the upper edge should go to ~70 Hz (which would reinstate the
  mains notch).
- Expected IEDs per confirmatory recording, for the pilot's pre-registered `mean top-N` (§9).

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
### Pilot pre-test — RUN, and the premise holds
Recording-level separation on the 90-train from out-of-fold CV scores, 49 epileptic vs 41 non-epileptic:

| statistic | AUC | 95% CI | | statistic | AUC | 95% CI |
|---|---|---|---|---|---|---|
| count ≥ 0.85 | 0.825 | [0.744, 0.904] | | mean top-10 | 0.795 | [0.701, 0.882] |
| q99 | 0.825 | [0.731, 0.909] | | mean score | 0.754 | [0.650, 0.853] |
| max | 0.824 | [0.728, 0.909] | | q90 | 0.752 | [0.650, 0.847] |
| mean top-3 | 0.810 | [0.718, 0.896] | | q95 | 0.746 | [0.641, 0.843] |
| mean top-5 | 0.805 | [0.714, 0.886] | | **n events** | **0.511** | [0.389, 0.632] |

- **The pilot's premise holds: AUC ≈ 0.82 at recording level**, CI comfortably clear of 0.5. Far stronger
  than the event-level FROC would suggest, and for the expected reason — pooling thousands of events into
  one number is a much easier question than pinpointing each discharge.
- **`n events` sits at chance (0.511) — an important control.** Epileptic and non-epileptic recordings
  produce the *same number* of L1/L2 candidates; the separation is entirely L3's doing. So the front end
  is not "seeing more spikes" in epileptic EEG — L3 is rating more of them IED-like. This also means the
  pilot must report **scores**, never raw event counts.
  - It sharpens §7 Q4: the 6× false-positive rate in epileptic recordings is not a generically spikier
    background, it is specifically more *IED-like* morphology. Consistent with either "real unmarked
    discharges" or "epileptic background is genuinely more IED-like", and does not separate them.
- **Pre-registered primary statistic: `mean top-N`, NOT the argmax.** The top six statistics are
  indistinguishable (0.795–0.825, CIs almost entirely overlapping), so the choice is made on transfer, as
  planned in advance:
  - `count ≥ thr` is doubly fragile — it scales with duration *and* depends on a threshold calibrated to
    Kural's score distribution;
  - `q99` means the top ~0.5 events on a 12.7 s recording but the top ~50 on a 17-minute one;
  - `max` saturates near 1.0 once a recording offers thousands of draws;
  - `mean top-N` always reads the N most IED-like events regardless of recording length, and maps directly
    onto "how many discharges do we expect".
  - **N must be fixed before the run.** On Kural (1 IED/recording) the best N is 3, i.e. ≈3× the expected
    IED count. **Ask the supervisor for the expected IEDs per confirmatory recording**; absent an answer,
    use N=10. The events CSV carries every score, so other N values remain computable afterwards as
    secondary/exploratory.
- **Selection caveat:** ten statistics were compared on the same 90 recordings, so the top AUC is
  optimistic by perhaps 0.02–0.04. Quote the pre-registered statistic's value with its interval, and say it
  was chosen here.

### AMPLITUDE CONFOUND — the recording-level signal is largely signal scale
Two controls, run in order, and the second explains the first.

- **Control 1 — remove the marked IED** from each epileptic recording and recompute: `mean top-5` falls
  only 0.805 → 0.747. Above chance, that is 0.305 → 0.247, so **~80% of the recording-level separation is
  not the annotated discharge.**
- **Control 2 — background amplitude alone** (per-channel MAD of the raw µV signal, median over the 19
  channels; MAD is robust so a single spike does not move it):
  - epileptic median **9.00 µV** vs non-epileptic **7.00 µV**;
  - amplitude *alone* separates the groups at **AUC 0.690, 95% CI [0.583, 0.789]**;
  - **Spearman corr(amplitude, `mean top-5`) = +0.708**;
  - and in the scatter of amplitude vs statistic, **the two groups do not separate vertically at any
    amplitude** — at a fixed background scale, epileptic and non-epileptic recordings score the same.
- **Conclusion: the recording-level separation is substantially a scale effect, not a morphology effect.**

**Mechanism (specific, and it was predicted before measuring):** L1 divides its sharpness statistic by each
channel's MAD, so it is **scale-blind** — which is why `n events` sits at chance (0.511). L3 has no such
normalisation: its windows are raw µV and the smoothing, registration and FPCA all operate on µV values,
while `StandardScaler` standardises component scores across the *whole training set*, not per recording. A
systematically larger recording therefore produces systematically larger FPCA scores and shifts along the
discriminant without anything in it being more IED-like.

**Three consequences were predicted in advance, then tested by re-running with normalisation.
Two confirmed, one REFUTED — and the refutation is the more useful result.**

| predicted consequence | before | after | verdict |
|---|---|---|---|
| §7 Q3 — FPs concentrated in the loud recordings | worst 9 hold 53%, worst single 22, 45/90 affected | worst 9 hold **39%**, worst single **11**, **58/90** affected | **confirmed** |
| §7 Q4 — 6× FP rate in epileptic recordings because they are louder | 14.8 vs 2.5 /min (**5.9×**) | 11.9 vs 7.3 /min (**1.6×**) | **confirmed** |
| §7 Q2 — misses expensive because loud recordings' mimics outrank quiet recordings' IEDs | median **+776** FPs, 14/20 >500 | median **+771** FPs, 14/21 >500 | **REFUTED** |

- **On Q3/Q4:** the *total* FP count cannot fall — the operating point fixes FP/min ≤10 by construction
  (176 → 187). What changed is **where** they come from: they redistributed from loud recordings to quiet
  ones. That is what "one global threshold now means the same thing in every recording" looks like, and it
  is the property a fixed threshold needs in order to transfer.
- **On Q2, the refutation matters more than the confirmations.** The misses cost just as much once the
  cross-recording unfairness was removed, so **they were never a calibration artefact**. Those ~21 IEDs are
  outranked by hundreds of events on a level playing field: L3 genuinely rates them less IED-like than the
  mimics. That is a **morphology/feature failure** — no threshold, calibration or normalisation change can
  reach them, only better features can.
  - Redirects effort: **error-analysis Phase 2 (plot the ~21 missed IEDs) is now the highest-value next
    step**, ahead of L4. What does an IED that L3 confidently ranks below hundreds of mimics look like?
- **A residual 1.6× FP asymmetry survives and can no longer be scale.** Either epileptic background is
  genuinely more IED-like, or there are unmarked discharges — the Q4 question, now cleanly posed.
- **Q1 unchanged:** clear 11/20 (55%), unclear 17/29 (59%). Certainty does not explain the misses, before
  or after normalisation.

**Consequences:**
- **Do not ship the pilot statistic as-is.** It would measure recording scale on the Neuronostics data,
  which transfers nowhere — and if his confirmatory/non-confirmatory groups happen to differ in acquisition
  scale, it would manufacture a spurious positive.
- The FROC itself is not invalidated (a hit is still a hit), but L3 carries a **nuisance sensitivity to
  recording scale** that will hurt on any dataset with different amplitude characteristics.

**Fix to test next — per-recording amplitude normalisation.** Divide each window by its own recording's
background MAD before the FDA pipeline, so L3 judges *relative prominence* — how far a transient stands out
from its local background, which is what a reader actually does — rather than absolute µV.
- It **self-calibrates on unseen data**, which is a prerequisite for external transfer, not just a fix here.
- **Prediction to test:** it should improve the CV FROC most at the **low-FP end**, by making the global
  threshold fair across recordings — quiet recordings' IEDs rise, loud recordings' mimics fall.
- **Risk:** absolute amplitude may carry genuine signal (some clinical criteria for IEDs involve absolute
  µV), so normalising could remove real information. Measure on the CV FROC, do not adopt on principle —
  same discipline as centring and mining.

### Normalisation — MEASURED, and adopted on transfer grounds, not performance
`cfg.normalise_amplitude=True` divides each window by `DetectionPipeline.background(X)` (median over
channels of per-channel MAD). Grouped CV, 90-train, everything else unchanged:

| | baseline | normalised |
|---|---|---|
| CV ROC-AUC | 0.828 [0.763, 0.889] | 0.831 [0.759, 0.894] |
| CV PR-AUC | 0.117 | **0.133** |
| sens @1 / 5 / 10 / 25 / 50 / 100 FP/min | 0.08 / 0.41 / 0.59 / 0.71 / 0.84 / 0.92 | 0.10 / **0.47** / 0.57 / 0.71 / 0.82 / 0.90 |
| hits / misses / FPs @ ≤10 FP/min | 29 / 20 / 176 | 28 / 21 / 187 |
| **recording-level `mean top-5`** | 0.805 | **0.687** |
| same, marked IED removed | 0.747 | **0.603** |

- **Detection effect is within noise.** The predicted shape appeared — gains at the low-FP end (+0.06 at
  ≤5 FP/min), small losses above — but +0.06 is 3 IEDs of 49 against a binomial SE of ~0.07. PR-AUC +14%
  relative is the most encouraging single number and is also not decisive.
  - **The user's scepticism was correct:** a discharge stays identifiable regardless of background
    loudness, so removing absolute scale neither helps nor hurts detection much.
- **The pilot number was the confounded one.** Recording-level separation falls 0.805 → **0.687**, and the
  marker-removed control 0.747 → 0.603. So roughly half of a much smaller signal is now the discharges,
  instead of a fifth of a large one. **0.687 is the honest expectation to give the supervisor.**
- **DECISION: adopt (`normalise_amplitude=True`), justified by generalisation, not by the CV.**
  A model whose scores depend on absolute µV cannot carry a fixed threshold to another lab's recordings:
  on unseen data every score shifts wholesale, the pre-registered threshold becomes meaningless, and the
  pilot statistic partly measures the acquisition setup. Normalised, the model self-calibrates against
  each recording's own background. **Say this in the writeup as a transfer property, not a result.**
- Also removes a confound that would otherwise have to be disclosed and caveated throughout the results.
- Invariant checked in `detection.py.__main__`: scaling a recording by 10× leaves the normalised windows
  bit-identical, and normalisation does change them versus baseline.

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
  - **Add a second transfer diagnostic: the downward fraction of events** (per recording, and over the
    top-scoring events). Ours are 51% and ~75% (§6). A sharp departure means their reference convention
    differs from Kural's, which matters for any polarity-dependent feature and is worth knowing even if the
    polarity column is not adopted — it is one extra column in `pilot_recordings.csv` and one sign per row
    in `pilot_events.csv`, computed from data the script already holds.
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
