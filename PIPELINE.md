# The IED detection pipeline

What the detector does, stage by stage, and why each choice is the one in place. This describes the
pipeline as built — `report.md` holds the experimental record, `CLAUDE.md` the working handover.

**Task.** Given a scalp EEG recording, find *where in time* interictal epileptiform discharges (IEDs)
occur, and don't fire on ordinary sharp transients.

```
EDF ─► preprocessing ─► L1 candidates ─► L2 events ─► L3 classifier ─► L4 decision ─► scored events
       detect_data      detect_stage1    detection     detection       detection
```

Every tunable lives in one place: `Config` in `src/detect_config.py`. Nothing downstream hard-codes a
threshold, a window or a sample rate.

---

## 1. Preprocessing — `detect_data.load_recording_path`

**The single entry point for anything applied to the signal before L1.** Every path into the cascade —
training, evaluation, `predict.py`, `pilot.py` — comes through this one function, driven entirely by
`cfg`. Since `cfg` is frozen into the saved model, an external run cannot preprocess differently from
training.

| step | setting | why |
|---|---|---|
| channel matching | 19 standard 10-20 electrodes | Present in all recordings. Names are normalised (`EEG Fp1-REF`, `Fp1-A1`, `E FP1-Ref` all map to `FP1`) and the old `T3/T4/T5/T6` nomenclature is aliased to `T7/T8/P7/P8`, so clinical files from any site load without editing. Matching is **exact after normalisation**, never by substring, so an extended array's `FC3` cannot masquerade as `C3`. |
| bandpass | 0.5–45 Hz, zero-phase | 0.5 Hz preserves the after-going slow wave, which is part of the discharge. 45 Hz already excludes 50 and 60 Hz mains, so no notch is needed. Zero-phase so spike timing and shape are preserved. Applied **before** resampling, so the anti-aliasing step has nothing left to remove. |
| resample | 250 Hz | Sufficient for a 20–70 ms spike, and halves every downstream cost. All windows are specified in **milliseconds** and converted with `cfg.samp()`, so the cascade means the same thing at any input rate. |
| bad channels | MAD > 3× median → soft; MAD > 10× or peak-to-peak < 0.5 µV → hard | Two thresholds because the costs are asymmetric. Excluding a channel from the average reference is nearly free if wrong; interpolating destroys real data, so that threshold is far above clean variation. "Dead" is tested on peak-to-peak, never on MAD — a quiet channel near the reference has tiny MAD but perfectly good signal. |
| reference | average, over the good channels only | A recorded reference differs between labs; an average reference does not. This is the choice that transfers. |
| interpolation | spherical spline, at most 2 channels | Beyond two of nineteen, reconstruction is fiction. Over the cap the recording is still processed — all bad channels are demoted to soft and the recording is flagged. |

**Interpolated channels generate no L1 candidates.** That single rule is the enforcement point for
"reconstructed data is not evidence": with no candidates, such a channel can never join an event, satisfy
the channel-count floor, or enter a spatial feature.

## 2. L1 — candidate generation (`detect_stage1`, class `Stage1`)

Per channel: Savitzky–Golay second derivative (42 ms window, order 3), absolute-valued, divided by that
channel's own MAD. Peaks above **6 MAD** become candidates, with 150 ms non-maximum suppression so one
biphasic complex yields one candidate per channel.

- **Curvature is what makes a spike look sharp**, so the second derivative is the natural statistic — large
  at a sharp transient, near zero on smooth background.
- **Per-channel MAD normalisation makes L1 scale-blind**: it asks "sharp relative to this channel's usual
  sharpness", so a loud recording does not produce more candidates than a quiet one.
- **The threshold is deliberately permissive.** L1 owes *recall* — it achieves 100% IED recall — and
  selectivity belongs to the layers below. Any hard filter here would be a permanent recall cap.

## 3. L2 — event consolidation (`detection.Consolidator`)

Turns per-channel candidates into multi-channel **events**.

**The test** — for a pair of candidates within 50 ms of each other, the cross-correlation of their
SG-smoothed ±150 ms windows must reach **0.7**. It is:

- **normalised**, so a diminished copy on a distant channel still matches;
- **lag-searched over ±30 ms**, so a propagating discharge still matches;
- **absolute-valued**, so the opposite side of a dipole — which is polarity-reversed — is not dropped.

**The assembly rule** — `grouping='components'`: single linkage, so events are connected components of the
"same spike" graph. A discharge's scalp field is a *chain*, not a star: across a dipole the far channels
resemble their neighbours but not the sharpest channel. Single linkage keeps such a discharge as one
event (1.31 events per IED, capturing 15 of 17 involved channels).

**One hard filter, and only one:** an event needs **≥2 distinct channels**. Single-channel transients are
artefact. The floor is 2 rather than 3 or 4 because a focal IED can legitimately occupy exactly two
adjacent electrodes. Everything else ambiguous is passed down as a soft feature.

The **representative channel** is the member with the largest peak-to-peak; its time becomes the event's
time.

## 4. L3 — the FDA classifier (`detection.Classifier`)

Scores the representative channel's **2 s window** as IED or not.

1. **Window cutting** — centred on the largest deviation of the smoothed signal within ±150 ms
   (`centre_on='amplitude'`), matching the annotation convention that the marker sits at the spike peak.
   Windows falling off a recording edge are dropped.
2. **Amplitude normalisation** — each window is divided by its recording's background amplitude (median
   over channels of per-channel MAD). L3 therefore judges **relative prominence** — how far a transient
   stands out from its own background, which is what a reader does — rather than absolute microvolts. This
   is what lets one fixed threshold mean the same thing on an unseen recording from another site.
3. **B-spline smoothing** — 70 basis functions, order 4, resampled onto a 100-point grid.
4. **Fisher–Rao elastic registration** — each curve is warped onto a template (a Karcher mean learned from
   training data and frozen into the model). Absorbs variable spacing between the spike and its
   after-going slow wave, which a rigid shift cannot.
5. **FPCA — 24 components.** FPCA orders components by *variance*, and the candidate pool is ~98.5%
   negatives, so the IED-discriminative direction is not among the leading few. Twenty-four components is
   what it takes to retain it.
6. **Polarity** — the sign of the central ±20 ms deflection is appended as a 25th feature. L1 and L2 are
   polarity-blind by design, so the sign is handed to the classifier explicitly.
7. **Standardise → logistic regression** (`C=1`, balanced class weights) → IED probability.

## 5. L4 — the event decision (`DetectionPipeline._fit_l4`)

A logistic regression over **(L3 score, number of channels, dipole)**, producing the final score. The raw
L3 value is retained as `l3_score`.

L3 sees one channel's waveform, so the shape of the discharge's field across the scalp is information the
cascade would otherwise discard entirely. Two aspects of it are used:

- **`n_channels`** — how widely the discharge spreads. Genuinely additional rather than a restatement of
  size: it correlates only +0.33 with the event's prominence, and where prominence is useless against the
  hardest false positives, channel count still separates them.
- **`dipole`** — the distance between the positive- and negative-member centroids, over the field's
  extent. A real IED is generated by a patch of synchronously firing cortex and therefore produces a
  **dipolar** field: two spatially coherent poles. Artefact does not. L2 groups on *absolute* correlation
  so that a dipole's polarity-reversed far side is not dropped, which means member sign is computed and
  then discarded — this feature recovers it. Because an average reference forces the field to sum to
  zero, both signs are nearly always present, so what is measured is whether they are spatially
  **organised**, not whether they exist.

The two are complementary: `dipole` carries no information on its own and only separates events once the
score and spread are already accounted for.

**The L3 scores L4 trains on are cross-validated** over the already-fitted representation. In-sample
scores are inflated for the positives, which would make L4 lean on the score and under-use the field.

## 6. Output and operating point

`predict()` returns one dict per event: `time`, `channel`, `n_channels`, `members`, `score`, `l3_score`,
`polarity`, `prominence`.

**The threshold is derived, not chosen.** `cfg.fp_budget` fixes the false-positive rate the detector is
quoted at, and the threshold is then whatever the cross-validated FROC gives at that budget —
`threshold_at()` takes the highest score still inside it. Nothing is tuned to make a number read well.

The budget is set from an **external** criterion, and there are two defensible ones:

- **clinical review burden** — at 5 FP/min a 20-minute recording yields ~100 events to review, at 10 it
  yields ~200;
- **comparability with published detectors**, which report 0.7–0.9 sensitivity at 0.5–5 FP/min.

`cfg.fp_budget = 10.0` is the pre-registered value the current threshold (**score ≥ 0.784**) and the
pilot were built on. Results should be quoted across several budgets with one named as primary — the full
curve is what makes the choice auditable.

## 7. How it is evaluated

The system metric is the **FROC** — sensitivity against false positives per minute — computed from
out-of-fold scores over a grouped 5-fold cross-validation of 90 recordings (49 IEDs). Folds are grouped by
recording, and everything label-derived is fitted inside each fold.

Results are reported as **hits, misses and false positives**. Correct rejections are deliberately not
counted: over continuous EEG there is no meaningful number of "non-events", and ~98.5% of candidates are
trivially negative, so specificity and accuracy would be inflated to the point of meaninglessness.

**Current cross-validated performance** (49 IEDs, out-of-fold):

| | |
|---|---|
| sensitivity @ ≤1 / ≤5 / ≤10 FP per min | 0.49 / 0.71 / 0.82 |
| hits / misses / false positives @ ≤10 FP/min | 40 / 9 / 172 |
| ROC-AUC | 0.869 |
| PR-AUC (chance 0.014) | 0.363 — 25× chance |
| median localisation error, over the 40 detected | 10 ms |
| median localisation error, over all 48 IEDs with an event | 12 ms |
| L1 / L2 recall | 1.00 / 0.98 |

Two null models must be cleared and both are: predicting the recording centre (0.29 at 4.0 FP/min) and
ranking candidates by raw prominence with no model at all (0.12 at ≤5 FP/min).

## 8. Files

| path | role |
|---|---|
| `src/detect_config.py` | `Config` — every tunable |
| `src/detect_data.py` | preprocessing, channel matching, electrode positions, train/test split |
| `src/detect_stage1.py` | L1 primitives |
| `src/detection.py` | L2, L3, L4 and `DetectionPipeline` |
| `src/detect_metrics.py` | FROC, localisation, cross-validation — evaluation only |
| `src/detect_plots.py` | one function per figure panel |
| `detect_model.joblib` | fitted pipeline: `Config` + classifier + L4 |
| `predict.py` | scan one EDF |
| `pilot.py` | scan a folder, write per-recording and per-event CSVs |

Each `src/` module runs its own self-check with `python src/<module>.py`.

**Loading a model verifies its provenance.** `DetectionPipeline.load` refuses a file whose stored `Config`
predates a field the current class defines, naming what is missing — because such a mismatch otherwise
produces plausible-looking output from meaningless scores.
