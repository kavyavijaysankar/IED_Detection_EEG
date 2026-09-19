# Recording-level evaluation on external unannotated EEG — what was tried and what came out

Companion to `recording_level_evaluation.md` (the protocol). This file records every analysis that was run
against the external pilot data, in the order it was run, with the numbers and the verdict for each. The
code is `IED Detection/recording_level_eval.ipynb` (12 sections, ~2 min for Run All); every figure is in
`figures/recording_level_*.png`. Nothing here refits a model — the notebook reads the two CSVs `pilot.py`
wrote and everything below is recomputable from them.

---

## 0. Provenance — what was evaluated

| | |
|---|---|
| data | `results/results_contrib/` and `results/results_noncontrib/` — `pilot_recordings.csv` (one row per file) and `pilot_events.csv` (one row per L2 event, with its L3/L4 score) |
| recordings | **25 confirmatory + 25 non-confirmatory**, 292,495 scored events, 0 failed files |
| montage | **19 channels matched on every file** — this run used the 19-electrode model, before the montage was widened to 25 (report.md §10, 2026-09-10). The numbers below belong to that model, not to the current pipeline |
| sample rates | 256 and 512 Hz on input, resampled to 250 by the loader |
| threshold | **0.766**, read from the CSV — the value `pilot.py` ran with. `recording_level_evaluation.md` still says 0.784; the code moved and the document did not. 0.784 is carried as a row in the sensitivity table |
| bad channels | 2 soft flags, 0 hard, 0 over cap — the two-threshold handling was near-inert here, as on Kural |
| duration | 20–50 min per recording (Kural was 11–14 s) |

The groups: **confirmatory** = the clinical report confirmed IEDs, but none are marked; **non-confirmatory** =
no IEDs reported. A second label arrived later for the non-confirmatory set — whether the *patient* has
epilepsy — and is used in §9–§11.

---

## 1. The question, and what it cannot answer

Without per-event annotation there are no hits, misses or false positives, so the FROC, sensitivity and
localisation error are all unavailable. What remains is whether **one number per recording** separates the
two groups. The unit of analysis is the recording.

Group separation does **not** show the detector fires *on* the IEDs. A detector responding to any correlate
of an epileptic recording — length, amplitude, acquisition site — separates the groups without detecting a
discharge. The annotated FROC on Kural carries the localisation claim; this analysis extends it to a larger
unannotated sample. They answer different questions and are reported as two results.

---

## 2. Pre-registration — fixed before the CSVs were opened

| | choice | fixed in |
|---|---|---|
| primary statistic | **detections per minute** at the run threshold | `recording_level_evaluation.md` |
| second statistic | **`mean top-N`**, N = 10 (mean of the 10 highest event scores in the recording) | report.md §9, pilot pre-test |
| primary analysis | AUC over recordings, 5,000-resample bootstrap 95% CI | protocol |
| secondary | one-sided Mann-Whitney (confirmatory > non-confirmatory), rank-biserial r = 2·AUC − 1 | protocol |
| threshold | whatever the run used; never re-chosen after seeing results | protocol |

Two documents pre-registered two different statistics. **Both are reported, side by side, always.** Every
other summary in this file is exploratory and is labelled so.

A rate, not a count, because detection count scales with recording length. That turned out to matter (§4).

---

## 3. Sample size

**25 per group — below the protocol's 30-per-group floor.** At this n:

- the AUC's bootstrap interval is about ±0.16 wide;
- the rank-sum test cannot reach one-sided p < 0.05 below **AUC 0.636**, and needs a true AUC of about
  **0.705** for 80% power.

So a null here is *no evidence of a difference*, not *evidence of none*, and every negative below has to be
read against those two numbers.

---

## 4. Duration — the groups differ in length

| group | n | median | range |
|---|---|---|---|
| confirmatory | 25 | 24.1 min | 20.1 – 49.5 |
| non-confirmatory | 25 | 21.2 min | 20.0 – 30.3 |

Reported because the protocol asks for it and because it is why every statistic in this file is a
**rate**, never a count — raw event count (`n_events`) reads AUC 0.773 for no reason other than the
confirmatory recordings being longer. Whether length leaks into the statistics that matter is tested in
§8a. Duration itself measures nothing.

---

## 5. Primary result — detections per minute: NULL

At threshold 0.766, confirmatory vs non-confirmatory:

| | confirmatory | non-confirmatory |
|---|---|---|
| median | 15.21 /min | 11.77 /min |
| IQR | 9.21 – 21.94 | 9.46 – 15.92 |
| mean rank | 27.6 | 23.4 |

- **AUC 0.582 [0.419, 0.741]**, rank-biserial r = +0.165
- Wilcoxon rank-sum / Mann-Whitney: U = 364 of 625, z = +0.999, **two-sided p = 0.322**, one-sided p = 0.161
- 20,000-shuffle permutation null reproduces both (p = 0.323 / 0.158) — the asymptotic test is fine at n = 25
- Hodges-Lehmann shift **+1.45 detections/min**

The direction is right, the size is small, and it sits well inside what 25 vs 25 recordings would produce
by chance. Observed 0.582 against a detectable-at-80%-power 0.705: **underpowered for an effect this
size, not disproven.**

**Why the rate fails where the top scores do not (§6):** at 0.766, a 20-minute clinical recording yields
150–1,300 above-threshold events. The threshold sits deep in the mimic mass, and mimic burden does not
differ between the groups — so the rate is measuring how many sharp transients a recording has, which
is the same in both.

---

## 6. Second pre-registered statistic — `mean top-10`: POSITIVE

| | AUC | 95% CI | MW p (one-sided) | r_rb | medians |
|---|---|---|---|---|---|
| **mean top-10** | **0.715** | [0.563, 0.858] | **0.0047** | +0.430 | 0.974 vs 0.966 |
| mean top-5 (exploratory) | 0.726 | [0.579, 0.863] | 0.0031 | +0.453 | 0.976 vs 0.972 |

The groups differ in *how convincing their best events are*, not in *how many events clear the bar*.

**This was predicted before the data arrived.** The Kural pilot pre-test (report.md §9, after amplitude
normalisation) put `mean top-5` at **0.666 [0.543, 0.779]** as the honest expectation. The external data
returned 0.726 [0.579, 0.863]. The interval covers the prediction; the prediction held.

Reporting this is legitimate only because report.md pre-registered it independently. Promoting it to the
headline *because* the rate failed would be the selection error the protocol names. It is the second
result, and the writeup must say the primary is null.

---

## 7. Threshold sensitivity — the rate's null is not a threshold artefact

AUC of detections/min, recomputed from the events CSV at each threshold:

| threshold | 0.50 | 0.60 | 0.70 | 0.75 | **0.766** | 0.784 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.597 | 0.587 | 0.598 | 0.592 | **0.582** | 0.574 | 0.586 | 0.582 | 0.594 | 0.651 |

Flat at 0.57–0.60 across the whole range, rising only at 0.95. The conclusion holds regardless of where
the threshold was fixed — and the rise at the very top is the lead into §11.

---

## 8. Threats to validity — each tested

### 8a. Duration

Three controls with different failure modes; agreement is what counts.

| statistic | raw | residualised on log-duration | 14 duration-matched pairs | **first 20 min only** |
|---|---|---|---|---|
| rate | 0.582 | 0.382 | 0.439 | **0.558** |
| mean top-10 | 0.715 | 0.622 | 0.566 | **0.690** |
| mean top-5 | 0.726 | 0.630 | 0.582 | **0.722** |

- **Truncating every recording to its first 20 minutes is the clean control** — no recording is shorter,
  so every one contributes exactly the same amount of EEG and duration cannot act at all. **`mean top-N`
  survives it** (0.722, 0.690) and the rate stays null (0.558).
- Residualising over-adjusts (duration is itself label-associated, so it removes real signal with the
  nuisance) and reads as a lower bound. Matching on 14 pairs (n = 28, CI ≈ [0.36, 0.80]) is too noisy
  to say anything.
- ρ(duration, `mean top-10`) = +0.48 — longer recordings offer more draws and so a higher top-N. Real,
  but the truncation control shows it is not what separates the groups.

**Verdict: the `mean top-N` separation is not a length artefact. The rate's null is not either.**

### 8b. The front end (L1 + L2, before any scoring)

`events/min` — consolidated L2 events per minute, before L3 sees anything — reads **AUC 0.483
[0.314, 0.644]**, p = 0.59. Both groups produce the same number of sharp transients. Any separation is
L3/L4's doing, exactly as on Kural (0.511). Raw candidate rate: 0.554, also chance.

### 8c. Background amplitude

The Kural confound (report.md §9): before normalisation, L3 scores tracked recording loudness (ρ = +0.708),
and normalisation collapsed that to +0.060.

| | external data |
|---|---|
| `background_uv` alone as classifier | AUC **0.640** [0.477, 0.792], p = 0.046; medians 6.17 vs 5.64 µV |
| ρ(amplitude, rate) | +0.161 (p = 0.26) |
| ρ(amplitude, mean top-10) | **+0.354** (p = 0.012) |

Not the +0.708 of the original confound, and amplitude alone barely reaches significance — but +0.354 is
not +0.060 either. **A residual amplitude channel is live on this data in a way it was not on Kural.**
It cannot explain `mean top-10` on its own (the amplitude AUC is lower and the correlation is moderate),
but it must be stated as a partial contributor. §9 has a further finding on this.

### 8d. Site and subject clustering

Filenames encode site and patient (`S06_001_EEG_003.edf` → site S06, subject S06_001).

| site | confirmatory | non-confirmatory |
|---|---|---|
| S01 | 5 | 7 |
| S02 | **0** | 6 |
| S06 | **9** | **0** |
| S07 | 6 | 6 |
| S08 | 5 | 6 |

- **9 of the 25 confirmatory recordings are one patient, `S06_001`** — also the longest (median 31 min) and
  loudest (8.55 vs 5.83 µV) in the set. 42 subjects for 50 recordings.
- Sites S06 and S02 appear in one group only, so acquisition is perfectly confounded with the label there.

| statistic | all 50 | subject-cluster bootstrap CI | shared sites only (n = 35) | drop S06_001 (n = 41) |
|---|---|---|---|---|
| rate | 0.582 | [0.41, 0.72] | 0.553 | 0.552 |
| mean top-10 | 0.715 | [0.55, 0.84] | 0.688 | 0.698 |

Per site (shared sites only): rate 0.60 / 0.50 / 0.50 — chance everywhere. mean top-10: S01 **0.51**,
S07 0.81, S08 0.80 — the signal is present at two sites and absent at one.

**Verdict: `mean top-N` survives every restriction (0.69–0.72) but is not consistent across sites; the
rate is chance under every restriction.** The one-patient cluster inflates nothing that survives.

### 8e. Negative-group contamination — cannot be tested here, must be stated

If "non-confirmatory" means the clinical report did not mention IEDs, some contain unreported discharges.
Contamination biases toward the null: a positive survives it, a null is harder to interpret. §9 makes this
concrete — 11 of the 23 labelled non-confirmatory recordings are from epileptic patients.

---

## 9. Exploratory — every other one-number summary (labelled, not headlined)

| statistic | AUC | 95% CI | MW p |
|---|---|---|---|
| mean top-5 | 0.726 | [0.579, 0.863] | 0.003 |
| mean top-3 | 0.722 | [0.573, 0.861] | 0.004 |
| **mean top-10 (pre-registered)** | 0.715 | [0.563, 0.858] | 0.005 |
| mean top-20 | 0.675 | [0.520, 0.825] | 0.017 |
| count ≥ 0.5 | 0.672 | [0.515, 0.818] | 0.019 |
| max score (top-1) | 0.665 | [0.506, 0.822] | 0.023 |
| background µV (confound) | 0.640 | [0.477, 0.792] | 0.046 |
| count ≥ 0.7 | 0.635 | [0.479, 0.783] | 0.052 |
| count ≥ 0.9 | 0.622 | [0.460, 0.777] | 0.070 |
| mean top-50 | 0.608 | [0.444, 0.765] | 0.097 |
| q95 | 0.598 | [0.433, 0.757] | 0.118 |
| p99 |µV| | 0.595 | [0.429, 0.749] | 0.126 |
| **detections/min (primary)** | 0.582 | [0.419, 0.741] | 0.161 |
| q90 | 0.579 | [0.414, 0.739] | 0.171 |
| q99 | 0.573 | [0.406, 0.737] | 0.191 |
| events/min (front-end control) | 0.483 | [0.314, 0.644] | 0.585 |

Sixteen statistics on the same 50 recordings: the best of them is optimistic by a few points of AUC,
exactly as the Kural pre-test warned. The `mean top-N` family (N = 3–10) sits together at 0.72; the
counts (`count ≥ 0.5/0.7/0.9`) are inflated by duration; the quantiles are chance.

---

## 10. Sub-analysis — does the rate tell epileptic from non-epileptic *within* the non-confirmatory group?

The supervisor supplied a patient-level epilepsy label for the non-confirmatory recordings (E / N). Two
were missing (`S02_006`, `S08_008`) and dropped rather than guessed: **11 epileptic, 12 non-epileptic**.

This asks something the main analysis cannot: among recordings with no discharges to find, does the
detector still respond to something about epileptic EEG? A positive answer is the correlate-confound; a
null is the reassuring one.

| statistic | AUC | 95% CI | p (two-sided) | medians E / N |
|---|---|---|---|---|
| **detections/min** | **0.515** | [0.265, 0.758] | **0.93** | 14.71 / 11.27 |
| mean top-10 | 0.561 | [0.308, 0.802] | 0.64 | 0.970 / 0.965 |
| background µV | **0.364** | [0.131, 0.617] | 0.28 | 5.41 / 6.28 |
| events/min | 0.598 | [0.348, 0.833] | 0.44 | 254.6 / 250.0 |

**No separation** — AUC 0.515. The medians differ but the ranks are interleaved. At 11 vs 12 the test cannot
reach p < 0.05 below AUC ≈ 0.70, so this is *no evidence*, not *evidence of none*.

**Two things worth more than the null itself:**

1. **The three-way comparison.** Splitting the non-confirmatory group by diagnosis and re-running against
   the confirmatory group: `mean top-10` gives **0.702 vs non-conf E** and **0.690 vs non-conf N**; the rate
   0.585 / 0.590. The same margin against both. Whatever `mean top-10` is detecting tracks **the recording
   being confirmatory**, not **the patient having epilepsy** — the result that argues *against* the
   correlate-confound reading and *for* "it responds to discharges". Weak at this n, right direction.
2. **Background amplitude runs backwards** (AUC 0.364 — non-epileptic patients' recordings are slightly
   louder). Kural's "epileptic recordings are louder" (6.26 vs 4.46 µV) does not reproduce. It was flagged
   as a possible Kural acquisition artefact; it looks like it was one.

---

## 11. Detections per minute across FP budgets, three groups

**The FP budget is read off this data, not imported from Kural.** `non-conf N` recordings have no IEDs and
no epilepsy, so every detection in them is a false positive by construction: their detections/min *is*
the empirical FP rate. Each budget below is inverted on those 12 recordings — the threshold at which
their median rate hits the target — and all three groups are read there. Medians, because the means are
outlier-driven and non-monotone (15.6 / 13.4 / 19.7 at 0.766) while the medians are not.

The hypothesis is ordered (IEDs present > epilepsy but none recorded > no epilepsy), so the test is
**Jonckheere-Terpstra** with a permutation null, not Kruskal-Wallis.

| FP budget | threshold | non-conf N | non-conf E | confirmatory | C / N | JT p |
|---|---|---|---|---|---|---|
| 1 | 0.942 | 0.98 | 1.48 | **2.06** | **2.10** | 0.123 |
| 2 | 0.924 | 2.01 | 2.37 | 2.99 | 1.49 | 0.152 |
| 5 | 0.875 | 5.00 | 5.43 | 5.72 | 1.14 | 0.258 |
| 10 | 0.789 | 10.00 | 12.05 | 12.87 | 1.29 | 0.179 |
| 25 | 0.563 | 25.02 | 27.55 | 30.76 | 1.23 | 0.104 |
| 50 | 0.362 | 50.02 | 52.50 | 57.25 | 1.14 | 0.159 |
| *Kural's 0.766* | 0.766 | 11.27 | 14.71 | 15.21 | 1.35 | 0.153 |

- **The ordering is right in all seven rows** — confirmatory > non-conf E > non-conf N, no exceptions.
  Never significant (JT p 0.10–0.26), never broken.
- **Separation is best at the tightest budget**: 2.1× at 1 FP/min, down to 1.14× by 5. The FROC's left
  end behaving as it should — the top-ranked detections are the informative ones, and loosening the
  threshold buries them in mimics. This is the mechanism behind §5's null.
- **The operating point transfers.** Kural's ≤10 FP/min threshold (0.766) delivers **11.27 FP/min** on
  external IED-free EEG; exactly 10 needs 0.789. A 13% miss on a threshold calibrated on 11–14 s clips
  and applied to 20–50 minute recordings from four other sites. Independent support for the
  normalise-for-transfer argument.
- The per-recording strips (`recording_level_fp_budget_strips.png`) show what the medians hide: IQRs
  overlap almost completely at every budget; the 2.1× at 1 FP/min is carried by 3–4 confirmatory
  recordings running 15–40 /min, not by a group-wide shift; and one non-epileptic recording is the top
  scorer at every budget (38 /min at the 1 FP/min point).
- `non-conf E` sitting between the other two at every budget is consistent with a small amount of unmarked
  epileptiform activity in epileptic patients whose recordings were called non-confirmatory. Suggestive at
  n = 11.

---

## 12. The best-separating FP budget — with the selection paid for

The protocol forbids selecting the threshold that maximises separation and quoting its p-value as if it
had been fixed. The search was run anyway, because the shape of the AUC-vs-budget curve is informative,
and the selection is **corrected for, not hidden**. Three numbers instead of one.

Grid: 338 operating points, 0.07–61.7 FP/min.

| | threshold | FP budget | AUC | medians C / E / N |
|---|---|---|---|---|
| **best** | 0.962 | **0.30 /min** | **0.685** | 1.04 / 0.53 / 0.30 |
| pre-registered | 0.766 | 11.27 /min | 0.582 | 15.21 / 14.71 / 11.27 |

- **Naive** (as if pre-registered — must not be quoted alone): AUC 0.685, 95% CI [0.531, 0.829],
  one-sided p = 0.013.
- **Selection-corrected p = 0.046** — max-statistic permutation. Shuffle the labels, take the best AUC across
  all 338 operating points, 2,000 times: on data with no group difference, best-of-grid averages **0.562**
  and its 95th percentile is **0.681**. The observed 0.685 clears that bar, just. The multiplicity is inside
  the null, so no further correction applies.
- **The search's optimism is ~0.06 AUC** — a true 0.5, searched over this grid, already reads 0.562.
- **Stability of the selection**: bootstrap the recordings and re-run the whole search 2,000 times.
  **82% of resamples pick a budget below 1 FP/min**; the top 8 grid points all lie in 0.07–0.37 FP/min.
  The *region* is stable; the exact point is not (95% range of the argmax spans the grid).

**How to report it:** §5's pre-registered null stays the headline. This is a secondary exploratory analysis,
quoted as *"separation is maximised at a tight operating point below 1 FP/min, AUC 0.685,
selection-corrected p = 0.046"*, with the naive CI shown as naive. A corrected p of 0.046 on 25 vs 25, just
inside the threshold, will not survive a stricter grid or a smaller sample — suggestive, not established.

Caveat: at 0.3 FP/min a 24-minute recording yields ~7 detections, so the rates are coarsely quantised, and
the one loud non-epileptic recording (28 /min) still tops every group.

---

## 13. What to conclude

**The pre-registered primary statistic does not separate the groups.** Detections/min at the fixed
threshold: AUC 0.582 [0.419, 0.741], p = 0.32. Robust to the threshold, to duration, to site restriction —
it is chance everywhere. The reason is mechanistic: at any budget above ~1 FP/min, the count is dominated
by mimics, and both groups have the same mimic burden (front-end control at 0.483).

**The second pre-registered statistic does.** `mean top-10`: AUC 0.715 [0.563, 0.858], p = 0.005. It
survives the length-equalisation control (0.690), site restriction (0.688), dropping the one-patient
cluster (0.698), and a subject-level bootstrap. It matches the Kural pre-test's prediction (0.666
predicted, 0.726 observed for top-5). It separates confirmatory recordings from epileptic *and*
non-epileptic non-confirmatory ones by the same margin, which points at discharges rather than a
diagnosis correlate. Against it: a residual amplitude correlation (+0.354) that was absent on Kural, and
inconsistency across sites (0.51 at S01, 0.80 at S07/S08).

**The groups differ in how convincing their best few events are, not in how many events clear a bar.**
Every angle — the top-N family, the rate ratios by budget, the best-budget search — says the same thing:
the information is in the top of the ranking, and the tighter the operating point the more of it shows.

**Three things the writeup has to say plainly:** n = 25 per group is below the protocol's floor and the
rate test could only have seen AUC ≥ 0.70; the groups differ in duration and in site
composition, and 9 of 25 confirmatory recordings are one patient; and none of this shows the detector
fires *on* the discharges — the Kural FROC does that, this extends it.

---

## 14. Reporting checklist (protocol §"Minimal reporting checklist"), filled

- **n per group**: 25 / 25 — below the 30 floor. Duration: confirmatory 24.1 min [20.1–49.5], non-confirmatory
  21.2 [20.0–30.3].
- **threshold**: 0.766, fixed in advance by `pilot.py`, not re-chosen; AUC across 0.5–0.95 spans 0.574–0.651.
- **primary statistic named before results**: detections per minute.
- **AUC with bootstrap CI**: 0.582 [0.419, 0.741].
- **Mann-Whitney p and rank-biserial r**: p = 0.161 (one-sided), 0.322 (two-sided); r = +0.165.
- **second pre-registered statistic**: mean top-10, AUC 0.715 [0.563, 0.858], p = 0.005, r = +0.430.
- **distribution plot**: `recording_level_distributions.png`.
- **AUC across alternative thresholds**: §7 table, `recording_level_threshold_sensitivity.png`.
- **contamination and correlate-confound limits**: stated in §8e and §1; 11 of 23 labelled non-confirmatory
  recordings are from epileptic patients.

---

## 15. Figures

| file | shows |
|---|---|
| `recording_level_distributions.png` | rate and mean top-10 by group, every recording |
| `recording_level_roc.png` | ROC of the two pre-registered statistics |
| `recording_level_ranksum_null.png` | permutation null of the rank-sum statistic for the rate |
| `recording_level_threshold_sensitivity.png` | AUC of the rate vs threshold with CI band |
| `recording_level_all_statistics.png` | forest plot of all 16 summaries, pre-registered ones marked |
| `recording_level_epilepsy_subgroup.png` | rate by confirmatory / non-conf E / non-conf N |
| `recording_level_fp_budgets.png` | median rate vs threshold, three groups; rate ratio vs FP budget |
| `recording_level_fp_budget_strips.png` | every recording at each of six FP budgets |
| `recording_level_best_budget.png` | AUC vs FP budget with the permutation bar; strips at the best budget |
