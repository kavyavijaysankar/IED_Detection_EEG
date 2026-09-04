# Recording-level evaluation on unannotated EEGs

## The problem

Two groups of recordings:

- **confirmatory** — known to contain IEDs, but the IEDs are not marked
- **non-confirmatory** — no IEDs

Without per-event annotation there are no hits, misses or false positives, so
FROC, sensitivity and localisation error are all unavailable. What remains is a
weaker but still meaningful question: does the detector's output separate the two
groups?

The unit of analysis is therefore the **recording**, not the detection.

---

## The summary statistic

One number per recording. Two defensible choices:

| Statistic | Question it asks |
|---|---|
| detections per minute above threshold | *how much* epileptiform activity is present |
| maximum event score in the recording | does the recording contain *at least one* convincing discharge |

Rate is the more natural fit if IED burden varies; maximum score is closer to what
a reader does when scanning a recording. **Pick one as primary before looking at
results.** Reporting whichever separates better is the same error as choosing a
statistical test after seeing the data.

Rate must be per minute, not a raw count. Detection count scales with recording
duration, so if the groups differ in mean length, a duration effect will appear as
a group effect.

---

## Two analyses

### Primary: discrimination (AUC)

AUC over recordings is the probability that a randomly chosen confirmatory
recording scores higher than a randomly chosen non-confirmatory one. It answers
the clinical question — can this recording be classified? — and has a published
comparator: Jing et al. (2020) report AUC 0.847 for whole-EEG classification from
SpikeNet scores.

```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# one row per recording
# labels: 1 = confirmatory, 0 = non-confirmatory
# rates:  detections per minute at the fixed threshold

auc = roc_auc_score(labels, rates)
fpr, tpr, thr = roc_curve(labels, rates)
```

Confidence interval by bootstrap over recordings:

```python
rng = np.random.default_rng(0)
boot = []
n = len(labels)
for _ in range(2000):
    idx = rng.integers(0, n, n)
    if len(np.unique(labels[idx])) < 2:      # skip degenerate resamples
        continue
    boot.append(roc_auc_score(labels[idx], rates[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
```

### Secondary: group difference

Evidence that the separation is not chance. Detection rates are typically skewed
and the groups have unequal variance, so use a rank-based test.

```python
from scipy.stats import mannwhitneyu

u, p = mannwhitneyu(rates[labels == 1], rates[labels == 0],
                    alternative='greater')
```

Report an effect size alongside the p-value. The rank-biserial correlation is the
natural one here, and relates directly to AUC:

```
r_rb = 2 * AUC - 1
```

### Always: show the distributions

A strip or box plot of rate by group, points overlaid. Both statistics above are
summaries of this picture; the overlap is the finding, whichever way it falls.

---

## Sample size

Size to the AUC, which is the more demanding of the two.

| n per group | approx. 95% CI half-width on AUC (true AUC ≈ 0.85) |
|---|---|
| 20 | ±0.13 |
| 30 | ±0.10 |
| 50 | ±0.07 |
| 80 | ±0.05 |

Below about 20 per group the interval spans 0.75 to 0.95, which cannot distinguish
a useful detector from an excellent one. **30 per group is a defensible floor; 50
is comfortable.**

For reference, the secondary test needs less: a Mann-Whitney at 80% power,
α = .05, needs roughly 27 per group for a large effect (d = 0.8) and 67 per group
for a moderate one (d = 0.5).

Groups need not be equal in size, but power is driven by the smaller one.

---

## Threshold handling

The threshold determines which events are counted, so it determines every rate and
therefore the AUC.

- **Fix it in advance.** Use the pre-registered operating point (`fp_budget = 10`,
  score ≥ 0.784). Do not select the threshold that maximises separation — that
  invalidates the result.
- **Report sensitivity to the choice** as a secondary analysis: recompute AUC
  across several budgets and show it as a small table or curve. If the conclusion
  holds across the range, say so; if it does not, that is a finding too.

---

## Threats to validity, to state explicitly

**Negative group contamination.** If "no IEDs" means the clinical report did not
mention any, some recordings may contain unreported discharges. Routine reports
miss IEDs — the real-world Persyst evaluation found PPV near 20% against clinical
reports as reference. Contamination biases toward the null, so a positive result
survives it; a null result is harder to interpret.

**Duration imbalance.** Check it directly and report the group medians. Using a
rate rather than a count handles the first-order effect, but very short recordings
give noisier rates.

**Multiple summaries.** If you try both rate and maximum score, say so and report
both. Reporting only the better one is selection on the outcome.

**What this design cannot show.** Group separation does not establish that the
detector fires *on the IEDs*. A detector responding to some correlate of the
epileptic recording — background differences, medication effects, artefact rate —
could separate the groups without detecting a single discharge. The annotated
FROC evaluation is what supports the localisation claim; this analysis extends it
to a larger, unannotated sample. State the two as answering different questions.

---

## Minimal reporting checklist

- n per group, and recording duration (median, range) per group
- threshold used, and that it was fixed in advance
- primary statistic named before results
- AUC with bootstrap 95% CI
- Mann-Whitney p and rank-biserial r
- distribution plot
- AUC across alternative thresholds
- explicit statement of the contamination and correlate-confound limits
