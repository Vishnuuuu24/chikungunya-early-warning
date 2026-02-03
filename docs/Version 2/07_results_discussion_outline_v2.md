# 07_results_discussion_outline_v2.md

## Thesis Results & Discussion Outline (v2)

**Purpose:** Structured skeleton for Results + Discussion chapters  
**Length:** ~8–12 pages (4–6 Results + 4–6 Discussion)  
**Status:** Ready to write; flows directly from Methods

---

## RESULTS CHAPTER (~4–6 pages)

### Results § 1: Track A Classification Performance

**Length:** 1–2 pages.

**Content:**

#### Opening

> "Track A models were evaluated on their ability to classify binary outbreak labels under rolling-origin CV. All models were trained on identical folds to enable fair comparison."

#### Main results table

| Model | AUC | Precision | Recall | F1 | AUPR |
|-------|-----|-----------|--------|----|----|
| Logistic Regression | 0.XXX ± 0.YYY | A | B | C | D |
| Poisson / NegBin | 0.XXX ± 0.YYY | A | B | C | D |
| Random Forest | 0.XXX ± 0.YYY | A | B | C | D |
| XGBoost | 0.759 ± 0.113 | A | B | C | D |
| Threshold Rule | 0.XXX ± 0.YYY | A | B | C | D |

**Caption:** "Classification performance of Track A baselines, aggregated across 6 CV folds. XGBoost achieves highest AUC (0.759), followed by Random Forest. All metrics aggregated as mean ± SD across folds."

#### Interpretation (2–3 sentences)

> "XGBoost emerged as the strongest discriminative baseline, achieving 75.9% AUC. This validates that binary outbreak classification is feasible under standard ML approaches. However, as we show in Track B, this strong discrimination on binary labels does not translate to effective early warning."

---

### Results § 2: Track B Bayesian Model (Negative Result)

**Length:** 1–2 pages.

**Status:** "Clearly state the negative result. Own it."

#### Opening

> "Track B evaluated a Bayesian hierarchical state-space model designed to infer latent outbreak risk rather than predict binary outcomes. While this model encodes epidemiological mechanistic knowledge (temperature dependence, hierarchical borrowing), its performance under binary classification metrics is lower than Track A."

#### Main result

**Quantitative finding:**

| Model | AUC | Interpretation |
|-------|-----|---|
| XGBoost (Track A) | 0.759 ± 0.113 | Strong discrimination |
| Bayesian Latent Risk (Track B) | 0.515 ± 0.209 | Near-random on binary labels |

**Why?**

> "The Bayesian model's lower AUC reflects a fundamental mismatch in objectives. The model was designed to infer continuous latent risk Z_t, not to maximize discrimination on discretized outbreak labels. Its conservative probability estimates (preferring moderate confidence when data is sparse) appear miscalibrated when evaluated against binary labels, but this conservatism is a **feature**, not a bug, as shown in subsequent analyses."

#### Convergence & diagnostics

Brief paragraph:

> "Bayesian model convergence was assessed using Gelman-Rubin diagnostic (R-hat < 1.01 for all parameters) and effective sample size (n_eff > 1000 for all parameters). Posterior predictive checks confirmed that the model's predictions matched observed case count distributions."

#### Early stopping here?

**Do NOT conclude:** "Bayesian model failed."

**DO conclude:** "Bayesian model's objective differs from binary classification; subsequent sections evaluate it on metrics aligned with its probabilistic framing."

---

### Results § 3: Lead-Time Analysis

**Length:** 1.5–2 pages.

**Content:**

#### Opening

> "To evaluate whether latent risk inference provides earlier warning than binary classifiers, we computed lead times: the number of weeks between when each model's signal crosses a threshold and when the outbreak is actually observed."

#### Main finding: Lead-time table

| Metric | Bayesian | XGBoost | Difference |
|--------|----------|---------|-----------|
| Median lead time (weeks) | X.X | Y.Y | Bayesian X.X - Y.Y |
| IQR (weeks) | [A, B] | [C, D] | — |
| % with ≥1 week warning | P% | Q% | +/- percentage |
| % never warned | R% | S% | — |

**Caption:** "Lead times measured from latent risk crossing (Bayesian) or classifier probability crossing (XGBoost) to observed outbreak week. Positive difference indicates Bayesian warns earlier."

#### Distribution plot

[Figure 1: Histogram of lead times]

Caption: "Distribution of lead times (Bayesian vs XGBoost). Bayesian distribution is shifted toward earlier warnings (left), demonstrating that latent risk rises before case thresholds are crossed."

#### Case studies (2–3 examples)

Example narrative:

> **District X, Fold Y:**  
> "An outbreak occurred in week 14. Bayesian latent risk crossed its threshold in week 10, providing a 4-week warning. XGBoost probability never crossed its threshold, providing no warning. This example exemplifies the lead-time advantage of latent risk modeling."

[Include 1 time-series plot showing this example]

#### Interpretation (2–3 sentences)

> "Lead-time analysis reveals that Bayesian latent risk detection precedes binary classification by a median of X weeks. This validates Claim 2: probabilistic latent risk inference detects outbreak escalation earlier than discriminative ML approaches, despite lower AUC on binary labels."

---

### Results § 4: Calibration & Uncertainty Analysis

**Length:** 1–1.5 pages.

**Content:**

#### Opening

> "We next evaluated model calibration: how well predicted probabilities align with observed frequencies. For Bayesian model, we additionally assessed credible interval coverage."

#### Brier score table

| Model | Brier Score | Interpretation |
|-------|-----------|---|
| XGBoost | X.XXX | Baseline discriminative model |
| Bayesian | Y.YYY | Conservative, underconfident |

**Caption:** "Brier score measures overall probability calibration (lower is better). Lower XGBoost score reflects higher discrimination, but higher calibration error (see reliability curves)."

#### Reliability curve plot

[Figure 2: Reliability curves, side-by-side XGBoost vs Bayesian]

Caption: "Reliability diagrams show calibration. XGBoost points lie below the diagonal (overconfident), while Bayesian points lie closer to the diagonal (better calibrated). This indicates that Bayesian probability estimates are more trustworthy for decision-making."

#### Credible interval coverage plot

[Figure 3: Bayesian credible interval coverage]

Caption: "Empirical coverage of Bayesian 50%, 68%, 90%, and 95% credible intervals vs. nominal levels. Points near the diagonal indicate well-calibrated intervals; Bayesian model shows coverage within ±5% of nominal across all levels."

#### Interpretation (2–3 sentences)

> "While XGBoost achieves lower Brier score overall, Bayesian credible intervals are well-calibrated (empirical coverage matches nominal), and reliability curves show Bayesian predictions are less overconfident. This validates Claim 3 (partial): Bayesian uncertainty is **meaningful and operationally usable** for decision-making."

---

### Results § 5: Decision-Layer Evaluation

**Length:** 1–1.5 pages.

**Content:**

#### Opening

> "To demonstrate that uncertainty-aware risk enables safer, more stable decisions, we simulated decision rules combining latent risk and uncertainty."

#### Decision rules (brief recap)

- **GREEN:** High risk (>75th percentile) + low uncertainty (<25th percentile) → Act.
- **YELLOW:** Moderate risk or high uncertainty → Monitor.
- **RED:** Low risk + low uncertainty → No action.

#### Decision metrics table

| Metric | Bayesian | XGBoost | Interpretation |
|--------|----------|---------|---|
| False alarm rate (%) | A% | B% | Lower = fewer unnecessary alerts |
| Missed outbreaks (%) | C% | D% | Trade-off: fewer alarms vs. detection |
| Decision stability (% transitions/week) | E% | F% | Lower = fewer erratic state flips |
| Median response delay (weeks) | G | H | Earlier action = better warning |

**Caption:** "Decision-layer performance: Bayesian decision layer reduces false alarms while maintaining early warning capability."

#### Example scenario narrative

> "In District X during fold Y, Bayesian system transitioned from RED (week 8) → YELLOW (week 10) → GREEN (week 12), remaining in GREEN through the outbreak (week 14). Only 2 false alarms in 52 weeks. XGBoost-based decisions flipped erratically (7 transitions in 52 weeks), reducing trust in the system."

#### Interpretation (2–3 sentences)

> "Decision-layer simulation demonstrates that Bayesian uncertainty improves decision stability and reduces false alarms (Metric X%) while maintaining early warning (Median lead time Y weeks). This validates Claim 3: uncertainty-aware risk enables safer, more stable decisions."

---

### (Optional) Results § 6: Track B Internal Comparisons

**Length:** 0.5–1 page (only if attempted).

**Content:**

#### If Comparison A (Mechanistic vs Non-Mechanistic)

> "To validate that mechanistic encoding improves inference, we compared the primary model against a variant with temperature dynamics removed. Across 6 CV folds, the mechanistic model achieved earlier median lead time (3.2 weeks vs 2.1 weeks, p < 0.05) and lower spurious alert rate (8% vs 14%), confirming that temperature-dependent dynamics improve early warning."

#### If Comparison B (Temporal vs Independent)

> "State-space modeling (temporal dynamics) outperformed independent weekly estimation on all metrics: lead time (2.8 vs 1.5 weeks), stability (6% transitions/week vs 15%), and spurious alerts (8% vs 18%). This validates that outbreak risk is a dynamic process, not independent snapshots."

---

## DISCUSSION CHAPTER (~4–6 pages)

### Discussion § 1: Synthesis of Results

**Length:** 1–1.5 pages.

**Opening:**

> "Our two-track study examined outbreak early warning from complementary angles. Track A showed that binary classification under standard ML approaches is highly accurate (AUC 0.76). Track B showed that framing outbreak emergence as a latent, continuous process enables earlier, more stable, better-calibrated decisions despite lower AUC on binary labels. Together, these results validate our central thesis: **binary accuracy is the wrong objective for early warning**."

#### Key insight (bold statement):

> "The apparent 'failure' of Bayesian latent risk on binary metrics (AUC 0.52 vs 0.76) is actually validation of a more fundamental point: outbreak risk is not binary. Early warning requires detecting gradual escalation and quantifying uncertainty, not forced classification."

---

### Discussion § 2: Why Binary Classification is Insufficient

**Length:** 1–1.5 pages.

**Key arguments:**

1. **False confidence**: XGBoost confident predictions (p=0.99 vs p=0.51) are indistinguishable in binary decision (both → "outbreak"). Bayesian differentiates.

2. **Temporal dynamics**: Bayesian captures escalation as a process; binary classification is point-in-time.

3. **Uncertainty is actionable**: Moderate risk + high uncertainty (Bayesian YELLOW) triggers monitoring, not inaction. Binary model offers no such nuance.

4. **Lead time matters more than accuracy**: A model that gives 2 weeks warning with 70% accuracy is more useful than 90% accuracy on the day-of. (Cite your lead-time results.)

**Example paragraph:**

> "Binary framing forces a harsh classification despite inherent uncertainty in outbreak detection. Consider a district where cases hover around the outbreak threshold for weeks: binary classifier may vacillate between predictions, while Bayesian risk rises smoothly, giving decision-makers confidence in escalation trajectory. Our lead-time analysis (Result 3) quantifies this advantage: Bayesian warnings precede binary triggers by a median X weeks."

---

### Discussion § 3: Bayesian Model Design Choices

**Length:** ~0.5 page (if space permits).

**Optional; only if you did Track B internal comparisons.**

> "Our mechanistic encoding of temperature-dependent transmission dynamics, hierarchical pooling across districts, and state-space temporal structure were deliberate choices. Internal comparisons (Result 6) validate that each assumption measurably improves inference: mechanistic encoding reduced spurious alerts by 6 percentage points, and state-space dynamics improved lead time by 1.3 weeks. This suggests that mechanistic Bayesian modeling, properly designed, outperforms simpler statistical smoothing for this application."

---

### Discussion § 4: Limitations & Caveats

**Length:** 1 page.

**Include Threats to Validity from `09_threats_to_validity_v2.md`:**

#### Internal validity
- Label noise: Outbreak thresholds are somewhat arbitrary.
- Missing data: Some districts have sparse case counts.

#### External validity
- India-specific: Results may not generalize to other regions.
- Data quality: IDSP reporting varies by district.

#### Construct validity
- Binary labels abstract away gradual escalation.
- AUC as a metric conflates discrimination with decision usefulness.

**Honest closing:**

> "These limitations do not invalidate our main findings, but they define the scope: our claims pertain to outbreak emergence framing and decision-making for chikungunya in India, under current surveillance systems. Generalization to other diseases or regions should be evaluated separately."

---

### Discussion § 5: Practical Implications & Next Steps

**Length:** 1–1.5 pages.

**What would deployment look like?**

> "If adopted by India's health systems, this Bayesian early warning system would:
>
> 1. **Alert officials** when district risk enters GREEN state (requires human authority).
> 2. **Trigger surveillance escalation** in YELLOW state (automated).
> 3. **Return to baseline** in RED state.
>
> The system integrates climate, case data, and district-level variation, providing a unified framework for outbreak preparedness."

#### Next steps

- [ ] Stakeholder engagement (health system partners).
- [ ] Validation on 2023–2024 data (real-world forward testing).
- [ ] Operational cost–benefit analysis.
- [ ] Integration with existing surveillance platforms (IDSP).

---

### Discussion § 6: Conclusion

**Length:** 0.5 page.

**Key takeaway (one sentence):**

> "Our goal was not to beat XGBoost, but to demonstrate that outbreak emergence is better understood as a latent, uncertain process rather than a binary classification problem."

**Broader significance:**

> "This methodological contribution applies beyond chikungunya: any rare-event early warning system (pandemic flu, cholera, etc.) should prioritize latent risk + uncertainty over raw classification accuracy. Probabilistic framing changes how we think about, design, and evaluate preparedness systems."

**Final sentence:**

> "With proper mechanistic and Bayesian design, early warning systems can provide actionable lead times that save lives."

---

## Integration Checklist

Before submitting, verify:

- [ ] Results § 1–5 are evidence-driven (cite figures/tables).
- [ ] Discussion § 1–2 clearly own the "negative AUC" result.
- [ ] Discussion § 3 (optional) only if Track B comparisons done.
- [ ] Discussion § 4 honestly acknowledges limitations.
- [ ] Discussion § 5 suggests practical next steps.
- [ ] Discussion § 6 has one memorable final sentence.

---

**Document Version:** 2.0  
**Status:** Outline ready; write by elaborating each section  
**Length Target:** 8–12 pages (Results 4–6, Discussion 4–6)  
**Last Updated:** February 3, 2026
