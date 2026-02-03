# 02_lead_time_analysis_spec_v2.md

## Lead-Time Analysis Specification

**Priority:** CRITICAL and non-negotiable  
**Purpose:** Validate Claim 2: "Latent risk inference captures outbreak escalation earlier than binary classifiers"  
**Effort:** Medium (~1–2 weeks implementation)  
**Output:** Numeric table + distribution plot + case studies + key insight

---

## 1. Conceptual Overview

### The Question

Your Bayesian model has lower AUC on binary outbreak labels (0.52 vs XGBoost 0.76). This seems like failure — but it's actually a **category mismatch**.

Lead-time analysis answers the real question: **Does Bayesian latent risk rise BEFORE XGBoost's binary prediction triggers?**

If yes, then Bayesian is doing something XGBoost cannot: early warning.

### Why This Matters

- AUC measures discrimination at a fixed threshold.
- Lead time measures **when** signals rise relative to ground truth.
- Early warning systems are judged on **how much advance warning they provide**, not on point accuracy.

---

## 2. Definitions (Exact & Reproducible)

### 2.1 Outbreak Week (Ground Truth)

**Definition:**  
A week \( t \) is an **outbreak week** if the observed case count \( Y_t \) exceeds a pre-specified threshold for that district.

**Threshold:** Use the **80th percentile** of historical case counts for each district (computed on training folds only, not test).

**Formal:**
- Let \( \mathcal{D}_i^{\text{train}} \) = case counts in district \( i \) during training folds.
- Threshold \( \tau_i = \text{quantile}(\mathcal{D}_i^{\text{train}}, 0.80) \).
- Outbreak week: \( Y_{i,t} > \tau_i \).

**Why 80th percentile?** Balances sensitivity (catches most outbreaks) with specificity (not too many false alarms).

**Robustness check:** Also compute lead time using 70th and 90th percentiles; report all three. Conclusion should be robust to choice.

---

### 2.2 Bayesian Risk Crossing Event

**Definition:**  
A week \( t \) is a **Bayesian risk crossing** if the latent risk posterior mean \( \hat{Z}_t \) exceeds a chosen threshold for that district.

**Threshold:** Use the **75th percentile** of the posterior predictive distribution of \( Z_t \) in test folds (computed as you generate predictions).

**Formal:**
- Posterior samples: \( \{Z_t^{(s)}\}_{s=1}^{S} \) where \( S \) = # of posterior samples (typically 2000–4000).
- District-specific threshold: \( \xi_i = \text{quantile}(\{\hat{Z}_{i,t} : t \in \text{test}\}, 0.75) \).
- Bayesian risk crossing: \( \hat{Z}_{i,t} > \xi_i \).

**Why 75th percentile?** Asymmetrically lower than 80% for outbreak (to reflect that Bayesian is conservative). This gives it a fair chance to rise earlier.

---

### 2.3 XGBoost Binary Prediction Event

**Definition:**  
A week \( t \) is an **XGBoost trigger** if the predicted probability \( \hat{p}_{i,t}^{\text{XGB}} \) exceeds a chosen threshold.

**Threshold:** Use **0.5** (standard classification threshold).

**Alternative:** Also report results at threshold = 0.3 (more permissive, more early warnings). Show both; emphasize that choice of threshold matters.

**Formal:**
- XGBoost trigger: \( \hat{p}_{i,t}^{\text{XGB}} > 0.5 \).

---

### 2.4 Lead-Time Calculation

**For each outbreak episode (district, outbreak week \( t^* \)):**

1. **Identify outbreak week:** \( t^* = \) first week where \( Y_{i,t^*} > \tau_i \).
2. **Identify Bayesian rising:** \( t_B = \) first week (before or on \( t^* \)) where \( \hat{Z}_{i,t_B} > \xi_i \).
3. **Identify XGBoost rising:** \( t_X = \) first week (before or on \( t^* \)) where \( \hat{p}_{i,t_X}^{\text{XGB}} > 0.5 \).
4. **Lead times:**
   - Bayesian lead time: \( L_B = t^* - t_B \) (weeks).
   - XGBoost lead time: \( L_X = t^* - t_X \) (weeks).
   - **Differential lead time:** \( \Delta L = L_X - L_B \) (positive = Bayesian earlier).

**Edge cases:**

- If Bayesian never rises before outbreak, set \( t_B = t^* + 1 \), so \( L_B = -1 \) (never warned).
- If XGBoost never rises before outbreak, set \( t_X = t^* + 1 \), so \( L_X = -1 \) (never warned).
- If both never rise, exclude this outbreak episode from lead-time comparison (but count it separately as "missed by both").

---

## 3. Aggregation & Summary Statistics

### 3.1 Overall Lead Time (Across All Test Folds & Districts)

**For each model (Bayesian vs XGBoost):**

- Median lead time (weeks).
- IQR (25th to 75th percentile).
- Mean (with SD).
- % of outbreaks with lead time ≥ 1 week (early warning achieved).
- % of outbreaks with lead time = −1 (never warned).

**Table format:**

| Metric | Bayesian | XGBoost | Interpretation |
|--------|----------|---------|-----------------|
| Median lead time (weeks) | X.X | Y.Y | ... |
| IQR | [A, B] | [C, D] | ... |
| % early warned (≥1 week) | P% | Q% | ... |
| % never warned | R% | S% | ... |

---

### 3.2 Per-Fold Lead Time

Repeat above table for each fold (2017, 2018, ..., 2022):

- Shows consistency of advantage across years.
- Highlights if advantage disappears in certain years (data quality? seasonal variation?).

---

### 3.3 Per-District Lead Time

For each district, show:

- # of outbreak episodes in test folds.
- Median lead time (Bayesian vs XGBoost).
- Differential lead time (\( \Delta L \)).

**Sort by \( \Delta L \) descending** to show which districts benefit most from Bayesian early warning.

**Output:** Small table (top 5 districts by Bayesian advantage) + note about heterogeneity.

---

## 4. Visualization

### 4.1 Lead-Time Distribution (Primary Plot)

**Type:** Histogram or violin plot (side-by-side).

**Data:** Lead time \( L_B \) and \( L_X \) for all outbreak episodes.

**Axes:**
- X-axis: Lead time (weeks). Range: \([-1, 12]\) (−1 = never warned; positive = weeks ahead of outbreak).
- Y-axis: Count (histogram) or density (violin).

**Key visual cues:**
- Color code: Bayesian (blue), XGBoost (orange).
- Overlay median lines and IQR bands.
- Annotate % early warned (≥1 week) for each model.

**Interpretation:** Visual proof that Bayesian distribution is skewed toward earlier lead times.

---

### 4.2 Differential Lead Time (Secondary Plot)

**Type:** Histogram.

**Data:** \( \Delta L = L_X - L_B \) for all outbreaks.

**Axes:**
- X-axis: \( \Delta L \) (weeks). Positive = Bayesian earlier.
- Y-axis: Count.

**Interpretation:** How many outbreaks have Bayesian warning earlier? By how many weeks on average?

---

### 4.3 Case Studies (Qualitative Validation)

**Select 3–5 outbreak episodes** that show the story clearly. For each:

**Plot:**

- X-axis: Time (weeks, e.g., weeks 1–30 around the outbreak).
- Y-axis: Risk/probability.
- Three lines:
  1. Bayesian posterior mean \( \hat{Z}_{i,t} \) (blue).
  2. XGBoost predicted probability \( \hat{p}_{i,t}^{\text{XGB}} \) (orange).
  3. Observed case count \( Y_{i,t} \) (red, scaled to same range).
- Threshold lines: Bayesian threshold \( \xi_i \) (blue dashed), XGBoost threshold 0.5 (orange dashed), outbreak threshold \( \tau_i \) (red solid).
- Annotations: First crossing points for each model, actual outbreak week.

**Narrative (per case study):**

> "District X, fold Y: Bayesian risk crossed threshold at week −4 (4 weeks before outbreak). XGBoost never crossed threshold. Outcome: Bayesian provides 4-week warning; XGBoost provides no warning."

---

## 5. Validation & Robustness Checks

### 5.1 Sensitivity to Threshold Choices

Recompute lead time using:

1. Outbreak threshold: 70th, 75th, 80th, 90th percentiles.
2. Bayesian threshold: 70th, 75th, 80th percentiles.
3. XGBoost threshold: 0.3, 0.5, 0.7.

**Output:** Table showing that conclusions are robust (e.g., "Bayesian earlier" holds across most threshold combinations).

---

### 5.2 Stratification by Outbreak Magnitude

Separate analysis by outbreak size:

- **Small outbreaks:** peak \( Y_t \) in [80th percentile, 95th percentile].
- **Large outbreaks:** peak \( Y_t \) in [95th percentile, 100th percentile].

**Hypothesis:** Bayesian may have larger lead time for large outbreaks (clearer signal).

---

### 5.3 False Alarm Rate During Lead-Time Window

For weeks before outbreak threshold is crossed:

- How many weeks did Bayesian cross threshold (false alarms)?
- How many weeks did XGBoost cross threshold (false alarms)?

**Metric:** Average # of false positives per district per year.

**Interpretation:** "Bayesian warns earlier, but does it over-alarm?"

---

## 6. Implementation Pseudocode

```python
def compute_lead_time(district, fold, bayesian_posterior, xgboost_probs, cases, thresholds):
    """
    Args:
        bayesian_posterior: array of shape (n_weeks, n_samples)
        xgboost_probs: array of shape (n_weeks,)
        cases: array of shape (n_weeks,)
        thresholds: dict with keys 'outbreak', 'bayesian', 'xgboost'
    
    Returns:
        lead_times_dict: {'bayesian': L_B, 'xgboost': L_X, 'differential': Delta_L, ...}
    """
    
    # Identify outbreak weeks
    outbreak_weeks = np.where(cases > thresholds['outbreak'])[0]
    
    # Compute posterior mean for Bayesian model
    bayesian_mean = np.mean(bayesian_posterior, axis=1)
    
    # For each outbreak episode
    lead_times = []
    for t_star in outbreak_weeks:
        # Find first crossing before or at t_star
        bayesian_crossing = np.where((bayesian_mean[:t_star+1] > thresholds['bayesian']))[0]
        xgboost_crossing = np.where((xgboost_probs[:t_star+1] > thresholds['xgboost']))[0]
        
        t_B = bayesian_crossing[-1] if len(bayesian_crossing) > 0 else t_star + 1
        t_X = xgboost_crossing[-1] if len(xgboost_crossing) > 0 else t_star + 1
        
        L_B = t_star - t_B
        L_X = t_star - t_X
        
        lead_times.append({
            'outbreak_week': t_star,
            'bayesian_lead_time': L_B,
            'xgboost_lead_time': L_X,
            'differential': L_X - L_B
        })
    
    return lead_times
```

---

## 7. Expected Findings & Interpretation

### 7.1 Optimistic Scenario

**Finding:**  
Bayesian median lead time: 2–3 weeks.  
XGBoost median lead time: −0.5 weeks (mostly misses or triggers too late).

**Interpretation:**

> "Even though Bayesian classification accuracy is lower, it detects outbreak escalation 2–3 weeks earlier on average. This validates that latent risk inference is better suited for early warning."

---

### 7.2 Conservative Scenario

**Finding:**  
Bayesian median lead time: 0–1 week.  
XGBoost median lead time: 0 weeks.

**Interpretation:**

> "Both models struggle with lead time, but Bayesian is slightly more conservative (lower false alarms, marginal early warning). This suggests outbreak emergence is highly nonlinear and difficult to predict far in advance. However, Bayesian's calibration (next analysis) offers more principled decision-making."

---

### 7.3 Negative Scenario

**Finding:**  
XGBoost has better or equal lead time.

**Mitigation:**

- This would contradict Claim 2.
- **Do NOT hide it.** Instead:
  - Investigate why (feature correlations? threshold choice?).
  - Restate that Bayesian strength is **calibration & uncertainty**, not lead time.
  - Claim 2 becomes "conditional": "Lead time may be comparable, but uncertainty enables safer decisions."
  - Move emphasis to Calibration (Workstream B).

---

## 8. What This Analysis Does NOT Do

❌ Does NOT claim Bayesian is "better" in absolute terms.  
❌ Does NOT optimize thresholds to inflate lead time.  
❌ Does NOT require new models or retraining.  
❌ Does NOT assume perfect outbreak labels (acknowledges label noise).  

---

## 9. Success Criteria

You will consider this workstream complete when:

✅ Lead time computed for all outbreak episodes in all test folds.  
✅ Numeric table (overall, per-fold, per-district) produced.  
✅ Distribution plots generated (histogram, violin, case studies).  
✅ Robustness checks show conclusions are stable across reasonable threshold choices.  
✅ False alarm analysis shows Bayesian does not over-alarm during lead-time window.  
✅ Interpretation aligns with Claim 2 (or gracefully degrades if negative finding).

---

## 10. Integration with Thesis

**Section in thesis:** Results → "Lead-Time Analysis" (2–3 pages).

**Figures to include:**

1. Lead-time distribution plot (histogram/violin).
2. Differential lead-time plot.
3. 1–2 case study time-series plots.

**Text to include:**

1. Definition of outbreak week, risk crossing, lead time.
2. Summary table (median lead time + IQR + % early warned).
3. Per-fold breakdown if variation is notable.
4. Interpretation: validation of Claim 2.

---

## 11. References & Context

- Existing CV strategy: `05_experiments.md` (v1), Section 5.2
- Bayesian model specification: `03_tdd.md` (v1), Section 3.4.2
- Observation model (Negative Binomial): `03_tdd.md` (v1), Section 3.1

---

**Document Version:** 2.0  
**Status:** Ready for implementation  
**Priority:** CRITICAL (start here)  
**Last Updated:** February 3, 2026
