# 5. EXPERIMENT & EVALUATION PROTOCOL DOCUMENT

**Project Name:** Chikungunya Early Warning & Decision System (India)

**Version:** 0.1

**Last Updated:** January 2026

---

## 5.1 Overview

This document specifies how models are evaluated, compared, and validated. It defines:
- Cross-validation strategy (temporal, spatial).
- Metrics for early-warning systems (not just classification accuracy).
- Model comparison and selection procedure.
- Reproducibility and reporting standards.

---

## 5.2 Cross-Validation Strategy

### 5.2.1 Why Temporal CV?

For time-series predictions, standard random k-fold violates temporal order and introduces **data leakage**: test data from the future can influence training.

**Example of leakage:**
- Random split: fold 1 trains on data from weeks 1–300 (mixed years).
- Fold 1 test: weeks 200–250 (mixed years).
- Problem: training data from weeks 251–300 is in the future relative to test weeks 200–250.

**Temporal CV prevents leakage** by respecting time order: train on past, test on future.

### 5.2.2 Primary Strategy: Rolling-Origin (Expanding Window)

**Definition:**

For each year $Y$ in test set:
- **Training set:** All data from 2010 through $Y-1$.
- **Test set:** All data from year $Y$ (entire year; all districts and weeks).
- **Horizon:** Predictions made for weeks in year $Y$; evaluated against actual outcomes in year $Y$ + next H weeks (typically 2–4 weeks into year $Y+1$ to capture delayed cases).

**Example (for India data, 2010–2022):**

| Fold | Train Years | Test Year | Test Start | Test End |
|------|-------------|-----------|------------|----------|
| Fold 2017 | 2010–2016 | 2017 | 2017-01-01 | 2017-12-31 + H weeks |
| Fold 2018 | 2010–2017 | 2018 | 2018-01-01 | 2018-12-31 + H weeks |
| Fold 2019 | 2010–2018 | 2019 | 2019-01-01 | 2019-12-31 + H weeks |
| Fold 2020 | 2010–2019 | 2020 | 2020-01-01 | 2020-12-31 + H weeks |
| Fold 2021 | 2010–2020 | 2021 | 2021-01-01 | 2021-12-31 + H weeks |
| Fold 2022 | 2010–2021 | 2022 | 2022-01-01 | 2022-12-31 + H weeks |

**Why expanding window?**
- Mirrors real-world deployment: at each year, we have all prior history.
- No "future" data leaks backward.
- More realistic for policy decisions.

**Total folds:** 6 (years 2017–2022); minimum 5 years for stability.

### 5.2.3 Alternative: Blocked k-Fold (if expanding window insufficient)

If test period is short or data sparse, use blocked k-fold:

**Definition:**
- Divide time series into K consecutive, non-overlapping blocks (e.g., 5 blocks).
- For each block k:
  - **Training:** blocks 1 to k-1.
  - **Test:** block k.
  - Blocks are consecutive in time; no future leakage.

**Example (5 blocks):**
```
2010–2013    2014–2015    2016–2017    2018–2019    2020–2022
─────────────────────────────────────────────────────────────
   Block 1      Block 2      Block 3      Block 4      Block 5

Fold 1: Train [Block 1], Test [Block 2]
Fold 2: Train [Blocks 1–2], Test [Block 3]
Fold 3: Train [Blocks 1–3], Test [Block 4]
Fold 4: Train [Blocks 1–4], Test [Block 5]
```

**Advantage:** Uses all data efficiently (each block serves as test exactly once).

**Disadvantage:** Earlier folds have less training data; may underestimate model capacity.

**Recommendation for this project:** Use **rolling-origin** (Fold 2017–2022) as primary; report blocked k-fold as sensitivity check.

---

## 5.3 Spatial Considerations

### 5.3.1 District-Level or State-Level?

**Primary:** District-level (700+ districts; more granular, more data).

**Why:**
- Chikungunya shows high spatial heterogeneity (endemic vs non-endemic regions).
- Early warning is operationally most useful at district level (vector control deployed locally).
- EpiClim provides district-level data.

**Optional variant:** State-level aggregation (28 states + 8 UTs = 36 units; fewer data points but more stable estimates). Can be tested in sensitivity analyses.

### 5.3.2 Spatial Hold-Out (Advanced Option)

**Definition:**
For one fold, hold out an entire state or region; train on other states; test on held-out state.

**Purpose:** Test generalization to unseen geographies.

**Example:**
```
Spatial Fold 1: Train on 27 states, test on Kerala.
Spatial Fold 2: Train on 27 states, test on Tamil Nadu.
...
Spatial Fold K: Train on 27 states, test on one state.
```

**Notes:**
- Computationally expensive (K additional model trainings).
- Useful for final validation but not primary CV strategy.
- Recommended only if time permits; secondary importance.

---

## 5.4 Label Definition & Threshold

### 5.4.1 Outbreak Definition (For Evaluation Only)

For each district $i$, week $t$, define:

$$Y_{i,t} = \begin{cases} 1 & \text{if } c_{i,t} > p_{75}(c_{i,\text{historical}}) \\ 0 & \text{otherwise} \end{cases}$$

where:
- $c_{i,t}$ = incidence (cases per 100k) in district $i$, week $t$.
- $p_{75}(c_{i,\text{historical}})$ = 75th percentile of $c_{i,\cdot}$ across all historical weeks in that district.

**Rationale:**
- Percentile-based: adapts to each district's baseline (endemic vs non-endemic).
- 75th percentile: represents elevated, but not extreme (avoids only catching massive outbreaks).
- Consistent across literature (WHO, CDC EWS papers use similar thresholds).

### 5.4.2 Sensitivity Analysis

Test multiple thresholds and report results:

| Threshold | Rationale | When to use |
|-----------|-----------|-------------|
| p_50 (median) | Very sensitive; catches all increases | High-priority zones; don't want to miss |
| p_75 (75th percentile) | Moderate; balanced | Primary definition (this document) |
| p_90 (90th percentile) | Specific; only major spikes | Low-resource settings; can't respond to everything |
| Fixed (e.g., 10 cases/100k) | Absolute threshold | For comparison with policy-defined thresholds |

**Report:** Compare AUC, sensitivity, specificity at each threshold; choose p_75 as primary, report others in appendix.

---

## 5.5 Metrics for Early-Warning Systems

### 5.5.1 Primary Metrics

#### 1. AUC-ROC (Area Under Receiver Operating Characteristic Curve)

**Definition:**
- Plot: True Positive Rate (sensitivity) vs. False Positive Rate (1 - specificity) as classification threshold varies.
- AUC ∈ [0, 1]; higher is better; 0.5 = random guessing.

**Computation:**
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred_proba)
```

**Interpretation:**
- AUC ≥ 0.80: good discrimination.
- AUC ≥ 0.85: strong discrimination.
- AUC < 0.70: poor; reconsider model.

**Why include:** Standard for comparing classifiers; threshold-independent.

---

#### 2. Lead Time (Weeks in Advance)

**Definition:**
- For a true outbreak (Y = 1) in week $t$, at what week $t - \tau$ did the model first predict high risk?
- Lead time = $\tau$ (positive = model precedes truth; negative = model lags).

**Computation:**
```
For each test district-year with outbreak:
  1. Find first week t where Y_t = 1.
  2. Look backward to find first week t-τ where model prediction P_t-τ > threshold.
  3. Record lead_time = τ (in weeks).

Report: median lead time, inter-quartile range.
```

**Interpretation:**
- Lead time ≥ 2 weeks: actionable (authorities can mobilize vector control).
- Lead time < 1 week: less useful (already too late).
- Lead time < 0: model lagged truth (failure).

**Why include:** **Core metric for early warning.** A model with 90% AUC but no lead time is useless operationally.

---

#### 3. False Alarm Rate

**Definition:**
- Fraction of weeks where model predicted outbreak, but no outbreak occurred.

$$\text{FAR} = \frac{\text{# of (district-year) with prediction = 1 but actual = 0}}{\text{# of predictions = 1}}$$

**Interpretation:**
- FAR < 15%: acceptable; stakeholders won't lose trust.
- FAR 15–25%: moderate; requires good explanation.
- FAR > 25%: high; authorities may ignore alerts.

**Why include:** High false alarms erode trust in the system; operationally critical.

---

#### 4. Sensitivity (True Positive Rate / Recall)

**Definition:**
$$\text{Sensitivity} = \frac{\text{# true positives}}{\text{# true positives + # false negatives}}$$

i.e., fraction of actual outbreaks the model detected.

**Interpretation:**
- Sensitivity ≥ 80%: good; catch most outbreaks.
- Sensitivity < 70%: concerning; miss too many.

**Why include:** Don't want the system missing outbreaks.

---

#### 5. Specificity (True Negative Rate)

**Definition:**
$$\text{Specificity} = \frac{\text{# true negatives}}{\text{# true negatives + # false positives}}$$

i.e., fraction of non-outbreak weeks correctly classified.

**Interpretation:**
- Specificity ≥ 70%: acceptable; few false alarms.
- Specificity < 60%: problematic.

**Why include:** Balance with sensitivity; operationally feasible if both balanced.

---

#### 6. Brier Score (Calibration)

**Definition:**
$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2$$

where $p_i$ = predicted probability, $y_i$ = actual outcome (0 or 1).

**Interpretation:**
- Brier ∈ [0, 1]; lower is better.
- Brier = 0.25: well-calibrated for 50–50 problem.
- Brier < 0.20: well-calibrated; probabilities close to actual frequencies.
- Brier > 0.30: poorly calibrated; over/under-confident.

**Why include:** Ensures predicted probabilities match reality. A model with high AUC but poor Brier is overconfident.

---

### 5.5.2 Secondary Metrics

#### F1 Score
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Harmonic mean of precision and recall; useful when class imbalance.

#### PPV (Positive Predictive Value) & NPV (Negative Predictive Value)
- PPV = P(actual outbreak | model predicts outbreak) = precision.
- NPV = P(no outbreak | model predicts no outbreak).

#### Calibration Plots
- For predicted probability bins [0–0.1), [0.1–0.2), ..., [0.9–1.0), compute actual frequency.
- Plot: predicted vs actual.
- Well-calibrated: points on diagonal.

---

## 5.6 Aggregation Across Folds & Regions

### 5.6.1 Averaging Across Folds

For each metric, compute **mean ± SD** across folds:

$$\text{Mean AUC} = \frac{1}{K} \sum_{k=1}^{K} \text{AUC}_k$$

$$\text{SD AUC} = \sqrt{\frac{1}{K} \sum_{k=1}^{K} (\text{AUC}_k - \text{Mean AUC})^2}$$

where K = number of folds (typically 6 for years 2017–2022).

**Report:** Mean ± SD; also report min/max across folds to show stability.

### 5.6.2 Aggregation Across Districts

Lead time and false alarm rate vary by district. Aggregate as:

**Lead time:**
- Per fold-district pair, compute median lead time (across outbreaks in that district-year).
- Aggregate across all fold-district pairs: report median, IQR.

**False alarm rate:**
- Per fold-district pair, compute FAR (false alarms / total alerts).
- Aggregate: report median FAR across all district-year combinations.

### 5.6.3 Aggregation Across Models

Create **comparison table:**

| Model | AUC (mean ± SD) | Lead Time (weeks) | FAR (%) | Brier | Runtime (s) |
|-------|-----------------|------------------|--------|-------|-------------|
| Threshold Rule | 0.65 ± 0.08 | 0.5 | 35% | 0.32 | < 1 |
| Logistic Reg. | 0.72 ± 0.09 | 1.2 | 20% | 0.28 | 2 |
| Poisson Reg. | 0.74 ± 0.08 | 1.5 | 18% | 0.27 | 3 |
| Random Forest | 0.78 ± 0.07 | 1.8 | 22% | 0.25 | 15 |
| XGBoost | 0.82 ± 0.06 | 2.0 | 19% | 0.23 | 45 |
| **Bayesian State-Space** | **0.84 ± 0.06** | **2.3** | **16%** | **0.20** | 180* |

*(180 = wall-clock time for MCMC; parallelizable)*

---

## 5.7 Reproducibility & Reporting

### 5.7.1 Code & Random Seeds

All models must be **deterministic** given the same seed:

```python
# In each model training script:
import random
import numpy as np
import torch  # if using PyTorch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Then train model...
# Result: identical output across runs
```

**Document:** In each script, specify seed and note if model is non-deterministic (some MCMC / bootstrap methods have inherent randomness; report this).

### 5.7.2 Hyperparameter Logging

For each model, save hyperparameters to JSON:

```json
{
  "model": "xgboost",
  "fold": "fold_2017",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0
  },
  "training_time_seconds": 45.3,
  "n_samples_train": 8500,
  "n_samples_test": 1450
}
```

### 5.7.3 Results Reporting

For each fold and model, save predictions + metrics:

**predictions_fold_{year}_model_{name}.csv:**
```
district_id,week,year,true_label,pred_prob,pred_binary,lead_time,uncertainty
230,10,2017,1,0.78,1,2.5,0.12
231,10,2017,0,0.22,0,-1,0.18
...
```

**metrics_fold_{year}_model_{name}.json:**
```json
{
  "fold": "fold_2017",
  "model": "xgboost",
  "auc_roc": 0.82,
  "sensitivity": 0.85,
  "specificity": 0.78,
  "false_alarm_rate": 0.19,
  "lead_time_median": 2.0,
  "lead_time_iqr": [1.2, 2.8],
  "brier_score": 0.23,
  "n_true_outbreaks": 45,
  "n_predictions": 1450
}
```

### 5.7.4 Final Comparison Document

Create **`results/comparison_final.md`:**

```markdown
# Model Comparison Results

## Summary Table
[Comparison table as in Section 5.6.3]

## Key Findings
- Best overall model: Bayesian State-Space (AUC 0.84, lead time 2.3 weeks).
- Compared to XGBoost baseline: +0.02 AUC, +0.3 weeks lead time, -3% FAR.
- Bayesian model also provides uncertainty quantification.

## Calibration Analysis
[Plots showing predicted vs actual probability by bin]

## Lead Time Distribution
[Histogram: lead times for all detected outbreaks]

## False Alarm Case Study
[Detailed analysis of false positives; any patterns?]

## Recommendation
Use Bayesian State-Space model for operational deployment.
- Strong discrimination (AUC 0.84).
- Adequate lead time (2.3 weeks, actionable).
- Reasonable false alarm rate (16%).
- Transparent uncertainty; interpretable.
```

---

## 5.8 Statistical Significance Testing

### 5.8.1 When to Use

Compare two models if:
- Performance difference ≥ 0.05 on a metric (e.g., AUC 0.82 vs 0.87).
- Want to know if difference is statistically significant or due to random variation.

### 5.8.2 Methods

**Method 1: Paired t-test (across folds)**
```python
from scipy.stats import ttest_rel

auc_model1 = [0.80, 0.82, 0.81, ...]  # AUC per fold
auc_model2 = [0.84, 0.83, 0.85, ...]

t_stat, p_value = ttest_rel(auc_model1, auc_model2)
print(f"t={t_stat:.3f}, p={p_value:.4f}")

if p_value < 0.05:
    print("Significant difference at α=0.05")
else:
    print("No significant difference")
```

**Method 2: Bootstrap (if few folds)**
```python
from scipy.stats import bootstrap

def auc_diff(x, y):
    return roc_auc_score(y, x) - roc_auc_score(y_alt, x_alt)

# Resample; compute CI
```

**Recommendation:** Use paired t-test (simple, sufficient for 6 folds).

---

## 5.9 Sensitivity Analyses

### 5.9.1 Variables to Test

| Variable | Range | Purpose |
|----------|-------|---------|
| Lookback window $L$ | 8, 12, 16 weeks | How much history needed? |
| Prediction horizon $H$ | 2, 3, 4 weeks | Trade-off: farther ahead = harder |
| Outbreak threshold | p_50, p_75, p_90 | How sensitive to label definition? |
| Feature set | minimal, standard, full | Which features drive performance? |

**Plan:** For each variable, train model and report metrics. Identify which choices matter most.

---

## 5.10 Timeline & Execution

### 5.10.1 Experiment Phases

| Phase | Task | Time | Output |
|-------|------|------|--------|
| Phase 1 | Data prep + feature engineering | 1–2 weeks | `panel_v01.parquet`, `features_v01.parquet` |
| Phase 2 | Train baselines (Track A) | 1–2 weeks | `predictions_*.csv`, `comparison_table.csv` |
| Phase 3 | Develop Bayesian model | 2–3 weeks | Stan code, posterior samples |
| Phase 4 | Fit Bayesian (MCMC) | 1–2 weeks (overnight runs) | `bayesian_posteriors.nc` |
| Phase 5 | Evaluate & compare all models | 1 week | `results_final.md`, plots |
| Phase 6 | Sensitivity analyses | 1 week | Appendix tables |
| Phase 7 | Write-up & documentation | 1–2 weeks | Methods section ready |

**Total:** ~10–14 weeks (can parallelize).

---

## 5.11 How This Protocol Evolves

As experiments progress, update this document:

| Version | Date | Change | Reason |
|---------|------|--------|--------|
| 0.1 | Jan 2026 | Initial protocol (rolling-origin, 6 folds, p_75 threshold) | Baseline |
| 0.2 | [future] | Add spatial hold-out if time permits | Robustness check |
| 0.3 | [future] | Adjust outbreak threshold based on initial results | Operational tuning |

---

**Next Step:** Read `06_playbook.md` for hands-on implementation guide.
