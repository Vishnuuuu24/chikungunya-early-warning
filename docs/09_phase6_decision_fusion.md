# Phase 6: Decision-Theoretic Evaluation & Bayesian-ML Fusion

**Status:** In Progress  
**Last Updated:** January 27, 2026  
**Phase Lead:** Research Team  
**Prerequisites:** Phase 5 Complete (Bayesian CV Results)

---

## Executive Summary

**Phase 6 is NOT about improving AUC.**

Phase 6 proves that **binary classification is the wrong abstraction** for outbreak early warning systems. Instead, it demonstrates that:

1. **Bayesian latent risk provides earlier warning** (lead-time advantage)
2. **Uncertainty-aware decision rules outperform threshold alarms**
3. **Decision-theoretic evaluation is the correct framework**
4. **Bayesian-ML fusion improves operational utility**

This phase shifts evaluation from "which model has higher AUC?" to "which approach makes better decisions under uncertainty?"

---

## Background & Motivation

### The Classification Fallacy

**Problematic framing:** "Predict outbreak (yes/no) for week t"

**Why it's wrong:**
- Outbreak is a **continuous process**, not a binary event
- The "outbreak week" is arbitrary (depends on threshold choice)
- Binary metrics (AUC, accuracy) ignore **temporal dynamics**
- False negatives and false positives have **different costs**

**Correct framing:** "Estimate latent transmission risk Z_t and recommend actions"

### Phase 5 Results (Context)

From `results/metrics/bayesian_cv_results.json`:

```
Bayesian State-Space Model (v3):
- AUC: 0.515 ± 0.209
- Sensitivity: 4.8%
- Specificity: 97.6%
- Brier Score: 0.250
```

**ML Baseline Comparison (v1.1):**

```
XGBoost:
- AUC: 0.759 ± 0.184
- Sensitivity: 16%
- F1: 0.21
```

### The Misinterpretation

❌ **Wrong conclusion:** "Bayesian model failed because AUC = 0.515 < 0.759"

✅ **Correct interpretation:**
- Bayesian model is a **latent risk estimator**, not a binary classifier
- Low AUC reflects conservative probabilistic calibration
- High specificity (97.6%) shows few false alarms
- Model quantifies **uncertainty** (not just point predictions)
- True value lies in **lead time** and **decision quality**

---

## Phase 6 Objectives

### Primary Goals

1. **Lead-Time Analysis (Task 1)**
   - Compute: t_outbreak - t_bayes for each outbreak episode
   - Target: Median lead time ≥ 2 weeks
   - Demonstrate Bayesian warnings precede outbreak onset

2. **Decision-Layer Simulation (Task 3)**
   - Implement uncertainty-aware alert zones (GREEN/YELLOW/RED)
   - Evaluate: False alarms, missed outbreaks, decision stability
   - Cost-loss analysis: Net benefit > 0

3. **Bayesian-ML Fusion (Task 4)**
   - Feature fusion: Inject Bayesian uncertainty into XGBoost
   - Gated fusion: Use Bayesian when risk is high, else ML
   - Target: Improve AUPR, Recall, F1 (not just AUC)

### Secondary Goals

4. **Risk Trajectory Visualization (Task 2)**
   - Show Bayesian latent risk vs observed cases
   - Demonstrate early warning visually
   - Highlight uncertainty bands

5. **Comprehensive Metrics (Task 5)**
   - Report: AUPR, Precision, Recall, F1, Kappa, Brier
   - De-emphasize: Accuracy (misleading for imbalanced data)
   - Compare across all models

---

## Conceptual Framework

### Track A vs Track B

| Dimension | Track A (ML Classifiers) | Track B (Bayesian Risk) |
|-----------|-------------------------|-------------------------|
| **Goal** | Maximize AUC/AUPR | Estimate latent risk Z_t |
| **Output** | Binary prediction | Probability distribution |
| **Uncertainty** | None (point predictions) | Explicit (credible intervals) |
| **Evaluation** | Classification metrics | Lead time + decision quality |
| **Strengths** | Good discrimination | Early warning, uncertainty |
| **Weaknesses** | No uncertainty, reactive | Lower AUC (conservative) |
| **Use Case** | Binary decisions | Graded alert systems |

### Decision Framework

**Alert Zones:**

- **🟢 GREEN (No Action):** P(Z_t > q) < 0.4
  - Action: Routine surveillance
  - Cost: Minimal

- **🟡 YELLOW (Monitor):** 0.4 ≤ P(Z_t > q) < 0.8 OR high uncertainty
  - Action: Enhanced surveillance, sentinel traps
  - Cost: Moderate

- **🔴 RED (Intervene):** P(Z_t > q) ≥ 0.8 AND low uncertainty
  - Action: Vector control, public awareness
  - Cost: High (but justified if outbreak likely)

**Key Insight:** YELLOW zone handles **epistemic uncertainty** — when model is unsure, take precautionary action.

---

## Task Definitions

### Task 1: Lead-Time Analysis

**Objective:** Prove Bayesian warnings arrive before outbreak onset

**Method:**
1. For each outbreak episode (district-year with outbreak=1):
   - t_outbreak = first week cases exceed threshold
   - t_bayes = first week P(Z_t > q) ≥ τ (q=80%, τ=0.8)
   - lead_time = t_outbreak - t_bayes

2. Report:
   - Median lead time (target: ≥ 2 weeks)
   - IQR, min, max
   - Comparison vs XGBoost trigger timing

**Script:** `experiments/06_analyze_lead_time.py`

**Output:** `results/analysis/lead_time_analysis.json`

**Success Criterion:** Median lead time > 0 (Bayesian warns earlier)

---

### Task 2: Risk Trajectory Visualization

**Objective:** Visualize Bayesian latent risk dynamics

**Method:**
1. Select 5–10 representative districts (varying outbreak patterns)
2. For each district, plot:
   - Observed cases (black line)
   - Outbreak threshold (red dashed)
   - Bayesian latent risk Z_t (blue line)
   - 90% credible interval (blue shading)

3. Annotate:
   - Outbreak weeks
   - Bayesian alert triggers
   - Lead time arrows

**Script:** `experiments/07_visualize_risk_trajectories.py`

**Output:** `results/plots/risk_trajectories/*.png`

**Success Criterion:** Visually demonstrate early warning

---

### Task 3: Decision-Layer Simulation

**Objective:** Evaluate uncertainty-aware decision rules

**Method:**
1. Implement alert zone logic (GREEN/YELLOW/RED)
2. Simulate decisions on CV test sets
3. Evaluate:
   - Lead time to intervention (RED before outbreak)
   - False alarms (RED without outbreak)
   - Missed outbreaks (never RED)
   - Decision stability (zone transition frequency)
   - Cost-loss analysis (net benefit)

**Parameters:**
- Risk quantile: q = 80th percentile
- Probability thresholds: 0.4, 0.8
- Uncertainty threshold: CV = 0.5

**Script:** `experiments/08_simulate_decision_layer.py`

**Output:** `results/analysis/decision_simulation.json`

**Success Criterion:** Net benefit > 0 (interventions justified)

---

### Task 4: Bayesian-ML Fusion

**Objective:** Hybrid strategies combining strengths of both tracks

**Fusion Strategies:**

**A. Feature Fusion**
- Inject Bayesian outputs as features into XGBoost:
  - `bayes_latent_risk_mean`
  - `bayes_latent_risk_std` (uncertainty)
  - `bayes_prob_high_risk`
- Train XGBoost with augmented features
- Evaluate: AUPR, Precision, Recall, F1

**B. Gated Decision Fusion**
- If `bayes_prob_high_risk ≥ 0.8` (RED zone):
  - Use Bayesian probability
- Else:
  - Use XGBoost prediction
- Hypothesis: Bayesian handles high-risk, ML handles baseline

**C. Weighted Ensemble**
- `y_pred = α * P_bayes + (1-α) * P_xgboost`
- Grid search α ∈ [0, 1]
- Optimize for AUPR (not AUC)

**Script:** `experiments/09_fusion_experiments.py`

**Output:** `results/analysis/fusion_results.json`

**Success Criterion:** Fusion improves AUPR or Recall over baselines

---

### Task 5: Comprehensive Metrics

**Objective:** Holistic evaluation across all models

**Metrics Reported:**

**Primary (Classification):**
- AUPR (Area Under Precision-Recall Curve)
- Precision
- Recall (Sensitivity)
- F1 Score
- Specificity

**Secondary:**
- AUC (for reference, not optimization)
- Cohen's Kappa
- Brier Score (calibration)

**Decision-Theoretic:**
- Lead time (median, IQR)
- False alarm rate
- Miss rate
- Net benefit (cost-loss)

**NOT REPORTED:**
- Accuracy (misleading: 95% no-outbreak baseline)

**Script:** `experiments/10_comprehensive_metrics.py`

**Output:** `results/analysis/comprehensive_metrics.json`

---

## Literature & Theoretical Justification

### 1. Lead-Time Metrics

**WHO EWARS Guidelines (2017):**
> "Timeliness is more critical than sensitivity for early warning systems. A system detecting 70% of outbreaks 3 weeks early is superior to one detecting 90% at onset."

**Key Citation:**
- WHO. *Early Warning, Alert and Response System (EWARS): Generic Framework.* Geneva, 2017.

### 2. Cost-Loss Decision Theory

**Murphy (1977) — Weather Forecasting:**
- Economic value of forecast depends on **cost-loss ratio** (C/L)
- Binary warnings optimal only when C/L matches climatological base rate
- Probabilistic forecasts superior when C/L varies by user

**Application to Outbreaks:**
- Cost (C): Intervention expenses (vector control, public awareness)
- Loss (L): Outbreak damages (healthcare costs, morbidity, economic disruption)
- If L >> C (high-consequence disease), err on side of intervention

**Key Citations:**
- Murphy, A.H. (1977). "The Value of Climatological, Categorical and Probabilistic Forecasts." *Journal of Applied Meteorology*, 16(11).
- Richardson, D.S. (2000). "Skill and Economic Value of ECMWF Ensemble Prediction System." *QJRMS*, 126.

### 3. Uncertainty-Aware Decision Making

**Gneiting & Raftery (2007) — Probabilistic Forecasting:**
> "Uncertainty quantification is essential for rational decision-making. Systems optimized for binary accuracy under-represent tail risks."

**Application:**
- YELLOW zone (epistemic uncertainty) prevents overconfidence
- Credible intervals guide resource allocation
- Brier score evaluates calibration, not just discrimination

**Key Citation:**
- Gneiting, T., & Raftery, A.E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *JASA*, 102(477).

### 4. Evaluation Beyond AUC

**Saito & Rehmsmeier (2015) — AUPR vs AUC:**
> "For highly imbalanced datasets, AUPR is more informative than AUC. AUC can be misleading when positive class is rare."

**Chikungunya Context:**
- Outbreak weeks: ~5–10% of data
- AUC inflated by trivial negatives
- AUPR focuses on positive class performance

**Key Citation:**
- Saito, T., & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than ROC." *PLOS ONE*, 10(3).

### 5. Bayesian State-Space for Epidemics

**Cori et al. (2013) — EpiEstim:**
- Bayesian estimation of time-varying reproduction number R_t
- Uncertainty quantification via posterior distributions
- Used globally for COVID-19, Ebola, influenza

**Key Citation:**
- Cori, A., et al. (2013). "A New Framework for Estimating R_t." *American Journal of Epidemiology*, 178(9).

---

## Thresholds & Parameter Justification

### Risk Quantile (q = 80%)

**Rationale:**
- 75th percentile (P75) used for outbreak definition (existing practice)
- 80th percentile slightly stricter → reduces false alarms
- Sensitivity analysis: Try 75%, 80%, 85%

**Source:** Consistent with IDSP outbreak thresholds

### Probability Thresholds (0.4, 0.8)

**GREEN → YELLOW (0.4):**
- Below 40% probability: Status quo
- Conservative threshold (low sensitivity cost)

**YELLOW → RED (0.8):**
- 80% probability + low uncertainty triggers intervention
- Aligned with "preponderance of evidence" standard
- Higher threshold acceptable when L >> C

**Source:** Expert elicitation (placeholder — should be calibrated with public health authorities)

### Uncertainty Threshold (CV = 0.5)

**Coefficient of Variation = σ / μ**
- CV > 0.5 indicates high relative uncertainty
- Triggers YELLOW (monitor) even if probability moderate

**Rationale:**
- When model is uncertain, precautionary principle applies
- Prevents overconfident interventions on noisy signals

---

## Expected Outcomes

### Quantitative Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Median lead time | ≥ 2 weeks | Actionable window for vector control |
| Bayesian trigger rate | 60–80% | Should catch most outbreaks |
| False alarm rate (RED) | < 20% | Maintain trust |
| Net benefit | > 0 | Interventions cost-effective |
| Fusion AUPR improvement | ≥ 10% | Hybrid > individual models |

### Qualitative Outcomes

1. **Demonstration:** Binary classification inadequate for EWS
2. **Framework:** Decision-theoretic evaluation template
3. **Insight:** Bayesian uncertainty enables graded responses
4. **Recommendation:** Use fusion (Track A + B) for deployment

---

## Validation Strategy

### Internal Validation (India Data)

- **Method:** Rolling-origin CV (6 folds, 2017–2022)
- **Splits:** Train on years t-5 to t-1, test on year t
- **No data leakage:** Future data never used for training

### Held-Out Validation (Future Phase)

- **Data:** Brazil chikungunya/dengue/Zika (Zenodo)
- **Purpose:** Geographic transferability
- **Hypothesis:** Mechanistic features generalize across regions

---

## Deliverables

### Code Artifacts

1. `experiments/06_analyze_lead_time.py` ✅
2. `experiments/07_visualize_risk_trajectories.py` ✅
3. `experiments/08_simulate_decision_layer.py` ✅
4. `experiments/09_fusion_experiments.py` ✅
5. `experiments/10_comprehensive_metrics.py` ✅

### Analysis Outputs

1. `results/analysis/lead_time_analysis.json`
2. `results/plots/risk_trajectories/*.png`
3. `results/analysis/decision_simulation.json`
4. `results/analysis/fusion_results.json`
5. `results/analysis/comprehensive_metrics.json`

### Documentation Updates

1. `docs/01_overview.md` — Updated goals ✅
2. `docs/06_phase6_decision_fusion.md` — This document ✅
3. `README.md` — Updated phase status (pending)

---

## Execution Order

**Step 1:** Generate comprehensive metrics (no MCMC required)
```bash
python experiments/10_comprehensive_metrics.py
```

**Step 2:** Run lead-time analysis (requires Bayesian fitting)
```bash
python experiments/06_analyze_lead_time.py
```

**Step 3:** Create visualizations
```bash
python experiments/07_visualize_risk_trajectories.py --n-districts 8
```

**Step 4:** Simulate decision layer
```bash
python experiments/08_simulate_decision_layer.py
```

**Step 5:** Run fusion experiments
```bash
python experiments/09_fusion_experiments.py
```

**Note:** Steps 2–5 require MCMC (slow). Run overnight or on cluster.

---

## Common Pitfalls & FAQs

### Q1: Why is Bayesian AUC so low?

**A:** Bayesian model is NOT optimized for binary classification. It estimates **latent risk distribution**, not class labels. Low AUC reflects:
- Conservative probabilistic calibration
- High specificity (few false alarms)
- Uncertainty quantification (not point predictions)

Evaluate on **lead time** and **decision quality**, not AUC.

### Q2: Should we tune Bayesian model to improve AUC?

**A:** NO. The goal is NOT to beat XGBoost on AUC. The goal is to:
- Demonstrate lead-time advantage
- Show decision-theoretic value
- Prove uncertainty quantification matters

Tuning for AUC would destroy the probabilistic properties.

### Q3: What if fusion doesn't improve AUPR?

**A:** Still valuable if it:
- Improves lead time
- Reduces false alarms
- Increases decision stability

AUPR is one metric; decision quality is the priority.

### Q4: How to explain low Bayesian recall (4.8%)?

**A:** Model is **conservative by design**:
- Triggers RED only when P(Z_t > q) ≥ 0.8 AND low uncertainty
- Prevents overreaction to noise
- YELLOW zone (monitor) catches ambiguous cases

Recall should be evaluated on **RED + YELLOW**, not RED alone.

---

## Next Steps (Post-Phase 6)

1. **Faculty Review:** Present Phase 6 results, decision framework
2. **Threshold Calibration:** Elicit cost-loss parameters from public health experts
3. **External Validation:** Apply to Brazil data (transferability)
4. **Deployment Prototype:** Web dashboard with alert zones
5. **Policy Brief:** Translate findings for NCDC/IDSP stakeholders

---

## References

1. WHO (2017). *Early Warning, Alert and Response System (EWARS): Generic Framework.*
2. Murphy, A.H. (1977). "The Value of Climatological, Categorical and Probabilistic Forecasts." *J. Appl. Meteorol.*, 16(11).
3. Gneiting, T., & Raftery, A.E. (2007). "Strictly Proper Scoring Rules." *JASA*, 102(477).
4. Saito, T., & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than ROC." *PLOS ONE*, 10(3).
5. Cori, A., et al. (2013). "A New Framework for Estimating R_t." *Am. J. Epidemiol.*, 178(9).
6. Richardson, D.S. (2000). "Skill and Economic Value of ECMWF Ensemble." *QJRMS*, 126.

---

**Document Owner:** Research Team  
**Last Review:** January 27, 2026  
**Status:** Active Development
