# 01_implementation_plan_v2.md

## Chikungunya Early Warning System — Phase 7 Implementation Plan (v2)

**Current Date:** February 3, 2026  
**Status:** Phase 7 (Analysis & Framing Validation)  
**Scope:** Lead-time analysis, calibration, minimal decision layer, optional Track B comparisons, thesis writing  
**Duration:** Weeks 1–8 (estimated 8 weeks to thesis completion)

---

## Executive Summary

This is **NOT** a new modeling phase. You have completed:

- ✅ Data ingestion and preprocessing
- ✅ Feature engineering (37 features)
- ✅ Track A: 5 supervised baselines (Logistic, RF, XGBoost, etc.)
- ✅ Track B: Bayesian hierarchical state-space model
- ✅ Fair 6-fold rolling-origin CV
- ✅ Honest comparison: XGBoost AUC ~0.76 vs Bayesian AUC ~0.52

**Phase 7 is validation of your framing**, not more models.

You will:

1. Demonstrate that your Bayesian model captures outbreak risk **earlier** (lead-time analysis).
2. Show that your Bayesian model is **better calibrated and uncertainty-aware** (calibration analysis).
3. Prove that uncertainty + risk enables **safer, more stable decisions** (minimal decision layer).
4. Optionally, validate that your modeling **assumptions matter** (Track B internal comparisons).
5. Write a defensible, publication-ready thesis.

---

## Overall Verdict (From External Review)

✅ **YES, this is the right plan**  
✅ **NO, it does not deviate from your original pipeline**  
✅ **YES, it finally reframes success away from "Bayesian beats XGBoost" toward "binary accuracy is the wrong objective"**

This v2 plan is better than your original conceptual plan because:

- It stops pretending the Bayesian model will beat XGBoost on AUC.
- It reframes success correctly around lead time, calibration, and decisions.
- It avoids last-minute scope creep (no new fancy models).

If you execute exactly this, you will have:

- A defensible thesis
- A mature research narrative
- No methodological embarrassment in front of reviewers

---

## Core Thesis Claims (What You Will Validate)

**Claim 1 (Problem framing):**  
Binary outbreak classification is the wrong abstraction for early warning. Outbreak risk is a continuous, hidden, uncertain process.

**Claim 2 (Bayesian advantage):**  
Latent risk inference (Bayesian model) captures outbreak escalation **earlier than** binary classifiers (XGBoost), even if raw classification accuracy is lower.

**Claim 3 (Decision usefulness):**  
Uncertainty-aware risk enables more stable, safer decisions than threshold-based binary predictions.

**Claim 4 (Modeling design):**  
The modeling assumptions we encode (mechanistic knowledge, temporal continuity, hierarchical pooling) measurably improve inference quality (optional, if time permits).

---

## Phase 7 Workstreams

### Workstream A: Lead-Time Analysis

**Objective:** Demonstrate Claim 2.

**What:** For each district-outbreak episode in your test folds (2017–2022):

1. Identify the outbreak week (from labels).
2. Identify the week when latent risk Z_t crosses a chosen threshold.
3. Identify the week when XGBoost probability crosses its threshold.
4. Compute lead time = outbreak week − risk crossing week.
5. Aggregate (median, IQR, per-fold, per-district).
6. Compare Bayesian vs XGBoost.

**Output:**

- Numeric table: median lead time (weeks), IQR.
- Distribution plot (histogram or violin plot).
- 2–3 case studies (narratives of specific districts/outbreaks).

**Effort:** Medium (1–2 weeks to implement + debug).

**Spec document:** `02_lead_time_analysis_spec_v2.md`

---

### Workstream B: Calibration & Uncertainty Sanity Check

**Objective:** Demonstrate Claim 3 (partial) — that Bayesian model is well-calibrated and knows when it is uncertain.

**What:**

1. Reliability diagram (predicted probability vs observed frequency) for both models.
2. Brier score for both models (overall probability calibration).
3. Credible interval coverage for Bayesian model (do 90% intervals contain observed events ~90% of the time?).

**Output:**

- Reliability curves (side-by-side XGBoost vs Bayesian).
- Coverage plot (Bayesian: nominal vs observed coverage).
- Short table: Brier score, coverage %, one-line interpretation.

**Effort:** Light (1 week).

**Spec document:** `03_calibration_uncertainty_spec_v2.md`

---

### Workstream C: Minimal Decision-Layer Simulation

**Objective:** Demonstrate Claim 3 — that uncertainty + risk enables better decisions.

**What:**

1. Define decision rules: GREEN / YELLOW / RED based on risk level + uncertainty.
2. For each week in test folds:
   - Compute decision state from Bayesian model.
   - Compute decision state from XGBoost model.
3. Measure:
   - False alarms (RED when no outbreak actually occurs).
   - Missed outbreaks (No action when outbreak occurs).
   - Decision stability (how often does state flip week-to-week?).
   - Response delay (weeks between first action and actual outbreak).

**Output:**

- Simple table: false alarms, missed, stability, delay (XGBoost vs Bayesian).
- Narrative: why minimal decision rules validate the framing.

**Effort:** Light (1 week).

**Spec document:** `04_decision_layer_minimal_spec_v2.md`

---

### Workstream D: Track B Internal Comparisons (Optional)

**Objective:** Validate Claim 4 (optional, **if time permits**).

**What:** Run 1–2 controlled ablations on the Bayesian model:

- Comparison A: Mechanistic (climate-driven) vs non-mechanistic (climate removed).
- Comparison B: Temporal (state-space dynamics) vs independent (no autocorrelation).

Do NOT re-fit from scratch; use same priors, same data, same CV splits.

**Metrics:** Lead time, credible interval width, smoothness of trajectory.

**Output:**

- Small table: effect of each assumption.
- 1–2 figures (risk trajectories under variant assumptions).

**Effort:** Medium if done; typically skipped if time is tight.

**Spec document:** `05_trackB_comparisons_spec_v2.md`

---

### Workstream E: Thesis Writing

**Objective:** Convert all analyses into a defensible, publication-ready thesis.

**What:**

1. Refine Methods using existing TDD + new specifications.
2. Write Results section combining:
   - Track A performance table.
   - Track B AUC vs binary labels (honest statement of negative result).
   - Lead-time analysis results.
   - Calibration plots.
   - Decision layer table.
3. Write Discussion explaining why Claim 1 is correct.

**Output:**

- Methods chapter (~5–7 pages).
- Results chapter (~4–6 pages + figures).
- Discussion chapter (~4–6 pages).

**Effort:** High, but guided by outlines in `06_thesis_methods_outline_v2.md` and `07_results_discussion_outline_v2.md`.

---

## Milestones & Timeline

| Week | Workstream | Deliverable | Status |
|------|-----------|-------------|--------|
| 1–2 | A (Lead time) | Analysis code + numeric results + plot | Implement |
| 2–3 | B (Calibration) | Reliability curves + Brier score table | Implement |
| 3–4 | C (Decision) | Decision metrics table | Implement |
| 4–5 | D (Optional) | One Track B comparison, if chosen | Implement if time |
| 5–6 | E (Writing) | Methods + Results draft | Write + iterate |
| 6–8 | E (Writing) | Discussion + full thesis | Polish + submit |

---

## Risk & Fallback Plan

### Risk 1: Lead-time analysis is messy (outbreaks rare, many weeks without crossing threshold)

**Mitigation:**

- Pre-compute distribution of outbreak frequencies by district + fold.
- Decide on aggregation level (per-district, per-fold, overall) before implementing.
- If too few outbreaks, report "insufficient data to measure lead time for [district]" instead of forcing a number.

---

### Risk 2: Calibration plots show Bayesian model is poorly calibrated

**Fallback:**

- This would actually strengthen your argument:
  - "Bayesian model is conservative; it reserves high probability for rare events."
  - This conservatism may explain why it triggers later (lower probability), but when it does trigger, it's more reliable.
- Reframe in Discussion as a feature, not a bug.

---

### Risk 3: Track B comparisons take too long

**Fallback:**

- Drop Workstream D (internal comparisons) entirely.
- You already have enough (Tracks A & B + calibration + decision layer).
- Optional label in the thesis: "Internal ablations left for future work."

---

### Risk 4: Time runs out before full thesis writing

**Fallback:**

- Prioritize order: Methods (required) → Results (required) → Discussion (required, but can be shorter).
- Get feedback early; avoid late rewrites.

---

## What You Will NOT Do

❌ Do NOT add more features to Bayesian just to chase AUC.  
❌ Do NOT add deep learning into Track B.  
❌ Do NOT fuse XGBoost + Bayesian (ensemble tricks break interpretability).  
❌ Do NOT re-tune thresholds to inflate metrics.  
❌ Do NOT reframe this as "Bayesian failed"; reframe as "binary framing failed."

---

## Success Criteria

By the end of Phase 7, you will have:

✅ **Quantitative proof that Bayesian model warns earlier** (Workstream A).  
✅ **Evidence that Bayesian uncertainty is meaningful** (Workstream B).  
✅ **Demonstration that uncertainty-aware decisions are stabler** (Workstream C).  
✅ **A complete, defensible thesis** (Workstream E).  
✅ **(Optional) Validation that modeling assumptions matter** (Workstream D, if time).

---

## Key Documentation References

- Existing data/features/models: `03_tdd.md` (v1)
- CV strategy & metrics: `05_experiments.md` (v1)
- New analysis specs:
  - `02_lead_time_analysis_spec_v2.md`
  - `03_calibration_uncertainty_spec_v2.md`
  - `04_decision_layer_minimal_spec_v2.md`
  - `05_trackB_comparisons_spec_v2.md`
- Thesis outlines:
  - `06_thesis_methods_outline_v2.md`
  - `07_results_discussion_outline_v2.md`
- Threats to validity: `09_threats_to_validity_v2.md`

---

## Executive Sign-Off

This plan has been reviewed and approved as:

- Coherent with your actual results.
- Avoiding false claims.
- Preserving novelty and methodological integrity.
- Realistic in scope and timeline.

**Recommendation:** Approve this plan. Do not expand it. Execute it.

---

**Document Version:** 2.0  
**Last Updated:** February 3, 2026  
**Status:** Ready for implementation
