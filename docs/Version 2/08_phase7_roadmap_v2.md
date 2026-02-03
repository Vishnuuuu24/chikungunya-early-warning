# 08_phase7_roadmap_v2.md

## Phase 7 Tactical Roadmap (Weeks 1–8)

**Current Status:** February 3, 2026  
**Objective:** Complete analysis + thesis writing by end of Week 8  
**Constraint:** No new models; only analysis & documentation

---

## Week-by-Week Breakdown

### WEEK 1: Lead-Time Analysis Implementation

**Spec:** `02_lead_time_analysis_spec_v2.md`

**Tasks:**

- [ ] **Day 1–2:** Load test folds + fitted models (Bayesian posteriors, XGBoost probabilities).
- [ ] **Day 3–4:** Define outbreak weeks (80th percentile threshold per district, computed on training).
- [ ] **Day 4–5:** Compute Bayesian risk crossings (75th percentile of posterior).
- [ ] **Day 5–6:** Compute XGBoost crossings (threshold = 0.5).
- [ ] **Day 7:** Calculate lead times (all outbreaks in test set).

**Outputs:**

- `lead_times_bayesian.csv` (outbreak_week, crossing_week, lead_time_weeks, district, fold).
- `lead_times_xgboost.csv` (same structure).
- Numeric summary: median, IQR, % early warned (overall + per-fold).

**Validation Checklist:**

- [ ] No rows with NaN or -9999 (missing values).
- [ ] Lead times make sense (check 3–5 outliers manually).
- [ ] Different threshold choices (70th, 75th, 80th, 90th percentiles) show consistent conclusions.

**By end of Week 1:** Results table ready to show advisor.

---

### WEEK 2: Lead-Time Visualization + Case Studies

**Spec:** `02_lead_time_analysis_spec_v2.md`, Sections 4.1–4.3

**Tasks:**

- [ ] **Day 1–2:** Plot 1: Histogram of lead times (Bayesian vs XGBoost, side-by-side).
- [ ] **Day 2:** Plot 2: Differential lead time histogram (\( \Delta L = L_X - L_B \)).
- [ ] **Day 3–4:** Select 2–3 case studies (outbreaks with clear early warning from Bayesian).
- [ ] **Day 4–5:** Plot 3–5: Time-series for each case (Z_t, p_XGB, cases, thresholds, crossings).
- [ ] **Day 6:** Annotate plots with narratives (e.g., "Bayesian warned 3 weeks early").

**Outputs:**

- `lead_time_distribution.png` (histogram).
- `differential_lead_time.png` (histogram).
- `case_study_1.png`, `case_study_2.png`, `case_study_3.png` (time-series).

**Validation:**

- [ ] Plots are publication-ready (labels, legends, captions).
- [ ] Case studies tell a clear story.

**Sensitivity Analysis:**

- [ ] Recompute lead times with 70th, 90th percentiles for outbreak threshold.
- [ ] Recompute with Bayesian threshold = 70th, 80th percentiles.
- [ ] Verify main conclusion holds across thresholds.

**By end of Week 2:** Lead-time manuscript section drafted (2–3 pages).

---

### WEEK 3: Calibration & Uncertainty Analysis

**Spec:** `03_calibration_uncertainty_spec_v2.md`

**Tasks:**

- [ ] **Day 1–2:** Compute Brier score for both XGBoost and Bayesian (overall + per-fold).
- [ ] **Day 2–3:** Generate reliability curves (10 bins, bin-wise observed frequency).
- [ ] **Day 4:** Plot: Side-by-side reliability curves (XGBoost vs Bayesian).
- [ ] **Day 5:** Compute Bayesian credible interval coverage (50%, 68%, 90%, 95% levels).
- [ ] **Day 6:** Plot: Coverage vs nominal (Bayesian only).
- [ ] **Day 7:** Create summary table (Brier score, coverage %, interpretation).

**Outputs:**

- `reliability_curves.png` (side-by-side XGBoost vs Bayesian).
- `credible_interval_coverage.png` (Bayesian).
- `calibration_summary_table.csv` (Brier, coverage, interpretation).

**Validation:**

- [ ] Brier scores make sense (range 0–1).
- [ ] Coverage values ~nominal level (e.g., 90% CI should have ~90% coverage).
- [ ] If Bayesian coverage is low, investigate (e.g., underconfident = wider intervals).

**By end of Week 3:** Calibration section drafted (1–2 pages).

---

### WEEK 4: Decision-Layer Simulation

**Spec:** `04_decision_layer_minimal_spec_v2.md`

**Tasks:**

- [ ] **Day 1–2:** Define decision rules (GREEN/YELLOW/RED thresholds from training set).
- [ ] **Day 2–3:** Apply decision rules to all test weeks.
- [ ] **Day 3–4:** Compute decision metrics (false alarms, missed, stability, delay).
- [ ] **Day 5:** Create decision metrics table (Bayesian vs XGBoost thresholds).
- [ ] **Day 6:** Sensitivity analysis (aggressive vs conservative rules).
- [ ] **Day 7:** Narrative for 1–2 example outbreaks.

**Outputs:**

- `decision_metrics_table.csv` (false alarms %, missed %, stability, delay).
- `sensitivity_analysis.csv` (3 Bayesian rules + XGBoost).
- `decision_flowchart.txt` or image.

**Validation:**

- [ ] False alarm rate + missed outbreak rate make sense (trade-off).
- [ ] Decision stability is interpretable (transitions per week).
- [ ] Example narrative is compelling (shows operational value).

**By end of Week 4:** Decision section drafted (1–2 pages).

---

### WEEK 5 (DECISION POINT): Optional Track B Comparisons

**Spec:** `05_trackB_comparisons_spec_v2.md`

**Status:** OPTIONAL. Only if you have time and energy.

**Decision:**

- **If yes:** Pick one comparison (Comparison A: Mechanistic vs non-mechanistic).
  - Effort: ~3–4 days.
  - Output: Lead time table, trajectory plots, interpretation.
- **If no:** Skip entirely. Move to Week 5 writing.

**If YES (Mechanistic vs Non-Mechanistic):**

- [ ] **Day 1:** Simulate non-mechanistic trajectories (AR(1) without temperature).
- [ ] **Day 2:** Compute lead times for both variants.
- [ ] **Day 3:** Plot trajectories (mechanistic vs non-mechanistic for 2–3 districts).
- [ ] **Day 4:** Create comparison table + interpretation.

**Outputs (if attempted):**

- `trackB_comparison_table.csv` (lead time, spurious alerts, stability: mechanistic vs non-mechanistic).
- `trajectory_comparison_1.png`, `trajectory_comparison_2.png`.

---

### WEEKS 5–6: Thesis Writing — Methods & Results

**Spec:** `06_thesis_methods_outline_v2.md`, `07_results_discussion_outline_v2.md`

**Tasks:**

- [ ] **Week 5 (3–4 days):**
  - Write Methods chapter (5–7 pages).
    - Data & preprocessing.
    - Feature engineering (pull from TDD table).
    - Track A description.
    - Track B model with equations.
    - CV strategy.
    - Metrics table.
  - Self-review: does every section reference appropriate spec doc?

- [ ] **Week 5 (3 days) + Week 6 (1 day):**
  - Write Results chapter (4–6 pages).
    - Result 1: Track A performance table.
    - Result 2: Track B AUC (negative result; own it).
    - Result 3: Lead-time analysis (embed figures from Week 2).
    - Result 4: Calibration (embed figures from Week 3).
    - Result 5: Decision layer (embed table from Week 4).
    - (Optional) Result 6: Track B comparisons (if attempted Week 5).

**Validation:**

- [ ] Methods chapter flows; each subsection has citations/references to specs.
- [ ] Results chapter is evidence-driven; every claim has a figure, table, or numerical citation.
- [ ] Negative Track B result (AUC 0.52) is stated clearly and reframed as "mismatch in objectives," not "model failure."

---

### WEEKS 6–8: Thesis Writing — Discussion & Finalization

**Spec:** `07_results_discussion_outline_v2.md`, `09_threats_to_validity_v2.md`

**Tasks:**

- [ ] **Week 6 (2–3 days):**
  - Write Discussion chapter (4–6 pages).
    - Synthesis of Tracks A & B results.
    - Why binary classification is insufficient (1.5 pages).
    - Optional: Bayesian design choices justified (if Track B comparisons done).
    - Limitations & threats to validity (pull from `09_threats_to_validity_v2.md`).
    - Practical implications & next steps.
    - Conclusion (one memorable sentence).

- [ ] **Week 6 (2–3 days):**
  - Integrate Introduction & Background chapters (if not yet started).
  - Ensure thesis has coherent narrative: problem → methods → results → discussion.

- [ ] **Week 7:**
  - Revise for clarity & flow.
  - Check all citations are formatted correctly.
  - Ensure all figures have captions; all tables have labels.
  - Proofread (spelling, grammar).

- [ ] **Week 8:**
  - Final review with advisor.
  - Incorporate feedback.
  - Submit.

**Validation:**

- [ ] Discussion owns the "Bayesian AUC is low" result and reframes it.
- [ ] Threats to validity (09_threats_to_validity_v2.md) are integrated.
- [ ] Practical implications section suggests deployment path.
- [ ] Final sentence is memorable (the one-liner your defense audience will remember).

---

## Risks & Fallback Plans

### Risk 1: Lead-time analysis shows no clear advantage for Bayesian

**Fallback:**

- Reframe: "Lead time comparable, but Bayesian is better calibrated."
- Emphasize calibration (Week 3) as the win.
- Acknowledge in Discussion that outbreak detection timing may be inherently difficult regardless of model.

---

### Risk 2: Decision-layer metrics show Bayesian has more false alarms

**Fallback:**

- This is actually OK if lead time is significantly better.
- Reframe as sensitivity–specificity trade-off: "Bayesian is more conservative, reducing missed outbreaks at the cost of some false alarms."
- Report false alarm rate explicitly.

---

### Risk 3: Time runs out before Track B comparisons

**Fallback (strongly recommended):**

- **Skip Track B comparisons entirely.** You already have enough (Tracks A & B + calibration + decision layer).
- Note in Discussion: "Internal ablations left for future work."
- Submit without `05_trackB_comparisons_spec_v2.md` content.

---

### Risk 4: Advisor asks for more validation / extra analyses

**Fallback:**

- Stay focused on these 8 weeks. Additional requests can be: (a) incorporated into discussion of future work, or (b) done after submission if time permits.
- Do not add new models during this phase.

---

## Critical Milestones (Non-Negotiable)

- **End of Week 2:** Lead-time analysis complete.
- **End of Week 4:** All three workstreams (lead time, calibration, decision) complete.
- **End of Week 6:** Methods + Results chapters drafted.
- **End of Week 8:** Full thesis submitted.

---

## How to Monitor Progress

**Weekly check-in (for yourself or with advisor):**

- [ ] Outputs from this week are clean + validated.
- [ ] No blockers / technical debt.
- [ ] On track for next week's tasks.

**Mid-point check (after Week 4):**

- [ ] All analysis outputs are publication-ready.
- [ ] Advisor has reviewed and approved direction.
- [ ] Writing starts on schedule (Week 5).

---

## What NOT to Do During These 8 Weeks

❌ Do NOT add new features.  
❌ Do NOT retrain models.  
❌ Do NOT change CV folds.  
❌ Do NOT optimize thresholds (they are illustrative).  
❌ Do NOT add new datasets.  
❌ Do NOT start new experiments.  

---

## Success Criteria (End of Week 8)

✅ Lead-time analysis complete + figures + narrative.  
✅ Calibration analysis complete + figures + table.  
✅ Decision-layer simulation complete + table + scenarios.  
✅ Methods chapter written (5–7 pages).  
✅ Results chapter written (4–6 pages).  
✅ Discussion chapter written (4–6 pages).  
✅ Threats to validity documented.  
✅ Full thesis submitted.  

---

**Document Version:** 2.0  
**Status:** Ready to execute  
**Start Date:** Week of February 3, 2026  
**End Date:** Week of February 24–March 3, 2026  
**Last Updated:** February 3, 2026
