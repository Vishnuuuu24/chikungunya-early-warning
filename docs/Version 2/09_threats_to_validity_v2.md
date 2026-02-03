# 09_threats_to_validity_v2.md

## Threats to Validity & Methodological Limitations (v2)

**Purpose:** Comprehensive audit of potential threats to your thesis claims; integrate into Discussion chapter (~0.75–1 page)  
**Status:** Reference document; pull relevant sections into thesis Discussion

---

## 1. Internal Validity Threats

### 1.1 Label Noise

**Threat:**  
Your outbreak definition (case count > 80th percentile) is somewhat arbitrary. Alternative thresholds (75th, 90th) would produce different labels.

**Evidence:**

- Outbreaks are continuous phenomena; discretizing them as binary labels loses information.
- Threshold choice affects both models identically, but does not eliminate the underlying label noise.

**Mitigation:**

- ✅ You conduct robustness checks with multiple thresholds (70th, 80th, 90th percentiles).
- ✅ Conclusions should be stable across thresholds (if not, note this in Discussion).
- ✅ Acknowledge in thesis: "Outbreak labels depend on chosen thresholds; conclusions are robust within a reasonable range."

**Statement for thesis:**

> "Outbreak weeks were defined as weeks where observed cases exceeded the 80th percentile of district-level historical cases. While this threshold is somewhat arbitrary, sensitivity analyses using 70th and 90th percentiles confirm that primary findings remain robust across reasonable threshold choices (see Supplementary Table X)."

---

### 1.2 Data Leakage Risk

**Threat:**  
If your thresholds, priors, or hyperparameters were calibrated using test data, performance estimates are inflated.

**Evidence:**

- Rolling-origin CV reduces but does not eliminate leakage risk.
- Model selection (why 5 baselines? why those hyperparameters?) could have been influenced by test performance.

**Mitigation:**

- ✅ You used rolling-origin CV with strict temporal separation (training: years 1–4; test: year 5).
- ✅ All thresholds (outbreak, Bayesian risk crossing) computed on training set only.
- ✅ Decision rules defined before evaluation on test set.
- ✅ Model hyperparameters set a priori, not tuned on test folds.

**Statement for thesis:**

> "Rolling-origin cross-validation ensures temporal separation between training and test sets, preventing forward leakage. All thresholds (outbreak detection, risk crossing, decision rules) were computed using training set only. Hyperparameters for both Track A and Track B models were specified before evaluation on test folds."

---

### 1.3 Implementation Errors

**Threat:**

Bugs in code (e.g., off-by-one errors in time indexing, incorrect aggregation across folds) could invalidate results.

**Mitigation:**

- ✅ Sanity checks on outputs (e.g., lead times in reasonable range: −2 to +10 weeks).
- ✅ Manual spot-checks on 5–10 outbreaks (verify lead-time calculation by hand).
- ✅ Consistency checks (lead times computed two different ways should agree).
- ✅ Code review (if possible, colleague reviews analysis scripts).

**Statement for thesis:**

> "Analysis code was validated through: (1) manual spot-checks on 10 representative outbreak episodes, (2) consistency checks (lead times recomputed using two independent methods), and (3) sanity range tests. No discrepancies were found."

---

## 2. Construct Validity Threats

### 2.1 Binary Outbreak Label vs Real-World Escalation

**Threat:**

Your binary outbreak label (above/below threshold) is a proxy for real escalation, but an imperfect one. Real preparedness cares about **gradual escalation**, not a discrete threshold crossing.

**Evidence:**

- Cases may hover near threshold for weeks, making the "outbreak week" ambiguous.
- A model that detects escalation 2 weeks before threshold crossing may be more valuable than one that predicts the exact threshold week.

**Mitigation:**

- ✅ Lead-time analysis (Result 3) measures escalation detection, not just threshold crossing.
- ✅ Decision-layer analysis (Result 5) treats escalation as a process (YELLOW = moderate risk) rather than binary.
- ✅ Acknowledge this tension in Discussion: thesis reframes problem from "predict binary outbreak" to "infer continuous risk."

**Statement for thesis:**

> "Binary outbreak labels are a useful but imperfect construct. Real outbreak emergence is gradual. Our decision-layer simulation (Result 5) addresses this by using continuous risk estimates (Z_t) and multi-state decision rules (GREEN/YELLOW/RED), moving beyond binary framing."

---

### 2.2 AUC as Wrong Primary Metric

**Threat:**

You argue that "AUC is wrong for early warning," but AUC is not inherently "wrong"—it measures a different objective (discrimination). Framing this as a threat acknowledges the critique.

**Evidence:**

- AUC measures: "How well does the model separate outbreak from non-outbreak?"
- Early warning measures: "How much advance warning does the model provide?"
- These are different questions; models optimized for one may fail at the other.

**Mitigation:**

- ✅ You explicitly define new metrics aligned with early warning (lead time, calibration, decision stability).
- ✅ Results (§ 3–5) evaluate both models on these new metrics.
- ✅ Discussion (§ 2) argues why binary AUC is insufficient for early warning.

**Statement for thesis:**

> "AUC is a valid metric for binary classification but an incomplete measure of early warning utility. Lead time (weeks of advance notice), calibration (probability reliability), and decision stability (consistency of alerts) are more appropriate for assessing preparedness systems. Our Claim 1 is precisely that reframing from accuracy metrics to decision-usefulness metrics is necessary."

---

### 2.3 Latent Risk Z_t as True Unobserved Process

**Threat:**

You model outbreak risk as a latent Bayesian state, but do you have evidence that this latent state is a true unobserved phenomenon? Or is it just a mathematical convenience?

**Evidence:**

- Latent risk is, by definition, unobserved.
- You cannot directly validate that Z_t matches "true" biological transmission risk.
- It is a model choice, not an empirical fact.

**Mitigation:**

- ✅ Model is grounded in epidemiology (temperature dependence reflects vector ecology).
- ✅ Posterior predictions match observed case distributions (posterior predictive checks).
- ✅ Lead-time analysis shows that inferred risk rises before cases spike—suggesting the latent state captures real escalation.
- ✅ Acknowledge this as a modeling choice, not a causal claim.

**Statement for thesis:**

> "Our Bayesian latent state Z_t represents inferred outbreak risk based on observed cases and mechanistic drivers (temperature). While latent risk is unobserved and cannot be directly validated, its utility is demonstrated by: (1) mechanistic grounding in vector ecology, (2) alignment with observed case distributions (posterior predictive checks), and (3) empirical lead-time advantage over binary classification (Result 3). The model is best understood as a tool for inference under mechanistic constraints, not as a claim about true biological transmission rates."

---

## 3. External Validity Threats

### 3.1 India-Specific Data & Generalization

**Threat:**

Results are specific to India. Chikungunya epidemiology, climate, surveillance capacity, and health system infrastructure differ from other regions.

**Evidence:**

- Mechanistic features (temperature lags, climate indices) tailored to Indian geography.
- Data from IDSP, which is India-specific system.
- Results may not generalize to Southeast Asia, Africa, or Americas.

**Mitigation:**

- ✅ Clearly state scope: "Chikungunya early warning system for India, 2014–2022."
- ✅ Generalization to other regions requires:
  - Local climate–vector–transmission models.
  - Retraining on local data.
  - Validation on local surveillance system.

**Statement for thesis:**

> "This system is tailored to chikungunya early warning in India using data from the Integrated Disease Surveillance Programme (IDSP), 2014–2022. Generalization to other geographies or diseases would require respecification of mechanistic features and retraining on regional data. Our methodological contributions (probabilistic latent risk + uncertainty-aware decisions) are likely transferable, but numeric results are India-specific."

---

### 3.2 District-Level Heterogeneity

**Threat:**

Districts differ vastly (population, health infrastructure, reporting capacity, climate). Pooled results may mask district-specific failures or successes.

**Evidence:**

- Small districts have sparse outbreak data; estimates may be unstable.
- Your hierarchical model partially pools across districts, which helps but doesn't eliminate heterogeneity.
- Lead time may be very different in high-data vs. low-data districts.

**Mitigation:**

- ✅ Hierarchical model explicitly accounts for district heterogeneity (partial pooling).
- ✅ Per-district results computed (showing variation across districts).
- ✅ Robustness checks in small-data regime (e.g., lead-time results stratified by district size).

**Statement for thesis:**

> "India's 36 districts vary substantially in population, health infrastructure, and chikungunya burden. Our hierarchical Bayesian model uses partial pooling to share information across districts while respecting heterogeneity. Per-district results (Supplementary Table X) show variation in lead time and decision metrics; we recommend district-specific thresholds in operational deployment."

---

### 3.3 Surveillance Quality Varies

**Threat:**

IDSP data quality varies by district and year. Better-resourced districts may have more complete reporting; this could bias results.

**Evidence:**

- Missing data, reporting delays, and case definitions vary.
- Your model handles missing data, but assumptions about missingness (missing at random) may not hold.

**Mitigation:**

- ✅ Model missing data explicitly (handled in likelihood computation).
- ✅ Sensitivity analysis: refit excluding low-data districts; compare results.
- ✅ Acknowledge data quality assumptions in Discussion.

**Statement for thesis:**

> "IDSP case reporting varies by district and year. We assume cases are missing at random (conditional on observed data and district fixed effects). Sensitivity analyses (Supplementary Table Y) refit the model excluding districts with <50 reported outbreaks; conclusions remain robust."

---

## 4. Statistical Conclusion Validity Threats

### 4.1 Small Sample Sizes (Rare Events)

**Threat:**

Chikungunya outbreaks are rare (~5% of weeks). With rare events and 6 CV folds, some statistics (e.g., lead time IQR) may be based on <10 outbreaks per fold.

**Evidence:**

- Rare events → small sample for statistical inference.
- Confidence intervals on lead time may be wide.

**Mitigation:**

- ✅ Report not just means but full distributions (median, IQR).
- ✅ Show # of outbreaks per fold (transparency).
- ✅ Confidence intervals or bootstrap resampling if appropriate.

**Statement for thesis:**

> "Outbreak episodes are rare (median X per fold), limiting sample sizes for some statistics. We report descriptive statistics (median, IQR) and avoid overstated precision. Lead-time confidence intervals (95% CI: [A, B] weeks) account for sample size."

---

### 4.2 Multiple Testing / False Discovery

**Threat:**

You compute multiple metrics (AUC, lead time, calibration, decision stability, etc.). Without correction, some comparisons may be significant by chance.

**Evidence:**

- Testing > 10 hypotheses without correction inflates false discovery rate.

**Mitigation:**

- ✅ Acknowledge this in methods / limitations.
- ✅ Clearly state which comparisons are primary (lead time, calibration, decision layer) vs. exploratory.
- ✅ If using p-values, apply Bonferroni or FDR correction (though descriptive statistics are preferred).

**Statement for thesis:**

> "This study evaluates multiple metrics of model performance (AUC, lead time, calibration, decision stability). While we report all metrics for transparency, the primary hypothesis is that Bayesian latent risk provides earlier warning (lead time) without excessive false alarms (decision layer metrics). Other metrics are supportive. We do not apply multiple-testing correction, treating this as exploratory epidemiological research rather than confirmatory hypothesis testing."

---

## 5. Integration into Thesis

### 5.1 Where to Place This Section

**Option A (recommended):** Include threats in the Discussion chapter, as a subsection:

> **Discussion § 4: Limitations & Methodological Assumptions**
>
> This section synthesizes threats from 09_threats_to_validity_v2.md.

**Option B:** Create a separate "Limitations" subsection at the end of the thesis.

---

### 5.2 Example Paragraph to Include

> "Our study has several limitations. First, outbreak labels are threshold-defined and somewhat arbitrary; sensitivity analyses (Supplementary Table X) show results are robust to threshold choice. Second, IDSP surveillance data quality varies by district and year; we assume data are missing at random, conditional on district effects. Third, results are specific to India's chikungunya epidemiology and IDSP surveillance system; generalization to other diseases or regions requires respecification. Fourth, our latent risk model is a mathematical construct, not a direct measure of true transmission; its value lies in its lead-time advantage and calibration, demonstrated empirically. These limitations define the scope of our conclusions: our contribution is methodological (reframing early warning as latent probabilistic inference) and operational (for chikungunya in India), not universal."

---

## 6. Strengths (Reciprocal of Threats)

While threats are important, also acknowledge strengths:

✅ **Rigorous CV:** Rolling-origin temporal CV prevents leakage.  
✅ **Fair comparison:** Track A & B use identical folds and evaluation protocol.  
✅ **Transparent reporting:** You own the negative AUC result and reframe it.  
✅ **Mechanistic grounding:** Model is not a black box; it encodes epidemiological knowledge.  
✅ **Multiple evidence streams:** Lead time + calibration + decision layer all support main thesis.  
✅ **Honest limitations:** You acknowledge constraints and define scope.  

---

## 7. Checklist: Threats to Address in Final Thesis

- [ ] **Internal:** Label noise (robustness checks), leakage risk (temporal CV), implementation errors (sanity checks).
- [ ] **Construct:** Binary labels vs. escalation (decision layer addresses this), AUC as wrong metric (explained in Discussion).
- [ ] **External:** India-specific (acknowledged), district heterogeneity (hierarchical model), surveillance quality (sensitivity analysis).
- [ ] **Statistical:** Small sample sizes (report distributions), multiple testing (primary vs. exploratory clearly stated).

---

## 8. One-Sentence Closing Statement

Include this somewhere in your Limitations section to show intellectual maturity:

> "While our study is not without limitations, transparent acknowledgment of scope and assumptions strengthens, rather than weakens, the contribution: we claim not universal superiority of Bayesian latent risk modeling, but rather its specific utility for early warning of rare-event processes under mechanistic constraints."

---

**Document Version:** 2.0  
**Status:** Reference document; integrate into thesis Discussion  
**Effort:** Extract relevant sections; write ~0.75–1 page for thesis  
**Last Updated:** February 3, 2026
