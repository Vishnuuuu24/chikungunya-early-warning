# 2. PRODUCT & PROBLEM REQUIREMENTS DOCUMENT (PRD)

**Project Name:** Chikungunya Early Warning & Decision System (India)

**Version:** 0.1

**Last Updated:** January 2026

---

## 2.1 Executive Summary

This PRD defines *what* the chikungunya early warning system must do, from user and stakeholder perspectives. It specifies inputs, outputs, functional and non-functional requirements, and success criteria. It does *not* specify how (that's in the TDD).

---

## 2.2 Use Cases

### Use Case 1: Weekly Risk Assessment
**Actor:** State surveillance officer (IDSP), state health department.

**Flow:**
1. Every Monday morning, new EpiClim data (cases, climate from previous week) is available.
2. Surveillance officer runs the system (or it auto-runs).
3. System generates: district-by-district risk scores + alerts for next 2–4 weeks.
4. Officer reviews a risk map; identifies high-risk districts.
5. Officer decides: should we increase surveillance? Start vector control? Alert district teams?

**Acceptance:**
- System must run in < 5 minutes (fresh data → risk scores).
- Risk estimates must be intuitive (e.g., probabilities 0–1, not abstract scores).
- Uncertainty must be visible (so officer knows when to be cautious).

### Use Case 2: Outbreak Investigation & Learning
**Actor:** Epidemiologist, research team.

**Flow:**
1. After an outbreak, team wants to understand: did the model predict it? How far ahead?
2. Team queries historical model outputs + true case data for that district/time.
3. System provides: lead time (how many weeks early), false alarms (false positives in retrospect).
4. Team uses this to validate model and improve future versions.

**Acceptance:**
- Historical predictions must be reproducible.
- Metrics must be well-defined and auditable.

### Use Case 3: Policy Briefing
**Actor:** State health secretary, ministry officials.

**Flow:**
1. Officials ask: "Is chikungunya under control? Where are the risks?"
2. System generates: national risk map, state-by-state summaries, trend plots.
3. Officials see: where risks are highest, are they increasing or decreasing, what % of population is at risk.

**Acceptance:**
- Summary statistics must be interpretable without technical background.
- Visualisations must support policy-level decisions (e.g., "allocate resources to top 5 highest-risk states").

---

## 2.3 Functional Requirements

### FR1: Data Ingestion
- **Requirement:** System must accept weekly chikungunya case counts (district, week, cases).
- **Format:** CSV, JSON, or direct database query (EpiClim).
- **Frequency:** Weekly (data arrives on Mondays or specified day).
- **Handling missing:** If a district has no data for a week, impute or flag (not error-out).

### FR2: Climate Data Integration
- **Requirement:** System must fetch or ingest weekly climate variables (temperature, rainfall, humidity) by district.
- **Source:** EpiClim, IMD, or equivalent.
- **Lags:** Must support multi-week lags (1–8 weeks back) to capture mosquito development delays.

### FR3: Feature Engineering
- **Requirement:** System must automatically compute features from raw case + climate data.
- **Features include:**
  - Case-based: lag-1, lag-2, lag-4, rolling means, growth rates, variance, autocorrelation.
  - Climate-based: lagged temperature, rainfall, degree-days above threshold.
  - Early-warning: trend slopes, variance spikes, skewness changes.
- **Optional spatial:** neighboring district incidence (if data available).

### FR4: Model Training & Inference
- **Requirement:** System must support multiple model families (baselines + Bayesian).
  - Baselines: threshold rules, logistic regression, Poisson, Random Forest, XGBoost.
  - Main: Hierarchical Bayesian state-space model.
- **Training:** Must update models weekly or on-demand (retrain on latest data).
- **Inference:** Must generate risk probabilities for each district for next H weeks (H = 2, 3, or 4).

### FR5: Risk Quantification & Uncertainty
- **Requirement:** For each district-week, output:
  - Point estimate: P(outbreak in next H weeks) ∈ [0, 1].
  - Uncertainty: 95% credible interval or equivalent confidence band.
  - Reasoning: feature importance or latent state interpretation.
- **Bayesian models:** Credible intervals from posterior samples.
- **Non-Bayesian models:** Calibration-based confidence bands or bootstrap.

### FR6: Decision Support
- **Requirement:** System must map risk + uncertainty → recommended action tier.
- **Tiers:**
  - **Tier 1 (Routine):** P(outbreak) < 0.3; continue normal monitoring.
  - **Tier 2 (Enhanced Surveillance):** 0.3 ≤ P < 0.6 or high uncertainty; increase lab testing, hotline monitoring.
  - **Tier 3 (Vector Control):** 0.6 ≤ P < 0.8, low uncertainty; deploy insecticide spraying, community awareness.
  - **Tier 4 (Emergency):** P(outbreak) ≥ 0.8, low uncertainty; mobilise all resources, consider public advisories.
- **Rationale:** Must be based on cost–loss framework (not arbitrary).

### FR7: Explainability
- **Requirement:** For each risk prediction, system must provide explanation in plain language.
- **Example:** "District A is at moderate risk (P = 0.62) because: recent cases trending up (+15% last 2 weeks), temperature optimal for mosquitoes (30°C avg), rainfall abundant. Confidence is moderate due to noise in case data."
- **Audience:** Non-technical public health staff.

### FR8: Reproducibility & Auditability
- **Requirement:**
  - All model parameters, hyperparameters, and data versions must be logged.
  - Same input data must always produce same output (reproducible).
  - Historical run can be re-run and checked.

### FR9: Validation & Comparison
- **Requirement:** System must support model comparison on held-out test data.
- **Metrics:** AUC, F1, sensitivity, specificity, lead time (weeks), false-alarm rate, Brier score, calibration curves.
- **Output:** Comparison tables and plots showing which model works best.

### FR10: Scalability Across Geographies
- **Requirement:** System must work at district level for all of India (28 states + 8 UTs, ~700 districts).
- **Bonus:** Must be adaptable to other countries (e.g., Brazil validation dataset).

---

## 2.4 Non-Functional Requirements

### NFR1: Performance
- **Model training:** Complete in < 1 hour on standard laptop (for weekly retraining).
- **Inference (risk prediction):** Generate risk for all districts + 4-week horizons in < 5 minutes.
- **Bayesian inference (MCMC):** Must complete in < 2 hours (acceptable for overnight batch jobs).

### NFR2: Interpretability
- **Principle:** Models should be understandable by epidemiologists, not just ML engineers.
- **Examples:**
  - Feature importance plots (which signals drove the prediction?).
  - Latent state visualization (how does risk evolve over time?).
  - Mechanistic explanation (climate → mosquitoes → risk chain must be visible).

### NFR3: Reliability & Robustness
- **Principle:** System must handle noisy, incomplete, or inconsistent data gracefully.
- **Examples:**
  - Missing climate data for a week: interpolate or use prior week's value.
  - Anomalous case count spike: flag as potential reporting error, don't blindly overreact.
  - Rare geographies: use hierarchical pooling to share information (don't train separate model per district).

### NFR4: Modularity & Extensibility
- **Principle:** Each block (data, features, models, decisions) must be loosely coupled.
- **Examples:**
  - Easy to swap one baseline model for another without changing rest of pipeline.
  - Easy to add a new feature (e.g., new climate variable) without rewriting everything.
  - Easy to add a new model (e.g., LSTM variant) with minimal code changes.

### NFR5: Documentation
- **Principle:** Code must be self-documenting; architecture must be clear.
- **Examples:**
  - Every function has docstring (input, output, assumptions).
  - Configuration files specify hyperparameters (not hard-coded).
  - Architecture diagrams exist.

### NFR6: Reproducibility in ML Sense
- **Principle:** Same code + same data + same random seed → identical output.
- **Implementation:**
  - Fix random seeds in all models.
  - Log all hyperparameters.
  - Version data snapshots.

---

## 2.5 Constraints & Assumptions

### Constraints
- **Data availability:** EpiClim case data has ~1–2 week reporting delay (acceptable).
- **Computational:** Bayesian MCMC fitting is slow; must use overnight batch jobs.
- **Geographic:** India has 28 states + 8 UTs; not all have equal surveillance coverage (some report more reliably than others).
- **Budget:** No real-time API or custom infrastructure (use free/open-source libraries).

### Assumptions
- **Data quality:** EpiClim and IMD data are reasonably accurate (with normal noise/bias).
- **Stationarity:** Historical outbreak patterns will continue (no major regime shifts due to vaccination or climate change over 1–2 year horizon).
- **User adoption:** State surveillance officers will use the system if it's simple and reliable.
- **Causality:** Climate variables have a causal or strongly predictive relationship with chikungunya (backed by literature).

---

## 2.6 Success Criteria (Measurable)

| Criterion | Target | Metric | Stakeholder |
|-----------|--------|--------|-------------|
| **Lead time** | ≥ 2 weeks | Median weeks between model alert and peak cases | Epi officer |
| **Sensitivity** | ≥ 80% | % of true outbreaks detected by model | Epi officer |
| **Specificity** | ≥ 70% | % of non-outbreak weeks correctly classified | Epi officer |
| **False alarm rate** | < 20% | % of model alerts that do not materialize | Epi officer |
| **Calibration** | Brier < 0.25 | Model predicted probability matches actual frequency | Researchers |
| **Runtime (train)** | < 1 hour | Time to retrain all models on fresh data | Engineer |
| **Runtime (infer)** | < 5 min | Time to generate risk for all districts + 4 weeks | Engineer |
| **Interpretability** | ≥ 80% | % of epidemiologists who understand explanation | Users |
| **Transferability** | Works on Brazil | Model trained on India; tested on Brazil data | Researchers |
| **Code quality** | Coverage > 80% | Unit test coverage | Engineer |

---

## 2.7 Out-of-Scope (For This Phase)

- Real-time web dashboard or API deployment.
- Multi-disease modeling (dengue, Zika, etc. simultaneously).
- Forecast horizon > 4 weeks.
- Detailed cost–loss economic analysis.
- Integration with IDSP or EpiClim databases (manual export/import acceptable).

---

## 2.8 Glossary

| Term | Definition |
|------|-----------|
| **EpiClim** | India disease surveillance database with weekly cases + climate, 2009–present. |
| **IDSP** | Integrated Disease Surveillance Program; official case reporting system in India. |
| **IMD** | India Meteorological Department; provides climate data. |
| **Outbreak** | Elevated case counts (e.g., > 75th historical percentile) in a district for a week. |
| **Lead time** | Number of weeks the model detects elevated risk before cases actually spike. |
| **Latent risk** | Hidden variable: true transmission intensity, inferred from climate + cases. |
| **State-space model** | Probabilistic model with latent state evolving over time + observations noisy function of state. |
| **Hierarchical model** | Model where parameters are shared across regions via global priors (partial pooling). |
| **Mechanistic** | Model explicitly encodes known cause-effect relationships (e.g., climate → mosquitoes → risk). |

---

## 2.9 How This PRD Evolves

This PRD is **v0.1** and reflects the current understanding. As implementation progresses:
- If a requirement is infeasible, log it as a note with rationale (don't just silently drop it).
- If new requirements emerge, add them with version bump (v0.2, v0.3, etc.).
- Any major change (e.g., shifting from district to state level) requires re-review with faculty.

---

**Next Step:** Read `03_tdd.md` for technical design (models, features, architecture).
