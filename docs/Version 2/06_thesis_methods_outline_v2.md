# 06_thesis_methods_outline_v2.md

## Thesis Methods Chapter Outline (v2)

**Purpose:** Skeleton for your Methods chapter; references existing specs + v1 docs to avoid redundancy  
**Length:** ~5–7 pages  
**Status:** Ready to write by assembling existing material

---

## 1. Track A vs Track B: Conceptual Framing

### Quick intro to set context

Add this early in Methods:

> **Problem framing:**  
> We compare two approaches to outbreak early warning. **Track A** (discriminative) treats outbreak prediction as a supervised classification problem, evaluating models on standard metrics (AUC, precision, recall). **Track B** (probabilistic) frames outbreak risk as a continuous, latent process, evaluating models on calibration, uncertainty, and decision usefulness.
>
> This dual-track comparison is designed to validate that binary classification is insufficient for early warning and that probabilistic latent risk modeling offers tangible advantages.

---

## 2. Methods Structure (with section references)

### 2.1 Data & Preprocessing

**What to write:** 2–3 paragraphs.

**Where to pull from:** `04_data_spec.md` (v1) + your implementation notes.

**Content:**
- Data sources (EpiClim, IDSP, etc.).
- Time period (2014–2022).
- District coverage.
- Case definitions (weekly aggregation, handling missing data).

---

### 2.2 Feature Engineering

**What to write:** 2–3 paragraphs + table.

**Where to pull from:** `03_tdd.md` (v1), Section 3.3 ("Feature Engineering").

**Content:**
- Overview of 37 features.
- Categories: lag features, climate features, spatial features.
- Table: [cite:1]
  - Column 1: Feature name
  - Column 2: Definition
  - Column 3: Category
  - Sort by category

Example table structure:

| Feature | Definition | Category |
|---------|-----------|----------|
| lag_cases_1 | Cases in previous week | Temporal |
| lag_cases_2 | Cases 2 weeks ago | Temporal |
| temp_mean_4w | Mean temperature last 4 weeks | Climate |
| (etc.) | | |

---

### 2.3 Track A: Supervised Baseline Models

**What to write:** 2–3 paragraphs.

**Where to pull from:** `03_tdd.md` (v1), Section 3.4.1.

**Content:**
- Five models: logistic regression, Poisson NB, random forest, XGBoost, threshold-based.
- Hyperparameters (brief; refer to code for details).
- Why each model (diversity of approach).
- Training procedure (same folds for all).

Key sentence:

> "All Track A models were trained to predict binary outbreak labels (1 = case count > 80th percentile, 0 = otherwise) on identical rolling-origin CV folds."

---

### 2.4 Track B: Bayesian Hierarchical State-Space Model

**What to write:** 3–4 paragraphs + equations.

**Where to pull from:** `03_tdd.md` (v1), Section 3.4.2.

**Content:**

#### 2.4.1 Model structure

\[ Z_{i,t} \mid Z_{i,t-1} \sim N(\alpha Z_{i,t-1} + \beta T_{i,t}, \sigma_{\text{proc},i}^2) \]

with priors on \( \alpha, \beta, \sigma_{\text{proc},i} \).

#### 2.4.2 Observation model

\[ Y_{i,t} \mid Z_{i,t} \sim \text{NegBin}(\mu_{i,t}, \phi_i) \]

where \( \mu_{i,t} = \exp(\eta_0 + Z_{i,t}) \) and \( \phi_i \) is the overdispersion parameter.

#### 2.4.3 Hierarchical priors

\[ \alpha_i \sim N(\mu_\alpha, \sigma_\alpha^2) \]
\[ \beta_i \sim N(\mu_\beta, \sigma_\beta^2) \]

District-level parameters share information via group-level hyperpriors.

#### 2.4.4 Interpretation

> "The latent state \( Z_{i,t} \) represents unobserved outbreak risk in district \( i \) at week \( t \). It evolves according to an AR(1) process with temperature forcing, encoding mechanistic knowledge of vector-borne disease dynamics. The observed case counts \( Y_{i,t} \) are noisy, dispersed observations of this latent risk."

---

### 2.5 Cross-Validation Strategy

**What to write:** 2 paragraphs.

**Where to pull from:** `05_experiments.md` (v1), Section 5.2.

**Content:**

- Rolling-origin CV (6 folds: 2017, 2018, ..., 2022).
- Training: years 1–4 of fold; test: year 5.
- Why rolling-origin? (temporal structure, no leakage).
- Ensures fair comparison: Track A & B use identical folds.

Key sentence:

> "Rolling-origin cross-validation was chosen to respect temporal dependence in case counts and to prevent data leakage that would artificially inflate performance estimates."

---

### 2.6 Evaluation Metrics

**What to write:** 2–3 paragraphs + table.

**Where to pull from:** `05_experiments.md` (v1), Section 5.3 + new specs (02, 03, 04).

**Content:**

#### Standard metrics (Track A & B for comparison):

| Metric | Definition | Why Used |
|--------|-----------|----------|
| AUC-ROC | Area under receiver-operator characteristic curve | Discrimination; threshold-independent |
| Precision | TP / (TP + FP) | Positive predictive value; false alarm rate |
| Recall | TP / (TP + FN) | Sensitivity; outbreak detection rate |
| F1 | Harmonic mean of precision & recall | Balanced metric |
| AUPR | Area under precision-recall curve | Better for imbalanced classes |
| Brier score | Mean squared error of probabilities | Calibration quality |

#### New metrics (Track B specific):

- **Lead time** (weeks): From risk crossing to observed outbreak.
- **Credible interval coverage** (%): Empirical vs. nominal coverage.
- **Decision stability** (%): Week-to-week decision state changes.
- **Response delay** (weeks): First action to actual outbreak.

---

## 3. Implementation Notes (Brief)

**What to write:** 1 paragraph.

**Content:**
- Software (Python, PyMC, scikit-learn, XGBoost).
- Computational resources (compute time, memory).
- Code availability (GitHub link, if public).

Example:

> "All models were implemented in Python 3.9 using scikit-learn (v1.0), XGBoost (v1.5), and PyMC (v4.0) for Bayesian inference. MCMC sampling used 4 chains, 2000 iterations per chain, with the first 1000 iterations discarded as burn-in. Posterior diagnostics (R-hat, n_eff) confirmed convergence."

---

## 4. What NOT to Replicate

❌ Do NOT repeat detailed feature definitions (put in table, reference TDD).  
❌ Do NOT rewrite full prior specifications (cite TDD, Section 3.4.2).  
❌ Do NOT include all hyperparameter details (refer to code repository).  
❌ Do NOT describe every model architecture (one paragraph per track is enough).

---

## 5. Revision Checklist

Before writing Results, ensure Methods covers:

- [ ] Data sources and preprocessing clearly explained.
- [ ] All 5 Track A models described (1 sentence each).
- [ ] Bayesian model structure with equations.
- [ ] Why Track B (mechanistic) was chosen.
- [ ] CV strategy and why temporal CV matters.
- [ ] All evaluation metrics defined (standard + new).
- [ ] Reference to existing docs (TDD, specs) to avoid duplication.

---

## 6. Example: How to Handle Equations

**If feature set is large (35+), don't list all:**

> "A total of 37 features were engineered across four categories: temporal lags (10 features: cases 1–10 weeks prior), climate lags (12 features: temperature, rainfall, and derived indices at various windows), spatial covariates (8 features: district-level population, health infrastructure), and temporal interactions (7 features: day-of-year dummies, seasonality). See [cite:X] for complete feature definitions."

[cite:X] = reference to Table in TDD or separate supplementary material.

---

## 7. Integration with Results

After finishing Methods, you will write:

- **Results § 1:** Track A performance table.
- **Results § 2:** Track B AUC result (the "negative" finding).
- **Results § 3:** Lead-time analysis.
- **Results § 4:** Calibration & uncertainty.
- **Results § 5:** Decision-layer evaluation.
- (Optional) **Results § 6:** Track B internal comparisons.

Each Results section flows naturally from a corresponding Methods subsection.

---

**Document Version:** 2.0  
**Status:** Outline ready; write by assembling references  
**Length Target:** 5–7 pages  
**Last Updated:** February 3, 2026
