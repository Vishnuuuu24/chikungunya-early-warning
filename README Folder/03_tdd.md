# 3. TECHNICAL DESIGN DOCUMENT (TDD)

**Project Name:** Chikungunya Early Warning & Decision System (India)

**Version:** 0.1

**Last Updated:** January 2026

---

## 3.1 Problem Formalization

### 3.1.1 Prediction Task (Exact)

**Unit:** District $i$, Week $t$

**Input:**
- Historical case counts: $C_{i,1}, C_{i,2}, \ldots, C_{i,t}$ (last $L$ weeks)
- Historical climate: $\mathbf{X}_{i,1}, \mathbf{X}_{i,2}, \ldots, \mathbf{X}_{i,t}$ where $\mathbf{X}_{i,t} = [\text{temp}, \text{rainfall}, \text{humidity}, \ldots]$
- Population: $P_i$

**Output:**
- Probability that district $i$ will be in "outbreak state" at week $t + H$:
  $$\hat{Y}_{i,t+H} = P(\text{outbreak at } t+H | \text{data up to } t)$$
- 95% Credible Interval (or confidence band): $[\hat{Y}^{lo}, \hat{Y}^{hi}]$
- Feature importance / explanation: which signals drove this prediction?

**Parameters:**
- $L$ = lookback window (typical: 12 weeks)
- $H$ = prediction horizon (typical: 2, 3, or 4 weeks ahead)
- Threshold for "outbreak": depends on task (see Section 3.1.3)

### 3.1.2 Notation

| Symbol | Meaning |
|--------|---------|
| $C_{i,t}$ | Case count in district $i$, week $t$ |
| $c_{i,t}$ | Incidence rate (cases per 100k population) in district $i$, week $t$ |
| $\mathbf{X}_{i,t}$ | Climate vector (temp, rainfall, etc.) for district $i$, week $t$ |
| $\mathbf{F}_{i,t}$ | Engineered feature vector for district $i$, week $t$ |
| $Y_{i,t}$ | Binary outbreak label (1 = outbreak, 0 = normal) at district $i$, week $t$ |
| $Z_{i,t}$ | Latent outbreak risk (continuous, 0–1 scale) for district $i$, week $t$ |
| $L$ | Lookback window (weeks of history used as input) |
| $H$ | Prediction horizon (weeks ahead to predict) |

### 3.1.3 Outbreak Definition (Label Convention)

**Primary definition (for Track A, supervised models):**

For week $t$ in district $i$:
$$Y_{i,t} = \begin{cases} 1 & \text{if } c_{i,t} > p_{75}(c_{i,\text{historical}}) \\ 0 & \text{otherwise} \end{cases}$$

where $p_{75}$ = 75th percentile of historical incidence in that district.

**Rationale:** Use percentile-based threshold to adapt to each district's baseline (high-endemic vs low-endemic).

**Alternative (if data permits):** Expert-labeled outbreak weeks from IDSP reports (validation).

**Track B (Bayesian) note:** Latent risk $Z_{i,t}$ is continuous; outbreak threshold applied only for evaluation, not training.

---

## 3.2 Data Flow & Pipeline (Block Diagram Description)

```
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 1: DATA ACQUISITION & INTEGRATION                         │
├─────────────────────────────────────────────────────────────────┤
│ Inputs:                                                          │
│  • EpiClim (weekly cases + climate, all India, 2009–2022)       │
│  • IDSP outbreak reports (validation reference)                 │
│  • India census 2011 (population by district)                   │
│  • District shapefiles (coordinates, neighbors)                 │
│  • [Optional] Google Trends (search volume index)               │
│                                                                  │
│ Processing:                                                     │
│  1. Load EpiClim; filter disease == "Chikungunya"              │
│  2. For each district, extract: (week, cases, temp, rain, ...) │
│  3. Merge population data → compute incidence per 100k          │
│  4. For spatial features: compute neighbor incidence (adjacency)│
│  5. Handle missing: forward-fill cases, interpolate climate     │
│                                                                  │
│ Output:                                                         │
│  Clean panel: (district, week, cases, incidence, climate, ...)  │
│  Stored as: data/processed/panel_chikungunya.parquet            │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 2: FEATURE ENGINEERING                                    │
├─────────────────────────────────────────────────────────────────┤
│ Input: Clean panel from Block 1                                 │
│                                                                  │
│ Feature Categories:                                             │
│                                                                  │
│ A) CASE-BASED FEATURES:                                         │
│    • Lags: c_lag_1, c_lag_2, c_lag_4, c_lag_8 (incidence)      │
│    • Moving avg: c_ma_2w, c_ma_4w (2-week, 4-week averages)   │
│    • Growth rate: (c_t - c_{t-1}) / c_{t-1} [clipped/bounded] │
│    • Variance (rolling): var_c_4w (variance last 4 weeks)      │
│    • Autocorrelation: acf_lag1_4w (lag-1 acf in 4-week window)│
│    • Trend: slope of linear fit over last 4 weeks              │
│    • Skewness: skew_c_4w (how right-skewed are recent cases?) │
│                                                                  │
│ B) CLIMATE MECHANISTIC FEATURES:                                │
│    • Lagged temp: temp_lag_1, temp_lag_2, ..., temp_lag_8     │
│    • Degree-days: sum(max(0, temp_t - 20°C)) over last 2 weeks│
│      (Aedes development threshold ≈ 20°C)                       │
│    • Lagged rainfall: rain_lag_1, ..., rain_lag_8              │
│    • Rainfall persistence: sum of rain last 4 weeks            │
│    • Temp anomaly: temp_t - historical_mean_temp[month]        │
│    • Humidity (if available): humidity_lag_1, ..., lag_4       │
│                                                                  │
│ C) EARLY-WARNING INDICATORS (NEW/UNIQUE):                       │
│    • Variance spike: var_c_4w / mean(var_c_52w_prior)         │
│    • Acf shift: change in acf_lag1 week-on-week               │
│    • Trend acceleration: d/dt of 4-week slope                  │
│    • Recent avg incidence: c_ma_2w normalized by historical    │
│                                                                  │
│ D) [OPTIONAL] SPATIAL FEATURES:                                 │
│    • Neighbors avg: mean(incidence) of adjacent districts       │
│    • Max neighbor: max incidence among 3 nearest districts      │
│    • Spatial lag (CAR-style): weighted neighbor incidence       │
│                                                                  │
│ E) [OPTIONAL] WEAK SIGNALS:                                     │
│    • Google Trends: search volume for "chikungunya" + region   │
│    • Trend of search volume (increasing? decreasing?)           │
│                                                                  │
│ Output:                                                         │
│  Feature matrix: (n_samples, n_features)                        │
│  Stored as: data/processed/features_engineering.parquet         │
│  Feature names: CSV describing each feature                     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 3A: MODEL PREPARATION (TRACK A & TRACK B)                 │
├─────────────────────────────────────────────────────────────────┤
│ Create labeled samples:                                         │
│  • For each (district, week): window of L weeks history → X     │
│    Next H weeks → Y = 1 if outbreak in [t+1, t+H], else 0     │
│  • Shape: (n_samples=all_districts × all_weeks, n_features)   │
│  • Store: data/processed/samples_supervised.parquet             │
│                                                                  │
│ Generate temporal CV splits (rolling-origin):                   │
│  • Fold 1: train 2010–2016, test 2017                          │
│  • Fold 2: train 2010–2017, test 2018                          │
│  • Fold 3: train 2010–2018, test 2019                          │
│  • Fold 4: train 2010–2019, test 2020                          │
│  • (No leakage; temporal order preserved)                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
┌─────────────────────┐       ┌─────────────────────┐
│ BLOCK 3A: TRACK A   │       │ BLOCK 3B: TRACK B   │
│ BASELINES (5 models)│       │ BAYESIAN (1 main)   │
└─────────────────────┘       └─────────────────────┘
         (parallel folds)           (serial MCMC)
        ↓                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 4: VALIDATION & COMPARISON                                │
├─────────────────────────────────────────────────────────────────┤
│ For each fold:                                                  │
│  • Train all Track A models + Track B model separately          │
│  • Evaluate on held-out test weeks                              │
│  • Compute: AUC, F1, sensitivity, specificity, lead time        │
│    Brier score, calibration, false alarm rate                  │
│  • Store results: results/fold_X_metrics.csv                    │
│                                                                  │
│ Aggregate across folds:                                         │
│  • Mean ± SD for each metric per model                          │
│  • Create comparison table                                      │
│  • Rank models by lead time + calibration                       │
│ Output:                                                         │
│  results/comparison_table.csv                                   │
│  results/plots/auc_curves.png, etc.                             │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 5: DECISION LAYER & OPERATIONALIZATION                    │
├─────────────────────────────────────────────────────────────────┤
│ For the chosen best model (likely Bayesian):                   │
│  1. Generate risk for all districts, next H weeks              │
│  2. Assign uncertainty intervals                               │
│  3. Apply cost–loss framework to determine alert thresholds    │
│  4. Map (probability, uncertainty) → action tier               │
│  5. Generate interpretable summary per district                │
│                                                                  │
│ Output:                                                         │
│  Risk map visualization (all districts, color-coded)            │
│  Decision table: (district, week, P_risk, uncertainty, action)  │
│  Briefing document (state-level summaries)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3.3 Feature Engineering Details

### 3.3.1 Feature Categories & Engineering

**Table: Feature Engineering Specification**

| Feature Category | Feature Name | Computation | Window | Why | Unique? |
|---|---|---|---|---|---|
| **Case Lags** | c_lag_1 | Incidence at t-1 | — | Previous week direct signal | No |
| | c_lag_2 | Incidence at t-2 | — | Slightly older signal | No |
| | c_lag_4 | Incidence at t-4 | — | Month-back reference | No |
| | c_lag_8 | Incidence at t-8 | — | 2-month back; seasonality | No |
| **Case Moving Avg** | c_ma_2w | mean(c_{t-1:t}) | 2 weeks | Smooth noise | No |
| | c_ma_4w | mean(c_{t-3:t}) | 4 weeks | Longer-term trend | No |
| **Case Growth** | c_growth_rate | (c_t - c_{t-1}) / (c_{t-1} + ε) | — | Acceleration; bounded ∈ [-1, 1] | No |
| **Case Variance** | var_c_4w | var(c_{t-3:t}) | 4 weeks | Volatility increase = early warning | **YES** |
| **Case Autocorr** | acf_lag1_4w | lag-1 autocorr(c_{t-3:t}) | 4 weeks | Loss of autocorr = transition? | **YES** |
| **Case Trend** | trend_c_4w | slope(t, c_{t-3:t}) | 4 weeks | Direction of change | **Partial** |
| **Case Skewness** | skew_c_4w | skewness(c_{t-3:t}) | 4 weeks | Distribution shape change | **YES** |
| **Temp Lags** | temp_lag_1 to temp_lag_8 | Mean temp at t-k | — | Known mosquito sensitivity | No |
| **Degree-Days** | dd_above_20 | sum(max(0, T_t - 20)) | 2 weeks | Aedes development (mech.) | **YES** |
| **Rainfall Lags** | rain_lag_1 to rain_lag_8 | Total rainfall at t-k | — | Vector habitat | No |
| **Rain Persist** | rain_persist_4w | sum(rain_{t-3:t}) | 4 weeks | Accumulated; habitat | **YES** |
| **Temp Anomaly** | temp_anom | T_t - mean(T[historical, same month]) | — | Deviation from norm | **YES** |
| **Var Spike Ratio** | var_spike_4w | var_c_4w / mean(var_c_52w_prior) | 4w vs 52w | Relative increase = alert | **YES** |
| **ACF Change** | acf_change | acf_lag1_4w - acf_lag1_4w_prior | — | Sudden loss of persistence? | **YES** |
| **Trend Accel** | trend_accel_4w | slope(t, slopes_4w_prior) | Rolling | Rate of trend change | **YES** |
| **Spatial Neighbor** | neighbor_avg_inc | mean(c_neighbors) / P_neighbors | — | Contagion signal | **YES** |

**Key: Unique = not typically used in standard chikungunya EWS papers (from literature review). These are the research contribution.**

---

## 3.4 Model Zoo

### 3.4.1 Track A: Supervised Baselines (5 Models)

#### Model A1: Threshold Rule (EWARS-style)

**What it does:**  
Triggers alert if: $c_{i,t} > \text{mean} + 2 \times \text{SD}$ (or percentile-based threshold).

**Form:**
$$\hat{Y}_{i,t} = \begin{cases} 1 & \text{if } c_{i,t} > \mu + k \sigma \\ 0 & \text{otherwise} \end{cases}$$

**Parameters:** $k$ (multiplier; typically 1.5–2.5)

**Role:** Naive baseline; shows how much smarter models gain over simple rules.

**Input/Output:**
- Input: single feature (case count)
- Output: binary alert (0 or 1)

---

#### Model A2: Logistic Regression

**What it does:**  
Linear combination of features → logistic transformation → probability.

**Form:**
$$P(\text{outbreak}) = \sigma(\beta_0 + \sum_j \beta_j F_{i,t,j})$$

where $\sigma(z) = \frac{1}{1+e^{-z}}$ (sigmoid).

**Parameters:** $\beta_0, \beta_1, \ldots, \beta_{n_f}$ (learned via max-likelihood)

**Role:** Interpretable baseline; coefficients show feature importance.

**Input/Output:**
- Input: feature vector $\mathbf{F}_{i,t}$ (all features)
- Output: probability ∈ [0, 1]

**Hyperparameters to tune:** Regularization (L1 / L2 / ElasticNet); $C$ parameter.

---

#### Model A3: Poisson / Negative Binomial Regression

**What it does:**  
Model future case *counts* (not binary outbreak).

**Form (Poisson):**
$$\mathbb{E}[C_{i,t+H} | \mathbf{F}_{i,t}] = \exp(\beta_0 + \sum_j \beta_j F_{i,t,j})$$

**Form (Negative Binomial):**  
Poisson + overdispersion parameter $\phi$ to handle variance > mean.

**Role:** Count-based prediction; natural for epidemiological data.

**Input/Output:**
- Input: feature vector
- Output: expected count; convert to probability (count > threshold = outbreak)

**Parameters:** $\beta$ coefficients, overdispersion $\phi$.

---

#### Model A4: Random Forest

**What it does:**  
Train many decision trees; average their predictions.

**Form:**
$$\hat{Y}_{i,t} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{F}_{i,t})$$

where $B$ = number of trees, $T_b$ = decision tree $b$.

**Role:** Non-linear baseline; robust; produces feature importance.

**Input/Output:**
- Input: feature vector
- Output: probability (average of tree votes)

**Hyperparameters:** $B$ (n_trees; typical 100–500), max_depth, min_samples_leaf, etc.

---

#### Model A5: XGBoost / LightGBM

**What it does:**  
Gradient-boosted trees; sequentially fit residuals of previous trees.

**Form (simplified):**
$$\hat{Y}_{i,t} = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{F}_{i,t})$$

where $h_m$ = weak learner (shallow tree), $\gamma_m$ = step size, $M$ = iterations.

**Role:** Often top accuracy on tabular data; modern ML baseline.

**Input/Output:**
- Input: feature vector
- Output: probability

**Hyperparameters:** max_depth, learning_rate, n_estimators, subsample, colsample_bytree, regularization (L1/L2).

---

### 3.4.2 Track B: Bayesian Hierarchical State-Space Model (Main)

**Core Idea:**  
- Latent risk $Z_{i,t}$ evolves over time (state equation).  
- Observed cases + features depend on $Z_{i,t}$ (observation equation).  
- Each district has own parameters, but they're partially pooled via priors.

#### State Equation (Latent Risk Dynamics)

$$Z_{i,t} | Z_{i,t-1}, \mathbf{X}_{i,t} \sim \mathcal{N}(\mu_{i,t}, \sigma_Z^2)$$

where:
$$\mu_{i,t} = \rho_i Z_{i,t-1} + \sum_j \alpha_{i,j} X_{i,t,j} + \tau_i$$

**Parameters:**
- $\rho_i$ = temporal autocorrelation for district $i$ (how much risk persists)  
- $\alpha_{i,j}$ = sensitivity of risk to feature $j$ in district $i$  
- $\tau_i$ = baseline risk level for district $i$  
- $\sigma_Z$ = process noise (how much randomness in risk evolution)

**Interpretation:**  
Latent risk drifts based on previous risk + current conditions (climate, cases, etc.) + noise.

#### Observation Equation (Linking Latent to Observed Cases)

$$C_{i,t} | Z_{i,t} \sim \text{Poisson}(\lambda_{i,t})$$

where:
$$\log \lambda_{i,t} = \gamma_i + \beta_i Z_{i,t} + \text{reporting noise}$$

**Parameters:**
- $\gamma_i$ = baseline case reporting rate for district $i$  
- $\beta_i$ = elasticity of cases to latent risk  
- Observation noise captures reporting delays and errors.

**Interpretation:**  
Observed cases are a noisy, lagged manifestation of latent risk.

#### Hierarchical Priors (Partial Pooling Across Districts)

For each district parameter (e.g., $\rho_i$):

$$\rho_i | \mu_\rho, \sigma_\rho \sim \mathcal{N}(\mu_\rho, \sigma_\rho^2)$$

$$\mu_\rho \sim \mathcal{N}(0.7, 0.1^2)$$ (prior: risk is persistent)

$$\sigma_\rho \sim \text{HalfNormal}(0.1)$$ (prior: some variation across districts, but not wild)

**Interpretation:**  
Each district learns from its own data, but is regularized toward the population average. Rarely-affected districts borrow strength from heavily-affected ones.

#### Plate Diagram (Conceptual)

```
         ┌──────────────────────┐
         │  Global Hyperpriors  │
         │  μ_ρ, σ_ρ, μ_α, ... │
         └────────────┬─────────┘
                      │
         ┌────────────▼──────────────┐
         │  District Parameters      │
         │  ╭─ ρ_i, α_i, τ_i        │  ← For each district i
         │  ├─ γ_i, β_i             │
         │  └─ (8–10 params each)   │
         └────────────┬──────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │  Time-Series Latent Risk Z_i,t     │  ← For each district i, week t
    │  ├─ Z_i,1 ~ prior                  │     (hundreds to thousands)
    │  ├─ Z_i,2 | Z_i,1 ~ state eqn.    │
    │  ├─ Z_i,3 | Z_i,2 ~ state eqn.    │
    │  └─ ...                            │
    └────────────┬──────────────────────┘
                 │
    ┌────────────▼─────────────────┐
    │  Observations: Cases C_i,t   │  ← Linked via obs. eqn.
    │  C_i,1 | Z_i,1 ~ Poisson    │
    │  C_i,2 | Z_i,2 ~ Poisson    │
    │  ...                         │
    └──────────────────────────────┘
```

#### Inference: MCMC Sampling

Use Stan (CmdStanPy) or PyMC3 to sample from posterior:

$$p(Z, \theta | C, \mathbf{X}) \propto p(C | Z, \theta) \cdot p(Z | \theta) \cdot p(\theta)$$

**Output:**  
- Posterior samples of $Z_{i,t}$ for all (district, week).  
- Posterior samples of parameters $\rho_i, \alpha_i, etc.$  
- From samples, compute:
  - Mean estimate $\hat{Z}_{i,t}$
  - 95% credible interval $[Z^{lo}_{i,t}, Z^{hi}_{i,t}]$
  - Probability that $Z_{i,t} > \text{threshold}$ (outbreak risk)

---

### 3.4.3 Track B Optional: Sequence Models (LSTM / CNN)

If exploring deep learning variants:

#### Model B1: LSTM (Long Short-Term Memory)

**What it does:**  
Processes time-series sequentially; maintains hidden state.

**Architecture:**
```
Input: (batch_size, seq_len=L, n_features)
       e.g., (32, 12, 35) = 32 districts, last 12 weeks, 35 features

LSTM Layer 1:
  └─ input_size=35, hidden_size=64, bidirectional=False
  └─ output: (batch_size, seq_len, 64)

LSTM Layer 2:
  └─ input_size=64, hidden_size=32
  └─ output: (batch_size, seq_len, 32)

Dense Layers:
  ├─ flatten / take last timestep: (batch_size, 32)
  ├─ Dense(16, ReLU)
  ├─ Dropout(0.3)
  └─ Dense(1, Sigmoid) → probability ∈ [0, 1]

Total params: ~2–5k
```

**Hyperparameters:** hidden_size, n_layers, dropout, learning_rate, batch_size.

---

#### Model B2: 1D CNN (Temporal Convolution)

**What it does:**  
Slides filters over time-axis to detect local patterns (e.g., spikes, trends).

**Architecture:**
```
Input: (batch_size, seq_len=L, n_features) = (32, 12, 35)

Conv1D Layers:
  ├─ filters=32, kernel_size=3, padding='same', ReLU
  │  └─ output: (batch_size, 12, 32)
  ├─ MaxPooling1D(pool_size=2)
  │  └─ output: (batch_size, 6, 32)
  ├─ filters=16, kernel_size=3, ReLU
  │  └─ output: (batch_size, 6, 16)
  └─ GlobalAveragePooling1D()
     └─ output: (batch_size, 16)

Dense Layers:
  ├─ Dense(32, ReLU)
  ├─ Dropout(0.3)
  └─ Dense(1, Sigmoid)

Total params: ~1–2k
```

**Hyperparameters:** n_filters, kernel_size, pooling strategy, learning_rate.

---

#### Model B3: CNN + LSTM (Hybrid)

**What it does:**  
CNN extracts local patterns; LSTM captures long-term dependencies.

**Architecture:**
```
Input: (batch_size, 12, 35)

Conv1D + Pooling (as above):
  └─ output: (batch_size, 6, 16)

LSTM:
  ├─ input_size=16, hidden_size=32
  └─ output: (batch_size, 6, 32)

Dense:
  ├─ flatten / last timestep: (batch_size, 32)
  ├─ Dense(16, ReLU)
  ├─ Dropout(0.3)
  └─ Dense(1, Sigmoid)

Total params: ~2–3k
```

**Rationale:** CNN fast at local patterns; LSTM robust at long-term context.

---

## 3.5 Training & Optimization Details

### 3.5.1 Loss Functions

| Model Type | Loss | Why |
|---|---|---|
| Logistic, RF, XGB | Binary Cross-Entropy (Log Loss) | Standard for classification |
| Poisson / NegBin | Poisson Deviance / Negative Log-Likelihood | Count-based; natural |
| LSTM, CNN, Hybrid | Binary Cross-Entropy + L2 Regularization | Classification; prevent overfit |
| Bayesian | Negative Log-Posterior (via MCMC) | Marginalizes over parameters |

### 3.5.2 Optimizers & Learning Schedules

| Model | Optimizer | Learning Rate | Schedule |
|---|---|---|---|
| Logistic | LBFGS / SGD | 0.01 (adaptive) | None (convergence-based) |
| RF / XGB | N/A (tree-based) | N/A | Tree-wise greedy |
| LSTM / CNN | Adam | 0.001 | Decay: ×0.5 if no improvement (patience=5) |
| Bayesian | HMC / NUTS (Stan) | N/A (adaptive) | Tuned step-size auto |

### 3.5.3 Regularization

| Model | Technique | Typical Value |
|---|---|---|
| Logistic | L2 penalty | $\lambda = 0.01$ |
| RF | max_depth, min_samples_leaf | depth ≤ 15, samples ≥ 5 |
| XGB | L1 + L2 + tree regularization | α=0.01, λ=1.0 |
| LSTM / CNN | Dropout | 0.2–0.3 |
| Bayesian | Prior variance | Set to keep posterior reasonable |

---

## 3.6 Uncertainty Quantification

### 3.6.1 Bayesian Models (Natural)

Posterior samples → credible intervals directly.

Example for Bayesian model:
- Draw 4 chains × 2000 samples = 8000 samples of $Z_{i,t}$.
- 95% credible interval: $[Z^{0.025}_{i,t}, Z^{0.975}_{i,t}]$ (2.5th & 97.5th percentiles).
- Point estimate: median or mean of samples.

### 3.6.2 Non-Bayesian Models

#### Calibration Curves + Confidence Bands

For each model, use held-out test data:
- Compute predicted probabilities $\hat{p}_k$ for samples $k=1, \ldots, n_{test}$.
- Bin predictions (e.g., [0–0.1), [0.1–0.2), ..., [0.9–1.0)).
- For each bin, compute actual frequency of outbreak.
- Plot actual vs predicted → calibration curve.
- Confidence band: 95% CI around bin frequencies (binomial).

#### Bootstrap Intervals (Optional)

- Resample training data with replacement; retrain model.
- Repeat 100 times; collect predictions on test set.
- For each test sample, compute 2.5th & 97.5th percentile across bootstrap runs.

---

## 3.7 Mechanistic Features (Emphasis)

**What makes a feature "mechanistic"?**

It directly encodes known biology/physics of chikungunya transmission:

1. **Degree-days above 20°C:**  
   - Aedes development is temperature-dependent (literature).  
   - Models explicitly accumulate heat above threshold.  
   - Not just a lag of temperature; captures the mechanism.

2. **Rainfall persistence (4-week sum):**  
   - Aedes breed in standing water.  
   - Accumulated rainfall is more relevant than single-week rainfall.  
   - Mechanistic: more water → more breeding sites.

3. **Trend acceleration:**  
   - As outbreak approaches, case growth accelerates.  
   - Captures non-linear dynamics; is statistically early-warning.

**Why include mechanistic features?**  
- Interpretability: stakeholders understand "warm + wet weather → risks."  
- Generalization: features encode biology, so they transfer to new data/regions.  
- Robustness: less prone to spurious correlations.

---

## 3.8 Latent Risk (Z_i,t) — Conceptual Summary

**What is Z_{i,t}?**

An unobserved "transmission intensity" dial for district $i$ at week $t$.

**Why latent?**

- We never directly measure how contagious the situation is.  
- We see cases (noisy, delayed) and climate (leading, informative).  
- The model infers $Z_{i,t}$ from these.

**How does it work?**

1. At time $t$, climate suggests conditions are optimal (warm, wet).  
2. Model infers $Z_{i,t}$ is high (latent risk increases).  
3. One to four weeks later, cases rise (observationally confirmed).  
4. By the time cases are visible, the model already signaled high $Z$.

**Decision relevance:**

- If $Z_{i,t}$ is high + uncertainty low → alert now (cases coming in ~2 weeks).  
- If $Z_{i,t}$ is high + uncertainty high → wait and update (conflicting signals; data too noisy).

---

## 3.9 Model Selection & Comparison Strategy

**How do we choose the final model?**

1. **Track A (baselines):** Run all 5 on same folds; record metrics.  
2. **Track B (Bayesian):** Run on same folds; record metrics.  
3. **Comparison table:** AUC, F1, lead time, false alarm rate, Brier score, calibration.  
4. **Choose best:** Pick model with best **trade-off** between:  
   - Lead time (higher is better; must be ≥ 2 weeks).  
   - False alarm rate (lower is better; must be < 20%).  
   - Calibration (Brier score; lower is better).  
   - Interpretability (subjective but important).

**Expected outcome:**  
Bayesian model likely wins on lead time + calibration. Track A models useful as comparators + sanity checks.

---

## 3.10 Version Control & Change Log

**This TDD is v0.1.** As implementation progresses:

| Version | Date | Change | Rationale |
|---------|------|--------|-----------|
| 0.1 | Jan 2026 | Initial design (Bayesian + 5 baselines) | Foundation |
| 0.2 | [future] | Add LSTM/CNN variants | If exploring deep learning |
| 0.3 | [future] | Spatial field modeling (if needed) | If neighboring-district effects important |

**Policy:** Any change to model architecture, features, or CV strategy must be logged here with rationale.

---

**Next Step:** Read `04_data_spec.md` for data formats, column names, and file structure.
