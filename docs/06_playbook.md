# 6. IMPLEMENTATION PLAYBOOK (VS CODE + COPILOT + LLM PROMPTING)

**Project Name:** Chikungunya Early Warning & Decision System (India)

**Version:** 0.1

**Last Updated:** January 2026

---

## 6.1 Overview

This playbook is your **day-to-day guide** for:
- Setting up the project environment (Python, packages, folder structure).
- Writing code using VS Code + Copilot (prompting best practices).
- Running experiments iteratively.
- Debugging and iterating when things don't work (they won't, first time).

It's intentionally **flexible**: Code, models, pipeline will evolve; this playbook adapts as you learn.

---

## 6.2 Environment Setup

### 6.2.1 Python Version & Packages

**Recommended Python:** 3.9 or 3.10 (stable, widely compatible).

**Create Virtual Environment:**
```bash
python3.9 -m venv chikungunya_ews_env
source chikungunya_ews_env/bin/activate  # On Windows: chikungunya_ews_env\Scripts\activate
```

**Install Core Packages:**
```bash
pip install --upgrade pip

# Data & numeric
pip install pandas numpy scipy scikit-learn

# ML & modeling
pip install scikit-learn xgboost lightgbm  # Baselines
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # LSTM/CNN (CPU)
# Or GPU if available: --index-url https://download.pytorch.org/whl/cu118

# Bayesian
pip install pymc arviz  # Bayesian modeling & visualization
pip install cmdstanpy  # Or: pip install stan

# Visualization & notebooks
pip install matplotlib seaborn jupyter scikit-plot

# Geospatial (optional)
pip install geopandas shapely

# Utilities
pip install pyyaml dotenv tqdm
```

**Save to `requirements.txt`:**
```bash
pip freeze > requirements.txt
```

(Later, reinstall via `pip install -r requirements.txt`.)

---

### 6.2.2 Project Folder Structure

```
chikungunya_ews/
├── .env                          # (ignored by git) local paths, API keys
├── .gitignore                    # Exclude data/, results/, .pyc, etc.
├── README.md                     # High-level project intro (for GitHub)
│
├── docs/
│   ├── 01_overview.md
│   ├── 02_prd.md
│   ├── 03_tdd.md
│   ├── 04_data_spec.md
│   ├── 05_experiments.md
│   └── 06_playbook.md (this file)
│
├── config/
│   ├── config_default.yaml       # Default hyperparams, paths
│   └── config_experiment_v01.yaml # Specific experiment variant
│
├── data/
│   ├── raw/
│   │   ├── .gitkeep             # Placeholder (don't commit raw data)
│   │   ├── README.md            # Instructions for downloading raw data
│   │   └── (large files stored locally or S3)
│   └── processed/
│       ├── panel_chikungunya_v01.parquet
│       └── features_engineered_v01.parquet
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Load YAML config
│   ├── logger.py                # Logging setup
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Load raw data (EpiClim, census, etc.)
│   │   ├── cleaner.py           # Data cleaning (imputation, filtering)
│   │   └── merger.py            # Merge datasets → panel
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py       # Compute all features
│   │   └── utils.py             # Helper functions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base model class
│   │   ├── baselines.py         # Logistic, RF, XGB, Poisson, Threshold
│   │   ├── bayesian.py          # Hierarchical Bayesian state-space
│   │   ├── deep_learning.py     # LSTM, CNN, CNN+LSTM (if exploring)
│   │   └── utils.py             # Training loops, predict methods
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # AUC, lead time, FAR, Brier, etc.
│   │   ├── cv.py                # Temporal CV splits
│   │   └── comparison.py        # Compare models, generate tables
│   │
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── threshold.py         # Cost-loss → alert thresholds
│   │   └── actions.py           # Map risk → action tier
│   │
│   └── utils/
│       ├── __init__.py
│       ├── plotting.py          # Visualization helpers
│       └── io.py                # Save/load models, results
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_features.ipynb        # Visualize engineered features
│   ├── 03_baseline_models.ipynb # Quick baseline tests
│   └── 04_results.ipynb         # Final comparison & plots
│
├── experiments/
│   ├── run_cv_fold_2017.py      # Script: run one fold
│   ├── run_cv_all.py            # Script: run all folds
│   ├── run_bayesian.py          # Script: MCMC inference
│   ├── run_comparison.py        # Script: model comparison
│   └── results/
│       ├── predictions_*.csv
│       ├── metrics_*.json
│       └── plots/
│           ├── auc_curves.png
│           └── lead_time_dist.png
│
└── stan_models/
    ├── hierarchical_statespace_v01.stan  # Bayesian model code
    └── hierarchical_statespace_v02.stan  # (Future variants)
```

---

### 6.2.3 Configuration Files

**`config/config_default.yaml`:**

```yaml
# Default configuration for chikungunya EWS

data:
  raw_epiclim: "data/raw/epiclim_chikungunya_2010_2022.csv"
  raw_census: "data/raw/india_census_2011_district.csv"
  processed_panel: "data/processed/panel_chikungunya_v01.parquet"
  processed_features: "data/processed/features_engineered_v01.parquet"

feature_engineering:
  lookback_window: 12          # L: weeks of history
  prediction_horizon: 3        # H: weeks ahead to predict
  case_lags: [1, 2, 4, 8]     # Which lags to include
  temp_lags: [1, 2, 4, 8]     # Climate lags
  rolling_windows: [4, 8]      # For variance, autocorr, etc.

labels:
  outbreak_percentile: 75      # p_75 threshold for label = 1
  min_district_completeness: 0.80  # 80% non-missing required

cv:
  strategy: "rolling_origin"   # or "blocked_kfold"
  test_years: [2017, 2018, 2019, 2020, 2021, 2022]  # Which years to test
  n_folds: 6

models:
  baselines:
    threshold:
      multiplier: 2.0
    logistic:
      C: 0.01
      solver: "lbfgs"
    poisson:
      alpha: 0.01
    rf:
      n_estimators: 100
      max_depth: 15
      min_samples_leaf: 5
    xgboost:
      n_estimators: 100
      max_depth: 5
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 0.8
  
  bayesian:
    n_chains: 4
    n_warmup: 1000
    n_sampling: 2000
    target_accept: 0.95

evaluation:
  metrics: [auc, f1, sensitivity, specificity, lead_time, false_alarm_rate, brier]
  random_seed: 42

output:
  results_dir: "experiments/results"
  plots_dir: "experiments/results/plots"
```

**Load in Python:**
```python
import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_config("config/config_default.yaml")
print(cfg["feature_engineering"]["lookback_window"])  # 12
```

---

## 6.3 Prompting Strategy for VS Code + Copilot

### 6.3.1 Copilot Best Practices

**When VS Code autocomplete / Copilot activates:**

1. **Give context in comments above the function:**
   ```python
   # Load EpiClim CSV; filter for chikungunya; merge with census population
   # Return pandas DataFrame with columns: district_id, week, cases, temp, rainfall, incidence_per_100k
   # Handle missing: forward-fill cases, interpolate climate
   def load_and_clean_epiclim(epiclim_path: str, census_path: str) -> pd.DataFrame:
       """
       ...
       """
       # Copilot reads the docstring + context and suggests the implementation
   ```

2. **Use type hints:**
   ```python
   def compute_case_lags(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
       # Copilot understands input/output; gives better suggestions
   ```

3. **Be specific with variable names:**
   ```python
   # BAD: features = df[...]  (ambiguous)
   # GOOD: case_lags_1_2_4_8 = df[...]  (self-documenting)
   ```

4. **If suggestion is close but not right:**
   - Accept partial suggestion (`Tab`).
   - Manually edit (Copilot does ~70% right; you do the last 30%).
   - Don't blindly accept; read and understand.

---

### 6.3.2 ChatGPT / Claude Prompting (for complex logic)

**When stuck or need to implement something complex:**

**Good Prompt Template:**
```
I'm building a chikungunya early warning system in Python using pandas + scikit-learn.

CONTEXT:
[Briefly: what you're doing]
- I have a DataFrame `df` with columns: district_id, week, cases, temp, rainfall
- I need to: compute features for all (district, week) pairs

TASK:
Create a function that:
1. For each district, compute rolling 4-week variance of case counts
2. Skip weeks with < 8 weeks of prior history (not enough data)
3. Return a new DataFrame with columns: district_id, week, feat_cases_var_4w

CONSTRAINTS:
- Must be efficient (vectorized; no loops over districts)
- Handle NaN properly (forward-fill missing cases)
- Include docstring + type hints

CODE SKELETON (optional; if you have partial code):
```python
def compute_case_variance_4w(df: pd.DataFrame) -> pd.DataFrame:
    # Your attempt / where stuck
    ...
```

EXPECTED OUTPUT FORMAT:
```python
# district_id | week | feat_cases_var_4w
#     10      |  20  |    45.2
#     10      |  21  |    38.1
#    ...
```
```

**Why this works:**
- **CONTEXT:** Model knows your setup; gives pandas-specific advice.
- **TASK:** Clear what you need.
- **CONSTRAINTS:** Helps avoid inefficient solutions.
- **CODE SKELETON:** Helps model continue where you got stuck.
- **OUTPUT FORMAT:** Ensures answer is what you expect.

---

### 6.3.3 Running Experiments with Prompts

**Scenario:** You want to train all baseline models on fold 2017.

**Prompt to ChatGPT:**
```
I have:
- Training data: X_train, y_train (5000 samples, 35 features)
- Test data: X_test, y_test (1000 samples)
- Models to train: LogisticRegression, RandomForest, XGBClassifier, PoissonRegressor

I want to:
1. Train each model with specific hyperparams (from config YAML)
2. Generate predictions on test set
3. Compute metrics: AUC, F1, sensitivity, lead_time
4. Save predictions to CSV
5. Log results to JSON

Can you provide a script structure that does this?
```

**Response** will give you a template; you adapt to your specific hyperparams.

---

## 6.4 Hands-On First Steps

### 6.4.1 Step 1: Data Loading (Week 1)

**Goal:** Load EpiClim data, verify structure, start cleaning.

**Code to write (with Copilot help):**

**File: `src/data/loader.py`**
```python
"""
Load and merge raw data sources (EpiClim, census, shapefiles).
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path

def load_epiclim(path: str) -> pd.DataFrame:
    """Load EpiClim CSV. Filter for chikungunya only."""
    df = pd.read_csv(path)
    df = df[df["disease"] == "Chikungunya"].copy()
    # Copilot will suggest: sort by date, handle dtypes, etc.
    return df

def load_census(path: str) -> pd.DataFrame:
    """Load Census 2011 population data."""
    df = pd.read_csv(path)
    return df[["district_id", "district_name", "population", "area_km2"]]

def merge_data(epiclim: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """Merge EpiClim + census by district."""
    return epiclim.merge(census, on="district_id", how="left")

if __name__ == "__main__":
    # Quick test
    epiclim = load_epiclim("data/raw/epiclim_chikungunya_2010_2022.csv")
    census = load_census("data/raw/india_census_2011_district.csv")
    panel = merge_data(epiclim, census)
    print(f"Loaded {len(panel)} rows, {panel.shape[1]} columns")
    print(panel.head())
    panel.to_parquet("data/processed/panel_chikungunya_v01.parquet")
```

**Run it:**
```bash
cd chikungunya_ews
python src/data/loader.py
```

**Verify:**
- Does parquet file exist?
- Are columns correct?
- Any NaN patterns?

---

### 6.4.2 Step 2: Feature Engineering (Week 2–3)

**Goal:** Compute all features (case lags, climate lags, early-warning indicators).

**Code skeleton (with Copilot):**

**File: `src/features/engineering.py`**
```python
"""
Compute engineered features from raw panel data.
"""
import pandas as pd
import numpy as np
from typing import List

def compute_case_lags(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """
    For each district-week, add columns for case lags.
    E.g., lags=[1, 2, 4, 8] → adds feat_cases_lag_1, ..., feat_cases_lag_8
    """
    df = df.copy()
    for lag in lags:
        df[f"feat_cases_lag_{lag}"] = df.groupby("district_id")["incidence_per_100k"].shift(lag)
    return df

def compute_rolling_variance_4w(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4-week rolling variance of incidence."""
    df = df.copy()
    df["feat_cases_var_4w"] = df.groupby("district_id")["incidence_per_100k"].rolling(
        window=4, min_periods=4
    ).var().reset_index(level=0, drop=True)
    return df

def compute_all_features(panel_path: str, cfg: dict) -> pd.DataFrame:
    """Load panel; compute all features; return feature matrix."""
    df = pd.read_parquet(panel_path)
    
    # Case features
    df = compute_case_lags(df, cfg["feature_engineering"]["case_lags"])
    df = compute_rolling_variance_4w(df)
    
    # Climate features (add similarly)
    # Early-warning features (add similarly)
    
    # Drop rows with NaN in features
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    df = df.dropna(subset=feature_cols)
    
    return df

if __name__ == "__main__":
    import yaml
    with open("config/config_default.yaml") as f:
        cfg = yaml.safe_load(f)
    
    features_df = compute_all_features(cfg["data"]["processed_panel"], cfg)
    print(f"Computed {len(features_df)} samples, {features_df.shape[1]} features")
    features_df.to_parquet(cfg["data"]["processed_features"])
```

**Prompt to ChatGPT if stuck:**
```
Help me write the `compute_climate_lags` function using pandas groupby().shift().
Input: df with columns district_id, week, temp_mean, rainfall_mm
Output: df with new columns feat_temp_lag_1, feat_temp_lag_2, ..., feat_rain_lag_1, etc.
```

---

### 6.4.3 Step 3: Quick Baseline (Week 3–4)

**Goal:** Train one simple model (logistic regression) to verify pipeline works.

**File: `experiments/01_quick_baseline.py`**
```python
"""
Quick test: train logistic regression on fold 2017; check metrics.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import yaml

# Load config
with open("config/config_default.yaml") as f:
    cfg = yaml.safe_load(f)

# Load features
features_df = pd.read_parquet(cfg["data"]["processed_features"])

# Split train/test by year (rolling-origin)
train_df = features_df[features_df["year"] < 2017]
test_df = features_df[features_df["year"] == 2017]

# Feature columns (drop metadata)
feature_cols = [c for c in train_df.columns if c.startswith("feat_")]

# Labels
from src.evaluation.cv import create_labels
train_df = create_labels(train_df, outbreak_percentile=75, horizon=3)
test_df = create_labels(test_df, outbreak_percentile=75, horizon=3)

# Drop NaN
train_df = train_df.dropna(subset=feature_cols + ["label_outbreak"])
test_df = test_df.dropna(subset=feature_cols + ["label_outbreak"])

# Train
X_train = train_df[feature_cols].values
y_train = train_df["label_outbreak"].values
model = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000)
model.fit(X_train, y_train)

# Test
X_test = test_df[feature_cols].values
y_test = test_df["label_outbreak"].values
y_pred = model.predict_proba(X_test)[:, 1]

# Metrics
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred > 0.5)
print(f"Fold 2017: AUC={auc:.3f}, F1={f1:.3f}")
```

**Run:**
```bash
cd chikungunya_ews
python experiments/01_quick_baseline.py
```

**If it works:** You have a working pipeline! Celebrate.

**If it fails:** Debug step-by-step.
- Print shapes: `print(X_train.shape, y_train.shape)`
- Check NaN: `print(X_train.isna().sum())`
- Copilot + ChatGPT to fix.

---

## 6.5 Iterative Development (The Reality)

### 6.5.1 Expect to Iterate

**First implementation:** 60% works.
- Code runs but metrics seem off.
- Features might be wrong.
- CV folds might have leakage.
- Models need tuning.

**Your cycle:**

1. **Run experiment** → output CSV with results.
2. **Plot results** (AUC curve, lead time histogram).
3. **Spot issues** (e.g., "Why is AUC so low?").
4. **Hypothesize cause** (wrong features? too much noise? CV leakage?).
5. **Fix** (adjust feature, retrain, replot).
6. **Repeat 20+ times** until satisfied.

### 6.5.2 Common Issues & Fixes

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| AUC = 0.50 (random) | CV leakage or all-zero labels | Check CV splits; verify label distribution |
| AUC = 0.99 (too good) | Leakage (test features contain future info) | Ensure features use only past data |
| NaN errors | Missing values in features | Check `feature_df.isna().sum()` per column |
| Model runs but metrics empty | Mismatched indices in y_test vs y_pred | Verify shapes match |
| Bayesian sampler slow | Too many parameters or chain issues | Reduce district count for testing; check diagnostics |
| Memory error on MCMC | Data too large | Aggregate to state level or subsample |

---

### 6.5.3 Debugging Workflow (In VS Code)

**Add breakpoints + print statements:**

```python
# In src/features/engineering.py

def compute_case_lags(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    df = df.copy()
    print(f"Input df shape: {df.shape}")
    print(f"Districts: {df['district_id'].nunique()}")
    
    for lag in lags:
        df[f"feat_cases_lag_{lag}"] = df.groupby("district_id")["incidence_per_100k"].shift(lag)
    
    print(f"Output df shape: {df.shape}")
    print(f"NaN in lag_1: {df['feat_cases_lag_1'].isna().sum()}")
    
    return df
```

**Run with debug prints:**
```bash
python -u experiments/01_quick_baseline.py 2>&1 | tee debug_log.txt
```

**Open `debug_log.txt`; trace through logic.**

---

## 6.6 Running Full Experiments (Later)

Once baselines work, scale up:

**File: `experiments/run_cv_all.py`**
```python
"""
Run all CV folds for all models (baseline + Bayesian).
This will take several hours.
"""
import sys
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
with open("config/config_default.yaml") as f:
    cfg = yaml.safe_load(f)

# Load data once
features_df = pd.read_parquet(cfg["data"]["processed_features"])

# Models to train
from src.models.baselines import *
from src.models.bayesian import BayesianStateSpace

models = {
    "threshold": ThresholdRule(cfg),
    "logistic": LogisticRegression(cfg),
    "rf": RandomForest(cfg),
    "xgboost": XGBoost(cfg),
    "bayesian": BayesianStateSpace(cfg),
}

# For each fold
all_results = []
for fold_year in cfg["cv"]["test_years"]:
    logger.info(f"=== FOLD {fold_year} ===")
    
    train_df = features_df[features_df["year"] < fold_year]
    test_df = features_df[features_df["year"] == fold_year]
    
    # Create labels
    train_df = create_labels(train_df, cfg)
    test_df = create_labels(test_df, cfg)
    
    # For each model
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        model.fit(train_df)
        preds = model.predict(test_df)
        metrics = compute_metrics(test_df["label_outbreak"], preds)
        
        result = {
            "fold": fold_year,
            "model": model_name,
            **metrics
        }
        all_results.append(result)
        logger.info(f"  → AUC={metrics['auc']:.3f}, lead_time={metrics['lead_time']:.1f}")

# Aggregate & save
results_df = pd.DataFrame(all_results)
results_df.to_csv(cfg["output"]["results_dir"] / "all_results.csv", index=False)
logger.info("✓ Saved all_results.csv")

# Print summary
print("\n=== SUMMARY ===")
print(results_df.groupby("model")[["auc", "f1", "lead_time"]].mean().round(3))
```

**Run (take your time; can run overnight):**
```bash
python experiments/run_cv_all.py 2>&1 | tee experiments/results/run_log.txt
```

---

## 6.7 Debugging with Copilot in Real-Time

**In VS Code, as you write:**

1. **Open a Python file in experiments/ or src/**
2. **Start typing a function (with a docstring + comment):**
   ```python
   def compute_lead_time(pred_prob, y_true, threshold=0.5):
       """
       For each true outbreak (y_true=1) in test set,
       find how many weeks before the actual outbreak
       the model predicted high probability (pred_prob > threshold).
       
       Return median lead time in weeks.
       """
       # Copilot will suggest an implementation
   ```
3. **Press `Ctrl+K, Ctrl+I`** (or your keybind) to trigger Copilot inline suggestions.
4. **Review suggestion; accept with `Tab` or edit manually.**

---

## 6.8 ChatGPT for High-Level Questions

**When unsure about approach:**

**Prompt:**
```
I'm training a chikungunya outbreak early warning system. My Bayesian model achieved:
- AUC: 0.84
- Lead time: 2.3 weeks
- False alarm rate: 16%

My XGBoost baseline achieved:
- AUC: 0.82
- Lead time: 2.0 weeks
- FAR: 19%

Is the Bayesian model significantly better? How do I decide which to deploy?
```

**Response** will give you decision-making guidance (statistical testing, operational considerations, etc.).

---

## 6.9 Version Control (Git)

**Initialize repo:**
```bash
cd chikungunya_ews
git init
git add .
git commit -m "Initial project structure + docs"
git remote add origin https://github.com/your_username/chikungunya_ews.git
git branch -M main
git push -u origin main
```

**As you iterate:**
```bash
git add src/features/engineering.py
git commit -m "Add degree-days feature; recomputed features_v01"

git add experiments/run_cv_all.py
git commit -m "Run CV fold 2017-2022; baseline AUC 0.78"
```

**.gitignore** (don't commit large files):
```
# Data
data/raw/
data/processed/

# Results
experiments/results/

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Notebooks
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# Environment
*.env
chikungunya_ews_env/
```

---

## 6.10 Documentation as You Go

**Keep a research journal:**

**File: `journal.md`**
```markdown
# Research Journal

## Week 1 (Jan 20–26, 2026)
- Loaded EpiClim data: 8.2M rows, 2010–2022
- Initial panel: 700 districts × 52 weeks/year ≈ 360k district-weeks
- Computed case lags (1,2,4,8) + rolling variance
- Quick baseline (logistic): AUC 0.72 on fold 2017

Issue: lead_time computation had a bug (off-by-one error)
Fix: Corrected week indexing

## Week 2 (Jan 27–Feb 2)
- Added climate mechanistic features (degree-days, rainfall persistence)
- AUC improved to 0.75 on fold 2017
- Ran full CV (2017–2022): mean AUC 0.73 ± 0.04

Next: Implement Bayesian state-space model

## Week 3 (Feb 3–9)
- [Update as you progress]
```

**Why?**
- Remember what you tried (you'll forget).
- Share with advisors/faculty ("Here's what I did last week").
- Useful for thesis/paper writing later.

---

## 6.11 Asking for Help Effectively

**When stuck, ask faculty or collaborate with clear context:**

**Email template:**
```
Subject: Issue with CV fold 2018 — AUC dropped unexpectedly

Hi [Faculty],

I'm debugging my chikungunya EWS implementation. On fold 2017, logistic regression 
achieves AUC 0.75. But on fold 2018, AUC drops to 0.62. This seems wrong.

WHAT I DID:
- Used same config (lookback=12, horizon=3)
- Same CV strategy (rolling-origin, train 2010–2017, test 2018)
- Same feature set

WHAT I OBSERVED:
- Fold 2017 label distribution: 8% outbreaks (balanced-ish)
- Fold 2018 label distribution: 2% outbreaks (very imbalanced)
- Fold 2018 feature distributions look similar to fold 2017

HYPOTHESIS:
Class imbalance (2% vs 8% outbreaks) is hurting model. Or bug in year 2018 data?

QUESTION:
Is class imbalance a known issue? Should I use class_weight='balanced' in logistic regression?
Or should I investigate the 2018 data?

Attached: debug_log_fold_2018.txt, AUC_comparison_plot.png

Thanks,
[Your name]
```

**Why this works:**
- Faculty understands the situation quickly.
- Not vague ("it doesn't work"); specific ("AUC dropped from 0.75 to 0.62").
- You've already hypothesized; faculty can validate or suggest better approach.

---

## 6.12 Evolution: From Baseline to Sophisticated (Timeline)

| Phase | Duration | Work | Output |
|-------|----------|------|--------|
| 1. Setup | 1 week | Env, data loading, quick baseline | `quick_baseline.py` works |
| 2. Features | 2 weeks | Engineer all features; verify distributions | `features_engineered_v01.parquet` |
| 3. Baselines | 2 weeks | Train all 5 baseline models; compare | `comparison_table.csv` |
| 4. Bayesian | 3 weeks | Implement state-space model in Stan; debug MCMC | `bayesian_v01.stan`, posterior samples |
| 5. Evaluation | 1 week | Full CV; compute all metrics; final comparison | `results_final.md`, plots |
| 6. Decision Layer | 1 week | Cost-loss analysis; alert thresholds | `decision_rules.json` |
| 7. Writeup | 2 weeks | Methods section, results plots, conclusions | Thesis chapter ready |

**Total:** ~12–14 weeks (can overlap).

---

## 6.13 Common Gotchas & How to Avoid

| Gotcha | Why it happens | How to avoid |
|--------|---|---|
| Data leakage in features | Using future data to predict past | Always compute features using only past data; verify no forward-looking info |
| Temporal CV not implemented | Forgot to implement temporal splits | Use `rolling_origin_split()` function; never random k-fold |
| NaN explosion | Missing data not handled | Impute early; document imputation strategy; check `isna().sum()` at each step |
| Model overfitting | Too many features, not enough data | Cross-validate; regularize (L2 penalty); monitor val loss |
| Bayesian MCMC slow | Too many parameters | Start with smaller subset; check diagnostics (Rhat, ESS) |
| Results not reproducible | Random seed not set | `np.random.seed(42)`, `torch.manual_seed(42)` at top of script |
| Metrics don't make sense | Computing metrics wrong | Double-check metric definitions; compare your values to sklearn reference |

---

## 6.14 Final Checklist Before Running Experiments

- [ ] Config file exists and is valid YAML.
- [ ] Data paths in config point to correct locations.
- [ ] Random seed is set (for reproducibility).
- [ ] Feature engineering script runs without error.
- [ ] Quick baseline (logistic on fold 2017) achieves sensible AUC (0.60–0.80).
- [ ] CV splits don't have leakage (check: train_year < test_year).
- [ ] Metrics functions tested manually on dummy data.
- [ ] Results directory exists and is writable.
- [ ] Git repo initialized; `.gitignore` configured.
- [ ] Journal.md started with notes.

---

**Next Step:** You're ready to start implementing. Open VS Code, create `src/data/loader.py`, and begin!

---

**End of Playbook (v0.1)**

As you implement, update this playbook with:
- Gotchas you encounter + solutions.
- Copilot/ChatGPT prompts that worked well.
- Commands that are handy.
- Debugging workflows you find effective.

This is a **living document** for the duration of your project.
