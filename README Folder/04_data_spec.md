# 4. DATA SPECIFICATION & SCHEMA DOCUMENT

**Project Name:** Chikungunya Early Warning & Decision System (India)

**Version:** 0.1

**Last Updated:** January 2026

---

## 4.1 Overview

This document specifies the structure, format, and schema of all raw and processed data used in the chikungunya early warning system. It is the source of truth for:
- Where data comes from (URLs, local paths, access methods).
- How data is structured (tables, columns, types).
- How data flows through the pipeline (raw → processed → features).

---

## 4.2 Raw Data Sources

### 4.2.1 EpiClim (Primary Source)

**Description:**  
India epidemiological surveillance + climate database. Contains weekly case counts for multiple diseases (including chikungunya), merged with meteorological variables by district.

**Access:**
- **URL:** https://www.epiclim.org/ (or database dump, if available)
- **License:** Open (check site)
- **Format:** CSV export or database query
- **Time coverage:** 2009 – present (updated weekly)
- **Spatial coverage:** All India districts (700+ as of 2011 Census)

**Key Columns (Raw):**

| Column Name | Type | Description | Example |
|---|---|---|---|
| district_id | int | District code (from Census 2011) | 230 |
| district_name | str | District name | "Bengaluru Urban" |
| state_name | str | State name | "Karnataka" |
| week | int | ISO week number (1–53) | 25 |
| year | int | Year | 2020 |
| disease | str | Disease name | "Chikungunya" |
| cases_suspected | int | Suspected cases | 145 |
| cases_confirmed | int | Confirmed cases (lab-verified) | 32 |
| deaths | int | Deaths | 0 |
| temp_mean | float | Mean temperature (°C) | 28.5 |
| temp_min | float | Min temperature (°C) | 21.2 |
| temp_max | float | Max temperature (°C) | 34.8 |
| rainfall_mm | float | Total rainfall (mm) | 12.5 |
| humidity_pct | float | Relative humidity (%) | 68.0 |
| lai | float | Leaf Area Index (vegetation proxy) | 2.1 |

**Notes:**
- `cases_suspected`: includes confirmed + probable + suspected.
- `cases_confirmed`: requires lab test (PCR, serology); subset of suspected.
- For modeling, use `cases_suspected` as primary (more complete).
- Climate variables are from India Meteorological Department (IMD), merged by district-week.
- Missing climate data is rare but can occur; see imputation strategy (Section 4.5).

---

### 4.2.2 IDSP Outbreak Reports (Validation Reference)

**Description:**  
Official weekly outbreak alerts from India's Integrated Disease Surveillance Program. States report diseases of epidemic concern; used to validate model predictions.

**Access:**
- **URL:** https://idsp.nic.in/ (website; may require API or manual download)
- **Format:** CSV or web scrape
- **Time coverage:** 2010 – present (inconsistent before 2010)

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| state_name | str | Reporting state |
| week | int | ISO week |
| year | int | Year |
| disease | str | Disease (e.g., "Chikungunya") |
| alert_status | str | "Alert" / "Outbreak" / "Monitoring" / None |
| confirmed_cases | int | Confirmed by state |
| details | str | Free-form outbreak description |

**Role in pipeline:**  
Used post-hoc for evaluation. *Not* used for model training (to avoid circular reasoning: if training labels came from IDSP, we'd be predicting what's already reported).

---

### 4.2.3 India Population Census 2011

**Description:**  
Population count by district; used to compute incidence rates (cases per 100k).

**Access:**
- **URL:** https://censusindia.gov.in/ (official; also available via local mirror)
- **Format:** CSV (typically district-level summary)
- **Time coverage:** 2011 Census (snapshot); used as fixed denominator

**Key Columns:**

| Column | Type | Description |
|--------|------|-------------|
| district_id | int | Census district code |
| district_name | str | District name |
| state_name | str | State |
| population | int | Total population 2011 |
| area_km2 | float | Area (km²) |

**Notes:**
- 2011 Census is now 15 years old. For very long-term analysis, consider US Census projections or interpolation.
- For 2026 analyses, population is ~1.4× the 2011 figure (rough estimate: 2% AAGR).
- For this project (short-term early warning), 2011 population is acceptable.

---

### 4.2.4 India District Shapefiles (Spatial Reference)

**Description:**  
GIS polygon boundaries for all India districts. Used for neighbor identification and visualization.

**Access:**
- **URL:** https://datameet.org/ (Datameet project; open GIS data for India)
- **Format:** Shapefile (.shp, .shx, .dbf, .prj) or GeoJSON
- **Projection:** WGS84 (EPSG:4326)

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| DISTRICT_C | int | District code |
| DIST_NAME | str | District name |
| ST_NAME | str | State name |
| geometry | geom | Polygon boundary |

**Role in pipeline:**  
- Identify neighboring districts (spatial adjacency) for optional neighbor features.
- Create visualizations (risk maps).

---

### 4.2.5 Google Trends (Optional Weak Signal)

**Description:**  
Search volume index for chikungunya-related keywords by region, week.

**Access:**
- **Tool:** Google Trends API (pytrends library, Python)
- **Format:** Time series JSON
- **Time coverage:** ~2004 – present
- **Spatial:** India-wide + state-level (limited granularity; not district-level)

**Example Query:**
```python
from pytrends.request import TrendReq
gt = TrendReq(hl='en-US')
gt.build_request({'chikungunya india': 1}, timeframe='2015-01-01 2022-12-31', geo='IN')
trends_df = gt.interest_over_time()
# Result: week, chikungunya_interest (0–100 index)
```

**Role:**  
- Optional; adds noise if not careful.  
- Use only if search trends are validated to correlate with cases.  
- Typically lagged or concurrent with case rise (not strongly predictive).

---

### 4.2.6 Brazil Validation Data (External Benchmark)

**Description:**  
For external validation, we use Brazil's arbovirus data (dengue, chikungunya, Zika) + climate.

**Access:**
- **URL:** Zenodo: "Brazil dengue and chikungunya data" (search for published datasets)
- **Platforms:** Mosqlimate (https://mosqlimate.org) — Brazil arbovirus + climate platform.
- **Format:** CSV or JSON
- **Time coverage:** 2014 – present
- **Spatial:** Municipality-level (~5500 municipalities)

**Key columns (similar to EpiClim):**

| Column | Type |
|--------|------|
| municipality_id | int |
| municipality_name | str |
| week | int |
| year | int |
| cases_chikungunya | int |
| temp_mean | float |
| rainfall_mm | float |

**Role:**  
- Test whether model trained on India generalizes to Brazil.  
- Secondary importance; primary focus is India.

---

## 4.3 Processed Data Formats (Pipeline Outputs)

### 4.3.1 Canonical Panel (Clean Data)

**Filename:** `data/processed/panel_chikungunya_v01.parquet` (or CSV)

**Description:**  
Unified table with one row per (district, week); all raw variables merged and cleaned.

**Schema:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| district_id | int | Census district code | 230 |
| district_name | str | District name | "Bengaluru Urban" |
| state_id | int | State code | 29 |
| state_name | str | State name | "Karnataka" |
| year | int | Year | 2020 |
| week | int | ISO week (1–53) | 25 |
| date_week_start | date | Start of ISO week | 2020-06-15 |
| cases_suspected | int | Suspected chikungunya cases | 145 |
| cases_confirmed | int | Confirmed cases | 32 |
| incidence_per_100k | float | cases_suspected / population × 100000 | 12.3 |
| temp_mean | float | Mean temperature (°C) | 28.5 |
| temp_min | float | Min temperature (°C) | 21.2 |
| temp_max | float | Max temperature (°C) | 34.8 |
| rainfall_mm | float | Rainfall (mm) | 12.5 |
| humidity_pct | float | Relative humidity (%) | 68.0 |
| lai | float | Leaf Area Index | 2.1 |
| population | int | Population (from Census 2011) | 11800000 |
| area_km2 | float | District area (km²) | 2190.0 |
| missing_flags | str | JSON: which columns had imputation? | '{"rainfall": "forward_fill"}' |

**Conventions:**
- **Sorting order:** (state_name, district_name, year, week)
- **Date range:** 2010-01-01 to latest (EpiClim coverage)
- **Missing data:** Handled as described in Section 4.5 (with `missing_flags` recorded)

**File format:**
- Primary: Parquet (efficient, preserves types)
- Alternative: CSV (for sharing, but larger file size)

---

### 4.3.2 Feature Engineering Output

**Filename:** `data/processed/features_engineered_v01.parquet`

**Description:**  
Row per (district, week); columns are all engineered features (case-based, climate-based, early-warning, spatial).

**Schema (partial listing; full list in TDD Section 3.3):**

| Column | Type | Description |
|--------|------|-------------|
| district_id | int | District |
| year | int | Year |
| week | int | Week |
| **Case Lags** | | |
| feat_cases_lag_1 | float | Incidence at t-1 (per 100k) |
| feat_cases_lag_2 | float | Incidence at t-2 |
| feat_cases_lag_4 | float | Incidence at t-4 |
| feat_cases_lag_8 | float | Incidence at t-8 |
| **Case Aggregates** | | |
| feat_cases_ma_2w | float | 2-week moving avg incidence |
| feat_cases_ma_4w | float | 4-week moving avg incidence |
| feat_cases_growth_rate | float | (c_t - c_{t-1}) / c_{t-1}, clipped [-1, 1] |
| **Case Volatility** | | |
| feat_cases_var_4w | float | Variance of incidence, last 4 weeks |
| feat_cases_acf_lag1_4w | float | Lag-1 autocorr, last 4 weeks |
| feat_cases_trend_4w | float | Slope of linear fit, last 4 weeks |
| feat_cases_skew_4w | float | Skewness, last 4 weeks |
| feat_cases_var_spike_ratio | float | var_4w / mean(var_52w_prior) |
| **Climate Lags** | | |
| feat_temp_lag_1 | float | Mean temp at t-1 (°C) |
| ... | | (similar for lag_2 through lag_8) |
| feat_temp_lag_8 | float | Mean temp at t-8 (°C) |
| **Climate Mechanistic** | | |
| feat_degree_days_above_20 | float | Accumulated heat (°C-days) above 20°C, last 2 weeks |
| feat_rain_lag_1 | float | Rainfall at t-1 (mm) |
| ... | | (similar for lag_2 through lag_8) |
| feat_rain_lag_8 | float | Rainfall at t-8 (mm) |
| feat_rain_persist_4w | float | Total rainfall, last 4 weeks (mm) |
| feat_temp_anomaly | float | Temp deviation from historical month-mean (°C) |
| **Early Warning Indicators** | | |
| feat_acf_change | float | Change in lag-1 acf: acf_4w - acf_4w_prior |
| feat_trend_accel_4w | float | Change in slope: slope_4w - slope_4w_prior |
| **Spatial (Optional)** | | |
| feat_neighbor_avg_incidence | float | Mean incidence of 3 nearest neighbors |
| feat_neighbor_max_incidence | float | Max incidence among neighbors |
| **Metadata** | | |
| feature_set_version | str | Version tag (e.g., "v01_mechanistic") |

**Handling missing features:**
- Features computed with < L historical weeks use NaN or forward-filled from prior week.
- For training/evaluation, rows with NaN are excluded (after setting aside test folds).

---

### 4.3.3 Labeled Samples (Supervised Datasets)

**Filename:** `data/processed/samples_supervised_v01.parquet`

**Description:**  
Rows are (district, week) samples; columns are features + label.

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| district_id | int | District |
| year | int | Year |
| week | int | Week |
| date_week_start | date | Week start date |
| feat_* | float | All engineered features (35+ columns) |
| label_outbreak_3w | int | Binary: outbreak in next 3 weeks? (0 or 1) |
| label_outbreak_3w_prob | float | Soft label (if available): prob outbreak (0–1) |
| split_fold | str | CV fold assignment ("fold_2017", "fold_2018", etc.) |

**Label convention:**
- `label_outbreak_3w = 1` if max incidence in weeks [t+1, t+3] > 75th percentile, else 0.
- Computed *after* feature engineering (to avoid leakage).

**CV fold assignment:**
- "fold_2017": train 2010–2016, test 2017.
- "fold_2018": train 2010–2017, test 2018.
- etc.

---

### 4.3.4 Model Predictions & Results

**Filename:** `results/predictions_model_{model_name}_fold_{fold}.csv`

**Description:**  
For each test sample (district-week), store model's predicted probability and actual label.

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| district_id | int | Test district |
| week | int | Test week |
| year | int | Test year |
| true_label | int | Actual outbreak (1 or 0) |
| pred_prob | float | Model predicted probability |
| pred_binary | int | Predicted class (1 if pred_prob > 0.5, else 0) |
| uncertainty | float | Uncertainty (std dev or credible interval width) |
| lead_time_weeks | int | How many weeks before peak cases did model alert? |
| model_name | str | Model identifier (e.g., "bayesian_v01", "xgboost_baseline") |
| fold | str | CV fold (e.g., "fold_2018") |

---

## 4.4 File Paths & Directory Structure

```
chikungunya_ews/
├── README.md (high-level project overview)
├── docs/
│   ├── 01_overview.md
│   ├── 02_prd.md
│   ├── 03_tdd.md
│   ├── 04_data_spec.md (this file)
│   ├── 05_experiments.md
│   └── 06_playbook.md
│
├── data/
│   ├── raw/
│   │   ├── epiclim_chikungunya_2010_2022.csv
│   │   ├── idsp_outbreaks_2010_2022.csv
│   │   ├── india_census_2011_district.csv
│   │   ├── india_districts_shapefile/
│   │   │   ├── india_districts.shp
│   │   │   ├── india_districts.shx
│   │   │   ├── india_districts.dbf
│   │   │   └── india_districts.prj
│   │   └── brazil_zenodo_data/ (if available)
│   │
│   └── processed/
│       ├── panel_chikungunya_v01.parquet
│       ├── features_engineered_v01.parquet
│       ├── samples_supervised_v01.parquet
│       └── cv_splits.json (fold assignments)
│
├── src/
│   ├── data_processing.py (load, clean, merge)
│   ├── feature_engineering.py (compute features)
│   ├── models/
│   │   ├── baselines.py (logistic, RF, XGB, Poisson, threshold)
│   │   ├── bayesian.py (Stan model + inference)
│   │   └── deep_learning.py (LSTM, CNN variants)
│   ├── evaluation.py (metrics, CV, comparison)
│   └── decision_layer.py (alerts, cost-loss, explanations)
│
├── notebooks/
│   ├── 01_eda.ipynb (exploratory data analysis)
│   ├── 02_features.ipynb (visualize engineered features)
│   ├── 03_baseline_models.ipynb (quick baseline tests)
│   └── 04_results_summary.ipynb (final comparison)
│
├── experiments/
│   ├── run_cv_fold_2017.sh (shell script for fold 2017)
│   ├── run_bayesian_inference.sh (MCMC on data)
│   └── results/
│       ├── predictions_bayesian_v01_fold_2017.csv
│       ├── predictions_xgboost_fold_2017.csv
│       ├── comparison_table.csv
│       └── plots/
│           ├── auc_curves.png
│           ├── lead_time_dist.png
│           ├── risk_map_latest.png
│           └── calibration_curves.png
│
└── config/
    ├── config_default.yaml (hyperparameters, paths)
    ├── config_experiment_v01.yaml (specific run)
    └── config_experiment_v02.yaml (variant)
```

---

## 4.5 Data Quality & Handling Missing Data

### 4.5.1 Missing Cases

**Scenario:** A district reports no data for a week (e.g., reporting gap).

**Strategy:**
- If gap ≤ 1 week: forward-fill (use previous week's value).
- If gap > 1 week: mark as missing; exclude from feature engineering window if lookback includes gap.
- Flag in `missing_flags` column for downstream auditing.

**Rationale:**  
Chikungunya cases don't reset to zero; reporting delays or gaps are common. Forward-fill is conservative.

### 4.5.2 Missing Climate

**Scenario:** Temperature or rainfall not available for a district-week.

**Strategy:**
- **Temperature:** Interpolate linearly between available values (or use regional average if > 2 weeks missing).
- **Rainfall:** Forward-fill (recent rainfall is most relevant for habitat formation).
- Flag in `missing_flags`.

**Rationale:**  
Climate data is more continuous than case data. Interpolation is reasonable for short gaps.

### 4.5.3 Anomalous Values

**Scenario:** A district reports 10,000 cases in a week (orders of magnitude higher than trend).

**Strategy:**
- Flag as potential reporting error or genuine large outbreak.
- Compute rolling z-score: z = (x - rolling_mean) / rolling_std.
- If |z| > 5, flag and optionally cap to rolling_mean + 3×rolling_std.
- Log flag in metadata; keep original for reference.

**Rationale:**  
Data quality varies; anomalies should be investigated but not blindly excluded.

### 4.5.4 Completeness Thresholds

**District-level:**
- Minimum 80% of weeks in analysis period (2010–2022).
- Districts with < 80% are excluded from modeling.

**Feature-level:**
- For each feature, compute % non-missing across all (district, week) in training set.
- Features with < 70% non-missing are excluded (too much imputation → unreliable).

---

## 4.6 Data Versioning & Snapshots

### 4.6.1 Version Tags

Each processed dataset includes a version tag:
- Format: `v01_mechanistic` (version + feature set type).
- When raw data or processing logic changes:
  - Increment minor version if small changes (e.g., new feature added).
  - Increment major version if structural change (e.g., new lookback window L).

### 4.6.2 Reproducibility

For reproducibility, always save:
1. Raw data snapshot (date downloaded, source URL).
2. Processing script version (git commit hash).
3. Output parquet file version tag.

**Example git commit message:**
```
Commit: 3a7f2e1
Data pipeline v01_mechanistic: compute features with L=12, H=3
- Added degree-days feature
- Filtered for 80% completeness
- Processed 2010-2022 EpiClim data
Output: features_engineered_v01.parquet (500k rows, 38 features)
```

---

## 4.7 Data Privacy & Licensing

### 4.7.1 India Data

- **EpiClim:** Open data; check license on website.
- **IDSP:** Government data; public health surveillance (open access expected).
- **Census 2011:** Public data; open access.
- **District shapefiles:** Datameet open project; CC license.

### 4.7.2 Brazil Data

- **Zenodo datasets:** Check individual dataset license (typically CC-BY or similar).
- **Mosqlimate:** Check terms of use.

### 4.7.3 Privacy Considerations

- No patient-level data used (only aggregated counts).
- District-level data is coarse enough to not pose privacy risk.
- Google Trends index is normalized (not absolute volume).

---

## 4.8 Data Access & Reproducibility

### 4.8.1 Downloading Data

**EpiClim:**
```bash
# Manual export from website or API (if available)
# Save to: data/raw/epiclim_chikungunya_2010_2022.csv
```

**Census data:**
```bash
# Download from censusindia.gov.in
# Or use R package: library(census2011)
```

**Shapefiles:**
```bash
# From Datameet
wget https://datameet.org/data/india_districts.zip
unzip india_districts.zip -d data/raw/india_districts_shapefile/
```

**Zenodo Brazil data:**
```bash
# Find dataset DOI; download via:
# https://zenodo.org/records/{DOI_ID}
```

### 4.8.2 Preprocessing Script

See `src/data_processing.py` for full pipeline:
```python
python src/data_processing.py \
    --input_epiclim data/raw/epiclim_chikungunya_2010_2022.csv \
    --input_census data/raw/india_census_2011_district.csv \
    --input_shapefile data/raw/india_districts_shapefile/india_districts.shp \
    --output_panel data/processed/panel_chikungunya_v01.parquet \
    --log_missing data/processed/missing_data_report.txt
```

---

## 4.9 How This Evolves

As you implement, you may discover:
- Additional data sources (e.g., lab confirmations from ICMR).
- New preprocessing needs (e.g., spatial smoothing, aggregation to state level).
- Data quality issues requiring new imputation strategies.

**Log all changes here with version bumps.**

**Example future entry:**

| Version | Date | Change | Impact |
|---------|------|--------|--------|
| 0.1 | Jan 2026 | Initial spec: EpiClim + census | Baseline |
| 0.2 | Feb 2026 | Added Google Trends weak signal | Extra feature column |
| 0.3 | Mar 2026 | Spatial aggregation to state level (per faculty feedback) | Different schema; rows by state-week |

---

**Next Step:** Read `05_experiments.md` for evaluation protocol and CV strategy.
