# Chikungunya Early Warning System — Version 1.1

**Frozen:** 2026-01-26  
**Phase:** Baseline Models (Track A)  
**Status:** Complete  
**Parent:** v1

---

## Changes from v1

| Aspect | v1 | v1.1 |
|--------|-----|------|
| Models | 3 (Threshold, Logistic, Random Forest) | **4** (+XGBoost) |
| Best AUC | 0.719 (Random Forest) | **0.759 (XGBoost)** |
| Best F1 | 0.569 (Random Forest) | **0.630 (XGBoost)** |
| Dependencies | libomp not required | libomp required for XGBoost |

**Summary:** Version 1.1 adds XGBoost to the baseline comparison after resolving macOS dependency issues (libomp). XGBoost achieves the best performance among all baselines.

---

## Models Included

| Model | Description |
|-------|-------------|
| **XGBoost** | Gradient boosting with scale_pos_weight for class imbalance |
| Random Forest | Ensemble of 100 decision trees with balanced class weights |
| Logistic Regression | L2-regularized linear classifier (C=0.01) |
| Threshold Rule | Simple μ + 2σ rule on lag-1 incidence |

## Evaluation Protocol

- **Cross-validation:** Rolling-origin temporal CV (expanding window)
- **Test years:** 2017, 2018, 2019, 2020, 2021 (2022 excluded due to insufficient samples)
- **Features:** 37 engineered features (case lags, climate, EWS indicators)
- **Valid samples:** 94 (after filtering for feature completeness)

## Results Summary

| Model | AUC | F1 | Sensitivity | Specificity | Brier |
|-------|-----|-----|-------------|-------------|-------|
| **XGBoost** | **0.759 ± 0.113** | **0.630 ± 0.052** | 0.634 | 0.753 | 0.201 |
| Random Forest | 0.719 ± 0.203 | 0.569 ± 0.165 | 0.590 | 0.707 | 0.218 |
| Logistic Regression | 0.666 ± 0.140 | 0.472 ± 0.312 | 0.466 | 0.786 | 0.231 |
| Threshold Rule | 0.604 ± 0.183 | 0.067 ± 0.133 | 0.040 | 0.971 | 0.355 |

### Interpretation

- **XGBoost** achieves the highest discriminative performance (AUC = 0.76) and best calibration (Brier = 0.20).
- XGBoost also shows the **lowest variance** across folds (AUC std = 0.11), indicating more stable predictions.
- **Random Forest** remains competitive but with higher variance.
- The improvement from v1's best (RF: 0.719) to v1.1's best (XGB: 0.759) represents a **5.6% relative improvement** in AUC.

### XGBoost Fold-Level Performance

| Fold | AUC | F1 | Test Samples |
|------|-----|-----|--------------|
| 2017 | 0.852 | 0.714 | 12 |
| 2018 | 0.768 | 0.667 | 15 |
| 2019 | 0.892 | 0.600 | 18 |
| 2020 | 0.571 | 0.571 | 10 |
| 2021 | 0.711 | 0.600 | 14 |

## Data Requirements

This version requires the processed features file:
```
data/processed/features_engineered_v01.parquet
```

**Additional requirement:** XGBoost requires libomp on macOS:
```bash
brew install libomp
```

## Reproduction

To reproduce v1.1 results from the project root:

```bash
cd versions/v1.1
bash run.sh
```

Or directly:
```bash
python experiments/03_train_baselines.py --config config/config_default.yaml
```

## Directory Contents

```
versions/v1.1/
├── README.md           # This file
├── run.sh              # Reproduction script
├── code/
│   ├── src/            # Frozen source modules
│   └── 03_train_baselines.py
├── config/
│   └── config_default.yaml
└── results/
    └── baseline_comparison.json
```

## Configuration

Key parameters from `config/config_default.yaml`:

| Parameter | Value |
|-----------|-------|
| Outbreak threshold | 75th percentile |
| Prediction horizon | 3 weeks |
| CV test years | [2017, 2018, 2019, 2020, 2021, 2022] |
| XGBoost n_estimators | 100 |
| XGBoost max_depth | 4 |
| XGBoost learning_rate | 0.1 |

## Limitations

1. **Sample size:** Only 94 valid samples after filtering for feature completeness
2. **Temporal coverage:** Sparse data in later years (2020–2022)
3. **Platform dependency:** XGBoost requires libomp on macOS

## Next Steps (for future versions)

- [ ] Implement Bayesian hierarchical model (Track B)
- [ ] Compute lead time metrics
- [ ] Generate prediction plots and calibration curves
- [ ] Feature importance analysis

---

*This version is frozen and should not be modified. Any improvements should be made in subsequent versions.*
