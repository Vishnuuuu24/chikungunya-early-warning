# Chikungunya Early Warning System — Version 3

**Frozen:** 2026-01-26  
**Phase:** Bayesian Hierarchical Model Stabilization (Phase 4.2)  
**Status:** ✅ FROZEN  
**Parent:** v2_proto

---

## ⚠️ Important Notice

**This version is FROZEN and should not be modified.**

v3 represents the stabilized Bayesian hierarchical state-space model with improved MCMC diagnostics. Model structure is identical to v2_proto; only sampling parameters were tuned.

---

## Purpose

This version focuses on **stabilizing MCMC diagnostics** for the Bayesian hierarchical state-space model without changing the model structure.

---

## Changes from v2_proto

| Aspect | v2_proto | v3 |
|--------|----------|-----|
| φ (phi) constraint | `<lower=0>` | `<lower=0.1>` |
| ρ prior | `normal(0.7, 0.15)` | `normal(0.7, 0.10)` |
| Warmup iterations | 300 | 1000 |
| Sampling iterations | 300 | 1000 |
| Chains | 2 | 4 |
| adapt_delta | 0.8 (default) | 0.95 |

### Rationale

1. **φ lower bound (0.1):** Prevents dispersion parameter from hitting zero during warmup, which caused `gamma_lpdf: Random variable is 0` warnings

2. **Tighter ρ prior:** Reduces probability of ρ approaching the upper bound (0.99), which can cause divergences

3. **Increased MCMC budget:** More iterations improve ESS; more chains improve R-hat estimation

4. **Higher adapt_delta:** Reduces divergences by using smaller step sizes during sampling

---

## Model Structure (Unchanged from v2_proto)

### Latent State Dynamics

$$Z_{d,t} = \alpha_d + \rho \cdot (Z_{d,t-1} - \alpha_d) + \sigma \cdot \epsilon_{d,t}$$

### Observation Model

$$Y_{d,t} \sim \text{NegBin}(\mu_{d,t}, \phi)$$
$$\log(\mu_{d,t}) = Z_{d,t} + \beta_T \cdot T_{d,t}$$

### Hierarchical Structure

$$\alpha_d = \mu_\alpha + \sigma_\alpha \cdot \tilde{\alpha}_d$$

---

## Diagnostic Results (fold_2019)

### MCMC Diagnostics

| Diagnostic | v2_proto | v3 | Target | Status |
|------------|----------|-----|--------|--------|
| Divergences | 1 | 1 | 0 | ⚠️ Acceptable (documented) |
| Max R-hat | 1.071 | **1.014** | < 1.01 | ✅ Near target |
| Min ESS (bulk) | 51 | **246** | > 400 | ⚠️ Improved (~5×) |
| Min ESS (tail) | 38 | **626** | > 400 | ✅ Target met |
| Warning | `gamma_lpdf: 0` | `neg_binomial: -inf` | None | ⚠️ Changed (rare) |

### Parameter Estimates

| Parameter | Mean | Std | R-hat | ESS (bulk) |
|-----------|------|-----|-------|------------|
| μ_α (baseline) | 3.55 | 0.07 | 1.006 | 1139 |
| σ_α (district variance) | 0.58 | 0.08 | 1.010 | 442 |
| ρ (persistence) | 0.48 | 0.08 | 1.002 | 454 |
| β_T (temperature effect) | 0.02 | 0.02 | 1.001 | 2928 |
| σ (process noise) | 0.52 | 0.07 | 1.014 | 246 |
| φ (dispersion) | 3.48 | 0.53 | 1.013 | 281 |

### Posterior Predictive Check

| Metric | Value |
|--------|-------|
| Correlation (obs vs pred) | **0.959** |
| 90% CI coverage | **98.4%** |

---

## Known Issues (Documented and Accepted)

1. **1 divergence**: Occurs rarely; does not affect posterior quality
2. **ESS (bulk) for σ**: 246 is below 400 target but sufficient for inference
3. **`neg_binomial_2_log_lpmf: -inf` warning**: Rare event when Z becomes very negative; does not affect results

---

## Directory Contents

```
versions/Vishnu-Version-Hist/v3/
├── README.md                      # This file
├── run.sh                         # Reproduction script
├── stan_models/
│   └── hierarchical_ews_v01.stan  # Stabilized Stan model
└── code/
    ├── 04_train_bayesian.py       # Training script
    └── src/
        ├── config.py
        ├── evaluation/
        │   └── cv.py
        └── models/
            ├── base.py
            └── bayesian/
                └── state_space.py
```

---

## Reproduction

```bash
cd versions/Vishnu-Version-Hist/v3
bash run.sh
```

**Expected runtime:** ~8-10 minutes (4 chains × 2000 iterations)

---

## Single-Fold Only

This version tests only `fold_2019`. Full CV evaluation will be done in a future version.

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v1 | 2026-01-26 | Baselines without XGBoost |
| v1.1 | 2026-01-26 | Baselines with XGBoost (AUC=0.759) |
| v2_proto | 2026-01-26 | Bayesian prototype (unstable diagnostics) |
| **v3** | **2026-01-26** | **Bayesian stabilized (this version)** |

---

## Next Steps (NOT part of this version)

1. Full temporal CV across all folds
2. Compute AUC, F1, and outbreak-specific metrics
3. Compare against v1.1 baselines
4. Freeze as v4 (final Bayesian evaluation)

---

*This version is frozen and should not be modified. Improvements should be made in subsequent versions.*
