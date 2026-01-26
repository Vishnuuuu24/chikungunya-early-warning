# Chikungunya Early Warning System — Version 2 Prototype

**Frozen:** 2026-01-26  
**Phase:** Bayesian Hierarchical Model (Phase 4.1)  
**Status:** Prototype (Diagnostic Test Only)  
**Parent:** v1.1

---

## ⚠️ Important Notice

**This is a PROTOTYPE version, not a final Bayesian model.**

This version captures the first working implementation of the Bayesian hierarchical state-space model. It is preserved as a methodological checkpoint before stabilization and tuning.

---

## Model Intent

The Bayesian model aims to improve upon baseline classifiers by:

1. **Modeling latent transmission risk** — A hidden state $Z_{d,t}$ represents underlying outbreak potential before cases manifest
2. **Uncertainty quantification** — Full posterior distributions enable probabilistic forecasting
3. **Partial pooling** — Districts with sparse data borrow strength from the population
4. **Mechanistic structure** — Climate effects enter through biologically plausible dynamics

---

## Model Structure

### Latent State Dynamics

$$Z_{d,t} = \alpha_d + \rho \cdot (Z_{d,t-1} - \alpha_d) + \sigma \cdot \epsilon_{d,t}$$

where:
- $Z_{d,t}$ = log-transmission risk for district $d$ at week $t$
- $\alpha_d$ = district-specific baseline (hierarchical)
- $\rho$ = autoregression coefficient
- $\sigma$ = process noise
- $\epsilon_{d,t} \sim \mathcal{N}(0,1)$ = innovation

### Observation Model

$$Y_{d,t} \sim \text{NegBin}(\mu_{d,t}, \phi)$$
$$\log(\mu_{d,t}) = Z_{d,t} + \beta_T \cdot T_{d,t}$$

where:
- $Y_{d,t}$ = observed cases
- $T_{d,t}$ = temperature anomaly
- $\phi$ = overdispersion parameter

### Hierarchical Structure

$$\alpha_d = \mu_\alpha + \sigma_\alpha \cdot \tilde{\alpha}_d$$
$$\tilde{\alpha}_d \sim \mathcal{N}(0,1)$$

(Non-centered parameterization for better sampling)

---

## What Works

| Aspect | Status |
|--------|--------|
| Stan model compilation | ✅ Compiles successfully |
| MCMC sampling | ✅ Completes without fatal errors |
| Posterior predictive | ✅ Correlation = 0.97 with observed data |
| 90% CI coverage | ✅ 98% (well-calibrated) |
| Parameter estimates | ✅ Sensible values |

### Single-Fold Results (fold_2019)

| Parameter | Estimate (Mean ± Std) |
|-----------|----------------------|
| μ_α (baseline) | 3.54 ± 0.07 |
| σ_α (district variance) | 0.59 ± 0.08 |
| ρ (persistence) | 0.38 ± 0.09 |
| β_T (temperature effect) | 0.02 ± 0.02 |
| σ (process noise) | 0.56 ± 0.06 |
| φ (dispersion) | 3.73 ± 0.64 |

---

## Known Issues

### MCMC Diagnostics (NOT YET STABLE)

| Diagnostic | Observed | Target | Status |
|------------|----------|--------|--------|
| Divergences | 1 | 0 | ⚠️ |
| Max R-hat | 1.071 | < 1.05 | ⚠️ |
| Min ESS (bulk) | 51 | > 100 | ⚠️ |
| Min ESS (tail) | 38 | > 100 | ⚠️ |

### Sampling Warnings

```
gamma_lpdf: Random variable is 0, but must be positive finite!
```

This occurs because the dispersion parameter φ occasionally samples near zero during warmup. A lower bound constraint should be added.

### Recommendations for Stabilization (NOT implemented in this version)

1. Add `<lower=0.1>` constraint on φ
2. Increase warmup to 500+ iterations
3. Increase samples to 500+ iterations
4. Use 4 chains instead of 2
5. Consider stronger prior on ρ

---

## Directory Contents

```
versions/v2_proto/
├── README.md                      # This file
├── run.sh                         # Reproduction script
├── stan_models/
│   └── hierarchical_ews_v01.stan  # Stan model code
└── code/
    ├── 04_train_bayesian.py       # Training script
    └── src/models/bayesian/
        └── state_space.py         # Python wrapper
```

---

## Reproduction

To reproduce the v2_proto diagnostic test:

```bash
cd versions/v2_proto
bash run.sh
```

This runs a single-fold test (fold_2019) with:
- 2 MCMC chains
- 300 warmup iterations
- 300 sampling iterations

**Expected runtime:** ~5-6 minutes

---

## Prerequisites

1. **Python environment** with project dependencies
2. **CmdStanPy** installed:
   ```bash
   pip install cmdstanpy
   python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
   ```
3. **Processed features file:**
   ```
   data/processed/features_engineered_v01.parquet
   ```

---

## Comparison to Baselines

| Model | AUC | Notes |
|-------|-----|-------|
| XGBoost (v1.1) | 0.759 | Best baseline |
| Bayesian (v2_proto) | — | Not evaluated (prototype) |

**Note:** This prototype does not include full CV evaluation or metric comparison. That will be done in a future version after stabilization.

---

## Next Steps (NOT part of this version)

1. Fix φ boundary constraint
2. Increase MCMC iterations for stable diagnostics
3. Run full temporal CV across all folds
4. Compute AUC, F1, and other metrics
5. Compare against v1.1 baselines
6. Freeze as v2 (final Bayesian)

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v1 | 2026-01-26 | Baselines without XGBoost |
| v1.1 | 2026-01-26 | Baselines with XGBoost (AUC=0.759) |
| **v2_proto** | 2026-01-26 | Bayesian prototype (this version) |

---

*This version is frozen and should not be modified. Improvements should be made in subsequent versions.*
