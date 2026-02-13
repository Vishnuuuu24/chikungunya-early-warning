# Feature Selection Strategy Discussion

## Current Status
- **Total Features**: 37 features across 5 categories
- **Approach**: Using all features with neutral-value imputation for missing data
- **Concern**: Potential "kitchen-sinking" vs mechanistically justified feature set

## Feature Taxonomy & Mechanistic Basis

### Tier 1: Core Features (8) - MECHANISTICALLY JUSTIFIED
**Must-have features with strong biological/epidemiological theory:**

1. **cases_lag_1** - Immediate autocorrelation (persistence)
2. **cases_lag_2** - Short memory (incubation period ~3-7 days)
3. **degree_days** - Vector development threshold (>20°C is biological constraint)
4. **temp_anomaly** - Climate shock indicator
5. **var_spike_ratio** - EWS: Critical slowing down (RESEARCH CONTRIBUTION)
6. **week_sin, week_cos** - Seasonal forcing (monsoon cycles)
7. **lat_norm** - Spatial risk gradient

**Missing Data Impact**: <20% for case lags, ~30% for EWS (requires 52w baseline)

### Tier 2: Conditional Features (4) - USEFUL BUT NOT CRITICAL
**Add if data completeness >70%:**

1. **cases_ma_4w** - Smoothed trend (redundant with lags but robust to noise)
2. **rain_persistence** - Vector breeding site proxy
3. **acf_change** - EWS: Loss of resilience
4. **case_growth_rate** - Exponential phase detector

**Missing Data Impact**: ~40% for EWS, ~25% for climate

### Tier 3: Drop Candidates (25) - REDUNDANT OR ATHEORETICAL
**Limited theoretical justification or highly correlated:**

- Long lags (4w, 8w) - Redundant with shorter lags + state-space AR(1)
- Higher-order moments (skewness, kurtosis) - Unstable with sparse data
- Excess climate lags - Correlation structure captured by temp_anomaly + degree_days
- Spatial neighbors - Already encoded in hierarchical structure

## Proposed Strategy

### Option A: Lock to Tier 1 (Recommended for Thesis)
**Pros:**
- Clear mechanistic justification for each feature
- Handles missing data gracefully (<30% missingness)
- Defensible against "fishing expedition" criticism
- Aligns with Papers 1-4 (focused feature sets)

**Cons:**
- May miss non-linear interactions in Tier 2
- Lower ceiling for predictive performance

### Option B: Use All 37 (Current Approach)
**Pros:**
- Maximizes signal capture (kitchen sink may work with hierarchical priors)
- Useful for exploratory phase

**Cons:**
- Hard to defend which features matter
- Induces bias via neutral-value imputation for sparse districts
- Thesis committee may question theoretical grounding

### Option C: Hybrid with Completeness Threshold
**Pros:**
- Adaptive to data availability
- Balances theory and performance

**Cons:**
- Introduces heterogeneity across districts
- Complex to document

## Recommendation for Thesis Defense

**Run ablation study comparing:**
1. **Baseline**: Tier 1 only (8 features)
2. **Enhanced**: Tier 1 + Tier 2 (12 features) where completeness >70%
3. **Full**: All 37 features (current)

**Decision Rule:**
- If AUC_full - AUC_baseline < 0.05 → Lock to Tier 1 and document mechanistic justification
- If AUC_full - AUC_baseline ≥ 0.05 → Use Full but clearly separate features in thesis:
  - **Core mechanistic** (Tier 1)
  - **Performance-enhancing** (Tier 2-3, empirically selected)

## Missing Data Handling

### Current Approach (Neutral-Value Imputation)
- Cases: 0 (no recent cases)
- Climate: mean(district) (average conditions)
- EWS: 0 (no signal)

**Problem**: Biases sparse districts toward null predictions

### Alternative (For Tier 1 Approach)
- Explicitly filter districts requiring >70% completeness in Tier 1 features
- Document as "data quality threshold" not arbitrary exclusion
- More honest than silent imputation bias

## Next Steps

1. ✅ Document this strategy
2. 🔄 Run full pipeline with current 37 features
3. ⏭️ Compare with Tier 1 ablation
4. ⏭️ Select final feature set for thesis reporting

---
**Date**: 2026-02-10  
**Status**: Discussion captured - awaiting full pipeline results before finalizing strategy
