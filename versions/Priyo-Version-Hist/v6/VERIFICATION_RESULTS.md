# V6 Verification Results

**Date**: February 5, 2026  
**Status**: ⏳ Analysis Running...

---

## Setup Summary

✅ **Environment Created**: Virtual environment `chik` with Python 3.14  
✅ **Dependencies Installed**: All packages including PyMC, CmdStan, XGBoost  
✅ **Sanity Check**: Passed (4/4 checks)  
✅ **Data Pipeline**: Panel + Features regenerated  

---

## Key Metrics to Compare (v5 → v6)

### Data Pipeline
| Metric | v5 (Old) | v6 (Expected) | v6 (Actual) | Status |
|--------|----------|---------------|-------------|--------|
| Total panel rows | 731 | 731 | 731 | ✓ |
| Feature columns | 37 | 37 | 37 | ✓ |
| Labeled samples | 101 | 101 | 101 | ✓ |
| Positive labels (p75) | 43 (42.6%) | 43 (42.6%) | 43 (42.6%) | ✓ |

### Lead-Time Analysis
| Metric | v5 (Old) | v6 (Expected) | v6 (Actual) | Status |
|--------|----------|---------------|-------------|--------|
| Valid test samples (all folds) | ~44 | ~400+ | ⏳ | Pending |
| Outbreak episodes detected | 4 | 15-25 | ⏳ | Pending |
| Successful CV folds | 3 of 6 | 5-6 of 6 | ⏳ | Pending |
| Bayesian warnings | 3 | ⏳ | ⏳ | Pending |
| XGBoost warnings | 1 | ⏳ | ⏳ | Pending |

---

## Analysis Progress

**Started**: Feb 5, 2026  
**Command**: `python experiments/06_analyze_lead_time.py`  
**Log File**: `/tmp/v6_leadtime.log`  
**Background PID**: Check with `ps aux | grep 06_analyze_lead_time`

### To Monitor Progress:
```bash
# Activate environment
source chik/bin/activate

# Check log in real-time
tail -f /tmp/v6_leadtime.log

# Check if process is still running
ps aux | grep 06_analyze_lead_time

# Check output files
ls -lh results/analysis/
```

---

## Expected Improvements from v6 Fixes

### 1. **Imputation Over Dropna** 
**Technical Change**: 
```python
# OLD (v5):
train_valid = train_df.dropna(subset=feature_cols + [target_col])

# NEW (v6):
train_imputed = impute_missing_features(train_df, feature_cols)
train_valid = train_imputed.dropna(subset=[target_col])
```

**Expected Impact**:
- Recover ~90% of data lost to aggressive filtering
- Valid test samples: 44 → 400+
- More outbreak episodes: 4 → 15-25

### 2. **Percentile Alignment** 
**Technical Change**: 
- `DEFAULT_OUTBREAK_PERCENTILE`: 80 → 75
- Now matches `config: outbreak_percentile: 75`

**Expected Impact**:
- More consistent outbreak detection
- Fewer mismatches between training labels and evaluation thresholds
- 2-3x more detected outbreaks

### 3. **Config-Driven Imputation**
**Technical Change**:
- `loader.py` now respects `config: imputation.cases: zero_fill`
- No more hardcoded `.fillna(0)`

**Expected Impact**:
- Clear documentation for thesis methods
- Defensible imputation strategy

---

## Files Generated (Check After Completion)

```
results/analysis/
├── lead_time_detail_all_folds.csv       [⏳ Pending]
├── lead_time_summary_overall.csv        [⏳ Pending]
├── lead_time_summary_by_fold.csv        [⏳ Pending]
└── lead_time_analysis_metadata.json     [⏳ Pending]
```

---

## Troubleshooting

### If Analysis Fails:
1. Check log file: `cat /tmp/v6_leadtime.log | tail -100`
2. Look for Python errors, memory issues, or Stan compilation errors
3. Check data files exist: `ls -lh data/processed/`
4. Verify config is correct: `cat config/config_default.yaml`

### Common Issues:
- **Stan compilation error**: CmdStan not properly installed
- **Memory error**: Too many MCMC samples (reduce in config)
- **NaN predictions**: Missing imputation in some features
- **Zero episodes**: Outbreak threshold too high

---

## Next Steps After Completion

1. **Verify Results**:
   ```bash
   python -c "import pandas as pd; df=pd.read_csv('results/analysis/lead_time_summary_overall.csv'); print(df)"
   ```

2. **Compare with v5**:
   ```bash
   diff versions/Vishnu-Version-Hist/v5/results/analysis/lead_time_summary_overall.csv \
        results/analysis/lead_time_summary_overall.csv
   ```

3. **Generate Phase 7 Visualizations**:
   ```bash
   python experiments/07_phase7_visualizations.py
   ```

4. **Update This Document** with actual results

---

## Notes

- Analysis may take 30-60 minutes depending on hardware
- Bayesian models use MCMC (1000 warmup + 1000 samples × 4 chains per fold)
- XGBoost is faster (~seconds per fold)
- Expected total folds: 6 (years 2017-2022)

---

**Last Updated**: Feb 5, 2026 (analysis started)
