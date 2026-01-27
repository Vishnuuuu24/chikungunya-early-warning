# Phase 6 Implementation Verification Checklist

## Pre-Flight Check

### Prerequisites ✅
- [x] Phase 5 complete (`results/metrics/bayesian_cv_results.json` exists)
- [x] Baseline results available (`results/metrics/baseline_comparison.json` exists)
- [x] Processed data available (`data/processed/features_engineered_v01.parquet`)
- [x] Python environment configured (PyStan, XGBoost, scikit-learn)

### Code Artifacts ✅
- [x] `experiments/06_analyze_lead_time.py` created
- [x] `experiments/07_visualize_risk_trajectories.py` created
- [x] `experiments/08_simulate_decision_layer.py` created
- [x] `experiments/09_fusion_experiments.py` created
- [x] `experiments/10_comprehensive_metrics.py` created
- [x] `run_phase6.sh` created and executable

### Documentation ✅
- [x] `docs/01_overview.md` updated (revised goals, Track A vs B)
- [x] `docs/09_phase6_decision_fusion.md` created (complete specification)
- [x] `results/analysis/README_PHASE6.md` created (quick-start guide)
- [x] `PHASE6_SUMMARY.md` created (implementation summary)

### Directory Structure ✅
- [x] `results/analysis/` directory exists
- [x] `results/plots/risk_trajectories/` directory exists

---

## Execution Plan

### Quick Test (Optional)
Run comprehensive metrics first (no MCMC, fast validation):
```bash
python experiments/10_comprehensive_metrics.py
```

**Expected output:** `results/analysis/comprehensive_metrics.json`
**Time:** ~1 minute
**Purpose:** Verify environment and data loading

### Full Execution
Run all Phase 6 tasks:
```bash
./run_phase6.sh
```

**Expected time:** 4-8 hours
**Recommendation:** Run overnight or on compute cluster

---

## Post-Execution Verification

### Output Files
After completion, verify these files exist:

**Analysis Results:**
- [ ] `results/analysis/comprehensive_metrics.json`
- [ ] `results/analysis/lead_time_analysis.json`
- [ ] `results/analysis/decision_simulation.json`
- [ ] `results/analysis/fusion_results.json`

**Visualizations:**
- [ ] `results/plots/risk_trajectories/*.png` (8 files)

### Sanity Checks

**Lead-Time Analysis:**
```bash
# Check median lead time
python -c "
import json
with open('results/analysis/lead_time_analysis.json') as f:
    data = json.load(f)
    median = data['aggregated']['median_lead_time']
    print(f'Median lead time: {median:.1f} weeks')
    assert median > 0, 'Lead time should be positive'
    print('✓ Lead-time sanity check passed')
"
```

**Decision Simulation:**
```bash
# Check net benefit
python -c "
import json
with open('results/analysis/decision_simulation.json') as f:
    data = json.load(f)
    benefit = data['aggregated']['net_benefit']
    print(f'Net benefit: {benefit:.2f}')
    assert benefit > 0, 'Net benefit should be positive'
    print('✓ Decision simulation sanity check passed')
"
```

**Fusion Results:**
```bash
# Check AUPR improvement
python -c "
import json
with open('results/analysis/fusion_results.json') as f:
    data = json.load(f)
    for strategy, metrics in data['aggregated'].items():
        if 'aupr_mean' in metrics:
            aupr = metrics['aupr_mean']
            print(f'{strategy}: AUPR = {aupr:.3f}')
    print('✓ Fusion results sanity check passed')
"
```

---

## Common Issues & Solutions

### Issue 1: MCMC Divergences

**Symptom:** Warnings about divergent transitions

**Check:**
```python
import json
with open('results/analysis/lead_time_analysis.json') as f:
    data = json.load(f)
    for fold in data['fold_results']:
        if 'diagnostics' in fold:
            n_div = fold['diagnostics']['n_divergences']
            if n_div > 10:
                print(f"WARNING: {fold['fold']} has {n_div} divergences")
```

**Solution:**
- Acceptable if < 1% of iterations
- If > 10%, consider increasing `adapt_delta` or `n_warmup`

### Issue 2: Out of Memory

**Symptom:** Python crashes during MCMC

**Solution:**
- Reduce `n_chains` from 4 to 2 in MCMC_CONFIG
- Reduce `n_samples` from 1000 to 500
- Run on machine with more RAM

### Issue 3: Slow Execution

**Symptom:** Taking > 12 hours

**Solution:**
- Check CPU usage (should be ~100% during MCMC)
- Consider running on cloud (AWS/GCP)
- Parallelize folds (modify scripts manually)

### Issue 4: Import Errors

**Symptom:** `ModuleNotFoundError` for PyStan or XGBoost

**Solution:**
```bash
pip install pystan xgboost scikit-learn matplotlib pandas numpy
```

---

## Interpretation Guidelines

### Lead-Time Analysis

**Good result:**
- Median lead time ≥ 2 weeks
- Positive lead percentage > 60%
- IQR shows consistency

**Example:**
```json
{
  "median_lead_time": 2.5,
  "mean_lead_time": 2.8,
  "positive_lead_pct": 75.0,
  "iqr": [1.0, 4.0]
}
```

**Interpretation:** "Bayesian model provides 2.5 weeks median lead time, with 75% of outbreaks warned in advance."

### Decision Simulation

**Good result:**
- Sensitivity > 0.6 (catch 60%+ of outbreaks)
- False alarm rate < 0.2 (< 20% false alarms)
- Net benefit > 0 (interventions justified)

**Example:**
```json
{
  "sensitivity": 0.68,
  "precision": 0.42,
  "false_alarm_rate": 0.15,
  "net_benefit": 12.5
}
```

**Interpretation:** "Decision layer caught 68% of outbreaks with 15% false alarm rate. Net benefit of 12.5 units shows interventions are cost-effective."

### Fusion Results

**Good result:**
- Feature fusion AUPR > baseline XGBoost
- Any strategy improves recall
- Gated decision shows higher precision

**Example:**
```json
{
  "feature_fusion": {
    "aupr_mean": 0.32,
    "recall_mean": 0.45
  },
  "xgboost_baseline": {
    "aupr_mean": 0.28,
    "recall_mean": 0.16
  }
}
```

**Interpretation:** "Feature fusion improved AUPR from 0.28 to 0.32 and recall from 16% to 45%."

---

## Next Actions After Verification

### Immediate
1. [ ] Review all JSON files
2. [ ] Examine risk trajectory plots (visual inspection)
3. [ ] Run sanity checks (scripts above)

### Short-Term
4. [ ] Interpret findings using `docs/09_phase6_decision_fusion.md`
5. [ ] Create summary bullet points for faculty
6. [ ] Draft results section (methods + findings)

### Medium-Term
7. [ ] Faculty presentation (slides + talking points)
8. [ ] Threshold calibration (if experts available)
9. [ ] External validation (Brazil data)

---

## Success Criteria

Phase 6 is successful if:

✅ **Lead-time analysis shows positive median lead time (≥2 weeks)**

✅ **Decision simulation shows net benefit > 0**

✅ **Fusion experiments show AUPR or recall improvement**

✅ **Documentation explains decision-theoretic framework clearly**

✅ **Results rebut "Bayesian AUC too low" criticism**

---

## Sign-Off

**Code Implementation:** ✅ Complete (January 27, 2026)

**Documentation:** ✅ Complete (January 27, 2026)

**Execution Status:** ⏳ Pending (awaiting `./run_phase6.sh`)

**Validation Status:** ⏳ Pending (post-execution)

**Faculty Review:** ⏳ Pending (post-validation)

---

**Ready to execute Phase 6.**

Use this checklist to track progress and verify outputs systematically.
