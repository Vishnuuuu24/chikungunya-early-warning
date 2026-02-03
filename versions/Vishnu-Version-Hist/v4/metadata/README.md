# Version v4: Lead-Time Analysis Logic Finalized

## Version Information

| Field | Value |
|-------|-------|
| **Version** | v4 |
| **Date** | February 3, 2026 |
| **Author** | Vishnu |
| **Status** | FROZEN (Immutable) |

---

## Purpose

This version captures the **finalized lead-time analysis logic** for the 
Chikungunya Early Warning System, immediately **BEFORE** full cross-validation 
execution.

v4 represents:
- Final, audited lead-time computation module
- Final experiment orchestration script
- Complete configuration state
- Ready-to-execute state for Phase 7 Workstream A

---

## What This Version Contains

### code/
```
code/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── common/
│   ├── data/
│   ├── decision/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── cv.py
│   │   ├── lead_time.py    ← CRITICAL: Lead-time analysis module
│   │   └── metrics.py
│   ├── features/
│   ├── labels/
│   ├── models/
│   └── visualization/
└── experiments/
    ├── 00_sanity_check.py
    ├── 01_build_panel.py
    ├── 02_build_features.py
    ├── 03_train_baselines.py
    ├── 04_train_bayesian.py
    ├── 05_evaluate_bayesian.py
    ├── 06_analyze_lead_time.py  ← CRITICAL: Full CV orchestration
    ├── 07_visualize_risk_trajectories.py
    ├── 08_simulate_decision_layer.py
    ├── 09_fusion_experiments.py
    └── 10_comprehensive_metrics.py
```

### config/
```
config/
└── config_default.yaml
```

### results/
Empty — no results have been generated yet.

### models/
Empty — no trained model artifacts saved in this version.

---

## What This Version Does NOT Contain

| Excluded | Reason |
|----------|--------|
| `data/` | Raw/processed data stored separately, not versioned |
| `notebooks/` | Exploratory notebooks not part of production pipeline |
| `results/` from previous runs | v4 captures pre-execution state |
| `__pycache__/`, `*.pyc` | Temporary/compiled files |
| Trained model weights | No execution has occurred |
| CSV/JSON output files | No results generated yet |

---

## Lead-Time Analysis Configuration (FROZEN)

These thresholds are **immutable** for v4 execution:

| Parameter | Value | Source |
|-----------|-------|--------|
| Outbreak Percentile | **80** | 80th percentile of training cases defines outbreak |
| Bayesian Threshold Percentile | **75** | 75th percentile of Z_t for risk crossing |
| XGBoost Threshold | **0.5** | Standard classification threshold |

---

## Cross-Validation Configuration (FROZEN)

| Parameter | Value |
|-----------|-------|
| CV Strategy | Rolling-origin (expanding window) |
| Test Years | 2017, 2018, 2019, 2020, 2021, 2022 |
| Number of Folds | 6 |

For each fold:
- Training: All years strictly **before** test year
- Test: Only the test year

---

## Key Files in This Version

### src/evaluation/lead_time.py

The core lead-time analysis module containing:
- `LeadTimeAnalyzer` class — main orchestrator
- `OutbreakEpisode` dataclass — ground truth representation
- `LeadTimeResult` dataclass — per-episode results
- `LeadTimeSummary` dataclass — aggregated statistics
- Threshold computation functions (training-only, no leakage)
- Episode identification logic
- Lead-time calculation with edge case handling
- Sanity check utilities

**Edge Case Handling:**
- Never-warned episodes: lead_time = -1 (sentinel)
- Both never-warn: excluded from differential comparison, counted separately

### experiments/06_analyze_lead_time.py

Full CV execution script that:
1. Loads feature-engineered data
2. Creates rolling-origin CV splits
3. For each fold:
   - Trains XGBoost and Bayesian models
   - Computes outbreak thresholds from TRAINING data only
   - Identifies outbreak episodes in TEST data
   - Computes lead times using LeadTimeAnalyzer
4. Aggregates results across all folds
5. Saves outputs:
   - `lead_time_detail_all_folds.csv`
   - `lead_time_summary_overall.csv`
   - `lead_time_summary_by_fold.csv`
   - `lead_time_analysis_metadata.json`

---

## Execution Status

| Item | Status |
|------|--------|
| Lead-time module implemented | ✅ Complete |
| Experiment script implemented | ✅ Complete |
| Synthetic tests passed | ✅ Complete |
| Import verification passed | ✅ Complete |
| Full CV execution | ❌ **NOT YET RUN** |
| Results generated | ❌ **NOT YET GENERATED** |

---

## Important Notice

**This version is a PRE-EXECUTION snapshot.**

No full cross-validation has been executed. No results exist.
This freeze ensures reproducibility: if issues arise during execution,
we can return to this exact state.

---

## Next Steps (After This Freeze)

1. Execute `python experiments/06_analyze_lead_time.py`
2. Review outputs in `results/analysis/`
3. Validate numeric correctness
4. Proceed to Week 2 visualization (if results are acceptable)

---

## Reproducibility Statement

To reproduce the lead-time analysis from this version:

```bash
# Navigate to project root
cd /path/to/Chikungunya

# Ensure dependencies are installed
pip install -r requirements.txt

# Run lead-time analysis
python experiments/06_analyze_lead_time.py
```

Expected outputs will be written to `results/analysis/`.

---

## Version History Context

| Version | Description |
|---------|-------------|
| v1 | Initial Track A baselines |
| v1.1 | Track A refinements |
| v2_proto | Bayesian prototype |
| v3 | Stabilized Bayesian model (MCMC tuning) |
| **v4** | **Lead-Time Analysis Logic Finalized** |

---

*This README is part of the immutable v4 freeze.*
*Do not modify any files in this version directory.*
