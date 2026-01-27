#!/bin/bash
#
# Phase 6 Master Execution Script
# 
# Runs all Phase 6 analysis tasks in sequence:
# 1. Comprehensive metrics (no MCMC, fast)
# 2. Lead-time analysis (MCMC required)
# 3. Risk trajectory visualization (MCMC required)
# 4. Decision-layer simulation (MCMC required)
# 5. Fusion experiments (MCMC required)
#
# Usage:
#   chmod +x run_phase6.sh
#   ./run_phase6.sh
#
# Note: Steps 2-5 require MCMC sampling and may take several hours.
#       Consider running overnight or on a compute cluster.

set -e  # Exit on error

echo "========================================"
echo "PHASE 6: Decision-Theoretic Evaluation"
echo "========================================"
echo ""
echo "WARNING: This will run MCMC sampling across multiple folds."
echo "Estimated time: 4-8 hours (depends on hardware)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "========================================"
echo "Step 0: Verify Phase 5 Results"
echo "========================================"
echo ""

if [ ! -f "results/metrics/bayesian_cv_results.json" ]; then
    echo "ERROR: Phase 5 results not found!"
    echo "Please run experiments/05_evaluate_bayesian.py first."
    exit 1
fi

echo "✓ Phase 5 results found"

echo ""
echo "========================================"
echo "Step 1: Comprehensive Metrics"
echo "========================================"
echo ""

python experiments/10_comprehensive_metrics.py

echo ""
echo "✓ Comprehensive metrics complete"

echo ""
echo "========================================"
echo "Step 2: Lead-Time Analysis"
echo "========================================"
echo ""
echo "This step fits Bayesian models for each CV fold."
echo "Expected time: 1-2 hours"
echo ""

python experiments/06_analyze_lead_time.py

echo ""
echo "✓ Lead-time analysis complete"

echo ""
echo "========================================"
echo "Step 3: Risk Trajectory Visualization"
echo "========================================"
echo ""
echo "This step fits Bayesian models for selected districts."
echo "Expected time: 30-60 minutes"
echo ""

python experiments/07_visualize_risk_trajectories.py --n-districts 8

echo ""
echo "✓ Risk trajectory plots complete"

echo ""
echo "========================================"
echo "Step 4: Decision-Layer Simulation"
echo "========================================"
echo ""
echo "This step fits Bayesian models and simulates decisions."
echo "Expected time: 1-2 hours"
echo ""

python experiments/08_simulate_decision_layer.py

echo ""
echo "✓ Decision-layer simulation complete"

echo ""
echo "========================================"
echo "Step 5: Fusion Experiments"
echo "========================================"
echo ""
echo "This step runs three fusion strategies across CV folds."
echo "Expected time: 2-3 hours"
echo ""

python experiments/09_fusion_experiments.py

echo ""
echo "✓ Fusion experiments complete"

echo ""
echo "========================================"
echo "PHASE 6 COMPLETE"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/analysis/comprehensive_metrics.json"
echo "  - results/analysis/lead_time_analysis.json"
echo "  - results/plots/risk_trajectories/*.png"
echo "  - results/analysis/decision_simulation.json"
echo "  - results/analysis/fusion_results.json"
echo ""
echo "Next steps:"
echo "  1. Review results JSON files"
echo "  2. Examine risk trajectory plots"
echo "  3. Read docs/09_phase6_decision_fusion.md for interpretation"
echo "  4. Prepare findings for faculty review"
echo ""
echo "✓✓✓ All Phase 6 tasks completed successfully! ✓✓✓"
