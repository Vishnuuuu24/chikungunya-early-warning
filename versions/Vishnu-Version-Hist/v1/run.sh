#!/bin/bash
# Chikungunya EWS - Version 1 Reproduction Script
# 
# This script reproduces the baseline model training and evaluation
# from the frozen v1 codebase.
#
# Prerequisites:
#   - Python 3.10+ with dependencies from requirements.txt
#   - Processed features file: data/processed/features_engineered_v01.parquet
#
# Usage:
#   cd versions/Vishnu-Version-Hist/v1
#   bash run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "============================================================"
echo "CHIKUNGUNYA EWS - VERSION 1 REPRODUCTION"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Script dir: $SCRIPT_DIR"
echo ""

# Verify processed data exists
FEATURES_FILE="$PROJECT_ROOT/data/processed/features_engineered_v01.parquet"
if [ ! -f "$FEATURES_FILE" ]; then
    echo "ERROR: Processed features file not found:"
    echo "  $FEATURES_FILE"
    echo ""
    echo "Please run the feature engineering pipeline first:"
    echo "  python experiments/02_build_features.py"
    exit 1
fi

echo "✓ Features file found"
echo ""

# Run baseline training
echo "Running baseline training..."
cd "$PROJECT_ROOT"
python experiments/03_train_baselines.py --config config/config_default.yaml

echo ""
echo "============================================================"
echo "V1 REPRODUCTION COMPLETE"
echo "============================================================"
echo "Results saved to: results/metrics/baseline_comparison.json"
