#!/bin/bash
# Chikungunya EWS - Version 2 Prototype Reproduction Script
# 
# This script reproduces the Bayesian model single-fold diagnostic test
# from the frozen v2_proto codebase.
#
# NOTE: This is a PROTOTYPE version. Diagnostics are NOT fully stable.
#
# Prerequisites:
#   - Python 3.10+ with dependencies from requirements.txt
#   - CmdStanPy installed (pip install cmdstanpy)
#   - CmdStan installed (python -c "import cmdstanpy; cmdstanpy.install_cmdstan()")
#   - Processed features file: data/processed/features_engineered_v01.parquet
#
# Usage:
#   cd versions/Vishnu-Version-Hist/v2_proto
#   bash run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "============================================================"
echo "CHIKUNGUNYA EWS - VERSION 2 PROTOTYPE REPRODUCTION"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Script dir: $SCRIPT_DIR"
echo ""
echo "⚠️  NOTE: This is a PROTOTYPE version."
echo "⚠️  MCMC diagnostics may show warnings (expected)."
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

# Verify CmdStanPy is available
python -c "import cmdstanpy" 2>/dev/null || {
    echo "ERROR: CmdStanPy not available."
    echo "Install with: pip install cmdstanpy"
    echo "Then run: python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'"
    exit 1
}

echo "✓ CmdStanPy available"
echo ""

# Run Bayesian model single-fold diagnostic test
echo "Running Bayesian model (single-fold diagnostic)..."
cd "$PROJECT_ROOT"
python experiments/04_train_bayesian.py \
    --fold fold_2019 \
    --n-warmup 300 \
    --n-samples 300 \
    --n-chains 2

echo ""
echo "============================================================"
echo "V2 PROTOTYPE REPRODUCTION COMPLETE"
echo "============================================================"
echo "This was a diagnostic test only (fold_2019)."
echo "See README.md for known issues and next steps."
