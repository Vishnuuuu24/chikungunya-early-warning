#!/bin/bash
# Run Bayesian model diagnostics for v3 (stabilized)
# Phase 4.2: Improved MCMC diagnostics

set -e

echo "=============================================="
echo "Chikungunya EWS - Bayesian Model v3 (Stabilized)"
echo "=============================================="

# Navigate to project root
cd "$(dirname "$0")/../../.."

# Run single-fold diagnostic test with stabilized settings
python versions/Vishnu-Version-Hist/v3/code/04_train_bayesian.py \
    --config config/config_default.yaml \
    --fold fold_2019 \
    --n-warmup 1000 \
    --n-samples 1000 \
    --n-chains 4 \
    --adapt-delta 0.95

echo ""
echo "v3 diagnostic run complete."
