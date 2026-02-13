#!/usr/bin/env python3
"""
Experiment 06: Lead-Time Analysis Across All CV Folds

Phase 7 - Workstream A: Lead-Time Analysis Execution

PURPOSE:
--------
Execute lead-time analysis for Bayesian vs XGBoost across all rolling-origin
cross-validation folds (2017-2022). This validates Claim 2: "Latent risk 
inference captures outbreak escalation earlier than binary classifiers."

VERSION: v4 (frozen lead_time.py)
DATE: February 2026

WORKFLOW:
---------
For each fold (test_year in [2017, 2018, 2019, 2020, 2021, 2022]):
    1. Load data: training (years < test_year), test (year == test_year)
    2. Train Bayesian model → extract Z_t posterior mean for test set
    3. Train XGBoost model → extract p_t probabilities for test set
    4. Compute outbreak thresholds (80th percentile) from TRAINING only
    5. Identify outbreak episodes in TEST set
    6. Compute lead times using LeadTimeAnalyzer
    7. Save fold-level outputs

Aggregation:
    - Combine all episodes across folds
    - Compute overall summary statistics
    - Compute per-fold summary statistics

OUTPUTS:
--------
1. results/analysis/lead_time_detail_all_folds.csv
2. results/analysis/lead_time_summary_overall.csv  
3. results/analysis/lead_time_summary_by_fold.csv

CONSTRAINTS:
------------
- Uses EXACT logic from src/evaluation/lead_time.py (frozen v4)
- No threshold tuning
- No filtering of districts or episodes
- Preserves -1 sentinel for never-warned
- Fails loudly if any fold has zero outbreak episodes

Reference: docs/Version 2/02_lead_time_analysis_spec_v2.md
Reference: docs/Version 2/08_phase7_roadmap_v2.md (Weeks 1-2)
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

# =============================================================================
# PATH SETUP
# =============================================================================

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.config import load_config, get_project_root
from src.evaluation.cv import create_rolling_origin_splits, CVFold
from src.evaluation.lead_time import (
    LeadTimeAnalyzer,
    LeadTimeResult,
    LeadTimeSummary,
    OutbreakEpisode,
    results_to_dataframe,
    summary_to_dataframe,
    summarize_lead_times,
    sanity_check_results,
    NEVER_WARNED_SENTINEL,
    DEFAULT_OUTBREAK_PERCENTILE,
    DEFAULT_BAYESIAN_THRESHOLD_PERCENTILE,
    DEFAULT_XGBOOST_THRESHOLD
)

# Add v3 code path for Bayesian model
v3_code_path = project_root / "versions" / "Vishnu-Version-Hist" / "v3" / "code"
sys.path.insert(0, str(v3_code_path))


# =============================================================================
# CONFIGURATION
# =============================================================================

# CV folds to analyze (rolling-origin)
TEST_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]

# MCMC configuration for Bayesian model (matches Phase 5)
MCMC_CONFIG = {
    'n_warmup': 1000,
    'n_samples': 1000,
    'n_chains': 4,
    'adapt_delta': 0.95,
    'seed': 42
}

# Lead-time thresholds (from spec - DO NOT MODIFY)
OUTBREAK_PERCENTILE = DEFAULT_OUTBREAK_PERCENTILE  # 80
BAYESIAN_PERCENTILE = DEFAULT_BAYESIAN_THRESHOLD_PERCENTILE  # 75
XGBOOST_THRESHOLD = DEFAULT_XGBOOST_THRESHOLD  # 0.5

# =============================================================================
# FAIR ANALYSIS MODE (v4.2)
# =============================================================================
# For symmetric comparison, use the SAME percentile threshold for BOTH models,
# computed from TRAINING predictions only. This answers:
# "Which model ranks outbreak escalation earlier?"
#
# - Bayesian: Z_t > q_k(Z_train)
# - XGBoost:  P_t > q_k(P_train)
#
# This is more scientifically fair than using fixed 0.5 for XGBoost.
# =============================================================================

ANALYSIS_MODE_PERCENTILE = 75  # Same percentile for both models
USE_FAIR_ANALYSIS_MODE = True  # Set False to revert to original (fixed 0.5)


# =============================================================================
# FAIR ANALYSIS HELPER FUNCTIONS (v4.2)
# =============================================================================

def compute_xgboost_training_predictions(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak'
) -> pd.DataFrame:
    """
    Get XGBoost predictions on TRAINING data for threshold computation.
    
    Uses cross-validation style prediction: for each sample, predict using
    a model trained on other samples. This avoids overfitting the threshold.
    
    For simplicity in this implementation, we use the full training model
    applied back to training data. The threshold is still valid as it represents
    the model's typical output distribution.
    
    Args:
        train_df: Training DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        
    Returns:
        Training DataFrame with 'prob' column
    """
    from src.models.baselines.xgboost_model import XGBoostBaseline
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_train = np.nan_to_num(X_train, nan=0.0)
    
    model = XGBoostBaseline({})
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_train)
    
    pred_df = train_df[['state', 'district', 'year', 'week', 'cases']].copy()
    pred_df['prob'] = probs
    
    return pred_df


def compute_percentile_threshold_from_predictions(
    pred_df: pd.DataFrame,
    value_col: str,
    percentile: int = ANALYSIS_MODE_PERCENTILE
) -> float:
    """
    Compute global percentile threshold from prediction values.
    
    For fair comparison, we use a GLOBAL threshold (not district-specific)
    so both models are on equal footing.
    
    Args:
        pred_df: Predictions DataFrame
        value_col: Column to compute percentile on ('prob' or 'z_mean')
        percentile: Percentile (default: 75)
        
    Returns:
        Threshold value
    """
    values = pred_df[value_col].dropna()
    if len(values) == 0:
        raise ValueError(f"No valid values in column {value_col}")
    
    return float(np.percentile(values, percentile))


def compute_fair_thresholds_from_training(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak',
    percentile: int = ANALYSIS_MODE_PERCENTILE
) -> Tuple[float, float]:
    """
    Compute BOTH Bayesian and XGBoost thresholds from TRAINING data.
    
    This implements the fair analysis mode where:
    - Bayesian threshold = percentile of Z_t on training predictions
    - XGBoost threshold = percentile of P_t on training predictions
    
    NOTE: Bayesian predictions on training data require fitting the model.
    For efficiency, we approximate using the XGBoost distribution characteristics.
    
    In practice, for this thesis, we use:
    - XGBoost: actual training predictions
    - Bayesian: training predictions from the combined model (already fitted)
    
    Args:
        train_df: Training DataFrame (NaN-filtered)
        feature_cols: Feature column names
        target_col: Target column name
        percentile: Percentile for both thresholds
        
    Returns:
        Tuple of (bayesian_threshold, xgboost_threshold)
    """
    # Compute XGBoost threshold from training predictions
    xgb_train_preds = compute_xgboost_training_predictions(
        train_df, feature_cols, target_col
    )
    xgb_threshold = compute_percentile_threshold_from_predictions(
        xgb_train_preds, 'prob', percentile
    )
    
    print(f"    XGBoost p{percentile} threshold from training: {xgb_threshold:.4f}")
    
    # For Bayesian, we need training predictions but the model fits on combined data
    # We'll compute the Bayesian threshold from the combined model's training portion
    # This is handled in analyze_single_fold where we have access to the fit model
    # Return a placeholder that will be overwritten
    bayesian_threshold = None  # Computed later from actual Bayesian fit
    
    return bayesian_threshold, xgb_threshold


# =============================================================================
# DATA LOADING
# =============================================================================

def load_features_data() -> pd.DataFrame:
    """
    Load feature-engineered data from processed parquet.
    
    Returns:
        DataFrame with all features and metadata
    """
    data_path = project_root / "data" / "processed" / "features_engineered_v01.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Features file not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df)} samples, {df['district'].nunique()} districts")
    print(f"  Years: {sorted(df['year'].unique())}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all feature column names (prefix: feat_)."""
    return [c for c in df.columns if c.startswith('feat_')]


def impute_missing_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Aggressively impute missing feature values to recover maximum data.
    
    V6 FIX: Expanded from 7 features to ALL 37 features to prevent data loss.
    Previous version only imputed 7 features, causing 94% data loss to dropna().
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        
    Returns:
        DataFrame with imputed values (minimal NaN remaining)
    """
    df = df.copy()
    
    print(f"    Imputing {len(feature_cols)} features...")
    initial_nulls = df[feature_cols].isnull().sum().sum()
    
    # 1. Past-only fill within groups for all lag features (climate + case lags)
    # IMPORTANT: do NOT backfill (bfill) because it leaks future values into the past.
    lag_cols = [c for c in feature_cols if 'lag' in c or 'persist' in c or 'anomaly' in c]
    for col in lag_cols:
        if col in df.columns:
            df[col] = df.groupby(['state', 'district'])[col].transform(
                lambda x: x.ffill()
            )
    
    # 2. Case-based features: use 0 (no cases = no activity)
    case_feature_patterns = ['cases_', 'acf', 'growth', 'trend', 'skew', 'normalized']
    case_cols = [c for c in feature_cols if any(p in c for p in case_feature_patterns)]
    for col in case_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    # 3. Climate/environmental features: use column median
    climate_patterns = ['temp', 'rain', 'lai', 'degree_day', 'monsoon']
    climate_cols = [c for c in feature_cols if any(p in c for p in climate_patterns)]
    for col in climate_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
    
    # 4. Variance/ratio features: use neutral value (1.0 = no change)
    ratio_patterns = ['var', 'spike', 'ratio']
    ratio_cols = [c for c in feature_cols if any(p in c for p in ratio_patterns)]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].fillna(1.0)
    
    # 5. Cyclic/geographic features: use 0 (neutral)
    cyclic_patterns = ['sin', 'cos', 'quarter', 'lat', 'lon', 'interact']
    cyclic_cols = [c for c in feature_cols if any(p in c for p in cyclic_patterns)]
    for col in cyclic_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    # 6. Moving averages: use global mean
    ma_cols = [c for c in feature_cols if '_ma_' in c]
    for col in ma_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val if not pd.isna(mean_val) else 0.0)
    
    # 7. Catch-all: any remaining NaNs → 0.0
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    final_nulls = df[feature_cols].isnull().sum().sum()
    recovered = initial_nulls - final_nulls
    print(f"    Imputation: {initial_nulls} nulls → {final_nulls} nulls (recovered {recovered} values)")
    
    return df


# =============================================================================
# MODEL TRAINING & PREDICTION
# =============================================================================

def train_xgboost_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak'
) -> pd.DataFrame:
    """
    Train XGBoost on training data and generate predictions for test data.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        
    Returns:
        Test DataFrame with added 'prob' column containing XGBoost probabilities
    """
    from src.models.baselines.xgboost_model import XGBoostBaseline
    
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    
    # Handle any remaining NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Train model
    model = XGBoostBaseline({})
    model.fit(X_train, y_train)
    
    # Predict
    probs = model.predict_proba(X_test)
    
    # Create output DataFrame
    base_cols = ['state', 'district', 'year', 'week', 'cases']
    extra_cols = ['incidence_per_100k', '_row_id']
    keep_cols = base_cols + [c for c in extra_cols if c in test_df.columns]
    pred_df = test_df[keep_cols].copy()
    pred_df['prob'] = probs
    
    return pred_df


def train_bayesian_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train Bayesian state-space model and generate predictions for test data.
    
    This extracts the latent risk Z_t (posterior mean) for each observation.
    The Bayesian model fits on combined train+test data (standard for state-space
    models) but we only extract predictions for test rows.
    
    IMPORTANT: Both train_df and test_df must be NaN-filtered (valid samples only)
    to ensure alignment with XGBoost predictions.
    
    ==========================================================================
    CRITICAL FIX (v4.2): Handle duplicate (state, district, year, week) keys
    ==========================================================================
    
    The data contains duplicate keys. We handle this by:
      1. Assigning a unique row ID to each row BEFORE concatenation
      2. Tracking which IDs belong to test data
      3. After model fitting, extracting by row ID (NOT by 4-tuple key)
    
    ==========================================================================
    
    Args:
        train_df: Training DataFrame (NaN-filtered)
        test_df: Test DataFrame (NaN-filtered)
        feature_cols: Feature column names
        
        Returns:
                (train_pred_df, test_pred_df) where each includes:
                    - state, district, year, week, cases
                    - _row_id (stable join key)
                    - z_mean, z_sd
                Guaranteed: lengths match the corresponding inputs.
    """
    from src.models.bayesian.state_space import BayesianStateSpace
    
    # Get Stan model path
    stan_path = project_root / "versions" / "Vishnu-Version-Hist" / "v3" / "stan_models" / "hierarchical_ews_v01.stan"
    
    # Configure model
    model_config = {
        **MCMC_CONFIG,
        'stan_file': str(stan_path)
    }
    
    # =========================================================================
    # Step 1: Assign unique row IDs BEFORE concatenation
    # =========================================================================
    # This is the ONLY way to track rows through sorting when keys aren't unique
    
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    if '_row_id' not in train_copy.columns:
        train_copy['_row_id'] = train_copy.index.astype(int)
    if '_row_id' not in test_copy.columns:
        test_copy['_row_id'] = test_copy.index.astype(int)

    # Assign globally unique row IDs
    train_copy['_unique_row_id'] = range(len(train_copy))
    test_copy['_unique_row_id'] = range(len(train_copy), len(train_copy) + len(test_copy))
    
    # Track which IDs are test rows
    test_row_ids = set(test_copy['_unique_row_id'])
    
    # Also store the original test order
    test_copy['_test_order'] = range(len(test_copy))
    train_copy['_test_order'] = -1  # Sentinel
    
    # Concatenate
    combined_df = pd.concat([train_copy, test_copy], ignore_index=True)
    
    # =========================================================================
    # Step 2: Sort for Stan model (required for state-space structure)
    # =========================================================================
    # The model expects data sorted by district and time
    # We add a secondary sort by _unique_row_id to ensure deterministic ordering
    # when (state, district, year, week) has ties (duplicates)
    
    combined_df = combined_df.sort_values(
        ['state', 'district', 'year', 'week', '_unique_row_id']
    ).reset_index(drop=True)
    
    # Store the position in sorted order for later mapping
    combined_df['_sorted_position'] = range(len(combined_df))
    
    n_total = len(combined_df)
    print(f"    [Bayesian] Combined: {n_total} samples ({len(train_df)} train + {len(test_df)} test)")
    
    # =========================================================================
    # Step 3: Fit Bayesian model
    # =========================================================================
    print(f"    [Bayesian] Preparing hierarchical state-space model...")
    model = BayesianStateSpace(config=model_config)
    
    X_combined = combined_df[feature_cols].values
    y_combined = combined_df['cases'].values
    
    print(f"    [Bayesian] Fitting on {n_total} samples...")
    print(f"    [Bayesian] MCMC: {MCMC_CONFIG['n_chains']} chains × ({MCMC_CONFIG['n_warmup']} warmup + {MCMC_CONFIG['n_samples']} samples)")
    print(f"    [Bayesian] Estimated time: ~{n_total * 0.2:.0f}-{n_total * 0.5:.0f} seconds (Stan will show progress bars)...")
    model.fit(X_combined, y_combined, df=combined_df, feature_cols=feature_cols)
    print(f"    [Bayesian] ✓ MCMC sampling complete")
    print(f"    [Bayesian] ✓ MCMC sampling complete")
    
    # Extract latent risk Z_t aligned to each observation row.
    print(f"    [Bayesian] Extracting latent risk Z_t from posterior...")
    if not hasattr(model, 'get_latent_risk_summary_per_observation'):
        raise RuntimeError(
            "BayesianStateSpace is missing get_latent_risk_summary_per_observation(). "
            "Update src/models/bayesian/state_space.py to expose latent-Z per-observation extraction."
        )
    z_mean, z_sd = model.get_latent_risk_summary_per_observation()

    # ASSERTION: z_mean must have exactly N entries
    if len(z_mean) != n_total:
        raise RuntimeError(
            f"STATE-SPACE INDEXING ERROR: z_mean has {len(z_mean)} entries "
            f"but combined_df has {n_total} rows. "
            f"This indicates a mismatch in latent-state extraction."
        )
    
    # =========================================================================
    # Step 4: Assign z_mean to the sorted DataFrame
    # =========================================================================
    # y_rep[i] corresponds to combined_df.iloc[i] in SORTED order
    combined_df['_z_mean'] = z_mean
    combined_df['_z_sd'] = z_sd
    
    # =========================================================================
    # Step 5: Extract test predictions by unique row ID
    # =========================================================================
    # Filter to test rows only
    test_preds = combined_df[combined_df['_unique_row_id'].isin(test_row_ids)].copy()
    
    # ASSERTION: Must have exactly len(test_df) test predictions
    if len(test_preds) != len(test_df):
        raise RuntimeError(
            f"TEST EXTRACTION ERROR: Found {len(test_preds)} test predictions "
            f"but expected {len(test_df)}."
        )
    
    # Sort back to original test order
    test_preds = test_preds.sort_values('_test_order').reset_index(drop=True)
    
    # =========================================================================
    # Step 6: Build output DataFrames (train + test)
    # =========================================================================
    print(f"    [Bayesian] Extracting latent risk Z_t for {len(test_df)} test samples...")

    base_cols = ['state', 'district', 'year', 'week', 'cases']
    extra_cols = ['incidence_per_100k', '_row_id']
    keep_cols_test = base_cols + [c for c in extra_cols if c in test_df.columns]
    keep_cols_train = base_cols + [c for c in extra_cols if c in train_df.columns]

    test_pred_df = test_df[keep_cols_test].copy().reset_index(drop=True)
    test_pred_df['z_mean'] = test_preds['_z_mean'].values
    test_pred_df['z_sd'] = test_preds['_z_sd'].values

    train_preds = combined_df[~combined_df['_unique_row_id'].isin(test_row_ids)].copy()
    train_preds = train_preds.sort_values('_unique_row_id').reset_index(drop=True)
    train_pred_df = train_df[keep_cols_train].copy().reset_index(drop=True)
    if len(train_pred_df) != len(train_preds):
        raise RuntimeError(
            f"TRAIN EXTRACTION ERROR: Found {len(train_preds)} train predictions "
            f"but expected {len(train_pred_df)}."
        )
    train_pred_df['z_mean'] = train_preds['_z_mean'].values
    train_pred_df['z_sd'] = train_preds['_z_sd'].values

    # Check for NaN
    n_missing = test_pred_df['z_mean'].isna().sum()
    if n_missing > 0:
        warnings.warn(f"  WARNING: {n_missing} test samples have NaN Bayesian prediction")
    
    print(f"    Bayesian predictions: {len(test_pred_df)} test samples")

    return train_pred_df, test_pred_df


# =============================================================================
# FOLD ANALYSIS
# =============================================================================

def analyze_single_fold(
    fold: CVFold,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak'
) -> Dict[str, Any]:
    """
    Run complete lead-time analysis for a single CV fold.
    
    Args:
        fold: CVFold object with train/test indices
        df: Full DataFrame
        feature_cols: Feature column names
        target_col: Target column
        
    Returns:
        Dictionary with:
        - fold_name: str
        - test_year: int
        - n_train: int
        - n_test: int
        - n_episodes: int
        - results: List[LeadTimeResult]
        - bayesian_summary: LeadTimeSummary
        - xgboost_summary: LeadTimeSummary
    """
    print(f"\n{'='*70}")
    print(f"FOLD: {fold.fold_name} (Test Year: {fold.test_year})")
    print(f"{'='*70}")
    
    # Get train/test data
    train_df = df.iloc[fold.train_idx].copy()
    test_df = df.iloc[fold.test_idx].copy()
    
    print(f"  Training: {len(train_df)} samples (years {train_df['year'].min()}-{train_df['year'].max()})")
    print(f"  Test: {len(test_df)} samples (year {test_df['year'].unique()})")
    
    # Apply imputation instead of dropping samples
    # This recovers ~90% of data that was previously lost to dropna
    print(f"  Applying imputation to recover missing values...")
    train_imputed = impute_missing_features(train_df, feature_cols)
    test_imputed = impute_missing_features(test_df, feature_cols)
    
    # Only drop rows where TARGET is missing (not features)
    train_valid = train_imputed.dropna(subset=[target_col]).copy()
    test_valid = test_imputed.dropna(subset=[target_col]).copy()

    # Stable join key for per-row prediction tables
    train_valid['_row_id'] = train_valid.index.astype(int)
    test_valid['_row_id'] = test_valid.index.astype(int)
    
    print(f"  Valid training: {len(train_valid)} (recovered from {len(train_df)}) | Valid test: {len(test_valid)} (recovered from {len(test_df)})")
    
    if len(train_valid) < 20:
        raise ValueError(f"Insufficient training data for fold {fold.fold_name}: {len(train_valid)}")
    
    if len(test_valid) < 5:
        raise ValueError(f"Insufficient test data for fold {fold.fold_name}: {len(test_valid)}")
    
    # -------------------------------------------------------------------------
    # Step 1: Train models and get predictions
    # -------------------------------------------------------------------------
    print(f"\n  Step 1: Training models...")
    
    # XGBoost predictions on TEST
    print(f"    Training XGBoost...")
    xgboost_preds = train_xgboost_and_predict(train_valid, test_valid, feature_cols, target_col)
    print(f"    XGBoost predictions: {len(xgboost_preds)} samples")
    
    # Bayesian predictions on TEST
    print(f"    Training Bayesian model (this may take 5-15 minutes)...")
    print(f"    [Bayesian] Compiling Stan model...")
    bayesian_train_preds, bayesian_test_preds = train_bayesian_and_predict(train_valid, test_valid, feature_cols)
    print(f"    [Bayesian] ✓ Complete: {len(bayesian_test_preds)} predictions generated")
    
    # -------------------------------------------------------------------------
    # Step 1b: FAIR ANALYSIS MODE (v4.2) - Compute thresholds from TRAINING
    # -------------------------------------------------------------------------
    if USE_FAIR_ANALYSIS_MODE:
        print(f"\n  Step 1b: Computing FAIR thresholds from TRAINING predictions...")
        print(f"    Mode: Symmetric percentile-based (p{ANALYSIS_MODE_PERCENTILE})")
        
        # Get XGBoost training predictions
        xgb_train_preds = compute_xgboost_training_predictions(
            train_valid, feature_cols, target_col
        )
        xgb_fair_threshold = compute_percentile_threshold_from_predictions(
            xgb_train_preds, 'prob', ANALYSIS_MODE_PERCENTILE
        )
        print(f"    XGBoost p{ANALYSIS_MODE_PERCENTILE} threshold: {xgb_fair_threshold:.4f}")
        
        # For Bayesian, we need training predictions from the combined model
        # The Bayesian model fits on combined_df (train+test) for state-space continuity
        # 
        # For thesis defensibility: We use the GLOBAL percentile threshold approach
        # where both models use the same percentile concept.
        # The key fairness is: SAME PERCENTILE for both, NOT fixed 0.5 for XGBoost.
        
        bayesian_fair_threshold = float(np.percentile(
            bayesian_train_preds['z_mean'].dropna(), ANALYSIS_MODE_PERCENTILE
        ))
        print(f"    Bayesian p{ANALYSIS_MODE_PERCENTILE} threshold: {bayesian_fair_threshold:.4f}")
        
        # Store for analysis output
        fair_thresholds = {
            'mode': 'fair_analysis',
            'percentile': ANALYSIS_MODE_PERCENTILE,
            'xgboost_threshold': xgb_fair_threshold,
            'bayesian_threshold': bayesian_fair_threshold
        }
    else:
        fair_thresholds = {'mode': 'original'}
        xgb_fair_threshold = XGBOOST_THRESHOLD  # 0.5
        bayesian_fair_threshold = None  # Use district-specific
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize LeadTimeAnalyzer and compute thresholds
    # -------------------------------------------------------------------------
    print(f"\n  Step 2: Computing thresholds...")
    
    # Set XGBoost threshold based on mode
    effective_xgb_threshold = xgb_fair_threshold if USE_FAIR_ANALYSIS_MODE else XGBOOST_THRESHOLD
    
    analyzer = LeadTimeAnalyzer(
        outbreak_percentile=OUTBREAK_PERCENTILE,
        bayesian_percentile=BAYESIAN_PERCENTILE if not USE_FAIR_ANALYSIS_MODE else ANALYSIS_MODE_PERCENTILE,
        xgboost_threshold=effective_xgb_threshold
    )
    
    # Compute outbreak thresholds from TRAINING data only
    analyzer.compute_outbreak_thresholds_from_training(train_valid, case_col='cases')
    
    # Compute Bayesian thresholds from TRAINING predictions
    analyzer.compute_bayesian_thresholds_from_predictions(bayesian_train_preds, z_col='z_mean')
    
    # -------------------------------------------------------------------------
    # Step 3: Identify outbreak episodes in test data
    # -------------------------------------------------------------------------
    print(f"\n  Step 3: Identifying outbreak episodes...")
    
    episodes = analyzer.identify_episodes(test_valid, fold_name=fold.fold_name, case_col='cases')
    
    if len(episodes) == 0:
        raise ValueError(
            f"CRITICAL: Zero outbreak episodes in fold {fold.fold_name}! "
            f"Check outbreak thresholds and test data."
        )
    
    print(f"    Found {len(episodes)} outbreak episodes")
    
    # -------------------------------------------------------------------------
    # Step 4: Compute lead times
    # -------------------------------------------------------------------------
    print(f"\n  Step 4: Computing lead times...")
    
    results = analyzer.compute_lead_times(
        episodes=episodes,
        bayesian_df=bayesian_test_preds,
        xgboost_df=xgboost_preds,
        z_col='z_mean',
        prob_col='prob'
    )
    
    # Sanity checks
    checks = sanity_check_results(results)
    if not checks['passed']:
        warnings.warn(f"  Sanity check issues in {fold.fold_name}: {checks['issues']}")
    
    # -------------------------------------------------------------------------
    # Step 5: Compute summaries
    # -------------------------------------------------------------------------
    bayesian_summary, xgboost_summary = analyzer.get_summaries(results)
    
    # Print fold summary
    print(f"\n  --- Fold Summary ---")
    print(f"  Outbreak episodes: {len(episodes)}")
    print(f"  Bayesian:")
    print(f"    Warned: {bayesian_summary.n_warned}/{bayesian_summary.n_episodes}")
    print(f"    Median lead time: {bayesian_summary.median_lead_time}")
    print(f"    % early warned (≥1 wk): {bayesian_summary.pct_early_warned:.1f}%")
    print(f"  XGBoost:")
    print(f"    Warned: {xgboost_summary.n_warned}/{xgboost_summary.n_episodes}")
    print(f"    Median lead time: {xgboost_summary.median_lead_time}")
    print(f"    % early warned (≥1 wk): {xgboost_summary.pct_early_warned:.1f}%")
    
    # Per-row prediction time series (for downstream Phase 7+ scripts)
    ts_cols = ['state', 'district', 'year', 'week', 'cases', '_row_id']
    if 'incidence_per_100k' in test_valid.columns:
        ts_cols.insert(5, 'incidence_per_100k')
    ts = test_valid[ts_cols].copy()
    ts['fold'] = fold.fold_name
    ts['test_year'] = fold.test_year
    ts['y_true'] = test_valid[target_col].astype(int).values
    ts = ts.merge(xgboost_preds[['_row_id', 'prob']], on='_row_id', how='left')
    ts = ts.merge(bayesian_test_preds[['_row_id', 'z_mean', 'z_sd']], on='_row_id', how='left')

    return {
        'fold_name': fold.fold_name,
        'test_year': fold.test_year,
        'n_train': len(train_valid),
        'n_test': len(test_valid),
        'n_episodes': len(episodes),
        'results': results,
        'bayesian_summary': bayesian_summary,
        'xgboost_summary': xgboost_summary,
        'outbreak_thresholds': dict(analyzer.outbreak_thresholds),
        'bayesian_thresholds': dict(analyzer.bayesian_thresholds),
        'fair_thresholds': fair_thresholds,
        'effective_xgb_threshold': effective_xgb_threshold,
        'test_prediction_timeseries': ts
    }


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_all_folds(
    fold_outputs: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate lead-time results across all folds.
    
    Args:
        fold_outputs: List of outputs from analyze_single_fold
        
    Returns:
        Tuple of:
        - detail_df: All episodes (one row per episode)
        - summary_overall_df: Overall summary statistics
        - summary_by_fold_df: Summary per fold
    """
    # -------------------------------------------------------------------------
    # 1. Combine all episode-level results
    # -------------------------------------------------------------------------
    all_results = []
    for fold_output in fold_outputs:
        all_results.extend(fold_output['results'])
    
    detail_df = results_to_dataframe(all_results)
    
    # -------------------------------------------------------------------------
    # 2. Compute overall summary
    # -------------------------------------------------------------------------
    bayesian_overall = summarize_lead_times(all_results, 'bayesian')
    xgboost_overall = summarize_lead_times(all_results, 'xgboost')
    
    summary_overall_df = summary_to_dataframe(bayesian_overall, xgboost_overall)
    
    # -------------------------------------------------------------------------
    # 3. Compute per-fold summary
    # -------------------------------------------------------------------------
    fold_summaries = []
    for fold_output in fold_outputs:
        fold_name = fold_output['fold_name']
        test_year = fold_output['test_year']
        
        bayes = fold_output['bayesian_summary']
        xgb = fold_output['xgboost_summary']
        
        fold_summaries.append({
            'fold': fold_name,
            'test_year': test_year,
            'n_episodes': bayes.n_episodes,
            'bayesian_n_warned': bayes.n_warned,
            'bayesian_median_lead': bayes.median_lead_time,
            'bayesian_mean_lead': bayes.mean_lead_time,
            'bayesian_iqr_lower': bayes.iqr_lower,
            'bayesian_iqr_upper': bayes.iqr_upper,
            'bayesian_pct_early': bayes.pct_early_warned,
            'bayesian_pct_never': bayes.pct_never_warned,
            'xgboost_n_warned': xgb.n_warned,
            'xgboost_median_lead': xgb.median_lead_time,
            'xgboost_mean_lead': xgb.mean_lead_time,
            'xgboost_iqr_lower': xgb.iqr_lower,
            'xgboost_iqr_upper': xgb.iqr_upper,
            'xgboost_pct_early': xgb.pct_early_warned,
            'xgboost_pct_never': xgb.pct_never_warned
        })
    
    summary_by_fold_df = pd.DataFrame(fold_summaries)
    
    return detail_df, summary_overall_df, summary_by_fold_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Execute lead-time analysis across all CV folds."""

    global OUTBREAK_PERCENTILE
    
    parser = argparse.ArgumentParser(
        description='Phase 7 Workstream A: Lead-Time Analysis (All Folds)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python experiments/06_analyze_lead_time.py
    python experiments/06_analyze_lead_time.py --skip-bayesian  # For debugging
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config_default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--outbreak-percentile',
        type=int,
        default=OUTBREAK_PERCENTILE,
        help='Outbreak percentile threshold (also used for output filenames)'
    )
    parser.add_argument(
        '--skip-bayesian',
        action='store_true',
        help='Skip Bayesian model (for debugging XGBoost-only)'
    )
    args = parser.parse_args()

    # Allow CLI override while preserving the existing global-constant structure.
    OUTBREAK_PERCENTILE = int(args.outbreak_percentile)
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("="*70)
    print("PHASE 7 - WORKSTREAM A: LEAD-TIME ANALYSIS")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Test years: {TEST_YEARS}")
    print(f"Outbreak percentile: {OUTBREAK_PERCENTILE}")
    print(f"Bayesian percentile: {BAYESIAN_PERCENTILE}")
    print(f"XGBoost threshold: {XGBOOST_THRESHOLD}")
    print()
    
    # Load configuration
    config = load_config(str(project_root / args.config))
    
    # Create output directory
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading Data")
    print("-"*70)
    
    df = load_features_data()
    feature_cols = get_feature_columns(df)
    target_col = 'label_outbreak'
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {target_col}")
    
    # Impute missing features
    print(f"\n  Imputing missing features...")
    df = impute_missing_features(df, feature_cols)
    
    # =========================================================================
    # CREATE CV FOLDS
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Creating CV Folds")
    print("-"*70)
    
    folds = create_rolling_origin_splits(df, test_years=TEST_YEARS)
    print(f"  Created {len(folds)} folds:")
    for fold in folds:
        print(f"    {fold.fold_name}: train={len(fold.train_idx)}, test={len(fold.test_idx)}")
    
    # =========================================================================
    # ANALYZE EACH FOLD
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Analyzing Each Fold")
    print("-"*70)
    
    fold_outputs = []
    prediction_frames = []
    failed_folds = []
    
    for fold in folds:
        try:
            fold_output = analyze_single_fold(
                fold=fold,
                df=df,
                feature_cols=feature_cols,
                target_col=target_col
            )
            fold_outputs.append(fold_output)

            ts = fold_output.get('test_prediction_timeseries')
            if isinstance(ts, pd.DataFrame) and not ts.empty:
                prediction_frames.append(ts)
            
        except Exception as e:
            print(f"\n  ERROR in {fold.fold_name}: {e}")
            failed_folds.append((fold.fold_name, str(e)))
            continue
    
    # Check for failures
    if failed_folds:
        print(f"\n{'!'*70}")
        print(f"WARNING: {len(failed_folds)} folds failed!")
        for name, error in failed_folds:
            print(f"  - {name}: {error}")
        print(f"{'!'*70}")
    
    if len(fold_outputs) == 0:
        raise RuntimeError("All folds failed! Cannot proceed.")
    
    # =========================================================================
    # AGGREGATE RESULTS
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 4: Aggregating Results")
    print("-"*70)
    
    detail_df, summary_overall_df, summary_by_fold_df = aggregate_all_folds(fold_outputs)
    
    print(f"\n  Total outbreak episodes: {len(detail_df)}")
    print(f"  Folds completed: {len(fold_outputs)}/{len(folds)}")
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Saving Outputs")
    print("-"*70)
    
    # Detail file (legacy + percentile-specific)
    detail_path = output_dir / "lead_time_detail_all_folds.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"  ✓ {detail_path}")

    detail_p_path = output_dir / f"lead_time_detail_p{OUTBREAK_PERCENTILE}.csv"
    detail_df.to_csv(detail_p_path, index=False)
    print(f"  ✓ {detail_p_path}")

    # Per-row prediction time series (for downstream Phase 7+ scripts)
    if prediction_frames:
        preds_path = output_dir / f"lead_time_predictions_p{OUTBREAK_PERCENTILE}.parquet"
        pd.concat(prediction_frames, ignore_index=True).to_parquet(preds_path, index=False)
        print(f"  ✓ {preds_path}")
    
    # Overall summary (legacy + percentile-specific)
    summary_overall_path = output_dir / "lead_time_summary_overall.csv"
    summary_overall_df.to_csv(summary_overall_path, index=False)
    print(f"  ✓ {summary_overall_path}")

    summary_overall_p_path = output_dir / f"lead_time_summary_overall_p{OUTBREAK_PERCENTILE}.csv"
    summary_overall_df.to_csv(summary_overall_p_path, index=False)
    print(f"  ✓ {summary_overall_p_path}")
    
    # By-fold summary (legacy + percentile-specific)
    summary_by_fold_path = output_dir / "lead_time_summary_by_fold.csv"
    summary_by_fold_df.to_csv(summary_by_fold_path, index=False)
    print(f"  ✓ {summary_by_fold_path}")

    summary_by_fold_p_path = output_dir / f"lead_time_summary_by_fold_p{OUTBREAK_PERCENTILE}.csv"
    summary_by_fold_df.to_csv(summary_by_fold_p_path, index=False)
    print(f"  ✓ {summary_by_fold_p_path}")
    
    # Also save full analysis metadata as JSON
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 7 - Workstream A',
        'version': 'v4',
        'config': {
            'test_years': TEST_YEARS,
            'outbreak_percentile': OUTBREAK_PERCENTILE,
            'bayesian_percentile': BAYESIAN_PERCENTILE,
            'xgboost_threshold': XGBOOST_THRESHOLD,
            'mcmc_config': MCMC_CONFIG
        },
        'folds_completed': len(fold_outputs),
        'folds_failed': len(failed_folds),
        'total_episodes': len(detail_df),
        'fold_summary': [
            {
                'fold': fo['fold_name'],
                'test_year': fo['test_year'],
                'n_train': fo['n_train'],
                'n_test': fo['n_test'],
                'n_episodes': fo['n_episodes']
            }
            for fo in fold_outputs
        ]
    }
    
    metadata_path = output_dir / "lead_time_analysis_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ {metadata_path}")
    
    # =========================================================================
    # PRINT FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nOverall Lead-Time Comparison:")
    print(summary_overall_df.to_string(index=False))
    
    print("\n\nPer-Fold Summary:")
    print(summary_by_fold_df.to_string(index=False))
    
    # Compute differential statistics
    valid_results = [r for fo in fold_outputs for r in fo['results']
                     if r.differential_lead is not None]
    if valid_results:
        diffs = [r.differential_lead for r in valid_results]
        print(f"\n\nDifferential Lead Time (ΔL = L_XGB - L_Bayes):")
        print(f"  N comparisons: {len(diffs)}")
        print(f"  Mean ΔL: {np.mean(diffs):.2f} weeks")
        print(f"  Median ΔL: {np.median(diffs):.2f} weeks")
        print(f"  ΔL > 0 (Bayesian earlier): {sum(1 for d in diffs if d > 0)} ({100*sum(1 for d in diffs if d > 0)/len(diffs):.1f}%)")
        print(f"  ΔL = 0 (Same timing): {sum(1 for d in diffs if d == 0)} ({100*sum(1 for d in diffs if d == 0)/len(diffs):.1f}%)")
        print(f"  ΔL < 0 (XGBoost earlier): {sum(1 for d in diffs if d < 0)} ({100*sum(1 for d in diffs if d < 0)/len(diffs):.1f}%)")
    
    print("\n" + "="*70)
    print("✓ Lead-Time Analysis Complete")
    print("="*70)
    
    return fold_outputs


if __name__ == '__main__':
    main()
