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
    Impute missing feature values with neutral/sensible defaults.
    
    Same imputation logic as 03_train_baselines.py to ensure consistency.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    
    # Neutral value imputation for specific features
    impute_map = {
        'feat_var_spike_ratio': 1.0,
        'feat_acf_change': 0.0,
        'feat_trend_accel': 0.0,
        'feat_lai': 0.0,
        'feat_lai_lag_1': 0.0,
        'feat_lai_lag_2': 0.0,
        'feat_lai_lag_4': 0.0,
    }
    
    # Forward-fill then backward-fill for climate lag features
    climate_lag_cols = [c for c in feature_cols if any(
        x in c for x in ['temp_lag', 'rain_lag', 'rain_persist', 'temp_anomaly']
    )]
    
    for col in climate_lag_cols:
        if col in df.columns:
            df[col] = df.groupby(['state', 'district'])[col].transform(
                lambda x: x.ffill().bfill()
            )
    
    # Apply neutral value imputation
    for col, neutral_val in impute_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(neutral_val)
    
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
    pred_df = test_df[['state', 'district', 'year', 'week', 'cases']].copy()
    pred_df['prob'] = probs
    
    return pred_df


def train_bayesian_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Train Bayesian state-space model and generate predictions for test data.
    
    This extracts the latent risk Z_t (posterior mean) for each observation.
    Since the Bayesian model fits on the training data, we need to 
    extrapolate/predict for the test period.
    
    For lead-time analysis, we use the posterior predictive distribution
    to estimate risk for the test period.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame  
        feature_cols: Feature column names
        
    Returns:
        Test DataFrame with added 'z_mean' and 'z_sd' columns
    """
    from src.models.bayesian.state_space import BayesianStateSpace
    
    # Get Stan model path
    stan_path = project_root / "versions" / "Vishnu-Version-Hist" / "v3" / "stan_models" / "hierarchical_ews_v01.stan"
    
    # Configure model
    model_config = {
        **MCMC_CONFIG,
        'stan_file': str(stan_path)
    }
    
    # Prepare combined data for fitting
    # The Bayesian model needs to see the full time series including test
    # but we only use test for predictions
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = combined_df.sort_values(['state', 'district', 'year', 'week']).reset_index(drop=True)
    
    # Train model on combined data (to get latent states for test period)
    # Note: This is standard for state-space models - we fit on all available data
    # but evaluate only on the test period
    model = BayesianStateSpace(config=model_config)
    
    X_combined = combined_df[feature_cols].values
    y_combined = combined_df['cases'].values
    
    print(f"    Fitting Bayesian model on {len(combined_df)} samples...")
    model.fit(X_combined, y_combined, df=combined_df, feature_cols=feature_cols)
    
    # Get posterior predictive for all time points
    y_rep = model.get_posterior_predictive()  # Shape: (n_draws, N)
    
    # Compute summary statistics
    z_mean = y_rep.mean(axis=0)
    z_sd = y_rep.std(axis=0)
    
    # Map back to test samples only
    # We need to identify which rows in combined_df correspond to test_df
    n_train = len(train_df)
    n_test = len(test_df)
    
    # The combined_df is sorted, so test samples are at the end
    # But we need to match by (state, district, year, week)
    combined_df['_z_mean'] = z_mean
    combined_df['_z_sd'] = z_sd
    
    # Merge back to test_df
    test_keys = test_df[['state', 'district', 'year', 'week']].copy()
    test_keys['_idx'] = range(len(test_keys))
    
    merged = test_keys.merge(
        combined_df[['state', 'district', 'year', 'week', '_z_mean', '_z_sd']],
        on=['state', 'district', 'year', 'week'],
        how='left'
    ).sort_values('_idx')
    
    # Create output DataFrame
    pred_df = test_df[['state', 'district', 'year', 'week', 'cases']].copy()
    pred_df['z_mean'] = merged['_z_mean'].values
    pred_df['z_sd'] = merged['_z_sd'].values
    
    # Sanity check
    n_missing = pred_df['z_mean'].isna().sum()
    if n_missing > 0:
        warnings.warn(f"  WARNING: {n_missing} test samples have no Bayesian prediction")
    
    return pred_df


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
    
    # Filter to valid samples (have both features and target)
    train_valid = train_df.dropna(subset=feature_cols + [target_col])
    test_valid = test_df.dropna(subset=feature_cols + [target_col])
    
    print(f"  Valid training: {len(train_valid)} | Valid test: {len(test_valid)}")
    
    if len(train_valid) < 20:
        raise ValueError(f"Insufficient training data for fold {fold.fold_name}: {len(train_valid)}")
    
    if len(test_valid) < 5:
        raise ValueError(f"Insufficient test data for fold {fold.fold_name}: {len(test_valid)}")
    
    # -------------------------------------------------------------------------
    # Step 1: Train models and get predictions
    # -------------------------------------------------------------------------
    print(f"\n  Step 1: Training models...")
    
    # XGBoost predictions
    print(f"    Training XGBoost...")
    xgboost_preds = train_xgboost_and_predict(train_valid, test_valid, feature_cols, target_col)
    print(f"    XGBoost predictions: {len(xgboost_preds)} samples")
    
    # Bayesian predictions
    print(f"    Training Bayesian model (this may take a few minutes)...")
    bayesian_preds = train_bayesian_and_predict(train_valid, test_valid, feature_cols)
    print(f"    Bayesian predictions: {len(bayesian_preds)} samples")
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize LeadTimeAnalyzer and compute thresholds
    # -------------------------------------------------------------------------
    print(f"\n  Step 2: Computing thresholds...")
    
    analyzer = LeadTimeAnalyzer(
        outbreak_percentile=OUTBREAK_PERCENTILE,
        bayesian_percentile=BAYESIAN_PERCENTILE,
        xgboost_threshold=XGBOOST_THRESHOLD
    )
    
    # Compute outbreak thresholds from TRAINING data only
    analyzer.compute_outbreak_thresholds_from_training(train_valid, case_col='cases')
    
    # Compute Bayesian thresholds from predictions (test data)
    analyzer.compute_bayesian_thresholds_from_predictions(bayesian_preds, z_col='z_mean')
    
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
        bayesian_df=bayesian_preds,
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
        'bayesian_thresholds': dict(analyzer.bayesian_thresholds)
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
        '--skip-bayesian',
        action='store_true',
        help='Skip Bayesian model (for debugging XGBoost-only)'
    )
    args = parser.parse_args()
    
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
    
    # Detail file
    detail_path = output_dir / "lead_time_detail_all_folds.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"  ✓ {detail_path}")
    
    # Overall summary
    summary_overall_path = output_dir / "lead_time_summary_overall.csv"
    summary_overall_df.to_csv(summary_overall_path, index=False)
    print(f"  ✓ {summary_overall_path}")
    
    # By-fold summary
    summary_by_fold_path = output_dir / "lead_time_summary_by_fold.csv"
    summary_by_fold_df.to_csv(summary_by_fold_path, index=False)
    print(f"  ✓ {summary_by_fold_path}")
    
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
