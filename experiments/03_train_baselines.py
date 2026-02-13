#!/usr/bin/env python3
"""
Experiment 03: Train Baseline Models

Trains all Track A baseline models with temporal CV:
- Threshold rule
- Logistic Regression
- Random Forest
- XGBoost

Output: results/metrics/baseline_comparison.json

Usage:
    python experiments/03_train_baselines.py
    python experiments/03_train_baselines.py --config config/config_baseline.yaml
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.config import load_config, get_project_root
from src.features.feature_sets import select_feature_columns
from src.evaluation.cv import create_rolling_origin_splits, prepare_train_test
from src.evaluation.metrics import compute_all_metrics, print_metrics

# Import models
from src.models.baselines.threshold import ThresholdBaseline
from src.models.baselines.logistic import LogisticBaseline
from src.models.baselines.random_forest import RandomForestBaseline

# XGBoost is optional (requires libomp on Mac)
try:
    from src.models.baselines.xgboost_model import XGBoostBaseline
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Warning: XGBoost not available ({e})")
    XGBOOST_AVAILABLE = False


def get_baseline_models(config: dict) -> dict:
    """Create all baseline model instances."""
    model_cfg = config.get('models', {}).get('baselines', {})
    
    models = {
        'threshold': ThresholdBaseline(model_cfg.get('threshold', {})),
        'logistic': LogisticBaseline(model_cfg.get('logistic', {})),
        'random_forest': RandomForestBaseline(model_cfg.get('random_forest', {})),
    }
    
    # Only add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBoostBaseline(model_cfg.get('xgboost', {}))
    
    return models


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Train model and compute metrics."""
    # Fit
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    metrics = compute_all_metrics(y_test, y_pred_proba)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_default.yaml",
        help="Path to config file"
    )
    # Feature set selection: FULL=37 features (all feat_*), CORE=20 features (sparse-robust subset)
    # Default is "core" for feature ablation experiment
    # To test with full features, run: python 03_train_baselines.py --feature-set=full
    parser.add_argument(
        "--feature-set",
        type=str,
        default="core",
        choices=["full", "core"],
        help="Which feature set to use: full (all feat_*) or core (sparse-robust subset)"
    )
    args = parser.parse_args()
    
    # Load config
    root = get_project_root()
    config_path = root / args.config
    cfg = load_config(str(config_path))
    
    # Load features
    features_path = root / cfg['data']['processed']['features']
    print("=" * 60)
    print("CHIKUNGUNYA EWS - TRAIN BASELINES")
    print("=" * 60)
    
    print(f"\nLoading features from {features_path}...")
    df = pd.read_parquet(features_path)
    print(f"  → {len(df)} rows, {len(df.columns)} columns")
    
    # Get feature columns
    feature_cols = select_feature_columns(df.columns, feature_set=args.feature_set)
    target_col = 'label_outbreak'
    
    print(f"  → Feature set: {args.feature_set}")
    print(f"  → {len(feature_cols)} features")
    print(f"  → Target: {target_col}")
    
    # Impute features with neutral values where missing
    # EWS features require baseline periods that don't exist for sparse districts
    # Climate features (LAI, rain, temp) have some missing raw values
    impute_map = {
        # EWS features - neutral values
        'feat_var_spike_ratio': 1.0,  # ratio of 1 = no spike
        'feat_acf_change': 0.0,       # no change in autocorrelation
        'feat_trend_accel': 0.0,      # no acceleration
        # LAI features - median imputation (0 is reasonable for sparse vegetation data)
        'feat_lai': 0.0,
        'feat_lai_lag_1': 0.0,
        'feat_lai_lag_2': 0.0,
        'feat_lai_lag_4': 0.0,
    }
    
    # Past-only imputation for climate lag features.
    # IMPORTANT: do NOT backfill (bfill) because it leaks future values into the past.
    climate_lag_cols = [c for c in feature_cols if any(x in c for x in ['temp_lag', 'rain_lag', 'rain_persist', 'temp_anomaly'])]
    for col in climate_lag_cols:
        if col in df.columns:
            # Group-wise forward fill (past-only)
            df[col] = df.groupby(['state', 'district'])[col].transform(
                lambda x: x.ffill()
            )
    
    # Apply neutral value imputation
    for col, neutral_val in impute_map.items():
        if col in df.columns:
            n_imputed = df[col].isna().sum()
            if n_imputed > 0:
                df[col] = df[col].fillna(neutral_val)
                print(f"  → Imputed {n_imputed} missing {col} with {neutral_val}")
    
    # Filter to valid samples.
    # IMPORTANT: Do not drop rows due to missing engineered features.
    # Many mechanistic/EWS features are undefined early in a district history
    # (rolling windows / long baselines). The baseline models handle NaNs via
    # neutral/forward-fill imputation above and `np.nan_to_num` inside each model.
    valid_df = df.dropna(subset=[target_col])
    print(f"  → {len(valid_df)} labeled samples")
    
    # Create CV splits
    test_years = cfg['cv']['test_years']
    folds = create_rolling_origin_splits(valid_df, test_years=test_years)
    print(f"\nCV Folds: {len(folds)}")
    for fold in folds:
        print(f"  {fold.fold_name}: train={len(fold.train_idx)}, test={len(fold.test_idx)}")
    
    # Get models
    models = get_baseline_models(cfg)
    print(f"\nModels: {list(models.keys())}")
    
    # Results storage
    all_results = {}
    
    # Train and evaluate each model
    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION")
    print("=" * 60)
    
    for model_name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*40}")
        
        fold_metrics = []
        
        for fold in folds:
            # Get train/test data
            train_df = valid_df.iloc[fold.train_idx]
            test_df = valid_df.iloc[fold.test_idx]
            
            X_train, y_train, X_test, y_test = prepare_train_test(
                train_df, test_df, feature_cols, target_col, drop_na=False
            )
            
            # Skip if insufficient data
            if len(X_train) < 10 or len(X_test) < 5:
                print(f"  {fold.fold_name}: Skipping (insufficient data)")
                continue
            
            if len(np.unique(y_train)) < 2:
                print(f"  {fold.fold_name}: Skipping (single class in train)")
                continue
            
            # Train and evaluate
            try:
                # Re-instantiate model for fresh training
                if model_name == 'threshold':
                    fold_model = ThresholdBaseline(cfg.get('models', {}).get('baselines', {}).get('threshold', {}))
                elif model_name == 'logistic':
                    fold_model = LogisticBaseline(cfg.get('models', {}).get('baselines', {}).get('logistic', {}))
                elif model_name == 'random_forest':
                    fold_model = RandomForestBaseline(cfg.get('models', {}).get('baselines', {}).get('random_forest', {}))
                else:
                    fold_model = XGBoostBaseline(cfg.get('models', {}).get('baselines', {}).get('xgboost', {}))
                
                metrics = train_and_evaluate(fold_model, X_train, y_train, X_test, y_test)
                metrics['fold'] = fold.fold_name
                metrics['test_year'] = fold.test_year
                fold_metrics.append(metrics)
                
                print(f"  {fold.fold_name}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, n={metrics['n_samples']}")
                
            except Exception as e:
                print(f"  {fold.fold_name}: ERROR - {e}")
                continue
        
        # Aggregate metrics
        if fold_metrics:
            agg = {
                'auc_mean': np.mean([m['auc'] for m in fold_metrics if not np.isnan(m['auc'])]),
                'auc_std': np.std([m['auc'] for m in fold_metrics if not np.isnan(m['auc'])]),
                'f1_mean': np.mean([m['f1'] for m in fold_metrics]),
                'f1_std': np.std([m['f1'] for m in fold_metrics]),
                'sensitivity_mean': np.mean([m['sensitivity'] for m in fold_metrics]),
                'specificity_mean': np.mean([m['specificity'] for m in fold_metrics]),
                'brier_mean': np.mean([m['brier'] for m in fold_metrics if not np.isnan(m['brier'])]),
                'n_folds': len(fold_metrics),
                'fold_results': fold_metrics
            }
            all_results[model_name] = agg
            
            print(f"\n  AGGREGATE: AUC={agg['auc_mean']:.3f}±{agg['auc_std']:.3f}, F1={agg['f1_mean']:.3f}±{agg['f1_std']:.3f}")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<15} {'AUC':>10} {'F1':>10} {'Sens':>10} {'Spec':>10} {'Brier':>10}")
    print("-" * 65)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<15} {results['auc_mean']:>10.3f} {results['f1_mean']:>10.3f} "
              f"{results['sensitivity_mean']:>10.3f} {results['specificity_mean']:>10.3f} "
              f"{results['brier_mean']:>10.3f}")
    
    # Save results
    results_dir = root / cfg['output']['metrics_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / 'baseline_comparison.json'
    
    # Convert to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable_results = {
        'timestamp': datetime.now().isoformat(),
        'config': str(config_path),
        'n_features': len(feature_cols),
        'n_samples': len(valid_df),
        'cv_folds': len(folds),
        'models': {}
    }
    
    for model_name, results in all_results.items():
        serializable_results['models'][model_name] = {
            k: convert_to_serializable(v) if not isinstance(v, list) else v
            for k, v in results.items()
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=convert_to_serializable)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    main()
