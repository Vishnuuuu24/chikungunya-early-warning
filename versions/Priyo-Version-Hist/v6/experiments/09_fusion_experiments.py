#!/usr/bin/env python3
"""
Phase 6 - Task 4: Fusion Experiments

Implement hybrid strategies combining Bayesian latent risk with ML classifiers.

Fusion Strategies:

Strategy A - Feature Fusion:
  Inject Bayesian outputs as features into XGBoost/CatBoost:
  - latent_risk_mean
  - latent_risk_std (uncertainty)
  - prob_high_risk
  Evaluate with AUPR, Precision, Recall, F1

Strategy B - Gated Decision Fusion:
  If Bayesian risk is high (RED zone) → Use Bayesian decision
  Else → Use ML prediction
  Evaluate decision quality and stability

Strategy C - Weighted Ensemble:
  weighted_prob = α * P_bayes + (1-α) * P_ml
  Optimize α for best performance

Output: results/analysis/fusion_results.json

Reference: Phase 6 faculty requirement for ML-Bayesian integration
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score,
    average_precision_score, cohen_kappa_score,
    brier_score_loss
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.evaluation.cv import create_rolling_origin_splits
from src.evaluation.metrics import compute_all_metrics

# Add v3 code path for Bayesian model
v3_code_path = project_root / "versions" / "Vishnu-Version-Hist" / "v3" / "code"
sys.path.insert(0, str(v3_code_path))

# Add v1.1 code path for XGBoost model
v1_1_code_path = project_root / "versions" / "Vishnu-Version-Hist" / "v1.1" / "code"
sys.path.insert(0, str(v1_1_code_path))


# =============================================================================
# CONFIGURATION
# =============================================================================

MCMC_CONFIG = {
    'n_warmup': 1000,
    'n_samples': 1000,
    'n_chains': 4,
    'adapt_delta': 0.95,
    'seed': 42
}

XGBOOST_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'auc'
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_processed_data() -> pd.DataFrame:
    """Load feature-engineered data."""
    data_path = project_root / "data" / "processed" / "features_engineered_v01.parquet"
    df = pd.read_parquet(data_path)
    return df


def compute_bayesian_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit Bayesian model and extract risk features for train and test sets.
    
    Returns augmented train_df and test_df with Bayesian features:
    - bayes_latent_risk_mean
    - bayes_latent_risk_std
    - bayes_prob_high_risk
    """
    from src.models.bayesian.state_space import BayesianStateSpace
    
    # Prepare training data
    valid_train = train_df.dropna(subset=['state', 'district', 'year', 'week', 'cases']).copy()
    if 'temp_celsius' in valid_train.columns:
        valid_train = valid_train.dropna(subset=['temp_celsius'])
    
    X_train = valid_train[feature_cols].values
    y_train = valid_train['cases'].values
    
    # Fit Bayesian model
    # NOTE: v6 uses Vishnu's v3 stabilized model (v0.2: phi>0.1, tighter rho prior)
    stan_path = project_root / "versions" / "Vishnu-Version-Hist" / "v3" / "stan_models" / "hierarchical_ews_v01.stan"
    
    model = BayesianStateSpace(
        stan_model_path=str(stan_path),
        n_warmup=MCMC_CONFIG['n_warmup'],
        n_samples=MCMC_CONFIG['n_samples'],
        n_chains=MCMC_CONFIG['n_chains'],
        adapt_delta=MCMC_CONFIG['adapt_delta'],
        seed=MCMC_CONFIG['seed']
    )
    
    print(f"  Fitting Bayesian model on {len(valid_train)} samples...")
    model.fit(X_train, y_train, df=valid_train, feature_cols=feature_cols)
    
    # Extract posterior predictive for training data
    y_rep_train = model.get_posterior_predictive()  # (n_draws, n_train)
    
    # Compute risk threshold (80th percentile)
    risk_threshold = np.percentile(y_rep_train.mean(axis=0), 80)
    
    # Compute Bayesian features for training set
    valid_train['bayes_latent_risk_mean'] = y_rep_train.mean(axis=0)
    valid_train['bayes_latent_risk_std'] = y_rep_train.std(axis=0)
    valid_train['bayes_prob_high_risk'] = (y_rep_train > risk_threshold).mean(axis=0)
    
    # For test set, we need to propagate features
    # Since we can't directly predict on new data with this state-space model,
    # we'll use district-level aggregates from training as proxies
    
    district_bayes = valid_train.groupby(['state', 'district']).agg({
        'bayes_latent_risk_mean': 'mean',
        'bayes_latent_risk_std': 'mean',
        'bayes_prob_high_risk': 'mean'
    }).reset_index()
    
    # Merge with test set
    test_augmented = test_df.merge(
        district_bayes,
        on=['state', 'district'],
        how='left'
    )
    
    # Fill missing with global means
    for col in ['bayes_latent_risk_mean', 'bayes_latent_risk_std', 'bayes_prob_high_risk']:
        test_augmented[col] = test_augmented[col].fillna(valid_train[col].mean())
    
    # Merge back with train
    train_augmented = train_df.merge(
        valid_train[['state', 'district', 'year', 'week', 
                     'bayes_latent_risk_mean', 'bayes_latent_risk_std', 'bayes_prob_high_risk']],
        on=['state', 'district', 'year', 'week'],
        how='left'
    )
    
    # Fill missing in train with means
    for col in ['bayes_latent_risk_mean', 'bayes_latent_risk_std', 'bayes_prob_high_risk']:
        train_augmented[col] = train_augmented[col].fillna(valid_train[col].mean())
    
    return train_augmented, test_augmented


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Train XGBoost baseline and return AUC and predictions."""
    from xgboost import XGBClassifier
    
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = 0.5
    
    return auc_score, y_pred_proba


def fusion_strategy_a_feature_fusion(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_feature_cols: List[str],
    target_col: str = 'outbreak_p75'
) -> Dict[str, Any]:
    """
    Strategy A: Feature Fusion
    Inject Bayesian features into XGBoost.
    """
    print("\n  Strategy A: Feature Fusion (XGBoost + Bayesian features)")
    
    # Compute Bayesian features
    train_aug, test_aug = compute_bayesian_features(train_df, test_df, base_feature_cols)
    
    # Prepare data
    bayes_feature_cols = ['bayes_latent_risk_mean', 'bayes_latent_risk_std', 'bayes_prob_high_risk']
    fusion_feature_cols = base_feature_cols + bayes_feature_cols
    
    # Drop NaN in target
    train_valid = train_aug.dropna(subset=[target_col])
    test_valid = test_aug.dropna(subset=[target_col])
    
    X_train_fusion = train_valid[fusion_feature_cols].values
    y_train = train_valid[target_col].values.astype(int)
    X_test_fusion = test_valid[fusion_feature_cols].values
    y_test = test_valid[target_col].values.astype(int)
    
    # Train XGBoost with fused features
    from xgboost import XGBClassifier
    
    model_fusion = XGBClassifier(**XGBOOST_PARAMS)
    model_fusion.fit(X_train_fusion, y_train)
    
    y_pred_proba_fusion = model_fusion.predict_proba(X_test_fusion)[:, 1]
    y_pred_fusion = (y_pred_proba_fusion >= 0.5).astype(int)
    
    # Compute metrics
    metrics = {}
    
    if len(np.unique(y_test)) > 1:
        metrics['auc'] = roc_auc_score(y_test, y_pred_proba_fusion)
        metrics['aupr'] = average_precision_score(y_test, y_pred_proba_fusion)
    else:
        metrics['auc'] = 0.5
        metrics['aupr'] = 0.0
    
    metrics['precision'] = precision_score(y_test, y_pred_fusion, zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred_fusion, zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred_fusion, zero_division=0)
    metrics['kappa'] = cohen_kappa_score(y_test, y_pred_fusion)
    metrics['brier'] = brier_score_loss(y_test, y_pred_proba_fusion)
    
    print(f"    AUC: {metrics['auc']:.3f}")
    print(f"    AUPR: {metrics['aupr']:.3f}")
    print(f"    F1: {metrics['f1']:.3f}")
    
    return {
        'strategy': 'feature_fusion',
        'n_base_features': len(base_feature_cols),
        'n_bayes_features': len(bayes_feature_cols),
        'n_total_features': len(fusion_feature_cols),
        'metrics': metrics
    }


def fusion_strategy_b_gated_decision(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_feature_cols: List[str],
    target_col: str = 'outbreak_p75'
) -> Dict[str, Any]:
    """
    Strategy B: Gated Decision Fusion
    If Bayesian risk is high → Use Bayesian decision
    Else → Use XGBoost prediction
    """
    print("\n  Strategy B: Gated Decision Fusion")
    
    # Compute Bayesian features
    train_aug, test_aug = compute_bayesian_features(train_df, test_df, base_feature_cols)
    
    # Train baseline XGBoost
    train_valid = train_aug.dropna(subset=[target_col])
    test_valid = test_aug.dropna(subset=[target_col])
    
    X_train = train_valid[base_feature_cols].values
    y_train = train_valid[target_col].values.astype(int)
    X_test = test_valid[base_feature_cols].values
    y_test = test_valid[target_col].values.astype(int)
    
    from xgboost import XGBClassifier
    
    model_xgb = XGBClassifier(**XGBOOST_PARAMS)
    model_xgb.fit(X_train, y_train)
    
    y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
    
    # Gated fusion logic
    # If bayes_prob_high_risk >= 0.8 (RED zone) → Use Bayesian probability
    # Else → Use XGBoost probability
    
    bayes_proba = test_valid['bayes_prob_high_risk'].values
    gated_threshold = 0.8
    
    y_pred_proba_gated = np.where(
        bayes_proba >= gated_threshold,
        bayes_proba,  # Use Bayesian
        y_pred_proba_xgb  # Use XGBoost
    )
    
    y_pred_gated = (y_pred_proba_gated >= 0.5).astype(int)
    
    # Compute metrics
    metrics = {}
    
    if len(np.unique(y_test)) > 1:
        metrics['auc'] = roc_auc_score(y_test, y_pred_proba_gated)
        metrics['aupr'] = average_precision_score(y_test, y_pred_proba_gated)
    else:
        metrics['auc'] = 0.5
        metrics['aupr'] = 0.0
    
    metrics['precision'] = precision_score(y_test, y_pred_gated, zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred_gated, zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred_gated, zero_division=0)
    metrics['kappa'] = cohen_kappa_score(y_test, y_pred_gated)
    metrics['brier'] = brier_score_loss(y_test, y_pred_proba_gated)
    
    # Count how often Bayesian gate was used
    n_bayes_used = (bayes_proba >= gated_threshold).sum()
    bayes_usage_rate = n_bayes_used / len(bayes_proba)
    
    print(f"    Bayesian gate used: {bayes_usage_rate*100:.1f}% of cases")
    print(f"    AUC: {metrics['auc']:.3f}")
    print(f"    AUPR: {metrics['aupr']:.3f}")
    print(f"    F1: {metrics['f1']:.3f}")
    
    return {
        'strategy': 'gated_decision',
        'gate_threshold': gated_threshold,
        'bayes_usage_rate': bayes_usage_rate,
        'metrics': metrics
    }


def fusion_strategy_c_weighted_ensemble(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_feature_cols: List[str],
    target_col: str = 'outbreak_p75'
) -> Dict[str, Any]:
    """
    Strategy C: Weighted Ensemble
    weighted_prob = α * P_bayes + (1-α) * P_xgboost
    Try different α values.
    """
    print("\n  Strategy C: Weighted Ensemble")
    
    # Compute Bayesian features
    train_aug, test_aug = compute_bayesian_features(train_df, test_df, base_feature_cols)
    
    # Train XGBoost
    train_valid = train_aug.dropna(subset=[target_col])
    test_valid = test_aug.dropna(subset=[target_col])
    
    X_train = train_valid[base_feature_cols].values
    y_train = train_valid[target_col].values.astype(int)
    X_test = test_valid[base_feature_cols].values
    y_test = test_valid[target_col].values.astype(int)
    
    from xgboost import XGBClassifier
    
    model_xgb = XGBClassifier(**XGBOOST_PARAMS)
    model_xgb.fit(X_train, y_train)
    
    y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
    y_pred_proba_bayes = test_valid['bayes_prob_high_risk'].values
    
    # Try different α values
    alpha_values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = []
    
    for alpha in alpha_values:
        y_pred_proba_weighted = alpha * y_pred_proba_bayes + (1 - alpha) * y_pred_proba_xgb
        y_pred_weighted = (y_pred_proba_weighted >= 0.5).astype(int)
        
        metrics = {}
        
        if len(np.unique(y_test)) > 1:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba_weighted)
            metrics['aupr'] = average_precision_score(y_test, y_pred_proba_weighted)
        else:
            metrics['auc'] = 0.5
            metrics['aupr'] = 0.0
        
        metrics['precision'] = precision_score(y_test, y_pred_weighted, zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred_weighted, zero_division=0)
        metrics['f1'] = f1_score(y_test, y_pred_weighted, zero_division=0)
        
        results.append({
            'alpha': alpha,
            'metrics': metrics
        })
    
    # Find best α
    best_result = max(results, key=lambda x: x['metrics']['aupr'])
    
    print(f"    Best α: {best_result['alpha']}")
    print(f"    Best AUPR: {best_result['metrics']['aupr']:.3f}")
    print(f"    Best AUC: {best_result['metrics']['auc']:.3f}")
    
    return {
        'strategy': 'weighted_ensemble',
        'alpha_grid': alpha_values,
        'results': results,
        'best_alpha': best_result['alpha'],
        'best_metrics': best_result['metrics']
    }


# =============================================================================
# MAIN FUSION EXPERIMENTS
# =============================================================================

def run_fusion_experiments_for_fold(
    df: pd.DataFrame,
    fold_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Run all fusion strategies for one CV fold.
    """
    print(f"\n{'='*60}")
    print(f"Fusion Experiments: {fold_name}")
    print(f"{'='*60}")
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    results = {
        'fold': fold_name,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'strategies': {}
    }
    
    # Run each strategy
    try:
        results['strategies']['feature_fusion'] = fusion_strategy_a_feature_fusion(
            train_df, test_df, feature_cols
        )
    except Exception as e:
        print(f"    ✗ Feature fusion failed: {e}")
        results['strategies']['feature_fusion'] = {'error': str(e)}
    
    try:
        results['strategies']['gated_decision'] = fusion_strategy_b_gated_decision(
            train_df, test_df, feature_cols
        )
    except Exception as e:
        print(f"    ✗ Gated decision failed: {e}")
        results['strategies']['gated_decision'] = {'error': str(e)}
    
    try:
        results['strategies']['weighted_ensemble'] = fusion_strategy_c_weighted_ensemble(
            train_df, test_df, feature_cols
        )
    except Exception as e:
        print(f"    ✗ Weighted ensemble failed: {e}")
        results['strategies']['weighted_ensemble'] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 6 Task 4: Fusion Experiments')
    parser.add_argument('--config', type=str, default='config/config_default.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='results/analysis/fusion_results.json',
                       help='Output JSON file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    print("Loading processed data...")
    df = load_processed_data()
    print(f"Loaded {len(df)} samples")
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    print(f"Using {len(feature_cols)} base features")
    
    # Create CV folds
    print("\nCreating rolling-origin CV splits...")
    folds = create_rolling_origin_splits(df)
    print(f"Created {len(folds)} folds")
    
    # Run experiments for each fold
    all_results = []
    for fold in folds:
        fold_result = run_fusion_experiments_for_fold(
            df, fold.name, fold.train_idx, fold.test_idx, feature_cols
        )
        all_results.append(fold_result)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATED FUSION RESULTS")
    print(f"{'='*60}")
    
    # Collect metrics across folds for each strategy
    strategy_names = ['feature_fusion', 'gated_decision', 'weighted_ensemble']
    aggregated = {}
    
    for strategy in strategy_names:
        strategy_metrics = []
        
        for fold_result in all_results:
            if strategy in fold_result['strategies'] and 'metrics' in fold_result['strategies'][strategy]:
                metrics = fold_result['strategies'][strategy]['metrics']
                if strategy == 'weighted_ensemble':
                    metrics = fold_result['strategies'][strategy]['best_metrics']
                strategy_metrics.append(metrics)
        
        if len(strategy_metrics) > 0:
            # Average metrics
            avg_metrics = {}
            for key in strategy_metrics[0].keys():
                values = [m[key] for m in strategy_metrics if m[key] is not None]
                if len(values) > 0:
                    avg_metrics[f'{key}_mean'] = float(np.mean(values))
                    avg_metrics[f'{key}_std'] = float(np.std(values))
            
            aggregated[strategy] = avg_metrics
    
    for strategy, metrics in aggregated.items():
        print(f"\n{strategy.upper()}:")
        if 'auc_mean' in metrics:
            print(f"  AUC: {metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
        if 'aupr_mean' in metrics:
            print(f"  AUPR: {metrics['aupr_mean']:.3f} ± {metrics['aupr_std']:.3f}")
        if 'f1_mean' in metrics:
            print(f"  F1: {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 6 - Task 4: Fusion Experiments',
        'strategies': {
            'feature_fusion': 'Inject Bayesian features into XGBoost',
            'gated_decision': 'Use Bayesian when high risk, else XGBoost',
            'weighted_ensemble': 'Weighted combination of probabilities'
        },
        'aggregated': aggregated,
        'fold_results': all_results
    }
    
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
