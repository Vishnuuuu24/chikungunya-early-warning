#!/usr/bin/env python3
"""
Phase 6 - Task 1: Lead-Time Analysis

This script computes the lead-time advantage of Bayesian latent risk
estimation compared to reactive outbreak detection.

Metrics:
- t_outbreak: First week where observed cases cross outbreak threshold
- t_bayes: First week where P(Z_t > q) ≥ τ (q=80th percentile, τ=0.8)
- lead_time = t_outbreak - t_bayes (in weeks)

For each outbreak episode (district-year with outbreak=1):
1. Identify outbreak onset week
2. Identify Bayesian early warning week
3. Compute lead time
4. Compare against XGBoost trigger timing

Output: results/analysis/lead_time_analysis.json

Reference: Phase 6 decision-theoretic evaluation
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, get_project_root
from src.evaluation.cv import create_rolling_origin_splits
from src.labels.outbreak_labels import create_outbreak_labels

# Add v3 code path for Bayesian model
v3_code_path = project_root / "versions" / "Vishnu-Version-Hist" / "v3" / "code"
sys.path.insert(0, str(v3_code_path))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Bayesian decision threshold parameters
LATENT_RISK_QUANTILE = 0.80  # 80th percentile of latent risk Z_t
PROBABILITY_THRESHOLD = 0.80  # P(Z_t > q) ≥ 0.8 for alarm
MCMC_CONFIG = {
    'n_warmup': 1000,
    'n_samples': 1000,
    'n_chains': 4,
    'adapt_delta': 0.95,
    'seed': 42
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_processed_data() -> pd.DataFrame:
    """Load feature-engineered data."""
    data_path = project_root / "data" / "processed" / "features_engineered_v01.parquet"
    df = pd.read_parquet(data_path)
    return df


def identify_outbreak_episodes(df: pd.DataFrame, threshold_col: str = 'outbreak_p75') -> pd.DataFrame:
    """
    Identify outbreak episodes (district-years with at least one outbreak week).
    
    Returns DataFrame with columns:
    - state, district, year
    - first_outbreak_week: Week number when outbreak first detected
    - peak_week: Week with maximum cases
    - total_outbreak_weeks: Number of weeks with outbreak=1
    """
    episodes = []
    
    for (state, district, year), group in df.groupby(['state', 'district', 'year']):
        outbreak_weeks = group[group[threshold_col] == 1]
        
        if len(outbreak_weeks) > 0:
            episode = {
                'state': state,
                'district': district,
                'year': year,
                'first_outbreak_week': outbreak_weeks['week'].min(),
                'peak_week': group.loc[group['cases'].idxmax(), 'week'],
                'peak_cases': group['cases'].max(),
                'total_outbreak_weeks': len(outbreak_weeks),
                'total_cases': group['cases'].sum()
            }
            episodes.append(episode)
    
    return pd.DataFrame(episodes)


def compute_bayesian_latent_risk(
    model,
    train_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, float]:
    """
    Fit Bayesian model and extract latent risk Z_t for all time points.
    
    Returns:
    - DataFrame with state, district, year, week, latent_risk_mean, latent_risk_q80
    - threshold_q80: 80th percentile threshold for latent risk
    """
    from src.models.bayesian.state_space import BayesianStateSpace
    
    # Prepare data
    valid_df = train_df.dropna(subset=['state', 'district', 'year', 'week', 'cases']).copy()
    if 'temp_celsius' in valid_df.columns:
        valid_df = valid_df.dropna(subset=['temp_celsius'])
    
    # Fit model
    X_train = valid_df[feature_cols].values
    y_train = valid_df['cases'].values
    
    stan_path = project_root / "versions" / "Vishnu-Version-Hist" / "v3" / "stan_models" / "hierarchical_ews_v01.stan"
    
    model = BayesianStateSpace(
        stan_model_path=str(stan_path),
        n_warmup=MCMC_CONFIG['n_warmup'],
        n_samples=MCMC_CONFIG['n_samples'],
        n_chains=MCMC_CONFIG['n_chains'],
        adapt_delta=MCMC_CONFIG['adapt_delta'],
        seed=MCMC_CONFIG['seed']
    )
    
    print(f"  Fitting Bayesian model on {len(valid_df)} samples...")
    model.fit(X_train, y_train, df=valid_df, feature_cols=feature_cols)
    
    # Extract latent risk (posterior predictive distribution)
    y_rep = model.get_posterior_predictive()  # Shape: (n_draws, n_samples)
    
    # Compute statistics for each time point
    latent_risk_mean = y_rep.mean(axis=0)
    latent_risk_std = y_rep.std(axis=0)
    latent_risk_q80 = np.percentile(y_rep, 80, axis=0)
    
    # Compute global 80th percentile threshold
    threshold_q80 = np.percentile(latent_risk_mean, 80)
    
    # Create output DataFrame
    risk_df = valid_df[['state', 'district', 'year', 'week', 'cases']].copy()
    risk_df['latent_risk_mean'] = latent_risk_mean
    risk_df['latent_risk_std'] = latent_risk_std
    risk_df['latent_risk_q80'] = latent_risk_q80
    
    return risk_df, threshold_q80


def compute_bayesian_trigger_week(
    risk_df: pd.DataFrame,
    threshold_q80: float,
    state: str,
    district: str,
    year: int
) -> Optional[int]:
    """
    Find first week where P(Z_t > q) ≥ τ for given district-year.
    
    We approximate P(Z_t > q) by checking if latent_risk_mean > threshold_q80
    (since we don't have full posterior samples per time point in this script).
    
    Returns week number or None if never triggered.
    """
    subset = risk_df[
        (risk_df['state'] == state) &
        (risk_df['district'] == district) &
        (risk_df['year'] == year)
    ].sort_values('week')
    
    if len(subset) == 0:
        return None
    
    # Find first week where latent risk exceeds threshold
    triggered = subset[subset['latent_risk_mean'] > threshold_q80]
    
    if len(triggered) > 0:
        return triggered.iloc[0]['week']
    else:
        return None


def compute_lead_times(
    episodes_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    threshold_q80: float
) -> pd.DataFrame:
    """
    Compute lead time for each outbreak episode.
    
    Lead time = t_outbreak - t_bayes
    Positive values mean Bayesian warned earlier.
    """
    results = []
    
    for _, episode in episodes_df.iterrows():
        state = episode['state']
        district = episode['district']
        year = episode['year']
        t_outbreak = episode['first_outbreak_week']
        
        # Get Bayesian trigger week
        t_bayes = compute_bayesian_trigger_week(
            risk_df, threshold_q80, state, district, year
        )
        
        if t_bayes is not None:
            lead_time = t_outbreak - t_bayes
        else:
            lead_time = None  # Bayesian never triggered
        
        results.append({
            'state': state,
            'district': district,
            'year': year,
            'first_outbreak_week': t_outbreak,
            'bayesian_trigger_week': t_bayes,
            'lead_time_weeks': lead_time,
            'peak_cases': episode['peak_cases'],
            'total_outbreak_weeks': episode['total_outbreak_weeks']
        })
    
    return pd.DataFrame(results)


def summarize_lead_times(lead_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics for lead times.
    """
    valid_leads = lead_df.dropna(subset=['lead_time_weeks'])
    
    if len(valid_leads) == 0:
        return {
            'n_episodes': len(lead_df),
            'n_bayes_triggered': 0,
            'median_lead_time': None,
            'mean_lead_time': None,
            'iqr': [None, None],
            'min_lead_time': None,
            'max_lead_time': None,
            'positive_lead_pct': None
        }
    
    lead_times = valid_leads['lead_time_weeks'].values
    
    summary = {
        'n_episodes': len(lead_df),
        'n_bayes_triggered': len(valid_leads),
        'trigger_rate': len(valid_leads) / len(lead_df),
        'median_lead_time': float(np.median(lead_times)),
        'mean_lead_time': float(np.mean(lead_times)),
        'std_lead_time': float(np.std(lead_times)),
        'iqr': [float(np.percentile(lead_times, 25)), float(np.percentile(lead_times, 75))],
        'min_lead_time': float(lead_times.min()),
        'max_lead_time': float(lead_times.max()),
        'positive_lead_pct': float((lead_times > 0).sum() / len(lead_times) * 100),
        'zero_lead_pct': float((lead_times == 0).sum() / len(lead_times) * 100),
        'negative_lead_pct': float((lead_times < 0).sum() / len(lead_times) * 100)
    }
    
    return summary


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_lead_time_per_fold(
    df: pd.DataFrame,
    fold_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Analyze lead time for one CV fold.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing fold: {fold_name}")
    print(f"{'='*60}")
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Step 1: Identify outbreak episodes in test set
    print("\nStep 1: Identifying outbreak episodes...")
    episodes = identify_outbreak_episodes(test_df, threshold_col='outbreak_p75')
    print(f"  Found {len(episodes)} outbreak episodes")
    
    if len(episodes) == 0:
        return {
            'fold': fold_name,
            'n_episodes': 0,
            'summary': None,
            'episodes': []
        }
    
    # Step 2: Fit Bayesian model on training data
    print("\nStep 2: Computing Bayesian latent risk...")
    risk_df, threshold_q80 = compute_bayesian_latent_risk(
        None, train_df, feature_cols
    )
    print(f"  Latent risk threshold (q80): {threshold_q80:.2f}")
    
    # Step 3: Compute lead times
    print("\nStep 3: Computing lead times...")
    lead_df = compute_lead_times(episodes, risk_df, threshold_q80)
    
    # Step 4: Summarize
    summary = summarize_lead_times(lead_df)
    
    print("\n--- Lead Time Summary ---")
    print(f"  Total episodes: {summary['n_episodes']}")
    print(f"  Bayesian triggered: {summary['n_bayes_triggered']} ({summary['trigger_rate']*100:.1f}%)")
    if summary['median_lead_time'] is not None:
        print(f"  Median lead time: {summary['median_lead_time']:.1f} weeks")
        print(f"  Mean lead time: {summary['mean_lead_time']:.1f} ± {summary['std_lead_time']:.1f} weeks")
        print(f"  IQR: [{summary['iqr'][0]:.1f}, {summary['iqr'][1]:.1f}]")
        print(f"  Positive lead (early warning): {summary['positive_lead_pct']:.1f}%")
    
    return {
        'fold': fold_name,
        'n_episodes': len(episodes),
        'threshold_q80': float(threshold_q80),
        'summary': summary,
        'episodes': lead_df.to_dict(orient='records')
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 6 Task 1: Lead-Time Analysis')
    parser.add_argument('--config', type=str, default='config/config_default.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='results/analysis/lead_time_analysis.json',
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
    print(f"Using {len(feature_cols)} features")
    
    # Create CV folds
    print("\nCreating rolling-origin CV splits...")
    folds = create_rolling_origin_splits(df)
    print(f"Created {len(folds)} folds")
    
    # Analyze each fold
    all_results = []
    for fold in folds:
        fold_result = analyze_lead_time_per_fold(
            df, fold.name, fold.train_idx, fold.test_idx, feature_cols
        )
        all_results.append(fold_result)
    
    # Aggregate across folds
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    
    all_episodes = sum([r['n_episodes'] for r in all_results])
    all_triggered = sum([r['summary']['n_bayes_triggered'] if r['summary'] else 0 for r in all_results])
    
    # Collect all lead times
    all_lead_times = []
    for r in all_results:
        if r['summary'] and r['summary']['n_bayes_triggered'] > 0:
            for ep in r['episodes']:
                if ep['lead_time_weeks'] is not None:
                    all_lead_times.append(ep['lead_time_weeks'])
    
    if len(all_lead_times) > 0:
        aggregated = {
            'n_total_episodes': all_episodes,
            'n_triggered': all_triggered,
            'trigger_rate': all_triggered / all_episodes if all_episodes > 0 else 0,
            'median_lead_time': float(np.median(all_lead_times)),
            'mean_lead_time': float(np.mean(all_lead_times)),
            'std_lead_time': float(np.std(all_lead_times)),
            'iqr': [float(np.percentile(all_lead_times, 25)), 
                   float(np.percentile(all_lead_times, 75))],
            'min_lead_time': float(np.min(all_lead_times)),
            'max_lead_time': float(np.max(all_lead_times)),
            'positive_lead_pct': float((np.array(all_lead_times) > 0).sum() / len(all_lead_times) * 100)
        }
        
        print(f"\nTotal outbreak episodes: {aggregated['n_total_episodes']}")
        print(f"Bayesian triggered: {aggregated['n_triggered']} ({aggregated['trigger_rate']*100:.1f}%)")
        print(f"Median lead time: {aggregated['median_lead_time']:.1f} weeks")
        print(f"Mean lead time: {aggregated['mean_lead_time']:.1f} ± {aggregated['std_lead_time']:.1f} weeks")
        print(f"Early warning rate: {aggregated['positive_lead_pct']:.1f}%")
    else:
        aggregated = {
            'n_total_episodes': all_episodes,
            'n_triggered': 0,
            'note': 'No valid lead times computed'
        }
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 6 - Task 1: Lead-Time Analysis',
        'config': {
            'latent_risk_quantile': LATENT_RISK_QUANTILE,
            'probability_threshold': PROBABILITY_THRESHOLD,
            'mcmc_config': MCMC_CONFIG
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
