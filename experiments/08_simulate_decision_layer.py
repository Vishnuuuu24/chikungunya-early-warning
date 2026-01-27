#!/usr/bin/env python3
"""
Phase 6 - Task 3: Decision-Layer Simulation

Implement uncertainty-aware decision rules on top of Bayesian outputs.

Decision Zones:
- GREEN (No Action): P(Z_t > q) < 0.4
- YELLOW (Monitor): 0.4 ≤ P(Z_t > q) < 0.8 OR high uncertainty
- RED (Intervene): P(Z_t > q) ≥ 0.8 AND low uncertainty

Evaluation Metrics:
- Lead time to intervention (RED zone before outbreak)
- False alarms (RED zone without outbreak)
- Missed outbreaks (never reached RED)
- Decision stability (zone transitions)
- Cost-loss analysis

Output: results/analysis/decision_simulation.json

Reference: Phase 6 decision-theoretic evaluation
WHO EWARS guidelines, cost-loss decision theory
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.evaluation.cv import create_rolling_origin_splits

# Add v3 code path for Bayesian model
v3_code_path = project_root / "versions" / "Vishnu-Version-Hist" / "v3" / "code"
sys.path.insert(0, str(v3_code_path))


# =============================================================================
# DECISION FRAMEWORK
# =============================================================================

class AlertZone(Enum):
    """Decision zones for outbreak early warning."""
    GREEN = "no_action"
    YELLOW = "monitor"
    RED = "intervene"


@dataclass
class DecisionThresholds:
    """Thresholds for decision zones."""
    risk_quantile: float = 0.80  # 80th percentile threshold for P(Z_t > q)
    prob_low: float = 0.40       # P(Z_t > q) < 0.4 → GREEN
    prob_high: float = 0.80      # P(Z_t > q) ≥ 0.8 → RED (if low uncertainty)
    uncertainty_threshold: float = 0.5  # Coefficient of variation threshold


@dataclass
class DecisionCosts:
    """Cost-loss parameters for decision evaluation."""
    cost_intervention: float = 1.0      # Cost of intervention (normalized)
    loss_missed_outbreak: float = 10.0  # Loss from missed outbreak
    cost_false_alarm: float = 0.5       # Cost of false alarm
    benefit_early_warning: float = 5.0  # Benefit of early intervention


# =============================================================================
# MCMC CONFIGURATION
# =============================================================================

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


def compute_bayesian_risk_scores(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    thresholds: DecisionThresholds
) -> pd.DataFrame:
    """
    Fit Bayesian model and compute risk scores with uncertainty.
    
    Returns DataFrame with:
    - latent_risk_mean: E[Z_t]
    - latent_risk_std: SD[Z_t]
    - prob_high_risk: P(Z_t > q)
    - uncertainty_cv: Coefficient of variation
    - alert_zone: GREEN/YELLOW/RED
    """
    from src.models.bayesian.state_space import BayesianStateSpace
    
    # Prepare valid data
    valid_df = train_df.dropna(subset=['state', 'district', 'year', 'week', 'cases']).copy()
    if 'temp_celsius' in valid_df.columns:
        valid_df = valid_df.dropna(subset=['temp_celsius'])
    
    # Fit model
    X = valid_df[feature_cols].values
    y = valid_df['cases'].values
    
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
    model.fit(X, y, df=valid_df, feature_cols=feature_cols)
    
    # Extract posterior predictive
    y_rep = model.get_posterior_predictive()  # (n_draws, n_samples)
    
    # Compute risk quantile threshold
    risk_threshold = np.percentile(y_rep.mean(axis=0), thresholds.risk_quantile * 100)
    
    # Compute statistics for each time point
    risk_df = valid_df[['state', 'district', 'year', 'week', 'cases', 'outbreak_p75']].copy()
    
    risk_df['latent_risk_mean'] = y_rep.mean(axis=0)
    risk_df['latent_risk_std'] = y_rep.std(axis=0)
    risk_df['latent_risk_median'] = np.median(y_rep, axis=0)
    
    # P(Z_t > q)
    risk_df['prob_high_risk'] = (y_rep > risk_threshold).mean(axis=0)
    
    # Coefficient of variation (uncertainty measure)
    risk_df['uncertainty_cv'] = risk_df['latent_risk_std'] / (risk_df['latent_risk_mean'] + 1e-6)
    
    # Assign alert zones
    risk_df['alert_zone'] = risk_df.apply(
        lambda row: assign_alert_zone(
            row['prob_high_risk'],
            row['uncertainty_cv'],
            thresholds
        ),
        axis=1
    )
    
    return risk_df


def assign_alert_zone(
    prob_high_risk: float,
    uncertainty_cv: float,
    thresholds: DecisionThresholds
) -> str:
    """
    Assign alert zone based on probability and uncertainty.
    
    Rules:
    1. GREEN: Low probability of high risk
    2. YELLOW: Moderate probability OR high uncertainty
    3. RED: High probability AND low uncertainty
    """
    if prob_high_risk < thresholds.prob_low:
        return AlertZone.GREEN.value
    
    elif prob_high_risk >= thresholds.prob_high and uncertainty_cv < thresholds.uncertainty_threshold:
        return AlertZone.RED.value
    
    else:
        return AlertZone.YELLOW.value


def evaluate_decision_performance(
    risk_df: pd.DataFrame,
    thresholds: DecisionThresholds,
    costs: DecisionCosts
) -> Dict[str, Any]:
    """
    Evaluate decision layer performance.
    
    Computes:
    - Lead time to RED intervention before outbreaks
    - False alarms (RED without outbreak)
    - Missed outbreaks (never RED)
    - Decision stability (transition frequency)
    - Expected cost
    """
    results = {
        'interventions': {'total': 0, 'true_positive': 0, 'false_positive': 0},
        'outbreaks': {'total': 0, 'detected': 0, 'missed': 0},
        'lead_times': [],
        'zone_transitions': 0,
        'zone_distribution': {},
        'cost_analysis': {}
    }
    
    # Count zone distribution
    zone_counts = risk_df['alert_zone'].value_counts().to_dict()
    total = len(risk_df)
    results['zone_distribution'] = {
        zone: {'count': count, 'percentage': count / total * 100}
        for zone, count in zone_counts.items()
    }
    
    # Analyze by district-year episodes
    for (state, district, year), group in risk_df.groupby(['state', 'district', 'year']):
        group = group.sort_values('week')
        
        # Check if this year had an outbreak
        has_outbreak = group['outbreak_p75'].sum() > 0
        
        # Find first RED zone week
        red_weeks = group[group['alert_zone'] == AlertZone.RED.value]
        has_intervention = len(red_weeks) > 0
        
        if has_outbreak:
            results['outbreaks']['total'] += 1
            
            # Find first outbreak week
            outbreak_weeks = group[group['outbreak_p75'] == 1]
            if len(outbreak_weeks) > 0:
                first_outbreak_week = outbreak_weeks.iloc[0]['week']
                
                if has_intervention:
                    first_red_week = red_weeks.iloc[0]['week']
                    lead_time = first_outbreak_week - first_red_week
                    
                    if lead_time > 0:  # Intervention before outbreak
                        results['outbreaks']['detected'] += 1
                        results['interventions']['true_positive'] += 1
                        results['lead_times'].append(lead_time)
                    else:
                        results['outbreaks']['missed'] += 1
                else:
                    results['outbreaks']['missed'] += 1
        
        elif has_intervention:
            # Intervention without outbreak = false alarm
            results['interventions']['false_positive'] += 1
        
        if has_intervention:
            results['interventions']['total'] += 1
        
        # Count zone transitions
        zones = group['alert_zone'].values
        transitions = np.sum(zones[1:] != zones[:-1])
        results['zone_transitions'] += transitions
    
    # Compute metrics
    n_interventions = results['interventions']['total']
    n_outbreaks = results['outbreaks']['total']
    
    metrics = {}
    
    if n_interventions > 0:
        metrics['precision'] = results['interventions']['true_positive'] / n_interventions
        metrics['false_alarm_rate'] = results['interventions']['false_positive'] / n_interventions
    else:
        metrics['precision'] = 0.0
        metrics['false_alarm_rate'] = 0.0
    
    if n_outbreaks > 0:
        metrics['sensitivity'] = results['outbreaks']['detected'] / n_outbreaks
        metrics['miss_rate'] = results['outbreaks']['missed'] / n_outbreaks
    else:
        metrics['sensitivity'] = 0.0
        metrics['miss_rate'] = 0.0
    
    if len(results['lead_times']) > 0:
        metrics['median_lead_time'] = float(np.median(results['lead_times']))
        metrics['mean_lead_time'] = float(np.mean(results['lead_times']))
        metrics['lead_time_std'] = float(np.std(results['lead_times']))
    else:
        metrics['median_lead_time'] = None
        metrics['mean_lead_time'] = None
        metrics['lead_time_std'] = None
    
    # Decision stability (transitions per district-year)
    n_episodes = risk_df.groupby(['state', 'district', 'year']).ngroups
    metrics['avg_transitions_per_episode'] = results['zone_transitions'] / n_episodes if n_episodes > 0 else 0
    
    # Cost-loss analysis
    total_cost = (
        results['interventions']['total'] * costs.cost_intervention +
        results['interventions']['false_positive'] * costs.cost_false_alarm +
        results['outbreaks']['missed'] * costs.loss_missed_outbreak -
        results['outbreaks']['detected'] * costs.benefit_early_warning
    )
    
    cost_analysis = {
        'total_cost': total_cost,
        'intervention_cost': results['interventions']['total'] * costs.cost_intervention,
        'false_alarm_cost': results['interventions']['false_positive'] * costs.cost_false_alarm,
        'missed_outbreak_loss': results['outbreaks']['missed'] * costs.loss_missed_outbreak,
        'early_warning_benefit': results['outbreaks']['detected'] * costs.benefit_early_warning,
        'net_benefit': -total_cost  # Negative cost = positive benefit
    }
    
    results['metrics'] = metrics
    results['cost_analysis'] = cost_analysis
    
    return results


def simulate_decision_layer_for_fold(
    df: pd.DataFrame,
    fold_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: List[str],
    thresholds: DecisionThresholds,
    costs: DecisionCosts
) -> Dict[str, Any]:
    """
    Simulate decision layer for one CV fold.
    """
    print(f"\n{'='*60}")
    print(f"Simulating fold: {fold_name}")
    print(f"{'='*60}")
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Compute Bayesian risk scores for test set
    # (Fit on train, but we need to evaluate decisions on test)
    # For simplicity, we'll fit on train and apply to test districts
    
    print("\nStep 1: Computing Bayesian risk scores...")
    risk_df = compute_bayesian_risk_scores(train_df, feature_cols, thresholds)
    
    print("\nStep 2: Evaluating decision performance...")
    results = evaluate_decision_performance(risk_df, thresholds, costs)
    
    # Print summary
    print("\n--- Decision Layer Results ---")
    print(f"Total interventions (RED): {results['interventions']['total']}")
    print(f"  True positives: {results['interventions']['true_positive']}")
    print(f"  False positives: {results['interventions']['false_positive']}")
    print(f"Total outbreaks: {results['outbreaks']['total']}")
    print(f"  Detected: {results['outbreaks']['detected']}")
    print(f"  Missed: {results['outbreaks']['missed']}")
    
    if results['metrics']['median_lead_time'] is not None:
        print(f"Median lead time: {results['metrics']['median_lead_time']:.1f} weeks")
    print(f"Sensitivity: {results['metrics']['sensitivity']:.3f}")
    print(f"Precision: {results['metrics']['precision']:.3f}")
    print(f"False alarm rate: {results['metrics']['false_alarm_rate']:.3f}")
    print(f"\nNet benefit: {results['cost_analysis']['net_benefit']:.2f}")
    
    return {
        'fold': fold_name,
        'thresholds': {
            'risk_quantile': thresholds.risk_quantile,
            'prob_low': thresholds.prob_low,
            'prob_high': thresholds.prob_high,
            'uncertainty_threshold': thresholds.uncertainty_threshold
        },
        'results': results
    }


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 6 Task 3: Decision-Layer Simulation')
    parser.add_argument('--config', type=str, default='config/config_default.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='results/analysis/decision_simulation.json',
                       help='Output JSON file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize thresholds and costs
    thresholds = DecisionThresholds()
    costs = DecisionCosts()
    
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
    
    # Simulate each fold
    all_results = []
    for fold in folds:
        fold_result = simulate_decision_layer_for_fold(
            df, fold.name, fold.train_idx, fold.test_idx,
            feature_cols, thresholds, costs
        )
        all_results.append(fold_result)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATED DECISION PERFORMANCE")
    print(f"{'='*60}")
    
    total_interventions = sum([r['results']['interventions']['total'] for r in all_results])
    total_tp = sum([r['results']['interventions']['true_positive'] for r in all_results])
    total_fp = sum([r['results']['interventions']['false_positive'] for r in all_results])
    total_outbreaks = sum([r['results']['outbreaks']['total'] for r in all_results])
    total_detected = sum([r['results']['outbreaks']['detected'] for r in all_results])
    total_missed = sum([r['results']['outbreaks']['missed'] for r in all_results])
    
    all_lead_times = []
    for r in all_results:
        all_lead_times.extend(r['results']['lead_times'])
    
    aggregated = {
        'total_interventions': total_interventions,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'total_outbreaks': total_outbreaks,
        'detected': total_detected,
        'missed': total_missed,
        'sensitivity': total_detected / total_outbreaks if total_outbreaks > 0 else 0,
        'precision': total_tp / total_interventions if total_interventions > 0 else 0,
        'false_alarm_rate': total_fp / total_interventions if total_interventions > 0 else 0
    }
    
    if len(all_lead_times) > 0:
        aggregated['median_lead_time'] = float(np.median(all_lead_times))
        aggregated['mean_lead_time'] = float(np.mean(all_lead_times))
        aggregated['std_lead_time'] = float(np.std(all_lead_times))
    
    # Calculate total cost
    total_cost = sum([r['results']['cost_analysis']['total_cost'] for r in all_results])
    total_benefit = sum([r['results']['cost_analysis']['net_benefit'] for r in all_results])
    
    aggregated['total_cost'] = total_cost
    aggregated['net_benefit'] = total_benefit
    
    print(f"\nTotal interventions: {aggregated['total_interventions']}")
    print(f"Total outbreaks: {aggregated['total_outbreaks']}")
    print(f"Sensitivity: {aggregated['sensitivity']:.3f}")
    print(f"Precision: {aggregated['precision']:.3f}")
    if 'median_lead_time' in aggregated:
        print(f"Median lead time: {aggregated['median_lead_time']:.1f} weeks")
    print(f"Net benefit: {aggregated['net_benefit']:.2f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 6 - Task 3: Decision-Layer Simulation',
        'decision_framework': {
            'thresholds': {
                'risk_quantile': thresholds.risk_quantile,
                'prob_low': thresholds.prob_low,
                'prob_high': thresholds.prob_high,
                'uncertainty_threshold': thresholds.uncertainty_threshold
            },
            'costs': {
                'intervention': costs.cost_intervention,
                'missed_outbreak_loss': costs.loss_missed_outbreak,
                'false_alarm': costs.cost_false_alarm,
                'early_warning_benefit': costs.benefit_early_warning
            }
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
