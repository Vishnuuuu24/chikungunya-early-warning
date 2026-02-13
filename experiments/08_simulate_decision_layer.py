#!/usr/bin/env python3
"""Experiment 08: Decision-Layer Simulation (Phase 7+ aligned)

This script simulates an uncertainty-aware decision layer using the per-row
out-of-sample predictions saved by Experiment 06:
  - ML probability: `prob`
  - Bayesian latent risk summary: `z_mean`, `z_sd`
  - True label: `y_true`

The older draft refit Bayesian models and evaluated on training data in places.
This version consumes the stored artifacts so it is reproducible and aligned
with the lead-time pipeline.

Output
- results/analysis/decision_simulation.json (default)
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config


def _preds_path(percentile: int) -> Path:
    analysis_dir = project_root / 'results' / 'analysis'
    path = analysis_dir / f'lead_time_predictions_p{percentile}.parquet'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Run experiments/06_analyze_lead_time.py first.'
        )
    return path


def load_predictions(percentile: int) -> pd.DataFrame:
    df = pd.read_parquet(_preds_path(percentile))
    needed = {'fold', 'prob', 'z_mean', 'z_sd', 'y_true'}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f'Predictions parquet missing required columns: {sorted(missing)}')
    return df


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def compute_bayes_prob_high_risk(df: pd.DataFrame, risk_quantile: float = 0.80) -> np.ndarray:
    """Compute P(Z > q) via Normal(mu=z_mean, sd=z_sd)."""
    work = df[['z_mean', 'z_sd']].copy().dropna()
    if work.empty:
        return np.full(shape=(len(df),), fill_value=np.nan, dtype=float)

    q = float(np.percentile(work['z_mean'].astype(float).values, risk_quantile * 100))
    mu = df['z_mean'].astype(float).values
    sd = np.maximum(df['z_sd'].astype(float).values, 1e-6)
    z = (q - mu) / sd
    return 1.0 - _normal_cdf(z)


def assign_zone(p_high: np.ndarray, yellow: float, red: float) -> np.ndarray:
    zone = np.full(shape=p_high.shape, fill_value='GREEN', dtype=object)
    zone[p_high >= yellow] = 'YELLOW'
    zone[p_high >= red] = 'RED'
    return zone


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        out['auc'] = float(roc_auc_score(y_true, y_prob))
        out['aupr'] = float(average_precision_score(y_true, y_prob))
    else:
        out['auc'] = 0.5
        out['aupr'] = 0.0

    out['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    out['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    out['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    return out


def simulate_fold(
    fold_df: pd.DataFrame,
    prob_threshold: float,
    yellow: float,
    red: float,
    risk_quantile: float,
) -> Dict[str, Any]:
    valid_mask = pd.to_numeric(fold_df['y_true'], errors='coerce').notna()
    df = fold_df.loc[valid_mask].copy()
    if df.empty:
        raise RuntimeError('No valid y_true rows in this fold')

    y_true = pd.to_numeric(df['y_true'], errors='coerce').astype(int).values
    p_high = compute_bayes_prob_high_risk(df, risk_quantile=risk_quantile)
    zone = assign_zone(p_high, yellow=yellow, red=red)

    # Alert definitions
    alert_any = (zone != 'GREEN').astype(int)
    alert_red = (zone == 'RED').astype(int)

    zone_counts = pd.Series(zone).value_counts().to_dict()
    outbreak_rate_by_zone = (
        pd.DataFrame({'zone': zone, 'y_true': y_true})
        .groupby('zone')['y_true']
        .mean()
        .to_dict()
    )

    # Evaluate alerts as binary decisions
    metrics_any = binary_metrics(y_true, alert_any.astype(float), threshold=0.5)
    metrics_red = binary_metrics(y_true, alert_red.astype(float), threshold=0.5)

    # Also evaluate p_high directly (for sanity)
    metrics_prob = binary_metrics(y_true, p_high, threshold=prob_threshold)

    return {
        'n_rows': int(len(df)),
        'zone_counts': zone_counts,
        'outbreak_rate_by_zone': {k: float(v) for k, v in outbreak_rate_by_zone.items()},
        'metrics': {
            'alert_any': metrics_any,
            'alert_red': metrics_red,
            'p_highrisk': metrics_prob,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 6 Task 3: Decision-Layer Simulation')
    parser.add_argument('--config', type=str, default='config/config_default.yaml', help='Path to config file')
    parser.add_argument(
        '--outbreak-percentile',
        type=int,
        default=75,
        help='Outbreak percentile used in Experiment 06 outputs',
    )
    parser.add_argument('--output', type=str, default='results/analysis/decision_simulation.json', help='Output JSON file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Thresholds (simple defaults; meant for decision-zone illustration)
    prob_threshold = float(config.get('evaluation', {}).get('probability_threshold', 0.5))
    yellow = float(config.get('decision_layer', {}).get('yellow_threshold', 0.5))
    red = float(config.get('decision_layer', {}).get('red_threshold', 0.8))
    risk_quantile = float(config.get('decision_layer', {}).get('risk_quantile', 0.80))

    preds = load_predictions(args.outbreak_percentile)

    fold_results: Dict[str, Any] = {}
    for fold_name, fold_df in preds.groupby('fold', sort=False):
        fold_results[str(fold_name)] = simulate_fold(
            fold_df,
            prob_threshold=prob_threshold,
            yellow=yellow,
            red=red,
            risk_quantile=risk_quantile,
        )

    output: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Experiment 08 - Decision Layer Simulation (prediction-level)',
        'input': {
            'predictions_parquet': str(_preds_path(args.outbreak_percentile)),
            'probability_threshold': prob_threshold,
            'yellow_threshold': yellow,
            'red_threshold': red,
            'risk_quantile': risk_quantile,
        },
        'fold_results': fold_results,
    }

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
