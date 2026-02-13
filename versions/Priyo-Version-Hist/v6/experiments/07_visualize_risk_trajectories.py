#!/usr/bin/env python3
"""
Phase 6 - Task 2: Risk Trajectory Visualization

Generate plots for representative districts showing:
- Observed cases (actual outbreak dynamics)
- Outbreak threshold (75th percentile)
- Bayesian latent risk Z_t (posterior mean)
- 90% credible intervals (uncertainty bands)

For 5-10 representative districts with varying outbreak patterns.

Output: results/plots/risk_trajectories/*.png

Reference: Phase 6 decision-theoretic evaluation
"""
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, get_repo_root
from src.labels.outbreak_labels import create_outbreak_labels

# Add v3 code path for Bayesian model (repo-level)
repo_root = get_repo_root()
v3_code_path = repo_root / "versions" / "Vishnu-Version-Hist" / "v3" / "code"
sys.path.insert(0, str(v3_code_path))


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

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
FIGSIZE = (14, 6)
DPI = 150


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_processed_data() -> pd.DataFrame:
    """Load feature-engineered data."""
    data_path = repo_root / "data" / "processed" / "features_engineered_v01.parquet"
    df = pd.read_parquet(data_path)
    return df


def select_representative_districts(df: pd.DataFrame, n_districts: int = 8) -> List[Tuple[str, str]]:
    """
    Select diverse representative districts based on outbreak patterns.
    
    Selection criteria:
    - High outbreak frequency (many outbreak weeks)
    - Moderate outbreak frequency
    - Low outbreak frequency
    - High peak intensity
    - Low peak intensity but recurring
    """
    district_stats = []
    
    # Prefer config-driven dynamic labels if present; fall back to legacy outbreak_pXX.
    outbreak_col = 'label_outbreak'
    if outbreak_col not in df.columns:
        legacy = [c for c in df.columns if c.startswith('outbreak_p')]
        if legacy:
            outbreak_col = legacy[0]

    for (state, district), group in df.groupby(['state', 'district']):
        stats = {
            'state': state,
            'district': district,
            'n_outbreak_weeks': group[outbreak_col].sum(skipna=True),
            'n_years': group['year'].nunique(),
            'max_cases': group['cases'].max(),
            'total_cases': group['cases'].sum(),
            'outbreak_rate': group[outbreak_col].mean(skipna=True)
        }
        district_stats.append(stats)
    
    stats_df = pd.DataFrame(district_stats)
    
    # Select representatives
    selected = []
    
    # 1. High outbreak frequency (top 2)
    high_freq = stats_df.nlargest(2, 'n_outbreak_weeks')
    selected.extend(zip(high_freq['state'], high_freq['district']))
    
    # 2. Moderate outbreak frequency (2-3 districts)
    moderate = stats_df[
        (stats_df['n_outbreak_weeks'] >= stats_df['n_outbreak_weeks'].quantile(0.4)) &
        (stats_df['n_outbreak_weeks'] <= stats_df['n_outbreak_weeks'].quantile(0.6))
    ].sample(min(2, len(stats_df)), random_state=42)
    selected.extend(zip(moderate['state'], moderate['district']))
    
    # 3. High peak intensity
    high_peak = stats_df.nlargest(2, 'max_cases')
    for state, district in zip(high_peak['state'], high_peak['district']):
        if (state, district) not in selected:
            selected.append((state, district))
    
    # 4. Fill remaining with diverse cases
    remaining = stats_df[~stats_df.apply(
        lambda x: (x['state'], x['district']) in selected, axis=1
    )].sample(min(n_districts - len(selected), len(stats_df)), random_state=42)
    selected.extend(zip(remaining['state'], remaining['district']))
    
    return selected[:n_districts]


def fit_bayesian_model_for_district(
    df: pd.DataFrame,
    state: str,
    district: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Fit Bayesian model for a specific district and extract latent risk.
    
    Returns:
    - DataFrame with week-level data and risk estimates
    - risk estimates are computed from latent state samples Z (not y_rep)
    """
    from src.models.bayesian.state_space import BayesianStateSpace
    
    # Filter to district
    district_df = df[
        (df['state'] == state) &
        (df['district'] == district)
    ].copy().sort_values(['year', 'week'])
    
    # Prepare valid data
    valid_df = district_df.dropna(subset=['cases', 'year', 'week']).copy()
    if 'temp_celsius' in valid_df.columns:
        valid_df = valid_df.dropna(subset=['temp_celsius'])
    
    # Get features and target
    X = valid_df[feature_cols].values
    y = valid_df['cases'].values
    
    # Fit model (v3 API: BayesianStateSpace(config=...))
    stan_path = repo_root / "versions" / "Vishnu-Version-Hist" / "v3" / "stan_models" / "hierarchical_ews_v01.stan"
    model_config = {
        **MCMC_CONFIG,
        'stan_file': str(stan_path),
        # Keep percentile config-driven when available; default to 75.
        'outbreak_percentile': 75,
    }
    model = BayesianStateSpace(config=model_config)
    
    print(f"    Fitting model on {len(valid_df)} weeks...")
    model.fit(X, y, df=valid_df, feature_cols=feature_cols)
    
    # Extract latent state samples (Z) aligned to each observation row.
    # IMPORTANT: Phase 7 "Bayesian risk" corresponds to latent Z, not y_rep.
    z_samples = model.get_latent_risk_samples_per_observation()  # (n_draws, n_obs)

    # Compute statistics
    valid_df['latent_risk_mean'] = z_samples.mean(axis=0)
    valid_df['latent_risk_std'] = z_samples.std(axis=0)
    valid_df['latent_risk_q05'] = np.percentile(z_samples, 5, axis=0)
    valid_df['latent_risk_q95'] = np.percentile(z_samples, 95, axis=0)
    valid_df['latent_risk_q25'] = np.percentile(z_samples, 25, axis=0)
    valid_df['latent_risk_q75'] = np.percentile(z_samples, 75, axis=0)

    return valid_df


def create_week_dates(year: int, week: int) -> datetime:
    """Convert year-week to approximate date."""
    # ISO week date (approximate)
    jan1 = datetime(year, 1, 1)
    start_of_year = jan1 - timedelta(days=jan1.weekday())
    week_start = start_of_year + timedelta(weeks=week - 1)
    return week_start


def plot_risk_trajectory(
    plot_df: pd.DataFrame,
    state: str,
    district: str,
    output_path: Path
):
    """
    Create risk trajectory plot for one district.
    
    Shows:
    - Observed cases (line + scatter)
    - Outbreak threshold (horizontal dashed line)
    - Bayesian latent risk mean (line)
    - 90% credible interval (shaded)
    - 50% credible interval (darker shaded)
    """
    # Create datetime index
    plot_df['date'] = plot_df.apply(lambda x: create_week_dates(int(x['year']), int(x['week'])), axis=1)
    plot_df = plot_df.sort_values('date')
    
    # Compute outbreak threshold (config-driven percentile, min 1.0)
    # This is for visualization only (not CV-safe).
    threshold = float(np.percentile(plot_df['cases'].dropna().values, 75)) if len(plot_df) else 1.0
    threshold = max(threshold, 1.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    # Plot observed cases
    ax.plot(plot_df['date'], plot_df['cases'], 
            'o-', color='black', linewidth=2, markersize=4,
            label='Observed Cases', alpha=0.7, zorder=3)
    
    # Plot outbreak threshold
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
              label=f'Outbreak Threshold (P75 = {threshold:.1f})', zorder=2)
    
    # Plot Bayesian latent risk
    ax.plot(plot_df['date'], plot_df['latent_risk_mean'],
           '-', color='#1f77b4', linewidth=2.5,
           label='Bayesian Latent Risk (Mean)', zorder=4)
    
    # Plot 90% credible interval
    ax.fill_between(plot_df['date'], 
                     plot_df['latent_risk_q05'],
                     plot_df['latent_risk_q95'],
                     color='#1f77b4', alpha=0.15,
                     label='90% Credible Interval', zorder=1)
    
    # Plot 50% credible interval
    ax.fill_between(plot_df['date'],
                     plot_df['latent_risk_q25'],
                     plot_df['latent_risk_q75'],
                     color='#1f77b4', alpha=0.25,
                     label='50% Credible Interval', zorder=1)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cases / Latent Risk', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk Trajectory: {district}, {state}', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    
    # Layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_path.name}")


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 6 Task 2: Risk Trajectory Visualization')
    parser.add_argument('--config', type=str, default='config/config_default.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='results/plots/risk_trajectories',
                       help='Output directory for plots')
    parser.add_argument('--n-districts', type=int, default=8,
                       help='Number of representative districts to plot')
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
    
    # Select representative districts
    print(f"\nSelecting {args.n_districts} representative districts...")
    selected = select_representative_districts(df, n_districts=args.n_districts)
    
    print("\nSelected districts:")
    for i, (state, district) in enumerate(selected, 1):
        print(f"  {i}. {district}, {state}")
    
    # Create output directory
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING RISK TRAJECTORY PLOTS")
    print(f"{'='*60}")
    
    for i, (state, district) in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Processing: {district}, {state}")
        
        try:
            # Fit Bayesian model
            plot_df = fit_bayesian_model_for_district(df, state, district, feature_cols)
            
            # Create plot
            safe_name = f"{district.replace(' ', '_')}_{state.replace(' ', '_')}.png"
            output_path = output_dir / safe_name
            
            plot_risk_trajectory(plot_df, state, district, output_path)
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
