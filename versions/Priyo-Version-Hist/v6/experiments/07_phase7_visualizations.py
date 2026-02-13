#!/usr/bin/env python3
"""
Experiment 07: Phase 7 Comprehensive Visualizations

PURPOSE:
--------
Generate all required figures for Phase 7 thesis chapters:
1. Track A Performance (ROC, AUC, Precision-Recall)
2. Lead-Time Analysis (distributions, comparisons, case studies)
3. Calibration & Uncertainty (reliability curves, Brier scores, credible intervals)
4. Decision Usefulness (state timelines, false alarm rates, stability)

VERSION: v4.2 (Phase 7 Complete)
DATE: February 2026

Reference: docs/Version 2/08_phase7_roadmap_v2.md
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =============================================================================
# PATH SETUP
# =============================================================================

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Output directories
FIGURE_ROOT = project_root / "results" / "figures" / "phase7"
TRACK_A_DIR = FIGURE_ROOT / "01_trackA_performance"
LEAD_TIME_DIR = FIGURE_ROOT / "02_lead_time_analysis"
CALIBRATION_DIR = FIGURE_ROOT / "03_calibration_uncertainty"
DECISION_DIR = FIGURE_ROOT / "04_decision_usefulness"

# Ensure directories exist
for d in [TRACK_A_DIR, LEAD_TIME_DIR, CALIBRATION_DIR, DECISION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Consistent color scheme
COLORS = {
    'bayesian': '#2E86AB',  # Blue
    'xgboost': '#E94F37',   # Red
    'rf': '#F39C12',        # Orange
    'logistic': '#27AE60',  # Green
    'baseline': '#95A5A6',  # Gray
    'outbreak': '#C0392B',  # Dark red
    'warning': '#F1C40F',   # Yellow
    'safe': '#27AE60',      # Green
}

# Figure settings
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure_with_description(
    fig: plt.Figure,
    filepath: Path,
    title: str,
    description: str,
    interpretation: str,
    caveats: str = ""
) -> None:
    """
    Save figure as PNG and create accompanying .txt description file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save PNG (without extension)
        title: Figure title
        description: What is shown
        interpretation: How to interpret
        caveats: Any caveats (optional)
    """
    # Save PNG
    png_path = filepath.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {png_path.name}")
    
    # Save description
    txt_path = filepath.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write(f"FIGURE: {title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"WHAT IS SHOWN:\n{description}\n\n")
        f.write(f"WHY IT MATTERS:\n{interpretation}\n\n")
        if caveats:
            f.write(f"CAVEATS:\n{caveats}\n")
    
    plt.close(fig)


def select_lead_time_file(prefix: str, ext: str, prefer_percentile: int = 75) -> Optional[Path]:
    """Pick lead-time output file, preferring a specific percentile if present."""
    analysis_dir = project_root / "results" / "analysis"
    preferred = analysis_dir / f"{prefix}_p{prefer_percentile}.{ext}"
    if preferred.exists():
        return preferred
    candidates = sorted(analysis_dir.glob(f"{prefix}_p*.{ext}"))
    if candidates:
        return candidates[0]
    return None


# =============================================================================
# SECTION 1: TRACK A PERFORMANCE FIGURES
# =============================================================================

def plot_auc_comparison_bar() -> None:
    """
    Create AUC comparison bar plot for all baseline models.
    """
    print("\n  Generating AUC comparison bar plot...")
    
    # Load baseline comparison results
    metrics_dir = project_root / "results" / "metrics"
    metrics_path = metrics_dir / "baseline_comparison.json"

    if not metrics_path.exists():
        # Sensitivity runs write suffixed files (e.g., baseline_comparison_p75.json).
        p75_path = metrics_dir / "baseline_comparison_p75.json"
        if p75_path.exists():
            metrics_path = p75_path
        else:
            candidates = sorted(metrics_dir.glob("baseline_comparison_p*.json"))
            if candidates:
                metrics_path = candidates[0]
            else:
                warnings.warn(f"Baseline comparison file not found: {metrics_path}")
                return
    
    with open(metrics_path, 'r') as f:
        baseline_results = json.load(f)
    
    # Extract AUC values - results are under 'models' key
    models_data = baseline_results.get('models', baseline_results)
    
    models = []
    aucs = []
    colors = []
    
    model_color_map = {
        'xgboost': COLORS['xgboost'],
        'random_forest': COLORS['rf'],
        'logistic': COLORS['logistic'],
        'threshold': COLORS['baseline'],
    }
    
    for model_name, metrics in models_data.items():
        if isinstance(metrics, dict) and 'auc_mean' in metrics:
            models.append(model_name.replace('_', ' ').title())
            aucs.append(metrics['auc_mean'])
            colors.append(model_color_map.get(model_name, COLORS['baseline']))
    
    # Add Bayesian (from Phase 5 results)
    bayesian_path = project_root / "results" / "metrics" / "bayesian_cv_results.json"
    if bayesian_path.exists():
        with open(bayesian_path, 'r') as f:
            bayesian_results = json.load(f)
        # Check in aggregated first, then top-level
        if 'aggregated' in bayesian_results and 'auc_mean' in bayesian_results['aggregated']:
            models.append('Bayesian')
            aucs.append(bayesian_results['aggregated']['auc_mean'])
            colors.append(COLORS['bayesian'])
        elif 'auc' in bayesian_results:
            models.append('Bayesian')
            aucs.append(bayesian_results['auc'])
            colors.append(COLORS['bayesian'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    bars = ax.bar(x, aucs, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Cross-Validation AUC Comparison: Track A Baselines vs Bayesian')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    save_figure_with_description(
        fig, TRACK_A_DIR / "auc_comparison_bar",
        title="AUC Comparison: Track A Baselines vs Bayesian",
        description="Bar chart comparing Area Under the ROC Curve (AUC-ROC) "
                   "for XGBoost, Random Forest, Logistic Regression, and Bayesian "
                   "hierarchical state-space model across rolling-origin CV folds.",
        interpretation="Higher AUC indicates better discrimination between outbreak "
                      "and non-outbreak weeks. XGBoost achieves ~0.76, while Bayesian "
                      "achieves ~0.52. This is expected: Bayesian is designed for "
                      "RISK INFERENCE, not binary classification. The comparison "
                      "motivates using lead-time as the primary evaluation metric.",
        caveats="Bayesian model output is continuous risk (Z_t), not probability. "
               "AUC computed using posterior predictive threshold."
    )


def plot_lead_time_distribution() -> None:
    """
    Create lead-time distribution comparison (Bayesian vs XGBoost).
    """
    print("\n  Generating lead-time distribution plot...")
    
    # Load lead-time results
    detail_path = select_lead_time_file("lead_time_detail", "csv")
    
    if detail_path is None or not detail_path.exists():
        warnings.warn("Lead-time detail file not found (lead_time_detail_p*.csv)")
        return
    
    df = pd.read_csv(detail_path)
    
    # Extract lead times (filter out -1 = never warned)
    bayesian_leads = df['lead_time_bayesian'][df['lead_time_bayesian'] >= 0].values
    xgboost_leads = df['lead_time_xgboost'][df['lead_time_xgboost'] >= 0].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Histogram
    ax1 = axes[0]
    bins = np.arange(-0.5, max(max(bayesian_leads, default=0), max(xgboost_leads, default=0)) + 1.5, 1)
    
    if len(bayesian_leads) > 0:
        ax1.hist(bayesian_leads, bins=bins, alpha=0.7, color=COLORS['bayesian'], 
                label=f'Bayesian (n={len(bayesian_leads)})', edgecolor='black')
    if len(xgboost_leads) > 0:
        ax1.hist(xgboost_leads, bins=bins, alpha=0.7, color=COLORS['xgboost'],
                label=f'XGBoost (n={len(xgboost_leads)})', edgecolor='black')
    
    ax1.set_xlabel('Lead Time (weeks)')
    ax1.set_ylabel('Count')
    ax1.set_title('A. Lead-Time Distribution')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel B: Box plot
    ax2 = axes[1]
    data_to_plot = []
    labels = []
    colors_box = []
    
    if len(bayesian_leads) > 0:
        data_to_plot.append(bayesian_leads)
        labels.append('Bayesian')
        colors_box.append(COLORS['bayesian'])
    if len(xgboost_leads) > 0:
        data_to_plot.append(xgboost_leads)
        labels.append('XGBoost')
        colors_box.append(COLORS['xgboost'])
    
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Lead Time (weeks)')
    ax2.set_title('B. Lead-Time Box Plot')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add warning statistics
    n_total = len(df)
    n_bayesian_warned = (df['lead_time_bayesian'] >= 0).sum()
    n_xgboost_warned = (df['lead_time_xgboost'] >= 0).sum()
    
    stats_text = (f"Total episodes: {n_total}\n"
                 f"Bayesian warned: {n_bayesian_warned} ({100*n_bayesian_warned/n_total:.0f}%)\n"
                 f"XGBoost warned: {n_xgboost_warned} ({100*n_xgboost_warned/n_total:.0f}%)")
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    save_figure_with_description(
        fig, LEAD_TIME_DIR / "lead_time_distribution",
        title="Lead-Time Distribution: Bayesian vs XGBoost",
        description="Panel A: Histogram of lead times (weeks before outbreak) for "
                   "episodes where models triggered. Panel B: Box plot showing median, "
                   "IQR, and outliers. Only includes episodes where models warned.",
        interpretation="Positive lead time = model warned before outbreak. "
                      "Higher lead time = more advance warning. Bayesian model "
                      "is designed to capture gradual risk escalation, potentially "
                      "warning earlier than binary classifiers.",
        caveats="Episodes where model never warned (lead_time = -1) are excluded "
               "from these plots. See summary table for never-warned rates."
    )


def plot_differential_lead_histogram() -> None:
    """
    Create differential lead-time (ΔL) histogram.
    """
    print("\n  Generating differential lead-time histogram...")
    
    # Load lead-time results
    detail_path = select_lead_time_file("lead_time_detail", "csv")
    
    if detail_path is None or not detail_path.exists():
        warnings.warn("Lead-time detail file not found (lead_time_detail_p*.csv)")
        return
    
    df = pd.read_csv(detail_path)
    
    # Filter to episodes where BOTH models warned (can compute ΔL)
    valid_mask = (df['lead_time_bayesian'] >= 0) & (df['lead_time_xgboost'] >= 0)
    delta_leads = df.loc[valid_mask, 'differential_lead'].dropna().values
    
    if len(delta_leads) == 0:
        print("    No valid differential lead times to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(min(delta_leads) - 0.5, max(delta_leads) + 1.5, 1)
    ax.hist(delta_leads, bins=bins, color='purple', alpha=0.7, edgecolor='black')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='ΔL = 0 (same timing)')
    ax.axvline(x=np.mean(delta_leads), color='blue', linestyle=':', linewidth=2,
              label=f'Mean ΔL = {np.mean(delta_leads):.2f}')
    
    ax.set_xlabel('Differential Lead Time ΔL = L_XGB - L_Bayes (weeks)')
    ax.set_ylabel('Count')
    ax.set_title('Differential Lead Time Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add interpretation zones
    ax.text(0.02, 0.95, '← Bayesian earlier', transform=ax.transAxes, fontsize=10,
           color='blue', verticalalignment='top')
    ax.text(0.98, 0.95, 'XGBoost earlier →', transform=ax.transAxes, fontsize=10,
           color='red', verticalalignment='top', horizontalalignment='right')
    
    save_figure_with_description(
        fig, LEAD_TIME_DIR / "differential_lead_histogram",
        title="Differential Lead Time (ΔL) Distribution",
        description="Histogram of ΔL = L_XGB - L_Bayes for episodes where both "
                   "models triggered. Positive ΔL means Bayesian warned earlier.",
        interpretation="If ΔL > 0: Bayesian provided more advance warning. "
                      "If ΔL < 0: XGBoost provided more advance warning. "
                      "If ΔL = 0: Both models warned at the same time.",
        caveats="Only includes episodes where BOTH models warned. "
               "Episodes where one or both never warned are excluded."
    )


def plot_per_fold_median_lead() -> None:
    """
    Create per-fold median lead-time comparison.
    """
    print("\n  Generating per-fold median lead-time plot...")
    
    # Load summary by fold
    summary_path = select_lead_time_file("lead_time_summary_by_fold", "csv")
    
    if summary_path is None or not summary_path.exists():
        warnings.warn("Summary by fold file not found (lead_time_summary_by_fold_p*.csv)")
        return
    
    df = pd.read_csv(summary_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Handle NaN values for plotting
    bayesian_medians = df['bayesian_median_lead'].fillna(0).values
    xgboost_medians = df['xgboost_median_lead'].fillna(0).values
    
    bars1 = ax.bar(x - width/2, bayesian_medians, width, label='Bayesian',
                  color=COLORS['bayesian'], edgecolor='black')
    bars2 = ax.bar(x + width/2, xgboost_medians, width, label='XGBoost',
                  color=COLORS['xgboost'], edgecolor='black')
    
    ax.set_xlabel('CV Fold (Test Year)')
    ax.set_ylabel('Median Lead Time (weeks)')
    ax.set_title('Median Lead Time by CV Fold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['test_year'].astype(str))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add episode counts as annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, max(bayesian_medians[i], xgboost_medians[i]) + 0.5,
               f'n={int(row["n_episodes"])}', ha='center', fontsize=9)
    
    save_figure_with_description(
        fig, LEAD_TIME_DIR / "per_fold_median_lead",
        title="Median Lead Time by CV Fold",
        description="Bar chart comparing median lead time (in weeks) for Bayesian "
                   "and XGBoost models across each rolling-origin CV fold. "
                   "Episode count (n) shown above each fold.",
        interpretation="Fold-level variation reflects year-to-year differences in "
                      "outbreak patterns. Some folds may have zero outbreaks or "
                      "models that never warned (shown as 0).",
        caveats="Folds with zero episodes or models that never warned show 0 median. "
               "Small n per fold limits statistical power."
    )


def plot_case_study_timeseries() -> None:
    """
    Create case study time series showing cases, Z_t, P_t, and threshold crossings.
    """
    print("\n  Generating case study time series...")
    
    preds_path = select_lead_time_file("lead_time_predictions", "parquet")
    detail_path = select_lead_time_file("lead_time_detail", "csv")

    if preds_path is None or detail_path is None or not preds_path.exists() or not detail_path.exists():
        warnings.warn(
            "Case study requires lead-time outputs: "
            "lead_time_predictions_p*.parquet and lead_time_detail_p*.csv"
        )
        return

    preds = pd.read_parquet(preds_path)
    episodes = pd.read_csv(detail_path)
    if episodes.empty:
        warnings.warn("No outbreak episodes found for case study plot")
        return

    # Pick first episode as a deterministic case study
    ep = episodes.iloc[0]
    state = ep['state']
    district = ep['district']
    year = int(ep['year'])

    ts = preds[(preds['state'] == state) & (preds['district'] == district) & (preds['year'] == year)].copy()
    if ts.empty:
        warnings.warn(f"No prediction time series found for {state}/{district}/{year}")
        return

    ts = ts.sort_values('week')
    weeks = ts['week'].to_numpy(dtype=int)
    cases = ts['cases'].to_numpy(dtype=float) if 'cases' in ts.columns else None
    prob = ts['prob'].to_numpy(dtype=float) if 'prob' in ts.columns else None
    z_mean = ts['z_mean'].to_numpy(dtype=float) if 'z_mean' in ts.columns else None

    outbreak_threshold = float(ep.get('outbreak_threshold', np.nan))
    bayes_threshold = float(ep.get('bayesian_threshold', np.nan))
    xgb_threshold = float(ep.get('xgboost_threshold', np.nan))
    t_outbreak = int(ep.get('first_outbreak_week', ep.get('outbreak_week', np.nan))) if not pd.isna(ep.get('first_outbreak_week', np.nan)) else None
    t_bayes = ep.get('bayesian_trigger_week', None)
    t_xgb = ep.get('xgboost_trigger_week', None)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel A: cases and outbreak threshold
    ax = axes[0]
    if cases is not None:
        ax.plot(weeks, cases, color='black', linewidth=1.5, label='Cases')
    if not np.isnan(outbreak_threshold):
        ax.axhline(outbreak_threshold, color=COLORS['outbreak'], linestyle='--', label='Outbreak threshold')
    if t_outbreak is not None:
        ax.axvline(t_outbreak, color=COLORS['outbreak'], linestyle=':', label='First outbreak week')
    ax.set_title(f"A. Observed Cases ({state} / {district}, {year})")
    ax.set_ylabel('Cases')
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left')

    # Panel B: Bayesian risk
    ax = axes[1]
    if z_mean is not None:
        ax.plot(weeks, z_mean, color=COLORS['bayesian'], linewidth=1.5, label='Bayesian z_mean')
    if not np.isnan(bayes_threshold):
        ax.axhline(bayes_threshold, color=COLORS['bayesian'], linestyle='--', label='Bayesian threshold')
    if pd.notna(t_bayes):
        ax.axvline(int(t_bayes), color=COLORS['bayesian'], linestyle=':')
    ax.set_title('B. Bayesian Latent Risk')
    ax.set_ylabel('z_mean')
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left')

    # Panel C: XGBoost probability
    ax = axes[2]
    if prob is not None:
        ax.plot(weeks, prob, color=COLORS['xgboost'], linewidth=1.5, label='XGBoost prob')
    if not np.isnan(xgb_threshold):
        ax.axhline(xgb_threshold, color=COLORS['xgboost'], linestyle='--', label='XGBoost threshold')
    if pd.notna(t_xgb):
        ax.axvline(int(t_xgb), color=COLORS['xgboost'], linestyle=':')
    ax.set_title('C. XGBoost Predicted Probability')
    ax.set_xlabel('Week')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left')

    plt.tight_layout()

    save_figure_with_description(
        fig,
        LEAD_TIME_DIR / "case_study_timeseries",
        title="Case Study Time Series",
        description="Three-panel time series for one outbreak episode: cases with outbreak threshold, "
                    "Bayesian latent risk (z_mean) with its threshold, and XGBoost probability with its threshold. "
                    "Vertical lines indicate the first threshold crossing weeks.",
        interpretation="Illustrates the temporal relationship between rising risk signals and the onset of outbreak "
                      "weeks in a single district-year.",
        caveats="This plot depends on stored per-row predictions produced by experiments/06_analyze_lead_time.py."
    )


def plot_calibration_curves() -> None:
    """
    Create reliability (calibration) curves for probability models.
    """
    print("\n  Generating calibration curves...")
    
    # Prefer Track A stored CV predictions if available; fallback to lead-time predictions.
    preds_dir = project_root / "results" / "predictions"
    baseline_preds_path = preds_dir / "baseline_cv_predictions_xgboost.parquet"
    lead_time_preds_path = select_lead_time_file("lead_time_predictions", "parquet")

    if not baseline_preds_path.exists():
        # Sensitivity runs write suffixed files (e.g., baseline_cv_predictions_xgboost_p75.parquet).
        p75_path = preds_dir / "baseline_cv_predictions_xgboost_p75.parquet"
        if p75_path.exists():
            baseline_preds_path = p75_path
        else:
            candidates = sorted(preds_dir.glob("baseline_cv_predictions_xgboost_p*.parquet"))
            if candidates:
                baseline_preds_path = candidates[0]

    if baseline_preds_path.exists():
        df = pd.read_parquet(baseline_preds_path)
        y_true = df['y_true'].to_numpy(dtype=int)
        y_prob = df['y_pred_proba'].to_numpy(dtype=float)
        source = str(baseline_preds_path)
    elif lead_time_preds_path is not None and lead_time_preds_path.exists():
        df = pd.read_parquet(lead_time_preds_path)
        if 'y_true' not in df.columns or 'prob' not in df.columns:
            warnings.warn(f"Lead-time predictions missing y_true/prob: {lead_time_preds_path}")
            return
        y_true = df['y_true'].to_numpy(dtype=int)
        y_prob = df['prob'].to_numpy(dtype=float)
        source = str(lead_time_preds_path)
    else:
        warnings.warn("No stored prediction probabilities found for calibration plot")
        return

    # Bin-based reliability curve (no sklearn dependency).
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    mean_pred = []
    frac_pos = []
    counts = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_pred.append(float(np.mean(y_prob[mask])))
        frac_pos.append(float(np.mean(y_true[mask])))
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, marker='o', color=COLORS['xgboost'], label='XGBoost')

    for x, y, n in zip(mean_pred, frac_pos, counts):
        ax.text(x, y, f"n={n}", fontsize=8, ha='left', va='bottom')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Fraction of Outbreaks')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    save_figure_with_description(
        fig,
        CALIBRATION_DIR / "calibration_curves",
        title="Calibration Curve (Reliability Diagram)",
        description="Reliability diagram for XGBoost probabilities binned into deciles. "
                    "Each point shows mean predicted probability vs observed outbreak rate in that bin.",
        interpretation="If points lie below the diagonal, the model is overconfident; above indicates underconfidence. "
                      "Calibration matters for decision thresholds and alert credibility.",
        caveats=f"Computed from stored predictions: {source}. Bayesian z_mean is a risk signal, not a probability."
    )


def plot_decision_state_timeline() -> None:
    """Plot decision-state timeline using stored lead-time predictions."""
    print("\n  Generating decision state timeline...")

    preds_path = select_lead_time_file("lead_time_predictions", "parquet")
    detail_path = select_lead_time_file("lead_time_detail", "csv")

    if preds_path is None or detail_path is None or not preds_path.exists() or not detail_path.exists():
        warnings.warn(
            "Decision timeline requires lead-time outputs: "
            "lead_time_predictions_p*.parquet and lead_time_detail_p*.csv"
        )
        return

    preds = pd.read_parquet(preds_path)
    episodes = pd.read_csv(detail_path)
    if episodes.empty:
        warnings.warn("No outbreak episodes found for decision timeline")
        return

    ep = episodes.iloc[0]
    state = ep['state']
    district = ep['district']
    year = int(ep['year'])

    ts = preds[(preds['state'] == state) & (preds['district'] == district) & (preds['year'] == year)].copy()
    if ts.empty:
        warnings.warn(f"No prediction time series found for {state}/{district}/{year}")
        return

    ts = ts.sort_values('week')
    weeks = ts['week'].to_numpy(dtype=int)
    cases = ts['cases'].to_numpy(dtype=float) if 'cases' in ts.columns else None
    prob = ts['prob'].to_numpy(dtype=float) if 'prob' in ts.columns else None
    z_mean = ts['z_mean'].to_numpy(dtype=float) if 'z_mean' in ts.columns else None

    outbreak_threshold = float(ep.get('outbreak_threshold', np.nan))
    bayes_threshold = float(ep.get('bayesian_threshold', np.nan))
    xgb_threshold = float(ep.get('xgboost_threshold', np.nan))
    t_outbreak = ep.get('first_outbreak_week', ep.get('outbreak_week', np.nan))
    if pd.isna(outbreak_threshold) or pd.isna(bayes_threshold) or pd.isna(xgb_threshold):
        warnings.warn("Decision timeline missing thresholds in lead_time_detail")
        return

    if cases is None or prob is None or z_mean is None:
        warnings.warn("Decision timeline requires cases, prob, and z_mean in predictions")
        return

    outbreak_mask = cases > outbreak_threshold
    bayes_mask = z_mean >= bayes_threshold
    xgb_mask = prob >= xgb_threshold

    fig, axes = plt.subplots(3, 1, figsize=(14, 4.5), sharex=True)

    def plot_state_row(ax, mask, color, label):
        ax.fill_between(weeks, 0, 1, color='#F0F0F0', step='mid')
        ax.fill_between(weeks, 0, 1, where=mask, color=color, alpha=0.85, step='mid')
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=35, va='center')
        ax.grid(axis='x', alpha=0.2)

    plot_state_row(axes[0], outbreak_mask, COLORS['outbreak'], 'Outbreak')
    plot_state_row(axes[1], bayes_mask, COLORS['bayesian'], 'Bayesian')
    plot_state_row(axes[2], xgb_mask, COLORS['xgboost'], 'XGBoost')

    if pd.notna(t_outbreak):
        for ax in axes:
            ax.axvline(int(t_outbreak), color=COLORS['outbreak'], linestyle=':', linewidth=1.5)

    axes[2].set_xlabel('Week')
    axes[0].set_title(f"Decision State Timeline ({state} / {district}, {year})")

    plt.tight_layout()

    save_figure_with_description(
        fig,
        DECISION_DIR / "decision_state_timeline",
        title="Decision State Timeline",
        description="Binary alert states over weeks for a single district-year. "
                    "Rows show outbreak weeks (cases above threshold), Bayesian alerts, "
                    "and XGBoost alerts, computed from stored predictions and thresholds.",
        interpretation="Shows how model alerts align with observed outbreak timing in a "
                      "real district-year sequence.",
        caveats="Uses the first available outbreak episode in lead-time outputs. "
               "Thresholds come from the corresponding lead_time_detail_p*.csv."
    )


def create_readme() -> None:
    """
    Create README.md for the Phase 7 figures directory.
    """
    print("\n  Creating README.md...")
    
    readme_content = """# Phase 7 Visualization Results

## Directory Structure

```
results/figures/phase7/
├── 01_trackA_performance/    # Baseline model comparison
│   ├── auc_comparison_bar.png
│   └── auc_comparison_bar.txt
├── 02_lead_time_analysis/    # Lead-time comparison (Bayesian vs XGBoost)
│   ├── lead_time_distribution.png
│   ├── differential_lead_histogram.png
│   ├── per_fold_median_lead.png
│   └── case_study_timeseries.png
├── 03_calibration_uncertainty/   # Calibration and uncertainty quantification
│   └── calibration_curves.png
└── README.md                     # This file
```

## Thesis Chapter Mapping

### Chapter: Methods (Track A Baselines)
- `01_trackA_performance/auc_comparison_bar.png`: Figure showing AUC comparison

### Chapter: Results (Lead-Time Analysis)
- `02_lead_time_analysis/lead_time_distribution.png`: Main lead-time comparison
- `02_lead_time_analysis/differential_lead_histogram.png`: ΔL distribution
- `02_lead_time_analysis/per_fold_median_lead.png`: Fold-level variability

### Chapter: Results (Calibration & Uncertainty)
- `03_calibration_uncertainty/calibration_curves.png`: Reliability

## Figure Standards

All figures include:
- Clear title
- Axis labels with units
- Legend (where applicable)
- Saved at 300 DPI
- Accompanying .txt file with interpretation

## Data Sources

- Lead-time results: `results/analysis/lead_time_*_p*.csv`
- Baseline metrics: `results/metrics/baseline_comparison.json`
- Bayesian metrics: `results/metrics/bayesian_cv_results.json`

## Generated

Date: {date}
Version: v4.2 (Phase 7 Complete)
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    readme_path = FIGURE_ROOT / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ {readme_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all visualization generation."""
    print("=" * 70)
    print("PHASE 7 VISUALIZATION GENERATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {FIGURE_ROOT}")
    
    # Section 1: Track A Performance
    print("\n" + "-" * 70)
    print("SECTION 1: Track A Performance Figures")
    print("-" * 70)
    plot_auc_comparison_bar()
    
    # Section 2: Lead-Time Analysis
    print("\n" + "-" * 70)
    print("SECTION 2: Lead-Time Analysis Figures")
    print("-" * 70)
    plot_lead_time_distribution()
    plot_differential_lead_histogram()
    plot_per_fold_median_lead()
    plot_case_study_timeseries()
    
    # Section 3: Calibration & Uncertainty
    print("\n" + "-" * 70)
    print("SECTION 3: Calibration & Uncertainty Figures")
    print("-" * 70)
    plot_calibration_curves()
    
    # Section 4: Decision Usefulness
    print("\n" + "-" * 70)
    print("SECTION 4: Decision Usefulness Figures")
    print("-" * 70)
    plot_decision_state_timeline()
    
    # Create README
    print("\n" + "-" * 70)
    print("DOCUMENTATION")
    print("-" * 70)
    create_readme()
    
    print("\n" + "=" * 70)
    print("✓ Phase 7 Visualizations Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
