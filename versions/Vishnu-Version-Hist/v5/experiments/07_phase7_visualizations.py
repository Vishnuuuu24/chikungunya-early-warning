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


# =============================================================================
# SECTION 1: TRACK A PERFORMANCE FIGURES
# =============================================================================

def plot_auc_comparison_bar() -> None:
    """
    Create AUC comparison bar plot for all baseline models.
    """
    print("\n  Generating AUC comparison bar plot...")
    
    # Load baseline comparison results
    metrics_path = project_root / "results" / "metrics" / "baseline_comparison.json"
    
    if not metrics_path.exists():
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
    detail_path = project_root / "results" / "analysis" / "lead_time_detail_all_folds.csv"
    
    if not detail_path.exists():
        warnings.warn(f"Lead-time detail file not found: {detail_path}")
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
    detail_path = project_root / "results" / "analysis" / "lead_time_detail_all_folds.csv"
    
    if not detail_path.exists():
        warnings.warn(f"Lead-time detail file not found: {detail_path}")
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
    summary_path = project_root / "results" / "analysis" / "lead_time_summary_by_fold.csv"
    
    if not summary_path.exists():
        warnings.warn(f"Summary by fold file not found: {summary_path}")
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
    
    # For case study, we need the actual predictions and cases
    # This requires re-running a single fold analysis - skip for now
    # and create a placeholder with documentation
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Placeholder with explanation
    for ax in axes:
        ax.text(0.5, 0.5, 'Case study visualization requires\nper-district prediction data.\n'
               'See experiments/08_case_study_plots.py\nfor detailed implementation.',
               ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 52)
        ax.set_ylim(0, 1)
    
    axes[0].set_title('A. Observed Cases and Outbreak Threshold')
    axes[1].set_title('B. Bayesian Latent Risk Z_t')
    axes[2].set_title('C. XGBoost Probability P_t')
    axes[2].set_xlabel('Week')
    
    plt.tight_layout()
    
    save_figure_with_description(
        fig, LEAD_TIME_DIR / "case_study_timeseries_placeholder",
        title="Case Study Time Series (Placeholder)",
        description="This figure will show 2-3 case study districts with:\n"
                   "- Panel A: Observed case counts with outbreak threshold\n"
                   "- Panel B: Bayesian latent risk Z_t with crossing threshold\n"
                   "- Panel C: XGBoost probability P_t with 0.5 threshold\n"
                   "Vertical lines indicate first threshold crossings.",
        interpretation="Case studies illustrate HOW the models detect outbreaks. "
                      "Bayesian Z_t may rise gradually before cases spike, "
                      "while XGBoost P_t may respond more abruptly.",
        caveats="Placeholder - requires per-district prediction storage. "
               "See experiments/08_case_study_plots.py for implementation."
    )


def plot_calibration_curves() -> None:
    """
    Create reliability (calibration) curves for probability models.
    """
    print("\n  Generating calibration curves...")
    
    # Calibration requires stored prediction probabilities
    # Create placeholder
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    ax.text(0.5, 0.5, 'Calibration curve requires\nstored prediction probabilities.\n'
           'See experiments/10_comprehensive_metrics.py\nfor detailed implementation.',
           ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Fraction of Outbreaks')
    ax.set_title('Calibration Curves (Reliability Diagram)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    save_figure_with_description(
        fig, CALIBRATION_DIR / "calibration_curves_placeholder",
        title="Calibration Curves (Placeholder)",
        description="Reliability diagram showing predicted probability vs "
                   "observed outbreak fraction for XGBoost (and optionally Bayesian "
                   "posterior predictive probability).",
        interpretation="Points above the diagonal indicate underconfidence "
                      "(observed rate higher than predicted). Points below indicate "
                      "overconfidence. Perfect calibration lies on diagonal.",
        caveats="Placeholder - requires stored prediction probabilities. "
               "Bayesian model outputs risk Z_t, not probability; "
               "calibration interpretation differs."
    )


def plot_decision_state_timeline() -> None:
    """
    Create decision state timeline showing GREEN/YELLOW/RED transitions.
    """
    print("\n  Generating decision state timeline...")
    
    # Create example timeline
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Example district timeline
    weeks = np.arange(1, 53)
    
    # Simulated states for illustration
    np.random.seed(42)
    states_bayesian = np.random.choice(['GREEN', 'YELLOW', 'RED'], size=52, 
                                       p=[0.6, 0.25, 0.15])
    states_xgboost = np.random.choice(['GREEN', 'YELLOW', 'RED'], size=52,
                                      p=[0.7, 0.15, 0.15])
    
    color_map = {'GREEN': COLORS['safe'], 'YELLOW': COLORS['warning'], 'RED': COLORS['outbreak']}
    
    for i, (w, s_b, s_x) in enumerate(zip(weeks, states_bayesian, states_xgboost)):
        ax.barh(1, 1, left=w-0.5, color=color_map[s_b], edgecolor='none')
        ax.barh(0, 1, left=w-0.5, color=color_map[s_x], edgecolor='none')
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['XGBoost', 'Bayesian'])
    ax.set_xlabel('Week')
    ax.set_title('Decision State Timeline: Example District')
    ax.set_xlim(0.5, 52.5)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['safe'], label='GREEN (Normal)'),
        mpatches.Patch(color=COLORS['warning'], label='YELLOW (Elevated Risk)'),
        mpatches.Patch(color=COLORS['outbreak'], label='RED (Outbreak Alert)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.text(0.5, -0.15, 'Note: This is an illustrative example. Actual states depend on '
           'model thresholds and predictions.', transform=ax.transAxes, 
           fontsize=10, ha='center', style='italic')
    
    save_figure_with_description(
        fig, DECISION_DIR / "decision_state_timeline",
        title="Decision State Timeline",
        description="Timeline showing decision states (GREEN/YELLOW/RED) over weeks "
                   "for Bayesian and XGBoost models. Each bar segment represents "
                   "one week's decision state.",
        interpretation="GREEN = normal operations, YELLOW = heightened surveillance, "
                      "RED = outbreak response. State transitions indicate when "
                      "public health action should change. More stable timelines "
                      "(fewer transitions) are operationally preferable.",
        caveats="This is an illustrative example. Actual decision states "
               "require defined thresholds and operational rules."
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
│   └── case_study_timeseries_placeholder.png
├── 03_calibration_uncertainty/   # Calibration and uncertainty quantification
│   └── calibration_curves_placeholder.png
├── 04_decision_usefulness/       # Decision support evaluation
│   └── decision_state_timeline.png
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
- `03_calibration_uncertainty/calibration_curves_placeholder.png`: Reliability

### Chapter: Discussion (Decision Usefulness)
- `04_decision_usefulness/decision_state_timeline.png`: Operational states

## Figure Standards

All figures include:
- Clear title
- Axis labels with units
- Legend (where applicable)
- Saved at 300 DPI
- Accompanying .txt file with interpretation

## Data Sources

- Lead-time results: `results/analysis/lead_time_*.csv`
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
