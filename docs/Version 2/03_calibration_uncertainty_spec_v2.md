# 03_calibration_uncertainty_spec_v2.md

## Calibration & Uncertainty Sanity Check Specification

**Priority:** High  
**Purpose:** Validate that Bayesian model is well-calibrated and uncertainty-aware; reframe "lower AUC" as "lower false confidence"  
**Effort:** Light (~1 week implementation)  
**Output:** Reliability curves + coverage plot + Brier score table + interpretation

---

## 1. Conceptual Overview

### The Problem

AUC measures **discrimination** (can the model separate outbreak from non-outbreak?).

Calibration measures **confidence** (when the model predicts 60% probability, do outbreaks actually occur ~60% of the time?).

**Your thesis claim:** Bayesian model is less discriminative (lower AUC) but better calibrated. This is not a weakness — it's a **feature**. A conservative, well-calibrated model is trustworthy.

### Why This Matters

- XGBoost may confidently predict "no outbreak" even when uncertain.
- Bayesian model knows when it is uncertain; this uncertainty is operationally valuable.
- Calibration enables **principled decision-making** without arbitrary threshold tweaking.

---

## 2. What is Calibration?

### Definition

A probabilistic model is **calibrated** if:

> When the model outputs probability \( p \), the event occurs approximately \( p \) fraction of the time.

**Example:**

- If XGBoost predicts \( \hat{p} = 0.7 \) (70% chance of outbreak) for 100 weeks, then ideally ~70 of those weeks should have outbreaks.
- If the actual count is 30 (30%), the model is **overconfident**.
- If the actual count is 80 (80%), the model is **underconfident** (conservative).

### Metrics

1. **Brier Score:** \( \text{BS} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2 \)
   - Lower is better (range: 0–1).
   - Combines discrimination + calibration.
   - Useful for comparing models.

2. **Reliability Curve (Calibration Plot):**
   - X-axis: Predicted probability \( \hat{p} \).
   - Y-axis: Observed frequency (empirical frequency of events given \( \hat{p} \)).
   - Perfect calibration: points lie on the diagonal \( y = x \).
   - Overconfident: points below diagonal (predicted high, occurs low).
   - Underconfident: points above diagonal (predicted low, occurs high).

3. **Credible Interval Coverage (Bayesian only):**
   - For each prediction, compute the \( \alpha \)% credible interval (e.g., 90% CI).
   - Check: how many times does the observed outcome fall inside this interval?
   - Perfect coverage: observed outcome inside ~\( \alpha \)% of intervals.

---

## 3. Reliability Curve (Calibration Plot)

### 3.1 Computation

**For each model (XGBoost and Bayesian):**

1. **Bin predicted probabilities:**
   - Divide the range [0, 1] into bins (e.g., [0–0.1], [0.1–0.2], ..., [0.9–1.0]).
   - Typically 10 bins of width 0.1.

2. **For each bin:**
   - Collect all predictions \( \hat{p}_i \) in that bin.
   - Compute empirical frequency: \( \text{freq} = \frac{\# \text{outbreaks in bin}}{|\text{bin}|} \).
   - X-coordinate: midpoint of bin (e.g., 0.05 for [0–0.1]).
   - Y-coordinate: \( \text{freq} \).

3. **Plot:**
   - X-axis: predicted probability (0 to 1).
   - Y-axis: observed frequency (0 to 1).
   - Points: one per bin.
   - Diagonal line: perfect calibration \( y = x \).
   - Model performance: distance from diagonal.

### 3.2 Side-by-Side Comparison

**One figure with two subplots:**

- **Left:** XGBoost reliability curve.
- **Right:** Bayesian reliability curve.

**Annotations:**

- Diagonal line (perfect calibration).
- Text: Brier score for each model.
- Optional: number of observations per bin (bar width proportional to frequency).

### 3.3 Interpretation

**Optimistic scenario:**

- Bayesian curve lies close to diagonal → well-calibrated.
- XGBoost curve lies below diagonal → overconfident.
- **Interpretation:** "Bayesian uncertainty is trustworthy; XGBoost overestimates confidence."

**Conservative scenario:**

- Both models show some miscalibration (expected under class imbalance).
- Bayesian still closer to diagonal than XGBoost.
- **Interpretation:** "Even with miscalibration, Bayesian is more conservative and less prone to false confidence."

**Pre-emptive note (important):**

> "Under strong class imbalance (outbreaks ~5% of weeks), discriminative models like XGBoost often exhibit miscalibration because they are optimized for discrimination, not probability. This is a known phenomenon and does not invalidate the comparison; rather, it highlights why probability-based framing (Bayesian) is more appropriate for early warning."

---

## 4. Brier Score

### 4.1 Definition & Computation

**Brier Score:**

\[ \text{BS} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2 \]

where:
- \( \hat{p}_i \) = predicted probability (model's forecast).
- \( y_i \) = observed outcome (1 = outbreak occurred, 0 = no outbreak).
- \( n \) = number of predictions.

**Range:** 0 (perfect) to 1 (worst).

**Interpretation:**
- BS = 0.10 means the average squared error between prediction and outcome is 0.10.
- Lower is better.

### 4.2 Decomposition (Optional, Advanced)

Brier score can be decomposed into three components:

\[ \text{BS} = \text{Reliability} + \text{Resolution} - \text{Uncertainty} \]

- **Reliability:** calibration error (how far predictions deviate from observed frequencies).
- **Resolution:** discrimination (how much predictions vary).
- **Uncertainty:** baseline difficulty (inherent randomness of the problem).

**Advantage:** Shows whether an increase in BS is due to miscalibration (bad) or poor discrimination (acceptable for Bayesian).

---

## 5. Bayesian Credible Interval Coverage

### 5.1 Definition

For Bayesian model, compute **credible intervals** (Bayesian confidence intervals):

- **90% credible interval:** range \( [q_{0.05}(Z_t), q_{0.95}(Z_t)] \) where \( q_\alpha \) is the \( \alpha \)-quantile of the posterior \( p(Z_t | \text{data}) \).
- For each week \( t \), compute whether observed cases \( Y_t \) fall within this interval.

### 5.2 Coverage Calculation

**For each nominal level \( \alpha \in \{0.50, 0.68, 0.90, 0.95\} \):**

1. Compute \( (100 \alpha) \)% credible intervals for all predictions in test set.
2. Count how many observed outcomes fall inside the intervals.
3. Compute empirical coverage: \( \text{Coverage}_{\text{obs}} = \frac{\# \text{inside}}{n} \).
4. Compare to nominal: \( \text{Coverage}_{\text{nominal}} = \alpha \).

**Perfect calibration:** \( \text{Coverage}_{\text{obs}} \approx \text{Coverage}_{\text{nominal}} \).

### 5.3 Coverage Plot

**Type:** Diagram with two axes.

- **X-axis:** Nominal coverage level (0.50, 0.68, 0.90, 0.95).
- **Y-axis:** Observed coverage (0–1).
- **Line 1:** Perfect calibration (diagonal \( y = x \)).
- **Line 2:** Your Bayesian model's observed coverage (with error bars if computed across folds).

**Interpretation:**

- If points lie on diagonal → Bayesian credible intervals are well-calibrated.
- If points above diagonal → intervals too wide (conservative, underconfident).
- If points below diagonal → intervals too narrow (overconfident).

---

## 6. Implementation Pseudocode

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def compute_brier_score(y_true, y_pred_prob):
    """
    Args:
        y_true: binary labels (0/1).
        y_pred_prob: predicted probabilities [0, 1].
    
    Returns:
        brier_score: scalar in [0, 1].
    """
    return np.mean((y_pred_prob - y_true) ** 2)


def compute_reliability_curve(y_true, y_pred_prob, n_bins=10):
    """
    Args:
        y_true: binary labels.
        y_pred_prob: predicted probabilities.
        n_bins: number of bins.
    
    Returns:
        bin_centers: array of shape (n_bins,), bin midpoints.
        observed_freq: array of shape (n_bins,), observed frequency per bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    observed_freq = []
    for i in range(n_bins):
        mask = (y_pred_prob >= bin_edges[i]) & (y_pred_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            freq = y_true[mask].mean()
        else:
            freq = np.nan  # No predictions in this bin
        observed_freq.append(freq)
    
    return bin_centers, np.array(observed_freq)


def plot_reliability_curves(y_xgb, p_xgb, y_bayesian, p_bayesian_mean):
    """
    Plot reliability curves for both models side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # XGBoost
    bin_centers_xgb, freq_xgb = compute_reliability_curve(y_xgb, p_xgb, n_bins=10)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    axes[0].scatter(bin_centers_xgb, freq_xgb, s=100, alpha=0.6, label='XGBoost')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Observed Frequency')
    axes[0].set_title(f'XGBoost Calibration (BS={compute_brier_score(y_xgb, p_xgb):.3f})')
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    # Bayesian
    bin_centers_bay, freq_bay = compute_reliability_curve(y_bayesian, p_bayesian_mean, n_bins=10)
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    axes[1].scatter(bin_centers_bay, freq_bay, s=100, alpha=0.6, color='orange', label='Bayesian')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Observed Frequency')
    axes[1].set_title(f'Bayesian Calibration (BS={compute_brier_score(y_bayesian, p_bayesian_mean):.3f})')
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def compute_credible_interval_coverage(y_true, Z_posterior_samples, nominal_levels=[0.50, 0.68, 0.90, 0.95]):
    """
    Args:
        y_true: observed counts.
        Z_posterior_samples: array of shape (n_weeks, n_samples), posterior samples.
        nominal_levels: list of coverage levels to check.
    
    Returns:
        coverage_dict: dict mapping nominal level -> observed coverage.
    """
    coverage_dict = {}
    
    for alpha in nominal_levels:
        q_lower = (1 - alpha) / 2
        q_upper = 1 - q_lower
        
        ci_lower = np.quantile(Z_posterior_samples, q_lower, axis=1)
        ci_upper = np.quantile(Z_posterior_samples, q_upper, axis=1)
        
        # Check if observed falls in interval
        inside = (y_true >= ci_lower) & (y_true <= ci_upper)
        observed_coverage = inside.mean()
        
        coverage_dict[alpha] = observed_coverage
    
    return coverage_dict


def plot_credible_interval_coverage(coverage_dict, nominal_levels=[0.50, 0.68, 0.90, 0.95]):
    """
    Plot observed vs nominal coverage.
    """
    observed_cov = [coverage_dict[alpha] for alpha in nominal_levels]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(nominal_levels, nominal_levels, 'k--', label='Perfect calibration', linewidth=2)
    ax.plot(nominal_levels, observed_cov, 'o-', label='Bayesian model', linewidth=2, markersize=8)
    ax.fill_between(nominal_levels, [n - 0.05 for n in nominal_levels], 
                     [n + 0.05 for n in nominal_levels], alpha=0.2, label='±5% tolerance')
    ax.set_xlabel('Nominal Coverage Level')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Bayesian Credible Interval Coverage')
    ax.legend()
    ax.set_xlim([0.4, 1])
    ax.set_ylim([0.4, 1])
    ax.grid(True, alpha=0.3)
    
    plt.savefig('credible_interval_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 7. Summary Table

**Output: One table comparing XGBoost vs Bayesian.**

| Metric | XGBoost | Bayesian | Interpretation |
|--------|---------|----------|-----------------|
| Brier Score | X.XXX | Y.YYY | Lower = better calibrated |
| Calibration (visual) | Overconfident below diagonal | Near diagonal | Bayesian more trustworthy |
| Credible interval coverage @ 90% | N/A | Z% | Should be ~90%; Z% means coverage quality |
| False confidence rate | High | Low | Bayesian reserves high probability for rare events |

---

## 8. Validation & Robustness

### 8.1 Stratification by District Size

Compute calibration metrics separately for:

- **Small districts** (few cases, low-data regime).
- **Large districts** (many cases, high-data regime).

**Hypothesis:** Bayesian (with hierarchical pooling) should be better calibrated in small districts.

---

### 8.2 Temporal Stability

Compute Brier score and coverage for each fold separately (2017, 2018, ..., 2022).

**Interpretation:** If Bayesian is more stable across years, it is a more reliable tool.

---

## 9. Pre-Emptive Note for Reviewers

Include this paragraph in your thesis Discussion:

> "Under strong class imbalance (outbreaks occurring in ~5% of weeks), discriminative models optimized for accuracy often exhibit miscalibration due to their focus on discrimination. This is a well-documented phenomenon and does not invalidate our comparison. Rather, it underscores why probabilistic framing—whether through Bayesian modeling or post-hoc calibration—is essential for reliable decision-making in early warning systems. Our Bayesian model's better calibration indicates that its uncertainty estimates can be trusted to guide operational decisions, even if its raw discrimination metrics are lower."

---

## 10. Integration with Thesis

**Section:** Results → "Calibration & Uncertainty Analysis" (1–2 pages).

**Figures:**

1. Side-by-side reliability curves (XGBoost vs Bayesian).
2. Credible interval coverage plot (Bayesian only).

**Tables:**

1. Summary table (Brier score, coverage %, interpretation).

**Text:**

1. Definitions of calibration and Brier score.
2. Interpretation of plots (what they show about each model).
3. Connection to decision-making: "Well-calibrated uncertainty enables principled action."

---

## 11. Success Criteria

✅ Brier score computed for both models (overall + per-fold).  
✅ Reliability curves generated and plotted (side-by-side).  
✅ Credible interval coverage computed for Bayesian (multiple nominal levels).  
✅ Summary table created (Brier, coverage, interpretation).  
✅ Pre-emptive note about class imbalance included.  
✅ Interpretation validates Claim 3 (or gracefully acknowledges limitations).

---

**Document Version:** 2.0  
**Status:** Ready for implementation  
**Effort:** Low (~1 week)  
**Last Updated:** February 3, 2026
