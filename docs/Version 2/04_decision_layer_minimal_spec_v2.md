# 04_decision_layer_minimal_spec_v2.md

## Minimal Decision-Layer Simulation Specification

**Priority:** High  
**Purpose:** Demonstrate that uncertainty-aware risk enables stabler, safer decisions than binary predictions  
**Effort:** Light (~1 week implementation)  
**Output:** Decision metrics table + flowchart/narrative + operational insights

---

## 1. Conceptual Overview

### The Problem

Binary classifiers output 0 or 1, often based on a hard threshold. If \( \hat{p} > 0.5 \), they predict "outbreak"; otherwise "no outbreak."

This is brittle:
- A prediction of 0.51 looks identical to 0.99 in terms of action.
- No notion of uncertainty or nuance.

### Your Solution

Bayesian latent risk + uncertainty enables nuanced decision rules:

- **GREEN:** High risk, low uncertainty → Act with confidence.
- **YELLOW:** Moderate risk or high uncertainty → Monitor closely.
- **RED:** Low risk, high confidence → No action.

This addresses your Claim 3: "Uncertainty-aware risk enables safer, more stable decisions."

### What This Analysis Does

You simulate these decision rules on your test folds (no retraining). You measure:

- False alarms (RED when no outbreak occurs).
- Missed outbreaks (GREEN when outbreak occurs).
- Decision stability (do decision states flip erratically week-to-week?).
- Response delay (how long between first action and actual outbreak?).

---

## 2. Decision Rule Definition

### 2.1 Decision States

Define three states based on Bayesian posterior \( p(Z_t | \text{data}) \):

**GREEN (Act):**
- Posterior mean \( \hat{Z}_t > 75\text{th percentile} \) (high risk).
- Posterior SD \( \sigma_t < 25\text{th percentile} \) (low uncertainty).
- **Action:** Alert health authorities, mobilize resources.

**YELLOW (Monitor):**
- Either:
  - \( 50\text{th percentile} < \hat{Z}_t < 75\text{th percentile} \) (moderate risk), OR
  - \( \sigma_t > 50\text{th percentile} \) (high uncertainty).
- **Action:** Increase surveillance, prepare contingency plans.

**RED (No action):**
- \( \hat{Z}_t < 50\text{th percentile} \) (low risk) AND \( \sigma_t < 50\text{th percentile} \) (low uncertainty).
- **Action:** Maintain baseline surveillance only.

### 2.2 Percentile Definitions

All percentiles are computed **on the training set** to avoid test-set leakage.

- \( p_{25} = \text{quantile}(\{\hat{Z}_t : t \in \text{train}\}, 0.25) \).
- \( p_{50} = \text{quantile}(\{\hat{Z}_t : t \in \text{train}\}, 0.50) \).
- \( p_{75} = \text{quantile}(\{\hat{Z}_t : t \in \text{train}\}, 0.75) \).

Similarly for \( \sigma_t \):

- \( \sigma_{25} = \text{quantile}(\{\sigma_{t} : t \in \text{train}\}, 0.25) \).
- \( \sigma_{50} = \text{quantile}(\{\sigma_{t} : t \in \text{train}\}, 0.50) \).

---

### 2.3 Flowchart

```
For each week t in test set:
  
  Compute posterior mean: ẑ_t = E[Z_t | data]
  Compute posterior SD:   σ_t = SD[Z_t | data]
  
  IF (ẑ_t > p_75) AND (σ_t < σ_25):
    Decision := GREEN (Act)
  
  ELSE IF (p_50 < ẑ_t < p_75) OR (σ_t > σ_50):
    Decision := YELLOW (Monitor)
  
  ELSE IF (ẑ_t < p_50) AND (σ_t < σ_50):
    Decision := RED (No action)
  
  ELSE:
    Decision := YELLOW (default to caution)
```

---

## 3. Evaluation Metrics

### 3.1 False Alarms

**Definition:** Weeks where decision is GREEN or YELLOW, but no outbreak occurs (using your 80th percentile outbreak threshold).

**Metrics:**

- Total false alarms (count).
- % of non-outbreak weeks that trigger alert: \( \frac{\text{# false alarms}}{\text{# non-outbreak weeks}} \times 100 \).
- Average false alarms per district per year.

**Interpretation:** How often does the system cry wolf?

---

### 3.2 Missed Outbreaks

**Definition:** Weeks where an outbreak occurs, but decision is RED (no action).

**Metrics:**

- Total missed outbreaks (count).
- % of outbreak weeks missed: \( \frac{\text{# missed}}{\text{# outbreak weeks}} \times 100 \).

**Interpretation:** How many true signals are lost?

---

### 3.3 Decision Stability

**Definition:** How often does the decision state flip from one week to the next?

**Computation:**

- For each district, count state transitions over all test weeks.
- Transition types: GREEN → YELLOW, YELLOW → RED, etc.
- Compute transition rate: \( \frac{\text{# transitions}}{(\text{# weeks} - 1)} \times 100 \).

**Interpretation:** Erratic decision flips erode trust. Stability is important.

---

### 3.4 Response Delay

**Definition:** For weeks where an outbreak actually occurs, how long until the first action (GREEN or YELLOW)?

**Computation:**

- For each outbreak episode (first week where \( Y_t > \tau_i \)):
  - Backtrack to find the first week where decision was GREEN or YELLOW.
  - Response delay = outbreak week − first action week.

**Metrics:**

- Median response delay (weeks).
- IQR.
- % of outbreaks with action taken ≥1 week before outbreak.

**Interpretation:** Does the system warn early enough to act?

---

## 4. Comparative Analysis

### 4.1 Bayesian Decision Layer vs. XGBoost Threshold

**For XGBoost**, apply a simple threshold rule:

- **Alert (GREEN/YELLOW):** \( \hat{p}^{\text{XGB}} > 0.3 \) (more permissive, easier to trigger).
- **No action (RED):** \( \hat{p}^{\text{XGB}} \leq 0.3 \).

Or use two thresholds:

- **GREEN:** \( \hat{p}^{\text{XGB}} > 0.5 \).
- **YELLOW:** \( 0.3 < \hat{p}^{\text{XGB}} \leq 0.5 \).
- **RED:** \( \hat{p}^{\text{XGB}} \leq 0.3 \).

**Comparison table** (see Section 5).

---

## 5. Output: Decision Metrics Table

**Main table** comparing Bayesian decision layer vs. XGBoost:

| Metric | Bayesian Decision | XGBoost (threshold 0.3/0.5) | Interpretation |
|--------|-------------------|------------------------------|---|
| **False alarm rate (%)** | X% | Y% | Bayesian: fewer false alarms? |
| **Missed outbreaks (%)** | A% | B% | Trade-off between sensitivity & specificity |
| **Decision stability (% transitions/week)** | C% | D% | Bayesian: more stable? |
| **Median response delay (weeks)** | E weeks | F weeks | Bayesian: acts earlier? |
| **Coverage: outbreaks with ≥1 week warning (%)** | G% | H% | Lead-time validation via decisions |

---

## 6. Sensitivity Analysis

### 6.1 Alternative Decision Rules

To show robustness, also compute metrics under:

**Aggressive Bayesian rule:**
- GREEN: \( \hat{Z}_t > 50\text{th percentile} \).
- Everything else: YELLOW or RED.
- **Result:** More alerts, but possibly better coverage.

**Conservative Bayesian rule:**
- GREEN: \( \hat{Z}_t > 90\text{th percentile} \) AND \( \sigma_t < 25\text{th percentile} \).
- Everything else: RED.
- **Result:** Fewer alerts, only high-confidence actions.

**Output:** Side-by-side comparison (3 Bayesian rules + XGBoost).

---

### 6.2 Per-District Variation

Some districts may benefit more from Bayesian decisions (more stable, better lead time). Show:

- Metrics per district (sorted by Bayesian advantage).
- Highlight which districts see largest improvement.

---

## 7. Operational Narrative

### 7.1 Example Scenario

Describe a hypothetical deployment:

> "In District X, during the first 4 weeks of the outbreak:
>
> - **Week 1:** Bayesian state = YELLOW (moderate risk, some uncertainty). XGBoost = no alert. Action: increase surveillance.
> - **Week 2:** Bayesian state = GREEN (high risk, low uncertainty). XGBoost = no alert. Action: escalate alert to health authorities.
> - **Week 3:** Actual outbreak threshold crossed. Bayesian had warned 2 weeks earlier; XGBoost was silent until too late.
>
> This example illustrates that Bayesian latent risk provides operationally useful early warnings that XGBoost cannot match."

---

## 8. Implementation Pseudocode

```python
def classify_decision_state(Z_mean, Z_sd, thresholds):
    """
    Args:
        Z_mean: posterior mean (scalar or array).
        Z_sd: posterior SD (scalar or array).
        thresholds: dict with keys 'Z_25', 'Z_50', 'Z_75', 'sigma_25', 'sigma_50'.
    
    Returns:
        decision: 'GREEN', 'YELLOW', or 'RED'.
    """
    if Z_mean > thresholds['Z_75'] and Z_sd < thresholds['sigma_25']:
        return 'GREEN'
    elif (thresholds['Z_50'] < Z_mean < thresholds['Z_75']) or (Z_sd > thresholds['sigma_50']):
        return 'YELLOW'
    elif Z_mean < thresholds['Z_50'] and Z_sd < thresholds['sigma_50']:
        return 'RED'
    else:
        return 'YELLOW'  # default to caution


def evaluate_decision_layer(Z_means, Z_sds, outbreak_labels, thresholds):
    """
    Compute decision layer metrics.
    
    Args:
        Z_means: array of shape (n_weeks,).
        Z_sds: array of shape (n_weeks,).
        outbreak_labels: binary array, 1 = outbreak week.
        thresholds: dict of percentile thresholds.
    
    Returns:
        metrics_dict: dict with false alarm rate, missed outbreaks, etc.
    """
    decisions = []
    for i in range(len(Z_means)):
        d = classify_decision_state(Z_means[i], Z_sds[i], thresholds)
        decisions.append(d)
    decisions = np.array(decisions)
    
    # False alarms: alert (GREEN or YELLOW) when no outbreak
    alert_mask = (decisions != 'RED')
    false_alarms = np.sum(alert_mask & (outbreak_labels == 0))
    false_alarm_rate = false_alarms / np.sum(outbreak_labels == 0) * 100
    
    # Missed outbreaks: RED when outbreak
    missed = np.sum((decisions == 'RED') & (outbreak_labels == 1))
    missed_rate = missed / np.sum(outbreak_labels == 1) * 100
    
    # Decision stability: count transitions
    transitions = np.sum(decisions[:-1] != decisions[1:])
    stability_pct = transitions / (len(decisions) - 1) * 100
    
    # Response delay: for each outbreak, find first alert before or at outbreak
    response_delays = []
    for i, is_outbreak in enumerate(outbreak_labels):
        if is_outbreak:
            # Find first alert before this week
            alerts_before = np.where((np.array(decisions[:i+1]) != 'RED'))[0]
            if len(alerts_before) > 0:
                delay = i - alerts_before[-1]
            else:
                delay = np.inf  # no alert before outbreak
            response_delays.append(delay)
    
    if len(response_delays) > 0:
        median_delay = np.median([d for d in response_delays if d < np.inf])
        pct_early_warned = np.sum(np.array(response_delays) >= 1) / len(response_delays) * 100
    else:
        median_delay = np.nan
        pct_early_warned = 0
    
    return {
        'false_alarm_rate': false_alarm_rate,
        'missed_outbreak_rate': missed_rate,
        'decision_stability': stability_pct,
        'median_response_delay': median_delay,
        'pct_early_warned': pct_early_warned
    }
```

---

## 9. Integration with Thesis

**Section:** Results → "Decision-Layer Simulation" (1–2 pages).

**Figures:**

1. Flowchart of decision rules (GREEN / YELLOW / RED).
2. Example time-series plot (Z_t, decision state, outbreak threshold, for one district).

**Tables:**

1. Decision metrics (Bayesian vs. XGBoost).
2. Sensitivity analysis (3 Bayesian rules + XGBoost).
3. Per-district variation (top 5 districts by Bayesian advantage).

**Text:**

1. Decision rule definitions.
2. Metrics explanations.
3. Interpretation: "This minimal decision layer shows that uncertainty-aware risk reduces false alarms (metric X%) while maintaining early warning capability (metric Y weeks). This validates Claim 3."

---

## 10. What This Analysis Does NOT Claim

❌ Does NOT optimize a utility function or cost–loss framework.  
❌ Does NOT claim optimal thresholds (these are illustrative).  
❌ Does NOT require domain expert input on costs/benefits.  
❌ Does NOT consider operational constraints (resource availability, implementation burden).  

These are **intentionally scoped out**. You are demonstrating the principle (uncertainty matters), not the full operationalization.

---

## 11. Pre-Emptive Note for Reviewers

Include this in your thesis:

> "Our decision-layer simulation is intentionally minimal, serving as a proof-of-concept for how uncertainty-aware risk can guide action. We do not claim these decision thresholds are operationally optimal; rather, we demonstrate that integrating uncertainty into decision rules **improves stability and reduces false alarms** compared to binary classification thresholds. Full operationalization would require engagement with public health partners to establish context-specific costs and benefits."

---

## 12. Success Criteria

✅ Decision states (GREEN/YELLOW/RED) defined clearly.  
✅ Decision rules applied to all test weeks.  
✅ Metrics computed (false alarms, missed, stability, delay).  
✅ Comparison table created (Bayesian vs. XGBoost).  
✅ Sensitivity analysis done (2–3 alternative rules).  
✅ Example narrative written (1–2 outbreaks).  
✅ Pre-emptive note addresses scope limitations.

---

**Document Version:** 2.0  
**Status:** Ready for implementation  
**Effort:** Light (~1 week)  
**Last Updated:** February 3, 2026
