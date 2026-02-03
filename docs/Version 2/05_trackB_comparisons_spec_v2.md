# 05_trackB_comparisons_spec_v2.md

## Track B Internal Comparisons Specification

**Priority:** Optional (do only if time permits)  
**Purpose:** Validate Claim 4: "Modeling assumptions (mechanistic, temporal, hierarchical) measurably improve inference quality"  
**Effort:** Medium (~2–3 weeks if attempted; skip if schedule is tight)  
**Output:** Ablation comparison table + trajectory plots + interpretation

---

## 1. Conceptual Overview

### The Goal

Track B comparisons are **NOT** about building a model zoo.

They are about answering: **Do the design choices we made (mechanistic encoding, temporal dynamics, hierarchical pooling) actually help?**

### Why This Matters (Optional)

If you show that removing a key design choice degrades performance, you strengthen the thesis:

> "Our modeling assumptions are not arbitrary; they encode real biological/epidemiological constraints that improve inference."

This moves from "We chose mechanistic design because it made sense" to "We chose it because it provably works."

However, this is a **nice-to-have**, not essential. If time is tight, skip this entire workstream.

---

## 2. Controlled Ablations (Hypothesis-Driven)

Each comparison changes **exactly one thing** while holding everything else fixed.

---

### Comparison A: Mechanistic vs Non-Mechanistic

#### A.1 What Changes

**Mechanistic variant:**
- Latent state dynamics: \( Z_t \mid Z_{t-1} \sim \text{Normal}(\alpha Z_{t-1} + \beta T_t, \sigma_{\text{proc}}^2) \).
- Temperature \( T_t \) enters the evolution of latent risk.
- Rationale: vector population (mosquitoes) is temperature-dependent; disease transmission follows.

**Non-mechanistic variant:**
- Remove temperature from dynamics: \( Z_t \mid Z_{t-1} \sim \text{Normal}(\alpha Z_{t-1}, \sigma_{\text{proc}}^2) \).
- Pure statistical smoothing (local level model).
- No explicit encoding of biology.

#### A.2 What Stays Fixed

- Priors (same hyperparameters).
- Data (same cases, same folds).
- Likelihood (Negative Binomial observation model).
- Hierarchical structure (same partial pooling).

#### A.3 Metrics to Compare

For each variant, compute (on test folds):

1. **Lead time** (vs. outbreak threshold crossing).
   - Does mechanistic variant warn earlier?

2. **Credible interval width** (posterior SD).
   - Is mechanistic variant more confident (narrower intervals)?

3. **Trajectory smoothness** (variance of risk over time).
   - Does mechanistic variant produce smoother, more interpretable escalations?

#### A.4 Expected Finding

**Hypothesis:** Mechanistic variant gives earlier, smoother, more stable risk rise.

**Why:** Temperature variability is natural; encoding it as a driver (rather than noise) reduces spurious fluctuations.

#### A.5 Interpretation

If confirmed:

> "Encoding mechanistic knowledge (temperature dependence of transmission) measurably improves early warning capability. This justifies the added model complexity."

If not confirmed:

> "Statistical smoothing (non-mechanistic) performs similarly to mechanistic modeling. This suggests that for this dataset, explicit temperature encoding provides limited gain over simpler dynamics."

---

### Comparison B: Temporal vs Independent (State-Space Dynamics)

#### B.1 What Changes

**Temporal variant (your current model):**
- Latent dynamics: \( Z_t \mid Z_{t-1} \sim \text{Normal}(f(Z_{t-1}, T_t), \sigma_{\text{proc}}^2) \).
- Outbreak risk is a **process** with memory.

**Independent variant:**
- No temporal dependence: \( Z_t \sim p(Z_t | Y_t, \text{other weekly features}) \).
- Risk estimated independently for each week.
- (Conceptually: Bayesian logistic regression applied week-by-week.)

#### B.2 What Stays Fixed

- Priors.
- Data (same cases, same folds).
- Hierarchical structure.
- Observation model.

#### B.3 Metrics to Compare

1. **Lead time**.
   - Does temporal model warn earlier?

2. **Number of spurious alerts** (weeks where risk crosses threshold but no outbreak follows).
   - Does temporal model reduce noise?

3. **Credible interval stability** (how much does posterior SD fluctuate week-to-week?).
   - Lower variability = more stable.

#### B.4 Expected Finding

**Hypothesis:** Temporal dynamics reduce noise and provide earlier warning by capturing escalation patterns.

**Why:** Temporal correlation reduces week-to-week noise; explosive risk escalation is smoother under state-space model.

#### B.5 Interpretation

If confirmed:

> "Modeling outbreak risk as a dynamic process (vs. independent weekly estimates) reduces noise and improves early warning. This justifies state-space formulation."

If not confirmed:

> "Independent Bayesian inference performs similarly. This suggests that for this dataset, temporal autocorrelation provides limited gain."

---

### Comparison C: Hierarchical vs Non-Hierarchical (Partial Pooling)

#### C.1 What Changes

**Hierarchical variant (your current model):**
- District parameters share information: \( \alpha_i \sim p(\alpha | \mu_\alpha, \sigma_\alpha) \).
- Partial pooling: small districts "borrow strength" from large districts.

**Non-hierarchical variant:**
- Each district independent: \( \alpha_i \sim p(\alpha | \text{weak prior}) \).
- No information sharing across districts.

#### C.2 What Stays Fixed

- Priors (district-level priors are now weak/uninformed).
- Data, likelihood, temporal dynamics.

#### C.3 Metrics to Compare

1. **Performance on small districts** (few cases, low-data regime).
   - Does hierarchical variant have higher stability/calibration?

2. **Variance of parameter estimates** (\( \alpha, \beta \) across districts).
   - Hierarchical: lower variance (pulled to mean).
   - Non-hierarchical: higher variance (individual estimates).

3. **Fold-to-fold stability** (correlation of Z_t between folds).
   - Hierarchical should be more stable (less variability).

4. **Overall lead time**.
   - Is there a trade-off? (Hierarchical smoother, but possibly less district-specific?)

#### C.4 Expected Finding

**Hypothesis:** Hierarchical pooling improves inference in low-data districts without hurting large districts.

**Why:** Small districts have few outbreak episodes; pooling from similar districts provides regularization.

#### C.5 Interpretation

If confirmed:

> "Hierarchical modeling robustly improves inference across heterogeneous districts, especially where data is sparse. This justifies the added model complexity."

If not confirmed:

> "Non-hierarchical inference performs competitively. This suggests that for this dataset, partial pooling provides minimal gain, and districts can be modeled independently."

---

## 3. Implementation Approach

### 3.1 No Re-Estimation Needed

**Important:** You do NOT need to re-run MCMC for each variant.

Instead:

1. Use your **existing trained models** (mechanistic, temporal, hierarchical).
2. For each variant, simulate predictions by:
   - Removing temperature term (Comparison A).
   - Computing independent Z_t from weekly Bayesian logistic model (Comparison B).
   - Using non-hierarchical priors retroactively (Comparison C, if feasible; otherwise, skip).

This keeps computational cost low.

---

### 3.2 Alternative: Selective Re-Training

If you want more rigorous comparisons, re-train variants on training folds only. This is more expensive but cleaner.

**If time permits:** Do this for Comparison A (mechanistic vs non-mechanistic) as your primary ablation.

**If time is tight:** Stick to post-hoc comparisons (simulations from existing models).

---

## 4. Output: Comparison Table

**Main table** (one per comparison):

### Comparison A: Mechanistic vs Non-Mechanistic

| Metric | Mechanistic | Non-Mechanistic | Difference | Winner |
|--------|-------------|-----------------|-----------|--------|
| Median lead time (weeks) | X.X | Y.Y | X.X - Y.Y | Mechanistic? |
| Spurious alerts (% of weeks) | A% | B% | A% - B% | Lower is better |
| Avg credible interval width | C | D | C - D | Narrower (mechanistic) = better? |
| Trajectory smoothness (avg SD of dZ/dt) | E | F | E - F | Smoother (lower SD) = better |

**Interpretation:** If mechanistic wins on 3+ metrics, claim is validated.

---

### Comparison B: Temporal vs Independent

| Metric | Temporal | Independent | Difference | Winner |
|--------|----------|-------------|-----------|--------|
| Median lead time (weeks) | X.X | Y.Y | X.X - Y.Y | Temporal? |
| Spurious alerts (%) | A% | B% | A% - B% | Lower is better |
| Stability (fold-to-fold correlation) | C | D | C - D | Higher = better |

---

### Comparison C: Hierarchical vs Non-Hierarchical

| Metric | Hierarchical | Non-Hierarchical | Difference | Winner |
|--------|-------------|-----------------|-----------|--------|
| Lead time (small districts, median weeks) | X.X | Y.Y | X.X - Y.Y | Hierarchical? |
| Parameter variance (std of α) | A | B | A - B | Lower = better |
| Fold-to-fold stability (small districts) | C | D | C - D | Higher = better |

---

## 5. Visualization

### 5.1 Risk Trajectory Comparison (Per District)

**For one district**, plot:

- X-axis: time (weeks).
- Y-axis: Z_t.
- **Line 1:** Mechanistic variant.
- **Line 2:** Non-mechanistic variant.
- **Shaded region:** Outbreak threshold.

**Interpretation:** Visually shows differences in smoothness, timing of escalation.

---

### 5.2 Comparison Across Districts

**Small facet plot** (3–4 districts, one per facet):

- Each facet shows mechanistic vs non-mechanistic (or temporal vs independent).
- Sorted by size (small districts first).

**Interpretation:** Shows whether variant advantage is consistent or district-specific.

---

## 6. Implementation Pseudocode (Mechanistic vs Non-Mechanistic)

```python
def simulate_non_mechanistic_trajectory(Z_previous, noise_sd, n_weeks):
    """
    Simulate Z_t without temperature (non-mechanistic).
    
    Z_t = alpha * Z_{t-1} + noise
    """
    Z = np.zeros(n_weeks)
    Z[0] = Z_previous
    
    for t in range(1, n_weeks):
        Z[t] = 0.95 * Z[t-1] + np.random.normal(0, noise_sd)
    
    return Z


def compare_mechanistic_vs_nonmechanistic(bayesian_posterior, temperature, 
                                         outbreak_labels, thresholds):
    """
    Compare mechanistic (current model) vs non-mechanistic trajectories.
    """
    n_weeks = len(temperature)
    
    # Mechanistic: use full posterior (already computed)
    Z_mechanistic = bayesian_posterior.mean(axis=1)  # posterior mean
    
    # Non-mechanistic: simulate AR(1) without temp
    Z_nonmech = simulate_non_mechanistic_trajectory(
        Z_mechanistic[0], 
        noise_sd=bayesian_posterior.std(axis=1).mean(),
        n_weeks=n_weeks
    )
    
    # Compute lead times
    lt_mech = compute_lead_time(Z_mechanistic, outbreak_labels, thresholds['outbreak'])
    lt_nonmech = compute_lead_time(Z_nonmech, outbreak_labels, thresholds['outbreak'])
    
    # Compare
    results = {
        'mechanistic': {
            'median_lead_time': np.median(lt_mech),
            'spurious_alerts': spurious_alert_rate(Z_mechanistic, outbreak_labels)
        },
        'non_mechanistic': {
            'median_lead_time': np.median(lt_nonmech),
            'spurious_alerts': spurious_alert_rate(Z_nonmech, outbreak_labels)
        }
    }
    
    return results
```

---

## 7. Priority & Sequencing

**If you have >4 weeks remaining:**  
Do Comparisons A & B. Comparison C is optional.

**If you have 2–3 weeks:**  
Do Comparison A only (mechanistic vs non-mechanistic).

**If you have <2 weeks:**  
Skip this entire workstream. Your thesis is already strong without it.

---

## 8. Pre-Emptive Note for Reviewers

Include this in your thesis if you attempt Comparisons:

> "To validate that our modeling assumptions (mechanistic encoding, temporal dynamics, hierarchical pooling) provide measurable benefit, we conducted controlled ablations. For each assumption, we compared the primary model against a simplified variant while holding all other components fixed. These comparisons are illustrative rather than exhaustive; a full architecture search is beyond the scope of this work."

---

## 9. What NOT to Do

❌ Do NOT build 10+ model variants (model zoo).  
❌ Do NOT optimize hyperparameters across variants (leakage risk).  
❌ Do NOT present all variants as equally valid (there is one primary model).  
❌ Do NOT report results on validation set without proper CV.

---

## 10. Integration with Thesis

**Section:** Results → "Model Assumptions Validation" (1–2 pages, optional).

**Figures:**

1. Comparison table (mechanistic vs non-mechanistic, if done).
2. Risk trajectory plots (2–3 districts, variant comparison).

**Text:**

1. Rationale for each comparison.
2. Expected hypotheses.
3. Findings (validated / not validated).
4. Interpretation: which assumptions matter most.

---

## 11. Success Criteria (If Attempted)

✅ Comparison A (mechanistic vs non-mechanistic) complete.  
✅ Metrics computed consistently (lead time, spurious alerts, stability).  
✅ Comparison table generated.  
✅ Trajectory plots show visual differences.  
✅ Interpretation aligns with hypothesis or gracefully acknowledges null finding.  

---

## 12. Decision Point: Should You Do This?

**Do this if:**
- You have ≥3 weeks.
- You want to make the "modeling assumptions matter" argument explicit.
- Your advisor suggests it.

**Skip this if:**
- You have <3 weeks.
- Your thesis deadline is firm.
- You have done enough to satisfy your committee.

---

**Document Version:** 2.0  
**Status:** Optional  
**Effort:** Medium (~2–3 weeks if attempted)  
**Last Updated:** February 3, 2026
