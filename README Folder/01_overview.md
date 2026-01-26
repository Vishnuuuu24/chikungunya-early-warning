# 1. PROJECT OVERVIEW & VISION

**Project Name:** Chikungunya Early Warning & Decision System (India)

**Version:** 0.1

**Last Updated:** January 2026

---

## 1.1 Problem Statement

Chikungunya is a mosquito-borne viral disease that causes periodic outbreaks in India, with significant morbidity. Current surveillance systems (IDSP, EpiClim) detect outbreaks *after* they have already begun—i.e., using cases as the signal.

**The gap:** By the time case counts rise, community transmission is already well-established. Public health authorities need *early warning* (2–4 weeks ahead) to deploy vector control and awareness campaigns before cases spike.

**The challenge:** Early warning requires integrating multiple noisy data sources (case counts, climate, weak signals) and translating probabilistic estimates into actionable decisions under uncertainty.

---

## 1.2 Vision & Goal

**Vision:**  
Build an operational early-warning and decision-support system that:
- Estimates weekly chikungunya outbreak risk per district in India.
- Quantifies uncertainty explicitly (credible intervals, confidence).
- Recommends graded public health actions (surveillance, vector control, emergency response).
- Is transparent, mechanistic, and transferable to other arboviruses (dengue, Zika).

**Primary Goal:**  
Achieve 2–4 weeks lead time in detecting outbreak transitions at district level, with acceptably low false alarm rates, using hierarchical Bayesian fusion of case + climate data.

**Secondary Goal:**  
Demonstrate that latent state-space modeling outperforms standard classification approaches for this problem, and provide a template for global early-warning systems.

---

## 1.3 Who This Is For

**Primary Users:**
- State surveillance officers (IDSP, state health departments).
- District epidemiologists and vector control teams.
- National Centre for Disease Control (NCDC) early warning cell (if it exists).

**Secondary Users:**
- Public health researchers and modelers.
- International teams working on arbovirus early warning (dengue, Zika, Mpox).

**Stakeholders:**
- Ministry of Health & Family Welfare (MoHFW), India.
- WHO country office, India.

---

## 1.4 System Architecture at a Glance (5 Blocks)

```
┌──────────────────────────────────────────────────────────────┐
│ BLOCK 1: DATA ACQUISITION                                    │
│ └─ EpiClim (cases, climate) + IDSP (validation)             │
│    └─ Population census, district shapefile                  │
└──────────────┬───────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────┐
│ BLOCK 2: FEATURE ENGINEERING                                 │
│ └─ Mechanistic climate features (lags, degree-days, etc.)   │
│    └─ Early-warning statistics (variance, autocorr., trend) │
│    └─ Optional spatial & weak signals                        │
└──────────────┬───────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────┐
│ BLOCK 3: LATENT RISK INFERENCE (CORE MODEL)                  │
│ └─ Track A: Supervised Baselines (5 models)                 │
│    └─ Track B: Bayesian Hierarchical State-Space (1 main)   │
│    └─ Compare on EWS metrics                                │
└──────────────┬───────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────┐
│ BLOCK 4: VALIDATION & COMPARISON                             │
│ └─ Rolling-origin temporal CV (no data leakage)             │
│    └─ Metrics: AUC, lead time, false alarm rate, Brier score│
└──────────────┬───────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────┐
│ BLOCK 5: DECISION LAYER                                      │
│ └─ Cost–loss analysis → alert thresholds                     │
│    └─ Map risk to action tiers (surveillance / vector ctrl) │
└──────────────────────────────────────────────────────────────┘
```

---

## 1.5 Key Concepts Explained Simply

### Mechanistic Model
- **What it is:** A model that follows the cause-effect chain explicitly.
- **For chikungunya:** Warm + wet weather → good mosquito conditions → more mosquitoes → more infections → cases reported weeks later.
- **Contrast:** Standard models just say "when numbers looked like this, an outbreak followed," without explaining why.

### Latent Risk (Hidden Variable)
- **What it is:** An internal "dial" the model maintains for how close a district is to an outbreak, which we never directly observe.
- **Why:** You see case counts and climate, but the true *transmission intensity* (how contagious the situation is) is hidden and inferred by the model.
- **Example:** District X might have low cases today but high latent risk (because conditions are perfect for mosquitoes); the model predicts cases will rise in 2–3 weeks.

### Lead Time
- **What it is:** How many weeks in advance the model detects an outbreak before case counts actually spike.
- **Target:** We aim for 2–4 weeks; more is better (gives time to mobilize).

### Early Warning System (EWS)
- **What it is:** A system that detects outbreaks early (before they're fully underway).
- **Components:** Features that signal instability (increasing variance, shifts in trend), coupled with a probabilistic model to make predictions.

---

## 1.6 Why This Matters (For India)

1. **Burden of disease:**  
   Chikungunya affects millions in India every few years; outbreaks cause long-term joint pain, healthcare overload.

2. **Vector control capacity:**  
   Mosquito control teams have limited resources. Early warning lets them concentrate efforts where they'll have the most impact.

3. **Scalability:**  
   EpiClim and IMD climate data exist and are publicly available. This system can be deployed nationwide.

4. **Transferability:**  
   Same approach (mechanistic Bayesian fusion) applies to dengue, Zika, and beyond.

---

## 1.7 Success Criteria (High-Level)

| Criterion | Target | Rationale |
|-----------|--------|-----------|
| **Lead time** | ≥ 2 weeks (ideally 3–4) | Enough time for vector control deployment |
| **False alarm rate** | < 20% | Don't cry wolf; maintain trust |
| **Sensitivity** | ≥ 80% | Catch most outbreaks; low miss rate |
| **Transferability** | Works on Brazil data | Shows generalization beyond India |
| **Interpretability** | Feature importance + mechanistic explanation | Public health stakeholders can understand why |

---

## 1.8 Scope & Out-of-Scope

### In Scope
- Weekly district-level outbreak probability estimation.
- Quantified uncertainty (credible intervals).
- Decision rules for action levels (rules of thumb, cost–loss framework).
- Model comparison and selection (which method works best?).
- Internal technical documentation (layer details, hyperparameters, etc.).

### Out of Scope
- Real-time model deployment (web APIs, dashboards, etc.) — that's Phase 2.
- Detailed logistics of vector control operations.
- Economic analysis of intervention costs (cost–loss framework is simplified).
- Forecasting beyond 4 weeks ahead.
- Multi-pathogen co-circulation dynamics (dengue + chikungunya simultaneously).

---

## 1.9 How This Document Fits In

This **Overview** anchors the big picture:
- **What problem** are we solving?
- **Who cares?**
- **What does success look like?**

The next documents get more detailed:
- **PRD** → What the system must do (functional requirements).
- **TDD** → How we'll build it (models, features, architecture).
- **Data Spec** → What the data looks like.
- **Experiments** → How we'll validate and compare.
- **Playbook** → How to actually run it in VS Code.

---

## 1.10 References & Resources

- **EpiClim:** https://www.epiclim.org/ (India surveillance + climate).
- **IDSP:** Integrated Disease Surveillance Program (case reports).
- **IMD:** India Meteorological Department (climate data).
- **Proposal:** "Modelling and Prediction of Infectious Disease Dynamics" (special issue response).
- **Spatial data:** Datameet India district shapefiles.
- **External validation:** Brazil Zenodo chikungunya/dengue/Zika + Mosqlimate.

---

**Next Step:** Read `02_prd.md` for detailed functional and non-functional requirements.
