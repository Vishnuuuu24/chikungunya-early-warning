# ADDENDUM: Versioning Strategy & Updated Folder Structure

**For:** Chikungunya Early Warning System (India)  
**Version:** Added to 06_playbook.md (Section 6.14-6.15)  
**Date:** January 26, 2026

---

## ADD THIS TO 06_playbook.md (After Section 6.13, Before Final Checklist)

---

## 6.14 Versioning Strategy (Working vs Frozen)

This project uses a **dual-layer versioning approach** that keeps development clean while maintaining reproducible milestones.

### 6.14.1 The Two Layers

**Layer 1: Active Development (Always changing)**
```
src/                ← Your working code (messy, evolving)
experiments/        ← Scripts you run (try things)
config/             ← Active config (experiments)
notebooks/          ← Exploratory work
```

**Layer 2: Frozen Milestones (Never touched once created)**
```
versions/           ← Stable snapshots (v1, v1.1, v1.2, v2, …)
```

**Layer 3: Version Control (Git)**
```
.git/               ← Tracks everything (including broken experiments)
```

### 6.14.2 Why This Structure?

| Scenario | What Breaks Without It |
|----------|----------------------|
| Model A works. You refactor src/models/. Model A breaks. Faculty asks "send me v1." | You scramble to find which commit had v1. Time lost. |
| You try 5 feature ideas. 1 works. Other 4 are debris in your repo. | Clutter. Hard to reproduce. Hard to explain. |
| You want to compare "logistic baseline" vs "Bayesian" outputs side-by-side. | You must re-run old code, hope it still works. |
| You're writing thesis. Need to cite exact model that achieved result X. | Can't remember which config, which data, which code version. |

**With versions/ folder:**
- Each v1, v1.2, v2 is self-contained and reproducible
- Faculty asks "send v1" → you zip versions/v1/ and done
- Comparison is easy (side-by-side folder structure)
- Thesis writing is traceable (each result tied to a version)

---

### 6.14.3 Folder Structure for Versions

```
versions/
├── v1/
│   ├── README.md                     ← What is this version?
│   ├── code/                         ← Frozen code snapshot
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   └── evaluation/
│   ├── config/
│   │   └── config_v1.yaml
│   ├── model/                        ← Trained artifacts
│   │   ├── model_artifact.pkl
│   │   ├── feature_list.json
│   │   └── model_name.txt
│   ├── results/
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   └── plots/
│   └── run.sh                        ← How to reproduce
│
├── v1.1/                             ← Bug fix or minor improvement
│   └── [same structure]
│
├── v1.2/                             ← Better features
│   └── [same structure]
│
├── v2/                               ← Major conceptual change
│   └── [same structure]
│
└── README.md                         ← Version roadmap
```

**Key rule:** Once created, nothing in `versions/vX.Y/` is ever modified. If you change anything, you create a new version.

---

### 6.14.4 Version Naming Convention

Use **semantic meaning**, not timestamps.

| Version | Meaning | Example |
|---------|---------|---------|
| **v1** | First end-to-end pipeline working | Logistic regression + simple features, baseline result |
| **v1.1** | Bug fix or dataset correction | Fixed data leakage in CV, re-ran, same model |
| **v1.2** | Feature improvement | Added mechanistic climate features, better AUC |
| **v2** | Major architectural change | Switched to Bayesian hierarchical model |
| **v2.1** | Tuning or prior improvement | Bayesian with better hierarchical priors |
| **v3** | New capability | Added decision layer + cost–loss |

**Why this works:**
- v1 → v1.2 clearly shows "same model, improving"
- v1.2 → v2 clearly shows "different approach"
- Faculty immediately understands the trajectory

---

### 6.14.5 Inside Each Version Folder: README.md

This is the **most important file** in each version. Template:

```markdown
# Version v1.2 — Logistic + Mechanistic Features

## Summary
- **Model:** Logistic Regression
- **Lookback window (L):** 12 weeks
- **Prediction horizon (H):** 3 weeks
- **Features:** Case lags (1,2,4,8) + degree-days + rainfall persistence
- **CV:** Rolling-origin (train 2010–Y-1, test Y; folds 2017–2022)

## Why This Version Exists
Improved upon v1.1 by adding mechanistic climate features (degree-days).
These features capture biological mechanism (Aedes development), 
not just statistical correlation. Result: +0.03 AUC, +0.2 weeks lead time.

## Performance (Mean ± SD Across 6 Folds)
- AUC: 0.74 ± 0.05
- Lead time: 1.6 ± 0.8 weeks
- False alarm rate: 21% ± 5%
- Brier score: 0.27 ± 0.02
- Sensitivity: 82%
- Specificity: 68%

## Data & Config
- Data: panel_chikungunya_v01.parquet (2010–2022, 700+ districts)
- Config: config_v1.2.yaml
- Features: feature_list.json (34 features)

## How to Reproduce
1. Place data in `data/processed/`
2. Copy config: `config/config_v1.2.yaml`
3. Run: `cd versions/v1.2 && bash run.sh`
4. Results appear in `results/`

## Comparison to Previous
| Metric | v1 | v1.1 | v1.2 |
|--------|----|----|------|
| AUC | 0.70 | 0.70 | 0.74 |
| Lead time | 1.2 | 1.2 | 1.6 |
| FAR | 26% | 22% | 21% |

## Status
✔ Stable & reproducible  
✔ Ready for thesis  
✔ Candidate baseline (before Bayesian)

## Files
- `code/` — frozen code snapshot
- `model/` — trained logistic model + artifacts
- `config/` — exact config used
- `results/` — predictions + metrics + plots
- `run.sh` — reproduction script

## Git Commit / Tag
v1.2 (commit hash if recording)
```

This README is gold for:
- Writing thesis methods
- Explaining to faculty
- Remembering what you did 6 months later
- Comparing versions side-by-side

---

### 6.14.6 What Gets Copied Into versions/vX.Y/code/

**Copy ONLY what was actually used:**

```
✅ DO copy:
   src/data/load_epiclim.py
   src/features/case_features.py
   src/models/logistic.py
   src/evaluation/metrics.py

❌ DO NOT copy:
   src/experiments/   (scripts, not logic)
   src/models/deep_learning/  (if not used in this version)
   test files
   broken exploratory code
   notebooks (keep in root/notebooks/)
```

**Why?** Forces clarity. If you copy it, it was used. Clean.

---

### 6.14.7 When to Create a New Version

Create a new version when **any** of these happen:

| Trigger | Example | New Version |
|---------|---------|-------------|
| Performance improves significantly | AUC 0.70 → 0.75 | v1 → v1.1 |
| Feature set changes | Add mechanistic climate | v1 → v1.2 |
| Bug fix (re-run needed) | Data leakage detected | v1 → v1.1 |
| Model architecture changes | Logistic → Bayesian | v1 → v2 |
| Sharing with faculty / advisor | For feedback | v_X |
| Using in thesis write-up | Cite exact version | v_X |
| Major conceptual shift | Add decision layer | v2 → v3 |

**Do NOT create a version for:**
- Minor code cleanup (Git is enough)
- Experimental branches that don't work
- Temporary debugging

---

### 6.14.8 Version Lifecycle (Concrete Timeline)

```
Week 1-2: ACTIVE DEVELOPMENT
  └─ Work in src/, experiments/, config/
  └─ Code breaks frequently, that's OK
  └─ No versions/ yet

Week 2: FIRST MILESTONE
  └─ Data loading works
  └─ Quick baseline (logistic) achieves AUC 0.70
  └─ Create: versions/v1/
  └─ Freeze code, model, config, results
  └─ Write versions/v1/README.md

Week 3-4: MORE DEVELOPMENT
  └─ Continue in src/
  └─ Try new features, more baselines
  └─ Refactor code (versions/v1 unchanged)

Week 3-4 (again): SECOND MILESTONE
  └─ Better features → AUC 0.74
  └─ Create: versions/v1.2/
  └─ Freeze
  └─ Write README

Week 4-6: BAYESIAN DEVELOPMENT
  └─ Implement Bayesian model in src/
  └─ Lots of MCMC tuning, prior exploration
  └─ versions/v1 and v1.2 stay frozen

Week 6: MAJOR MILESTONE
  └─ Bayesian model works well, AUC 0.84
  └─ Create: versions/v2/
  └─ This is a jump from v1 (different model)
  └─ Write detailed README comparing to v1

Week 6-8: FINAL DEVELOPMENT
  └─ Tune Bayesian, add decision layer
  └─ versions/v2 stays frozen
  └─ Create versions/v2.1 if significant improvement

Week 8+: WRITE THESIS
  └─ All versions are ready to cite
  └─ Can reproduce any version
  └─ "Our final model (v2.1) achieved…"
```

---

### 6.14.9 Daily Workflow (Practical Steps)

**Every day, this is what you do:**

```
Morning:
  1. Open src/, experiments/
  2. Make changes, run experiments, break things
  3. Git commit: "WIP: trying new feature X"

When something works:
  1. Run full CV, compute metrics
  2. Result looks good → note version number
  3. Decide: is this worth freezing?

Freezing a version (when performance is good):
  1. Create folder: versions/v1.2/
  2. Copy: src/ → versions/v1.2/code/
  3. Copy: config_latest.yaml → versions/v1.2/config/config_v1.2.yaml
  4. Save: trained model → versions/v1.2/model/
  5. Save: results → versions/v1.2/results/
  6. Write: versions/v1.2/README.md (use template)
  7. Write: versions/v1.2/run.sh (reproduction script)
  8. Git commit: "Release: v1.2 (logistic + mechanistic features)"
  9. Git tag: git tag -a v1.2 -m "Logistic regression with degree-days"

Continue:
  1. Go back to src/
  2. Refactor, experiment, break things (versions/v1.2 safe)
  3. When next thing works → create v1.3 or v2
```

---

### 6.14.10 Example: Creating versions/v1.2/ (Step-by-Step)

You've been developing. Logistic regression achieves AUC 0.74 with mechanistic features. Time to freeze v1.2.

**Step 1: Create folder structure**
```bash
mkdir -p versions/v1.2/code
mkdir -p versions/v1.2/config
mkdir -p versions/v1.2/model
mkdir -p versions/v1.2/results
```

**Step 2: Copy code (selectively)**
```bash
cp -r src/data/ versions/v1.2/code/
cp -r src/features/ versions/v1.2/code/
cp -r src/models/baselines/ versions/v1.2/code/models/
cp -r src/evaluation/ versions/v1.2/code/
```

**Step 3: Copy config**
```bash
cp config/config_latest.yaml versions/v1.2/config/config_v1.2.yaml
```

**Step 4: Copy trained model**
```bash
cp results/logistic_model_v1.2.pkl versions/v1.2/model/
cp results/feature_list_v1.2.json versions/v1.2/model/
echo "logistic_regression" > versions/v1.2/model/model_name.txt
```

**Step 5: Copy results**
```bash
cp -r results/metrics_v1.2/ versions/v1.2/results/
cp results/predictions_v1.2.csv versions/v1.2/results/
cp -r results/plots_v1.2/ versions/v1.2/results/plots/
```

**Step 6: Write README**
(Use template from Section 6.14.5)

**Step 7: Write run.sh**
```bash
cat > versions/v1.2/run.sh << 'EOF'
#!/bin/bash
# Reproduction script for v1.2

# Load data
python code/data/load_epiclim.py

# Build features
python code/features/build_features.py --config config/config_v1.2.yaml

# Train & evaluate (rolling-origin CV)
python code/models/baselines/logistic.py --config config/config_v1.2.yaml

# Generate results
python code/evaluation/comparison.py --config config/config_v1.2.yaml

echo "✓ v1.2 reproduction complete"
EOF
chmod +x versions/v1.2/run.sh
```

**Step 8: Git commit & tag**
```bash
git add versions/v1.2/
git commit -m "Release: v1.2 (logistic + mechanistic features, AUC 0.74)"
git tag -a v1.2 -m "Logistic regression with degree-days & rainfall persistence"
git push origin v1.2
```

**Done!** versions/v1.2/ is now frozen and reproducible.

---

### 6.14.11 Comparing Versions

You can now easily compare:

```bash
# Side-by-side folders
ls versions/

# Read performance
cat versions/v1.2/README.md    # AUC 0.74
cat versions/v2/README.md      # AUC 0.84

# Re-run old version
cd versions/v1.2 && bash run.sh

# Check what changed
diff versions/v1.2/config/config_v1.2.yaml versions/v2/config/config_v2.yaml
```

---

### 6.14.12 Checklist: Creating a Version

When you decide a version is ready:

- [ ] Performance metrics finalized
- [ ] Code works end-to-end
- [ ] Config file is well-documented
- [ ] Model artifact saved (pkl, pt, nc, etc.)
- [ ] results/ folder contains: metrics.json, predictions.csv, plots/
- [ ] README.md written (use template from 6.14.5)
- [ ] run.sh written and tested (can reproduce from scratch)
- [ ] Git commit made: `"Release: vX.Y (description)"`
- [ ] Git tag created: `git tag vX.Y`
- [ ] Pushed to remote: `git push origin vX.Y`

---

## 6.15 Updated Final Checklist

- [ ] Config file exists and is valid YAML.
- [ ] Data paths in config point to correct locations.
- [ ] Random seed is set (for reproducibility).
- [ ] Feature engineering script runs without error.
- [ ] Quick baseline (logistic on fold 2017) achieves sensible AUC (0.60–0.80).
- [ ] CV splits don't have leakage (check: train_year < test_year).
- [ ] Metrics functions tested manually on dummy data.
- [ ] Results directory exists and is writable.
- [ ] Git repo initialized; `.gitignore` configured.
- [ ] Journal.md started with notes.
- [ ] **Understand versioning strategy (Section 6.14).**
- [ ] **Know when to create a new version.**
- [ ] **versions/ folder created and structure understood.**

---

