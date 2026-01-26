# 📌 VERSIONING + STRUCTURE GUIDE (Final)

**Chikungunya Early Warning System**  
**Version + Folder Structure Integration**  
**January 26, 2026**

---

## ✅ What You Now Have (Updated)

### Original 11 Documents
- ✓ 01_overview.md
- ✓ 02_prd.md
- ✓ 03_tdd.md
- ✓ 04_data_spec.md
- ✓ 05_experiments.md
- ✓ 06_playbook.md
- ✓ README_DOCUMENTS.md
- ✓ START_HERE.md
- ✓ QUICK_SUMMARY.txt
- ✓ INDEX.md
- ✓ DOCUMENTS_MANIFEST.txt

### NEW (3 files)
- ✅ **07_versioning_addendum.md** — Full versioning strategy (add to 06_playbook.md)
- ✅ **README_PRODUCTION.md** — Root project README (replace generic README.md)
- ✅ **VERSIONING_FOLDER_STRUCTURE.md** — This guide

---

## 🗂️ Complete Folder Structure (With Versioning)

```
chikungunya_ews/
│
├── docs/                        📘 Design documents
│   ├── 01_overview.md
│   ├── 02_prd.md
│   ├── 03_tdd.md
│   ├── 04_data_spec.md
│   ├── 05_experiments.md
│   ├── 06_playbook.md           ← (Add Section 6.14-6.15 from addendum)
│   ├── 07_versioning_addendum.md ← (NEW - versioning guide)
│   ├── README_DOCUMENTS.md
│   ├── START_HERE.md
│   ├── etc...
│
├── src/                         🧠 Active development (always changing)
│   ├── __init__.py
│   ├── common/
│   ├── data/
│   ├── features/
│   ├── labels/
│   ├── models/
│   ├── evaluation/
│   ├── decision/
│   └── visualization/
│
├── config/                      ⚙️ Configuration (no hardcoding)
│   ├── config_default.yaml
│   ├── config_baseline.yaml
│   └── config_bayesian.yaml
│
├── data/
│   ├── raw/                     ⛔ Never commit
│   ├── interim/
│   └── processed/
│
├── experiments/                 🧪 Orchestration scripts
│   ├── 00_sanity_check.py
│   ├── 01_build_panel.py
│   ├── 02_build_features.py
│   ├── 03_train_baselines.py
│   ├── 04_train_bayesian.py
│   ├── 05_run_evaluation.py
│   └── 06_generate_reports.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_checks.ipynb
│   └── 03_results_review.ipynb
│
├── stan_models/
│   └── hierarchical_statespace_v01.stan
│
├── versions/                    🧊 FROZEN SNAPSHOTS (Your idea!)
│   ├── v1/
│   │   ├── README.md           ← What is this version?
│   │   ├── code/               ← Frozen source
│   │   │   ├── data/
│   │   │   ├── features/
│   │   │   ├── models/
│   │   │   └── evaluation/
│   │   ├── config/
│   │   │   └── config_v1.yaml
│   │   ├── model/              ← Trained artifacts
│   │   │   ├── model_artifact.pkl
│   │   │   ├── feature_list.json
│   │   │   └── model_name.txt
│   │   ├── results/
│   │   │   ├── metrics.json
│   │   │   ├── predictions.csv
│   │   │   └── plots/
│   │   └── run.sh              ← Reproduction script
│   │
│   ├── v1.1/
│   │   └── [same structure]
│   │
│   ├── v1.2/
│   │   └── [same structure]
│   │
│   ├── v2/
│   │   └── [same structure]
│   │
│   └── README.md               ← Version roadmap
│
├── results/                     📈 Current experiment outputs
│   ├── predictions/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── .gitignore                   ← Configured for this structure
├── requirements.txt
├── .env.example
├── README.md                    ← (Use README_PRODUCTION.md content)
├── journal.md                   ← Research log
└── LICENSE
```

---

## 🧠 Three Layers Explained

### Layer 1: Active Development (`src/`, `experiments/`, `config/`)
- Where you work every day
- Code is messy, evolving, breaking is OK
- Refactor freely
- Git commits: frequent, include broken states

### Layer 2: Frozen Milestones (`versions/`)
- Snapshots of working code + trained models
- Never modified once created
- Self-contained (can be zipped and sent to faculty)
- Tied to git tags for traceability

### Layer 3: Version Control (`git/`)
- Tracks all development (including broken experiments)
- Branches for features
- Tags for versions
- Remote for backup

**Example workflow:**
```
You work in src/ (messy)
  → git commit "WIP: trying new features"
  → git commit "Fixed bug in feature_engineering"
  → Run experiments, get good results
  → Freeze as versions/v1.2/
  → git commit "Release: v1.2"
  → git tag v1.2
  → Continue in src/ (still messy)
  → versions/v1.2/ stays frozen ✓
```

---

## 📝 Version Naming (Semantics Matter)

| Pattern | Meaning | When |
|---------|---------|------|
| v1.0 | First end-to-end working pipeline | Week 2-3: data + baseline works |
| v1.1 | Bug fix or minor correction | Data leakage found & fixed |
| v1.2 | Feature improvement (same model) | Better features, higher AUC |
| v2.0 | Major architectural change | Switch from logistic to Bayesian |
| v2.1 | Tuning/optimization (same model) | Better priors, higher AUC |
| v3.0 | New capability added | Decision layer + cost–loss |

**Why?** Faculty immediately sees trajectory: v1 → v1.2 (improving), v1.2 → v2 (different), v2 → v2.1 (tuning).

---

## 🔄 When to Create a Version

### ✅ YES, Create a Version If:

```
□ Performance improvement significant (ΔAUc ≥ 0.03)
□ Feature set changed conceptually
□ Bug fix found (re-run needed)
□ Model architecture changed
□ Sharing with faculty for feedback
□ Using in thesis write-up
□ Any code you might want to reproduce later
```

### ❌ NO, Don't Create a Version If:

```
□ Minor code cleanup (git commit is enough)
□ Experimental idea that didn't work
□ Temporary debugging (will delete code)
□ Random seed tuning (not meaningful)
```

---

## 📄 Inside versions/vX.Y/ : README Template

**This file is ESSENTIAL.** It's what makes versions reproducible.

```markdown
# Version v1.2 — Logistic + Mechanistic Features

## Summary (3 lines)
- Model: Logistic Regression
- Features: Case lags + degree-days + rainfall persistence
- CV: Rolling-origin (2017–2022)

## Why This Version Exists (2–3 sentences)
Improved v1.1 by adding mechanistic climate features capturing Aedes 
development biology. Result: +0.03 AUC, +0.2 weeks lead time.

## Performance (Copy from eval results)
- AUC: 0.74 ± 0.05
- Lead time: 1.6 ± 0.8 weeks
- False alarm rate: 21% ± 5%
- Brier score: 0.27 ± 0.02

## Data & Config
- Data: panel_chikungunya_v01.parquet
- Config: config_v1.2.yaml
- Features: 34 (list in feature_list.json)

## Reproduce
cd versions/v1.2 && bash run.sh

## Status
✔ Stable & reproducible
✔ Ready for thesis

## Files
- code/  → frozen code used
- model/ → trained model artifact
- config/ → exact config
- results/ → predictions, metrics, plots
```

**This README should be copy-paste-able into thesis methods.**

---

## 🎯 Daily Workflow (Practical)

### Morning: Active Development
```bash
cd chikungunya_ews
git checkout main                 # or your feature branch
git pull

# Work in src/
vim src/models/logistic.py
vim src/features/engineering.py

# Try locally
python experiments/03_train_baselines.py

# Git track as you go
git add src/models/logistic.py
git commit -m "WIP: improved logistic model"
```

### When Something Works
```bash
# Run full CV
python experiments/05_run_evaluation.py

# Check results
cat results/metrics_latest.json

# Looks good? Decide: freeze as new version?
# YES → proceed to "Freezing a Version" below
# NO → continue dev, commit as WIP
```

### Freezing a Version
```bash
# Create structure
mkdir -p versions/v1.2/{code,config,model,results}

# Copy code
cp -r src/data src/features src/models/baselines src/evaluation versions/v1.2/code/

# Copy config
cp config/config_latest.yaml versions/v1.2/config/config_v1.2.yaml

# Copy trained model
cp results/logistic_model_v1.2.pkl versions/v1.2/model/
cp results/feature_list_v1.2.json versions/v1.2/model/

# Copy results
cp results/metrics_v1.2.json versions/v1.2/results/
cp -r results/plots_v1.2/ versions/v1.2/results/plots/

# Write README (use template above)
vim versions/v1.2/README.md

# Write reproduction script
cat > versions/v1.2/run.sh << 'EOF'
#!/bin/bash
python code/data/load_epiclim.py
python code/features/build_features.py --config config/config_v1.2.yaml
python code/models/baselines/logistic.py --config config/config_v1.2.yaml
python code/evaluation/comparison.py
EOF
chmod +x versions/v1.2/run.sh

# Git track the version
git add versions/v1.2/
git commit -m "Release: v1.2 (logistic + mechanistic, AUC 0.74)"
git tag -a v1.2 -m "Logistic regression with degree-days"

# Continue development
git checkout main (or next feature branch)
vim src/...  # Keep working
```

---

## 📊 Version Comparison

**Easy side-by-side:**

```bash
# List all
ls -la versions/

# Compare README
cat versions/v1.2/README.md
cat versions/v2/README.md

# Compare metrics
diff versions/v1.2/results/metrics.json versions/v2/results/metrics.json

# Run old version
cd versions/v1.2 && bash run.sh

# Check differences
diff versions/v1.2/config/config_v1.2.yaml versions/v2/config/config_v2.yaml
```

---

## 🔗 How Versioning + Docs Fit Together

**Your documents explain WHY.** Your versions show WHAT.

```
Thesis Methods Section:
  "Our approach uses mechanistic features..." (cite 03_tdd.md)
  "We evaluate via rolling-origin CV..." (cite 05_experiments.md)
  "Our baseline achieves AUC 0.74..." 
    → Link to: versions/v1.2/README.md (proof!)
  "Our Bayesian model achieves AUC 0.84..."
    → Link to: versions/v2/README.md (proof!)
```

**Every result is tied to a frozen, reproducible version.**

---

## ✅ Checklist: Before First Version

- [ ] Understand the 3 layers (active/frozen/git)
- [ ] Know version naming (v1.0, v1.1, v1.2, v2.0, etc.)
- [ ] Read Section 6.14 of 06_playbook.md (versioning workflow)
- [ ] Know when to create a version (section above)
- [ ] Have README template memorized (or bookmarked)
- [ ] Know `versions/vX.Y/` structure
- [ ] Understand `run.sh` reproduction script
- [ ] Git comfortable with tagging

---

## 🚀 Timeline: When Versions Happen

```
Week 1: Active dev
  └─ No versions yet

Week 2: First milestone
  └─ Create versions/v1/
  └─ Data loading + quick baseline works

Week 3-4: More dev
  └─ Try features
  └─ Create versions/v1.2/ when better

Week 4-6: Bayesian dev
  └─ Work in src/
  └─ versions/v1 and v1.2 stay frozen

Week 6: Major milestone
  └─ Create versions/v2/
  └─ This is the jump (different model)

Week 8+: Thesis writing
  └─ All versions are ready to reference
  └─ Reproducible + traceable
```

---

## 🎓 For Your Faculty

When they ask "Can you send me your model?":
```bash
cd chikungunya_ews
tar -czf versions/v2.tar.gz versions/v2/
# Send versions/v2.tar.gz

# They can do:
tar -xzf versions/v2.tar.gz
cd versions/v2
bash run.sh
# Boom — reproduced!
```

---

## 📝 Final Integration Checklist

### Update 06_playbook.md
- [ ] Add Section 6.14 (paste from 07_versioning_addendum.md)
- [ ] Add Section 6.15 (updated checklist)

### Update README.md (root)
- [ ] Replace with README_PRODUCTION.md content
- [ ] Ensure mentions `versions/` folder

### Create folder structure
- [ ] Create: `versions/` folder (empty, with README.md)
- [ ] Create: `src/` folder structure
- [ ] Create: `experiments/`, `config/`, `notebooks/`, `stan_models/`

### Git configuration
- [ ] `.gitignore` includes: `data/raw/`, `data/processed/`, `results/`, `.env`
- [ ] But does NOT ignore: `versions/` (we want to track frozen versions)

### Documentation
- [ ] Save all 14 files (docs + new ones) to `docs/` folder
- [ ] Print or bookmark `docs/07_versioning_addendum.md`

---

## 🎯 Bottom Line

You now have:

✅ **Active development** in `src/` (messy, evolving)  
✅ **Frozen milestones** in `versions/` (reproducible snapshots)  
✅ **Git tracking** for full history  
✅ **Documentation** explaining everything  
✅ **Folder structure** that won't need refactoring  

This is how professional research projects are run.

**You're ready. Let's build! 🚀**

---

**Questions?** See `docs/06_playbook.md` (after adding Section 6.14).

