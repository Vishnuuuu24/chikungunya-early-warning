# Chikungunya Early Warning System (India)

**A Bayesian hierarchical state-space model for district-level outbreak prediction**

---

## 📋 Quick Start

1. **New here?** Start with `docs/START_HERE.md`
2. **Want to understand?** Read `docs/01_overview.md` 
3. **Ready to code?** Follow `docs/06_playbook.md`
4. **Git started:**
   ```bash
   git clone <repo>
   cd chikungunya_ews
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 📦 What's In This Repo

This is a **production-grade research project** with:
- Complete technical documentation (docs/)
- Clean code structure (src/)
- Reproducible experiments (experiments/)
- Frozen version milestones (versions/)
- Git-tracked development history

---

## 🗂️ Folder Structure

```
chikungunya_ews/
├── docs/                        📘 Design documents (source of truth)
│   ├── START_HERE.md           ← Start here
│   ├── 01_overview.md          ← Problem & vision
│   ├── 02_prd.md               ← Requirements
│   ├── 03_tdd.md               ← Technical design
│   ├── 04_data_spec.md         ← Data formats
│   ├── 05_experiments.md       ← Evaluation
│   └── 06_playbook.md          ← Implementation guide
│
├── src/                         🧠 Source code (clean, testable)
│   ├── data/                   ← BLOCK 1: Data loading
│   ├── features/               ← BLOCK 2: Feature engineering
│   ├── labels/                 ← Label creation (no leakage)
│   ├── models/                 ← BLOCK 3: All models
│   ├── evaluation/             ← BLOCK 4: Validation & metrics
│   ├── decision/               ← BLOCK 5: Decision layer
│   └── visualization/          ← Plotting utilities
│
├── config/                      ⚙️ Configuration (NO hardcoding)
│   ├── config_default.yaml
│   ├── config_baseline.yaml
│   └── config_bayesian.yaml
│
├── data/
│   ├── raw/                    ⛔ Never commit (download separately)
│   ├── interim/                ← Temporary / debug outputs
│   └── processed/              ← Canonical datasets (versioned)
│
├── experiments/                 🧪 Runnable scripts
│   ├── 00_sanity_check.py
│   ├── 01_build_panel.py
│   ├── 02_build_features.py
│   ├── 03_train_baselines.py
│   ├── 04_train_bayesian.py
│   ├── 05_run_evaluation.py
│   └── 06_generate_reports.py
│
├── notebooks/                   📓 Exploratory analysis
│   ├── 01_eda.ipynb
│   ├── 02_feature_checks.ipynb
│   └── 03_results_review.ipynb
│
├── stan_models/                 📊 Bayesian model source
│   └── hierarchical_statespace_v01.stan
│
├── versions/                    🧊 FROZEN SNAPSHOTS (key feature)
│   ├── v1/                     ← First working baseline
│   ├── v1.2/                   ← Improved features
│   ├── v2/                     ← Bayesian model
│   └── README.md               ← Version guide
│
├── results/                     📈 Experiment outputs
│   ├── predictions/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── .gitignore                   ← Git configuration
├── .env.example                 ← Local config template
├── journal.md                   ← Research log
└── LICENSE

```

---

## 🚀 Development Workflow

### Working Layers (3 levels)

**Layer 1: Active Development** (always changing)
- Write code in `src/`
- Run experiments in `experiments/`
- Break things, refactor, try ideas

**Layer 2: Frozen Milestones** (never touched)
- `versions/Vishnu-Version-Hist/v1.2/` = one working model + results
- Faculty asks "send v1.2" → zip that folder
- Easy side-by-side comparison

**Layer 3: Version Control** (Git)
- Tracks everything (broken experiments too)
- Tag releases: `git tag v1.2`
- Push to remote

**See** `docs/07_versioning_addendum.md` for detailed workflow.

---

## 🔄 Pipeline (5 Blocks)

```
BLOCK 1: DATA
  └─ Load EpiClim, Census, climate → panel

BLOCK 2: FEATURES  
  └─ Compute 35+ mechanistic + statistical features

BLOCK 3: MODELS
  ├─ Track A: 5 baselines (logistic, RF, XGB, Poisson, threshold)
  └─ Track B: Bayesian hierarchical state-space

BLOCK 4: EVALUATION
  └─ Temporal CV, metrics (AUC, lead time, FAR, Brier)

BLOCK 5: DECISION
  └─ Cost–loss → alert thresholds → actions
```

**Each block is:** isolated, testable, versioned, documented.

---

## 🎯 Quick Links

| I want to... | Go to... |
|---|---|
| Understand the project | `docs/01_overview.md` |
| Know what we're building | `docs/02_prd.md` |
| Understand the models | `docs/03_tdd.md` |
| Load and process data | `src/data/` |
| Run a baseline | `experiments/03_train_baselines.py` |
| Check results | `versions/Vishnu-Version-Hist/v1.2/results/` |
| Write thesis methods | `docs/` (copy as needed) |
| Fix a bug | `journal.md` + `docs/06_playbook.md` |
| Compare models | `ls versions/` + read READMEs |

---

## 📊 Key Concepts

### Mechanistic Model
- Follows the cause-effect chain: **climate → mosquitoes → risk → cases**
- Not just statistical correlation

### Latent Risk (Z_t)
- Hidden variable: true transmission intensity
- Inferred from observed climate + cases

### Hierarchical Bayesian
- Shared parameters across districts
- Partial pooling: borrow strength from neighbors

### Lead Time
- How many weeks before cases spike does the model warn?
- Target: ≥ 2 weeks (actionable)

### Temporal CV (Rolling-Origin)
- Train on past, test on future (never leakage)
- Train 2010–2016, test 2017; train 2010–2017, test 2018; etc.

---

## 🧪 Usage Examples

### Build the panel
```bash
python experiments/01_build_panel.py --config config/config_default.yaml
```

### Engineer features
```bash
python experiments/02_build_features.py --config config/config_default.yaml
```

### Train baselines (Track A)
```bash
python experiments/03_train_baselines.py --config config/config_baseline.yaml
```

### Train Bayesian model (Track B)
```bash
python experiments/04_train_bayesian.py --config config/config_bayesian.yaml
```

### Evaluate all models
```bash
python experiments/05_run_evaluation.py
```

### Generate report
```bash
python experiments/06_generate_reports.py
```

### Reproduce a frozen version
```bash
cd versions/Vishnu-Version-Hist/v1.2
bash run.sh
```

---

## 📚 Documentation Map

| Purpose | Document |
|---------|----------|
| **Get started (5 min)** | `docs/START_HERE.md` |
| **Understand problem** | `docs/01_overview.md` |
| **Know requirements** | `docs/02_prd.md` |
| **Understand models** | `docs/03_tdd.md` |
| **Data formats** | `docs/04_data_spec.md` |
| **Evaluation protocol** | `docs/05_experiments.md` |
| **How to code** | `docs/06_playbook.md` |
| **Versioning workflow** | `docs/07_versioning_addendum.md` |
| **Everything else** | `docs/README_DOCUMENTS.md` + `docs/INDEX.md` |

---

## 🔧 Environment Setup

**Python 3.9+**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Bayesian inference (optional)
pip install cmdstanpy

# For development
pip install pytest black flake8
```

---

## 📝 Configuration

Copy `.env.example` to `.env` and set local paths:

```bash
cp .env.example .env
# Edit: local data paths, API keys, etc.
```

All configs are YAML (in `config/`). No hardcoding.

---

## 🎓 For Faculty / Advisors

To review this project:

1. **5-minute overview:**  
   `docs/01_overview.md` + `docs/02_prd.md` Section 2.3

2. **Technical details:**  
   `docs/03_tdd.md` (equations, models, features)

3. **Evaluation protocol:**  
   `docs/05_experiments.md` (CV strategy, metrics)

4. **Reproducibility:**  
   `versions/Vishnu-Version-Hist/v1.2/run.sh` (one-command replay)

---

## 🔄 Version History

| Version | Model | AUC | Lead Time | Status |
|---------|-------|-----|-----------|--------|
| v1 | Logistic baseline | 0.70 | 1.2 weeks | ✓ Working |
| v1.2 | Logistic + features | 0.74 | 1.6 weeks | ✓ Stable |
| v2 | Bayesian hierarchical | 0.84 | 2.3 weeks | ✓ Production |
| v2.1 | Bayesian (tuned) | 0.85 | 2.5 weeks | ✓ Final |

**Explore:** `cd versions/vX.Y && cat README.md`

---

## 📖 Research Log

See `journal.md` for:
- Weekly progress notes
- Decisions made and why
- Bugs encountered and fixed
- Ideas for future work

---

## 🤝 Contributing

For team collaboration:
1. Create a feature branch: `git checkout -b feature/my-idea`
2. Work in `src/` and `experiments/`
3. Commit frequently: `git commit -m "WIP: description"`
4. When working: create a version: `versions/vX.Y/`
5. Push to remote

**Never edit files in `versions/`** — they're frozen.

---

## 📋 Checklist: Before You Start

- [ ] Read `docs/START_HERE.md`
- [ ] Read `docs/01_overview.md`
- [ ] Read `docs/02_prd.md`
- [ ] Environment setup complete
- [ ] `pip install -r requirements.txt` works
- [ ] `.env` configured (or not needed locally)
- [ ] Can run: `python experiments/00_sanity_check.py`

---

## 📞 Help

**Stuck?**
1. Search the docs (Ctrl+F)
2. Check `docs/06_playbook.md` Section 6.5 (debugging)
3. Review `journal.md` (what did I try before?)
4. Ask faculty with document reference

**Common questions answered in:**
- `docs/README_DOCUMENTS.md` (FAQ)
- `docs/INDEX.md` (full index)
- `docs/DOCUMENTS_MANIFEST.txt` (lookup by topic)

---

## 📄 License

(Add your license here)

---

## 👤 Author & Contact

**Project:** Chikungunya Early Warning System (India)  
**Started:** January 2026  
**Faculty Advisor:** [Your advisor]  
**Institution:** [Your institution]  

---

## 🎯 Project Goals

✅ Predict chikungunya outbreak risk 2–4 weeks in advance  
✅ Quantify uncertainty explicitly  
✅ Support district-level decision-making  
✅ Demonstrate Bayesian approach outperforms baselines  
✅ Provide reproducible, publishable code & results  

---

**Latest Update:** January 26, 2026  
**Status:** Ready for implementation ✓  
**Next:** Start with `docs/START_HERE.md`

