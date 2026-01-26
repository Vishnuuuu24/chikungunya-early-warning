# Versions Directory

This folder contains **frozen, reproducible snapshots** of working models.

## Structure

Each version folder contains:
```
versions/Vishnu-Version-Hist/vX.Y/
├── README.md           # What is this version?
├── code/               # Frozen source code
├── config/             # Exact config used
├── model/              # Trained model artifacts
├── results/            # Predictions, metrics, plots
└── run.sh              # Reproduction script
```

## Version Naming

| Version | Meaning |
|---------|---------|
| v1 | First working baseline |
| v1.1 | Bug fix or minor correction |
| v1.2 | Feature improvement (same model) |
| v2 | Major architectural change (Bayesian) |

## Rules

1. **Never modify** a frozen version
2. If changes needed → create new version
3. Each version is self-contained and reproducible
4. Write detailed README for each version

## Current Versions

*No versions created yet. First version will be created after baseline models work.*
