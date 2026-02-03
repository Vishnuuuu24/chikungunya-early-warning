# Chikungunya Early Warning System - Research Journal

## Project Start: January 26, 2026

---

### Week 1 (Jan 26, 2026)

**Setup:**
- Created complete project folder structure
- Downloaded India Census 2011 district population data (640 districts)
- Documented datasets in `data/raw/README.md`

**Data Sources:**
1. **EpiClim** (`Epiclim_Final_data.csv`)
   - 8,985 total rows, 731 Chikungunya-specific
   - Years: 2009–2022
   - Climate: precipitation, LAI, temperature (Kelvin)
   
2. **Census 2011** (`india_census_2011_district.csv`)
   - 640 districts, 118 demographic columns
   - Source: GitHub (nishusharma1608/India-Census-2011-Analysis)
   - URL: https://raw.githubusercontent.com/nishusharma1608/India-Census-2011-Analysis/master/india-districts-census-2011.csv

**Decisions Made:**
- [x] Use strict "Chikungunya" filter only (not variants)
- [x] Convert temperature from Kelvin to Celsius
- [x] Use fuzzy matching for district names between datasets
- [x] Apply 2% annual growth rate for 2011→target year population projection

**Next Steps:**
- [ ] Test data loading pipeline
- [ ] Build panel dataset
- [ ] Begin feature engineering

---

## Notes

*Add research notes, decisions, and observations here as the project progresses.*
## 2026-01-26 — Phase 4.1 Bayesian Prototype Complete

- v1: Baseline models (RF, Logistic, Threshold)
- v1.1: Baselines + XGBoost (best AUC = 0.759)
- Phase 4.1: Bayesian hierarchical state-space model
  - Stan implementation
  - Single-fold test (fold_2019)
  - Model compiles, samples, PPC looks good
  - Diagnostics not yet stable (expected)
- Next step: Freeze v2_proto, then Phase 4.2 (stabilization)

