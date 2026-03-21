# Classifier Research Summary

## Candidate Classifiers
- rf_base: Literature RF (SD, CORR, NDAI)
- rf_full: Full RF (base + AE features)
- hgb_original: HGB (base engineered features)

## Model Assumption Checks
- Tree ensembles: nonparametric, no normality/linearity assumptions.
- Temporal stability checked via SMD between training and holdout.
- Largest |SMD|:
  - NDAI_DF_AF: |SMD|=1.147, train skew=2.481, test skew=1.226
  - DF: |SMD|=0.748, train skew=-0.676, test skew=-1.762
  - CORR: |SMD|=0.537, train skew=-0.639, test skew=0.099

## Fit Assessment (Temporal Holdout: O013490)
- hgb_full: ROC AUC=0.9976, F1=0.9780
- hgb_original: ROC AUC=0.9958, F1=0.9605
- rf_base: ROC AUC=0.9945, F1=0.9558
- rf_full: ROC AUC=0.9941, F1=0.9725

## Selected Classifier
- Selected: hgb_original (ROC AUC=0.9958, F1=0.9605).
- Baseline rf_base: ROC AUC=0.9945, F1=0.9558.