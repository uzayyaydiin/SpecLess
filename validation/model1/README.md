# SpecLess Model 1 Validation

This directory contains validation experiments associated with SpecLess Model 1.

## Experiments

### KiDS + VIKING internal validation

A morphology classifier was independently trained and evaluated using KiDS DR5 + VIKING photometry.

Input bands:

u, g, r, i, Z, Y, J, H, Ks

The complete feature space contains 45 features: 9 magnitudes and 36 colour indices.

Dataset size:

- Total: 16,342 galaxies
- Training: 13,073 galaxies
- Independent test: 3,269 galaxies

Random Forest achieved the highest balanced accuracy on the independent test set:

- Accuracy: 0.932701
- Balanced accuracy: 0.931321
- Macro F1: 0.928194
- ROC-AUC: 0.976626

XGBoost achieved:

- Accuracy: 0.928418
- Balanced accuracy: 0.927411
- Macro F1: 0.923724
- ROC-AUC: 0.977526

The lightweight XGBoost model is included in this repository.
The Random Forest model is retained as a separate research artefact because of its larger binary size.

### SDSS to KiDS transfer

Cross-survey transfer uses 10 common photometric features:

u, g, r, i, u_g, u_r, u_i, g_r, g_i, r_i

XGBoost achieved:

- Accuracy: 0.946548
- Balanced accuracy: 0.945002
- Macro F1: 0.939456
- ROC-AUC: 0.984414

### KiDS to SDSS transfer

The reverse transfer experiment uses the same 10-feature common photometric space.

XGBoost achieved:

- Accuracy: 0.886526
- Balanced accuracy: 0.891594
- Macro F1: 0.880202
- ROC-AUC: 0.947709

This directory stores validation metadata, compact result tables, feature definitions, and lightweight trained models. Large source catalogues are not distributed directly with the Python package.
