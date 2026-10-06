# SpecLess Model 2: Photometric Activity Classification

Model 2 is the activity classification component of SpecLess.

It performs four class classification between:

- Seyfert
- Starburst
- Star forming
- LINER

The released classifier operates on photometric catalogue measurements and does not require spectroscopy at inference time.

## 1. Class Definition

The original research catalogue used the following class codes:

`3 = Seyfert`

`4 = Starburst`

`5 = Star forming`

`6 = LINER`

The released software maps these labels internally to:

`0 = SEYFERT`

`1 = STARBURST`

`2 = STARFORMING`

`3 = LINER`

The mapping between the research catalogue and the software output is therefore explicit and fixed.

## 2. Reference Data

The spectroscopically labelled parent sample contains 133,886 galaxies.

| Class | Number |
|---|---:|
| Star forming | 60,000 |
| Starburst | 60,000 |
| Seyfert | 7,264 |
| LINER | 6,622 |
| Total | 133,886 |

The final SDSS and WISE photometric sample contains 84,652 galaxies.

| Class | Number |
|---|---:|
| Star forming | 37,518 |
| Starburst | 36,924 |
| Seyfert | 5,990 |
| LINER | 4,220 |
| Total | 84,652 |

Spectroscopic information is used to define the reference classes during model development.

The released classifier itself uses photometric measurements at inference time.

## 3. Input Photometry

The public `ActivityClassifier` requires eight magnitude columns:

`u`

`g`

`r`

`i`

`z`

`W1`

`W2`

`W3`

The optical measurements correspond to SDSS photometry.

The infrared measurements correspond to WISE photometry.

Users do not need to calculate colour features manually.

## 4. Feature Construction

SpecLess automatically constructs all pairwise colours from the eight input magnitudes.

The Model 2 feature representation contains:

| Feature type | Number |
|---|---:|
| Magnitudes | 8 |
| Pairwise colours | 28 |
| Total features | 36 |

Feature construction is implemented in:

`src/specless_astro/activity/features.py`

The prediction interface is implemented in:

`src/specless_astro/activity/classifier.py`

## 5. Train and Test Split

The final photometric catalogue was divided using a stratified 80/20 train and test split with:

`random_state = 42`

This produced:

`Training sample before balancing = 67,721`

`Independent test sample = 16,931`

The independent test sample was kept unchanged for final evaluation.

## 6. Training Sample Balancing

Balancing was applied only to the training sample.

The two dominant classes were undersampled to:

`Star forming = 10,000`

`Starburst = 10,000`

The smaller classes were retained:

`Seyfert = 4,792`

`LINER = 3,376`

The resulting final training sample contained:

`28,168 galaxies`

Inverse frequency sample weights were also applied during XGBoost training.

The independent test sample was not artificially balanced.

## 7. XGBoost Configuration

The final Model 2 XGBoost configuration reproduced during package reconstruction was:

```text
objective        = multi:softprob
num_class        = 4
n_estimators     = 800
learning_rate    = 0.05
max_depth        = 6
subsample        = 0.8
colsample_bytree = 0.8
min_child_weight = 1
reg_alpha        = 0
reg_lambda       = 1
eval_metric      = mlogloss
tree_method      = hist
random_state     = 42
n_jobs           = -1
```

The original saved Model 2 training artifact was not available during package reconstruction.

The final classifier was therefore reconstructed from the final catalogue and archived analysis configuration.

The reconstructed summary metrics matched the stored original Model 2 results exactly.

## 8. Native SDSS and WISE Performance

### Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.827476 |
| Balanced accuracy | 0.774225 |
| Macro F1 | 0.766127 |
| Macro ROC AUC | 0.957359 |

### XGBoost

| Metric | Value |
|---|---:|
| Accuracy | 0.821688 |
| Balanced accuracy | 0.786524 |
| Macro F1 | 0.763706 |
| Macro ROC AUC | 0.957192 |

The public `ActivityClassifier` uses the XGBoost model.

The stored native performance results are available at:

`validation/model2/native/model2_final_performance.csv`

## 9. Released Model Artifact

The packaged Model 2 XGBoost model is stored at:

`src/specless_astro/resources/model2/model2_xgboost_final.json`

Associated metadata are stored at:

`src/specless_astro/resources/model2/model2_model_info.json`

The released classifier uses a native XGBoost Booster representation.

## 10. Prediction Interface

A catalogue can be classified with:

```python
import pandas as pd
from specless_astro import ActivityClassifier

catalog = pd.read_csv("catalog.csv")

model = ActivityClassifier()
result = model.predict(catalog)
```

The returned table contains the original catalogue information together with SpecLess prediction columns.

Principal outputs are:

`specless_class`

`specless_label`

`p_seyfert`

`p_starburst`

`p_starforming`

`p_liner`

`confidence`

## 11. Output Interpretation

For each object:

`p_seyfert`

is the probability assigned to the Seyfert class.

`p_starburst`

is the probability assigned to the Starburst class.

`p_starforming`

is the probability assigned to the Star forming class.

`p_liner`

is the probability assigned to the LINER class.

`confidence`

is the probability associated with the predicted class.

The categorical prediction is returned in:

`specless_label`

and the internal numerical class in:

`specless_class`

## 12. Frozen Regression Verification

A fixed reference sample is stored at:

`tests/data/model2_reference_sample.csv`

The sample contains:

`20 objects`

with:

`5 objects per class`

The corresponding tests are implemented in:

`tests/test_model2.py`

The regression tests verify:

- construction of the complete 36 feature representation
- required input bands
- frozen class predictions
- frozen class probabilities
- model metadata

These tests are intended to detect unintended numerical changes across software versions.

## 13. KiDS and VIKING External Validation

The KiDS and VIKING validation catalogue contains 5,521 labelled galaxies.

| Class | Number |
|---|---:|
| Star forming | 2,459 |
| Starburst | 2,337 |
| Seyfert | 439 |
| LINER | 286 |
| Total | 5,521 |

The validation products are stored under:

`validation/model2/kids_viking/`

## 14. Native KiDS and VIKING Performance

The KiDS and VIKING native experiment produced the following results.

### Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.77828 |
| Balanced accuracy | 0.64645 |
| Macro F1 | 0.65767 |
| Macro ROC AUC | 0.93893 |

### XGBoost

| Metric | Value |
|---|---:|
| Accuracy | 0.76652 |
| Balanced accuracy | 0.65693 |
| Macro F1 | 0.65592 |
| Macro ROC AUC | 0.92905 |

The corresponding results are stored in:

`validation/model2/kids_viking/model2_kids_viking_performance.csv`

## 15. SDSS and WISE to KiDS and VIKING Transfer

The SDSS and WISE trained classifier was evaluated on the external KiDS and VIKING catalogue.

### Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.72507 |
| Balanced accuracy | 0.65861 |
| Macro F1 | 0.65018 |
| Macro ROC AUC | 0.94800 |

### XGBoost

| Metric | Value |
|---|---:|
| Accuracy | 0.793531 |
| Balanced accuracy | 0.758591 |
| Macro F1 | 0.752844 |
| Macro ROC AUC | 0.953315 |

The corresponding result file is:

`validation/model2/kids_viking/model2_sdsswise_train_kidsviking_external_test.csv`

This experiment evaluates cross survey behaviour and is distinct from the native SDSS and WISE independent test evaluation.

## 16. Mid Infrared Controlled Comparison

The contribution of WISE information was examined using five fold cross validation.

Relevant files are:

`validation/model2/kids_viking/model2_kidsviking_wise_controlled_comparison.csv`

`validation/model2/kids_viking/model2_mir_cv_folds.csv`

`validation/model2/kids_viking/model2_mir_cv_summary.csv`

### KiDS and VIKING

XGBoost produced:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.77867 | 0.01592 |
| Balanced accuracy | 0.68522 | 0.03211 |
| Macro F1 | 0.68932 | 0.02875 |
| Macro ROC AUC | 0.93576 | 0.00527 |

### KiDS, VIKING and WISE

XGBoost produced:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.82739 | 0.01545 |
| Balanced accuracy | 0.77541 | 0.02785 |
| Macro F1 | 0.77974 | 0.02130 |
| Macro ROC AUC | 0.95738 | 0.00745 |

## 17. Interpretability

Model 2 interpretability products are stored under:

`validation/model2/interpretability/`

The public repository contains:

- global SHAP results
- Seyfert specific SHAP results
- Starburst specific SHAP results
- Star forming specific SHAP results
- LINER specific SHAP results
- XGBoost feature importance products

These files describe model behaviour and are not additional training data.

## 18. Survey Transfer

Application to another survey requires caution.

Different photometric systems can differ in:

- filter transmission
- magnitude calibration
- limiting depth
- source selection
- photometric uncertainty
- redshift distribution
- signal to noise
- survey specific systematics

Matching column names alone does not imply identical photometric domains.

External applications should therefore be validated independently before predictions are used for scientific interpretation.

## 19. DESI Legacy Surveys

DESI Legacy Surveys are part of the broader SpecLess survey portability programme.

A numerical DESI Legacy Survey result is not included in the current Model 2 documentation because no corresponding verified public Model 2 validation artifact is currently indexed in the repository.

DESI results should be added here only after the associated catalogue, experiment definition, and result artifact are publicly identifiable.

## 20. LSST

LSST is not a training or validation source for the current Model 2 release.

The connection to LSST is methodological.

SpecLess is designed for photometric catalogue level inference and can therefore be adapted to future large imaging survey catalogues.

Any LSST application will require independent validation and survey specific feature mapping before scientific use.

No LSST performance metric is claimed for the current release.

## 21. Model Scope

The current Model 2 classifier distinguishes only:

- Seyfert
- Starburst
- Star forming
- LINER

Objects outside these four classes may still receive one of the available labels.

The model should therefore be applied to samples whose astrophysical population is reasonably compatible with the released classification problem.

## 22. Limitations

The current public Model 2 release should be interpreted with the following limitations:

1. The classifier is restricted to four activity classes.
2. Native test performance does not guarantee equivalent performance on another survey.
3. Survey dependent photometric differences can introduce domain shift.
4. Spectroscopic information is not used during inference.
5. The required photometric bands must be available.
6. Classification probabilities should not automatically be interpreted as perfectly calibrated astrophysical probabilities.
7. External survey applications require independent validation.
8. The current Model 2 release does not include a verified public DESI Legacy Survey validation artifact.

## 23. Related Repository Files

Classifier:

`src/specless_astro/activity/classifier.py`

Feature construction:

`src/specless_astro/activity/features.py`

Model artifact:

`src/specless_astro/resources/model2/model2_xgboost_final.json`

Model metadata:

`src/specless_astro/resources/model2/model2_model_info.json`

Frozen regression sample:

`tests/data/model2_reference_sample.csv`

Regression tests:

`tests/test_model2.py`

Native validation:

`validation/model2/native/`

KiDS and VIKING validation:

`validation/model2/kids_viking/`

Interpretability products:

`validation/model2/interpretability/`

General data documentation:

`DATA.md`

Reproducibility documentation:

`REPRODUCIBILITY.md`

Results map:

`RESULTS_INDEX.md`

## 24. Version

This document describes Model 2 as distributed with:

`SpecLess v0.2.1`
