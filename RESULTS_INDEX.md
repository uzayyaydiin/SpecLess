# Results Index

This document provides a map of the principal scientific results, validation products, model artifacts, and regression resources distributed with SpecLess.

Its purpose is to help researchers identify which repository files correspond to the different stages of model evaluation and validation without having to reconstruct the project structure manually.

This index describes:

`SpecLess v0.2.1`

---

## 1. Model 1: Galaxy Morphology

Model 1 performs binary classification between:

- Elliptical
- Spiral

The released classifier is based on XGBoost.

### 1.1 Released model

Model file:

`src/specless_astro/resources/model1/model1_xgboost_final.json`

Model metadata:

`src/specless_astro/resources/model1/model1_model_info.json`

Classifier implementation:

`src/specless_astro/morphology/classifier.py`

Feature construction:

`src/specless_astro/morphology/features.py`

### 1.2 Native independent test results

Final independent test sample:

`35,554 galaxies`

Class distribution:

| Class | Number |
|---|---:|
| Elliptical | 12,888 |
| Spiral | 22,666 |

Final XGBoost results:

| Metric | Value |
|---|---:|
| Accuracy | 0.937475 |
| Balanced accuracy | 0.938761 |
| Macro precision | 0.928644 |
| Macro F1 | 0.933181 |
| ROC AUC | 0.982557 |
| Elliptical recall | 0.943436 |
| Spiral recall | 0.934086 |

Random Forest comparison:

| Metric | Value |
|---|---:|
| Accuracy | 0.936294 |
| Balanced accuracy | 0.930805 |
| Macro F1 | 0.931043 |
| ROC AUC | 0.980704 |

Model 1 validation material is stored under:

`validation/model1/`

### 1.3 Cross survey results

Model 1 includes KiDS based survey transfer experiments.

For the SDSS to KiDS XGBoost experiment using the common 10 feature representation:

| Metric | Value |
|---|---:|
| Accuracy | 0.946548 |
| Balanced accuracy | 0.945002 |
| Macro F1 | 0.939456 |
| ROC AUC | 0.984414 |

For the reverse KiDS to SDSS XGBoost experiment:

| Metric | Value |
|---|---:|
| Accuracy | 0.886526 |
| Balanced accuracy | 0.891594 |
| Macro F1 | 0.880202 |
| ROC AUC | 0.947709 |

These are cross survey experiments and should not be interpreted as replacements for the native SDSS and WISE test evaluation.

### 1.4 Regression sample

Frozen Model 1 reference catalogue:

`tests/data/model1_reference_sample.csv`

Regression tests:

`tests/test_model1.py`

The reference sample contains 20 fixed objects used to verify predictions and numerical consistency across software changes.

---

## 2. Model 2: Galaxy Activity Classification

Model 2 performs four class classification between:

- Seyfert
- Starburst
- Star forming
- LINER

The released classifier is based on XGBoost.

### 2.1 Released model

Model file:

`src/specless_astro/resources/model2/model2_xgboost_final.json`

Model metadata:

`src/specless_astro/resources/model2/model2_model_info.json`

Classifier implementation:

`src/specless_astro/activity/classifier.py`

Feature construction:

`src/specless_astro/activity/features.py`

### 2.2 Native independent test results

Final independent test sample:

`16,931 galaxies`

Final XGBoost results:

| Metric | Value |
|---|---:|
| Accuracy | 0.821688 |
| Balanced accuracy | 0.786524 |
| Macro F1 | 0.763706 |
| Macro ROC AUC | 0.957192 |

Random Forest comparison:

| Metric | Value |
|---|---:|
| Accuracy | 0.827476 |
| Balanced accuracy | 0.774225 |
| Macro F1 | 0.766127 |
| Macro ROC AUC | 0.957359 |

The principal stored native performance file is:

`validation/model2/native/model2_final_performance.csv`

---

## 3. Model 2 KiDS and VIKING Validation

The KiDS and VIKING validation catalogue contains:

`5,521 galaxies`

Class distribution:

| Class | Number |
|---|---:|
| Star forming | 2,459 |
| Starburst | 2,337 |
| Seyfert | 439 |
| LINER | 286 |

The associated validation directory is:

`validation/model2/kids_viking/`

### 3.1 Native KiDS and VIKING performance

Results file:

`validation/model2/kids_viking/model2_kids_viking_performance.csv`

XGBoost:

| Metric | Value |
|---|---:|
| Accuracy | 0.76652 |
| Balanced accuracy | 0.65693 |
| Macro F1 | 0.65592 |
| Macro ROC AUC | 0.92905 |

Random Forest:

| Metric | Value |
|---|---:|
| Accuracy | 0.77828 |
| Balanced accuracy | 0.64645 |
| Macro F1 | 0.65767 |
| Macro ROC AUC | 0.93893 |

---

## 4. SDSS and WISE to KiDS and VIKING Transfer

This experiment evaluates a classifier trained in the SDSS and WISE domain on the external KiDS and VIKING sample.

Results file:

`validation/model2/kids_viking/model2_sdsswise_train_kidsviking_external_test.csv`

XGBoost:

| Metric | Value |
|---|---:|
| Accuracy | 0.793531 |
| Balanced accuracy | 0.758591 |
| Macro F1 | 0.752844 |
| Macro ROC AUC | 0.953315 |

Random Forest:

| Metric | Value |
|---|---:|
| Accuracy | 0.72507 |
| Balanced accuracy | 0.65861 |
| Macro F1 | 0.65018 |
| Macro ROC AUC | 0.94800 |

This experiment is one of the principal external survey tests of Model 2.

---

## 5. Mid Infrared Controlled Experiments

The effect of including WISE information was examined through controlled cross validation experiments.

Relevant files:

`validation/model2/kids_viking/model2_kidsviking_wise_controlled_comparison.csv`

`validation/model2/kids_viking/model2_mir_cv_folds.csv`

`validation/model2/kids_viking/model2_mir_cv_summary.csv`

### 5.1 KiDS and VIKING

Five fold XGBoost results:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.77867 | 0.01592 |
| Balanced accuracy | 0.68522 | 0.03211 |
| Macro F1 | 0.68932 | 0.02875 |
| Macro ROC AUC | 0.93576 | 0.00527 |

### 5.2 KiDS, VIKING and WISE

Five fold XGBoost results:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.82739 | 0.01545 |
| Balanced accuracy | 0.77541 | 0.02785 |
| Macro F1 | 0.77974 | 0.02130 |
| Macro ROC AUC | 0.95738 | 0.00745 |

---

## 6. Model 2 Interpretability

Interpretability products are stored under:

`validation/model2/interpretability/`

The directory contains:

- global SHAP results
- class specific SHAP results for Seyfert
- class specific SHAP results for Starburst
- class specific SHAP results for Star forming
- class specific SHAP results for LINER
- XGBoost feature importance products

These products describe the behaviour of the trained classifier and are not additional training datasets.

---

## 7. Model 2 Regression Sample

Frozen Model 2 reference catalogue:

`tests/data/model2_reference_sample.csv`

Regression tests:

`tests/test_model2.py`

The reference sample contains 20 objects:

`5 objects per class`

The tests verify:

- required input columns
- construction of the 36 model features
- predicted classes
- class probabilities
- metadata loading

---

## 8. Automated Testing

The complete software test suite is located under:

`tests/`

The public release contains eight automated tests covering Model 1 and Model 2.

The test suite can be executed with:

```bash
pytest -v
```

Continuous integration is configured through:

`.github/workflows/tests.yml`

The workflow tests the package on:

- Python 3.10
- Python 3.11
- Python 3.12

---

## 9. Model Feature Definitions

### Model 1

Required magnitudes:

`u, g, r, i, z, W1, W2`

Automatically generated colours:

`21`

Total features:

`28`

Implementation:

`src/specless_astro/morphology/features.py`

### Model 2

Required magnitudes:

`u, g, r, i, z, W1, W2, W3`

Automatically generated colours:

`28`

Total features:

`36`

Implementation:

`src/specless_astro/activity/features.py`

---

## 10. Documentation Map

Main project documentation:

`README.md`

Data documentation:

`DATA.md`

Reproducibility documentation:

`REPRODUCIBILITY.md`

Model 2 documentation:

`docs/model2.md`

Release history:

`CHANGELOG.md`

License:

`LICENSE`

---

## 11. DESI Legacy Surveys

DESI Legacy Surveys are part of the broader SpecLess cross survey portability work.

A numerical DESI result is not indexed in this document unless a corresponding public result artifact can be directly identified in the repository.

When verified DESI outputs are added publicly, they should be indexed here with:

- exact file path
- sample definition
- sample size
- feature representation
- training survey
- target survey
- evaluation metrics
- corresponding software version

---

## 12. LSST

LSST is currently treated as a future application domain rather than a source of released training or validation data.

No LSST performance result is indexed for SpecLess v0.2.1.

---

## 13. Result Interpretation

Results in this repository fall into three categories.

### Native test results

Evaluation using an independent test sample drawn from the same primary data domain as the training sample.

### External survey results

Evaluation using data from a different survey or photometric system.

### Interpretability products

Derived quantities intended to examine model behaviour, such as SHAP values and feature importance.

These categories should not be compared as though they represent identical experimental conditions.

---

## 14. Repository Result Structure

A simplified map of the result related repository structure is:

```text
validation/
├── model1/
└── model2/
    ├── native/
    │   └── model2_final_performance.csv
    ├── kids_viking/
    │   ├── model2_kids_viking_performance.csv
    │   ├── model2_kidsviking_wise_controlled_comparison.csv
    │   ├── model2_mir_cv_folds.csv
    │   ├── model2_mir_cv_summary.csv
    │   └── model2_sdsswise_train_kidsviking_external_test.csv
    └── interpretability/
```

Packaged model artifacts are stored separately under:

```text
src/specless_astro/resources/
├── model1/
└── model2/
```

Regression reference catalogues are stored under:

```text
tests/data/
```

---

## 15. Version

This result index corresponds to:

`SpecLess v0.2.1`

The index should be updated whenever a new public validation experiment, pretrained model, or scientifically relevant result artifact is added.
