# Reproducibility

This document describes the reproducibility framework of **SpecLess**, including the public pretrained models, feature construction, train and test samples, validation products, frozen regression samples, and the procedures used to verify numerical consistency.

SpecLess is designed as a photometry based galaxy classification framework. Spectroscopic information is used to construct or validate reference labels during model development, but the released classifiers operate on photometric measurements at inference time.

The current public software release is:

`specless-astro v0.2.1`

---

## 1. Reproducibility Scope

Reproducibility in SpecLess is considered at three levels.

### 1.1 Software reproducibility

A clean Python environment can install the public release directly from PyPI:

```bash
pip install specless-astro==0.2.1
```

The released package contains:

- the Model 1 pretrained XGBoost model
- the Model 2 pretrained XGBoost model
- model metadata
- feature construction code
- prediction interfaces
- regression tests
- compact frozen reference samples

The package has been successfully installed and loaded in a clean environment using:

```python
from specless_astro import MorphologyClassifier, ActivityClassifier
```

Both classifiers are distributed with the package and do not require downloading external model weights.

### 1.2 Prediction reproducibility

Frozen reference catalogues are included under:

```text
tests/data/
```

These samples are used to check that future software changes do not alter the expected classifier behaviour.

The corresponding regression tests are:

```text
tests/test_model1.py
tests/test_model2.py
```

### 1.3 Scientific result reproducibility

The repository contains validation products under:

```text
validation/model1/
validation/model2/
```

These files preserve the numerical outputs used to evaluate the released classifiers, including internal performance, external survey transfer experiments, and interpretability products.

Large source catalogues and intermediate survey tables are not distributed with the Python package.

The public repository therefore currently supports direct reproduction of packaged predictions and inspection of validation results, while complete reconstruction of every upstream catalogue from the original survey archives requires the corresponding catalogue selection and cross matching workflow.

---

# 2. Model 1: Galaxy Morphology

## 2.1 Scientific target

Model 1 performs binary morphological classification between:

- Elliptical
- Spiral

The internal class mapping is:

```text
0 = Elliptical
1 = Spiral
```

The reference morphological labels were constructed from Galaxy Zoo 1 classifications and matched to SDSS photometric information.

The final classifier performs inference using photometric measurements only.

---

## 2.2 Input photometry

The public `MorphologyClassifier` requires seven magnitude columns:

```text
u
g
r
i
z
W1
W2
```

These correspond to SDSS optical photometry and WISE mid infrared photometry.

No user supplied colour columns are required.

---

## 2.3 Feature construction

From the seven input magnitudes, SpecLess automatically constructs all 21 pairwise colour combinations.

The resulting Model 1 feature vector therefore contains:

```text
7 magnitudes
21 colours
-----------
28 features
```

Feature generation is implemented in:

```text
src/specless_astro/morphology/features.py
```

The prediction interface is implemented in:

```text
src/specless_astro/morphology/classifier.py
```

---

## 2.4 Final samples

The final fixed samples used for Model 1 contain:

| Sample | Number of galaxies |
|---|---:|
| Training sample | 142,212 |
| Independent test sample | 35,554 |
| Elliptical galaxies in training sample | 51,553 |
| Spiral galaxies in training sample | 90,659 |
| Elliptical galaxies in test sample | 12,888 |
| Spiral galaxies in test sample | 22,666 |

The final training and test catalogues contain no overlapping `objID` values.

The test sample was not used during final model fitting.

---

## 2.5 Released model

The released Model 1 XGBoost model is stored at:

```text
src/specless_astro/resources/model1/model1_xgboost_final.json
```

Associated model metadata are stored at:

```text
src/specless_astro/resources/model1/model1_model_info.json
```

The exact original training hyperparameter configuration should be taken from the archived training configuration or model metadata rather than reconstructed from assumptions. Hyperparameters not explicitly recorded in the public repository are not inferred in this document.

---

## 2.6 Independent test performance

The released XGBoost morphology classifier produced the following performance on the fixed independent test sample:

| Metric | Value |
|---|---:|
| Accuracy | 0.937475 |
| Balanced accuracy | 0.938761 |
| Macro precision | 0.928644 |
| Macro F1 | 0.933181 |
| ROC AUC | 0.982557 |
| Elliptical recall | 0.943436 |
| Spiral recall | 0.934086 |

A Random Forest comparison model produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.936294 |
| Balanced accuracy | 0.930805 |
| Macro F1 | 0.931043 |
| ROC AUC | 0.980704 |

The public package uses the XGBoost classifier as the released Model 1 estimator.

---

## 2.7 Frozen prediction verification

A fixed 20 object reference sample is stored at:

```text
tests/data/model1_reference_sample.csv
```

The regression tests compare current package predictions against the frozen reference predictions.

During packaging verification, the reproduced Model 1 numerical results agreed with the original results to better than:

```text
|Delta| < 5e-7
```

for the evaluated metrics.

---

## 2.8 Cross survey validation

Model 1 was additionally examined outside the native SDSS and WISE domain.

Validation products are stored under:

```text
validation/model1/
```

The cross survey work includes KiDS based experiments.

For an SDSS to KiDS transfer experiment using the common 10 feature representation, the XGBoost classifier produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.946548 |
| Balanced accuracy | 0.945002 |
| Macro F1 | 0.939456 |
| ROC AUC | 0.984414 |

For the reverse KiDS to SDSS experiment, the XGBoost classifier produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.886526 |
| Balanced accuracy | 0.891594 |
| Macro F1 | 0.880202 |
| ROC AUC | 0.947709 |

These experiments should be interpreted as survey transfer tests rather than as replacements for the native Model 1 evaluation.

---

# 3. Model 2: Galaxy Activity Classification

## 3.1 Scientific target

Model 2 performs four class activity classification between:

- Seyfert
- Starburst
- Star forming
- LINER

The original research catalogue used the following class codes:

```text
3 = Seyfert
4 = Starburst
5 = Star forming
6 = LINER
```

For the released software model, these labels are internally mapped to:

```text
0 = SEYFERT
1 = STARBURST
2 = STARFORMING
3 = LINER
```

The mapping between research labels and software output labels is therefore explicit and deterministic.

---

## 3.2 Reference catalogue

The spectroscopically labelled parent sample contains:

```text
133,886 galaxies
```

with the following class composition:

| Class | Number |
|---|---:|
| Star forming | 60,000 |
| Starburst | 60,000 |
| Seyfert | 7,264 |
| LINER | 6,622 |

The final catalogue with the photometric measurements required by Model 2 contains:

```text
84,652 galaxies
```

distributed as:

| Class | Number |
|---|---:|
| Star forming | 37,518 |
| Starburst | 36,924 |
| Seyfert | 5,990 |
| LINER | 4,220 |

No redshift feature is required by the released classifier.

---

## 3.3 Input photometry

The public `ActivityClassifier` requires eight magnitude columns:

```text
u
g
r
i
z
W1
W2
W3
```

These correspond to five optical SDSS magnitudes and three WISE mid infrared magnitudes.

---

## 3.4 Feature construction

All pairwise colours are automatically generated from the eight input magnitudes.

The final Model 2 feature vector therefore contains:

```text
8 magnitudes
28 colours
-----------
36 features
```

Feature generation is implemented in:

```text
src/specless_astro/activity/features.py
```

The prediction interface is implemented in:

```text
src/specless_astro/activity/classifier.py
```

---

# 4. Model 2 Train and Test Construction

## 4.1 Fixed split

The final photometric catalogue was divided using a stratified 80/20 train and test split with:

```text
random_state = 42
```

This produced:

```text
Training sample before balancing = 67,721
Independent test sample           = 16,931
```

The test sample was kept unchanged for final evaluation.

---

## 4.2 Training only class balancing

Class balancing was applied only to the training sample.

The two largest classes were randomly undersampled to:

```text
Star forming = 10,000
Starburst    = 10,000
```

The smaller classes were retained:

```text
Seyfert = 4,792
LINER   = 3,376
```

The resulting final training sample contained:

```text
28,168 galaxies
```

The independent test distribution was not artificially balanced.

---

## 4.3 Sample weighting

Inverse frequency sample weights were applied during XGBoost training to reduce the influence of the remaining class imbalance.

Balancing therefore consisted of:

1. training only undersampling of the two dominant classes
2. inverse frequency sample weighting during model fitting

The test catalogue was not modified by either procedure.

---

# 5. Model 2 XGBoost Configuration

The final Model 2 XGBoost configuration reproduced during independent reconstruction was:

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

The original Model 2 model file was not available during package reconstruction.

The classifier was therefore reconstructed from the final catalogue and archived analysis configuration.

The reconstructed results matched the stored original Model 2 results exactly for the reported summary metrics.

---

# 6. Model 2 Independent Test Performance

The final XGBoost classifier produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.8216880279 |
| Balanced accuracy | 0.7865243976 |
| Macro F1 | 0.7637061004 |
| Macro ROC AUC | 0.9571918660 |

The differences between the reconstructed and stored summary results were:

```text
0
```

for all four reported metrics.

The Random Forest comparison model produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.827476 |
| Balanced accuracy | 0.774225 |
| Macro F1 | 0.766127 |
| Macro ROC AUC | 0.957359 |

The released package uses XGBoost because the final model selection considered the complete performance profile, including balanced classification performance and external survey behaviour.

---

# 7. Released Model 2 Artifact

The packaged model is stored at:

```text
src/specless_astro/resources/model2/model2_xgboost_final.json
```

Associated metadata are stored at:

```text
src/specless_astro/resources/model2/model2_model_info.json
```

The released classifier uses the native XGBoost Booster representation and constructs an XGBoost `DMatrix` internally for inference.

---

# 8. Model 2 Frozen Prediction Verification

A fixed 20 object regression sample is stored at:

```text
tests/data/model2_reference_sample.csv
```

The sample contains five objects from each of the four classes.

Expected probabilities and predictions are frozen in the regression test workflow.

The corresponding tests verify:

- construction of the complete 36 feature vector
- required input band checking
- predictions on the fixed reference catalogue
- class probabilities
- model metadata

The Model 2 tests are implemented in:

```text
tests/test_model2.py
```

---

# 9. Model 2 External Survey Validation

External validation uses KiDS and VIKING photometry, with WISE information included where required by the experiment.

The final KiDS and VIKING validation catalogue contains:

```text
5,521 galaxies
```

distributed as:

| Class | Number |
|---|---:|
| Star forming | 2,459 |
| Starburst | 2,337 |
| Seyfert | 439 |
| LINER | 286 |

The validation products are stored under:

```text
validation/model2/kids_viking/
```

Important files include:

```text
model2_kids_viking_performance.csv
model2_kidsviking_wise_controlled_comparison.csv
model2_mir_cv_folds.csv
model2_mir_cv_summary.csv
model2_sdsswise_train_kidsviking_external_test.csv
```

---

## 9.1 KiDS and VIKING native experiment

The XGBoost classifier produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.76652 |
| Balanced accuracy | 0.65693 |
| Macro F1 | 0.65592 |
| Macro ROC AUC | 0.92905 |

The corresponding Random Forest results were:

| Metric | Value |
|---|---:|
| Accuracy | 0.77828 |
| Balanced accuracy | 0.64645 |
| Macro F1 | 0.65767 |
| Macro ROC AUC | 0.93893 |

---

## 9.2 SDSS and WISE to KiDS and VIKING transfer

For the SDSS and WISE trained XGBoost classifier evaluated on the KiDS and VIKING external catalogue:

| Metric | Value |
|---|---:|
| Accuracy | 0.793531 |
| Balanced accuracy | 0.758591 |
| Macro F1 | 0.752844 |
| Macro ROC AUC | 0.953315 |

The corresponding Random Forest transfer experiment produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.72507 |
| Balanced accuracy | 0.65861 |
| Macro F1 | 0.65018 |
| Macro ROC AUC | 0.94800 |

---

## 9.3 Mid infrared controlled cross validation

Five fold cross validation was also performed to examine the contribution of the WISE information.

For KiDS and VIKING without the additional WISE information, XGBoost produced:

```text
Accuracy          = 0.77867 +/- 0.01592
Balanced accuracy = 0.68522 +/- 0.03211
Macro F1          = 0.68932 +/- 0.02875
Macro ROC AUC     = 0.93576 +/- 0.00527
```

For KiDS, VIKING and WISE together:

```text
Accuracy          = 0.82739 +/- 0.01545
Balanced accuracy = 0.77541 +/- 0.02785
Macro F1          = 0.77974 +/- 0.02130
Macro ROC AUC     = 0.95738 +/- 0.00745
```

The corresponding fold level and summary products are retained in the validation directory.

---

# 10. Interpretability Products

Model 2 interpretability products are stored under:

```text
validation/model2/interpretability/
```

The repository contains:

- global SHAP results
- class specific SHAP results for all four activity classes
- XGBoost feature importance products

These products are intended to support interpretation of the trained classifier rather than to define new input features.

---

# 11. DESI Legacy Surveys

DESI Legacy Surveys are part of the broader SpecLess cross survey development programme.

However, numerical DESI Legacy Survey results are not listed in this reproducibility document unless a corresponding public validation artifact can be directly identified in the repository.

This prevents undocumented or preliminary experiments from being presented as reproducible public results.

When the corresponding verified outputs are added to the repository, they should be documented here using the same structure as the KiDS and VIKING experiments:

- catalogue definition
- sample size
- common feature set
- train and test direction
- model configuration
- evaluation metrics
- exact output file

---

# 12. LSST

LSST is not presented as a training or validation data source for the current public release.

The connection to LSST is methodological.

SpecLess is designed around catalogue level photometric inference, allowing the framework to be adapted to large imaging surveys where spectroscopy is unavailable for the majority of detected sources.

Any future LSST application should therefore be treated as a new survey domain and independently validated before scientific use.

---

# 13. Automated Tests

The repository currently contains regression and interface tests for both public models.

The complete test suite can be executed with:

```bash
pytest -v
```

The tests include checks for:

- Model 1 feature generation
- Model 2 feature generation
- missing required photometric bands
- model metadata
- frozen prediction outputs
- frozen probability outputs

GitHub Actions automatically runs the test suite for supported Python environments.

The public repository currently tests Python:

```text
3.10
3.11
3.12
```

---

# 14. Runtime Dependencies

The public package declares the following minimum runtime dependencies:

```text
numpy >= 1.26
pandas >= 2.0
scikit-learn >= 1.4
xgboost >= 3.1, < 4
```

The package requires:

```text
Python >= 3.10
```

These are package compatibility requirements and should not automatically be interpreted as the exact historical environment in which every original research experiment was performed.

For strict historical reconstruction, the exact environment associated with a specific experiment should be recorded separately from the minimum installation requirements.

---

# 15. Public Package Verification

The final public release was independently installed from PyPI in a clean virtual environment using:

```bash
pip install specless-astro==0.2.1
```

The installed package reported:

```text
VERSION: 0.2.1
MODEL 1 OK
MODEL 2 OK
```

This verifies that both public model artifacts are included in the PyPI distribution and can be loaded outside the development repository.

---

# 16. Repository Artifacts Relevant to Reproducibility

The main reproducibility related paths are:

```text
src/specless_astro/
├── morphology/
│   ├── classifier.py
│   └── features.py
├── activity/
│   ├── classifier.py
│   └── features.py
└── resources/
    ├── model1/
    │   ├── model1_model_info.json
    │   └── model1_xgboost_final.json
    └── model2/
        ├── model2_model_info.json
        └── model2_xgboost_final.json
```

Regression resources:

```text
tests/
├── test_model1.py
├── test_model2.py
└── data/
    ├── model1_reference_sample.csv
    └── model2_reference_sample.csv
```

Scientific validation resources:

```text
validation/
├── model1/
└── model2/
    ├── native/
    ├── kids_viking/
    └── interpretability/
```

---

# 17. What the Public Repository Reproduces Directly

The current public repository allows a researcher to directly reproduce or verify:

1. installation of the released SpecLess software
2. loading of both pretrained classifiers
3. deterministic feature construction from the required photometric bands
4. predictions from the released Model 1 classifier
5. predictions from the released Model 2 classifier
6. frozen regression predictions on reference catalogues
7. packaged model metadata
8. stored internal validation metrics
9. stored KiDS and VIKING external validation results
10. Model 2 SHAP and feature importance products
11. automated software regression tests

---

# 18. What Is Not Reconstructed Solely from the Public Package

The current PyPI package does not contain the complete large survey catalogues used during model development.

The following are therefore distinct from simply installing the package:

- downloading the original survey catalogues
- reproducing all archive queries
- reconstructing every cross match from raw survey catalogues
- rebuilding the complete labelled parent catalogues
- reproducing every exploratory experiment performed during development

These operations require the corresponding public survey data and catalogue construction procedures.

Their absence from the installable Python package does not affect inference with the released pretrained classifiers, but it should be considered when distinguishing software reproducibility from complete end to end reconstruction of the research data pipeline.

---

# 19. Reproducibility Principles

The SpecLess repository follows the following principles:

- reported numerical results should be traceable to stored outputs
- released classifiers should be accompanied by frozen regression tests
- class mappings should be explicit
- feature definitions should be deterministic
- training and test samples should remain logically separated
- external survey results should be identified separately from native survey performance
- preliminary experiments should not be presented as reproducible final results
- unsupported numerical values should not be reconstructed from memory or assumption
- package compatibility requirements should not be confused with historical training environments
- future model changes should be versioned and documented

---

# 20. Versioning

This document describes the public SpecLess release:

```text
v0.2.1
```

Future changes to model weights, feature definitions, class mappings, or preprocessing behaviour should result in an explicit software version update and corresponding regression test update.

Changes affecting scientific predictions should be documented in the project changelog.
