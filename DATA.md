# Data

This document describes the astronomical catalogues, class definitions, photometric inputs, sample construction, and external survey datasets used in the development and validation of SpecLess.

The installable Python package does not distribute the complete survey catalogues used during model development. Large catalogue products remain separate from the software distribution because they originate from external astronomical surveys and can be substantially larger than the package itself.

The public repository instead contains pretrained model files, compact regression samples, validation products, model metadata, and the software required for photometric inference.

## 1. Data Roles in SpecLess

The datasets used in SpecLess have different roles.

### Training and internal evaluation

These datasets are used to construct labelled samples and train the released classifiers.

### External validation

These datasets are used to evaluate how a classifier behaves when applied outside its native survey domain.

### Future survey portability

Some surveys are part of the broader SpecLess portability programme but are not necessarily represented by a released pretrained classifier or a complete public validation product in the current software version.

These roles should not be treated as interchangeable.

## 2. Model 1 Data

### 2.1 Scientific objective

Model 1 separates galaxies into two morphological classes:

1. Elliptical
2. Spiral

The released model performs classification using photometric catalogue measurements only.

### 2.2 Morphological reference labels

The morphological reference sample was constructed using Galaxy Zoo 1 classifications.

The Elliptical sample was selected using a debiased Galaxy Zoo 1 elliptical probability criterion:

`p_elliptical >= 0.80`

The resulting reference labels were matched to SDSS catalogue information.

Quality control included checks on spectroscopic galaxy classification and photometric data quality.

The working selection included:

` spectral_class = GALAXY `

and the development workflow excluded problematic spectroscopic or photometric entries where required by the final sample construction.

### 2.3 Final Model 1 sample

The final fixed sample used for the released Model 1 experiment contains:

| Sample | Elliptical | Spiral | Total |
|---|---:|---:|---:|
| Training | 51,553 | 90,659 | 142,212 |
| Test | 12,888 | 22,666 | 35,554 |

The training and test samples contain no overlapping SDSS `objID` values.

The independent test sample was not used during final model fitting.

### 2.4 Model 1 photometry

The released classifier requires:

`u, g, r, i, z, W1, W2`

The optical bands correspond to SDSS photometry.

The mid infrared bands correspond to WISE photometry.

From these seven input magnitudes, SpecLess constructs all 21 pairwise colours automatically.

The complete Model 1 feature vector therefore contains 28 features.

## 3. Model 2 Data

### 3.1 Scientific objective

Model 2 separates galaxies into four activity classes:

1. Seyfert
2. Starburst
3. Star forming
4. LINER

The reference labels originate from spectroscopically classified galaxy samples, while the released classifier itself uses photometric information for inference.

### 3.2 Parent labelled sample

The final spectroscopically labelled parent sample contains 133,886 galaxies.

| Class | Number |
|---|---:|
| Star forming | 60,000 |
| Starburst | 60,000 |
| Seyfert | 7,264 |
| LINER | 6,622 |
| Total | 133,886 |

The research catalogue used the following class codes:

`3 = Seyfert`

`4 = Starburst`

`5 = Star forming`

`6 = LINER`

The public software maps these classes internally to:

`0 = SEYFERT`

`1 = STARBURST`

`2 = STARFORMING`

`3 = LINER`

### 3.3 Final photometric sample

After requiring the photometric measurements needed by the released Model 2 classifier, the final sample contains 84,652 galaxies.

| Class | Number |
|---|---:|
| Star forming | 37,518 |
| Starburst | 36,924 |
| Seyfert | 5,990 |
| LINER | 4,220 |
| Total | 84,652 |

### 3.4 Train and test split

The final photometric catalogue was divided using a stratified 80 percent training and 20 percent test split with:

`random_state = 42`

This produced:

`Training sample before balancing = 67,721`

`Independent test sample = 16,931`

The test sample was retained in its original class distribution.

### 3.5 Training sample balancing

Balancing was applied only to the training sample.

The two dominant classes were undersampled to:

`Star forming = 10,000`

`Starburst = 10,000`

The smaller classes were retained:

`Seyfert = 4,792`

`LINER = 3,376`

The final Model 2 training set therefore contains:

`28,168 galaxies`

Inverse frequency sample weights were also used during XGBoost training.

The independent test sample was not artificially balanced.

### 3.6 Model 2 photometry

The released Model 2 classifier requires:

`u, g, r, i, z, W1, W2, W3`

The optical bands correspond to SDSS photometry.

The infrared bands correspond to WISE photometry.

SpecLess constructs all 28 pairwise colours automatically from these eight magnitudes.

The complete Model 2 feature vector therefore contains 36 features.

## 4. KiDS and VIKING External Validation

KiDS and VIKING were used to investigate cross survey behaviour outside the native SDSS and WISE domain.

The final KiDS and VIKING validation catalogue used for Model 2 contains 5,521 galaxies.

| Class | Number |
|---|---:|
| Star forming | 2,459 |
| Starburst | 2,337 |
| Seyfert | 439 |
| LINER | 286 |
| Total | 5,521 |

The combined catalogue contains optical and near infrared measurements from KiDS and VIKING, with WISE information included in experiments where mid infrared information was required.

Validation products are stored under:

`validation/model2/kids_viking/`

Important public result files include:

`model2_kids_viking_performance.csv`

`model2_kidsviking_wise_controlled_comparison.csv`

`model2_mir_cv_folds.csv`

`model2_mir_cv_summary.csv`

`model2_sdsswise_train_kidsviking_external_test.csv`

These products are external validation outputs and are not training data for the released SDSS and WISE classifier.

## 5. Model 1 Cross Survey Validation

Model 1 was also examined using KiDS based external survey experiments.

The public validation material is stored under:

`validation/model1/`

These experiments examine survey transfer behaviour and should be distinguished from the native SDSS and WISE test sample.

The common feature representation used for a specific transfer experiment may differ from the 28 feature representation used by the released native classifier because external surveys do not necessarily provide identical filter systems.

## 6. DESI Legacy Surveys

DESI Legacy Surveys are part of the broader SpecLess cross survey portability programme.

DESI Legacy Surveys should only be described as a completed public validation component where a corresponding catalogue, result file, or documented analysis artifact is available in the repository.

The current data documentation therefore does not assign numerical DESI validation results unless they can be directly connected to a verified public artifact.

This distinction is intentional and prevents preliminary or incomplete experiments from being presented as final reproducible results.

## 7. LSST

LSST is a future application domain for SpecLess rather than a training source for the current public release.

The motivation is methodological.

SpecLess is designed for catalogue level photometric inference and is therefore compatible in principle with the scale of future imaging surveys where spectroscopy will not exist for the majority of detected galaxies.

Application to LSST data will require survey specific photometric mapping, validation, and domain shift assessment before scientific use.

No LSST performance metric is claimed for the current release.

## 8. Why the Full Survey Catalogues Are Not Included

The full research catalogues are not distributed inside the Python package.

There are several reasons.

First, the catalogues originate from external astronomical surveys and remain subject to the data access, citation, and redistribution practices of those projects.

Second, the full tables are substantially larger than the compact software package required for inference.

Third, most users of SpecLess do not require the original training catalogues in order to apply the released pretrained classifiers.

The public repository instead contains the components required for software verification and scientific inspection:

1. pretrained model artifacts
2. model metadata
3. deterministic feature construction code
4. compact frozen reference samples
5. regression tests
6. stored validation metrics
7. external survey validation products
8. interpretability products

## 9. Frozen Reference Samples

Compact reference samples are included for regression testing.

Model 1:

`tests/data/model1_reference_sample.csv`

Model 2:

`tests/data/model2_reference_sample.csv`

These samples are not intended to reproduce the complete training process.

Their purpose is to verify that software changes do not unexpectedly alter predictions or probabilities produced by the released model artifacts.

## 10. Validation Products

Scientific validation results are stored separately from the installable package logic.

The principal validation directories are:

`validation/model1/`

`validation/model2/native/`

`validation/model2/kids_viking/`

`validation/model2/interpretability/`

These directories contain stored metrics and analysis products used to evaluate the released models and investigate their behaviour outside the native training domain.

## 11. Interpretability Data Products

Model 2 interpretability products include:

1. global SHAP outputs
2. class specific SHAP outputs
3. XGBoost feature importance products

These files are stored under:

`validation/model2/interpretability/`

They are derived analysis products and should not be interpreted as additional training data.

## 12. Data Separation Principles

SpecLess follows the following data handling principles:

1. Training and independent test samples are kept logically separate.
2. External survey catalogues are treated separately from native survey evaluation.
3. Class balancing is applied only to training data where explicitly documented.
4. Frozen regression samples are used only for software verification.
5. Preliminary survey experiments are not presented as final validation results without a corresponding verified artifact.
6. Photometric inference is kept separate from the spectroscopic information used to construct reference labels.
7. Large external survey catalogues are not embedded unnecessarily inside the Python package.

## 13. Recommended Use of New Catalogues

A new catalogue supplied to a released classifier must contain the required photometric input bands.

For Model 1:

`u, g, r, i, z, W1, W2`

For Model 2:

`u, g, r, i, z, W1, W2, W3`

Users applying SpecLess to a survey other than the native survey system should not assume that matching column names imply identical photometric domains.

Differences in:

1. filter transmission
2. photometric calibration
3. depth
4. source selection
5. redshift distribution
6. signal to noise
7. survey systematics

can introduce domain shift.

External survey applications should therefore be validated independently before the resulting classifications are used for scientific inference.

## 14. Data Provenance and Future Documentation

Future public data products should include, where applicable:

1. source survey
2. catalogue release
3. selection criteria
4. class definition
5. cross match procedure
6. matching radius
7. photometric quality criteria
8. final object count
9. feature mapping
10. associated model version

This information should be recorded before a new validation dataset is treated as a formal SpecLess benchmark.

## 15. Current Release

This document describes the datasets associated with:

`SpecLess v0.2.1`

and should be updated when new public datasets, survey validation products, or model versions are added.
