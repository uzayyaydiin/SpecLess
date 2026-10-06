# Model 1: Galaxy Morphology Classification

Model 1 is the morphological classification component of SpecLess.

It performs binary classification between:

- Elliptical galaxies
- Spiral galaxies

The released classifier operates on photometric catalogue measurements and does not require spectroscopy at inference time.

## 1. Class Definition

The internal class mapping used by the released model is:

`0 = Elliptical`

`1 = Spiral`

The morphological reference labels were constructed using Galaxy Zoo 1 classifications matched to SDSS catalogue data.

For the elliptical reference sample, the adopted Galaxy Zoo 1 debiased probability criterion was:

`p_elliptical >= 0.80`

## 2. Input Photometry

The public `MorphologyClassifier` requires seven input magnitude columns:

`u`

`g`

`r`

`i`

`z`

`W1`

`W2`

The optical measurements correspond to SDSS photometry.

The mid infrared measurements correspond to WISE photometry.

Users do not need to calculate colour features manually.

## 3. Feature Construction

SpecLess automatically constructs all pairwise colours from the seven input magnitudes.

The feature representation contains:

| Feature type | Number |
|---|---:|
| Magnitudes | 7 |
| Pairwise colours | 21 |
| Total features | 28 |

Feature construction is implemented in:

`src/specless_astro/morphology/features.py`

The prediction interface is implemented in:

`src/specless_astro/morphology/classifier.py`

## 4. Final Dataset

The final Model 1 dataset was divided into fixed training and independent test samples.

| Sample | Elliptical | Spiral | Total |
|---|---:|---:|---:|
| Training | 51,553 | 90,659 | 142,212 |
| Test | 12,888 | 22,666 | 35,554 |

The training and test samples contain no overlapping SDSS `objID` values.

The independent test sample was not used during final model fitting.

## 5. Released Classifier

The public Model 1 classifier uses XGBoost.

The released model artifact is:

`src/specless_astro/resources/model1/model1_xgboost_final.json`

Associated metadata are stored in:

`src/specless_astro/resources/model1/model1_model_info.json`

The exact training configuration should be taken from the archived model metadata and training records where available.

Hyperparameters that are not explicitly preserved in the public artifacts should not be reconstructed from assumption.

## 6. Independent Test Performance

The released XGBoost classifier produced the following results on the fixed independent test sample:

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

## 7. Prediction Interface

A catalogue can be classified with:

```python id="g6w77y"
import pandas as pd
from specless_astro import MorphologyClassifier

catalog = pd.read_csv("catalog.csv")

model = MorphologyClassifier()
result = model.predict(catalog)
```

The returned table contains the original catalogue information together with SpecLess prediction columns.

Principal prediction outputs are:

` specless_class `

` specless_label `

` p_elliptical `

` p_spiral `

` confidence `

## 8. Example Output Interpretation

For each object:

`p_elliptical`

is the model probability associated with the Elliptical class.

`p_spiral`

is the model probability associated with the Spiral class.

`confidence`

represents the probability assigned to the predicted class.

The categorical prediction is available through:

`specless_label`

and the internal numerical class through:

`specless_class`

## 9. Regression Testing

A frozen reference sample is included at:

`tests/data/model1_reference_sample.csv`

The corresponding regression tests are implemented in:

`tests/test_model1.py`

The reference sample contains 20 fixed objects.

These tests verify that package changes do not unexpectedly alter predictions produced by the released model.

During package verification, the reproduced numerical results agreed with the original stored results to better than:

`|Delta| < 5e-7`

for the evaluated metrics.

## 10. Cross Survey Validation

Model 1 was examined outside the native SDSS and WISE domain using KiDS based experiments.

Validation products are stored under:

`validation/model1/`

For an SDSS to KiDS transfer experiment using a common 10 feature representation, XGBoost produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.946548 |
| Balanced accuracy | 0.945002 |
| Macro F1 | 0.939456 |
| ROC AUC | 0.984414 |

For the reverse KiDS to SDSS experiment:

| Metric | Value |
|---|---:|
| Accuracy | 0.886526 |
| Balanced accuracy | 0.891594 |
| Macro F1 | 0.880202 |
| ROC AUC | 0.947709 |

These experiments evaluate survey transfer behaviour and are distinct from the native independent test evaluation.

They should not be interpreted as identical experimental conditions.

## 11. Survey Transfer

Applying a pretrained classifier to another survey requires caution.

Different surveys may differ in:

- filter transmission curves
- magnitude systems
- calibration
- depth
- photometric uncertainties
- source selection
- redshift distributions
- detection limits

For this reason, matching column names alone does not guarantee that a catalogue is in the same photometric domain as the original training sample.

External applications should be independently validated before classifications are used for scientific inference.

## 12. Required Input Structure

At minimum, an input table must contain the following columns:

```text id="j7f0gv"
u
g
r
i
z
W1
W2
```

Colour columns should not be supplied as replacements for these magnitudes.

SpecLess constructs the model colours internally to preserve the feature definition used by the released classifier.

## 13. Model Scope

Model 1 is intended for catalogue level photometric morphology classification.

The released model distinguishes only:

- Elliptical
- Spiral

It does not provide separate classes for:

- irregular galaxies
- mergers
- lenticular galaxies
- detailed Hubble subtypes
- morphological disturbance

Objects outside the two released classes may therefore receive one of the available binary labels even when they are not well represented by the training distribution.

## 14. Limitations

The current public Model 1 release should be interpreted with the following limitations:

1. The classifier is trained for a binary morphology problem.
2. Performance on the native test sample does not guarantee identical performance on another survey.
3. Survey dependent photometric differences can introduce domain shift.
4. The model does not use image morphology directly.
5. The model does not use spectroscopy during inference.
6. The released classifier requires the specified photometric bands.
7. Classification probabilities should not automatically be interpreted as perfectly calibrated astrophysical probabilities.
8. External survey applications require independent validation.

## 15. Related Repository Files

Classifier:

`src/specless_astro/morphology/classifier.py`

Feature construction:

`src/specless_astro/morphology/features.py`

Model artifact:

`src/specless_astro/resources/model1/model1_xgboost_final.json`

Model metadata:

`src/specless_astro/resources/model1/model1_model_info.json`

Frozen regression sample:

`tests/data/model1_reference_sample.csv`

Regression tests:

`tests/test_model1.py`

Validation directory:

`validation/model1/`

General data documentation:

`DATA.md`

Reproducibility documentation:

`REPRODUCIBILITY.md`

Results map:

`RESULTS_INDEX.md`

## 16. Version

This document describes Model 1 as distributed with:

`SpecLess v0.2.1`
