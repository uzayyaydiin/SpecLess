# SpecLess Model 2 — Photometric Activity Classification

## Overview

SpecLess Model 2 classifies galaxies into four activity-related classes using photometric information only at inference time:

- Seyfert
- Starburst
- Star-forming
- LINER

The native model uses SDSS optical photometry together with WISE mid-infrared photometry.

## Data

The spectroscopic ground-truth parent sample contains 133,886 galaxies:

- Star-forming: 60,000
- Starburst: 60,000
- Seyfert: 7,264
- LINER: 6,622

The final SDSS+WISE photometric sample contains 84,652 galaxies:

- Star-forming: 37,518
- Starburst: 36,924
- Seyfert: 5,990
- LINER: 4,220

## Input features

The public classifier requires eight magnitude columns:

- u, g, r, i, z
- W1, W2, W3

SpecLess automatically derives 28 colour features, producing 36 model features in total.

## Training

The catalogue is divided with a stratified 80/20 train-test split using random_state=42.

- Initial training sample: 67,721
- Independent test sample: 16,931

Training-only undersampling produces:

- Star-forming: 10,000
- Starburst: 10,000
- Seyfert: 4,792
- LINER: 3,376
- Total training sample: 28,168

Inverse-frequency sample weights are applied during XGBoost training.

## Native SDSS+WISE performance

### Random Forest

- Accuracy: 0.8275
- Balanced Accuracy: 0.7742
- Macro F1: 0.7661
- Macro ROC-AUC: 0.9574

### XGBoost

- Accuracy: 0.8217
- Balanced Accuracy: 0.7865
- Macro F1: 0.7637
- Macro ROC-AUC: 0.9572

The packaged ActivityClassifier uses the XGBoost model.

## KiDS/VIKING validation

The KiDS+VIKING validation catalogue contains 5,521 labelled galaxies.

SDSS+WISE to KiDS/VIKING external XGBoost performance:

- Accuracy: 0.7935
- Balanced Accuracy: 0.7586
- Macro F1: 0.7528
- Macro ROC-AUC: 0.9533

## Usage

```python
from specless_astro import ActivityClassifier

model = ActivityClassifier()
result = model.predict(catalog)
```

The input catalogue must contain u, g, r, i, z, W1, W2, and W3.

The returned table includes class labels, four class probabilities, and prediction confidence.

## Current scope

The current Model 2 release contains native SDSS+WISE classification and KiDS/VIKING validation products.

DESI/Legacy Survey validation is not included in the current Model 2 release.
