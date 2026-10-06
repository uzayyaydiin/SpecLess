<p align="center">
  <img src="https://raw.githubusercontent.com/uzayyaydiin/SpecLess/main/docs/assets/specless-logo.png" width="420" alt="SpecLess logo">
</p>

<h1 align="center">SpecLess</h1>

<p align="center">
  <strong>Photometry Based Galaxy Classification</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/specless-astro/">
    <img src="https://img.shields.io/pypi/v/specless-astro" alt="PyPI">
  </a>
  <a href="https://github.com/uzayyaydiin/SpecLess/actions/workflows/tests.yml">
    <img src="https://github.com/uzayyaydiin/SpecLess/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/specless-astro" alt="Python versions">
  <img src="https://img.shields.io/github/license/uzayyaydiin/SpecLess" alt="License">
</p>

## Overview

**SpecLess** is an open-source Python package for photometry-based machine-learning classification of galaxies.

The package provides pretrained classifiers and automated photometric feature construction for galaxy classification without requiring spectroscopy at inference time.

The current release contains two classification models:

- **Model 1:** Elliptical and Spiral galaxy morphology
- **Model 2:** Seyfert, Starburst, Star-forming, and LINER activity classification

The package is distributed on PyPI as `specless-astro` and imported in Python as `specless_astro`.

## Surveys and Validation

SpecLess was developed and evaluated using photometric data from multiple astronomical surveys and catalogues.

Current development and validation work includes:

- **SDSS**  optical photometry and spectroscopic ground-truth samples
- **AllWISE**  mid-infrared photometry used by the pretrained classifiers
- **KiDS**  external optical survey validation
- **VIKING**  near-infrared photometry used together with KiDS in cross-survey validation
- **DESI Legacy Surveys**  external survey testing and cross-survey portability analysis

SpecLess is designed for photometry-based galaxy classification across heterogeneous survey data, with scalability toward upcoming large imaging datasets such as **LSST**.

## Installation

Install the public release from PyPI:

```bash
pip install specless-astro
```

The two classifiers can then be imported with:

```python
from specless_astro import MorphologyClassifier, ActivityClassifier
```

## Model 1: Galaxy Morphology

Model 1 performs binary morphological classification:

- Elliptical
- Spiral

The pretrained classifier uses XGBoost.

### Required photometry

Seven photometric bands are required:

- SDSS `u`, `g`, `r`, `i`, `z`
- WISE `W1`, `W2`

SpecLess automatically constructs 21 colour features from the seven input magnitudes, producing a total of 28 model features.

### Data

| Sample | Number of galaxies |
|---|---:|
| Training sample | 142,212 |
| Independent test sample | 35,554 |
| Elliptical galaxies in test sample | 12,888 |
| Spiral galaxies in test sample | 22,666 |

### Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.9375 |
| Balanced accuracy | 0.9388 |
| Macro F1 | 0.9332 |
| ROC AUC | 0.9826 |
| Elliptical recall | 0.9434 |
| Spiral recall | 0.9341 |

### Example

```python
import pandas as pd
from specless_astro import MorphologyClassifier

catalog = pd.read_csv("catalog.csv")

model = MorphologyClassifier()
result = model.predict(catalog)

print(
    result[
        [
            "specless_label",
            "p_elliptical",
            "p_spiral",
            "confidence",
        ]
    ]
)
```

## Model 2: Galaxy Activity Classification

Model 2 performs four-class activity classification:

- Seyfert
- Starburst
- Star-forming
- LINER

The pretrained classifier uses XGBoost.

### Required photometry

Eight photometric bands are required:

- SDSS `u`, `g`, `r`, `i`, `z`
- WISE `W1`, `W2`, `W3`

SpecLess automatically constructs 28 colour features from the eight input magnitudes, producing a total of 36 model features.

### Data

The spectroscopically labelled parent sample contains 133,886 galaxies.

The final SDSS and WISE photometric sample contains 84,652 galaxies:

| Class | Number of galaxies |
|---|---:|
| Star-forming | 37,518 |
| Starburst | 36,924 |
| Seyfert | 5,990 |
| LINER | 4,220 |

The independent test sample contains 16,931 galaxies.

### Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.8217 |
| Balanced accuracy | 0.7865 |
| Macro F1 | 0.7637 |
| Macro ROC AUC | 0.9572 |

### Example

```python
import pandas as pd
from specless_astro import ActivityClassifier

catalog = pd.read_csv("catalog.csv")

model = ActivityClassifier()
result = model.predict(catalog)

print(
    result[
        [
            "specless_label",
            "p_seyfert",
            "p_starburst",
            "p_starforming",
            "p_liner",
            "confidence",
        ]
    ]
)
```

## Cross-Survey Validation

The repository contains validation products associated with external and cross-survey experiments.

These include:

- KiDS and VIKING validation products
- SDSS and WISE to KiDS and VIKING transfer experiments
- feature-importance products
- SHAP-based interpretation products

For Model 2, the SDSS and WISE trained XGBoost classifier obtained the following results in the KiDS and VIKING external validation experiment:

| Metric | Score |
|---|---:|
| Accuracy | 0.7935 |
| Balanced accuracy | 0.7586 |
| Macro F1 | 0.7528 |
| Macro ROC AUC | 0.9533 |

## Repository Structure

```text
SpecLess/
├── src/
│   └── specless_astro/
│       ├── morphology/
│       ├── activity/
│       └── resources/
├── tests/
├── validation/
│   ├── model1/
│   └── model2/
├── docs/
│   └── assets/
├── pyproject.toml
├── LICENSE
└── README.md
```

## Testing

The repository contains regression tests for both classifiers, photometric feature construction, required input handling, model metadata, and frozen reference predictions.

Tests are also run through GitHub Actions.

## Data and Reproducibility

Large survey catalogues used during model development are not distributed directly with the Python package.

The public repository contains the pretrained model files, feature definitions, model metadata, compact reference samples, regression tests, validation metrics, cross-survey results, and interpretation products required for software use and verification.

## Documentation

Additional model and validation documentation is available in the [`docs`](docs) and [`validation`](validation) directories.

## Citation

The formal citation for SpecLess will be added following publication of the associated scientific paper.

## Author

**Uzay Aydın**  
Astronomy and Space Sciences  
Erciyes University, Türkiye

## License

SpecLess is distributed under the **BSD 3-Clause License**.

See [`LICENSE`](LICENSE) for the full license text.
