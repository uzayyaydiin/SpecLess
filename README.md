<p align="center">
  <img src="https://raw.githubusercontent.com/uzayyaydiin/SpecLess/main/docs/assets/specless-logo.png" width="420" alt="SpecLess logo">
</p>

# SpecLess

**SpecLess** is an open-source Python framework for photometry-based machine-learning classification of galaxies.

The project is designed to provide reusable pretrained models, photometric feature-engineering tools, survey-validation workflows, and a collaborative framework for galaxy classification without requiring spectroscopy at inference time.

## Current status

SpecLess is currently under active development.

### Model 1: Galaxy Morphology

Binary classification:

- Elliptical
- Spiral

Primary classifier:

- XGBoost

Required photometric bands:

- SDSS `u`, `g`, `r`, `i`, `z`
- WISE `W1`, `W2`

SpecLess automatically constructs the 21 colour features used by the pretrained classifier, producing a total of 28 photometric features.

The current Model 1 classifier was trained on 142,212 galaxies and evaluated on an independent test sample of 35,554 galaxies.

## Development installation

Clone the repository and install in editable mode:

```bash
pip install -e .
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

---


```bash
cat > LICENSE <<'EOF'
BSD 3-Clause License

Copyright (c) 2026, Uzay Aydın
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
