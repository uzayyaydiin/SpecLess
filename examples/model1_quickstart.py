import pandas as pd
from specless_astro import MorphologyClassifier

catalog = pd.read_csv("catalog.csv")

model = MorphologyClassifier()
result = model.predict(catalog)

print(
    result[
        [
            "specless_class",
            "specless_label",
            "p_elliptical",
            "p_spiral",
            "confidence",
        ]
    ]
)
