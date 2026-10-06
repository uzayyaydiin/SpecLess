import pandas as pd
from specless_astro import ActivityClassifier

catalog = pd.read_csv("catalog.csv")

model = ActivityClassifier()
result = model.predict(catalog)

print(
    result[
        [
            "specless_class",
            "specless_label",
            "p_seyfert",
            "p_starburst",
            "p_starforming",
            "p_liner",
            "confidence",
        ]
    ]
)
