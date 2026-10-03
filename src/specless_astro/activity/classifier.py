import json
from importlib import resources

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import (
    BASE_BANDS,
    MODEL2_FEATURES,
    build_activity_features,
)


CLASS_NAMES = {
    0: "SEYFERT",
    1: "STARBURST",
    2: "STARFORMING",
    3: "LINER",
}


class ActivityClassifier:
    """
    SpecLess Model 2 photometric galaxy activity classifier.

    Classes
    -------
    SEYFERT
    STARBURST
    STARFORMING
    LINER
    """

    def __init__(self):
        self.required_bands = BASE_BANDS.copy()
        self.features = MODEL2_FEATURES.copy()

        self._metadata = self._load_metadata()
        self._model = self._load_model()

    @staticmethod
    def _resource_dir():
        return resources.files(
            "specless_astro.resources.model2"
        )

    def _load_metadata(self):
        path = (
            self._resource_dir()
            / "model2_model_info.json"
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _load_model(self):
        path = (
            self._resource_dir()
            / "model2_xgboost_final.json"
        )

        model = xgb.Booster()
        model.load_model(str(path))

        return model

    def predict(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Classify galaxies using photometric measurements.

        Parameters
        ----------
        catalog : pandas.DataFrame
            Must contain:
            u, g, r, i, z, W1, W2, W3

        Returns
        -------
        pandas.DataFrame
            Original catalogue with SpecLess predictions
            and class probabilities.
        """

        features = build_activity_features(catalog)

        matrix = xgb.DMatrix(
            features,
            feature_names=self.features,
        )

        probabilities = self._model.predict(matrix)

        predicted_index = np.argmax(
            probabilities,
            axis=1,
        )

        result = catalog.copy()

        result["specless_class"] = predicted_index

        result["specless_label"] = [
            CLASS_NAMES[int(i)]
            for i in predicted_index
        ]

        result["p_seyfert"] = probabilities[:, 0]
        result["p_starburst"] = probabilities[:, 1]
        result["p_starforming"] = probabilities[:, 2]
        result["p_liner"] = probabilities[:, 3]

        result["confidence"] = np.max(
            probabilities,
            axis=1,
        )

        return result

    def info(self):
        """
        Return Model 2 metadata.
        """
        return self._metadata
