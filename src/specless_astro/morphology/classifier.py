"""Morphological galaxy classification with SpecLess Model 1."""

from __future__ import annotations

import json
from importlib.resources import as_file, files

import pandas as pd
import xgboost as xgb

from .features import BASE_BANDS, build_model1_features


class MorphologyClassifier:
    """
    SpecLess Model 1 classifier for elliptical and spiral galaxies.

    The classifier requires seven photometric magnitudes:

        u, g, r, i, z, W1, W2

    The remaining 21 colour features are generated automatically.
    """

    def __init__(self) -> None:
        self.metadata = self._load_metadata()
        self.model = self._load_model()

        self.class_mapping = self.metadata["class_mapping"]
        self.features = self.metadata["features"]

        self._validate_model_features()

    @staticmethod
    def _load_metadata() -> dict:
        """Load Model 1 metadata distributed with SpecLess."""

        metadata_path = files("specless_astro").joinpath(
            "resources",
            "model1",
            "model1_model_info.json",
        )

        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_model() -> xgb.XGBClassifier:
        """Load the pretrained Model 1 XGBoost classifier."""

        model_resource = files("specless_astro").joinpath(
            "resources",
            "model1",
            "model1_xgboost_final.json",
        )

        model = xgb.XGBClassifier()

        with as_file(model_resource) as model_path:
            model.load_model(str(model_path))

        return model

    def _validate_model_features(self) -> None:
        """Check consistency between metadata and the trained model."""

        model_features = self.model.get_booster().feature_names

        if model_features is None:
            return

        if list(model_features) != list(self.features):
            raise RuntimeError(
                "The feature order stored in the trained model does not "
                "match the SpecLess Model 1 metadata."
            )

    @property
    def required_bands(self) -> list[str]:
        """Photometric bands required for inference."""

        return BASE_BANDS.copy()

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict galaxy morphology.

        Parameters
        ----------
        data
            Pandas DataFrame containing at least:
            u, g, r, i, z, W1, W2.

        Returns
        -------
        pandas.DataFrame
            Copy of the input catalogue with SpecLess predictions,
            class probabilities, and confidence appended.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "SpecLess expects a pandas.DataFrame as input."
            )

        features = build_model1_features(data)

        predicted_class = self.model.predict(features).astype(int)
        probabilities = self.model.predict_proba(features)

        result = data.copy()

        result["specless_class"] = predicted_class
        result["specless_label"] = [
            self.class_mapping[str(value)]
            for value in predicted_class
        ]

        result["p_elliptical"] = probabilities[:, 0]
        result["p_spiral"] = probabilities[:, 1]
        result["confidence"] = probabilities.max(axis=1)

        return result

    def info(self) -> dict:
        """Return metadata describing the pretrained classifier."""

        return self.metadata.copy()
