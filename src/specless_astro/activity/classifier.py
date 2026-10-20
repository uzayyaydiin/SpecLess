"""Multi-survey galaxy activity classification with SpecLess."""

from __future__ import annotations

import json
from importlib.resources import (
    as_file,
    files,
)

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import (
    build_activity_features,
    normalise_activity_survey,
    required_activity_columns,
)


PROBABILITY_COLUMN_MAP = {

    "SEYFERT":
        "p_seyfert",

    "STARBURST":
        "p_starburst",

    "STARFORMING":
        "p_starforming",

    "LINER":
        "p_liner",
}


class ActivityClassifier:
    """
    SpecLess photometric galaxy activity classifier.

    Supported survey models
    -----------------------
    sdss
        SDSS + AllWISE activity model.

    desi
        DESI Legacy Surveys activity model.

    Parameters
    ----------
    survey : str, default="sdss"
        Survey-specific photometric model.
    """

    SUPPORTED_SURVEYS = (
        "sdss",
        "desi",
    )


    def __init__(
        self,
        survey: str = "sdss",
    ) -> None:

        self.survey = (
            normalise_activity_survey(
                survey
            )
        )

        self._metadata = (
            self._load_metadata()
        )

        self.features = list(
            self._metadata[
                "features"
            ]
        )

        self.class_mapping = {
            int(key):
                value

            for key, value
            in self._metadata[
                "class_mapping"
            ].items()
        }

        self.model_api = (
            self._metadata.get(
                "model_api",
                "booster_multiclass",
            )
        )

        self._model = (
            self._load_model()
        )

        self._validate_model_features()


    @staticmethod
    def available_surveys() -> list[str]:
        """
        Return supported activity survey models.
        """

        return [
            "sdss",
            "desi",
        ]


    def _resource_dir(
        self,
    ):

        return (
            files(
                "specless_astro"
            )
            .joinpath(
                "resources",
                "activity",
                self.survey,
            )
        )


    def _load_metadata(
        self,
    ) -> dict:

        metadata_resource = (
            self._resource_dir()
            .joinpath(
                "metadata.json"
            )
        )

        with metadata_resource.open(
            "r",
            encoding="utf-8",
        ) as file:

            return (
                json.load(
                    file
                )
            )


    def _load_model(
        self,
    ):

        model_resource = (
            self._resource_dir()
            .joinpath(
                "model.json"
            )
        )


        if self.model_api in {
            "booster",
            "booster_multiclass",
        }:

            model = (
                xgb.Booster()
            )

            with as_file(
                model_resource
            ) as model_path:

                model.load_model(
                    str(
                        model_path
                    )
                )

            return model


        if (
            self.model_api
            == "xgb_classifier"
        ):

            model = (
                xgb.XGBClassifier()
            )

            with as_file(
                model_resource
            ) as model_path:

                model.load_model(
                    str(
                        model_path
                    )
                )

            return model


        raise RuntimeError(
            "Unsupported activity model API: "
            f"{self.model_api}"
        )


    def _validate_model_features(
        self,
    ) -> None:
        """
        Check model and metadata consistency.
        """

        expected_count = (
            len(
                self.features
            )
        )


        if self.model_api in {
            "booster",
            "booster_multiclass",
        }:

            model_count = (
                self._model
                .num_features()
            )

            model_features = (
                self._model
                .feature_names
            )


        elif (
            self.model_api
            == "xgb_classifier"
        ):

            model_count = (
                self._model
                .n_features_in_
            )

            model_features = (
                self._model
                .get_booster()
                .feature_names
            )


        else:

            raise RuntimeError(
                "Unknown activity model API."
            )


        if (
            int(
                model_count
            )
            != expected_count
        ):

            raise RuntimeError(
                "Activity model feature count "
                "does not match metadata."
            )


        if (
            model_features is not None
            and list(
                model_features
            )
            != list(
                self.features
            )
        ):

            raise RuntimeError(
                "Activity model feature order "
                "does not match metadata."
            )


    @property
    def required_columns(
        self,
    ) -> list[str]:

        return (
            required_activity_columns(
                self.survey
            )
        )


    @property
    def required_bands(
        self,
    ) -> list[str]:
        """
        Backward-compatible alias.

        For DESI this contains both
        magnitude and magnitude-error columns.
        """

        return (
            self.required_columns
        )


    def _predict_probabilities(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:

        if self.model_api in {
            "booster",
            "booster_multiclass",
        }:

            matrix = (
                xgb.DMatrix(
                    features,
                    feature_names=
                        self.features,
                )
            )

            probabilities = (
                self._model
                .predict(
                    matrix
                )
            )


        elif (
            self.model_api
            == "xgb_classifier"
        ):

            probabilities = (
                self._model
                .predict_proba(
                    features
                )
            )


        else:

            raise RuntimeError(
                "Unsupported activity model API."
            )


        probabilities = (
            np.asarray(
                probabilities,
                dtype=float,
            )
        )


        if (
            probabilities.ndim
            != 2
        ):

            raise RuntimeError(
                "Activity model did not return "
                "a two-dimensional probability matrix."
            )


        if (
            probabilities.shape[1]
            != len(
                self.class_mapping
            )
        ):

            raise RuntimeError(
                "Probability matrix class count "
                "does not match metadata."
            )


        return (
            probabilities
        )


    def predict_proba(
        self,
        catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return class probabilities.
        """

        features = (
            build_activity_features(
                catalog,
                survey=self.survey,
            )
        )

        features = (
            features[
                self.features
            ]
        )

        probabilities = (
            self._predict_probabilities(
                features
            )
        )


        output = {}


        for class_index in sorted(
            self.class_mapping
        ):

            class_name = (
                self.class_mapping[
                    class_index
                ]
            )

            if (
                class_name
                not in
                PROBABILITY_COLUMN_MAP
            ):

                raise RuntimeError(
                    "Unknown activity class in metadata: "
                    f"{class_name}"
                )

            column_name = (
                PROBABILITY_COLUMN_MAP[
                    class_name
                ]
            )

            output[
                column_name
            ] = (
                probabilities[
                    :,
                    class_index
                ]
            )


        return (
            pd.DataFrame(
                output,
                index=catalog.index,
            )
        )


    def predict(
        self,
        catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Classify galaxies from photometry.
        """

        features = (
            build_activity_features(
                catalog,
                survey=self.survey,
            )
        )

        features = (
            features[
                self.features
            ]
        )

        probabilities = (
            self._predict_probabilities(
                features
            )
        )

        predicted_index = (
            np.argmax(
                probabilities,
                axis=1,
            )
            .astype(int)
        )


        result = (
            catalog.copy()
        )


        result[
            "specless_class"
        ] = (
            predicted_index
        )


        result[
            "specless_label"
        ] = [

            self.class_mapping[
                int(
                    index
                )
            ]

            for index
            in predicted_index
        ]


        for class_index in sorted(
            self.class_mapping
        ):

            class_name = (
                self.class_mapping[
                    class_index
                ]
            )

            output_column = (
                PROBABILITY_COLUMN_MAP[
                    class_name
                ]
            )

            result[
                output_column
            ] = (
                probabilities[
                    :,
                    class_index
                ]
            )


        result[
            "confidence"
        ] = (
            probabilities.max(
                axis=1
            )
        )


        return result


    def info(
        self,
    ) -> dict:
        """
        Return metadata for the selected
        activity model.
        """

        return (
            self._metadata.copy()
        )
