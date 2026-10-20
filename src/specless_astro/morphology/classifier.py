"""Multi-survey morphology classification with SpecLess."""

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
    build_morphology_features,
    normalise_morphology_survey,
    required_morphology_bands,
)


class MorphologyClassifier:
    """
    SpecLess photometric morphology classifier.

    Supported survey models
    -----------------------
    sdss
        SDSS + AllWISE morphology model.

    kids
        KiDS + VIKING morphology model.

    Parameters
    ----------
    survey : str, default="sdss"
        Survey-specific photometric model.
    """

    SUPPORTED_SURVEYS = (
        "sdss",
        "kids",
    )


    def __init__(
        self,
        survey: str = "sdss",
    ) -> None:

        self.survey = (
            normalise_morphology_survey(
                survey
            )
        )

        self.metadata = (
            self._load_metadata()
        )

        self.features = list(
            self.metadata[
                "features"
            ]
        )

        self.class_mapping = (
            self.metadata[
                "class_mapping"
            ]
        )

        self.model_api = (
            self.metadata.get(
                "model_api",
                "xgb_classifier",
            )
        )

        self.model = (
            self._load_model()
        )

        self._validate_model_features()


    @staticmethod
    def available_surveys() -> list[str]:
        """
        Return supported morphology
        survey models.
        """

        return [
            "sdss",
            "kids",
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
                "morphology",
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

        with as_file(
            model_resource
        ) as model_path:

            if (
                self.model_api
                == "xgb_classifier"
            ):

                model = (
                    xgb.XGBClassifier()
                )

                model.load_model(
                    str(
                        model_path
                    )
                )

                return model


            if (
                self.model_api
                == "booster_binary"
            ):

                model = (
                    xgb.Booster()
                )

                model.load_model(
                    str(
                        model_path
                    )
                )

                return model


        raise RuntimeError(
            "Unsupported morphology model API: "
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


        if (
            self.model_api
            == "xgb_classifier"
        ):

            model_count = (
                self.model
                .n_features_in_
            )

            model_features = (
                self.model
                .get_booster()
                .feature_names
            )


        elif (
            self.model_api
            == "booster_binary"
        ):

            model_count = (
                self.model
                .num_features()
            )

            model_features = (
                self.model
                .feature_names
            )


        else:

            raise RuntimeError(
                "Unknown morphology model API."
            )


        if (
            int(
                model_count
            )
            != expected_count
        ):

            raise RuntimeError(
                "Morphology model feature count "
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
                "Morphology model feature order "
                "does not match metadata."
            )


    @property
    def required_bands(
        self,
    ) -> list[str]:

        return (
            required_morphology_bands(
                self.survey
            )
        )


    @property
    def required_columns(
        self,
    ) -> list[str]:

        return (
            self.required_bands
        )


    def _predict_probabilities(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return probabilities in the order:

        column 0 = Elliptical
        column 1 = Spiral
        """

        if (
            self.model_api
            == "xgb_classifier"
        ):

            probabilities = (
                self.model
                .predict_proba(
                    features
                )
            )

            return (
                np.asarray(
                    probabilities,
                    dtype=float,
                )
            )


        matrix = (
            xgb.DMatrix(
                features,
                feature_names=
                    self.features,
            )
        )

        p_spiral = (
            self.model
            .predict(
                matrix
            )
        )

        p_spiral = (
            np.asarray(
                p_spiral,
                dtype=float,
            )
            .reshape(-1)
        )

        p_elliptical = (
            1.0
            - p_spiral
        )

        return (
            np.column_stack(
                [
                    p_elliptical,
                    p_spiral,
                ]
            )
        )


    def predict_proba(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return morphology class probabilities.
        """

        features = (
            build_morphology_features(
                data,
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

        return (
            pd.DataFrame(
                {
                    "p_elliptical":
                        probabilities[
                            :,
                            0
                        ],

                    "p_spiral":
                        probabilities[
                            :,
                            1
                        ],
                },
                index=data.index,
            )
        )


    def predict(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict elliptical or spiral morphology.
        """

        probabilities = (
            self.predict_proba(
                data
            )
        )

        probability_array = (
            probabilities[
                [
                    "p_elliptical",
                    "p_spiral",
                ]
            ]
            .to_numpy()
        )

        predicted_class = (
            np.argmax(
                probability_array,
                axis=1,
            )
            .astype(int)
        )

        result = (
            data.copy()
        )

        result[
            "specless_class"
        ] = predicted_class

        result[
            "specless_label"
        ] = [
            self.class_mapping[
                str(
                    value
                )
            ]
            for value
            in predicted_class
        ]

        result[
            "p_elliptical"
        ] = (
            probability_array[
                :,
                0
            ]
        )

        result[
            "p_spiral"
        ] = (
            probability_array[
                :,
                1
            ]
        )

        result[
            "confidence"
        ] = (
            probability_array.max(
                axis=1
            )
        )

        return result


    def info(
        self,
    ) -> dict:
        """
        Return metadata for the selected
        morphology model.
        """

        return (
            self.metadata.copy()
        )
