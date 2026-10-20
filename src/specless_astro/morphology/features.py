"""Feature engineering utilities for SpecLess morphology classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# SURVEY DEFINITIONS
# ============================================================

SURVEY_ALIASES = {

    "sdss":
        "sdss",

    "sdss_wise":
        "sdss",

    "sdss+wise":
        "sdss",

    "sdss+allwise":
        "sdss",

    "allwise":
        "sdss",


    "kids":
        "kids",

    "kids_viking":
        "kids",

    "kids+viking":
        "kids",

    "viking":
        "kids",
}


SURVEY_BANDS = {

    "sdss": [
        "u",
        "g",
        "r",
        "i",
        "z",
        "W1",
        "W2",
    ],

    "kids": [
        "u",
        "g",
        "r",
        "i",
        "Z",
        "Y",
        "J",
        "H",
        "Ks",
    ],
}


# ============================================================
# SURVEY NORMALISATION
# ============================================================


def normalise_morphology_survey(
    survey: str,
) -> str:
    """
    Convert a survey alias to the canonical
    SpecLess morphology survey key.
    """

    if not isinstance(
        survey,
        str,
    ):

        raise TypeError(
            "survey must be a string."
        )

    key = (
        survey
        .strip()
        .lower()
    )

    if key not in SURVEY_ALIASES:

        raise ValueError(
            "Unsupported morphology survey. "
            "Supported surveys are "
            "'sdss' and 'kids'."
        )

    return (
        SURVEY_ALIASES[
            key
        ]
    )


# ============================================================
# REQUIRED BANDS
# ============================================================


def required_morphology_bands(
    survey: str,
) -> list[str]:
    """
    Return the required photometric bands
    for one morphology survey model.
    """

    key = (
        normalise_morphology_survey(
            survey
        )
    )

    return (
        SURVEY_BANDS[
            key
        ].copy()
    )


# ============================================================
# FEATURE NAMES
# ============================================================


def morphology_feature_names(
    survey: str,
) -> list[str]:
    """
    Return the complete feature order for
    one morphology model.
    """

    bands = (
        required_morphology_bands(
            survey
        )
    )

    colours = []

    for i, band1 in enumerate(
        bands
    ):

        for band2 in bands[
            i + 1:
        ]:

            colours.append(
                f"{band1}_{band2}"
            )

    return (
        bands
        + colours
    )


# ============================================================
# INPUT VALIDATION
# ============================================================


def validate_morphology_photometry(
    data: pd.DataFrame,
    survey: str,
) -> None:
    """
    Validate input photometry before
    feature construction.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):

        raise TypeError(
            "SpecLess expects a pandas.DataFrame "
            "as input."
        )

    bands = (
        required_morphology_bands(
            survey
        )
    )

    missing = [
        band
        for band in bands
        if band not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required photometric bands: "
            + ", ".join(
                missing
            )
        )

    values = (
        data[
            bands
        ]
    )

    if values.isna().any().any():

        bad_columns = (
            values.columns[
                values.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "NaN values detected in required bands: "
            + ", ".join(
                bad_columns
            )
        )

    numeric_values = (
        values.to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            "Infinite or non-finite values "
            "detected in photometry."
        )


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================


def build_morphology_features(
    data: pd.DataFrame,
    survey: str = "sdss",
) -> pd.DataFrame:
    """
    Construct the survey-specific morphology
    feature matrix.
    """

    key = (
        normalise_morphology_survey(
            survey
        )
    )

    validate_morphology_photometry(
        data,
        key,
    )

    bands = (
        SURVEY_BANDS[
            key
        ]
    )

    features = (
        data[
            bands
        ]
        .astype(float)
        .copy()
    )

    for i, band1 in enumerate(
        bands
    ):

        for band2 in bands[
            i + 1:
        ]:

            feature_name = (
                f"{band1}_{band2}"
            )

            features[
                feature_name
            ] = (
                features[
                    band1
                ]
                - features[
                    band2
                ]
            )

    ordered_features = (
        morphology_feature_names(
            key
        )
    )

    return (
        features[
            ordered_features
        ]
    )


# ============================================================
# BACKWARD COMPATIBILITY WITH SPECLESS 0.2.x
# ============================================================

BASE_BANDS = (
    SURVEY_BANDS[
        "sdss"
    ].copy()
)

MODEL1_FEATURES = (
    morphology_feature_names(
        "sdss"
    )
)

COLOUR_FEATURES = (
    MODEL1_FEATURES[
        len(BASE_BANDS):
    ]
)


def validate_photometry(
    data: pd.DataFrame,
) -> None:
    """
    Backward-compatible SDSS morphology
    photometry validation.
    """

    validate_morphology_photometry(
        data,
        survey="sdss",
    )


def build_model1_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backward-compatible SDSS morphology
    feature builder.
    """

    return (
        build_morphology_features(
            data,
            survey="sdss",
        )
    )
