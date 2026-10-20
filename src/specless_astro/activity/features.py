"""Feature engineering utilities for SpecLess activity classifiers."""

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


    "desi":
        "desi",

    "legacy":
        "desi",

    "desi_legacy":
        "desi",

    "desi legacy":
        "desi",

    "desi legacy surveys":
        "desi",
}


# ============================================================
# SDSS + ALLWISE
# ============================================================


SDSS_BASE_BANDS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "W1",
    "W2",
    "W3",
]


def _sdss_feature_names() -> list[str]:

    colours = []

    for i, band1 in enumerate(
        SDSS_BASE_BANDS
    ):

        for band2 in (
            SDSS_BASE_BANDS[
                i + 1:
            ]
        ):

            colours.append(
                f"{band1}_{band2}"
            )

    return (
        SDSS_BASE_BANDS
        + colours
    )


SDSS_FEATURES = (
    _sdss_feature_names()
)


# ============================================================
# DESI LEGACY SURVEYS
# ============================================================


DESI_MAG_COLUMNS = [
    "mag0_g",
    "mag0_r",
    "mag0_z",
    "mag0_w1",
    "mag0_w2",
    "mag0_w3",
]


DESI_ERROR_COLUMNS = [
    "mag_err0_g",
    "mag_err0_r",
    "mag_err0_z",
    "mag_err0_w1",
    "mag_err0_w2",
    "mag_err0_w3",
]


DESI_COLOUR_FEATURES = [
    "g-r_0",
    "g-z_0",
    "g-w1_0",
    "g-w2_0",
    "g-w3_0",

    "r-z_0",
    "r-w1_0",
    "r-w2_0",
    "r-w3_0",

    "z-w1_0",
    "z-w2_0",
    "z-w3_0",

    "w1-w2_0",
    "w1-w3_0",
    "w2-w3_0",
]


DESI_FEATURES = (
    DESI_MAG_COLUMNS
    + DESI_ERROR_COLUMNS
    + DESI_COLOUR_FEATURES
)


# ============================================================
# SURVEY NORMALISATION
# ============================================================


def normalise_activity_survey(
    survey: str,
) -> str:
    """
    Convert an activity survey alias
    to the canonical SpecLess survey key.
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
            "Unsupported activity survey. "
            "Supported surveys are "
            "'sdss' and 'desi'."
        )

    return (
        SURVEY_ALIASES[
            key
        ]
    )


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================


def required_activity_columns(
    survey: str,
) -> list[str]:
    """
    Return required catalogue columns
    for one activity model.
    """

    key = (
        normalise_activity_survey(
            survey
        )
    )

    if key == "sdss":

        return (
            SDSS_BASE_BANDS.copy()
        )

    return (
        DESI_MAG_COLUMNS
        + DESI_ERROR_COLUMNS
    )


# ============================================================
# GENERIC VALIDATION
# ============================================================


def _validate_columns(
    data: pd.DataFrame,
    required_columns: list[str],
) -> None:

    if not isinstance(
        data,
        pd.DataFrame,
    ):

        raise TypeError(
            "SpecLess expects a pandas.DataFrame "
            "as input."
        )

    missing = [
        column
        for column
        in required_columns
        if column
        not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required photometric columns: "
            + ", ".join(
                missing
            )
        )

    values = (
        data[
            required_columns
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
            "NaN values detected in required columns: "
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
# SDSS FEATURE BUILDER
# ============================================================


def build_sdss_activity_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the 36 features used by the
    SDSS+AllWISE activity model.
    """

    _validate_columns(
        data,
        SDSS_BASE_BANDS,
    )

    features = (
        data[
            SDSS_BASE_BANDS
        ]
        .astype(float)
        .copy()
    )

    for i, band1 in enumerate(
        SDSS_BASE_BANDS
    ):

        for band2 in (
            SDSS_BASE_BANDS[
                i + 1:
            ]
        ):

            name = (
                f"{band1}_{band2}"
            )

            features[
                name
            ] = (
                features[
                    band1
                ]
                - features[
                    band2
                ]
            )

    return (
        features[
            SDSS_FEATURES
        ]
    )


# ============================================================
# DESI FEATURE BUILDER
# ============================================================


def build_desi_activity_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the 27 features used by the
    DESI Legacy activity model.
    """

    required = (
        DESI_MAG_COLUMNS
        + DESI_ERROR_COLUMNS
    )

    _validate_columns(
        data,
        required,
    )

    features = (
        data[
            required
        ]
        .astype(float)
        .copy()
    )


    magnitude_map = {

        "g":
            features[
                "mag0_g"
            ],

        "r":
            features[
                "mag0_r"
            ],

        "z":
            features[
                "mag0_z"
            ],

        "w1":
            features[
                "mag0_w1"
            ],

        "w2":
            features[
                "mag0_w2"
            ],

        "w3":
            features[
                "mag0_w3"
            ],
    }


    pairs = [

        ("g", "r"),
        ("g", "z"),

        ("g", "w1"),
        ("g", "w2"),
        ("g", "w3"),

        ("r", "z"),

        ("r", "w1"),
        ("r", "w2"),
        ("r", "w3"),

        ("z", "w1"),
        ("z", "w2"),
        ("z", "w3"),

        ("w1", "w2"),
        ("w1", "w3"),
        ("w2", "w3"),
    ]


    for band1, band2 in pairs:

        feature_name = (
            f"{band1}-{band2}_0"
        )

        features[
            feature_name
        ] = (
            magnitude_map[
                band1
            ]
            - magnitude_map[
                band2
            ]
        )


    return (
        features[
            DESI_FEATURES
        ]
    )


# ============================================================
# GENERAL ACTIVITY FEATURE BUILDER
# ============================================================


def build_activity_features(
    data: pd.DataFrame,
    survey: str = "sdss",
) -> pd.DataFrame:
    """
    Build survey-specific activity features.
    """

    key = (
        normalise_activity_survey(
            survey
        )
    )

    if key == "sdss":

        return (
            build_sdss_activity_features(
                data
            )
        )

    return (
        build_desi_activity_features(
            data
        )
    )


# ============================================================
# BACKWARD COMPATIBILITY WITH SPECLESS 0.2.x
# ============================================================


BASE_BANDS = (
    SDSS_BASE_BANDS.copy()
)

MODEL2_FEATURES = (
    SDSS_FEATURES.copy()
)

COLOR_FEATURES = (
    SDSS_FEATURES[
        len(
            SDSS_BASE_BANDS
        ):
    ]
)
