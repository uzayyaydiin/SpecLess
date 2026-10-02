"""Feature engineering utilities for SpecLess morphology models."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_BANDS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "W1",
    "W2",
]


COLOUR_FEATURES = [
    "u_g",
    "u_r",
    "u_i",
    "u_z",
    "u_W1",
    "u_W2",
    "g_r",
    "g_i",
    "g_z",
    "g_W1",
    "g_W2",
    "r_i",
    "r_z",
    "r_W1",
    "r_W2",
    "i_z",
    "i_W1",
    "i_W2",
    "z_W1",
    "z_W2",
    "W1_W2",
]


MODEL1_FEATURES = BASE_BANDS + COLOUR_FEATURES


def validate_photometry(data: pd.DataFrame) -> None:
    """Validate the photometric input required by Model 1."""

    missing = [band for band in BASE_BANDS if band not in data.columns]

    if missing:
        raise ValueError(
            "Missing required photometric bands: "
            + ", ".join(missing)
        )

    values = data[BASE_BANDS]

    if values.isna().any().any():
        bad_columns = values.columns[values.isna().any()].tolist()

        raise ValueError(
            "NaN values detected in required bands: "
            + ", ".join(bad_columns)
        )

    numeric_values = values.to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Infinite or non-finite values detected in photometry."
        )


def build_model1_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate the 28 photometric features used by Model 1."""

    validate_photometry(data)

    features = data[BASE_BANDS].astype(float).copy()

    for i, band1 in enumerate(BASE_BANDS):
        for band2 in BASE_BANDS[i + 1:]:
            feature_name = f"{band1}_{band2}"
            features[feature_name] = features[band1] - features[band2]

    return features[MODEL1_FEATURES]

