import numpy as np
import pandas as pd


BASE_BANDS = [
    "u", "g", "r", "i", "z",
    "W1", "W2", "W3",
]


COLOR_FEATURES = [
    "u_g", "u_r", "u_i", "u_z", "u_W1", "u_W2", "u_W3",
    "g_r", "g_i", "g_z", "g_W1", "g_W2", "g_W3",
    "r_i", "r_z", "r_W1", "r_W2", "r_W3",
    "i_z", "i_W1", "i_W2", "i_W3",
    "z_W1", "z_W2", "z_W3",
    "W1_W2", "W1_W3", "W2_W3",
]


MODEL2_FEATURES = BASE_BANDS + COLOR_FEATURES


def build_activity_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 36 photometric features required by SpecLess Model 2.

    Required input bands
    --------------------
    u, g, r, i, z, W1, W2, W3

    Returns
    -------
    pandas.DataFrame
        Eight magnitudes plus 28 colour features.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        band for band in BASE_BANDS
        if band not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required photometric bands: "
            + ", ".join(missing)
        )

    base = data[BASE_BANDS].copy()

    if base.isna().any().any():
        raise ValueError(
            "Input photometry contains missing values."
        )

    if not np.isfinite(base.to_numpy(dtype=float)).all():
        raise ValueError(
            "Input photometry contains non-finite values."
        )

    features = base.copy()

    for colour in COLOR_FEATURES:
        band1, band2 = colour.split("_")
        features[colour] = (
            features[band1] - features[band2]
        )

    return features[MODEL2_FEATURES]
