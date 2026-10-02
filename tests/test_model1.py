"""Automated tests for SpecLess Model 1."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from specless_astro import MorphologyClassifier
from specless_astro.morphology.features import (
    MODEL1_FEATURES,
    build_model1_features,
)


REFERENCE_FILE = (
    Path(__file__).parent
    / "data"
    / "model1_reference_sample.csv"
)


def test_feature_engineering_produces_28_features():
    """Seven photometric bands must generate the 28 Model 1 features."""

    data = pd.DataFrame(
        {
            "u": [18.0],
            "g": [17.0],
            "r": [16.5],
            "i": [16.2],
            "z": [16.0],
            "W1": [15.0],
            "W2": [14.8],
        }
    )

    features = build_model1_features(data)

    assert features.shape == (1, 28)
    assert features.columns.tolist() == MODEL1_FEATURES

    assert features.loc[0, "u_g"] == pytest.approx(1.0)
    assert features.loc[0, "g_r"] == pytest.approx(0.5)
    assert features.loc[0, "W1_W2"] == pytest.approx(0.2)


def test_missing_photometric_band_raises_error():
    """Inference must fail clearly when a required band is absent."""

    data = pd.DataFrame(
        {
            "u": [18.0],
            "g": [17.0],
            "r": [16.5],
            "i": [16.2],
            "z": [16.0],
            "W1": [15.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required photometric bands",
    ):
        build_model1_features(data)


def test_model1_reference_predictions():
    """Model 1 must reproduce the frozen reference predictions."""

    reference = pd.read_csv(REFERENCE_FILE)

    model = MorphologyClassifier()

    input_catalog = reference[
        [
            "sdss_objid",
            "ra",
            "dec",
            "u",
            "g",
            "r",
            "i",
            "z",
            "W1",
            "W2",
        ]
    ].copy()

    result = model.predict(input_catalog)

    np.testing.assert_array_equal(
        result["specless_class"].to_numpy(),
        reference["expected_class"].to_numpy(),
    )

    np.testing.assert_allclose(
        result["p_elliptical"].to_numpy(),
        reference["expected_p_elliptical"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        result["p_spiral"].to_numpy(),
        reference["expected_p_spiral"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_classifier_metadata():
    """Packaged metadata must describe the intended Model 1 release."""

    model = MorphologyClassifier()

    info = model.info()

    assert info["task"] == "Elliptical vs Spiral"
    assert info["primary_model"] == "XGBoost"
    assert info["n_features"] == 28
    assert info["class_mapping"] == {
        "0": "Elliptical",
        "1": "Spiral",
    }
