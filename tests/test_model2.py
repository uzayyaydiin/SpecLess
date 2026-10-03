from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from specless_astro import ActivityClassifier
from specless_astro.activity.features import build_activity_features


REFERENCE = (
    Path(__file__).parent
    / "data"
    / "model2_reference_sample.csv"
)


def test_model2_feature_engineering():
    df = pd.read_csv(REFERENCE)

    bands = df[
        ["u", "g", "r", "i", "z", "W1", "W2", "W3"]
    ]

    features = build_activity_features(bands)

    assert features.shape[1] == 36

    assert np.allclose(
        features["u_g"],
        bands["u"] - bands["g"],
    )

    assert np.allclose(
        features["W1_W3"],
        bands["W1"] - bands["W3"],
    )


def test_model2_missing_band():
    df = pd.read_csv(REFERENCE)

    bands = df[
        ["u", "g", "r", "i", "z", "W1", "W2"]
    ]

    with pytest.raises(ValueError):
        build_activity_features(bands)


def test_model2_reference_predictions():
    df = pd.read_csv(REFERENCE)

    bands = df[
        ["u", "g", "r", "i", "z", "W1", "W2", "W3"]
    ]

    model = ActivityClassifier()
    result = model.predict(bands)

    np.testing.assert_array_equal(
        result["specless_class"].to_numpy(),
        df["expected_class"].to_numpy(),
    )

    np.testing.assert_array_equal(
        result["specless_label"].to_numpy(),
        df["expected_label"].to_numpy(),
    )

    np.testing.assert_allclose(
        result["p_seyfert"].to_numpy(),
        df["expected_p_seyfert"].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        result["p_starburst"].to_numpy(),
        df["expected_p_starburst"].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        result["p_starforming"].to_numpy(),
        df["expected_p_starforming"].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        result["p_liner"].to_numpy(),
        df["expected_p_liner"].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_model2_metadata():
    model = ActivityClassifier()
    info = model.info()

    assert info["n_features"] == 36

    assert info["classes"] == {
        "0": "SEYFERT",
        "1": "STARBURST",
        "2": "STARFORMING",
        "3": "LINER",
    }

    assert np.isclose(
        info["test_metrics"]["accuracy"],
        0.8216880278778572,
    )

    assert np.isclose(
        info["test_metrics"]["balanced_accuracy"],
        0.7865243975574545,
    )

    assert np.isclose(
        info["test_metrics"]["macro_f1"],
        0.7637061003679557,
    )

    assert np.isclose(
        info["test_metrics"]["macro_roc_auc"],
        0.9571918660268934,
    )
