import numpy as np
import pandas as pd
import pytest

import specless_astro

from specless_astro import (
    ActivityClassifier,
    MorphologyClassifier,
)


def test_version():

    assert (
        specless_astro.__version__
        == "0.3.0"
    )


def test_available_surveys():

    assert (
        MorphologyClassifier
        .available_surveys()
        == ["sdss", "kids"]
    )

    assert (
        ActivityClassifier
        .available_surveys()
        == ["sdss", "desi"]
    )


def test_default_surveys():

    assert (
        MorphologyClassifier()
        .survey
        == "sdss"
    )

    assert (
        ActivityClassifier()
        .survey
        == "sdss"
    )


def test_sdss_morphology_initialises():

    model = (
        MorphologyClassifier(
            survey="sdss"
        )
    )

    assert (
        len(
            model.features
        )
        == 28
    )

    assert (
        model.required_bands
        == [
            "u",
            "g",
            "r",
            "i",
            "z",
            "W1",
            "W2",
        ]
    )


def test_kids_morphology_initialises():

    model = (
        MorphologyClassifier(
            survey="kids"
        )
    )

    assert (
        len(
            model.features
        )
        == 45
    )

    assert (
        model.required_bands
        == [
            "u",
            "g",
            "r",
            "i",
            "Z",
            "Y",
            "J",
            "H",
            "Ks",
        ]
    )


def test_kids_morphology_prediction():

    data = pd.DataFrame(
        {
            "u": [
                20.1,
                19.4,
            ],

            "g": [
                18.8,
                18.1,
            ],

            "r": [
                17.9,
                17.3,
            ],

            "i": [
                17.5,
                16.9,
            ],

            "Z": [
                17.3,
                16.7,
            ],

            "Y": [
                17.0,
                16.5,
            ],

            "J": [
                16.8,
                16.2,
            ],

            "H": [
                16.6,
                16.0,
            ],

            "Ks": [
                16.5,
                15.9,
            ],
        }
    )

    model = (
        MorphologyClassifier(
            survey="kids"
        )
    )

    result = (
        model.predict(
            data
        )
    )

    probability_sum = (
        result[
            [
                "p_elliptical",
                "p_spiral",
            ]
        ]
        .sum(
            axis=1
        )
        .to_numpy()
    )

    assert np.allclose(
        probability_sum,
        1.0,
        atol=1e-6,
    )

    assert {
        "specless_class",
        "specless_label",
        "p_elliptical",
        "p_spiral",
        "confidence",
    }.issubset(
        result.columns
    )


def test_sdss_activity_initialises():

    model = (
        ActivityClassifier(
            survey="sdss"
        )
    )

    assert (
        len(
            model.features
        )
        == 36
    )

    assert (
        model.required_columns
        == [
            "u",
            "g",
            "r",
            "i",
            "z",
            "W1",
            "W2",
            "W3",
        ]
    )


def test_desi_activity_initialises():

    model = (
        ActivityClassifier(
            survey="desi"
        )
    )

    assert (
        len(
            model.features
        )
        == 27
    )

    assert (
        model.required_columns
        == [
            "mag0_g",
            "mag0_r",
            "mag0_z",
            "mag0_w1",
            "mag0_w2",
            "mag0_w3",
            "mag_err0_g",
            "mag_err0_r",
            "mag_err0_z",
            "mag_err0_w1",
            "mag_err0_w2",
            "mag_err0_w3",
        ]
    )


def test_desi_activity_prediction():

    data = pd.DataFrame(
        {
            "mag0_g": [
                19.0,
                20.0,
            ],

            "mag0_r": [
                18.5,
                19.3,
            ],

            "mag0_z": [
                18.1,
                18.9,
            ],

            "mag0_w1": [
                17.5,
                18.2,
            ],

            "mag0_w2": [
                17.2,
                17.9,
            ],

            "mag0_w3": [
                16.5,
                17.0,
            ],

            "mag_err0_g": [
                0.03,
                0.04,
            ],

            "mag_err0_r": [
                0.03,
                0.04,
            ],

            "mag_err0_z": [
                0.03,
                0.04,
            ],

            "mag_err0_w1": [
                0.04,
                0.05,
            ],

            "mag_err0_w2": [
                0.05,
                0.06,
            ],

            "mag_err0_w3": [
                0.08,
                0.10,
            ],
        }
    )

    model = (
        ActivityClassifier(
            survey="desi"
        )
    )

    result = (
        model.predict(
            data
        )
    )

    probability_sum = (
        result[
            [
                "p_starforming",
                "p_starburst",
                "p_seyfert",
                "p_liner",
            ]
        ]
        .sum(
            axis=1
        )
        .to_numpy()
    )

    assert np.allclose(
        probability_sum,
        1.0,
        atol=1e-6,
    )

    assert {
        "specless_class",
        "specless_label",
        "p_starforming",
        "p_starburst",
        "p_seyfert",
        "p_liner",
        "confidence",
    }.issubset(
        result.columns
    )


@pytest.mark.parametrize(
    "classifier,survey",
    [
        (
            MorphologyClassifier,
            "desi",
        ),

        (
            ActivityClassifier,
            "kids",
        ),

        (
            MorphologyClassifier,
            "invalid",
        ),

        (
            ActivityClassifier,
            "invalid",
        ),
    ],
)
def test_invalid_survey(
    classifier,
    survey,
):

    with pytest.raises(
        ValueError
    ):

        classifier(
            survey=survey
        )


def test_nan_rejected():

    data = pd.DataFrame(
        {
            "u": [
                np.nan
            ],

            "g": [
                18.8
            ],

            "r": [
                17.9
            ],

            "i": [
                17.5
            ],

            "Z": [
                17.3
            ],

            "Y": [
                17.0
            ],

            "J": [
                16.8
            ],

            "H": [
                16.6
            ],

            "Ks": [
                16.5
            ],
        }
    )

    model = (
        MorphologyClassifier(
            survey="kids"
        )
    )

    with pytest.raises(
        ValueError
    ):

        model.predict(
            data
        )
