"""
Step 12: calibration procedure validation.

The key test: deliberately construct data with a KNOWN, systematic
miscalibration (overconfidence via squaring/renormalizing true
probabilities -- a realistic pattern for how tree ensembles and other
models actually get overconfident), and verify Platt scaling and
isotonic regression measurably correct it on held-out data. This
validates the calibration *procedure*, independent of whether any of
this project's actual models happen to need much correcting.
"""

import numpy as np
import pytest

from calibration.calibrate import IsotonicCalibrator, PlattCalibrator
from calibration.reliability import expected_calibration_error


def make_overconfident_synthetic(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    true_probs = rng.dirichlet([1.5, 1.0, 1.5], size=n)  # valid 3-class probabilities
    actual = np.array([rng.choice(3, p=p) for p in true_probs])

    # Squaring then renormalizing pushes the already-larger class higher
    # and the smaller ones lower -- a known, systematic overconfidence
    # distortion, monotonic per-class (so a calibrator *can* fix it) but
    # badly miscalibrated in raw form.
    distorted = true_probs**3
    distorted /= distorted.sum(axis=1, keepdims=True)
    return distorted, actual, true_probs


@pytest.fixture(scope="module")
def synthetic_split():
    train_probs, train_actual, _ = make_overconfident_synthetic(4000, seed=1)
    test_probs, test_actual, _ = make_overconfident_synthetic(4000, seed=2)
    return train_probs, train_actual, test_probs, test_actual


@pytest.mark.parametrize("Calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_calibration_reduces_ece_on_known_miscalibration(synthetic_split, Calibrator):
    train_probs, train_actual, test_probs, test_actual = synthetic_split

    cal = Calibrator().fit(train_probs, train_actual)
    calibrated_test = cal.transform(test_probs)

    for c in range(3):
        indicator = (test_actual == c).astype(float)
        raw_ece = expected_calibration_error(test_probs[:, c], indicator)
        cal_ece = expected_calibration_error(calibrated_test[:, c], indicator)
        assert cal_ece < raw_ece * 0.6, (
            f"{Calibrator.__name__} class {c}: expected substantial ECE improvement, "
            f"raw={raw_ece:.4f} calibrated={cal_ece:.4f}"
        )


def test_calibrated_probabilities_are_valid_distributions(synthetic_split):
    train_probs, train_actual, test_probs, _ = synthetic_split
    for Calibrator in [PlattCalibrator, IsotonicCalibrator]:
        cal = Calibrator().fit(train_probs, train_actual)
        calibrated = cal.transform(test_probs)
        assert np.all(calibrated >= 0)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)


def test_well_calibrated_input_is_left_roughly_unchanged():
    """If the raw probabilities are already well-calibrated (no
    distortion), calibration shouldn't make things meaningfully worse --
    a basic 'do no harm on already-good input' check."""
    rng = np.random.default_rng(3)
    true_probs = rng.dirichlet([1.5, 1.0, 1.5], size=4000)
    actual = np.array([rng.choice(3, p=p) for p in true_probs])
    train_probs, train_actual = true_probs[:2000], actual[:2000]
    test_probs, test_actual = true_probs[2000:], actual[2000:]

    cal = IsotonicCalibrator().fit(train_probs, train_actual)
    calibrated_test = cal.transform(test_probs)

    for c in range(3):
        indicator = (test_actual == c).astype(float)
        raw_ece = expected_calibration_error(test_probs[:, c], indicator)
        cal_ece = expected_calibration_error(calibrated_test[:, c], indicator)
        assert cal_ece < raw_ece + 0.03, (
            f"calibration should not meaningfully hurt already-calibrated input "
            f"(class {c}: raw={raw_ece:.4f} calibrated={cal_ece:.4f})"
        )
