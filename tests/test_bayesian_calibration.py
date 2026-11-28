from types import SimpleNamespace

import numpy as np

from model.temporal.post_processing.bayesian_calibration import (
    _posterior_predictive_draws,
    _save_inference_data,
    _standardize_design,
)


def test_calibration_design_is_fit_from_training_data_only():
    x_train, x_test, stats = _standardize_design(
        np.asarray([1.0, 3.0]), np.asarray([1000.0]),
        np.asarray([[2.0], [4.0]]), np.asarray([[2000.0]]),
    )
    np.testing.assert_allclose(x_train.mean(axis=0), 0.0)
    assert stats["mean"] == [2.0, 3.0]
    assert x_test[0, 0] > 100.0


def test_student_t_posterior_predictions_are_retained_and_clipped_at_zero():
    posterior = {
        "alpha": np.zeros((2, 3)),
        "beta": np.zeros((2, 3, 2)),
        "sigma": np.ones((2, 3)) * 0.1,
        "nu": np.ones((2, 3)) * 5.0,
    }
    corrected, arrays = _posterior_predictive_draws(
        SimpleNamespace(posterior=posterior), np.zeros((2, 2)),
        correction_mean=-10.0, correction_scale=1.0,
        original_prediction=np.asarray([0.1, 0.2]), count=4, seed=42,
    )
    assert corrected.shape == (4, 2)
    assert np.all(corrected >= 0.0)
    assert arrays["beta"].shape == (6, 2)


def test_inference_data_uses_its_netcdf_method(tmp_path):
    class FakeInference:
        def __init__(self):
            self.saved = None

        def to_netcdf(self, path):
            self.saved = path

    inference = FakeInference()
    output = tmp_path / "posterior.nc"
    assert _save_inference_data(inference, output) == output
    assert inference.saved == str(output)
