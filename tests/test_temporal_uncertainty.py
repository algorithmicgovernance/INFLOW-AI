import numpy as np
import pytest

from model.temporal.ablation.metrics import compute_metrics
from model.temporal.ablation.temporal_model_candidates.elastic_net_model import (
    ElasticNetAblationModel,
)


def test_metrics_report_95_interval_width_and_calibration():
    samples = np.asarray([
        [-2.0, -1.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    metrics = compute_metrics(
        np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]), samples,
    )
    assert metrics.calibration == 1.0
    assert metrics.calibration_error_95 == pytest.approx(0.05)
    assert metrics.ci95_mean_width > 0.0
    assert metrics.ci95_median_width > 0.0


def test_deterministic_candidate_produces_residual_bootstrap_samples():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = 0.5 * x[:, 0] + np.sin(x[:, 0])
    model = ElasticNetAblationModel(seed=42)
    model.fit(x, y)
    samples = model.predict_samples(x[:3])
    assert samples.shape == (100, 3)
    assert np.all(np.std(samples, axis=0) > 0.0)
