import json

import numpy as np

from model.temporal.ablation.temporal_ablation_runner import (
    _baseline_payload,
    _baseline_predictions,
)


def test_baselines_use_origin_target_and_training_only_calendar_period():
    persistence, climatology = _baseline_predictions(
        origin_dates=np.array(["2023-01-01", "2023-02-01"], dtype="datetime64[D]"),
        target_dates=np.array(["2023-03-01", "2023-04-01"], dtype="datetime64[D]"),
        train_target_dates=np.array(
            ["2020-03-01", "2021-03-01", "2020-04-01", "2021-04-01"],
            dtype="datetime64[D]",
        ),
        train_target_raw=np.array([2.0, 4.0, 6.0, 10.0]),
        history_dates=np.array(["2023-01-01", "2023-02-01"], dtype="datetime64[D]"),
        history_target_raw=np.array([11.0, 12.0]),
    )

    np.testing.assert_allclose(persistence, [11.0, 12.0])
    np.testing.assert_allclose(climatology, [3.0, 8.0])


def test_baseline_payload_exposes_per_lead_and_average_metrics():
    metrics = {
        "persistence": {1: {"mae": 2.0}, 2: {"mae": 4.0}},
        "climatology": {1: {"mae": 1.0}, 2: {"mae": 3.0}},
    }
    payload = _baseline_payload(metrics)

    assert payload["persistence_avg_mae"] == 3.0
    assert payload["climatology_lead_2_mae"] == 3.0
    combined = json.loads(payload["baseline_metrics_by_lead_time"])
    assert combined["persistence"]["lead_periods_1"]["mae"] == 2.0
