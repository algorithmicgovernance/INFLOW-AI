import json
from datetime import datetime

import numpy as np

import model.temporal.ablation.temporal_best_model_viz as viz


def test_best_model_plot_converts_fraction_values_to_km2(monkeypatch, tmp_path):
    captured = {}
    dates = [datetime(2020, 1, 1)]
    monkeypatch.setattr(
        viz,
        "_compute_predicted_series",
        lambda row, lead: (dates, [0.1], dates, [0.2], [-0.1], [0.3], dates, [0.1]),
    )
    monkeypatch.setattr(
        viz,
        "convert_target_fraction_to_km2",
        lambda values, target: np.asarray(values, dtype=float) * 1000.0,
    )
    monkeypatch.setattr(
        viz,
        "_render_svg",
        lambda path, title, actual_dates, actual_values, test_dates, pred_values, lower, upper, split: captured.update(
            actual=actual_values, predicted=pred_values, lower=lower, upper=upper
        ),
    )
    monkeypatch.setattr(
        viz,
        "get_cfg",
        lambda key, default=None: str(tmp_path) if key == "ablation.visualization.output_dir" else default,
    )
    row = {
        "metrics_by_lead_time": json.dumps({"lead_periods_1": {"peak_f1": 0.5}}),
        "training_cutoff_date": "2019-01-01",
        "dataset_target_column": "percent_inundation",
    }

    viz.plot_best_model_results(row)

    assert captured["actual"] == [100.0]
    assert captured["predicted"] == [200.0]
    assert captured["lower"] == [0.0]
    assert captured["upper"] == [300.0]
