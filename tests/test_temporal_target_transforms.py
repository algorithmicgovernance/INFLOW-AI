import numpy as np
import pandas as pd
import joblib

from model.temporal.ablation.temporal_data_pipeline import (
    _direct_forecast_target,
    _fit_train_only_transforms,
)
from model.temporal.ablation.temporal_ablation_runner import build_grid


def _target_frame():
    dates = pd.date_range("2018-01-01", periods=15, freq="YS")
    return pd.DataFrame({"date": dates, "target_raw": np.arange(15, dtype=float)})


def test_first_difference_is_direct_cumulative_change_from_origin():
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="10D"),
        "target_raw": [10.0, 11.0, 15.0, 14.0, 20.0],
    })
    transformed = _direct_forecast_target(
        frame, target_type="first_differenced", horizon=3,
        training_cutoff_date="2020-01-21",
    )
    assert transformed.loc[0, "target"] == 4.0
    assert transformed.loc[0, "reconstruction_component"] == 10.0


def test_deseasonalised_statistics_exclude_post_cutoff_values():
    frame = _target_frame()
    cutoff = "2027-01-01"
    baseline = _direct_forecast_target(
        frame, target_type="deseasonalised", horizon=1,
        training_cutoff_date=cutoff,
    )
    changed = frame.copy()
    changed.loc[changed["date"] > cutoff, "target_raw"] += 10000.0
    after = _direct_forecast_target(
        changed, target_type="deseasonalised", horizon=1,
        training_cutoff_date=cutoff,
    )
    np.testing.assert_allclose(
        baseline.loc[:9, "reconstruction_component"],
        after.loc[:9, "reconstruction_component"],
    )
    np.testing.assert_allclose(
        baseline.loc[:9, "reconstruction_scale"],
        after.loc[:9, "reconstruction_scale"],
    )


def test_differenced_anomaly_reconstructs_from_origin_only():
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=4, freq="10D"),
        "target_raw": [2.0, 4.0, 8.0, 16.0],
    })
    transformed = _direct_forecast_target(
        frame, target_type="differenced_anomaly", horizon=2,
        training_cutoff_date="2020-01-11",
    )
    assert transformed.loc[0, "target"] == 6.0
    assert transformed.loc[0, "reconstruction_component"] == 2.0


def test_feature_seasonal_standardization_uses_training_period_only(tmp_path):
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        "feature": [1.0, 3.0, 1000.0],
    })
    path = tmp_path / "scaler.pkl"
    _fit_train_only_transforms(
        frame, pd.Series([True, True, False]), str(path),
        seasonally_standardize_features=True,
    )
    bundle = joblib.load(path)
    assert bundle["seasonal_stats"]["feature"]["mean"]["01-01"] == 2.0


def test_ablation_grid_expands_both_feature_standardization_settings():
    grid = build_grid(
        ["elastic_net"], ["2020-01-01"], [False], ["raw"], ["viirs"], 42,
        seasonally_standardize_features_values=[False, True],
    )
    assert [item.seasonally_standardize_features for item in grid] == [False, True]


def test_ablation_grid_expands_transformer_history_without_duplicating_tabular_models():
    grid = build_grid(
        ["elastic_net", "transformer"], ["2020-01-01"], [False], ["raw"],
        ["viirs"], 42, transformer_history_lengths=[12, 24, 36],
    )
    elastic = [item for item in grid if item.model_type == "elastic_net"]
    transformers = [item for item in grid if item.model_type == "transformer"]
    assert len(elastic) == 1
    assert [item.transformer_history_length for item in transformers] == [12, 24, 36]
