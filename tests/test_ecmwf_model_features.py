import os
import sys
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import model.ablation.feature_selection as feature_selection
from model.ablation.temporal_data_pipeline import (
    _add_optional_feature_blocks,
    _ecmwf_leads,
)


def test_all_four_ecmwf_windows_are_recognized_and_not_historically_lagged():
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=15, freq="10D"),
        "ordinary_rainfall": np.arange(15, dtype=float),
        **{
            f"ecmwf_rainfall_ssd_lead_{lead}": np.arange(15, dtype=float) + lead
            for lead in range(1, 5)
        },
    })
    assert _ecmwf_leads(list(frame.columns)) == {1, 2, 3, 4}
    result, lagged_features, _ = _add_optional_feature_blocks(
        frame,
        target_for_lags=pd.Series(np.arange(15, dtype=float)),
        autoregressive=False,
    )
    assert "ordinary_rainfall_lag_1" in result
    assert any(name == "ordinary_rainfall_lag_1" for name in lagged_features)
    assert not any(name.startswith("ecmwf_") and "_lag_" in name for name in result.columns)


def test_feature_selection_can_correlation_prune_ecmwf_windows(monkeypatch):
    rng = np.random.default_rng(42)
    base = rng.normal(size=40)
    names = [f"ecmwf_rainfall_ssd_lead_{lead}" for lead in range(1, 5)] + ["optional"]
    # Deliberately make the forecast windows highly correlated so normal
    # train-only redundancy pruning is exercised.
    matrix = np.column_stack([base * lead for lead in range(1, 5)] + [rng.normal(size=40)])

    settings = {
        "ablation.pipeline.feature_selection.enabled": True,
        "ablation.pipeline.feature_selection.near_constant_variance_threshold": 1e-8,
        "ablation.pipeline.feature_selection.high_correlation_threshold": 0.90,
        "ablation.pipeline.feature_selection.cv_splits": 1,
        "ablation.pipeline.feature_selection.candidate_feature_counts": [1],
        "ablation.pipeline.feature_selection.include_all_features_candidate": False,
        "ablation.pipeline.feature_selection.cache.enabled": False,
        "ablation.pipeline.feature_selection.elastic_net.alpha": 0.001,
        "ablation.pipeline.feature_selection.elastic_net.l1_ratio": 0.5,
        "ablation.pipeline.feature_selection.elastic_net.max_iter": 10000,
        "ablation.artifacts.base_dir": "unused",
    }
    monkeypatch.setattr(
        feature_selection,
        "get_cfg",
        lambda key, default=None: settings.get(key, default),
    )
    selected = feature_selection.select_features_with_cv(
        X_train=matrix[:30],
        X_test=matrix[30:],
        y_train=base[:30],
        feature_names=names,
        model_type="linear_regression",
        seed=42,
        dataset_fingerprint="test",
        configuration_signature={"lead": 1},
    )
    assert len(set(names[:4]).intersection(selected.selected_feature_names)) < 4
    assert any(name in selected.removed_correlated_features for name in names[1:4])


def test_feature_selection_cache_is_shared_across_candidate_models(monkeypatch, tmp_path):
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(30, 4))
    target = matrix[:, 0] - matrix[:, 1]
    settings = {
        "ablation.pipeline.feature_selection.enabled": True,
        "ablation.pipeline.feature_selection.selection_model_type": "elastic_net",
        "ablation.pipeline.feature_selection.near_constant_variance_threshold": 1e-8,
        "ablation.pipeline.feature_selection.high_correlation_threshold": 0.995,
        "ablation.pipeline.feature_selection.cv_splits": 1,
        "ablation.pipeline.feature_selection.candidate_feature_counts": [2],
        "ablation.pipeline.feature_selection.include_all_features_candidate": False,
        "ablation.pipeline.feature_selection.cache.enabled": True,
        "ablation.pipeline.feature_selection.elastic_net.alpha": 0.001,
        "ablation.pipeline.feature_selection.elastic_net.l1_ratio": 0.5,
        "ablation.pipeline.feature_selection.elastic_net.max_iter": 10000,
        "ablation.artifacts.base_dir": str(tmp_path),
    }
    monkeypatch.setattr(feature_selection, "get_cfg", lambda key, default=None: settings.get(key, default))
    common = {
        "X_train": matrix[:20],
        "X_test": matrix[20:],
        "y_train": target[:20],
        "feature_names": ["a", "b", "c", "d"],
        "seed": 42,
        "dataset_fingerprint": "same-lead-dataset",
    }
    first = feature_selection.select_features_with_cv(
        **common,
        model_type="random_forest",
        configuration_signature={"model_type": "random_forest", "lead_period": 2},
    )
    second = feature_selection.select_features_with_cv(
        **common,
        model_type="linear_regression",
        configuration_signature={"model_type": "linear_regression", "lead_period": 2},
    )
    different_lead = feature_selection.select_features_with_cv(
        **common,
        model_type="linear_regression",
        configuration_signature={"model_type": "linear_regression", "lead_period": 3},
    )
    assert first.used_cache is False
    assert second.used_cache is True
    assert first.cache_path == second.cache_path
    assert first.selected_feature_names == second.selected_feature_names
    assert different_lead.used_cache is False
    assert different_lead.cache_path != first.cache_path


def test_feature_lags_are_created_without_dataframe_fragmentation_warning(monkeypatch):
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=20, freq="10D"),
        **{f"feature_{index}": np.arange(20, dtype=float) for index in range(100)},
    })
    monkeypatch.setattr(
        "model.ablation.temporal_data_pipeline.get_cfg",
        lambda key, default=None: {
            "ablation.pipeline.optional_feature_blocks.include_feature_lag_features": True,
            "ablation.pipeline.optional_feature_blocks.calendar": True,
            "ablation.pipeline.optional_feature_blocks.lag_steps": [1, 2, 3, 6, 12],
        }.get(key, default),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, lagged, _ = _add_optional_feature_blocks(
            frame, target_for_lags=pd.Series(np.arange(20, dtype=float)), autoregressive=False
        )
    assert len(lagged) == 500
    assert result.shape[1] == 605
    assert not any("highly fragmented" in str(item.message) for item in caught)


def test_optional_feature_blocks_replace_existing_calendar_columns(monkeypatch):
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="10D"),
        "feature": np.arange(10, dtype=float),
        "month": -1,
        "day_of_year": -1,
    })
    monkeypatch.setattr(
        "model.ablation.temporal_data_pipeline.get_cfg",
        lambda key, default=None: {
            "ablation.pipeline.optional_feature_blocks.include_feature_lag_features": False,
            "ablation.pipeline.optional_feature_blocks.calendar": True,
            "ablation.pipeline.optional_feature_blocks.lag_steps": [1],
        }.get(key, default),
    )
    result, _, _ = _add_optional_feature_blocks(
        frame, target_for_lags=pd.Series(np.arange(10, dtype=float)), autoregressive=False
    )
    assert result.columns.is_unique
    assert result["month"].tolist() == result["date"].dt.month.tolist()
    assert result["day_of_year"].tolist() == result["date"].dt.dayofyear.tolist()
