from __future__ import annotations

import itertools
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from processing.config import get_cfg
from .temporal_data_pipeline import TargetType, prepare_temporal_dataset
from .experiment_logger import ExperimentLogger
from .feature_selection import select_features_with_cv
from .metrics import compute_metrics
from .temporal_model_candidates import TEMPORAL_MODEL_REGISTRY
from .utils import set_global_seed

ARTIFACTS_DIR = Path(get_cfg("ablation.artifacts.base_dir", "model/temporal/ablation/artifacts"))
LOG_PATH = str(get_cfg("ablation.experiments.log_path", "model/temporal/ablation/ablation_experiment_log.csv"))
WEIGHTS_DIR = ARTIFACTS_DIR / "experiment_weights"
BEST_MODEL_INFO_PATH = Path("model/temporal/best_temporal_model.json")


@dataclass
class TemporalAblationConfig:
    model_type: str
    training_cutoff_date: str
    autoregressive: bool
    target_type: TargetType
    inundation_product: str
    seed: int = 42
    ablation_stage: str = "model_selection"
    feature_block_name: str = "fixed_preprocessing"
    include_ecmwf_hindcasts: bool | None = None
    include_target_lag_features: bool | None = None
    include_feature_lag_features: bool | None = None
    seasonally_standardize_features: bool = False
    transformer_history_length: int | None = None


def _feature_block_kwargs(config: TemporalAblationConfig) -> dict[str, bool | None]:
    return {
        "include_ecmwf_hindcasts": config.include_ecmwf_hindcasts,
        "include_target_lag_features": config.include_target_lag_features,
        "include_feature_lag_features": config.include_feature_lag_features,
        "seasonally_standardize_features": config.seasonally_standardize_features,
    }


def _run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _weights_path(
    model_type: str,
    inundation_product: str,
    target_type: str,
    autoregressive: bool,
    cutoff: str,
    run_id: str,
    lead_period: int | None = None,
) -> str:
    safe_cutoff = cutoff.replace("-", "")
    ar_flag = "ar" if autoregressive else "noar"
    lead_tag = f"_lead{int(lead_period)}" if lead_period is not None else ""
    filename = (
        f"{model_type}_{inundation_product}_{target_type}_{ar_flag}_{safe_cutoff}{lead_tag}_{run_id}.pkl"
    )
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(WEIGHTS_DIR / filename)


def _write_best_model_metadata(row: dict[str, Any]) -> None:
    BEST_MODEL_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_INFO_PATH.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")


def _as_log_row(
    config: TemporalAblationConfig, run_id: str, status: str, payload: dict[str, Any]
) -> dict[str, Any]:
    base = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "status": status,
        **asdict(config),
    }
    base.update(payload)
    return base


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[complete]"
    filled = max(0, min(width, int(round(width * current / total))))
    remaining = width - filled
    return f"[{'#' * filled}{'-' * remaining}]"


def format_progress_message(current: int, total: int, config: TemporalAblationConfig) -> str:
    bar = _progress_bar(current, total)
    return (
        f"Progress {current}/{total} {bar} | "
        f"model={config.model_type} | cutoff={config.training_cutoff_date} | "
        f"ar={config.autoregressive} | target={config.target_type} | product={config.inundation_product} | "
        f"stage={config.ablation_stage} | blocks={config.feature_block_name}"
        f" | seasonal_features={config.seasonally_standardize_features}"
        f" | history={config.transformer_history_length or 'n/a'}"
    )


def _rolling_dry_baseline_past_365_days(
    history_dates: np.ndarray,
    history_target_raw: np.ndarray,
    test_dates: np.ndarray,
) -> np.ndarray:
    """Dry baseline per test timestamp = min observed inundation over prior 365 days."""
    hist = pd.DataFrame({
        "date": pd.to_datetime(history_dates),
        "target_raw": history_target_raw.astype(np.float64),
    }).sort_values("date")

    hist = hist.dropna(subset=["date", "target_raw"]).drop_duplicates(subset=["date"], keep="last")
    hist = hist.set_index("date")

    # Use only historical values strictly before each timestamp.
    rolling_min = hist["target_raw"].rolling("365D", closed="left").min()
    fallback_min = hist["target_raw"].expanding(min_periods=1).min().shift(1)
    baseline = rolling_min.fillna(fallback_min)

    aligned = baseline.reindex(pd.to_datetime(test_dates))
    # Last safety fallback for any unresolved gaps.
    aligned = aligned.fillna(float(hist["target_raw"].min()))
    return aligned.to_numpy(dtype=np.float64)


def _resolve_extreme_event_top_fraction() -> float:
    top_fraction = float(get_cfg("ablation.pipeline.metrics.extreme_event_top_fraction", 0.10))
    if top_fraction <= 0.0 or top_fraction >= 1.0:
        raise ValueError("ablation.pipeline.metrics.extreme_event_top_fraction must be in (0, 1).")
    return top_fraction


def _derive_event_threshold_from_training_rel_true(
    y_train_raw: np.ndarray,
    train_baseline: np.ndarray,
    top_fraction: float,
) -> float:
    denom = np.maximum(np.abs(train_baseline.astype(np.float64)), 1e-6)
    rel_true_train = (y_train_raw.astype(np.float64) - train_baseline.astype(np.float64)) / denom
    valid_rel = rel_true_train[np.isfinite(rel_true_train)]
    if valid_rel.size == 0:
        raise RuntimeError("Unable to derive event threshold: no finite rel_true values in training set.")

    quantile = 1.0 - float(top_fraction)
    return float(np.quantile(valid_rel, quantile))


def _recursive_raw_reconstruction(
    target_type: str,
    transformed_points: np.ndarray,
    transformed_samples: np.ndarray,
    reconstruction_components: np.ndarray,
    origin_dates: np.ndarray,
    history_dates: np.ndarray,
    history_target_raw: np.ndarray,
    reconstruction_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert direct lead targets without using post-origin observations."""
    points = np.asarray(transformed_points, dtype=np.float64)
    samples = np.asarray(transformed_samples, dtype=np.float64)
    components = np.asarray(reconstruction_components, dtype=np.float64)
    scales = (
        np.ones_like(components)
        if reconstruction_scales is None
        else np.asarray(reconstruction_scales, dtype=np.float64)
    )
    if points.ndim != 2 or samples.ndim != 3 or components.shape != points.shape or scales.shape != points.shape:
        raise ValueError("Recursive reconstruction expects point/component (obs, lead) and samples (draw, obs, lead).")

    raw_points = np.empty_like(points)
    raw_samples = np.empty_like(samples)
    for lead in range(points.shape[1]):
        raw_points[:, lead] = np.maximum(
            points[:, lead] * scales[:, lead] + components[:, lead], 0.0
        )
        raw_samples[:, :, lead] = np.maximum(
            samples[:, :, lead] * scales[:, lead][np.newaxis, :]
            + components[:, lead][np.newaxis, :], 0.0
        )
    return raw_points, raw_samples


def _average_metrics_across_leads(lead_metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    if not lead_metrics:
        return {}

    metric_names = sorted(next(iter(lead_metrics.values())).keys())
    averaged: dict[str, float] = {}
    for metric_name in metric_names:
        values = np.array(
            [lead_metrics[lead][metric_name] for lead in sorted(lead_metrics.keys())],
            dtype=np.float64,
        )
        finite_values = values[np.isfinite(values)]
        averaged[metric_name] = float(np.mean(finite_values)) if finite_values.size > 0 else float("nan")
    return averaged


def _flatten_metrics_by_lead(lead_metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for lead in sorted(lead_metrics.keys()):
        for metric_name, value in lead_metrics[lead].items():
            flattened[f"lead_{lead}_{metric_name}"] = float(value)
    return flattened


def _baseline_predictions(
    *, origin_dates: np.ndarray, target_dates: np.ndarray,
    train_target_dates: np.ndarray, train_target_raw: np.ndarray,
    history_dates: np.ndarray, history_target_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return persistence-at-origin and training-only seasonal climatology."""
    history = pd.DataFrame({
        "date": pd.to_datetime(history_dates),
        "value": np.asarray(history_target_raw, dtype=np.float64),
    }).dropna().drop_duplicates(subset="date", keep="last").set_index("date")
    origin_index = pd.DatetimeIndex(pd.to_datetime(origin_dates))
    missing_origins = origin_index.difference(history.index)
    if not missing_origins.empty:
        raise RuntimeError(
            "Cannot calculate persistence baseline: target data are missing at forecast origins "
            f"{[str(value.date()) for value in missing_origins[:5]]}."
        )
    persistence = history.loc[origin_index, "value"].to_numpy(dtype=np.float64)

    climatology_frame = pd.DataFrame({
        "date": pd.to_datetime(train_target_dates),
        "value": np.asarray(train_target_raw, dtype=np.float64),
    }).dropna()
    climatology_frame["period"] = climatology_frame["date"].dt.strftime("%m-%d")
    seasonal_means = climatology_frame.groupby("period")["value"].mean()
    target_periods = pd.DatetimeIndex(pd.to_datetime(target_dates)).strftime("%m-%d")
    climatology = seasonal_means.reindex(target_periods)
    if climatology.isna().any():
        missing_periods = sorted(set(target_periods[climatology.isna()]))
        raise RuntimeError(
            "Cannot calculate climatology baseline: no training target data for calendar periods "
            f"{missing_periods}."
        )
    return persistence, climatology.to_numpy(dtype=np.float64)


def _baseline_payload(
    metrics_by_baseline: dict[str, dict[int, dict[str, float]]]
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    combined: dict[str, dict[str, dict[str, float]]] = {}
    for baseline_name, metrics_by_lead in metrics_by_baseline.items():
        averaged = _average_metrics_across_leads(metrics_by_lead)
        lead_payload = {
            f"lead_periods_{lead}": values for lead, values in sorted(metrics_by_lead.items())
        }
        combined[baseline_name] = lead_payload
        payload[f"{baseline_name}_metrics_by_lead_time"] = json.dumps(lead_payload, sort_keys=True)
        payload.update({f"{baseline_name}_avg_{key}": value for key, value in averaged.items()})
        payload.update({
            f"{baseline_name}_{key}": value
            for key, value in _flatten_metrics_by_lead(metrics_by_lead).items()
        })
    payload["baseline_metrics_by_lead_time"] = json.dumps(combined, sort_keys=True)
    return payload


def run_single_temporal_ablation(
    config: TemporalAblationConfig, logger: ExperimentLogger | None = None
) -> dict[str, Any]:
    if logger is None:
        logger = ExperimentLogger(LOG_PATH)

    run_id = _run_id()
    set_global_seed(config.seed)

    try:
        if config.model_type not in TEMPORAL_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{config.model_type}'. "
                f"Available: {sorted(TEMPORAL_MODEL_REGISTRY.keys())}"
            )

        if config.model_type == "transformer":
            return _run_transformer_ablation(config, run_id, logger)

        configured_dataset = prepare_temporal_dataset(
            inundation_product=config.inundation_product,
            target_type=config.target_type,
            autoregressive=config.autoregressive,
            training_cutoff_date=config.training_cutoff_date,
            run_id=run_id,
            model_type=config.model_type,
            **_feature_block_kwargs(config),
        )
        max_lead_period = int(configured_dataset.properties.get("dataset_forecast_horizon_periods", 1))
        if max_lead_period < 1:
            raise RuntimeError("Configured forecast horizon periods resolved to < 1.")

        event_top_fraction = _resolve_extreme_event_top_fraction()
        lead_metrics_by_period: dict[int, dict[str, float]] = {}
        baseline_metrics: dict[str, dict[int, dict[str, float]]] = {
            "persistence": {}, "climatology": {}
        }
        lead_metrics_payload: dict[str, dict[str, float]] = {}
        model_weights_by_lead: dict[str, str] = {}
        extreme_threshold_by_lead: dict[str, float] = {}
        feature_selection_best_k_by_lead: dict[str, int] = {}
        selected_feature_count_by_lead: dict[str, int] = {}
        selected_feature_names_by_lead: dict[str, str] = {}
        lead_outputs: dict[int, tuple[Any, np.ndarray, np.ndarray]] = {}

        feature_selection_enabled_all = True
        feature_selection_used_cache_all = True
        reference_feature_selection = None

        for lead_period in range(1, max_lead_period + 1):
            dataset = prepare_temporal_dataset(
                inundation_product=config.inundation_product,
                target_type=config.target_type,
                autoregressive=config.autoregressive,
                training_cutoff_date=config.training_cutoff_date,
                run_id=run_id,
                model_type=config.model_type,
                forecast_horizon_periods_override=lead_period,
                **_feature_block_kwargs(config),
            )

            feature_selection = select_features_with_cv(
                X_train=dataset.X_train,
                X_test=dataset.X_test,
                y_train=dataset.y_train,
                feature_names=dataset.feature_names,
                model_type=config.model_type,
                seed=config.seed,
                dataset_fingerprint=str(dataset.properties.get("dataset_fingerprint", "unknown")),
                configuration_signature={
                    "model_type": config.model_type,
                    "training_cutoff_date": config.training_cutoff_date,
                    "autoregressive": config.autoregressive,
                    "target_type": config.target_type,
                    "inundation_product": config.inundation_product,
                    "lead_period": lead_period,
                    "feature_block_name": config.feature_block_name,
                    **_feature_block_kwargs(config),
                },
            )

            if reference_feature_selection is None:
                reference_feature_selection = feature_selection

            feature_selection_enabled_all = feature_selection_enabled_all and feature_selection.enabled
            feature_selection_used_cache_all = feature_selection_used_cache_all and feature_selection.used_cache

            model = TEMPORAL_MODEL_REGISTRY[config.model_type](seed=config.seed)
            model.fit(feature_selection.X_train, dataset.y_train)

            y_pred_transformed = model.predict(feature_selection.X_test)
            y_pred_samples_transformed = model.predict_samples(feature_selection.X_test)
            lead_outputs[lead_period] = (
                dataset,
                np.asarray(y_pred_transformed),
                np.asarray(y_pred_samples_transformed),
            )

            model_path = _weights_path(
                model_type=config.model_type,
                inundation_product=config.inundation_product,
                target_type=config.target_type,
                autoregressive=config.autoregressive,
                cutoff=config.training_cutoff_date,
                run_id=run_id,
                lead_period=lead_period,
            )
            model.save_weights(model_path)

            model_weights_by_lead[str(lead_period)] = model_path
            feature_selection_best_k_by_lead[str(lead_period)] = int(feature_selection.best_k)
            selected_feature_count_by_lead[str(lead_period)] = int(len(feature_selection.selected_feature_names))
            selected_feature_names_by_lead[str(lead_period)] = "|".join(feature_selection.selected_feature_names)

        # Score every lead over the same origins. The longest lead determines
        # the deployable time range; intersect defensively with all other leads.
        reference_dataset = lead_outputs[max_lead_period][0]
        common_set = set(pd.to_datetime(reference_dataset.origin_test_dates))
        for dataset, _, _ in lead_outputs.values():
            common_set.intersection_update(pd.to_datetime(dataset.origin_test_dates))
        common_origins = np.asarray([
            date for date in reference_dataset.origin_test_dates
            if pd.Timestamp(date) in common_set
        ])
        if common_origins.size == 0:
            raise RuntimeError("No common held-out origins across forecast leads.")

        point_columns = []
        sample_columns = []
        true_columns = []
        component_columns = []
        scale_columns = []
        target_date_columns = []
        for lead_period in range(1, max_lead_period + 1):
            dataset, point_values, sample_values = lead_outputs[lead_period]
            positions = {pd.Timestamp(date): index for index, date in enumerate(dataset.origin_test_dates)}
            indices = np.asarray([positions[pd.Timestamp(date)] for date in common_origins], dtype=int)
            point_columns.append(point_values[indices])
            sample_columns.append(sample_values[:, indices])
            true_columns.append(dataset.y_test_raw[indices])
            component_columns.append(dataset.raw_reconstruction_component_test[indices])
            scale_columns.append(dataset.raw_reconstruction_scale_test[indices])
            target_date_columns.append(dataset.test_dates[indices])

        transformed_points = np.stack(point_columns, axis=1)
        transformed_samples = np.stack(sample_columns, axis=2)
        true_raw = np.stack(true_columns, axis=1)
        components = np.stack(component_columns, axis=1)
        scales = np.stack(scale_columns, axis=1)
        target_dates = np.stack(target_date_columns, axis=1)
        predicted_raw, predicted_samples_raw = _recursive_raw_reconstruction(
            target_type=config.target_type,
            transformed_points=transformed_points,
            transformed_samples=transformed_samples,
            reconstruction_components=components,
            origin_dates=common_origins,
            history_dates=reference_dataset.history_dates,
            history_target_raw=reference_dataset.history_target_raw,
            reconstruction_scales=scales,
        )

        for lead_index, lead_period in enumerate(range(1, max_lead_period + 1)):
            dataset = lead_outputs[lead_period][0]
            dry_baseline = _rolling_dry_baseline_past_365_days(
                dataset.history_dates, dataset.history_target_raw, target_dates[:, lead_index]
            )
            train_baseline = _rolling_dry_baseline_past_365_days(
                dataset.history_dates, dataset.history_target_raw, dataset.train_dates
            )
            event_change_threshold = _derive_event_threshold_from_training_rel_true(
                dataset.y_train_raw, train_baseline, event_top_fraction
            )
            metrics = compute_metrics(
                y_true_raw=true_raw[:, lead_index],
                y_pred_raw=predicted_raw[:, lead_index],
                y_pred_samples_raw=predicted_samples_raw[:, :, lead_index],
                dry_season_baseline=dry_baseline,
                event_change_threshold=event_change_threshold,
            )
            lead_metrics_by_period[lead_period] = metrics.to_dict()
            persistence_pred, climatology_pred = _baseline_predictions(
                origin_dates=common_origins,
                target_dates=target_dates[:, lead_index],
                train_target_dates=dataset.train_dates,
                train_target_raw=dataset.y_train_raw,
                history_dates=dataset.history_dates,
                history_target_raw=dataset.history_target_raw,
            )
            for baseline_name, baseline_pred in (
                ("persistence", persistence_pred), ("climatology", climatology_pred)
            ):
                baseline_metrics[baseline_name][lead_period] = compute_metrics(
                    y_true_raw=true_raw[:, lead_index], y_pred_raw=baseline_pred,
                    dry_season_baseline=dry_baseline,
                    event_change_threshold=event_change_threshold,
                ).to_dict()
            lead_metrics_payload[f"lead_periods_{lead_period}"] = metrics.to_dict()
            extreme_threshold_by_lead[str(lead_period)] = float(event_change_threshold)

        averaged_metrics = _average_metrics_across_leads(lead_metrics_by_period)
        flattened_lead_metrics = _flatten_metrics_by_lead(lead_metrics_by_period)

        if reference_feature_selection is None:
            raise RuntimeError("Internal error: no lead-specific feature selection results were computed.")

        payload = {
            "model_weights_path": model_weights_by_lead.get(str(max_lead_period), ""),
            "model_weights_path_by_lead": json.dumps(model_weights_by_lead, sort_keys=True),
            "dry_baseline_definition": "rolling_min_past_365_days",
            "extreme_event_definition": "rel_true_above_training_quantile_threshold",
            "extreme_event_top_fraction": event_top_fraction,
            "extreme_event_threshold": extreme_threshold_by_lead.get(str(max_lead_period), float("nan")),
            "extreme_event_threshold_by_lead": json.dumps(extreme_threshold_by_lead, sort_keys=True),
            "metrics_target_space": "raw",
            "metrics_averaging_scope": "equal_weight_across_lead_periods",
            "metrics_by_lead_time": json.dumps(lead_metrics_payload, sort_keys=True),
            "baseline_definitions": (
                "persistence=observed_target_at_forecast_origin;"
                "climatology=training_target_mean_for_target_month_day"
            ),
            **_baseline_payload(baseline_metrics),
            "feature_selection_enabled": feature_selection_enabled_all,
            "feature_selection_used_cache": feature_selection_used_cache_all,
            "feature_selection_cache_path": reference_feature_selection.cache_path,
            "feature_selection_best_k": feature_selection_best_k_by_lead.get(str(max_lead_period), -1),
            "feature_selection_best_k_by_lead": json.dumps(feature_selection_best_k_by_lead, sort_keys=True),
            "selected_feature_count": selected_feature_count_by_lead.get(str(max_lead_period), 0),
            "selected_feature_count_by_lead": json.dumps(selected_feature_count_by_lead, sort_keys=True),
            "selected_feature_names": selected_feature_names_by_lead.get(str(max_lead_period), ""),
            "selected_feature_names_by_lead": json.dumps(selected_feature_names_by_lead, sort_keys=True),
            "removed_constant_features": "|".join(reference_feature_selection.removed_constant_features),
            "removed_correlated_features": "|".join(reference_feature_selection.removed_correlated_features),
            "feature_selection_cv_mean_mae_by_k": json.dumps(
                {str(k): float(v) for k, v in reference_feature_selection.cv_mean_mae_by_k.items()},
                sort_keys=True,
            ),
            "dataset_multi_lead_enabled": True,
            "dataset_lead_period_min": 1,
            "dataset_lead_period_max": int(max_lead_period),
            "dataset_forecast_horizon_periods": int(max_lead_period),
            **configured_dataset.properties,
            **averaged_metrics,
            **{f"avg_{k}": v for k, v in averaged_metrics.items()},
            **flattened_lead_metrics,
        }
        row = _as_log_row(config, run_id, "success", payload)
        logger.append(row)
        return row

    except Exception as e:
        error_payload = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(limit=5),
        }
        row = _as_log_row(config, run_id, "failed", error_payload)
        logger.append(row)
        return row


def _run_transformer_ablation(
    config: TemporalAblationConfig, run_id: str, logger: ExperimentLogger
) -> dict[str, Any]:
    """Train and score one joint, non-autoregressive multi-horizon model."""
    try:
        from .temporal_transformer_pipeline import prepare_temporal_transformer_dataset

        dataset = prepare_temporal_transformer_dataset(
            inundation_product=config.inundation_product,
            target_type=config.target_type,
            autoregressive=config.autoregressive,
            training_cutoff_date=config.training_cutoff_date,
            run_id=run_id,
            seed=config.seed,
            **_feature_block_kwargs(config),
            history_length=config.transformer_history_length,
        )
        model = TEMPORAL_MODEL_REGISTRY[config.model_type](
            seed=config.seed, history_length=config.transformer_history_length
        )
        model.fit(dataset.X_train, dataset.y_train)
        predicted = model.predict(dataset.X_test)
        samples = model.predict_samples(dataset.X_test)
        horizon_count = int(dataset.y_train.shape[1])
        if predicted.shape != dataset.y_test.shape:
            raise RuntimeError(
                f"Transformer prediction shape {predicted.shape} does not match targets {dataset.y_test.shape}."
            )

        metrics_by_lead: dict[int, dict[str, float]] = {}
        baseline_metrics: dict[str, dict[int, dict[str, float]]] = {
            "persistence": {}, "climatology": {}
        }
        metrics_payload: dict[str, dict[str, float]] = {}
        thresholds: dict[str, float] = {}
        event_top_fraction = _resolve_extreme_event_top_fraction()
        predicted_raw_all, sample_raw_all = _recursive_raw_reconstruction(
            target_type=config.target_type,
            transformed_points=predicted,
            transformed_samples=samples,
            reconstruction_components=dataset.reconstruction_test,
            origin_dates=dataset.test_origin_dates,
            history_dates=dataset.configured_dataset.history_dates,
            history_target_raw=dataset.configured_dataset.history_target_raw,
            reconstruction_scales=dataset.reconstruction_scale_test,
        )
        for horizon in range(horizon_count):
            predicted_raw = predicted_raw_all[:, horizon]
            sample_raw = sample_raw_all[:, :, horizon]
            dry_baseline = _rolling_dry_baseline_past_365_days(
                dataset.configured_dataset.history_dates,
                dataset.configured_dataset.history_target_raw,
                dataset.test_target_dates[:, horizon],
            )
            train_baseline = _rolling_dry_baseline_past_365_days(
                dataset.configured_dataset.history_dates,
                dataset.configured_dataset.history_target_raw,
                dataset.train_target_dates[:, horizon],
            )
            threshold = _derive_event_threshold_from_training_rel_true(
                dataset.y_train_raw[:, horizon], train_baseline, event_top_fraction
            )
            metrics = compute_metrics(
                y_true_raw=dataset.y_test_raw[:, horizon],
                y_pred_raw=predicted_raw,
                y_pred_samples_raw=sample_raw,
                dry_season_baseline=dry_baseline,
                event_change_threshold=threshold,
            )
            lead = horizon + 1
            metrics_by_lead[lead] = metrics.to_dict()
            persistence_pred, climatology_pred = _baseline_predictions(
                origin_dates=dataset.test_origin_dates,
                target_dates=dataset.test_target_dates[:, horizon],
                train_target_dates=dataset.train_target_dates[:, horizon],
                train_target_raw=dataset.y_train_raw[:, horizon],
                history_dates=dataset.configured_dataset.history_dates,
                history_target_raw=dataset.configured_dataset.history_target_raw,
            )
            for baseline_name, baseline_pred in (
                ("persistence", persistence_pred), ("climatology", climatology_pred)
            ):
                baseline_metrics[baseline_name][lead] = compute_metrics(
                    y_true_raw=dataset.y_test_raw[:, horizon], y_pred_raw=baseline_pred,
                    dry_season_baseline=dry_baseline, event_change_threshold=threshold,
                ).to_dict()
            metrics_payload[f"lead_periods_{lead}"] = metrics.to_dict()
            thresholds[str(lead)] = float(threshold)

        model_path = _weights_path(
            config.model_type,
            config.inundation_product,
            config.target_type,
            config.autoregressive,
            config.training_cutoff_date,
            run_id,
        )
        model.save_weights(model_path)
        selection = dataset.feature_selection
        selected_names = selection.selected_feature_names
        weights_by_lead = {str(lead): model_path for lead in range(1, horizon_count + 1)}
        selected_by_lead = {
            str(lead): "|".join(selected_names) for lead in range(1, horizon_count + 1)
        }
        averaged = _average_metrics_across_leads(metrics_by_lead)
        payload = {
            "model_weights_path": model_path,
            "model_weights_path_by_lead": json.dumps(weights_by_lead, sort_keys=True),
            "dry_baseline_definition": "rolling_min_past_365_days",
            "extreme_event_definition": "rel_true_above_training_quantile_threshold",
            "extreme_event_top_fraction": event_top_fraction,
            "extreme_event_threshold": thresholds[str(horizon_count)],
            "extreme_event_threshold_by_lead": json.dumps(thresholds, sort_keys=True),
            "metrics_target_space": "raw",
            "metrics_averaging_scope": "equal_weight_across_lead_periods",
            "metrics_by_lead_time": json.dumps(metrics_payload, sort_keys=True),
            "baseline_definitions": (
                "persistence=observed_target_at_forecast_origin;"
                "climatology=training_target_mean_for_target_month_day"
            ),
            **_baseline_payload(baseline_metrics),
            "feature_selection_enabled": selection.enabled,
            "feature_selection_used_cache": selection.used_cache,
            "feature_selection_cache_path": selection.cache_path,
            "feature_selection_best_k": selection.best_k,
            "feature_selection_best_k_by_lead": json.dumps(
                {str(lead): selection.best_k for lead in range(1, horizon_count + 1)}, sort_keys=True
            ),
            "selected_feature_count": len(selected_names),
            "selected_feature_count_by_lead": json.dumps(
                {str(lead): len(selected_names) for lead in range(1, horizon_count + 1)}, sort_keys=True
            ),
            "selected_feature_names": "|".join(selected_names),
            "selected_feature_names_by_lead": json.dumps(selected_by_lead, sort_keys=True),
            "removed_constant_features": "|".join(selection.removed_constant_features),
            "removed_correlated_features": "|".join(selection.removed_correlated_features),
            "feature_selection_cv_mean_mae_by_k": json.dumps(selection.cv_mean_mae_by_k, sort_keys=True),
            "dataset_multi_lead_enabled": True,
            "dataset_joint_multi_horizon_model": True,
            "dataset_history_sequence_length": int(dataset.X_train.history.shape[1]),
            "dataset_history_feature_names": "|".join(dataset.history_feature_names),
            "dataset_future_feature_names": "|".join(dataset.future_feature_names),
            "dataset_lead_period_min": 1,
            "dataset_lead_period_max": horizon_count,
            "dataset_forecast_horizon_periods": horizon_count,
            **dataset.configured_dataset.properties,
            **averaged,
            **{f"avg_{key}": value for key, value in averaged.items()},
            **_flatten_metrics_by_lead(metrics_by_lead),
        }
        row = _as_log_row(config, run_id, "success", payload)
        logger.append(row)
        return row
    except Exception as error:
        row = _as_log_row(
            config,
            run_id,
            "failed",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(limit=5),
            },
        )
        logger.append(row)
        return row


def run_temporal_ablation_grid(
    configs: Iterable[TemporalAblationConfig], log_csv_path: str = LOG_PATH
) -> list[dict[str, Any]]:
    logger = ExperimentLogger(log_csv_path)
    configs = list(configs)
    results = []

    total = len(configs)
    for index, config in enumerate(configs, start=1):
        status_label = "running"
        progress_message = format_progress_message(index, total, config)
        print(f"\r{progress_message} | status={status_label}", end="", flush=True)
        result = run_single_temporal_ablation(config, logger=logger)
        status_label = result.get("status", "unknown")
        print(f"\r{progress_message} | status={status_label}", end="", flush=True)
        results.append(result)

    print()
    successful = [row for row in results if row.get("status") == "success"]
    # The operational retraining/forecast path is currently lead-specific. Keep
    # joint neural models in the ablation results without publishing one as the
    # deployable runtime model until that separate path supports sequence input.
    deployable = [row for row in successful if not row.get("dataset_joint_multi_horizon_model", False)]
    if deployable:
        best = max(
            deployable,
            key=lambda row: (
                float(row.get("avg_peak_f1", row.get("peak_f1", float("-inf")))),
                float(row.get("avg_peak_auc", row.get("peak_auc", float("-inf")))),
            ),
        )
        _write_best_model_metadata(best)
    return results


def _ranking_key(row: dict[str, Any], metric: str) -> tuple[float, float]:
    def finite(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float("-inf")
        return result if np.isfinite(result) else float("-inf")

    return finite(row.get(metric)), finite(row.get("avg_peak_auc", row.get("peak_auc")))


def build_temporal_feature_ablation_configs(
    top_rows: list[dict[str, Any]],
    feature_blocks: list[dict[str, Any]],
    fixed_preprocessing: dict[str, Any],
) -> list[TemporalAblationConfig]:
    """Expand winning model configurations over explicit feature-block removals."""
    configs: list[TemporalAblationConfig] = []
    seen: set[tuple[Any, ...]] = set()
    for row in top_rows:
        autoregressive = bool(row["autoregressive"])
        # Stage-one results are the fixed-preprocessing reference; do not rerun
        # an equivalent effective block combination in stage two.
        fixed_signature = (
            bool(fixed_preprocessing["include_ecmwf_hindcasts"]),
            bool(fixed_preprocessing["include_target_lag_features"]) and autoregressive,
            bool(fixed_preprocessing["include_feature_lag_features"]),
            bool(row.get("seasonally_standardize_features", False)),
        )
        for block in feature_blocks:
            ecmwf = bool(block["include_ecmwf_hindcasts"])
            target_lags = bool(block["include_target_lag_features"])
            feature_lags = bool(block["include_feature_lag_features"])
            seasonal_features = bool(row.get("seasonally_standardize_features", False))
            effective_signature = (ecmwf, target_lags and autoregressive, feature_lags, seasonal_features)
            if effective_signature == fixed_signature:
                continue
            signature = (
                row["model_type"],
                row["training_cutoff_date"],
                autoregressive,
                row["target_type"],
                row["inundation_product"],
                row.get("transformer_history_length"),
                *effective_signature,
            )
            if signature in seen:
                continue
            seen.add(signature)
            configs.append(
                TemporalAblationConfig(
                    model_type=str(row["model_type"]),
                    training_cutoff_date=str(row["training_cutoff_date"]),
                    autoregressive=autoregressive,
                    target_type=str(row["target_type"]),
                    inundation_product=str(row["inundation_product"]),
                    seed=int(row.get("seed", 42)),
                    ablation_stage="feature_blocks",
                    feature_block_name=str(block["name"]),
                    include_ecmwf_hindcasts=ecmwf,
                    include_target_lag_features=target_lags,
                    include_feature_lag_features=feature_lags,
                    seasonally_standardize_features=seasonal_features,
                    transformer_history_length=(
                        int(row["transformer_history_length"])
                        if row.get("transformer_history_length") not in (None, "", "nan")
                        and not pd.isna(row.get("transformer_history_length"))
                        else None
                    ),
                )
            )
    return configs


def run_staged_temporal_ablation(
    model_configs: Iterable[TemporalAblationConfig],
    *,
    feature_blocks: list[dict[str, Any]],
    fixed_preprocessing: dict[str, Any],
    top_n: int,
    ranking_metric: str,
    log_csv_path: str = LOG_PATH,
) -> list[dict[str, Any]]:
    """Rank model choices first, then ablate features for only the winners."""
    if top_n < 1:
        raise ValueError("ablation.staged.top_n_model_configurations must be at least one.")
    stage_one = run_temporal_ablation_grid(model_configs, log_csv_path=log_csv_path)
    successful = [row for row in stage_one if row.get("status") == "success"]
    ranked = sorted(successful, key=lambda row: _ranking_key(row, ranking_metric), reverse=True)
    top_rows = ranked[:top_n]
    if not top_rows:
        return stage_one
    stage_two_configs = build_temporal_feature_ablation_configs(
        top_rows, feature_blocks, fixed_preprocessing
    )
    stage_two = (
        run_temporal_ablation_grid(stage_two_configs, log_csv_path=log_csv_path)
        if stage_two_configs
        else []
    )
    combined = stage_one + stage_two
    deployable = [
        row for row in combined
        if row.get("status") == "success" and not row.get("dataset_joint_multi_horizon_model", False)
    ]
    if deployable:
        _write_best_model_metadata(max(deployable, key=lambda row: _ranking_key(row, ranking_metric)))
    return combined


def build_grid(
    model_types: list[str],
    cutoff_dates: list[str],
    autoregressive_values: list[bool],
    target_types: list[TargetType],
    inundation_products: list[str],
    seed: int,
    ablation_stage: str = "model_selection",
    feature_block_name: str = "fixed_preprocessing",
    include_ecmwf_hindcasts: bool | None = None,
    include_target_lag_features: bool | None = None,
    include_feature_lag_features: bool | None = None,
    seasonally_standardize_features_values: list[bool] | None = None,
    transformer_history_lengths: list[int] | None = None,
) -> list[TemporalAblationConfig]:
    configs = []
    seasonal_values = seasonally_standardize_features_values or [False]
    configured_histories = transformer_history_lengths or [12]
    for model_type, cutoff, autoregressive, target_type, product, seasonal_features in itertools.product(
        model_types,
        cutoff_dates,
        autoregressive_values,
        target_types,
        inundation_products,
        seasonal_values,
    ):
        histories = configured_histories if model_type == "transformer" else [None]
        for history_length in histories:
            configs.append(TemporalAblationConfig(
                model_type=model_type,
                training_cutoff_date=cutoff,
                autoregressive=autoregressive,
                target_type=target_type,
                inundation_product=product,
                seed=seed,
                ablation_stage=ablation_stage,
                feature_block_name=feature_block_name,
                include_ecmwf_hindcasts=include_ecmwf_hindcasts,
                include_target_lag_features=include_target_lag_features,
                include_feature_lag_features=include_feature_lag_features,
                seasonally_standardize_features=seasonal_features,
                transformer_history_length=history_length,
            ))
    return configs


# Public compatibility aliases retained for existing integrations and notebooks.
AblationConfig = TemporalAblationConfig
run_single_ablation = run_single_temporal_ablation
run_ablation_grid = run_temporal_ablation_grid
build_feature_ablation_configs = build_temporal_feature_ablation_configs
run_staged_ablation = run_staged_temporal_ablation
