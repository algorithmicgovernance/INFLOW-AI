from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
import ast
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from processing import cleaning_utils
from processing.config import get_cfg, get_config
from processing.data_cleaning import process_albert
from processing.data_cleaning import process_gridded_moisture
from processing.data_cleaning import process_gridded_rainfall
from processing.data_cleaning import process_inundation_viirs
from processing.data_cleaning import process_kyoga
from processing.data_cleaning import process_rainfall
from processing.data_cleaning import process_teleconnections
from processing.data_cleaning import process_victoria

from ..ablation.temporal_data_pipeline import (
    _add_optional_feature_blocks,
    _add_post_scale_cumulative_features,
    _build_temporal_feature_table,
    _fit_train_only_transforms,
    _direct_forecast_target,
    _seasonal_period,
    _target_transform,
)
from ..ablation.temporal_model_candidates import TEMPORAL_MODEL_REGISTRY
from ..ablation.pca_processing import apply_train_only_pca


BEST_MODEL_INFO_PATH = Path("model/temporal/best_temporal_model.json")
ARTIFACTS_DIR = Path(str(get_cfg("ablation.artifacts.base_dir", "model/temporal/ablation/artifacts")))
RUNTIME_RETRAINED_WEIGHTS_DIR = ARTIFACTS_DIR / "runtime_retrained_weights"


@dataclass
class TemporalForecastBundle:
    target_column: str
    target_product: str
    model_row: dict[str, Any]
    origin_date: pd.Timestamp
    future_dates: list[pd.Timestamp]
    history_dates: list[pd.Timestamp]
    history_values_raw: np.ndarray
    predicted_raw: np.ndarray
    lower_raw: np.ndarray
    upper_raw: np.ndarray


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "0.0", "false", "no", "n", "", "nan"}:
        return False
    try:
        return float(normalized) != 0.0
    except ValueError:
        return False


def _parse_pipe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item).strip()]
    return [item for item in text.split("|") if item]


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}
    return json.loads(text)


def _parse_feature_names_by_lead(value: Any) -> dict[int, list[str]]:
    payload = _parse_json_dict(value)
    return {int(k): _parse_pipe_list(v) for k, v in payload.items()}


def _parse_weights_by_lead(value: Any) -> dict[int, str]:
    payload = _parse_json_dict(value)
    return {int(k): str(v) for k, v in payload.items()}


def _sync_runtime_config_from_row(row: dict[str, Any]) -> None:
    cfg = get_config()
    cfg.setdefault("runtime", {})
    cfg.setdefault("ablation", {}).setdefault("pipeline", {}).setdefault("optional_feature_blocks", {})

    cfg["runtime"]["target_product"] = str(row["inundation_product"]).strip().lower()
    target_column = row.get("dataset_target_column") or row.get("target_column") or "percent_inundation"
    cfg["ablation"]["pipeline"]["target_column"] = str(target_column).strip()
    cfg["ablation"]["pipeline"]["optional_feature_blocks"]["include_target_lag_features"] = _as_bool(
        row.get("dataset_target_lag_feature_toggle", True)
    )
    selected_names = str(row.get("selected_feature_names", ""))
    inferred_feature_lags = "_lag_" in selected_names
    cfg["ablation"]["pipeline"]["optional_feature_blocks"]["include_feature_lag_features"] = (
        _as_bool(row.get("dataset_feature_lag_feature_toggle", inferred_feature_lags))
        or inferred_feature_lags
    )
    cfg["ablation"]["pipeline"]["seasonally_standardize_features"] = _as_bool(
        row.get("dataset_seasonally_standardized_features", row.get("seasonally_standardize_features", False))
    )

    lag_steps = _parse_pipe_list(row.get("dataset_feature_lag_steps")) or _parse_pipe_list(
        row.get("dataset_target_lag_steps")
    )
    if lag_steps:
        cfg["ablation"]["pipeline"]["optional_feature_blocks"]["lag_steps"] = [int(v) for v in lag_steps]


def load_best_model_row() -> dict[str, Any]:
    if not BEST_MODEL_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Runner output is missing: {BEST_MODEL_INFO_PATH}. "
            "Run the ablation runner before deployment."
        )
    row = json.loads(BEST_MODEL_INFO_PATH.read_text(encoding="utf-8"))
    if "model_type" not in row and "model_architecture" in row:
        row["model_type"] = row["model_architecture"]
    if "dataset_target_column" not in row and "target_column" in row:
        row["dataset_target_column"] = row["target_column"]
    return row


def write_best_model_metadata(row: dict[str, Any], output_path: Path = BEST_MODEL_INFO_PATH) -> Path:
    features_by_lead = _parse_feature_names_by_lead(row.get("selected_feature_names_by_lead"))
    weights_by_lead = _parse_weights_by_lead(row.get("model_weights_path_by_lead"))
    all_features = sorted({feature for features in features_by_lead.values() for feature in features})
    lag_steps = [int(v) for v in (_parse_pipe_list(row.get("dataset_feature_lag_steps")) or _parse_pipe_list(row.get("dataset_target_lag_steps")))]

    payload = dict(row)
    payload.update({
        "run_id": str(row.get("run_id", "")),
        "model_architecture": str(row.get("model_type", "")),
        "training_cutoff": str(row.get("training_cutoff_date", "")),
        "inundation_product": str(row.get("inundation_product", "")),
        "target_column": str(row.get("dataset_target_column", "")),
        "target_type": str(row.get("target_type", "")),
        "autoregressive_values": _as_bool(row.get("autoregressive", False)),
        "lag_steps": lag_steps,
        "lead_count": int(float(row.get("dataset_forecast_horizon_periods", 1))),
        "calibration": float(row.get("calibration", float("nan"))),
        "roc_auc": float(row.get("peak_auc", float("nan"))),
        "f1_score": float(row.get("peak_f1", float("nan"))),
        "selected_feature_names": all_features,
        "selected_feature_names_by_lead": {str(k): v for k, v in sorted(features_by_lead.items())},
        "model_weights_path_by_lead": {str(k): v for k, v in sorted(weights_by_lead.items())},
    })
    if "dataset_target_column" not in payload and "target_column" in payload:
        payload["dataset_target_column"] = payload["target_column"]
    if "model_type" not in payload and "model_architecture" in payload:
        payload["model_type"] = payload["model_architecture"]

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def update_temporal_sources_for_best_model(row: dict[str, Any]) -> None:
    _sync_runtime_config_from_row(row)

    process_victoria.update_victoria()
    process_albert.update_albert()
    process_kyoga.update_kyoga()
    process_rainfall.update_rainfall()
    process_teleconnections.update_teleconnections()

    product = str(row["inundation_product"]).strip().lower()
    if product == "viirs":
        process_inundation_viirs.update_inundation()
        if bool(get_cfg("runtime.viirs.use_preliminary_for_deployment", True)):
            operational_h5, operational_csv, operational_downloads = (
                process_inundation_viirs.prepare_operational_inundation()
            )
            cfg = get_config()
            cfg.setdefault("runtime", {})["operational_target_temporal_path"] = operational_csv
            cfg.setdefault("paths", {}).setdefault("historic", {})["viirs_h5"] = operational_h5
            cfg["paths"].setdefault("downloads", {})["inundation_viirs"] = operational_downloads
    else:
        raise ValueError(f"Unsupported inundation product '{product}'.")

    process_gridded_rainfall.update_gridded_rainfall()
    process_gridded_moisture.update_gridded_moisture()


def retrain_best_model_on_available_data(row: dict[str, Any]) -> dict[str, Any]:
    """Retrain selected lead models on all currently available labeled data.

    The runtime retraining keeps the selected architecture, feature subsets,
    target transform, and preprocessing pipeline from the chosen best row,
    while refreshing model weights on all observations available up to the
    most recent observed target date.
    """

    _sync_runtime_config_from_row(row)
    product = str(row["inundation_product"]).strip().lower()
    target_type = str(row["target_type"]).strip()
    autoregressive = _as_bool(row.get("autoregressive", False))
    seed = int(float(row.get("seed", 42)))
    model_type = str(row.get("model_type") or row.get("model_architecture")).strip()

    feature_names_by_lead = _parse_feature_names_by_lead(row.get("selected_feature_names_by_lead"))
    if not feature_names_by_lead:
        raise RuntimeError("Cannot retrain runtime model: selected_feature_names_by_lead is empty.")

    lead_count = int(float(row.get("dataset_forecast_horizon_periods", row.get("dataset_lead_period_max", 1))))

    raw_temporal, cumulative_sources, _, _ = _build_temporal_feature_table(product)
    raw_temporal = raw_temporal.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    observed_target_history = raw_temporal.dropna(subset=["target_raw"]).copy()
    if observed_target_history.empty:
        raise RuntimeError("Cannot retrain runtime model without observed target history.")

    target_anchor_date = pd.to_datetime(observed_target_history["date"]).max()
    transformed_target = _target_transform(
        raw_temporal[["date", "target_raw"]].copy(), target_type=target_type,
        product=product, training_cutoff_date=str(target_anchor_date.date()),
    )
    non_null_target_raw = int(raw_temporal["target_raw"].notna().sum())
    non_null_transformed_target = int(transformed_target["target"].notna().sum())
    logging.info(
        "Runtime retraining target coverage: %d non-null target_raw values, %d non-null transformed targets, anchor=%s.",
        non_null_target_raw,
        non_null_transformed_target,
        target_anchor_date.date(),
    )

    base_features = raw_temporal.drop(columns=["target_raw"]).copy()
    if "period_order" in base_features.columns and bool(get_cfg("ablation.pipeline.drop_metadata_columns", True)):
        base_features = base_features.drop(columns=["period_order"])

    features_with_blocks, _, _ = _add_optional_feature_blocks(
        base_features,
        target_for_lags=transformed_target["target"],
        autoregressive=autoregressive,
    )
    timestamp_tag = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    RUNTIME_RETRAINED_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    origin_dates = pd.to_datetime(features_with_blocks["date"])
    saved_scaler = joblib.load(str(row["dataset_scaler_path"]))
    unavailable_trained_features = [
        name for name in saved_scaler["feature_cols"] if name not in features_with_blocks.columns
    ]
    runtime_scaler_path = str(row["dataset_scaler_path"])
    runtime_pca_path = str(row.get("dataset_pca_path", ""))

    if unavailable_trained_features:
        runtime_scaler_path = str(
            RUNTIME_RETRAINED_WEIGHTS_DIR / f"preprocessor_{product}_{timestamp_tag}.pkl"
        )
        preprocessing_mask = origin_dates <= target_anchor_date
        transformed_features = _fit_train_only_transforms(
            features_with_blocks,
            train_mask=preprocessing_mask,
            scaler_output_path=runtime_scaler_path,
            seasonally_standardize_features=_as_bool(
                row.get("dataset_seasonally_standardized_features", False)
            ),
        )
        transformed_features = _add_post_scale_cumulative_features(
            transformed_features, cumulative_sources
        )
        if _as_bool(row.get("dataset_pca_enabled", False)):
            saved_pca = joblib.load(str(row["dataset_pca_path"]))
            runtime_pca_path = str(
                RUNTIME_RETRAINED_WEIGHTS_DIR / f"pca_{product}_{timestamp_tag}.pkl"
            )
            transformed_features, _ = apply_train_only_pca(
                feature_df=transformed_features,
                train_mask=preprocessing_mask,
                output_path=runtime_pca_path,
                n_components=min(
                    int(saved_pca["pca"].n_components_),
                    len(transformed_features.columns) - 1,
                ),
                whiten=bool(saved_pca["pca"].whiten),
                seed=seed,
            )
        for lead, names in feature_names_by_lead.items():
            feature_names_by_lead[lead] = [
                name for name in names if name in transformed_features.columns
            ]
            if not feature_names_by_lead[lead]:
                raise RuntimeError(
                    f"All selected features for lead {lead} were removed by the recent-data rule."
                )
        logging.warning(
            "Runtime preprocessing was refit after stale products removed %d trained features.",
            len(unavailable_trained_features),
        )
    else:
        transformed_features = _apply_saved_feature_transforms(
            features_with_blocks, cumulative_sources, row
        )

    origin_dates = pd.to_datetime(transformed_features["date"])
    runtime_weights_by_lead: dict[str, str] = {}

    for lead in range(1, lead_count + 1):
        if lead not in feature_names_by_lead:
            raise RuntimeError(f"Missing selected feature names for lead {lead} in chosen best row.")

        selected_features = feature_names_by_lead[lead]
        missing = [feature for feature in selected_features if feature not in transformed_features.columns]
        if missing:
            raise RuntimeError(f"Runtime retraining missing selected features for lead {lead}: {missing[:10]}")

        direct_target = _direct_forecast_target(
            raw_temporal[["date", "target_raw"]], target_type=target_type,
            horizon=lead, training_cutoff_date=str(target_anchor_date.date()),
        )
        y_shifted = direct_target["target"]
        train_mask = (origin_dates <= target_anchor_date) & y_shifted.notna()
        shifted_label_count = int(y_shifted.notna().sum())
        trainable_row_count = int(train_mask.sum())
        logging.info(
            "Runtime retraining lead %d coverage: %d non-null shifted labels, %d trainable rows.",
            lead,
            shifted_label_count,
            trainable_row_count,
        )

        if trainable_row_count == 0:
            raise RuntimeError(f"Runtime retraining has no labeled samples for lead {lead}.")

        x_train = transformed_features.loc[train_mask, selected_features].to_numpy(dtype=np.float64)
        y_train = y_shifted.loc[train_mask].to_numpy(dtype=np.float64)

        model = TEMPORAL_MODEL_REGISTRY[model_type](seed=seed)
        model.fit(x_train, y_train)

        weights_path = RUNTIME_RETRAINED_WEIGHTS_DIR / (
            f"{model_type}_{product}_{target_type}_runtime_fullhistory_lead{lead}_{timestamp_tag}.pkl"
        )
        model.save_weights(str(weights_path))
        runtime_weights_by_lead[str(lead)] = str(weights_path)

    updated_row = dict(row)
    updated_row["model_weights_path_by_lead"] = json.dumps(runtime_weights_by_lead, sort_keys=True)
    updated_row["selected_feature_names_by_lead"] = json.dumps(
        {str(lead): names for lead, names in feature_names_by_lead.items()}, sort_keys=True
    )
    updated_row["dataset_scaler_path"] = runtime_scaler_path
    updated_row["dataset_pca_path"] = runtime_pca_path
    updated_row["model_weights_path"] = runtime_weights_by_lead.get(str(lead_count), "")
    updated_row["runtime_retrained_at"] = timestamp_tag
    updated_row["runtime_retrained_until_target_date"] = str(target_anchor_date.date())
    return updated_row


def _apply_saved_feature_transforms(
    features_df: pd.DataFrame,
    cumulative_sources: list[str],
    row: dict[str, Any],
) -> pd.DataFrame:
    scaler_bundle = joblib.load(str(row["dataset_scaler_path"]))
    feature_cols = list(scaler_bundle["feature_cols"])
    missing = [col for col in feature_cols if col not in features_df.columns]
    if missing:
        raise RuntimeError(f"Current feature table is missing trained feature columns: {missing[:10]}")

    imputer = scaler_bundle["imputer"]
    scaler = scaler_bundle["scaler"]

    if (
        imputer.__class__.__name__ == "SimpleImputer"
        and not hasattr(imputer, "_fill_dtype")
        and hasattr(imputer, "_fit_dtype")
    ):
        logging.warning(
            "Loaded a legacy SimpleImputer artifact without _fill_dtype; patching it from _fit_dtype "
            "to stay compatible with the active scikit-learn runtime."
        )
        imputer._fill_dtype = imputer._fit_dtype

    raw_features = features_df[feature_cols].copy()
    if bool(scaler_bundle.get("seasonally_standardize_features", False)):
        periods = pd.to_datetime(features_df["date"]).dt.strftime("%m-%d")
        for column, stats in scaler_bundle.get("seasonal_stats", {}).items():
            means = periods.map(stats["mean"])
            stds = periods.map(stats["std"])
            fallback_mean = float(np.nanmean(list(stats["mean"].values())))
            fallback_std = float(np.nanmean(list(stats["std"].values())))
            fallback_std = fallback_std if np.isfinite(fallback_std) and fallback_std > 1e-6 else 1.0
            raw_features[column] = (
                raw_features[column] - means.fillna(fallback_mean)
            ) / stds.fillna(fallback_std).clip(lower=1e-6)
    missing_cells = int(raw_features.isna().sum().sum())
    total_cells = int(raw_features.shape[0] * raw_features.shape[1])
    imputation_ratio = missing_cells / total_cells if total_cells else 0.0
    warning_threshold = float(get_cfg("ablation.pipeline.imputation_warning_threshold", 0.10))
    if imputation_ratio >= warning_threshold:
        logging.warning(
            "High degree of imputation in deployment predictors: %.1f%% of feature cells "
            "(%d/%d) will be imputed; threshold is %.1f%%.",
            imputation_ratio * 100.0,
            missing_cells,
            total_cells,
            warning_threshold * 100.0,
        )
    else:
        logging.info(
            "Deployment predictor imputation: %.2f%% of feature cells (%d/%d).",
            imputation_ratio * 100.0,
            missing_cells,
            total_cells,
        )

    full_imputed = imputer.transform(raw_features)
    full_scaled = scaler.transform(full_imputed)
    scaled_df = pd.DataFrame(full_scaled, columns=feature_cols, index=features_df.index)
    scaled_df.insert(0, "date", features_df["date"].values)
    scaled_df = _add_post_scale_cumulative_features(scaled_df, cumulative_sources)

    if _as_bool(row.get("dataset_pca_enabled", False)):
        pca_bundle = joblib.load(str(row["dataset_pca_path"]))
        pca_feature_cols = list(pca_bundle["feature_cols"])
        missing = [col for col in pca_feature_cols if col not in scaled_df.columns]
        if missing:
            raise RuntimeError(f"Current feature table is missing PCA input columns: {missing[:10]}")
        transformed = pca_bundle["pca"].transform(scaled_df[pca_feature_cols].to_numpy(dtype=np.float64))
        pca_cols = [f"pca_{i + 1}" for i in range(transformed.shape[1])]
        out_df = pd.DataFrame(transformed, columns=pca_cols, index=scaled_df.index)
        out_df.insert(0, "date", scaled_df["date"].values)
        return out_df

    return scaled_df


def _restore_model_wrapper(model_type: str, seed: int, model_path: str):
    wrapper = TEMPORAL_MODEL_REGISTRY[model_type](seed=seed)
    payload = joblib.load(model_path)
    wrapper.model = payload
    if model_type == "persistence":
        wrapper.last_target_ = float(payload["last_target"])
        wrapper.train_std_ = float(payload.get("train_std", 0.0))
    return wrapper


def _seasonal_mean_by_month_day(history_dates: list[pd.Timestamp], history_values: np.ndarray) -> dict[str, float]:
    series = pd.Series(history_values.astype(np.float64), index=pd.to_datetime(history_dates))
    return series.groupby(series.index.strftime("%m-%d")).mean().to_dict()


def _reconstruct_future_raw_predictions(
    target_type: str,
    target_product: str,
    history_dates: list[pd.Timestamp],
    history_values: np.ndarray,
    future_dates: list[pd.Timestamp],
    transformed_point_predictions: dict[int, float],
    transformed_sample_predictions: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_type = str(target_type).strip()
    history_raw = [float(v) for v in history_values.astype(np.float64)]
    history_frame = pd.DataFrame({"date": pd.to_datetime(history_dates), "value": history_values})
    history_frame["period"] = history_frame["date"].dt.strftime("%m-%d")
    level_stats = history_frame.groupby("period")["value"].agg(["mean", "std"])
    difference_frame = history_frame.copy()
    difference_frame["difference"] = difference_frame["value"].diff(1)
    difference_stats = difference_frame.groupby("period")["difference"].agg(["mean", "std"])

    point_predictions: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for lead_index, future_date in enumerate(future_dates, start=1):
        point_transformed = float(transformed_point_predictions[lead_index])
        sample_transformed = np.asarray(transformed_sample_predictions[lead_index], dtype=np.float64)

        if target_type == "raw":
            point_raw = point_transformed
            sample_raw = sample_transformed
        elif target_type == "deseasonalised":
            period = future_date.strftime("%m-%d")
            component = float(level_stats["mean"].get(period, np.mean(history_raw)))
            scale = max(float(level_stats["std"].get(period, np.std(history_raw))), 1e-6)
            point_raw = point_transformed * scale + component
            sample_raw = sample_transformed * scale + component
        elif target_type == "seasonally_differenced":
            intervening = future_dates[:lead_index]
            means = [float(difference_stats["mean"].get(date.strftime("%m-%d"), 0.0)) for date in intervening]
            fallback_std = float(difference_frame["difference"].std())
            fallback_std = fallback_std if np.isfinite(fallback_std) and fallback_std > 1e-6 else 1.0
            stds = [float(difference_stats["std"].get(date.strftime("%m-%d"), fallback_std)) for date in intervening]
            component = float(history_values[-1]) + float(np.sum(means))
            scale = float(np.sqrt(np.sum(np.maximum(np.asarray(stds), 1e-6) ** 2)))
            point_raw = point_transformed * scale + component
            sample_raw = sample_transformed * scale + component
        elif target_type in {"first_differenced", "differenced_anomaly"}:
            point_raw = point_transformed + float(history_values[-1])
            sample_raw = sample_transformed + float(history_values[-1])
        else:
            raise NotImplementedError(
                f"Future raw reconstruction is not yet implemented for target_type '{target_type}'."
            )

        point_raw = max(point_raw, 0.0)
        sample_raw = np.maximum(sample_raw, 0.0)
        point_predictions.append(point_raw)
        interval = float(get_cfg("forecast.interval_percent", 80)) / 100.0
        if not 0.0 < interval < 1.0:
            raise ValueError("forecast.interval_percent must be between 0 and 100")
        tail = (1.0 - interval) / 2.0
        lower_bounds.append(float(np.quantile(sample_raw, tail)))
        upper_bounds.append(float(np.quantile(sample_raw, 1.0 - tail)))

        # Lead models predict directly from the common t=0; do not feed a
        # shorter-lead prediction into a longer-lead inverse transform.

    return (
        np.asarray(point_predictions, dtype=np.float64),
        np.asarray(lower_bounds, dtype=np.float64),
        np.asarray(upper_bounds, dtype=np.float64),
    )


def generate_temporal_forecast(row: dict[str, Any]) -> TemporalForecastBundle:
    _sync_runtime_config_from_row(row)
    product = str(row["inundation_product"]).strip().lower()
    target_column = str(row.get("dataset_target_column") or row.get("target_column") or "percent_inundation").strip()
    target_type = str(row["target_type"]).strip()
    autoregressive = _as_bool(row.get("autoregressive", False))
    seed = int(float(row.get("seed", 42)))
    lead_count = int(float(row.get("dataset_forecast_horizon_periods", row.get("dataset_lead_period_max", 1))))

    raw_temporal, cumulative_sources, _, _ = _build_temporal_feature_table(product)
    raw_temporal = raw_temporal.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    observed_target_history = raw_temporal.dropna(subset=["target_raw"]).copy()
    if observed_target_history.empty:
        raise RuntimeError("Cannot generate forecast without observed target history.")
    target_anchor_date = pd.to_datetime(observed_target_history["date"]).max()
    transformed_target = _target_transform(
        raw_temporal[["date", "target_raw"]].copy(), target_type=target_type,
        product=product, training_cutoff_date=str(target_anchor_date.date()),
    )

    base_features = raw_temporal.drop(columns=["target_raw"]).copy()
    if "period_order" in base_features.columns and bool(get_cfg("ablation.pipeline.drop_metadata_columns", True)):
        base_features = base_features.drop(columns=["period_order"])

    features_with_blocks, _, _ = _add_optional_feature_blocks(
        base_features,
        target_for_lags=transformed_target["target"],
        autoregressive=autoregressive,
    )
    transformed_features = _apply_saved_feature_transforms(features_with_blocks, cumulative_sources, row)
    eligible_features = transformed_features[pd.to_datetime(transformed_features["date"]) <= target_anchor_date]
    if eligible_features.empty:
        raise RuntimeError(
            "Cannot find feature row on or before the last observed target date for forecast anchoring."
        )
    latest_features = eligible_features.iloc[-1]
    origin_date = target_anchor_date

    forecast_end = origin_date + timedelta(days=int(get_cfg(
        "forecast.date_generation_buffer_days", 120
    )))
    future_date_strings = cleaning_utils.get_dates_of_interest(
        origin_date.strftime("%Y-%m-%d"),
        forecast_end.strftime("%Y-%m-%d"),
        target_product=product,
    )[1 : lead_count + 1]
    future_dates = [pd.Timestamp(date_str) for date_str in future_date_strings]

    feature_names_by_lead = _parse_feature_names_by_lead(row.get("selected_feature_names_by_lead"))
    weights_by_lead = _parse_weights_by_lead(row.get("model_weights_path_by_lead"))

    transformed_point_predictions: dict[int, float] = {}
    transformed_sample_predictions: dict[int, np.ndarray] = {}
    for lead in range(1, lead_count + 1):
        if lead not in feature_names_by_lead or lead not in weights_by_lead:
            raise RuntimeError(f"Missing saved features or weights for lead {lead}.")

        selected_features = feature_names_by_lead[lead]
        missing = [feature for feature in selected_features if feature not in transformed_features.columns]
        if missing:
            raise RuntimeError(f"Latest feature frame is missing selected features for lead {lead}: {missing[:10]}")

        x_latest = latest_features[selected_features].to_numpy(dtype=np.float64).reshape(1, -1)
        model = _restore_model_wrapper(
            str(row.get("model_type") or row.get("model_architecture")),
            seed,
            weights_by_lead[lead],
        )
        point_pred = np.asarray(model.predict(x_latest), dtype=np.float64).reshape(-1)
        sample_pred = np.asarray(model.predict_samples(x_latest), dtype=np.float64)
        if sample_pred.ndim == 1:
            sample_pred = sample_pred[np.newaxis, :]
        if sample_pred.shape[1] != 1:
            raise RuntimeError(f"Expected a single forecast sample column for lead {lead}, got shape {sample_pred.shape}")

        transformed_point_predictions[lead] = float(point_pred[0])
        transformed_sample_predictions[lead] = sample_pred[:, 0]

    history_dates = [pd.Timestamp(value) for value in pd.to_datetime(observed_target_history["date"]).tolist()]
    history_values = observed_target_history["target_raw"].to_numpy(dtype=np.float64)
    predicted_raw, lower_raw, upper_raw = _reconstruct_future_raw_predictions(
        target_type=target_type,
        target_product=product,
        history_dates=history_dates,
        history_values=history_values,
        future_dates=future_dates,
        transformed_point_predictions=transformed_point_predictions,
        transformed_sample_predictions=transformed_sample_predictions,
    )

    return TemporalForecastBundle(
        target_column=target_column,
        target_product=product,
        model_row=row,
        origin_date=origin_date,
        future_dates=future_dates,
        history_dates=history_dates,
        history_values_raw=history_values,
        predicted_raw=predicted_raw,
        lower_raw=lower_raw,
        upper_raw=upper_raw,
    )
