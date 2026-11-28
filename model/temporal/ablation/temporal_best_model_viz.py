from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import joblib
import pandas as pd

from processing.config import get_cfg
from .temporal_data_pipeline import prepare_temporal_dataset
from .temporal_ablation_runner import _recursive_raw_reconstruction
from .temporal_model_candidates import TEMPORAL_MODEL_REGISTRY
from .utils import set_global_seed
from ..runtime.area_reconversion import convert_target_fraction_to_km2


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_svg_points(values: list[float], width: int, height: int, padding: int = 40) -> str:
    if not values:
        return ""
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    span = max_value - min_value
    points: list[str] = []
    for index, value in enumerate(values):
        x = padding + (index / max(len(values) - 1, 1)) * (width - 2 * padding)
        y = height - padding - ((value - min_value) / span) * (height - 2 * padding)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _format_title(row: dict[str, Any], lead_period: int, metrics: dict[str, Any]) -> str:
    return (
        f"Best Model: {row.get('model_type', 'unknown')} | "
        f"run_id={row.get('run_id', 'unknown')} | target_type={row.get('target_type', 'unknown')} | "
        f"autoregressive={row.get('autoregressive', 'unknown')} | "
        f"Lead {lead_period} | F1 {metrics.get('peak_f1', float('nan')):.3f}"
    )


def _compute_predicted_series(row: dict[str, Any], lead_period: int) -> tuple[
    list[datetime], list[float], list[datetime], list[float], list[float], list[float],
    list[datetime], list[float],
]:
    training_cutoff_date = str(row.get("training_cutoff_date") or get_cfg("ablation.experiments.training_cutoff_dates", [""])[0])
    seed = int(row.get("seed", get_cfg("ablation.experiments.seed", 42)))
    config_row = {
        "model_type": str(row.get("model_type", "")),
        "target_type": str(row.get("target_type", "")),
        "autoregressive": _coerce_bool(row.get("autoregressive", False)),
        "inundation_product": str(row.get("inundation_product", "viirs")),
        "training_cutoff_date": training_cutoff_date,
        "seed": seed,
    }

    set_global_seed(seed)
    selected_by_lead = json.loads(str(row["selected_feature_names_by_lead"]))
    weights_by_lead = json.loads(str(row["model_weights_path_by_lead"]))
    all_leads = sorted(int(value) for value in selected_by_lead)
    if lead_period not in all_leads:
        raise ValueError(f"Lead {lead_period} is not present in selected_feature_names_by_lead.")

    # Load every direct lead model and align them to the common forecast origins
    # used by ablation scoring. Each inverse term is derived from training data
    # and/or the observed value at t=0, never an observation between t and t+h.
    lead_outputs: dict[int, tuple[Any, np.ndarray, np.ndarray]] = {}
    for lead in all_leads:
        dataset = prepare_temporal_dataset(
            inundation_product=config_row["inundation_product"],
            target_type=config_row["target_type"],
            autoregressive=config_row["autoregressive"],
            training_cutoff_date=config_row["training_cutoff_date"],
            run_id=str(row.get("run_id", "")),
            model_type=config_row["model_type"],
            forecast_horizon_periods_override=lead,
            include_ecmwf_hindcasts=_coerce_bool(row.get("include_ecmwf_hindcasts", True)),
            include_target_lag_features=_coerce_bool(row.get("include_target_lag_features", True)),
            include_feature_lag_features=_coerce_bool(row.get("include_feature_lag_features", True)),
        )
        selected_names = selected_by_lead[str(lead)]
        if isinstance(selected_names, str):
            selected_names = [name for name in selected_names.split("|") if name]
        indices = [dataset.feature_names.index(name) for name in selected_names]
        model = TEMPORAL_MODEL_REGISTRY[config_row["model_type"]](seed=seed)
        model.model = joblib.load(weights_by_lead[str(lead)])
        if config_row["model_type"] == "persistence":
            model.last_target_ = float(model.model["last_target"])
            model.train_std_ = float(model.model.get("train_std", 0.0))
        elif hasattr(model, "_residual_distribution") and model._residual_distribution().size < 2:
            # Backfill uncertainty for weights saved before residual distributions
            # became part of model artifacts.
            model._store_training_residuals(dataset.X_train[:, indices], dataset.y_train)
        x_test = dataset.X_test[:, indices]
        lead_outputs[lead] = (
            dataset, np.asarray(model.predict(x_test)), np.asarray(model.predict_samples(x_test))
        )

    reference_dataset = lead_outputs[max(all_leads)][0]
    common_origins = set(pd.to_datetime(reference_dataset.origin_test_dates))
    for dataset, _, _ in lead_outputs.values():
        common_origins.intersection_update(pd.to_datetime(dataset.origin_test_dates))
    ordered_origins = np.asarray([
        date for date in reference_dataset.origin_test_dates
        if pd.Timestamp(date) in common_origins
    ])
    if ordered_origins.size == 0:
        raise RuntimeError("No common held-out origins across plotted forecast leads.")

    point_columns = []
    sample_columns = []
    component_columns = []
    scale_columns = []
    true_columns = []
    target_date_columns = []
    for lead in all_leads:
        dataset, predictions, samples = lead_outputs[lead]
        positions = {pd.Timestamp(date): index for index, date in enumerate(dataset.origin_test_dates)}
        indices = np.asarray([positions[pd.Timestamp(date)] for date in ordered_origins], dtype=int)
        point_columns.append(predictions[indices])
        sample_columns.append(samples[:, indices])
        component_columns.append(dataset.raw_reconstruction_component_test[indices])
        scale_columns.append(dataset.raw_reconstruction_scale_test[indices])
        true_columns.append(dataset.y_test_raw[indices])
        target_date_columns.append(dataset.test_dates[indices])

    transformed_points = np.stack(point_columns, axis=1)
    transformed_samples = np.stack(sample_columns, axis=2)
    components = np.stack(component_columns, axis=1)
    scales = np.stack(scale_columns, axis=1)
    y_pred_raw_all, y_samples_raw_all = _recursive_raw_reconstruction(
        target_type=config_row["target_type"],
        transformed_points=transformed_points,
        transformed_samples=transformed_samples,
        reconstruction_components=components,
        origin_dates=ordered_origins,
        history_dates=reference_dataset.history_dates,
        history_target_raw=reference_dataset.history_target_raw,
        reconstruction_scales=scales,
    )
    lead_index = all_leads.index(lead_period)
    dataset = lead_outputs[lead_period][0]
    y_pred_raw = y_pred_raw_all[:, lead_index]
    lower_95 = np.maximum(np.quantile(y_samples_raw_all[:, :, lead_index], 0.025, axis=0), 0.0)
    upper_95 = np.maximum(np.quantile(y_samples_raw_all[:, :, lead_index], 0.975, axis=0), lower_95)
    aligned_test_dates = np.stack(target_date_columns, axis=1)[:, lead_index]
    aligned_test_actual = np.stack(true_columns, axis=1)[:, lead_index]

    split_date = datetime.strptime(training_cutoff_date, "%Y-%m-%d")
    history_dates = [datetime.fromtimestamp(ts / 1_000_000_000.0) if isinstance(ts, (int, np.integer)) else datetime.fromisoformat(str(ts)) for ts in dataset.history_dates]
    history_values = [float(value) for value in dataset.history_target_raw]

    history_before_split = [
        (date_value, value) for date_value, value in zip(history_dates, history_values) if date_value <= split_date
    ]

    actual_dates = [date_value for date_value, _ in history_before_split]
    actual_values = [value for _, value in history_before_split]
    test_dates = [datetime.fromisoformat(str(ts)) for ts in aligned_test_dates]
    test_actual_values = [float(value) for value in aligned_test_actual]
    actual_dates.extend(test_dates)
    actual_values.extend(test_actual_values)

    return (
        actual_dates, actual_values, test_dates, [float(value) for value in y_pred_raw],
        lower_95.tolist(), upper_95.tolist(), test_dates, test_actual_values,
    )


def _render_svg(
    output_path: Path, title: str, actual_dates: list[datetime], actual_values: list[float],
    test_dates: list[datetime], pred_values: list[float], lower_95: list[float],
    upper_95: list[float], split_date: datetime | None,
) -> None:
    if not actual_dates:
        raise ValueError("No actual dates available for plotting")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates
    from matplotlib.ticker import StrMethodFormatter

    prediction_dates = test_dates

    if split_date is None:
        plot_split_date = None
    else:
        plot_split_date = split_date

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual_dates, actual_values, color="#1f77b4", linewidth=2.5, label="Actual")
    if prediction_dates and pred_values:
        ax.plot(prediction_dates, pred_values, color="#d62728", linestyle="--", linewidth=2.5, label="Predicted")
        ax.fill_between(
            prediction_dates, np.maximum(lower_95, 0.0), upper_95,
            color="#d62728", alpha=0.14, edgecolor="none", linewidth=0,
            label="95% confidence interval",
        )
    if plot_split_date is not None:
        ax.axvline(plot_split_date, color="#222", linestyle=":", linewidth=1.5, label="Train/Test Split")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Inundated Area (km²)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if actual_dates and prediction_dates:
        all_dates = actual_dates + prediction_dates
        min_date = min(all_dates)
        max_date = max(all_dates)
        ax.set_xlim(min_date - timedelta(days=30), max_date + timedelta(days=30))

    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    min_value = min(actual_values + pred_values) if pred_values else min(actual_values)
    max_value = max(actual_values + pred_values) if pred_values else max(actual_values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    ax.set_ylim(min_value, max_value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_model_results(row: dict[str, Any], split_date: str | None = None) -> list[Path]:
    if not row:
        raise ValueError("A row must be provided to plot")

    payload = row.get("metrics_by_lead_time", "") or ""
    if not payload:
        raise ValueError("The selected row does not contain metrics_by_lead_time")

    try:
        lead_metrics_payload = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("metrics_by_lead_time is not valid JSON") from exc

    lead_periods = sorted(int(key.replace("lead_periods_", "")) for key in lead_metrics_payload.keys())
    if not lead_periods:
        raise ValueError("No lead periods were found in metrics_by_lead_time")

    split_dt = datetime.strptime(split_date, "%Y-%m-%d") if split_date else None
    if split_dt is None:
        split_dt = datetime.strptime(str(row.get("training_cutoff_date", "2018-07-21")), "%Y-%m-%d")

    output_dir = Path(str(get_cfg(
        "ablation.visualization.output_dir", "model/temporal/ablation/artifacts/best_model_plots"
    )))
    output_paths: list[Path] = []
    for lead_period in lead_periods:
        metrics = lead_metrics_payload[f"lead_periods_{lead_period}"]
        actual_dates, actual_values, test_dates, pred_values, lower_95, upper_95, _, _ = _compute_predicted_series(row, lead_period)
        target_column = str(
            row.get("dataset_target_column")
            or row.get("target_column")
            or get_cfg("ablation.pipeline.target_column", "percent_inundation")
        )
        actual_values = convert_target_fraction_to_km2(actual_values, target_column).tolist()
        pred_values = convert_target_fraction_to_km2(pred_values, target_column).tolist()
        lower_95 = np.maximum(convert_target_fraction_to_km2(lower_95, target_column), 0.0).tolist()
        upper_95 = convert_target_fraction_to_km2(upper_95, target_column).tolist()
        title = _format_title(row, lead_period, metrics)
        output_path = output_dir / f"best_model_plot_lead{lead_period}.png"
        _render_svg(
            output_path, title, actual_dates, actual_values, test_dates, pred_values,
            lower_95, upper_95, split_dt,
        )
        print(f"Plot written to {output_path}")
        output_paths.append(output_path)
    return output_paths
