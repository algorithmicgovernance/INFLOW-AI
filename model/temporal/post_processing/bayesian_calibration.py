"""Lead-specific Bayesian Student-t calibration of the published temporal model."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd

from processing.config import get_cfg
from model.temporal.ablation.metrics import compute_metrics
from model.temporal.ablation.temporal_ablation_runner import (
    _average_metrics_across_leads,
    _derive_event_threshold_from_training_rel_true,
    _resolve_extreme_event_top_fraction,
    _rolling_dry_baseline_past_365_days,
)
from model.temporal.ablation.temporal_best_model_viz import _render_svg
from model.temporal.ablation.temporal_data_pipeline import prepare_temporal_dataset
from model.temporal.ablation.temporal_model_candidates import TEMPORAL_MODEL_REGISTRY
from model.temporal.runtime.area_reconversion import convert_target_fraction_to_km2


@dataclass
class CalibrationData:
    lead: int
    feature_names: list[str]
    train_dates: np.ndarray
    test_dates: np.ndarray
    history_dates: np.ndarray
    history_raw: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    original_train_raw: np.ndarray
    original_test_raw: np.ndarray
    original_test_samples_raw: np.ndarray
    covariates_train: np.ndarray
    covariates_test: np.ndarray


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else json.loads(str(value))


def _selected_names(row: dict[str, Any], lead: int) -> list[str]:
    value = _parse_mapping(row["selected_feature_names_by_lead"])[str(lead)]
    return [str(item) for item in value] if isinstance(value, list) else [item for item in str(value).split("|") if item]


def _load_calibration_data(row: dict[str, Any], lead: int) -> CalibrationData:
    if bool(row.get("dataset_joint_multi_horizon_model", False)) or row.get("model_type") == "transformer":
        raise NotImplementedError(
            "Bayesian post-processing currently requires the deployed lead-specific model artifacts."
        )
    dataset = prepare_temporal_dataset(
        inundation_product=str(row["inundation_product"]),
        target_type=str(row["target_type"]),
        autoregressive=_as_bool(row.get("autoregressive", False)),
        training_cutoff_date=str(row["training_cutoff_date"]),
        run_id=str(row["run_id"]),
        model_type=str(row["model_type"]),
        forecast_horizon_periods_override=lead,
        include_ecmwf_hindcasts=_as_bool(row.get("include_ecmwf_hindcasts", True)),
        include_target_lag_features=_as_bool(row.get("include_target_lag_features", True)),
        include_feature_lag_features=_as_bool(row.get("include_feature_lag_features", True)),
        seasonally_standardize_features=_as_bool(
            row.get("dataset_seasonally_standardized_features", row.get("seasonally_standardize_features", False))
        ),
    )
    names = _selected_names(row, lead)
    indices = [dataset.feature_names.index(name) for name in names]
    weights = _parse_mapping(row["model_weights_path_by_lead"])
    wrapper = TEMPORAL_MODEL_REGISTRY[str(row["model_type"])](seed=int(float(row.get("seed", 42))))
    wrapper.model = joblib.load(str(weights[str(lead)]))
    x_train = dataset.X_train[:, indices]
    x_test = dataset.X_test[:, indices]
    if hasattr(wrapper, "_residual_distribution") and wrapper._residual_distribution().size < 2:
        wrapper._store_training_residuals(x_train, dataset.y_train)

    train_pred = np.asarray(wrapper.predict(x_train), dtype=np.float64)
    test_pred = np.asarray(wrapper.predict(x_test), dtype=np.float64)
    test_samples = np.asarray(wrapper.predict_samples(x_test), dtype=np.float64)
    train_raw = np.maximum(
        train_pred * dataset.raw_reconstruction_scale_train
        + dataset.raw_reconstruction_component_train, 0.0,
    )
    test_raw = np.maximum(
        test_pred * dataset.raw_reconstruction_scale_test
        + dataset.raw_reconstruction_component_test, 0.0,
    )
    test_samples_raw = np.maximum(
        test_samples * dataset.raw_reconstruction_scale_test[np.newaxis, :]
        + dataset.raw_reconstruction_component_test[np.newaxis, :], 0.0,
    )
    return CalibrationData(
        lead, names, dataset.train_dates, dataset.test_dates,
        dataset.history_dates, dataset.history_target_raw,
        dataset.y_train_raw.astype(np.float64), dataset.y_test_raw.astype(np.float64),
        train_raw, test_raw, test_samples_raw, x_train.astype(np.float64), x_test.astype(np.float64),
    )


def _standardize_design(
    prediction_train: np.ndarray, prediction_test: np.ndarray,
    covariates_train: np.ndarray, covariates_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    train = np.column_stack([prediction_train, covariates_train])
    test = np.column_stack([prediction_test, covariates_test])
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return (train - mean) / std, (test - mean) / std, {
        "mean": mean.tolist(), "std": std.tolist(),
    }


def _fit_student_t_nuts(
    x_train: np.ndarray, correction_train: np.ndarray, *, settings: dict[str, Any], lead: int,
):
    cache_root = Path(tempfile.gettempdir()) / "inflow-pytensor"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTENSOR_FLAGS", f"base_compiledir={cache_root}")
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "Bayesian calibration requires PyMC and ArviZ. Install project requirements first."
        ) from exc

    correction_mean = float(np.mean(correction_train))
    correction_scale = float(np.std(correction_train))
    if not np.isfinite(correction_scale) or correction_scale < 1e-8:
        correction_scale = 1.0
    y = (correction_train - correction_mean) / correction_scale
    coords = {"coefficient": np.arange(x_train.shape[1])}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="coefficient")
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu_minus_two = pm.Exponential("nu_minus_two", lam=1.0 / 10.0)
        nu = pm.Deterministic("nu", nu_minus_two + 2.0)
        mu = alpha + pm.math.dot(x_train, beta)
        pm.StudentT("correction", nu=nu, mu=mu, sigma=sigma, observed=y)
        inference = pm.sample(
            draws=int(settings["draws"]), tune=int(settings["tune"]),
            chains=int(settings["chains"]), cores=int(settings["cores"]),
            target_accept=float(settings["target_accept"]),
            random_seed=int(settings["random_seed"]) + lead,
            return_inferencedata=True, progressbar=True,
        )
    return inference, correction_mean, correction_scale, az


def _posterior_predictive_draws(
    inference, x_test: np.ndarray, correction_mean: float, correction_scale: float,
    original_prediction: np.ndarray, *, count: int, seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    posterior = inference.posterior
    alpha = np.asarray(posterior["alpha"]).reshape(-1)
    beta = np.asarray(posterior["beta"]).reshape(-1, x_test.shape[1])
    sigma = np.asarray(posterior["sigma"]).reshape(-1)
    nu = np.asarray(posterior["nu"]).reshape(-1)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(alpha), size=min(count, len(alpha)), replace=False)
    mu = alpha[selected, np.newaxis] + beta[selected] @ x_test.T
    standardized = mu + sigma[selected, np.newaxis] * rng.standard_t(
        nu[selected, np.newaxis], size=mu.shape
    )
    corrections = standardized * correction_scale + correction_mean
    corrected = np.maximum(original_prediction[np.newaxis, :] + corrections, 0.0)
    return corrected, {
        "alpha": alpha, "beta": beta, "sigma": sigma, "nu": nu,
        "selected_posterior_indices": selected,
    }


def _save_inference_data(inference, output_path: str | Path, arviz_module=None) -> Path:
    """Persist an ArviZ InferenceData across old and new ArviZ APIs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(inference, "to_netcdf"):
        inference.to_netcdf(str(output_path))
    elif arviz_module is not None and hasattr(arviz_module, "to_netcdf"):
        arviz_module.to_netcdf(inference, str(output_path))
    else:
        raise RuntimeError(
            "The installed ArviZ version cannot serialize InferenceData to NetCDF."
        )
    return output_path


def _plot_corrected(
    row: dict[str, Any], data: CalibrationData, corrected_samples: np.ndarray,
    metrics: dict[str, float], output_path: Path,
) -> None:
    cutoff = pd.Timestamp(row["training_cutoff_date"])
    history_dates = pd.to_datetime(data.history_dates)
    history_mask = history_dates <= cutoff
    actual_dates = list(history_dates[history_mask]) + list(pd.to_datetime(data.test_dates))
    actual_raw = list(data.history_raw[history_mask]) + list(data.y_test_raw)
    point = np.mean(corrected_samples, axis=0)
    lower = np.maximum(np.quantile(corrected_samples, 0.025, axis=0), 0.0)
    upper = np.quantile(corrected_samples, 0.975, axis=0)
    target_column = str(row.get("dataset_target_column", "percent_inundation"))
    actual_km2 = convert_target_fraction_to_km2(actual_raw, target_column).tolist()
    point_km2 = convert_target_fraction_to_km2(point, target_column).tolist()
    lower_km2 = np.maximum(convert_target_fraction_to_km2(lower, target_column), 0.0).tolist()
    upper_km2 = convert_target_fraction_to_km2(upper, target_column).tolist()
    title = (
        f"Best Model (Bayesian corrected): {row['model_type']} | Lead {data.lead} | "
        f"F1 {metrics.get('peak_f1', float('nan')):.3f}"
    )
    _render_svg(
        output_path, title, actual_dates, actual_km2, list(pd.to_datetime(data.test_dates)),
        point_km2, lower_km2, upper_km2, cutoff.to_pydatetime(),
    )


def _settings() -> dict[str, Any]:
    prefix = "ablation.post_processing.bayesian_calibration"
    return {
        key: get_cfg(f"{prefix}.{key}", default)
        for key, default in {
            "draws": 1000, "tune": 1000, "chains": 4, "cores": 1,
            "target_accept": 0.90, "posterior_predictive_draws": 1000,
            "random_seed": 42,
            "output_dir": "model/temporal/post_processing/artifacts",
        }.items()
    }


def run_bayesian_calibration(
    best_model_path: str | Path = "model/temporal/best_temporal_model.json",
) -> dict[str, Any]:
    settings = _settings()
    row = json.loads(Path(best_model_path).read_text(encoding="utf-8"))
    output_dir = Path(str(settings["output_dir"])) / str(row["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison: dict[str, Any] = {
        "source_run_id": row["run_id"], "model_type": row["model_type"],
        "target_type": row["target_type"], "training_cutoff_date": row["training_cutoff_date"],
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "mcmc": {key: settings[key] for key in ("draws", "tune", "chains", "cores", "target_accept")},
        "leads": {},
    }
    original_by_lead: dict[int, dict[str, float]] = {}
    corrected_by_lead: dict[int, dict[str, float]] = {}
    lead_count = int(float(row.get("dataset_forecast_horizon_periods", 6)))
    for lead in range(1, lead_count + 1):
        data = _load_calibration_data(row, lead)
        x_train, x_test, design_stats = _standardize_design(
            data.original_train_raw, data.original_test_raw,
            data.covariates_train, data.covariates_test,
        )
        correction_train = data.y_train_raw - data.original_train_raw
        inference, correction_mean, correction_scale, az = _fit_student_t_nuts(
            x_train, correction_train, settings=settings, lead=lead,
        )
        summary = az.summary(inference, var_names=["alpha", "beta", "sigma", "nu"])
        diagnostics = {
            "maximum_r_hat": float(summary["r_hat"].max()),
            "minimum_bulk_ess": float(summary["ess_bulk"].min()),
            "minimum_tail_ess": float(summary["ess_tail"].min()),
            "divergences": int(np.asarray(inference.sample_stats["diverging"]).sum()),
        }
        posterior_path = output_dir / f"lead_{lead}_posterior.nc"
        _save_inference_data(inference, posterior_path, az)
        corrected_samples, posterior_arrays = _posterior_predictive_draws(
            inference, x_test, correction_mean, correction_scale, data.original_test_raw,
            count=int(settings["posterior_predictive_draws"]),
            seed=int(settings["random_seed"]) + lead,
        )
        draws_path = output_dir / f"lead_{lead}_posterior_draws.npz"
        np.savez_compressed(
            draws_path, corrected_test_raw=corrected_samples,
            original_test_raw=data.original_test_raw, observed_test_raw=data.y_test_raw,
            test_dates=data.test_dates, feature_names=np.asarray(["original_prediction", *data.feature_names]),
            design_mean=np.asarray(design_stats["mean"]), design_std=np.asarray(design_stats["std"]),
            correction_mean=correction_mean, correction_scale=correction_scale,
            **posterior_arrays,
        )
        dry_test = _rolling_dry_baseline_past_365_days(
            data.history_dates, data.history_raw, data.test_dates
        )
        dry_train = _rolling_dry_baseline_past_365_days(
            data.history_dates, data.history_raw, data.train_dates
        )
        threshold = _derive_event_threshold_from_training_rel_true(
            data.y_train_raw, dry_train, _resolve_extreme_event_top_fraction()
        )
        original_metrics = compute_metrics(
            data.y_test_raw, data.original_test_raw, data.original_test_samples_raw,
            dry_test, threshold,
        ).to_dict()
        corrected_point = np.mean(corrected_samples, axis=0)
        corrected_metrics = compute_metrics(
            data.y_test_raw, corrected_point, corrected_samples, dry_test, threshold,
        ).to_dict()
        original_by_lead[lead] = original_metrics
        corrected_by_lead[lead] = corrected_metrics
        plot_path = output_dir / f"best_model_corrected_lead{lead}.png"
        _plot_corrected(row, data, corrected_samples, corrected_metrics, plot_path)
        comparison["leads"][f"lead_periods_{lead}"] = {
            "original": original_metrics, "bayesian_corrected": corrected_metrics,
            "posterior_path": str(posterior_path), "posterior_draws_path": str(draws_path),
            "corrected_plot_path": str(plot_path), "selected_covariates": data.feature_names,
            "mcmc_diagnostics": diagnostics,
        }
    comparison["average_original"] = _average_metrics_across_leads(original_by_lead)
    comparison["average_bayesian_corrected"] = _average_metrics_across_leads(corrected_by_lead)
    comparison_path = output_dir / "bayesian_calibration_comparison.json"
    comparison["comparison_path"] = str(comparison_path)
    comparison_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    return comparison
