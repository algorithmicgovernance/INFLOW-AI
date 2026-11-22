from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from processing.config import get_cfg

from .temporal_data_pipeline import PreparedTemporalDataset, prepare_temporal_dataset
from .feature_selection import FeatureSelectionResult, select_features_with_cv
from .models.transformer_model import TransformerInput


_ECMWF_PATTERN = re.compile(r"^(ecmwf_rainfall(?:_[a-z0-9]+)?)_lead_([1-4])$")


@dataclass
class TemporalTransformerDataset:
    X_train: TransformerInput
    X_test: TransformerInput
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    reconstruction_test: np.ndarray
    train_target_dates: np.ndarray
    test_target_dates: np.ndarray
    train_origin_dates: np.ndarray
    test_origin_dates: np.ndarray
    feature_selection: FeatureSelectionResult
    configured_dataset: PreparedTemporalDataset
    history_feature_names: list[str]
    future_feature_names: list[str]


def _combined_lookup(dataset: PreparedTemporalDataset, value_name: str) -> dict[pd.Timestamp, object]:
    """Index a value by origin date without assuming identical lead splits."""
    train_name = value_name.replace("_test", "_train")
    test_name = value_name.replace("_train", "_test")
    origins = np.concatenate([dataset.origin_train_dates, dataset.origin_test_dates])
    values = np.concatenate([getattr(dataset, train_name), getattr(dataset, test_name)])
    return {pd.Timestamp(date): value for date, value in zip(origins, values)}


def _common_origin_dates(
    base: PreparedTemporalDataset, lead_datasets: list[PreparedTemporalDataset]
) -> np.ndarray:
    """Return base-ordered origins having a valid target at every lead."""
    available = []
    for dataset in lead_datasets:
        available.append(
            set(pd.to_datetime(np.concatenate([dataset.origin_train_dates, dataset.origin_test_dates])))
        )
    common = set.intersection(*available)
    base_dates = np.concatenate([base.origin_train_dates, base.origin_test_dates])
    return np.asarray([date for date in base_dates if pd.Timestamp(date) in common])


def _aligned_targets(
    base_dates: np.ndarray,
    lead_datasets: list[PreparedTemporalDataset],
    value_name: str,
) -> np.ndarray:
    lookups = [_combined_lookup(dataset, value_name) for dataset in lead_datasets]
    return np.asarray(
        [[lookup[pd.Timestamp(date)] for lookup in lookups] for date in base_dates],
        dtype=np.float32,
    )


def _aligned_target_dates(
    base_dates: np.ndarray, lead_datasets: list[PreparedTemporalDataset]
) -> np.ndarray:
    lookups = []
    for dataset in lead_datasets:
        origins = np.concatenate([dataset.origin_train_dates, dataset.origin_test_dates])
        dates = np.concatenate([dataset.train_dates, dataset.test_dates])
        lookups.append({pd.Timestamp(origin): date for origin, date in zip(origins, dates)})
    return np.asarray(
        [[lookup[pd.Timestamp(origin)] for lookup in lookups] for origin in base_dates]
    )


def _split_feature_indices(feature_names: list[str], selected_indices: np.ndarray):
    history_indices: list[int] = []
    future_entries: list[tuple[int, str, int]] = []
    for index in selected_indices.tolist():
        match = _ECMWF_PATTERN.match(feature_names[index])
        if match:
            future_entries.append((index, match.group(1), int(match.group(2))))
        elif not feature_names[index].startswith("ecmwf_"):
            history_indices.append(index)
    if not history_indices:
        raise RuntimeError("Transformer feature selection removed every historical observation feature.")
    return history_indices, future_entries


def _future_tokens(X: np.ndarray, entries: list[tuple[int, str, int]]) -> tuple[np.ndarray, list[str]]:
    channel_names = sorted({channel for _, channel, _ in entries})
    if not channel_names:
        # Retain four positional decoder queries for the no-ECMWF ablation.
        return np.zeros((len(X), 4, 1), dtype=np.float32), ["no_forecast"]
    channel_lookup = {name: index for index, name in enumerate(channel_names)}
    output = np.zeros((len(X), 4, len(channel_names)), dtype=np.float32)
    for feature_index, channel, lead in entries:
        output[:, lead - 1, channel_lookup[channel]] = X[:, feature_index]
    return output, channel_names


def prepare_temporal_transformer_dataset(
    *,
    inundation_product: str,
    target_type: str,
    autoregressive: bool,
    training_cutoff_date: str,
    run_id: str,
    seed: int,
    include_ecmwf_hindcasts: bool | None = None,
    include_target_lag_features: bool | None = None,
    include_feature_lag_features: bool | None = None,
) -> TemporalTransformerDataset:
    """Create one sequence dataset containing every output lead simultaneously."""
    configured = prepare_temporal_dataset(
        inundation_product=inundation_product,
        target_type=target_type,
        autoregressive=autoregressive,
        training_cutoff_date=training_cutoff_date,
        run_id=run_id,
        model_type="transformer",
        include_ecmwf_hindcasts=include_ecmwf_hindcasts,
        include_target_lag_features=include_target_lag_features,
        include_feature_lag_features=include_feature_lag_features,
    )
    horizon_count = int(configured.properties["dataset_forecast_horizon_periods"])
    lead_datasets = [
        prepare_temporal_dataset(
            inundation_product=inundation_product,
            target_type=target_type,
            autoregressive=autoregressive,
            training_cutoff_date=training_cutoff_date,
            run_id=run_id,
            model_type="transformer",
            forecast_horizon_periods_override=lead,
            include_ecmwf_hindcasts=include_ecmwf_hindcasts,
            include_target_lag_features=include_target_lag_features,
            include_feature_lag_features=include_feature_lag_features,
        )
        for lead in range(1, horizon_count + 1)
    ]

    # Seasonal transforms and lead shifts can produce different earliest valid
    # origins. Restrict all arrays to origins available at every output lead.
    base = lead_datasets[-1]
    all_origins = _common_origin_dates(base, lead_datasets)
    cutoff = pd.Timestamp(training_cutoff_date)
    initial_train_mask = pd.to_datetime(all_origins) <= cutoff
    initial_test_mask = ~initial_train_mask
    base_origins = np.concatenate([base.origin_train_dates, base.origin_test_dates])
    base_X = np.concatenate([base.X_train, base.X_test], axis=0)
    base_X_lookup = {pd.Timestamp(date): row for date, row in zip(base_origins, base_X)}
    all_X = np.asarray([base_X_lookup[pd.Timestamp(date)] for date in all_origins], dtype=np.float32)

    all_y = _aligned_targets(all_origins, lead_datasets, "y_train")
    all_y_raw = _aligned_targets(all_origins, lead_datasets, "y_train_raw")
    all_reconstruction = _aligned_targets(
        all_origins, lead_datasets, "raw_reconstruction_component_train"
    )
    all_target_dates = _aligned_target_dates(all_origins, lead_datasets)
    y_train = all_y[initial_train_mask]

    # Elastic-net ranking gives the neural model the same leakage-safe feature-selection
    # policy without repeatedly fitting a Transformer inside feature-count CV.
    selection_target = np.mean(y_train, axis=1)
    feature_selection = select_features_with_cv(
        X_train=all_X[initial_train_mask],
        X_test=all_X[initial_test_mask],
        y_train=selection_target,
        feature_names=base.feature_names,
        model_type="elastic_net",
        seed=seed,
        dataset_fingerprint=str(base.properties.get("dataset_fingerprint", "unknown")),
        configuration_signature={
            "model_type": "transformer",
            "training_cutoff_date": training_cutoff_date,
            "autoregressive": autoregressive,
            "target_type": target_type,
            "inundation_product": inundation_product,
            "horizon_count": horizon_count,
            "include_ecmwf_hindcasts": include_ecmwf_hindcasts,
            "include_target_lag_features": include_target_lag_features,
            "include_feature_lag_features": include_feature_lag_features,
        },
    )
    history_indices, future_entries = _split_feature_indices(
        base.feature_names, feature_selection.selected_feature_indices
    )

    history_length = int(get_cfg("ablation.pipeline.transformer.history_length", 12))
    if history_length < 1:
        raise ValueError("Transformer history_length must be at least one.")
    if len(all_X) < history_length:
        raise RuntimeError("Not enough observations for the configured Transformer history length.")

    histories = np.stack(
        [all_X[index - history_length + 1 : index + 1, history_indices] for index in range(history_length - 1, len(all_X))]
    ).astype(np.float32)
    futures, future_names = _future_tokens(all_X[history_length - 1 :], future_entries)
    origins = all_origins[history_length - 1 :]
    targets = all_y[history_length - 1 :]
    targets_raw = all_y_raw[history_length - 1 :]
    reconstruction = all_reconstruction[history_length - 1 :]
    target_dates = all_target_dates[history_length - 1 :]
    train_mask = pd.to_datetime(origins) <= cutoff
    test_mask = ~train_mask
    history_names = [base.feature_names[index] for index in history_indices]

    return TemporalTransformerDataset(
        X_train=TransformerInput(histories[train_mask], futures[train_mask]),
        X_test=TransformerInput(histories[test_mask], futures[test_mask]),
        y_train=targets[train_mask],
        y_test=targets[test_mask],
        y_train_raw=targets_raw[train_mask],
        y_test_raw=targets_raw[test_mask],
        reconstruction_test=reconstruction[test_mask],
        train_target_dates=target_dates[train_mask],
        test_target_dates=target_dates[test_mask],
        train_origin_dates=origins[train_mask],
        test_origin_dates=origins[test_mask],
        feature_selection=feature_selection,
        configured_dataset=configured,
        history_feature_names=history_names,
        future_feature_names=future_names,
    )


# Compatibility aliases for the previous generic temporal names.
TransformerDataset = TemporalTransformerDataset
prepare_transformer_dataset = prepare_temporal_transformer_dataset

