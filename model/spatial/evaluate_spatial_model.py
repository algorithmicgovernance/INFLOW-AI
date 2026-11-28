"""Validate spatial change detection against persistence and GloFAS baselines.

ConvLSTM and GloFAS scores are converted to binary tiles using the known
observed count of inundated pixels for each tile and lead. Persistence is a
strict no-change baseline: its prediction is always the original binary mask.

Evaluation treats each pixel as one of four origin-to-target classes: dry to
dry, dry to flood, flood to dry, and flood to flood. Per-class precision,
recall, F1, and IoU make onset and recession skill explicit rather than letting
the much more common stable pixels dominate the headline accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from processing.config import get_cfg, resolve_forecast_horizon_periods
from .channels import load_raster_on_target_grid
from .train_convlstm import (
    _load_static,
    _paths,
    _sample_index,
    _sequence_class,
    _validate_inputs,
)
from processing import cleaning_utils


LOG = logging.getLogger(__name__)
BASELINE_NAMES = ("convlstm", "persistence", "glofas_20_year", "glofas_100_year")
TRANSITION_NAMES = ("stable_dry", "dry_to_flood", "flood_to_dry", "stable_flood")


def _top_n_mask(scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Select exactly the requested number of highest-scoring pixels per map."""
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-np.inf)
    counts = np.asarray(counts, dtype=np.int64)
    flat = scores.reshape((-1, scores.shape[-2] * scores.shape[-1]))
    output = np.zeros_like(flat, dtype=bool)
    for index, count in enumerate(counts.reshape(-1)):
        count = int(np.clip(count, 0, flat.shape[1]))
        if count == flat.shape[1]:
            output[index] = True
        elif count:
            chosen = np.argpartition(flat[index], -count)[-count:]
            output[index, chosen] = True
    return output.reshape(scores.shape)


def _transition_classes(origin: np.ndarray, state: np.ndarray) -> np.ndarray:
    origin = np.asarray(origin, dtype=bool)
    state = np.asarray(state, dtype=bool)
    # 0 stable dry, 1 dry->flood, 2 flood->dry, 3 stable flood.
    return np.where(origin, np.where(state, 3, 2), np.where(state, 1, 0)).astype(np.uint8)


def _persistence_mask(origin: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the origin mask without using any future inundation information."""
    return np.repeat(np.asarray(origin, dtype=bool)[:, np.newaxis], horizon, axis=1)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _metrics(binary: np.ndarray, transitions: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = (float(value) for value in binary)
    total = tn + fp + fn + tp
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    transition_support = transitions.sum(axis=1).astype(np.float64)
    transition_predicted = transitions.sum(axis=0).astype(np.float64)
    transition_true_positives = np.diag(transitions).astype(np.float64)
    transition_total = float(transitions.sum())
    transition_recalls = np.asarray([
        _safe_ratio(transition_true_positives[index], transition_support[index])
        for index in range(4)
    ])
    transition_precisions = np.asarray([
        (_safe_ratio(transition_true_positives[index], transition_predicted[index])
         if transition_predicted[index] else 0.0)
        for index in range(4)
    ])
    transition_f1s = np.asarray([
        (_safe_ratio(2.0 * transition_precisions[index] * transition_recalls[index],
                     transition_precisions[index] + transition_recalls[index])
         if transition_precisions[index] + transition_recalls[index] else 0.0)
        for index in range(4)
    ])
    transition_ious = np.asarray([
        _safe_ratio(
            transition_true_positives[index],
            transition_support[index] + transition_predicted[index]
            - transition_true_positives[index],
        )
        for index in range(4)
    ])
    transition_accuracies = np.asarray([
        _safe_ratio(
            transition_total - transition_support[index] - transition_predicted[index]
            + 2.0 * transition_true_positives[index],
            transition_total,
        )
        for index in range(4)
    ])
    result = {
        "accuracy": _safe_ratio(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        "iou": _safe_ratio(tp, tp + fp + fn),
        "transition_accuracy": _safe_ratio(float(np.trace(transitions)), float(transitions.sum())),
        "dry_origin_accuracy": _safe_ratio(
            transitions[0, 0] + transitions[1, 1], transition_support[0] + transition_support[1]
        ),
        "wet_origin_accuracy": _safe_ratio(
            transitions[2, 2] + transitions[3, 3], transition_support[2] + transition_support[3]
        ),
        "transition_macro_recall": float(np.nanmean(transition_recalls)),
        "transition_macro_precision": float(np.nanmean(transition_precisions)),
        "transition_macro_f1": float(np.nanmean(transition_f1s)),
        "transition_macro_iou": float(np.nanmean(transition_ious)),
        "transition_macro_accuracy": float(np.nanmean(transition_accuracies)),
    }
    for index, name in enumerate(TRANSITION_NAMES):
        true_positive = int(transition_true_positives[index])
        false_positive = int(transition_predicted[index] - transition_true_positives[index])
        false_negative = int(transition_support[index] - transition_true_positives[index])
        result[f"{name}_precision"] = float(transition_precisions[index])
        result[f"{name}_recall"] = float(transition_recalls[index])
        result[f"{name}_f1"] = float(transition_f1s[index])
        result[f"{name}_accuracy"] = float(transition_accuracies[index])
        result[f"{name}_iou"] = float(transition_ious[index])
        result[f"{name}_support"] = int(transition_support[index])
        result[f"{name}_predicted"] = int(transition_predicted[index])
        result[f"{name}_true_positive"] = true_positive
        result[f"{name}_false_positive"] = false_positive
        result[f"{name}_false_negative"] = false_negative
    return result


def _change_detection_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert wide validation metrics into an analysis-friendly class table."""
    output = []
    for row in rows:
        for name in TRANSITION_NAMES:
            output.append({
                "model": row["model"],
                "lead_period": row["lead_period"],
                "transition": name,
                "accuracy": row[f"{name}_accuracy"],
                "precision": row[f"{name}_precision"],
                "recall": row[f"{name}_recall"],
                "f1": row[f"{name}_f1"],
                "iou": row[f"{name}_iou"],
                "support": row[f"{name}_support"],
                "predicted": row[f"{name}_predicted"],
                "true_positive": row[f"{name}_true_positive"],
                "false_positive": row[f"{name}_false_positive"],
                "false_negative": row[f"{name}_false_negative"],
            })
    return output


def _update_counts(
    binary_counts: np.ndarray,
    transition_counts: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    origins: np.ndarray,
) -> None:
    horizon = actual.shape[1]
    for lead in range(horizon):
        truth = actual[:, lead].reshape(-1)
        estimate = predicted[:, lead].reshape(-1)
        binary_code = truth.astype(np.uint8) * 2 + estimate.astype(np.uint8)
        # Codes: 0 TN, 1 FP, 2 FN, 3 TP.
        values = np.bincount(binary_code, minlength=4)
        binary_counts[lead] += values[[0, 1, 2, 3]]

        origin = origins.reshape(-1)
        actual_transition = _transition_classes(origin, truth)
        predicted_transition = _transition_classes(origin, estimate)
        matrix = np.bincount(
            actual_transition * 4 + predicted_transition, minlength=16
        ).reshape(4, 4)
        transition_counts[lead] += matrix


def evaluate_spatial_model(model=None, model_path: str | None = None) -> tuple[Path, Path]:
    """Evaluate ConvLSTM and persistence/GloFAS baselines on validation tiles."""
    import tensorflow as tf

    product = cleaning_utils.resolve_target_product(get_cfg("runtime.target_product", "viirs"))
    paths = _paths(product, model_path)
    if model is None:
        model = tf.keras.models.load_model(paths.model)
    total_times, height, width = _validate_inputs(paths)
    horizon = resolve_forecast_horizon_periods()
    sequence_length = int(get_cfg("spatial_training.sequence_length", 6))
    patch_size = int(get_cfg("spatial_training.patch_size", 64))
    stride = int(get_cfg("spatial_training.stride", 32))
    patches_per_timestep = int(get_cfg("spatial_training.patches_per_timestep", 16))
    batch_size = int(get_cfg("spatial_training.batch_size", 4))
    cache_mb = int(get_cfg("spatial_training.hdf5_cache_mb", 64))
    validation_fraction = float(get_cfg("spatial_training.validation_fraction", 0.2))
    seed = int(get_cfg("spatial_training.seed", 42))
    threshold = float(get_cfg("spatial_validation.inundation_threshold", 0.5))
    split_time = int(round(total_times * (1.0 - validation_fraction)))
    final_origin_exclusive = total_times - horizon + 1
    samples = _sample_index(
        range(split_time, final_origin_exclusive), height, width, patch_size, stride,
        patches_per_timestep=patches_per_timestep, seed=seed + 1,
    )
    static = _load_static(paths, height, width)
    Sequence = _sequence_class(tf)
    validation = Sequence(
        paths, samples, static, sequence_length, patch_size, batch_size,
        False, seed, cache_mb, horizon,
    )

    glofas_grids = {
        "glofas_20_year": load_raster_on_target_grid(
            get_cfg("spatial_validation.glofas_20_year_path", "data/maps/glofas/merged_raster_SSD_20_20260604164952.tif"),
            height, width,
        ),
        "glofas_100_year": load_raster_on_target_grid(
            get_cfg("spatial_validation.glofas_100_year_path", "data/maps/glofas/merged_raster_SSD_100_20260604172751.tif"),
            height, width,
        ),
    }
    binary = {name: np.zeros((horizon, 4), dtype=np.int64) for name in BASELINE_NAMES}
    transitions = {name: np.zeros((horizon, 4, 4), dtype=np.int64) for name in BASELINE_NAMES}

    try:
        for batch_index in range(len(validation)):
            x, y = validation[batch_index]
            actual = y[..., 0] >= threshold
            counts = actual.sum(axis=(-2, -1))
            origin = x[:, -1, ..., 0] >= threshold
            model_scores = np.asarray(model.predict_on_batch(x), dtype=np.float32)[..., 0]
            if model_scores.shape != actual.shape:
                raise ValueError(
                    f"Spatial model output {model_scores.shape} does not match targets {actual.shape}."
                )
            score_sets = {
                "convlstm": model_scores,
            }
            persistence = _persistence_mask(origin, horizon)
            _update_counts(
                binary["persistence"], transitions["persistence"],
                actual, persistence, origin,
            )
            batch_samples = samples[
                batch_index * batch_size:(batch_index + 1) * batch_size
            ]
            for name, grid in glofas_grids.items():
                patches = np.stack([
                    grid[row:row + patch_size, col:col + patch_size]
                    for _, row, col in batch_samples
                ])
                score_sets[name] = np.repeat(patches[:, np.newaxis], horizon, axis=1)
            for name, scores in score_sets.items():
                prediction = _top_n_mask(scores, counts)
                _update_counts(binary[name], transitions[name], actual, prediction, origin)
    finally:
        validation.close()

    rows = []
    for name in BASELINE_NAMES:
        for lead in range(horizon):
            rows.append({
                "model": name,
                "lead_period": lead + 1,
                **_metrics(binary[name][lead], transitions[name][lead]),
            })
        rows.append({
            "model": name,
            "lead_period": "overall",
            **_metrics(binary[name].sum(axis=0), transitions[name].sum(axis=0)),
        })

    csv_path = paths.model.with_suffix(".validation.csv")
    json_path = paths.model.with_suffix(".validation.json")
    change_csv_path = paths.model.with_suffix(".change_detection.validation.csv")
    change_rows = _change_detection_rows(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with change_csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(change_rows[0]))
        writer.writeheader()
        writer.writerows(change_rows)
    json_path.write_text(json.dumps({
        "target_product": product,
        "forecast_horizon_periods": horizon,
        "known_pixel_count_applied": True,
        "known_pixel_count_models": ["convlstm", "glofas_20_year", "glofas_100_year"],
        "persistence_definition": "origin_mask_repeated_unchanged",
        "transition_classes": list(TRANSITION_NAMES),
        "change_detection_metrics": ["f1", "accuracy", "precision", "recall", "iou"],
        "change_detection_csv": str(change_csv_path),
        "results": rows,
        "change_detection_results": change_rows,
    }, indent=2, allow_nan=True), encoding="utf-8")
    LOG.info(
        "Saved spatial validation results to %s, %s, and %s",
        csv_path, json_path, change_csv_path,
    )
    from readme_status import update_spatial_validation_readme
    update_spatial_validation_readme(
        change_rows,
        forecast_horizon_periods=horizon,
        csv_path=change_csv_path,
        json_path=json_path,
    )
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate_spatial_model(model_path=args.model_path)


if __name__ == "__main__":
    main()
