"""Train the patch-based ConvLSTM used by the spatial prediction pipeline.

Run from the project root after preparing the configured target-product data:

    python3 -m model.spatial.train_convlstm

The model consumes the same five channels as ``model.make_spatial_prediction``:
inundation, rainfall, soil moisture, elevation, and basin attributes. Historic
grids are streamed from HDF5 by batch, so the full space-time cube is never
expanded in memory.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import threading
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from processing import cleaning_utils
from processing.config import get_cfg, resolve_forecast_horizon_periods
from .channels import CHANNEL_NAMES, ecmwf_patch, load_glofas_grid


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingPaths:
    inundation: Path
    rainfall: Path
    moisture: Path
    elevation: Path
    basin_attributes: Path
    ecmwf: Path
    glofas: Path
    model: Path


def _paths(target_product: str, output: str | None = None) -> TrainingPaths:
    aligned = Path(get_cfg("paths.historic.root", "data/historic")) / f"{target_product}-aligned"
    inundation_name = "inundation_viirs.h5" if target_product == "viirs" else "inundation_modis.h5"
    return TrainingPaths(
        inundation=aligned / inundation_name,
        rainfall=aligned / "gridded_rainfall.h5",
        moisture=aligned / "gridded_moisture.h5",
        elevation=Path(get_cfg("spatial_training.elevation_path", "data/maps/elevation.npy")),
        basin_attributes=Path(
            get_cfg("spatial_training.basin_attributes_path", "data/maps/basin_attributes.npy")
        ),
        ecmwf=aligned / "ecmwf_hindcasts.h5",
        glofas=Path(get_cfg(
            "spatial_training.glofas_path", "data/maps/flood_return_period/glofas_10_year_rp.tif"
        )),
        model=Path(output or get_cfg("spatial_training.model_path", "model/spatial_model.keras")),
    )


def _patch_starts(size: int, patch_size: int, stride: int) -> list[int]:
    if patch_size > size:
        raise ValueError(f"Patch size {patch_size} exceeds grid dimension {size}.")
    starts = list(range(0, size - patch_size + 1, stride))
    final = size - patch_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _validate_inputs(paths: TrainingPaths) -> tuple[int, int, int]:
    missing = [str(path) for path in paths.__dict__.values() if path != paths.model and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing spatial training inputs: " + ", ".join(missing))

    specs = ((paths.inundation, "inundation"), (paths.rainfall, "rainfall"), (paths.moisture, "moisture"))
    shapes = []
    for path, key in specs:
        with h5py.File(path, "r") as hf:
            if key not in hf:
                raise KeyError(f"{path} does not contain dataset '{key}'.")
            if hf[key].ndim != 3:
                raise ValueError(f"{path}:{key} must have shape (time, height, width).")
            shapes.append(tuple(map(int, hf[key].shape)))
    height, width = shapes[0][1:]
    if any(shape[1:] != (height, width) for shape in shapes):
        raise ValueError(f"Spatial grid mismatch between HDF5 inputs: {shapes}")
    with h5py.File(paths.ecmwf, "r") as hf:
        if "X" not in hf or hf["X"].ndim != 4 or hf["X"].shape[1] != 4:
            raise ValueError(f"{paths.ecmwf}:X must have shape (time, 4, height, width).")
        ecmwf_times = int(hf["X"].shape[0])
    return min(*(shape[0] for shape in shapes), ecmwf_times), height, width


def _load_static(paths: TrainingPaths, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elevation = np.asarray(np.load(paths.elevation, mmap_mode="r"), dtype=np.float32)
    basin = np.asarray(np.load(paths.basin_attributes, mmap_mode="r"), dtype=np.float32)
    if elevation.ndim == 2:
        elevation = elevation[np.newaxis, ...]
    if basin.ndim == 2:
        basin = basin[np.newaxis, ...]
    # Deployment currently uses the first basin-attribute layer.
    elevation = elevation[0, :height, :width]
    basin = basin[0, :height, :width]
    if elevation.shape != (height, width) or basin.shape != (height, width):
        raise ValueError("Static maps do not cover the configured historic grid.")
    glofas = load_glofas_grid(paths.glofas, height, width)
    return np.nan_to_num(elevation), np.nan_to_num(basin), glofas


def _sample_index(
    target_times: range,
    height: int,
    width: int,
    patch_size: int,
    stride: int,
    patches_per_timestep: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    rows = _patch_starts(height, patch_size, stride)
    cols = _patch_starts(width, patch_size, stride)
    locations = [(row, col) for row in rows for col in cols]
    rng = np.random.default_rng(seed)
    samples = []
    for time in target_times:
        selected = locations
        if patches_per_timestep and patches_per_timestep < len(locations):
            indices = rng.choice(len(locations), size=patches_per_timestep, replace=False)
            selected = [locations[index] for index in sorted(indices)]
        samples.extend((time, row, col) for row, col in selected)
    return np.asarray(samples, dtype=np.int32)


def _sequence_class(tf):
    class SpatialSequence(tf.keras.utils.Sequence):
        def __init__(
            self, paths, samples, static, sequence_length, patch_size, batch_size,
            shuffle, seed, hdf5_cache_mb, forecast_horizon,
        ):
            super().__init__()
            self.paths = paths
            self.samples = np.asarray(samples, dtype=np.int32)
            self.elevation, self.basin, self.glofas = static
            self.sequence_length = int(sequence_length)
            self.patch_size = int(patch_size)
            self.batch_size = int(batch_size)
            self.forecast_horizon = int(forecast_horizon)
            self.shuffle = bool(shuffle)
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(len(self.samples))
            self.hdf5_cache_bytes = int(hdf5_cache_mb) * 1024 * 1024
            self._files = None
            self._lock = threading.Lock()
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.samples) / self.batch_size))

        def on_epoch_end(self):
            if self.shuffle:
                # Shuffle timestep groups while keeping patches from the same
                # timestep adjacent. This reuses decompressed HDF5 time chunks.
                times = np.unique(self.samples[:, 0])
                self.rng.shuffle(times)
                self.order = np.concatenate([
                    np.flatnonzero(self.samples[:, 0] == time) for time in times
                ])

        def _open_files(self):
            if self._files is None:
                options = {"mode": "r", "rdcc_nbytes": self.hdf5_cache_bytes, "rdcc_nslots": 10007}
                self._files = (
                    h5py.File(self.paths.inundation, **options),
                    h5py.File(self.paths.rainfall, **options),
                    h5py.File(self.paths.moisture, **options),
                    h5py.File(self.paths.ecmwf, **options),
                )
            return self._files

        def close(self):
            if self._files is not None:
                for handle in self._files:
                    handle.close()
                self._files = None

        def __del__(self):
            self.close()

        def __getitem__(self, batch_index):
            chosen = self.order[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            batch = self.samples[chosen]
            x = np.empty(
                (len(batch), self.sequence_length, self.patch_size, self.patch_size, len(CHANNEL_NAMES)),
                dtype=np.float32,
            )
            y = np.empty(
                (len(batch), self.forecast_horizon, self.patch_size, self.patch_size, 1),
                dtype=np.float32,
            )
            with self._lock:
                inun_hf, rain_hf, moist_hf, ecmwf_hf = self._open_files()
                inundation = inun_hf["inundation"]
                rainfall = rain_hf["rainfall"]
                moisture = moist_hf["moisture"]
                ecmwf = ecmwf_hf["X"]
                source_transform = tuple(ecmwf_hf.attrs["transform"])
                source_crs = ecmwf_hf.attrs.get("crs", "EPSG:4326")
                for index, (target_time, row, col) in enumerate(batch):
                    start = int(target_time) - self.sequence_length
                    row_slice = slice(int(row), int(row) + self.patch_size)
                    col_slice = slice(int(col), int(col) + self.patch_size)
                    x[index, ..., 0] = inundation[start:target_time, row_slice, col_slice]
                    x[index, ..., 1] = rainfall[start:target_time, row_slice, col_slice]
                    x[index, ..., 2] = moisture[start:target_time, row_slice, col_slice]
                    x[index, ..., 3] = self.elevation[row_slice, col_slice][np.newaxis, ...]
                    x[index, ..., 4] = self.basin[row_slice, col_slice][np.newaxis, ...]
                    x[index, ..., 9] = self.glofas[row_slice, col_slice][np.newaxis, ...]
                    for step, time_index in enumerate(range(start, int(target_time))):
                        x[index, step, ..., 5:9] = ecmwf_patch(
                            ecmwf, time_index, int(row), int(col), self.patch_size,
                            source_transform, source_crs,
                        )
                    target_end = int(target_time) + self.forecast_horizon
                    y[index, ..., 0] = inundation[target_time:target_end, row_slice, col_slice]
            return np.nan_to_num(x, copy=False), np.clip(np.nan_to_num(y, copy=False), 0.0, 1.0)

    return SpatialSequence


def _build_model(
    tf, sequence_length: int, patch_size: int, filters: list[int], forecast_horizon: int
):
    kernel_size = tuple(int(value) for value in get_cfg("spatial_training.kernel_size", [3, 3]))
    activation = str(get_cfg("spatial_training.activation", "relu"))
    output_activation = str(get_cfg("spatial_training.output_activation", "sigmoid"))
    model = tf.keras.Sequential(name="spatial_convlstm")
    model.add(tf.keras.Input(shape=(sequence_length, patch_size, patch_size, len(CHANNEL_NAMES))))
    for index, count in enumerate(filters):
        model.add(tf.keras.layers.ConvLSTM2D(
            filters=int(count),
            kernel_size=kernel_size,
            activation=activation,
            padding="same",
            return_sequences=index < len(filters) - 1,
        ))
        model.add(tf.keras.layers.BatchNormalization())
    # Keep probabilities/loss in float32 even when GPU mixed precision is enabled.
    model.add(tf.keras.layers.Conv2D(
        forecast_horizon, kernel_size, activation=output_activation, padding="same", dtype="float32"
    ))
    model.add(tf.keras.layers.Permute((3, 1, 2)))
    model.add(tf.keras.layers.Reshape((forecast_horizon, patch_size, patch_size, 1)))
    optimizer_name = str(get_cfg("spatial_training.optimizer", "adam")).lower()
    learning_rate = float(get_cfg("spatial_training.learning_rate", 0.001))
    optimizer = tf.keras.optimizers.get(optimizer_name)
    optimizer.learning_rate = learning_rate
    model.compile(
        optimizer=optimizer,
        loss=str(get_cfg("spatial_training.loss", "mean_squared_error")),
        metrics=list(get_cfg("spatial_training.metrics", ["mae"])),
    )
    return model


def train(output: str | None = None) -> Path:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Spatial training requires TensorFlow (Python <3.13); install requirements.txt.") from exc

    product = cleaning_utils.resolve_target_product(get_cfg("runtime.target_product", "viirs"))
    paths = _paths(product, output)
    total_times, height, width = _validate_inputs(paths)
    sequence_length = int(get_cfg("spatial_training.sequence_length", 6))
    forecast_horizon = resolve_forecast_horizon_periods()
    patch_size = int(get_cfg("spatial_training.patch_size", 64))
    stride = int(get_cfg("spatial_training.stride", 32))
    patches_per_timestep = int(get_cfg("spatial_training.patches_per_timestep", 16))
    batch_size = int(get_cfg("spatial_training.batch_size", 2))
    hdf5_cache_mb = int(get_cfg("spatial_training.hdf5_cache_mb", 64))
    max_epochs = int(get_cfg(
        "spatial_training.max_epochs", get_cfg("spatial_training.epochs", 200)
    ))
    validation_fraction = float(get_cfg("spatial_training.validation_fraction", 0.2))
    seed = int(get_cfg("spatial_training.seed", 42))
    filters = [int(value) for value in get_cfg("spatial_training.filters", [32])]
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("spatial_training.validation_fraction must be between zero and one.")

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if bool(get_cfg("spatial_training.mixed_precision", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    first_target = sequence_length
    split_time = int(round(total_times * (1.0 - validation_fraction)))
    last_target_exclusive = total_times - forecast_horizon + 1
    train_target_end = split_time - forecast_horizon + 1
    if train_target_end <= first_target or split_time >= last_target_exclusive:
        raise RuntimeError("Not enough time periods for the configured chronological split.")
    train_samples = _sample_index(
        range(first_target, train_target_end), height, width, patch_size, stride,
        patches_per_timestep=patches_per_timestep, seed=seed,
    )
    validation_samples = _sample_index(
        range(split_time, last_target_exclusive), height, width, patch_size, stride,
        patches_per_timestep=patches_per_timestep, seed=seed + 1,
    )
    static = _load_static(paths, height, width)
    Sequence = _sequence_class(tf)
    train_data = Sequence(
        paths, train_samples, static, sequence_length, patch_size, batch_size,
        True, seed, hdf5_cache_mb, forecast_horizon,
    )
    validation_data = Sequence(
        paths, validation_samples, static, sequence_length, patch_size, batch_size,
        False, seed, hdf5_cache_mb, forecast_horizon,
    )

    LOG.info(
        "Training %s ConvLSTM: %d periods, %d future leads, %d train patches, "
        "%d validation patches, grid=%dx%d",
        product.upper(), total_times, forecast_horizon, len(train_samples),
        len(validation_samples), height, width,
    )
    model = _build_model(tf, sequence_length, patch_size, filters, forecast_horizon)
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    configured_backup_dir = str(get_cfg("spatial_training.backup_dir", "")).strip()
    backup_dir = Path(configured_backup_dir) if configured_backup_dir else (
        paths.model.parent / f".{paths.model.stem}_training_backup"
    )
    monitor = str(get_cfg("spatial_training.checkpoint_monitor", "val_loss"))
    monitor_mode = str(get_cfg("spatial_training.checkpoint_mode", "min"))
    callbacks = [
        # This is the only published model artifact. It is replaced solely when
        # validation loss improves.
        tf.keras.callbacks.ModelCheckpoint(
            paths.model,
            monitor=monitor,
            mode=monitor_mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # Training-state backup supports resuming an interrupted fit. It is not
        # a candidate model and is deleted automatically after successful fit.
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir),
            save_freq=int(get_cfg("spatial_training.backup_frequency_batches", 100)),
            double_checkpoint=True,
            delete_checkpoint=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=int(get_cfg("spatial_training.early_stopping_patience", 10)),
            min_delta=float(get_cfg("spatial_training.early_stopping_min_delta", 1.0e-5)),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=float(get_cfg("spatial_training.reduce_lr_factor", 0.5)),
            patience=int(get_cfg("spatial_training.reduce_lr_patience", 5)),
            min_lr=float(get_cfg("spatial_training.minimum_learning_rate", 1.0e-6)), verbose=1,
        ),
    ]
    try:
        history = model.fit(
            train_data, validation_data=validation_data, epochs=max_epochs, callbacks=callbacks
        )
    except KeyboardInterrupt:
        LOG.warning(
            "Training interrupted. The best completed validation checkpoint remains at %s; "
            "rerun the command to resume from %s.",
            paths.model,
            backup_dir,
        )
        raise
    finally:
        train_data.close()
        validation_data.close()
    # Reload the published checkpoint so downstream validation always evaluates
    # the best epoch, regardless of why fitting stopped.
    model = tf.keras.models.load_model(paths.model)
    metadata_path = paths.model.with_suffix(".training.json")
    metadata_path.write_text(json.dumps({
        "target_product": product,
        "sequence_length": sequence_length,
        "forecast_horizon_periods": forecast_horizon,
        "output_shape": [forecast_horizon, patch_size, patch_size, 1],
        "patch_size": patch_size,
        "stride": stride,
        "patches_per_timestep": patches_per_timestep,
        "hdf5_cache_mb": hdf5_cache_mb,
        "channels": list(CHANNEL_NAMES),
        "train_period_end_index": split_time - 1,
        "validation_period_start_index": split_time,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }, indent=2), encoding="utf-8")
    LOG.info("Saved best spatial model to %s", paths.model)
    if bool(get_cfg("spatial_validation.enabled", True)):
        from .evaluate_spatial_model import evaluate_spatial_model
        evaluate_spatial_model(model=model, model_path=str(paths.model))
    return paths.model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Override spatial_training.model_path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(args.output)


if __name__ == "__main__":
    main()
