"""Shared daily-to-target-period streaming for gridded TAMSAT products."""
from __future__ import annotations

import calendar
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import geopandas as gpd
import h5py
import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from tqdm.auto import tqdm

from data.stats import gridded_stats
from .. import cleaning_utils


DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})")
DOMAIN_BOUNDS = (15.837321509670957, -4.029166662242848, 23.424907051000087, 36.30367723700005)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class PeriodFiles:
    start: str
    files: tuple[Path, ...]


@dataclass
class GridMasks:
    height: int
    width: int
    transform: rasterio.Affine
    crs: object
    inside: np.ndarray
    regions: list[tuple[str, np.ndarray]]


def inventory_daily_files(folder: str | Path) -> dict[pd.Timestamp, Path]:
    """Return one canonical local NetCDF per date, preferring non-duplicate names."""
    selected: dict[pd.Timestamp, Path] = {}
    for path in resolve_project_path(folder).rglob("*.nc"):
        match = DATE_PATTERN.search(path.name)
        if not match:
            continue
        date = pd.Timestamp("-".join(match.groups()))
        previous = selected.get(date)
        if previous is None or ("(" in previous.name and "(" not in path.name):
            selected[date] = path
    return selected


def complete_periods(files_by_date: dict[pd.Timestamp, Path]) -> list[PeriodFiles]:
    """Group local daily files into complete calendar dekads."""
    buckets: dict[pd.Timestamp, list[tuple[pd.Timestamp, Path]]] = {}
    for date, path in files_by_date.items():
        day = 1 if date.day <= 10 else 11 if date.day <= 20 else 21
        start = date.replace(day=day).normalize()
        buckets.setdefault(start, []).append((date, path))
    periods = []
    for start, entries in sorted(buckets.items()):
        expected = 10 if start.day in (1, 11) else calendar.monthrange(start.year, start.month)[1] - 20
        entries.sort()
        expected_dates = list(pd.date_range(start, periods=expected, freq="D"))
        if [date for date, _ in entries] == expected_dates:
            periods.append(PeriodFiles(start.strftime("%Y-%m-%d"), tuple(path for _, path in entries)))
    return periods


def _validate_daily_file(path: Path, variable_name: str) -> str | None:
    try:
        with h5py.File(path, "r") as dataset:
            missing = {"lat", "lon", variable_name}.difference(dataset.keys())
            if missing:
                raise ValueError(f"missing datasets {sorted(missing)}")
            values = dataset[variable_name]
            if values.ndim != 3 or values.shape[0] < 1:
                raise ValueError(f"unexpected {variable_name} shape {values.shape}")
            if values.shape[-2:] != (len(dataset["lat"]), len(dataset["lon"])):
                raise ValueError("coordinate and data shapes differ")
        return None
    except Exception as hdf_error:
        try:
            with nc.Dataset(path, "r") as dataset:
                missing = {"lat", "lon", variable_name}.difference(dataset.variables)
                if missing:
                    raise ValueError(f"missing variables {sorted(missing)}")
                if dataset.variables[variable_name].ndim != 3:
                    raise ValueError(f"unexpected {variable_name} dimensions")
            return None
        except Exception as netcdf_error:
            return f"HDF5: {hdf_error}; NetCDF: {netcdf_error}"


def _write_validation_cache(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_daily_files(
    periods: list[PeriodFiles], variable_name: str, max_workers: int = 8
) -> list[tuple[Path, str]]:
    """Validate changed files concurrently and reuse cached results for unchanged files."""
    paths = list(dict.fromkeys(path.resolve() for period in periods for path in period.files))
    if not paths:
        return []
    common_root = Path(os.path.commonpath([str(path.parent) for path in paths]))
    cache_path = common_root / ".inflow_validation_cache.json"
    cache = {"schema": 1, "files": {}}
    try:
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        if loaded.get("schema") == 1 and isinstance(loaded.get("files"), dict):
            cache = loaded
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    invalid: list[tuple[Path, str]] = []
    pending: list[tuple[Path, os.stat_result]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError as error:
            invalid.append((path, str(error)))
            continue
        entry = cache["files"].get(str(path), {})
        unchanged = (
            entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
            and entry.get("variable") == variable_name
        )
        if unchanged:
            if entry.get("error"):
                invalid.append((path, str(entry["error"])))
        else:
            pending.append((path, stat))

    if pending:
        workers = max(1, min(int(max_workers), len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_validate_daily_file, path, variable_name): (path, stat)
                for path, stat in pending
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Validating {variable_name} files"
            ):
                path, stat = futures[future]
                try:
                    error = future.result()
                except Exception as unexpected:
                    error = str(unexpected)
                cache["files"][str(path)] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "variable": variable_name,
                    "error": error,
                }
                if error:
                    invalid.append((path, error))
        _write_validation_cache(cache_path, cache)
    return sorted(invalid, key=lambda item: str(item[0]))


def quarantine_invalid_files(invalid: list[tuple[Path, str]]) -> list[Path]:
    """Move unreadable files aside so downloads can replace canonical paths."""
    quarantined = []
    for path, _ in invalid:
        candidate = path.with_suffix(path.suffix + ".invalid")
        counter = 2
        while candidate.exists():
            candidate = path.with_suffix(path.suffix + f".invalid.{counter}")
            counter += 1
        path.replace(candidate)
        quarantined.append(candidate)
    return quarantined


def build_grid_masks(catchments_path: str) -> GridMasks:
    """Rasterize catchment and regional masks once for all periods."""
    height, width = map(int, cleaning_utils.MASK_REGIONS_REF_SHAPE)
    transform = rasterio.Affine(*cleaning_utils.MASK_REGIONS_REF_TRANSFORM)
    crs = cleaning_utils.MASK_REGIONS_REF_CRS
    catchments_path = resolve_project_path(catchments_path)
    if not catchments_path.exists():
        raise FileNotFoundError(f"Catchment shapefile not found: {catchments_path}")
    catchments = gpd.read_file(catchments_path).to_crs(crs)
    inside = geometry_mask(
        catchments.geometry, out_shape=(height, width), transform=transform, invert=True
    )
    regions = cleaning_utils.extract_regions(
        admin0_path=resolve_project_path(cleaning_utils.ADMIN0_PATH),
        admin1_path=resolve_project_path(cleaning_utils.ADMIN1_PATH),
        abyei_region_path=resolve_project_path(cleaning_utils.ABYEI_PATH),
    ).to_crs(crs)
    region_masks = []
    for _, region in regions.iterrows():
        code = gridded_stats.region_to_code_dict.get(region["region"], region["region"])
        mask = geometry_mask(
            [region.geometry], out_shape=(height, width), transform=transform,
            invert=True, all_touched=True,
        ) & inside
        region_masks.append((str(code), mask))
    return GridMasks(height, width, transform, crs, inside, region_masks)


def _read_daily_subset(path, variable_name, bounds):
    """Read a subset, falling back to HDF5 when NetCDF metadata is malformed."""
    north, south, west, east = bounds
    try:
        # TAMSAT NetCDF4 files are HDF5 containers. Reading their arrays through
        # h5py avoids failures caused only by malformed NetCDF metadata.
        with h5py.File(path, "r") as dataset:
            lats = np.asarray(dataset["lat"][:])
            lons = np.asarray(dataset["lon"][:])
            lat_indices = np.flatnonzero((lats >= south) & (lats <= north))
            lon_indices = np.flatnonzero((lons >= west) & (lons <= east))
            lat_slice = slice(lat_indices.min(), lat_indices.max() + 1)
            lon_slice = slice(lon_indices.min(), lon_indices.max() + 1)
            values = np.asarray(dataset[variable_name][0, lat_slice, lon_slice], dtype=np.float32)
            fill_value = dataset[variable_name].attrs.get("_FillValue")
            if fill_value is not None:
                values[values == np.asarray(fill_value).reshape(-1)[0]] = np.nan
            return lats[lat_slice], lons[lon_slice], values
    except OSError:
        with nc.Dataset(path, "r") as dataset:
            lats = np.asarray(dataset.variables["lat"][:])
            lons = np.asarray(dataset.variables["lon"][:])
            lat_indices = np.flatnonzero((lats >= south) & (lats <= north))
            lon_indices = np.flatnonzero((lons >= west) & (lons <= east))
            if not len(lat_indices) or not len(lon_indices):
                raise ValueError(f"Requested domain is outside {path}")
            lat_slice = slice(lat_indices.min(), lat_indices.max() + 1)
            lon_slice = slice(lon_indices.min(), lon_indices.max() + 1)
            values = np.ma.filled(
                dataset.variables[variable_name][0, lat_slice, lon_slice], np.nan
            ).astype(np.float32, copy=False)
            return lats[lat_slice], lons[lon_slice], values


def stream_period_grid(
    files: tuple[Path, ...], variable_name: str, masks: GridMasks,
    bounds: tuple[float, float, float, float] = DOMAIN_BOUNDS,
) -> np.ndarray:
    """Average one complete period on its source grid, then reproject once."""
    total = None
    count = None
    source_lats = source_lons = None
    for path in files:
        lats, lons, values = _read_daily_subset(path, variable_name, bounds)
        if total is None:
            total = np.zeros(values.shape, dtype=np.float64)
            count = np.zeros(values.shape, dtype=np.uint16)
            source_lats = lats
            source_lons = lons
        valid = np.isfinite(values)
        total[valid] += values[valid]
        count[valid] += 1
    source = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0).astype(np.float32)
    if source_lats[0] < source_lats[-1]:
        source = source[::-1]
        source_lats = source_lats[::-1]
    if source_lons[0] > source_lons[-1]:
        source = source[:, ::-1]
        source_lons = source_lons[::-1]
    lat_step = abs(float(source_lats[1] - source_lats[0]))
    lon_step = abs(float(source_lons[1] - source_lons[0]))
    source_transform = rasterio.transform.from_origin(
        float(source_lons.min()) - lon_step / 2,
        float(source_lats.max()) + lat_step / 2,
        lon_step, lat_step,
    )
    output = np.full((masks.height, masks.width), np.nan, dtype=np.float32)
    reproject(
        source=source, destination=output, src_transform=source_transform,
        src_crs="EPSG:4326", src_nodata=np.nan, dst_transform=masks.transform,
        dst_crs=masks.crs, dst_nodata=np.nan, resampling=Resampling.bilinear,
    )
    output[~masks.inside] = np.nan
    return output


def temporal_row(date: str, grid: np.ndarray, prefix: str, masks: GridMasks) -> dict[str, float | str]:
    valid = np.isfinite(grid) & masks.inside
    row: dict[str, float | str] = {
        "date": date,
        prefix: float(np.nansum(grid[valid])) if valid.any() else np.nan,
    }
    for code, region_mask in masks.regions:
        cells = valid & region_mask
        row[f"{prefix}_{code}"] = float(np.nanmean(grid[cells])) if cells.any() else np.nan
    return row


def _hdf_dates(hdf: h5py.File) -> list[str]:
    return list(hdf["dates"].asstr()[:]) if "dates" in hdf else []


def _atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append(
    hdf_path: str | Path, periods: list[PeriodFiles], dataset_name: str,
    prefix: str, target_product: str, masks: GridMasks,
    loader: Callable[[PeriodFiles], np.ndarray],
) -> list[dict[str, float | str]]:
    rows = []
    with h5py.File(hdf_path, "a") as hdf:
        values = hdf.get(dataset_name)
        dates = hdf.get("dates")
        for period in tqdm(periods, desc=f"Streaming {prefix} into H5"):
            grid = loader(period)
            if values is None:
                values = hdf.create_dataset(
                    dataset_name, shape=(0, *grid.shape), maxshape=(None, *grid.shape),
                    chunks=(1, *grid.shape), dtype=np.float32,
                    compression="lzf", shuffle=True,
                )
            if dates is None:
                dates = hdf.create_dataset(
                    "dates", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8")
                )
            index = values.shape[0]
            values.resize(index + 1, axis=0)
            dates.resize(index + 1, axis=0)
            values[index] = grid
            dates[index] = period.start
            rows.append(temporal_row(period.start, grid, prefix, masks))
        hdf.attrs["target_product"] = target_product
        hdf.attrs["aggregation"] = "mean daily value within complete calendar dekad"
        hdf.flush()
    return rows


def _csv_from_hdf(hdf_path, csv_path, dataset_name, prefix, masks):
    rows = []
    with h5py.File(hdf_path, "r") as hdf:
        dates = _hdf_dates(hdf)
        values = hdf[dataset_name]
        if len(dates) != values.shape[0]:
            raise ValueError(f"{dataset_name} HDF5 values/date lengths differ")
        for index, date in enumerate(tqdm(dates, desc=f"Rebuilding {prefix} CSV")):
            rows.append(temporal_row(date, values[index], prefix, masks))
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, csv_path)
    return frame


def validate_outputs(hdf_path, csv_path, dataset_name, prefix, expected_dates=None) -> list[str]:
    frame = pd.read_csv(csv_path)
    csv_dates = pd.to_datetime(frame.get("date"), errors="coerce")
    if frame.empty or csv_dates.isna().any() or csv_dates.duplicated().any() or not csv_dates.is_monotonic_increasing:
        raise ValueError(f"{prefix} CSV dates are empty, invalid, duplicated, or unsorted")
    columns = [column for column in frame if column == prefix or column.startswith(prefix + "_")]
    if not columns or frame[columns].isna().all().any():
        raise ValueError(f"{prefix} CSV contains an entirely empty feature")
    with h5py.File(hdf_path, "r") as hdf:
        if dataset_name not in hdf or "dates" not in hdf:
            raise ValueError(f"{dataset_name} HDF5 is missing values or dates")
        dates = _hdf_dates(hdf)
        if len(dates) != hdf[dataset_name].shape[0]:
            raise ValueError(f"{dataset_name} HDF5 values/date lengths differ")
    if dates != list(csv_dates.dt.strftime("%Y-%m-%d")):
        raise ValueError(f"{prefix} HDF5 and CSV dates differ")
    if expected_dates is not None and dates != list(expected_dates):
        raise ValueError(f"{prefix} outputs do not match available complete periods")
    return dates


def sync_product(
    *, periods: list[PeriodFiles], hdf_path: str | Path, csv_path: str | Path,
    dataset_name: str, prefix: str, target_product: str, masks: GridMasks,
    loader: Callable[[PeriodFiles], np.ndarray], rebuild: bool = False,
) -> tuple[str, int]:
    """Transactionally bootstrap or append one date-indexed gridded product."""
    if not periods:
        raise RuntimeError(f"No complete daily periods available for {prefix}")
    expected = [period.start for period in periods]
    hdf_path, csv_path = Path(hdf_path), Path(csv_path)
    existing: list[str] = []
    if not rebuild and hdf_path.exists():
        try:
            with h5py.File(hdf_path, "r") as hdf:
                existing = _hdf_dates(hdf)
                valid = dataset_name in hdf and len(existing) == hdf[dataset_name].shape[0]
            if not valid:
                rebuild = True
            elif not csv_path.exists():
                _csv_from_hdf(hdf_path, csv_path, dataset_name, prefix, masks)
            else:
                try:
                    validate_outputs(hdf_path, csv_path, dataset_name, prefix)
                except ValueError:
                    _csv_from_hdf(hdf_path, csv_path, dataset_name, prefix, masks)
        except (OSError, KeyError, ValueError):
            rebuild = True
    else:
        rebuild = True
    if not rebuild and expected[:len(existing)] != existing:
        rebuild = True

    if rebuild:
        hdf_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_hdf = tempfile.mkstemp(prefix=f".{dataset_name}.", suffix=".tmp", dir=hdf_path.parent)
        os.close(fd)
        os.unlink(temporary_hdf)
        fd, temporary_csv = tempfile.mkstemp(prefix=f".{prefix}.", suffix=".tmp", dir=csv_path.parent)
        os.close(fd)
        os.unlink(temporary_csv)
        try:
            rows = _append(temporary_hdf, periods, dataset_name, prefix, target_product, masks, loader)
            _atomic_csv(pd.DataFrame(rows), temporary_csv)
            validate_outputs(temporary_hdf, temporary_csv, dataset_name, prefix, expected)
            os.replace(temporary_hdf, hdf_path)
            os.replace(temporary_csv, csv_path)
        finally:
            for temporary in (temporary_hdf, temporary_csv):
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return "rebuilt", len(periods)

    pending = periods[len(existing):]
    if pending:
        rows = _append(hdf_path, pending, dataset_name, prefix, target_product, masks, loader)
        frame = pd.concat([pd.read_csv(csv_path), pd.DataFrame(rows)], ignore_index=True)
        frame = frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        _atomic_csv(frame, csv_path)
    validate_outputs(hdf_path, csv_path, dataset_name, prefix, expected)
    return "appended" if pending else "current", len(pending)
