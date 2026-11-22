"""Process ECMWF rainfall forecasts onto the target-product timeline.

Each target period has four ensemble-mean, non-overlapping forecast totals for
days [0,10), [10,20), [20,30), and [30,40). Forecasts are stored on a compact
common 1.5-degree grid; temporal and regional features are derived from HDF5.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import h5py
import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject, transform_bounds
from tqdm.auto import tqdm

from data.stats import gridded_stats
from .. import cleaning_utils
from ..config import get_cfg

LOG = logging.getLogger(__name__)
BASE_URL = "https://gws-access.jasmin.ac.uk/public/tamsat/tamsat_alert/jess_hindcasts/"
SCHEMA_VERSION = "target-aligned-coarse-relative-10day-v1"
LEAD_START_DAYS = np.array([0, 10, 20, 30], dtype=np.int16)
LEAD_END_DAYS = np.array([10, 20, 30, 40], dtype=np.int16)
MAX_INIT_AGE_DAYS = 6


def download_ecmwf_hindcasts(base_url=None, download_path=None, max_workers=4, timeout=60):
    """Download missing NetCDF files; preserve existing nonempty files."""
    base_url = base_url or get_cfg("sources.ecmwf.base_url", BASE_URL)
    folder = Path(download_path or Path(get_cfg("paths.downloads.root", "data/downloads")) / "ecmwf")
    folder.mkdir(parents=True, exist_ok=True)
    response = requests.get(base_url, timeout=timeout, headers={"User-Agent": "INFLOW-AI/1.0"})
    response.raise_for_status()
    names = sorted(set(re.findall(r"[A-Za-z0-9_.-]+\.nc\b", response.text)))
    if not names:
        raise RuntimeError(f"No .nc filenames found at {base_url}")

    def get_one(name):
        destination = folder / name
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        partial = folder / f"{name}.part"
        try:
            with requests.get(urljoin(base_url, name), stream=True, timeout=timeout) as download:
                download.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in download.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if partial.stat().st_size == 0:
                raise RuntimeError(f"Empty download: {name}")
            partial.replace(destination)
            return destination
        finally:
            partial.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(tqdm(pool.map(get_one, names), total=len(names), desc="Download ECMWF"))


def _blocks(init, last_valid, product=None):
    """Return four fixed 10-day lead blocks if 40 complete forecast days exist."""
    del product
    init = pd.Timestamp(init).normalize()
    if init + pd.Timedelta(days=40) > pd.Timestamp(last_valid).normalize():
        return []
    return [(init + pd.Timedelta(days=int(start)), init + pd.Timedelta(days=int(end)))
            for start, end in zip(LEAD_START_DAYS, LEAD_END_DAYS)]


def _run_slices(times, source, file):
    """Identify forecast sequences by gaps, without splitting New Year crossings."""
    if source == "forecast":
        match = re.search(r"(20\d{6})", file.stem)
        if not match:
            raise ValueError(f"Cannot identify forecast initialization date in {file.name}")
        init = pd.Timestamp(match.group(1))
        positions = np.flatnonzero(times == init)
        if not len(positions):
            raise ValueError(f"Filename initialization {init} not in time coordinate: {file}")
        return [slice(int(positions[0]), len(times))]
    gaps = np.flatnonzero(np.diff(times.values) > np.timedelta64(1, "D")) + 1
    boundaries = np.concatenate(([0], gaps, [len(times)]))
    return [slice(int(start), int(end)) for start, end in zip(boundaries[:-1], boundaries[1:])]


def _target_periods(product):
    path = Path(cleaning_utils.get_target_temporal_path(product))
    if not path.exists():
        raise FileNotFoundError(f"Target temporal CSV not found: {path}")
    frame = pd.read_csv(path)
    start_column = "period_start" if "period_start" in frame else "date"
    if start_column not in frame:
        raise ValueError(f"Target temporal CSV has no period_start/date column: {path}")
    starts = pd.to_datetime(frame[start_column], errors="coerce").dt.normalize()
    if "period_end" in frame:
        ends = pd.to_datetime(frame["period_end"], errors="coerce").dt.normalize()
    else:
        ends = starts + pd.Timedelta(days=9)
    periods = pd.DataFrame({"target_start": starts, "target_end": ends}).dropna()
    periods = periods.sort_values("target_start").drop_duplicates("target_start").reset_index(drop=True)
    if periods.empty:
        raise ValueError(f"No valid target periods in {path}")
    return periods


def _inventory_runs(files, source):
    """Read only metadata needed to map target periods to forecast runs."""
    runs = []
    for path in tqdm(files, desc="Inventory ECMWF runs"):
        with xr.open_dataset(path) as dataset:
            required = {"tp", "time", "number", "latitude", "longitude"}
            missing = required.difference(dataset.variables)
            if missing:
                raise ValueError(f"{path.name} missing {sorted(missing)}")
            times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
            members = int(dataset.sizes["number"])
        if not times.is_monotonic_increasing or times.has_duplicates:
            raise ValueError(f"Time must be sorted and unique: {path.name}")
        for run_slice in _run_slices(times, source, path):
            run_times = times[run_slice]
            if not len(run_times):
                continue
            blocks = _blocks(run_times[0], run_times[-1])
            if blocks:
                runs.append({"init": pd.Timestamp(run_times[0]), "path": path,
                             "blocks": blocks, "ensemble_size": members})
    runs.sort(key=lambda item: item["init"])
    duplicate_dates = pd.Series([run["init"] for run in runs]).duplicated()
    if duplicate_dates.any():
        raise ValueError("Duplicate ECMWF initialization dates found across source files")
    return runs


def _select_runs(periods, runs, max_age_days=MAX_INIT_AGE_DAYS):
    """Select the latest forecast available on or before every target period."""
    selected = {}
    if not runs:
        return selected
    initializations = np.array([run["init"].to_datetime64() for run in runs])
    for row, target in enumerate(periods["target_start"]):
        position = int(np.searchsorted(initializations, target.to_datetime64(), side="right") - 1)
        if position < 0:
            continue
        age = int((target - runs[position]["init"]).days)
        if 0 <= age <= max_age_days:
            selected[row] = {**runs[position], "init_age_days": age}
    return selected


def _source_transform(latitudes, longitudes):
    lat_step = abs(float(latitudes[1] - latitudes[0]))
    lon_step = abs(float(longitudes[1] - longitudes[0]))
    return rasterio.transform.from_origin(float(np.min(longitudes)) - lon_step / 2,
                                          float(np.max(latitudes)) + lat_step / 2,
                                          lon_step, lat_step)


def _common_grid(sample_file):
    """Create a stable coarse grid over the reference domain plus one-cell margin."""
    with xr.open_dataset(sample_file) as dataset:
        latitudes = dataset.latitude.values
        longitudes = dataset.longitude.values
    resolution = max(abs(float(latitudes[1] - latitudes[0])),
                     abs(float(longitudes[1] - longitudes[0])))
    west, south, east, north = transform_bounds(
        cleaning_utils.MASK_REGIONS_REF_CRS,
        "EPSG:4326",
        *cleaning_utils.MASK_REGIONS_REF_BOUNDS,
    )
    west = math.floor((west - resolution) / resolution) * resolution
    east = math.ceil((east + resolution) / resolution) * resolution
    south = math.floor((south - resolution) / resolution) * resolution
    north = math.ceil((north + resolution) / resolution) * resolution
    width = int(round((east - west) / resolution))
    height = int(round((north - south) / resolution))
    transform = rasterio.transform.from_origin(west, north, resolution, resolution)
    latitudes = north - resolution * (np.arange(height) + 0.5)
    longitudes = west + resolution * (np.arange(width) + 0.5)
    return {"height": height, "width": width, "transform": transform,
            "crs": "EPSG:4326", "latitude": latitudes, "longitude": longitudes}


def _to_common_grid(data, latitudes, longitudes, grid):
    if latitudes[0] < latitudes[-1]:
        data = data[::-1, :]
    if longitudes[0] > longitudes[-1]:
        data = data[:, ::-1]
    output = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
    reproject(np.asarray(data, dtype=np.float32), output,
              src_transform=_source_transform(latitudes, longitudes), src_crs="EPSG:4326",
              src_nodata=np.nan, dst_transform=grid["transform"], dst_crs=grid["crs"],
              dst_nodata=np.nan, resampling=Resampling.bilinear)
    return output


def _backup_path(path):
    candidate = path.with_name(path.name + ".pre-coarse.bak")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(path.name + f".pre-coarse.{counter}.bak")
        counter += 1
    return candidate


def _same_timeline(hdf, periods, grid, source_file_count):
    if hdf.attrs.get("schema_version") != SCHEMA_VERSION:
        return False
    if hdf["X"].shape != (len(periods), 4, grid["height"], grid["width"]):
        return False
    if int(hdf.attrs.get("source_file_count", -1)) != source_file_count:
        return False
    stored = hdf["target_period_starts"].asstr()[:]
    expected = periods["target_start"].dt.strftime("%Y-%m-%d").to_numpy()
    return np.array_equal(stored, expected)


def _ensure_hdf5(path, periods, selected, grid, product, source, temporal_path,
                 source_file_count):
    if path.exists():
        with h5py.File(path, "r") as existing:
            compatible = _same_timeline(existing, periods, grid, source_file_count)
        if compatible:
            return
        backup = _backup_path(path)
        path.replace(backup)
        LOG.warning("Moved incompatible ECMWF HDF5 to %s", backup)
        if temporal_path.exists():
            csv_backup = _backup_path(temporal_path)
            temporal_path.replace(csv_backup)
            LOG.warning("Moved incompatible ECMWF CSV to %s", csv_backup)

    count = len(periods)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("X", shape=(count, 4, grid["height"], grid["width"]),
                           maxshape=(None, 4, grid["height"], grid["width"]),
                           chunks=(1, 1, grid["height"], grid["width"]), dtype="float32",
                           fillvalue=np.nan, compression="gzip", compression_opts=4, shuffle=True)
        hdf.create_dataset("target_period_starts", data=periods["target_start"].dt.strftime("%Y-%m-%d"),
                           maxshape=(None,), dtype=string_type)
        hdf.create_dataset("target_period_ends", data=periods["target_end"].dt.strftime("%Y-%m-%d"),
                           maxshape=(None,), dtype=string_type)
        init_times = hdf.create_dataset("init_times", shape=(count,), maxshape=(None,), dtype=string_type)
        source_files = hdf.create_dataset("source_files", shape=(count,), maxshape=(None,), dtype=string_type)
        ages = hdf.create_dataset("init_age_days", shape=(count,), maxshape=(None,), dtype="int16", fillvalue=-1)
        members = hdf.create_dataset("ensemble_size", shape=(count,), maxshape=(None,), dtype="int16")
        status = hdf.create_dataset("status", shape=(count,), maxshape=(None,), dtype="uint8", fillvalue=2)
        hdf.create_dataset("lead_start_days", data=LEAD_START_DAYS)
        hdf.create_dataset("lead_end_days", data=LEAD_END_DAYS)
        hdf.create_dataset("latitude", data=grid["latitude"].astype(np.float32))
        hdf.create_dataset("longitude", data=grid["longitude"].astype(np.float32))
        for row, run in selected.items():
            init_times[row] = run["init"].strftime("%Y-%m-%d")
            source_files[row] = run["path"].name
            ages[row] = run["init_age_days"]
            members[row] = run["ensemble_size"]
            status[row] = 0  # selected but not yet processed
        hdf.attrs.update(schema_version=SCHEMA_VERSION, target_product=product, source=source,
                         units="mm", aggregation="ensemble mean; four relative 10-day totals",
                         max_initialization_age_days=MAX_INIT_AGE_DAYS,
                         source_file_count=source_file_count,
                         crs=grid["crs"], transform=tuple(grid["transform"])[:6],
                         data_revision=0, csv_revision=-1)


def _process_pending(path, selected, grid):
    with h5py.File(path, "r+") as hdf:
        pending = {row: run for row, run in selected.items() if hdf["status"][row] == 0}
    if not pending:
        LOG.info("All selected ECMWF target periods are already processed")
        return False

    by_file = {}
    for row, run in pending.items():
        by_file.setdefault(run["path"], []).append((row, run))
    with h5py.File(path, "r+") as hdf:
        for source_path, rows in tqdm(by_file.items(), desc="Process ECMWF target forecasts"):
            with xr.open_dataset(source_path) as dataset:
                precipitation = dataset["tp"]
                latitudes = dataset.latitude.values
                longitudes = dataset.longitude.values
                for row, run in rows:
                    cube = np.empty((4, grid["height"], grid["width"]), dtype=np.float32)
                    for lead, (start, end) in enumerate(run["blocks"]):
                        rainfall = (precipitation.sel(time=end) - precipitation.sel(time=start)).mean(
                            "number", skipna=True)
                        cube[lead] = _to_common_grid(rainfall.values, latitudes, longitudes, grid)
                    hdf["X"][row] = cube
                    hdf["status"][row] = 1
            hdf.flush()
        hdf.attrs["data_revision"] = int(hdf.attrs.get("data_revision", 0)) + 1
    return True


def _reference_grid_and_masks():
    height, width = map(int, cleaning_utils.MASK_REGIONS_REF_SHAPE)
    transform = rasterio.Affine(*cleaning_utils.MASK_REGIONS_REF_TRANSFORM)
    crs = cleaning_utils.MASK_REGIONS_REF_CRS
    import geopandas as gpd
    catchments = gpd.read_file(get_cfg(
        "paths.maps.catchments", "data/maps/inflow_catchments/INFLOW_all_cmts.shp")).to_crs(crs)
    inside = geometry_mask(catchments.geometry, out_shape=(height, width),
                           transform=transform, invert=True)
    regions = cleaning_utils.extract_regions().to_crs(crs)
    masks = []
    for _, region in regions.iterrows():
        code = gridded_stats.region_to_code_dict.get(region["region"], region["region"])
        mask = geometry_mask([region.geometry], out_shape=(height, width), transform=transform,
                             invert=True, all_touched=True) & inside
        masks.append((code, mask))
    return {"height": height, "width": width, "transform": transform,
            "crs": crs, "inside": inside, "region_masks": masks}


def _to_reference(data, source_transform, reference):
    output = np.full((reference["height"], reference["width"]), np.nan, dtype=np.float32)
    reproject(data, output, src_transform=source_transform, src_crs="EPSG:4326", src_nodata=np.nan,
              dst_transform=reference["transform"], dst_crs=reference["crs"], dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    output[~reference["inside"]] = np.nan
    return output


def _csv_is_current(hdf_path, csv_path):
    if not csv_path.exists():
        return False
    with h5py.File(hdf_path, "r") as hdf:
        return int(hdf.attrs.get("csv_revision", -1)) == int(hdf.attrs.get("data_revision", 0))


def _write_temporal_csv_from_hdf5(hdf_path, csv_path):
    """Derive one temporal-feature row per target period directly from HDF5."""
    if _csv_is_current(hdf_path, csv_path):
        LOG.info("ECMWF temporal CSV is already current")
        return
    reference = _reference_grid_and_masks()
    rows = []
    with h5py.File(hdf_path, "r") as hdf:
        starts = hdf["target_period_starts"].asstr()[:]
        ends = hdf["target_period_ends"].asstr()[:]
        init_times = hdf["init_times"].asstr()[:]
        ages = hdf["init_age_days"][:]
        members = hdf["ensemble_size"][:]
        statuses = hdf["status"][:]
        source_transform = rasterio.Affine(*hdf.attrs["transform"])
        for index in tqdm(range(len(starts)), desc="Extract ECMWF temporal features"):
            available = bool(statuses[index] == 1)
            row = {"date": starts[index], "period_start": starts[index],
                   "period_end": ends[index], "ecmwf_init_time": init_times[index],
                   "ecmwf_init_age_days": int(ages[index]) if available else np.nan,
                   "ecmwf_has_forecast": available,
                   "ecmwf_ensemble_size": int(members[index]) if available else np.nan}
            for lead in range(4):
                total = np.nan
                regional = {code: np.nan for code, _ in reference["region_masks"]}
                if available:
                    rainfall = _to_reference(hdf["X"][index, lead], source_transform, reference)
                    valid = np.isfinite(rainfall)
                    if valid.any():
                        total = float(np.nansum(rainfall))
                    for code, mask in reference["region_masks"]:
                        cells = mask & valid
                        if cells.any():
                            regional[code] = float(np.nanmean(rainfall[cells]))
                lead_number = lead + 1
                row[f"ecmwf_rainfall_lead_{lead_number}"] = total
                for code, value in regional.items():
                    row[f"ecmwf_rainfall_{code}_lead_{lead_number}"] = value
            rows.append(row)
        revision = int(hdf.attrs.get("data_revision", 0))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with h5py.File(hdf_path, "r+") as hdf:
        hdf.attrs["csv_revision"] = revision


def process_ecmwf_hindcasts(download_path=None, output_h5=None, output_csv=None,
                            target_product=None, skip_download=False, source="hindcast"):
    """Build target-aligned coarse-grid ECMWF HDF5 and temporal CSV outputs."""
    product = cleaning_utils.resolve_target_product(
        target_product or get_cfg("runtime.target_product", "viirs"))
    folder = Path(download_path or Path(get_cfg("paths.downloads.root", "data/downloads")) / "ecmwf")
    files = sorted(folder.glob("*.nc"))
    files = [path for path in files if ("hin" in path.stem.lower()) == (source == "hindcast")]
    if not files and not skip_download and source == "hindcast":
        download_ecmwf_hindcasts(download_path=folder)
        files = [path for path in sorted(folder.glob("*.nc")) if "hin" in path.stem.lower()]
    if not files:
        raise FileNotFoundError(f"No {source} NetCDF files in {folder}")

    output = Path(output_h5 or cleaning_utils.get_target_historic_path("ecmwf_hindcasts.h5", product))
    temporal = Path(output_csv or cleaning_utils.get_target_historic_path(
        "ecmwf_hindcasts_temporal.csv", product))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporal.parent.mkdir(parents=True, exist_ok=True)

    grid = _common_grid(files[0])
    periods = _target_periods(product)
    if output.exists():
        with h5py.File(output, "r") as hdf:
            complete = (
                _same_timeline(hdf, periods, grid, len(files))
                and not np.any(hdf["status"][:] == 0)
            )
        if complete:
            LOG.info("ECMWF HDF5 is complete and unchanged; skipping source inventory")
            _write_temporal_csv_from_hdf5(output, temporal)
            return output

    runs = _inventory_runs(files, source)
    selected = _select_runs(periods, runs)
    LOG.info("Mapped %d of %d target periods to ECMWF runs on a %dx%d coarse grid",
             len(selected), len(periods), grid["height"], grid["width"])
    _ensure_hdf5(output, periods, selected, grid, product, source, temporal, len(files))
    _process_pending(output, selected, grid)
    _write_temporal_csv_from_hdf5(output, temporal)
    LOG.info("ECMWF processing complete: %s and %s", output, temporal)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-path")
    parser.add_argument("--output-h5")
    parser.add_argument("--output-csv")
    parser.add_argument("--target-product", choices=["viirs", "modis"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--source", choices=["hindcast", "forecast"], default="hindcast")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    process_ecmwf_hindcasts(arguments.download_path, arguments.output_h5, arguments.output_csv,
                            arguments.target_product, arguments.skip_download, arguments.source)


if __name__ == "__main__":
    main()
