import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.data_cleaning.process_ecmwf import (
    LEAD_END_DAYS,
    LEAD_START_DAYS,
    SCHEMA_VERSION,
    _blocks,
    _ensure_hdf5,
    _inventory_runs,
    _merge_runs,
    _normalized_times,
    _run_slices,
    _select_runs,
)


def test_archived_forecast_placeholder_times_are_reconstructed_from_filename():
    expected = pd.date_range("2025-01-01", periods=47, freq="D")
    malformed = expected.to_numpy().copy()
    malformed[13:] = np.datetime64("1900-01-01")
    dataset = xr.Dataset(coords={"time": malformed})
    result = _normalized_times(
        dataset, "forecast", Path("sfc-24ac-ecmwf-20250101.nc")
    )
    assert result.equals(expected)


def test_forecast_time_repair_rejects_nonplaceholder_disagreement():
    expected = pd.date_range("2025-01-01", periods=47, freq="D")
    malformed = expected.to_numpy().copy()
    malformed[5] = np.datetime64("2025-01-08")
    malformed[13:] = np.datetime64("1900-01-01")
    dataset = xr.Dataset(coords={"time": malformed})
    with pytest.raises(ValueError, match="inconsistent non-placeholder times"):
        _normalized_times(dataset, "forecast", Path("sfc-24ac-ecmwf-20250101.nc"))


def test_ecmwf_inventory_reuses_unchanged_file_cache(tmp_path, monkeypatch):
    source = tmp_path / "sfc-24ac-ecmwf-20250101.nc"
    cache = tmp_path / "inventory.json"
    times = pd.date_range("2025-01-01", periods=47, freq="D")
    xr.Dataset(
        data_vars={"tp": (("time", "number", "latitude", "longitude"),
                          np.zeros((47, 1, 2, 2), dtype=np.float32))},
        coords={
            "time": times,
            "number": [1],
            "latitude": [1.0, 0.0],
            "longitude": [30.0, 31.0],
        },
    ).to_netcdf(source)

    first = _inventory_runs([source], "forecast", "forecast_2025", cache_path=cache)
    assert len(first) == 1
    assert cache.exists()

    monkeypatch.setattr(
        "processing.data_cleaning.process_ecmwf.xr.open_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = _inventory_runs([source], "forecast", "forecast_2025", cache_path=cache)
    assert second[0]["init"] == first[0]["init"]


def test_live_forecast_replaces_same_date_from_older_archive():
    init = pd.Timestamp("2026-01-01")
    old = {"init": init, "path": Path("archive.nc"), "source_kind": "forecast_2025"}
    live = {"init": init, "path": Path("live.nc"), "source_kind": "forecast_rt"}
    merged = _merge_runs([old], [live])
    assert len(merged) == 1
    assert merged[0]["path"].name == "live.nc"


def test_hindcast_runs_are_split_on_gaps_not_calendar_years():
    first = pd.date_range("2019-12-15", periods=47, freq="D")
    second = pd.date_range("2020-12-15", periods=47, freq="D")
    times = pd.DatetimeIndex(first.append(second))
    slices = _run_slices(times, "hindcast", Path("hindcast.nc"))
    assert [(item.start, item.stop) for item in slices] == [(0, 47), (47, 94)]


def test_latest_nonfuture_initialization_is_selected_with_age_limit():
    periods = pd.DataFrame({
        "target_start": pd.to_datetime(["2020-09-01", "2020-09-11", "2021-01-01"]),
        "target_end": pd.to_datetime(["2020-09-10", "2020-09-20", "2021-01-10"]),
    })
    runs = []
    for date in pd.to_datetime(["2020-08-31", "2020-09-09"]):
        runs.append({"init": date, "path": Path("source.nc"),
                     "blocks": _blocks(date, date + pd.Timedelta(days=46)),
                     "ensemble_size": 10})
    selected = _select_runs(periods, runs)
    assert selected[0]["init"] == pd.Timestamp("2020-08-31")
    assert selected[0]["init_age_days"] == 1
    assert selected[1]["init"] == pd.Timestamp("2020-09-09")
    assert selected[1]["init_age_days"] == 2
    assert 2 not in selected


def test_target_aligned_coarse_hdf5_schema(tmp_path):
    periods = pd.DataFrame({
        "target_start": pd.to_datetime(["2020-01-01", "2020-01-11"]),
        "target_end": pd.to_datetime(["2020-01-10", "2020-01-20"]),
    })
    init = pd.Timestamp("2020-01-01")
    selected = {0: {"init": init, "path": Path("source.nc"),
                    "blocks": _blocks(init, init + pd.Timedelta(days=46)),
                    "ensemble_size": 10, "init_age_days": 0}}
    grid = {
        "height": 3,
        "width": 4,
        "transform": rasterio.transform.from_origin(20, 20, 1.5, 1.5),
        "crs": "EPSG:4326",
        "latitude": np.array([19.25, 17.75, 16.25]),
        "longitude": np.array([20.75, 22.25, 23.75, 25.25]),
    }
    path = tmp_path / "ecmwf.h5"
    csv_path = tmp_path / "ecmwf.csv"
    _ensure_hdf5(path, periods, selected, grid, "viirs", "hindcast", csv_path, 178)
    with h5py.File(path, "r") as hdf:
        assert hdf["X"].shape == (2, 4, 3, 4)
        assert hdf.attrs["schema_version"] == SCHEMA_VERSION
        assert list(hdf["status"][:]) == [0, 2]
        assert list(hdf["init_times"].asstr()[:]) == ["2020-01-01", ""]
        np.testing.assert_array_equal(hdf["lead_start_days"][:], LEAD_START_DAYS)
        np.testing.assert_array_equal(hdf["lead_end_days"][:], LEAD_END_DAYS)
