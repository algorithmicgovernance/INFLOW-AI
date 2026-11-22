import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.data_cleaning.process_ecmwf import (
    LEAD_END_DAYS,
    LEAD_START_DAYS,
    SCHEMA_VERSION,
    _blocks,
    _ensure_hdf5,
    _run_slices,
    _select_runs,
)


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
