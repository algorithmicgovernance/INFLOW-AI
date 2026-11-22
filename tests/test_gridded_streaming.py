from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from processing.data_cleaning import process_gridded_rainfall_cumulative as cumulative
from processing.data_cleaning import gridded_streaming
from processing.data_cleaning.gridded_streaming import (
    GridMasks, PeriodFiles, _read_daily_subset, complete_periods, validate_daily_files,
)


def test_complete_periods_excludes_incomplete_dekads():
    files = {
        date: Path(f"day-{date.day}.nc")
        for date in pd.date_range("2020-01-01", "2020-01-19", freq="D")
    }
    periods = complete_periods(files)
    assert [period.start for period in periods] == ["2020-01-01"]
    assert len(periods[0].files) == 10


def test_hdf5_fallback_reads_file_without_valid_netcdf_metadata(tmp_path):
    path = tmp_path / "sm2020_01_01.v2.3.1.nc"
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("lat", data=np.array([2.0, 1.0, 0.0]))
        hdf.create_dataset("lon", data=np.array([30.0, 31.0, 32.0]))
        hdf.create_dataset("sm_c4grass", data=np.arange(9, dtype=np.float32).reshape(1, 3, 3))
    period = PeriodFiles("2020-01-01", (path,))
    assert validate_daily_files([period], "sm_c4grass") == []
    lats, lons, values = _read_daily_subset(path, "sm_c4grass", (2.0, 0.0, 30.0, 32.0))
    assert values.shape == (3, 3)
    assert lats.tolist() == [2.0, 1.0, 0.0]
    assert lons.tolist() == [30.0, 31.0, 32.0]


def test_daily_validation_cache_skips_unchanged_files(monkeypatch, tmp_path):
    path = tmp_path / "rfe2020_01_01.v3.1.nc"
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("lat", data=np.array([1.0]))
        hdf.create_dataset("lon", data=np.array([30.0]))
        hdf.create_dataset("rfe", data=np.ones((1, 1, 1), dtype=np.float32))
    period = PeriodFiles("2020-01-01", (path,))
    calls = 0
    original = gridded_streaming._validate_daily_file

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(gridded_streaming, "_validate_daily_file", counted)
    assert validate_daily_files([period], "rfe") == []
    assert calls == 1
    assert validate_daily_files([period], "rfe") == []
    assert calls == 1

    path.touch()
    assert validate_daily_files([period], "rfe") == []
    assert calls == 2


def test_cumulative_rainfall_rebuild_then_append(monkeypatch, tmp_path):
    rainfall_hdf = tmp_path / "rainfall.h5"
    cumulative_hdf = tmp_path / "cumulative.h5"
    cumulative_csv = tmp_path / "cumulative.csv"

    def write_rainfall(values, dates):
        with h5py.File(rainfall_hdf, "w") as hdf:
            hdf.create_dataset("rainfall", data=np.asarray(values, dtype=np.float32))
            hdf.create_dataset(
                "dates", data=np.asarray(dates, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )

    write_rainfall(
        [np.ones((2, 2)), np.full((2, 2), 2.0)],
        ["2020-01-01", "2020-01-11"],
    )
    masks = GridMasks(2, 2, None, None, np.ones((2, 2), dtype=bool), [])
    monkeypatch.setattr(cumulative, "GR_RAINFALL_H5_PATH", str(rainfall_hdf))
    monkeypatch.setattr(cumulative, "build_grid_masks", lambda path: masks)

    cumulative.update_gridded_rainfall_cumulative(
        data_path=str(cumulative_hdf), temporal_data_path=str(cumulative_csv)
    )
    with h5py.File(cumulative_hdf, "r") as hdf:
        np.testing.assert_allclose(hdf["cumulative_rainfall"][0], 1.0)
        np.testing.assert_allclose(hdf["cumulative_rainfall"][1], 3.0)

    write_rainfall(
        [np.ones((2, 2)), np.full((2, 2), 2.0), np.full((2, 2), 4.0)],
        ["2020-01-01", "2020-01-11", "2020-01-21"],
    )
    cumulative.update_gridded_rainfall_cumulative(
        data_path=str(cumulative_hdf), temporal_data_path=str(cumulative_csv)
    )
    with h5py.File(cumulative_hdf, "r") as hdf:
        assert hdf["cumulative_rainfall"].shape[0] == 3
        np.testing.assert_allclose(hdf["cumulative_rainfall"][2], 7.0)
        assert hdf["dates"].asstr()[:].tolist() == [
            "2020-01-01", "2020-01-11", "2020-01-21"
        ]

