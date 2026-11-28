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

import processing.data_cleaning.process_ecmwf as ecmwf
from processing.data_cleaning.process_ecmwf import (
    LEAD_END_DAYS,
    LEAD_START_DAYS,
    SCHEMA_VERSION,
    _blocks,
    _cleanup_obsolete_backups,
    _deduplicate_inventory_runs,
    _ensure_hdf5,
    _inventory_runs,
    _merge_runs,
    _normalized_times,
    _run_slices,
    _select_runs,
)


def test_successful_migration_cleanup_removes_only_ecmwf_backups(tmp_path):
    hdf = tmp_path / "ecmwf.h5"
    csv = tmp_path / "ecmwf.csv"
    backups = [
        tmp_path / "ecmwf.h5.pre-coarse.bak",
        tmp_path / "ecmwf.h5.pre-coarse.1.bak",
        tmp_path / "ecmwf.csv.pre-coarse.bak",
    ]
    unrelated = tmp_path / "ecmwf.h5.other.bak"
    for path in [*backups, unrelated]:
        path.write_text("backup", encoding="utf-8")

    removed = _cleanup_obsolete_backups(hdf, csv)

    assert set(removed) == set(backups)
    assert all(not path.exists() for path in backups)
    assert unrelated.exists()


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


def test_corrupt_ecmwf_file_is_redownloaded_before_inventory(tmp_path, monkeypatch):
    source = tmp_path / "sfc-24ac-ecmwf-20250101.nc"
    source.write_bytes(b"not-netcdf")
    cache = tmp_path / "inventory.json"
    calls = {"inspect": 0, "download": 0}

    def inspect(path, source_type, source_kind):
        calls["inspect"] += 1
        if calls["inspect"] == 1:
            raise ValueError("invalid file")
        init = pd.Timestamp("2025-01-01")
        return [{
            "init": init, "path": path,
            "blocks": _blocks(init, init + pd.Timedelta(days=46)),
            "ensemble_size": 100, "source_kind": source_kind,
        }]

    def redownload(base_url, folder, **kwargs):
        calls["download"] += 1
        source.write_bytes(b"replacement")
        return [source]

    monkeypatch.setattr(ecmwf, "_inspect_inventory_file", inspect)
    monkeypatch.setattr(ecmwf, "download_ecmwf_files", redownload)
    runs = _inventory_runs(
        [source], "forecast", "forecast_2025", cache_path=cache,
        base_url="https://example.test/",
    )
    assert len(runs) == 1
    assert calls == {"inspect": 2, "download": 1}
    assert list(tmp_path.glob("*.corrupt"))


def test_unrepairable_ecmwf_file_is_quarantined_and_skipped(tmp_path, monkeypatch):
    source = tmp_path / "sfc-24ac-ecmwf-20250101.nc"
    source.write_bytes(b"not-netcdf")
    monkeypatch.setattr(
        ecmwf, "_inspect_inventory_file",
        lambda *args: (_ for _ in ()).throw(ValueError("invalid file")),
    )
    monkeypatch.setattr(
        ecmwf, "download_ecmwf_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("server failed")),
    )
    runs = _inventory_runs(
        [source], "forecast", "forecast_2025",
        cache_path=tmp_path / "inventory.json", base_url="https://example.test/",
    )
    assert runs == []
    assert not source.exists()
    assert list(tmp_path.glob("*.corrupt"))


def test_duplicate_initialization_prefers_canonical_file_over_copy(tmp_path):
    canonical = tmp_path / "sfc-24ac-ecmwf-20260601.nc"
    duplicate = tmp_path / "sfc-24ac-ecmwf-20260601_COPY.nc"
    canonical.write_bytes(b"canonical")
    duplicate.write_bytes(b"larger-copy-file")
    init = pd.Timestamp("2026-06-01")
    common = {
        "init": init,
        "blocks": _blocks(init, init + pd.Timedelta(days=46)),
        "ensemble_size": 100,
        "source_kind": "forecast_rt",
    }
    result = _deduplicate_inventory_runs(
        [{**common, "path": duplicate}, {**common, "path": canonical}],
        "forecast_rt",
    )
    assert len(result) == 1
    assert result[0]["path"] == canonical


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
