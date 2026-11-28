from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from processing.data_cleaning import process_gridded_rainfall as rainfall
from processing.data_cleaning.gridded_streaming import GridMasks, PeriodFiles


def test_rainfall_bootstrap_then_append_keeps_hdf_and_csv_aligned(monkeypatch, tmp_path):
    hdf_path = tmp_path / "gridded_rainfall.h5"
    csv_path = tmp_path / "gridded_rainfall_temporal.csv"
    periods = [
        PeriodFiles("2020-01-01", (Path("period-0.nc"),)),
        PeriodFiles("2020-01-11", (Path("period-1.nc"),)),
    ]
    processed = []

    monkeypatch.setattr(rainfall, "GR_RAINFALL_H5_PATH", str(hdf_path))
    monkeypatch.setattr(rainfall, "download_gridded_rainfall", lambda *args, **kwargs: None)
    monkeypatch.setattr(rainfall, "inventory_daily_files", lambda folder: {pd.Timestamp("2020-01-01"): Path("x")})
    monkeypatch.setattr(rainfall, "complete_periods", lambda files: list(periods))
    monkeypatch.setattr(rainfall, "validate_daily_files", lambda periods, variable: [])
    masks = GridMasks(3, 4, None, None, np.ones((3, 4), dtype=bool), [])
    monkeypatch.setattr(rainfall, "build_grid_masks", lambda path: masks)

    def fake_grid(files, variable, grid_masks):
        path = files[0]
        processed.append(path.name)
        value = float(int(path.stem.split("-")[-1]) + 1)
        return np.full((3, 4), value, dtype=np.float32)

    monkeypatch.setattr(rainfall, "stream_period_grid", fake_grid)

    rainfall.update_gridded_rainfall(
        download_folder=str(tmp_path / "downloads"),
        download_path=str(tmp_path / "daily"),
        extract_folder=str(tmp_path / "domain"),
        dekads_path=str(tmp_path / "dekads"),
        temporal_data_path=str(csv_path),
    )

    assert processed == ["period-0.nc", "period-1.nc"]
    with h5py.File(hdf_path, "r") as hdf:
        assert hdf["rainfall"].shape == (2, 3, 4)
        assert hdf["dates"].asstr()[:].tolist() == ["2020-01-01", "2020-01-11"]
    assert pd.read_csv(csv_path)["date"].tolist() == ["2020-01-01", "2020-01-11"]

    periods.append(PeriodFiles("2020-01-21", (Path("period-2.nc"),)))
    processed.clear()
    rainfall.update_gridded_rainfall(
        download_folder=str(tmp_path / "downloads"),
        download_path=str(tmp_path / "daily"),
        extract_folder=str(tmp_path / "domain"),
        dekads_path=str(tmp_path / "dekads"),
        temporal_data_path=str(csv_path),
    )

    assert processed == ["period-2.nc"]
    with h5py.File(hdf_path, "r") as hdf:
        assert hdf["rainfall"].shape == (3, 3, 4)
        assert hdf["dates"].asstr()[:].tolist() == [
            "2020-01-01", "2020-01-11", "2020-01-21"
        ]
    rainfall._validate_rainfall_outputs(
        hdf_path, csv_path, expected_dates=["2020-01-01", "2020-01-11", "2020-01-21"]
    )


def test_missing_temporal_csv_is_recreated_from_hdf(monkeypatch, tmp_path):
    hdf_path = tmp_path / "rainfall.h5"
    csv_path = tmp_path / "rainfall.csv"
    with h5py.File(hdf_path, "w") as hdf:
        hdf.create_dataset("rainfall", data=np.ones((2, 2, 2), dtype=np.float32))
        hdf.create_dataset(
            "dates", data=np.asarray(["2020-01-01", "2020-01-11"], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
    regions = pd.DataFrame({"region": []})
    rainfall._temporal_csv_from_hdf(hdf_path, csv_path, regions)
    assert pd.read_csv(csv_path)["date"].tolist() == ["2020-01-01", "2020-01-11"]
    rainfall._validate_rainfall_outputs(hdf_path, csv_path)

