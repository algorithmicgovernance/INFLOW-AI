import os
import sys
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing import cleaning_utils
import processing.data_cleaning.process_inundation_viirs as viirs
from processing.data_cleaning.process_ecmwf import _blocks
from processing.data_cleaning.process_inundation_viirs import (
    parse_viirs_filename,
    viirs_target_filename,
)


def test_viirs_v2_date_filename_is_parsed_as_dekad():
    parsed = parse_viirs_filename("south_sudan_flood_2024-02-21.tif")
    assert parsed["period_type"] == "D3"
    assert parsed["period_order"] == 2
    assert parsed["period_start"].isoformat() == "2024-02-21"
    assert parsed["period_end"].isoformat() == "2024-02-29"
    assert viirs_target_filename("south_sudan_flood_2024-02-21.tif") == "2024-02-21.tif"


def test_viirs_v2_compact_and_dekad_number_filenames_are_supported():
    assert parse_viirs_filename("floodmap_20240111.tif")["period_type"] == "D2"
    assert parse_viirs_filename("floodmap_202401d3.tif")["period_start"].day == 21


def test_viirs_v2_rejects_non_dekadal_dates():
    try:
        parse_viirs_filename("floodmap_2024-01-16.tif")
    except ValueError as error:
        assert "not a dekad start" in str(error)
    else:
        raise AssertionError("Non-dekadal VIIRS date was accepted")


def test_viirs_target_grouping_and_ecmwf_blocks_use_fixed_relative_leads():
    dates = [datetime(2024, 1, day) for day in range(1, 32)]
    groups, _ = cleaning_utils.group_dates_by_target_period(dates, "viirs")
    assert [group[0].day for group in groups] == [1, 11, 21]
    assert [len(group) for group in groups] == [10, 10, 11]
    blocks = _blocks(pd.Timestamp("2024-01-07"), pd.Timestamp("2024-02-22"), "viirs")
    assert blocks == [
        (pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-17")),
        (pd.Timestamp("2024-01-17"), pd.Timestamp("2024-01-27")),
        (pd.Timestamp("2024-01-27"), pd.Timestamp("2024-02-06")),
        (pd.Timestamp("2024-02-06"), pd.Timestamp("2024-02-16")),
    ]


def test_remote_listing_failure_retries_then_raises(monkeypatch):
    attempts = []

    def fail_request(*args, **kwargs):
        attempts.append(1)
        raise requests.exceptions.SSLError("unexpected EOF")

    monkeypatch.setattr(viirs.requests, "get", fail_request)
    monkeypatch.setattr(viirs.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Unable to verify remote VIIRS archive"):
        viirs.list_remote_tif_files(max_retries=3, retry_delay_seconds=0)
    assert len(attempts) == 3


def test_failed_listing_cannot_trigger_legacy_rebuild(tmp_path, monkeypatch):
    h5_path = tmp_path / "viirs.h5"
    csv_path = tmp_path / "viirs.csv"
    download_path = tmp_path / "downloads"
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("inundation", data=np.zeros((1, 2, 2), dtype=np.uint8),
                          maxshape=(None, 2, 2))

    monkeypatch.setattr(
        viirs,
        "download_new_inundation",
        lambda path: (_ for _ in ()).throw(RuntimeError("listing unavailable")),
    )
    rebuild_calls = []
    monkeypatch.setattr(viirs, "rebuild_viirs_history",
                        lambda *args: rebuild_calls.append(args))

    with pytest.raises(RuntimeError, match="listing unavailable"):
        viirs.update_inundation(
            h5_file_path=str(h5_path),
            download_path=str(download_path),
            temporal_data_path=str(csv_path),
        )
    assert rebuild_calls == []


def test_prepare_data_can_use_an_exact_validated_offline_cache(tmp_path, monkeypatch):
    h5_path = tmp_path / "viirs.h5"
    csv_path = tmp_path / "viirs.csv"
    download_path = tmp_path / "downloads"
    download_path.mkdir()
    names = ["2020-01-01.tif", "2020-01-11.tif"]
    for name in names:
        (download_path / name).touch()
    pd.DataFrame({
        "file_name": names,
        "period_start": ["2020-01-01", "2020-01-11"],
    }).to_csv(csv_path, index=False)
    height, width = cleaning_utils.MASK_REGIONS_REF_SHAPE
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset(
            "inundation",
            shape=(2, height, width),
            maxshape=(None, height, width),
            dtype=np.uint8,
            chunks=(1, height, width),
            fillvalue=0,
        )

    monkeypatch.setattr(
        viirs,
        "download_new_inundation",
        lambda path: (_ for _ in ()).throw(RuntimeError("listing unavailable")),
    )
    viirs.update_inundation(
        h5_file_path=str(h5_path),
        download_path=str(download_path),
        temporal_data_path=str(csv_path),
        allow_offline_existing=True,
    )
    with h5py.File(h5_path, "r") as hf:
        assert hf["inundation"].attrs["dataset_version"] == "2.0"
        assert hf["inundation"].attrs["cadence"] == "dekadal"


def test_preliminary_catalogue_keeps_latest_version_per_dekad(monkeypatch):
    class Response:
        text = (
            "dekad,version,source,flooded_km2,file\n"
            "2026-10-01,0,VIIRS,10,prelim/ssdmask_20261001_prelim_20261015.tif\n"
            "2026-10-01,2,VIIRS,12,prelim/ssdmask_20261001_prelim_20261031.tif\n"
            "2026-10-11,0,VIIRS,11,prelim/ssdmask_20261011_prelim_20261023.tif\n"
        )

    monkeypatch.setattr(viirs, "_request_with_retries", lambda url: Response())
    result = viirs._latest_preliminary_catalogue()
    assert result["version"].tolist() == [2, 0]
    assert result["dekad"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-10-01", "2026-10-11"
    ]


def test_operational_preliminary_view_does_not_modify_final_history(tmp_path, monkeypatch):
    historic_h5 = tmp_path / "historic.h5"
    historic_csv = tmp_path / "historic.csv"
    prelim_dir = tmp_path / "prelim"
    operational_h5 = tmp_path / "operational.h5"
    prelim_h5 = tmp_path / "prelim.h5"
    operational_csv = tmp_path / "operational.csv"
    original = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
    with h5py.File(historic_h5, "w") as output:
        output.create_dataset("inundation", data=original)
    pd.DataFrame({
        "file_name": ["2026-10-01.tif"], "period_start": ["2026-10-01"],
        "year": [2026], "month": [10], "period_order": [0],
        "percent_inundation": [0.25],
    }).to_csv(historic_csv, index=False)
    catalogue = pd.DataFrame({
        "dekad": pd.to_datetime(["2026-10-01", "2026-10-11"]),
        "version": [2, 0],
        "file": ["prelim/finalized.tif", "prelim/ssdmask_20261011_prelim_20261023.tif"],
    })
    preliminary = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    configured = {
        "paths.operational.viirs_h5": str(operational_h5),
        "paths.operational.viirs_prelim_h5": str(prelim_h5),
        "paths.operational.viirs_temporal": str(operational_csv),
    }
    monkeypatch.setattr(viirs, "get_cfg", lambda key, default=None: configured.get(key, default))
    monkeypatch.setattr(viirs, "_latest_preliminary_catalogue", lambda: catalogue)
    monkeypatch.setattr(
        viirs, "_download_latest_preliminary",
        lambda rows, path: ["ssdmask_20261011_prelim_20261023.tif"],
    )
    monkeypatch.setattr(viirs, "load_shapefile", lambda path: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(
        viirs, "process_and_clip_rasters",
        lambda files, path, catchments: ([preliminary], files, {}),
    )
    monkeypatch.setattr(viirs.cleaning_utils, "extract_regions", lambda: None)

    result_h5, result_csv, _ = viirs.prepare_operational_inundation(
        historic_h5_path=historic_h5,
        historic_temporal_path=historic_csv,
        prelim_download_path=prelim_dir,
    )
    with h5py.File(historic_h5, "r") as source:
        np.testing.assert_array_equal(source["inundation"][:], original)
    with h5py.File(result_h5, "r") as source:
        assert source["inundation"].shape == (2, 2, 2)
        np.testing.assert_array_equal(source["inundation"][1], preliminary)
    result = pd.read_csv(result_csv)
    assert result["map_status"].tolist() == ["final", "preliminary"]
