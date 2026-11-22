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
