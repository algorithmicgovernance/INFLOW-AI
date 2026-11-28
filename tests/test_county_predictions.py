from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import box

from model.spatial.county_predictions import _plot_year_by_year, _trigger_summary
from processing.data_cleaning import process_inundation_viirs as viirs


def test_county_trigger_uses_historic_annual_range_percentiles():
    dates = []
    values = []
    for year, annual_range in zip(range(2021, 2026), [0.01, 0.02, 0.03, 0.04, 0.05]):
        dates.extend([f"{year}-01-01", f"{year}-12-21"])
        values.extend([0.01, 0.01 + annual_range])
    dates.extend(["2026-01-01", "2026-10-21"])
    values.extend([0.01, 0.02])
    result = _trigger_summary(
        pd.Series(pd.to_datetime(dates)),
        np.asarray(values),
        pd.DatetimeIndex(["2026-11-01", "2026-12-21"]),
        np.asarray([0.03, 0.055]),
    )
    assert result["trigger_active"] is True
    assert result["trigger_level"] == "moderate"
    assert result["forecast_peak_date"] == "2026-12-21"
    assert np.isclose(result["forecast_increase_fraction"], 0.045)


def test_zero_inundation_county_plot_omits_year_annotations(tmp_path, monkeypatch):
    annotations = []
    monkeypatch.setattr(
        "matplotlib.axes.Axes.annotate",
        lambda self, *args, **kwargs: annotations.append((args, kwargs)),
    )

    _plot_year_by_year(
        pd.Series(pd.to_datetime(["2025-01-01", "2025-12-21", "2026-01-01"])),
        np.zeros(3),
        pd.DatetimeIndex(pd.to_datetime(["2026-11-01", "2026-11-11"])),
        np.zeros(2),
        "Zero County",
        tmp_path / "zero.png",
    )

    assert annotations == []


def test_county_temporal_csv_aggregates_each_h5_mask(tmp_path, monkeypatch):
    h5_path = tmp_path / "viirs.h5"
    temporal_path = tmp_path / "inundation_viirs_temporal.csv"
    output_path = tmp_path / "inundation_viirs_county_temporal.csv"
    with h5py.File(h5_path, "w") as output:
        output.create_dataset("inundation", data=np.asarray([
            [[1, 0], [0, 1]],
            [[0, 1], [1, 1]],
        ], dtype=np.uint8))
    pd.DataFrame({
        "file_name": ["a.tif", "b.tif"],
        "period_start": ["2026-01-01", "2026-01-11"],
        "percent_inundation_ssd": [0.5, 0.75],
    }).to_csv(temporal_path, index=False)
    counties = gpd.GeoDataFrame({
        "ADM2_EN": ["Left", "Right"], "ADM2_PCODE": ["SS01", "SS02"],
        "ADM1_EN": ["State", "State"], "ADM1_PCODE": ["S1", "S1"],
        "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
    }, crs="EPSG:6933")
    counties.attrs["source_path"] = "counties.shp"
    labels = np.asarray([[1, 2], [1, 2]], dtype=np.uint16)
    monkeypatch.setattr(viirs, "county_reference_labels", lambda path=None: (counties, labels))

    result_path, _, _ = viirs.build_county_viirs_temporal_csv(
        h5_path, temporal_path, output_path=output_path,
    )

    result = pd.read_csv(result_path)
    assert "percent_inundation_ssd" not in result
    assert result["percent_inundation_ss01"].tolist() == [0.5, 0.5]
    assert result["percent_inundation_ss02"].tolist() == [0.5, 1.0]
    assert Path(result_path).name == "inundation_viirs_county_temporal.csv"
