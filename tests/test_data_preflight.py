from pathlib import Path

import pandas as pd
import pytest
import h5py

import model.temporal.data.preflight as preflight


def _write_sources(folder: Path, availability: list[bool]):
    dates = pd.date_range("2020-01-01", periods=len(availability), freq="10D")
    target = pd.DataFrame({
        "period_start": dates,
        "percent_inundation_ssd": [0.1] * len(dates),
    })
    target_path = folder / "inundation_viirs_temporal.csv"
    target.to_csv(target_path, index=False)
    with h5py.File(folder / "inundation_viirs.h5", "w") as output:
        output.create_dataset("inundation", shape=(len(dates), 1, 1))
    pd.DataFrame({
        "date": dates,
        "ecmwf_has_forecast": availability,
        "ecmwf_rainfall_ssd_lead_1": [1.0 if value else None for value in availability],
    }).to_csv(folder / "ecmwf_hindcasts_temporal.csv", index=False)
    with h5py.File(folder / "ecmwf_hindcasts.h5", "w") as output:
        output.create_dataset("X", shape=(len(dates), 4, 1, 1))
    return target_path


def _configure(monkeypatch, tmp_path):
    values = {
        "ablation.pipeline.target_column": "percent_inundation_ssd",
        "data_preflight.minimum_target_rows": 3,
        "data_preflight.maximum_target_missing_fraction": 0.0,
        "data_preflight.minimum_date_coverage_fraction": 1.0,
        "data_preflight.maximum_numeric_missing_fraction": 0.5,
        "data_preflight.ecmwf.minimum_available_fraction_after_first": 0.9,
        "data_preflight.ecmwf.maximum_trailing_missing_periods": 1,
        "data_preflight.report_paths.ablation": str(tmp_path / "report.json"),
    }
    monkeypatch.setattr(preflight, "get_cfg", lambda key, default=None: values.get(key, default))


def test_data_preflight_reports_timeseries_length_and_passes_complete_sources(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    target = _write_sources(tmp_path, [True, True, True, True])
    report = preflight.audit_temporal_data(
        "viirs", mode="ablation", aligned_dir=tmp_path, target_path=target
    )
    assert report["passed"] is True
    assert report["target_periods"] == 4
    assert (tmp_path / "report.json").exists()


def test_data_preflight_blocks_large_or_trailing_ecmwf_gaps(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    target = _write_sources(tmp_path, [True, True, False, False])
    with pytest.raises(RuntimeError, match="Data preflight failed"):
        preflight.audit_temporal_data(
            "viirs", mode="ablation", aligned_dir=tmp_path, target_path=target
        )
    report = pd.read_json(tmp_path / "report.json", typ="series")
    assert report["passed"] is False


def test_recent_source_coverage_flags_predictors_older_than_two_calendar_months(tmp_path):
    source = tmp_path / "rainfall.csv"
    pd.DataFrame({
        "date": ["2024-03-14", "2024-03-15"],
        "rainfall": [1.0, 2.0],
    }).to_csv(source, index=False)

    result = preflight.assess_recent_source_coverage(
        source,
        pd.Timestamp("2024-05-16"),
        maximum_missing_months=2,
    )

    assert result["excluded_from_training"] is True
    assert result["latest_usable_date"] == "2024-03-15"
    assert result["required_since_date"] == "2024-03-16"


def test_recent_source_coverage_keeps_data_on_two_month_boundary(tmp_path):
    source = tmp_path / "rainfall.csv"
    pd.DataFrame({"date": ["2024-03-16"], "rainfall": [1.0]}).to_csv(source, index=False)

    result = preflight.assess_recent_source_coverage(
        source,
        pd.Timestamp("2024-05-16"),
        maximum_missing_months=2,
    )

    assert result["excluded_from_training"] is False
