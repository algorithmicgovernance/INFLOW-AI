from datetime import date

import pandas as pd
import pytest

import early_action_trigger
from early_action_trigger import (
    configured_trigger_percentiles,
    evaluate_early_action_trigger,
    evaluate_trigger_series,
)


def _history_with_annual_ranges():
    dates = []
    values = []
    for year, annual_range in zip(range(2021, 2026), [0.01, 0.02, 0.03, 0.04, 0.05]):
        dates.extend([f"{year}-01-01", f"{year}-12-21"])
        values.extend([0.01, 0.01 + annual_range])
    dates.extend(["2026-01-01", "2026-10-21"])
    values.extend([0.01, 0.02])
    return dates, values


def test_trigger_percentiles_are_loaded_from_config(monkeypatch):
    configured = {"moderate": 0.75, "severe": 0.85, "extreme": 0.97}
    monkeypatch.setattr(
        early_action_trigger, "get_cfg",
        lambda path, default=None: configured.get(path.rsplit(".", 1)[-1], default),
    )
    assert configured_trigger_percentiles() == configured


def test_percentile_trigger_uses_forecast_maximum_and_anchored_annual_minimum(tmp_path):
    history_dates, history_values = _history_with_annual_ranges()
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- EARLY_ACTION_TRIGGER_STATUS_START -->old"
        "<!-- EARLY_ACTION_TRIGGER_STATUS_END -->",
        encoding="utf-8",
    )
    result = evaluate_early_action_trigger(
        history_dates=history_dates,
        history_values=history_values,
        forecast_dates=["2026-11-01", "2026-12-21"],
        forecast_values=[0.03, 0.055],
        output_dir=tmp_path,
        percentiles={"moderate": 0.80, "severe": 0.90, "extreme": 0.95},
        alert_date=date(2026, 11, 28),
        readme_path=readme,
        area_km2=100_000,
    )

    assert result.severity == "moderate"
    assert result.prediction_date == pd.Timestamp("2026-12-21")
    assert result.annual_minimum_fraction == pytest.approx(0.01)
    assert result.increase_fraction == pytest.approx(0.045)
    assert result.window_start == pd.Timestamp("2025-12-21")
    assert result.window_end == pd.Timestamp("2026-12-21")
    assert "Moderate alert triggered" in (tmp_path / "TRIGGER ACTIVATED.txt").read_text()
    assert "| Forecast increase | 4,500 km² |" in readme.read_text()


def test_trigger_levels_use_strict_historic_percentile_thresholds():
    history_dates, history_values = _history_with_annual_ranges()
    result = evaluate_trigger_series(
        history_dates=history_dates,
        history_values=history_values,
        forecast_dates=["2026-11-01", "2026-12-21"],
        forecast_values=[0.02, 0.08],
        percentiles={"moderate": 0.80, "severe": 0.90, "extreme": 0.95},
    )
    assert result["trigger_level"] == "extreme"
    assert result["historic_window_count"] == 4


def test_inactive_trigger_removes_obsolete_alert(tmp_path):
    alert_path = tmp_path / "TRIGGER ACTIVATED.txt"
    alert_path.write_text("obsolete", encoding="utf-8")
    history_dates, history_values = _history_with_annual_ranges()
    result = evaluate_early_action_trigger(
        history_dates=history_dates,
        history_values=history_values,
        forecast_dates=["2026-11-01", "2026-12-21"],
        forecast_values=[0.011, 0.012],
        output_dir=tmp_path,
        percentiles={"moderate": 0.80, "severe": 0.90, "extreme": 0.95},
    )
    assert result.activated is False
    assert not alert_path.exists()


def test_no_history_signal_requires_five_percent_forecast_extent():
    dates = []
    values = []
    for year in range(2021, 2027):
        dates.extend([f"{year}-01-01", f"{year}-10-21"])
        values.extend([0.0, 0.0])

    noise = evaluate_trigger_series(
        history_dates=dates, history_values=values,
        forecast_dates=["2026-11-01", "2026-12-21"],
        forecast_values=[0.0, 0.001],
        no_history_minimum_forecast_fraction=0.05,
    )
    material = evaluate_trigger_series(
        history_dates=dates, history_values=values,
        forecast_dates=["2026-11-01", "2026-12-21"],
        forecast_values=[0.01, 0.051],
        no_history_minimum_forecast_fraction=0.05,
    )

    assert noise["trigger_active"] is False
    assert noise["no_history_forecast_gate_passed"] is False
    assert material["trigger_level"] == "extreme"
    assert material["no_history_forecast_gate_passed"] is True
