"""Percentile-based S-EAP early-action triggers for inundation forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from processing.config import get_cfg

LOG = logging.getLogger(__name__)
README_STATUS_START = "<!-- EARLY_ACTION_TRIGGER_STATUS_START -->"
README_STATUS_END = "<!-- EARLY_ACTION_TRIGGER_STATUS_END -->"
TRIGGER_LEVELS = ("moderate", "severe", "extreme")


@dataclass(frozen=True)
class EarlyActionTriggerResult:
    activated: bool
    severity: str | None
    annual_minimum_fraction: float | None
    forecast_maximum_fraction: float | None
    increase_fraction: float | None
    prediction_date: pd.Timestamp | None
    window_start: pd.Timestamp | None
    window_end: pd.Timestamp | None
    thresholds: dict[str, float]
    historic_window_count: int
    area_km2: float
    output_path: Path | None
    message: str | None


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _area(value: float | None, area_km2: float) -> str:
    return "n/a" if value is None else f"{float(value) * area_km2:,.0f} km²"


def configured_trigger_percentiles() -> dict[str, float]:
    defaults = {"moderate": 0.80, "severe": 0.90, "extreme": 0.95}
    configured = {
        level: float(get_cfg(f"early_action_trigger.percentiles.{level}", quantile))
        for level, quantile in defaults.items()
    }
    values = [configured[level] for level in TRIGGER_LEVELS]
    if any(not 0.0 < value < 1.0 for value in values) or values != sorted(values):
        raise ValueError(
            "early_action_trigger.percentiles must be ordered fractions between 0 and 1"
        )
    return configured


def _series_frame(dates: Iterable[object], values: Iterable[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates), errors="coerce"),
        "value": pd.to_numeric(pd.Series(list(values)), errors="coerce"),
    }).dropna().sort_values("date")


def evaluate_trigger_series(
    *,
    history_dates: Iterable[object],
    history_values: Iterable[float],
    forecast_dates: Iterable[object],
    forecast_values: Iterable[float],
    percentiles: Mapping[str, float] | None = None,
    no_history_minimum_forecast_fraction: float | None = None,
) -> dict[str, object]:
    """Compare forecast increase with historic anchored annual flood ranges."""
    quantiles = dict(percentiles or configured_trigger_percentiles())
    history = _series_frame(history_dates, history_values)
    forecast = _series_frame(forecast_dates, forecast_values)
    if history.empty or forecast.empty:
        return {
            "trigger_active": False, "trigger_level": None,
            "annual_minimum": None, "forecast_maximum": None,
            "increase": None, "forecast_peak_date": None,
            "window_start": None, "window_end": None,
            "thresholds": {}, "historic_annual_increases": [],
            "historic_window_count": 0,
            "historic_signal_available": False,
            "no_history_forecast_gate_passed": False,
        }

    window_end = pd.Timestamp(forecast["date"].max()).normalize()
    window_start = window_end - pd.DateOffset(years=1)
    current_history = history.loc[
        (history["date"] > window_start) & (history["date"] <= window_end), "value"
    ]
    annual_minimum = None if current_history.empty else float(current_history.min())
    peak_row = forecast.loc[forecast["value"].idxmax()]
    forecast_maximum = float(peak_row["value"])
    peak_date = pd.Timestamp(peak_row["date"])

    historic_increases: list[float] = []
    for end_year in range(int(history["date"].min().year) + 1, int(window_end.year)):
        try:
            historic_end = window_end.replace(year=end_year)
        except ValueError:
            historic_end = window_end.replace(year=end_year, day=28)
        historic_start = historic_end - pd.DateOffset(years=1)
        values = history.loc[
            (history["date"] > historic_start) & (history["date"] <= historic_end), "value"
        ]
        if len(values) >= 2:
            historic_increases.append(float(values.max() - values.min()))

    thresholds = {
        level: float(np.quantile(historic_increases, quantile))
        for level, quantile in quantiles.items()
    } if historic_increases else {}
    increase = None if annual_minimum is None else forecast_maximum - annual_minimum
    severity = None
    historic_signal_available = bool(historic_increases) and max(historic_increases) > 0.0
    no_history_forecast_gate_passed = (
        no_history_minimum_forecast_fraction is None
        or historic_signal_available
        or forecast_maximum >= no_history_minimum_forecast_fraction
    )
    if increase is not None:
        for level in reversed(TRIGGER_LEVELS):
            if (
                level in thresholds
                and increase > thresholds[level]
                and no_history_forecast_gate_passed
            ):
                severity = level
                break

    return {
        "trigger_active": severity is not None,
        "trigger_level": severity,
        "annual_minimum": annual_minimum,
        "forecast_maximum": forecast_maximum,
        "increase": increase,
        "forecast_peak_date": peak_date,
        "window_start": window_start,
        "window_end": window_end,
        "thresholds": thresholds,
        "historic_annual_increases": historic_increases,
        "historic_window_count": len(historic_increases),
        "historic_signal_available": historic_signal_available,
        "no_history_forecast_gate_passed": no_history_forecast_gate_passed,
    }


def _update_readme_status(result: EarlyActionTriggerResult, readme_path: str | Path) -> None:
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    if README_STATUS_START not in text or README_STATUS_END not in text:
        raise ValueError("README is missing early-action trigger status markers.")
    severity = "Not activated" if result.severity is None else result.severity.title()
    colour = {None: "lightgrey", "moderate": "yellow", "severe": "orange", "extreme": "red"}[result.severity]
    badge = f"![Trigger activation](https://img.shields.io/badge/S--EAP_trigger-{severity.replace(' ', '_')}-{colour})"
    value = lambda item: _area(item, result.area_km2)
    date_value = lambda item: "n/a" if item is None else item.date().isoformat()
    rows = [
        README_STATUS_START, badge, "",
        "| Latest trigger calculation | Value |", "|---|---|",
        f"| Highest trigger level | **{severity}** |",
        f"| Annual window | {date_value(result.window_start)} to {date_value(result.window_end)} |",
        f"| Forecast peak date | {date_value(result.prediction_date)} |",
        f"| Maximum forecast extent | {value(result.forecast_maximum_fraction)} |",
        f"| Annual-window minimum extent | {value(result.annual_minimum_fraction)} |",
        f"| Forecast increase | {value(result.increase_fraction)} |",
    ]
    for level in TRIGGER_LEVELS:
        rows.append(f"| {level.title()} threshold | {value(result.thresholds.get(level))} |")
    rows.extend([
        f"| Historic annual windows | {result.historic_window_count} |",
        f"| Trigger file | {'`' + str(result.output_path) + '`' if result.output_path else 'Not generated'} |",
        README_STATUS_END,
    ])
    prefix, remainder = text.split(README_STATUS_START, 1)
    _, suffix = remainder.split(README_STATUS_END, 1)
    path.write_text(prefix + "\n".join(rows) + suffix, encoding="utf-8")


def update_trigger_readme_status(
    result: EarlyActionTriggerResult, readme_path: str | Path = "README.md"
) -> None:
    _update_readme_status(result, readme_path)


def evaluate_early_action_trigger(
    *,
    history_dates: Iterable[object], history_values: Iterable[float],
    forecast_dates: Iterable[object], forecast_values: Iterable[float],
    output_dir: str | Path,
    percentiles: Mapping[str, float] | None = None,
    alert_date: date | None = None,
    output_filename: str | None = None,
    readme_path: str | Path | None = None,
    area_km2: float | None = None,
) -> EarlyActionTriggerResult:
    summary = evaluate_trigger_series(
        history_dates=history_dates, history_values=history_values,
        forecast_dates=forecast_dates, forecast_values=forecast_values,
        percentiles=percentiles,
    )
    output_path = Path(output_dir) / str(output_filename or get_cfg(
        "early_action_trigger.output_filename", "TRIGGER ACTIVATED.txt"
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if area_km2 is None:
        from model.temporal.runtime.area_reconversion import target_area_scale_km2
        target_column = str(get_cfg(
            "early_action_trigger.target_column", "percent_inundation_ssd"
        ))
        area_km2 = target_area_scale_km2(target_column)
    area_km2 = float(area_km2)
    severity = summary["trigger_level"]
    message = None
    published_path = None
    if severity is not None:
        message = (
            f"{severity.title()} alert triggered on {(alert_date or date.today()).isoformat()}. "
            f"The maximum six-dekad forecast flood extent is {_area(summary['forecast_maximum'], area_km2)} "
            f"on {summary['forecast_peak_date'].date().isoformat()}, an increase of "
            f"{_area(summary['increase'], area_km2)} from the annual-window minimum of "
            f"{_area(summary['annual_minimum'], area_km2)}. This exceeds the historic "
            f"{severity} threshold of {_area(summary['thresholds'][severity], area_km2)}."
        )
        output_path.write_text(message + "\n", encoding="utf-8")
        published_path = output_path
    else:
        output_path.unlink(missing_ok=True)

    result = EarlyActionTriggerResult(
        activated=severity is not None, severity=severity,
        annual_minimum_fraction=summary["annual_minimum"],
        forecast_maximum_fraction=summary["forecast_maximum"],
        increase_fraction=summary["increase"], prediction_date=summary["forecast_peak_date"],
        window_start=summary["window_start"], window_end=summary["window_end"],
        thresholds=summary["thresholds"], historic_window_count=summary["historic_window_count"],
        area_km2=area_km2,
        output_path=published_path, message=message,
    )
    if readme_path is not None:
        _update_readme_status(result, readme_path)
    return result


def evaluate_trigger_from_prediction_csv(
    prediction_csv: str | Path, **kwargs,
) -> EarlyActionTriggerResult:
    prediction_csv = Path(prediction_csv)
    frame = pd.read_csv(prediction_csv)
    required = {"date", "actual_fraction", "predicted_fraction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing trigger columns: {sorted(missing)}")
    history = frame[frame["actual_fraction"].notna()]
    forecast = frame[frame["predicted_fraction"].notna()]
    return evaluate_early_action_trigger(
        history_dates=history["date"], history_values=history["actual_fraction"],
        forecast_dates=forecast["date"], forecast_values=forecast["predicted_fraction"],
        output_dir=prediction_csv.parent, **kwargs,
    )
