"""County-level VIIRS histories, spatial forecasts, triggers, plots, and map."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

from processing import cleaning_utils
from processing.config import get_cfg
from processing.data_cleaning import process_inundation_viirs
from early_action_trigger import evaluate_trigger_series


@dataclass(frozen=True)
class CountyPredictionArtifacts:
    output_dir: Path
    historic_csv: Path
    forecast_csv: Path
    county_trigger_csv: Path
    state_trigger_csv: Path
    trigger_map: Path
    active_counties: tuple[str, ...]
    plot_paths: tuple[Path, ...]


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _pixel_area_km2() -> float:
    transform = cleaning_utils.MASK_REGIONS_REF_TRANSFORM
    return abs(transform[0] * transform[4] - transform[1] * transform[3]) / 1_000_000.0


def _observation_dates(frame: pd.DataFrame) -> pd.Series:
    if "period_start" in frame:
        return pd.to_datetime(frame["period_start"], errors="coerce")
    if "date" in frame:
        return pd.to_datetime(frame["date"], errors="coerce")
    raise ValueError("County VIIRS history requires a period_start or date column.")


def _trigger_summary(
    history_dates: pd.Series,
    history_fraction: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    forecast_fraction: np.ndarray,
) -> dict[str, object]:
    evaluation = evaluate_trigger_series(
        history_dates=history_dates,
        history_values=history_fraction,
        forecast_dates=forecast_dates,
        forecast_values=forecast_fraction,
        percentiles=None,
        no_history_minimum_forecast_fraction=float(get_cfg(
            "early_action_trigger.no_history_minimum_forecast_fraction", 0.05
        )),
    )
    thresholds = evaluation["thresholds"]
    increase = evaluation["increase"]
    level = evaluation["trigger_level"]
    return {
        "trigger_active": evaluation["trigger_active"],
        "trigger_level": level,
        "moderate_trigger": level in {"moderate", "severe", "extreme"},
        "severe_trigger": level in {"severe", "extreme"},
        "extreme_trigger": level == "extreme",
        "forecast_peak_date": (
            None if evaluation["forecast_peak_date"] is None
            else evaluation["forecast_peak_date"].date().isoformat()
        ),
        "annual_window_start": (
            None if evaluation["window_start"] is None
            else evaluation["window_start"].date().isoformat()
        ),
        "annual_window_end": (
            None if evaluation["window_end"] is None
            else evaluation["window_end"].date().isoformat()
        ),
        "forecast_maximum_fraction": evaluation["forecast_maximum"],
        "annual_minimum_fraction": evaluation["annual_minimum"],
        "forecast_increase_fraction": increase,
        "moderate_threshold_fraction": thresholds.get("moderate", np.nan),
        "severe_threshold_fraction": thresholds.get("severe", np.nan),
        "extreme_threshold_fraction": thresholds.get("extreme", np.nan),
        "historic_annual_window_count": evaluation["historic_window_count"],
        "historic_signal_available": evaluation["historic_signal_available"],
        "no_history_forecast_gate_passed": evaluation["no_history_forecast_gate_passed"],
    }


def _plot_year_by_year(
    history_dates: pd.Series,
    history_km2: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    forecast_km2: np.ndarray,
    county_name: str,
    output_path: Path,
) -> None:
    combined = pd.DataFrame({
        "date": list(history_dates) + list(forecast_dates),
        "km2": list(history_km2) + list(forecast_km2),
        "predicted": [False] * len(history_dates) + [True] * len(forecast_dates),
    }).dropna(subset=["date", "km2"])
    combined["year"] = combined["date"].dt.year
    combined["plot_day"] = combined["date"].dt.dayofyear
    leap = combined["date"].dt.is_leap_year & (combined["plot_day"] > 59)
    combined.loc[leap, "plot_day"] -= 1

    fig, ax = plt.subplots(figsize=(12, 8))
    years = sorted(combined["year"].unique())
    forecast_years = set(forecast_dates.year)
    cmap = plt.get_cmap("winter")
    year_labels = []
    show_year_labels = bool((combined["km2"].abs() > 1e-12).any())
    for index, year in enumerate(years):
        group = combined[combined["year"] == year]
        actual = group[~group["predicted"]]
        predicted = group[group["predicted"]]
        if year in forecast_years:
            if not actual.empty:
                ax.plot(actual["plot_day"], actual["km2"], color="#d62728", linewidth=2,
                        label="Actual" if index == 0 or "Actual" not in ax.get_legend_handles_labels()[1] else "_nolegend_")
            if not predicted.empty:
                if not actual.empty:
                    ax.plot(
                        [actual.iloc[-1]["plot_day"], predicted.iloc[0]["plot_day"]],
                        [actual.iloc[-1]["km2"], predicted.iloc[0]["km2"]],
                        color="#d62728", linestyle="--", linewidth=2,
                    )
                ax.plot(predicted["plot_day"], predicted["km2"], color="#d62728",
                        linestyle="--", linewidth=2, label="Spatial forecast", zorder=5)
            label_color = "#d62728"
        else:
            color = cmap(index / max(len(years) - 1, 1))
            ax.plot(group["plot_day"], group["km2"], color=color, alpha=0.7, zorder=2)
            label_color = color
        if not group.empty and show_year_labels:
            year_labels.append({
                "year": int(year),
                "x": float(group.iloc[-1]["plot_day"]),
                "y": float(group.iloc[-1]["km2"]),
                "color": label_color,
            })
    month_starts = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
    ax.set_title(f"{county_name} Inundated Area, Year-by-Year Comparison")
    ax.set_xlabel("Month of Year")
    ax.set_ylabel("Inundated Area (km²)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1.0)
    label_gap = y_span * 0.035
    ordered_labels = sorted(year_labels, key=lambda item: item["y"])
    label_positions = []
    for item in ordered_labels:
        label_y = max(item["y"], label_positions[-1] + label_gap) if label_positions else item["y"]
        label_positions.append(label_y)
    if label_positions and label_positions[-1] > y_max:
        shift = label_positions[-1] - y_max
        label_positions = [position - shift for position in label_positions]
    if label_positions and label_positions[0] < y_min:
        shift = y_min - label_positions[0]
        label_positions = [position + shift for position in label_positions]

    current_year = int(forecast_dates.max().year)
    for item, label_y in zip(ordered_labels, label_positions):
        is_current = item["year"] == current_year
        ax.annotate(
            str(item["year"]),
            xy=(item["x"], item["y"]),
            xytext=(377, label_y),
            color=item["color"],
            fontsize=10 if is_current else 8,
            fontweight="bold" if is_current else "normal",
            ha="left", va="center",
            zorder=10 if is_current else 6,
            arrowprops={
                "arrowstyle": "-", "color": item["color"],
                "alpha": 0.8 if is_current else 0.5,
                "linewidth": 1.2 if is_current else 0.7,
            },
            annotation_clip=False,
        )
    ax.set_xlim(1, 405)
    ax.set_xticks(month_starts.dayofyear)
    ax.set_xticklabels(month_starts.strftime("%b"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(get_cfg("outputs.graph_dpi", 220)), bbox_inches="tight")
    plt.close(fig)


def run_county_level_predictions(
    *,
    inundation_h5_path: str | Path,
    temporal_history_path: str | Path,
    forecast_dates,
    forecast_masks: list[np.ndarray],
    prediction_root: str | Path,
    counties_path: str | Path | None = None,
) -> CountyPredictionArtifacts:
    """Create all county operational products from thresholded spatial masks."""
    output_dir = Path(prediction_root) / get_cfg(
        "county_predictions.output_subdirectory", "county-level predictions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    historic_csv, counties, county_labels = (
        process_inundation_viirs.build_county_viirs_temporal_csv(
            inundation_h5_path,
            temporal_history_path,
            counties_path=counties_path,
        )
    )
    history = pd.read_csv(historic_csv)
    history_dates = _observation_dates(history)
    forecast_dates = pd.DatetimeIndex(pd.to_datetime(forecast_dates))
    if len(forecast_masks) != len(forecast_dates):
        raise ValueError("County forecasts require one thresholded spatial mask per date.")
    pixel_area = _pixel_area_km2()
    forecasts = []
    triggers = []
    plots = []

    for county_index, county in enumerate(counties.itertuples(index=False), start=1):
        county_mask = county_labels == county_index
        code = str(county.ADM2_PCODE)
        column = f"percent_inundation_{code.lower()}"
        cells = int(county_mask.sum())
        fractions = np.asarray(
            [np.count_nonzero(mask[county_mask]) / cells for mask in forecast_masks],
            dtype=np.float64,
        )
        counts = np.rint(fractions * cells).astype(int)
        for lead, (date, count, fraction) in enumerate(
            zip(forecast_dates, counts, fractions), start=1
        ):
            forecasts.append({
                "county_code": code, "county": county.ADM2_EN,
                "state_code": county.ADM1_PCODE, "state": county.ADM1_EN,
                "lead_period": lead, "date": date.date().isoformat(),
                "inundated_pixel_count": int(count), "county_pixel_count": cells,
                "predicted_fraction": float(fraction),
                "predicted_km2": float(count * pixel_area),
            })
        summary = _trigger_summary(
            history_dates, history[column].to_numpy(dtype=np.float64),
            forecast_dates, fractions,
        )
        triggers.append({
            "county_code": code, "county": county.ADM2_EN,
            "state_code": county.ADM1_PCODE, "state": county.ADM1_EN,
            **summary,
            "county_area_km2": float(cells * pixel_area),
            "forecast_maximum_km2": float(summary["forecast_maximum_fraction"] * cells * pixel_area),
            "annual_minimum_km2": float(
                summary["annual_minimum_fraction"] * cells * pixel_area
            ),
            "forecast_increase_km2": float(
                summary["forecast_increase_fraction"] * cells * pixel_area
            ),
            "moderate_threshold_km2": float(summary["moderate_threshold_fraction"] * cells * pixel_area),
            "severe_threshold_km2": float(summary["severe_threshold_fraction"] * cells * pixel_area),
            "extreme_threshold_km2": float(summary["extreme_threshold_fraction"] * cells * pixel_area),
        })
        county_dir = output_dir / f"{code.lower()}_{_safe_name(county.ADM2_EN)}"
        county_dir.mkdir(parents=True, exist_ok=True)
        series = pd.DataFrame({
            "date": list(history_dates) + list(forecast_dates),
            "actual_fraction": list(history[column].to_numpy(dtype=np.float64)) + [np.nan] * len(forecast_dates),
            "predicted_fraction": [np.nan] * len(history) + list(fractions),
        })
        series["actual_inundated_km2"] = series["actual_fraction"] * cells * pixel_area
        series["predicted_inundated_km2"] = series["predicted_fraction"] * cells * pixel_area
        series.to_csv(county_dir / "county_inundation_timeseries.csv", index=False)
        plot_path = county_dir / "prediction_year_by_year_comparison.png"
        _plot_year_by_year(
            history_dates,
            history[column].to_numpy(dtype=np.float64) * cells * pixel_area,
            forecast_dates,
            counts * pixel_area,
            str(county.ADM2_EN),
            plot_path,
        )
        plots.append(plot_path)

    forecast_csv = output_dir / "county_spatial_predictions.csv"
    county_trigger_csv = output_dir / "county_early_action_triggers.csv"
    pd.DataFrame(forecasts).to_csv(forecast_csv, index=False)
    trigger_frame = pd.DataFrame(triggers)
    trigger_frame.to_csv(county_trigger_csv, index=False)

    state_rows = []
    county_cells = {
        str(row.ADM2_PCODE): int(np.count_nonzero(county_labels == index))
        for index, row in enumerate(counties.itertuples(index=False), start=1)
    }
    for (state_code, state_name), members in counties.groupby(["ADM1_PCODE", "ADM1_EN"]):
        codes = members["ADM2_PCODE"].astype(str).tolist()
        weights = np.asarray([county_cells[code] for code in codes], dtype=np.float64)
        historic_matrix = np.column_stack([
            history[f"percent_inundation_{code.lower()}"].to_numpy(dtype=np.float64)
            for code in codes
        ])
        historic_fraction = np.average(historic_matrix, axis=1, weights=weights)
        state_forecasts = pd.DataFrame(forecasts)
        state_forecasts = state_forecasts[state_forecasts["state_code"] == state_code]
        predicted_fraction = (
            state_forecasts.groupby("lead_period")["inundated_pixel_count"].sum()
            .reindex(range(1, len(forecast_dates) + 1)).to_numpy(dtype=np.float64)
            / weights.sum()
        )
        state_rows.append({
            "state_code": state_code,
            "state": state_name,
            **_trigger_summary(
                history_dates, historic_fraction, forecast_dates,
                predicted_fraction,
            ),
        })
    state_trigger_csv = output_dir / "state_early_action_triggers.csv"
    pd.DataFrame(state_rows).to_csv(state_trigger_csv, index=False)

    map_frame = counties.to_crs("EPSG:4326").copy()
    severity_by_code = trigger_frame.set_index("county_code")["trigger_level"]
    map_frame["trigger_level"] = map_frame["ADM2_PCODE"].map(severity_by_code).fillna("inactive")
    severity_colours = {
        "inactive": "#d9d9d9", "moderate": "#f1c40f",
        "severe": "#f39c12", "extreme": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(10, 11))
    map_frame.plot(
        ax=ax,
        color=map_frame["trigger_level"].map(severity_colours),
        edgecolor="white", linewidth=0.35,
    )
    ax.set_title("County Early-Action Triggers from Historic Annual Flood-Range Percentiles")
    ax.axis("off")
    ax.legend(handles=[
        Patch(facecolor=severity_colours["extreme"], label="Extreme (>95th percentile)"),
        Patch(facecolor=severity_colours["severe"], label="Severe (>90th percentile)"),
        Patch(facecolor=severity_colours["moderate"], label="Moderate (>80th percentile)"),
        Patch(facecolor=severity_colours["inactive"], label="Inactive / not evaluable"),
    ], loc="lower left")
    trigger_map = output_dir / "county_early_action_trigger_map.png"
    fig.savefig(trigger_map, dpi=int(get_cfg("outputs.graph_dpi", 220)), bbox_inches="tight")
    plt.close(fig)

    active_counties = tuple(
        f"{row.county} ({row.state}) — {str(row.trigger_level).title()}"
        for row in trigger_frame[trigger_frame["trigger_active"]].itertuples(index=False)
    )
    return CountyPredictionArtifacts(
        output_dir, historic_csv, forecast_csv, county_trigger_csv,
        state_trigger_csv, trigger_map, active_counties, tuple(plots),
    )
