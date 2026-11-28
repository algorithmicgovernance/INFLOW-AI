"""Generated README status blocks for completed model and deployment runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


ABLATION_START = "<!-- ABLATION_STATUS_START -->"
ABLATION_END = "<!-- ABLATION_STATUS_END -->"
DEPLOYMENT_START = "<!-- DEPLOYMENT_STATUS_START -->"
DEPLOYMENT_END = "<!-- DEPLOYMENT_STATUS_END -->"
SPATIAL_VALIDATION_START = "<!-- SPATIAL_VALIDATION_STATUS_START -->"
SPATIAL_VALIDATION_END = "<!-- SPATIAL_VALIDATION_STATUS_END -->"
COUNTY_PREDICTIONS_START = "<!-- COUNTY_PREDICTIONS_STATUS_START -->"
COUNTY_PREDICTIONS_END = "<!-- COUNTY_PREDICTIONS_STATUS_END -->"


def _replace_block(path: Path, start: str, end: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise ValueError(f"README is missing generated status markers {start} / {end}.")
    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    path.write_text(prefix + start + "\n" + body.strip() + "\n" + end + suffix, encoding="utf-8")


def _number(row: dict[str, Any], key: str, digits: int = 4) -> str:
    try:
        return f"{float(row[key]):.{digits}f}"
    except (KeyError, TypeError, ValueError):
        return "n/a"


def generate_spatial_prediction_gif(
    output_dir: str | Path, *, duration_ms: int = 1200, maximum_width: int = 1000,
) -> Path | None:
    """Build a compact animated lead-time summary from spatial map outputs."""
    from PIL import Image, ImageColor

    output = Path(output_dir)
    frames = sorted((output / "spatial_predictions").glob("lead_*/flood_change_map.png"))
    if not frames:
        return None
    images = []
    gif_frames = []
    shared_palette = None
    try:
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            if image.width > maximum_width:
                height = round(image.height * maximum_width / image.width)
                image = image.resize((maximum_width, height), Image.Resampling.LANCZOS)
            images.append(image)
        # GIF is an indexed-colour format. Letting Pillow choose a separate
        # adaptive palette for every frame makes identical flood categories
        # appear to change colour between leads. The first spatial map contains
        # the complete categorical legend, so use its palette for every frame.
        shared_palette = images[0].quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        # Adaptive quantization can merge the two orange decrease categories
        # because their colours are relatively close. Reserve exact entries for
        # every categorical map colour before applying the shared palette.
        flood_category_colors = (
            "#ffffff", "#d3d3d3",
            "#f4a261", "#e76f51", "#9b2226",
            "#a8dadc", "#457b9d", "#1d3557",
        )
        palette = shared_palette.getpalette()
        for index, color in enumerate(flood_category_colors):
            palette[index * 3:(index + 1) * 3] = ImageColor.getrgb(color)
        shared_palette.putpalette(palette)
        gif_frames = [
            image.quantize(palette=shared_palette, dither=Image.Dither.NONE)
            for image in images
        ]
        gif_path = output / "spatial_predictions" / "spatial_predictions_leads_1_to_6.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return gif_path
    finally:
        for frame in gif_frames:
            frame.close()
        if shared_palette is not None:
            shared_palette.close()
        for image in images:
            image.close()


def update_ablation_readme(
    row: dict[str, Any], *, total: int, successful: int, failed: int,
    readme_path: str | Path = "README.md",
) -> None:
    model = str(row.get("model_type", "unknown")).replace("_", " ").title()
    try:
        peak_fraction = float(row.get("extreme_event_top_fraction", 0.10))
        peak_percent = f"{peak_fraction * 100:g}%"
    except (TypeError, ValueError):
        peak_percent = "configured upper-tail fraction"
    body = f"""
The latest completed ablation published **{model}** with a
`{row.get('target_type', 'unknown')}` target and `{row.get('feature_block_name', 'unknown')}`
feature block (run `{row.get('run_id', 'unknown')}`).

Here, a **peak event** is an inundation value whose relative increase above the
rolling minimum observed during the preceding 365 days reaches the
lead-specific threshold. Relative increase is `(value - rolling minimum) /
max(abs(rolling minimum), 1e-6)`, and the threshold is learned from the upper
{peak_percent} of training-set relative increases. Held-out observations are
not used to set this threshold.

| Latest ablation | Value |
|---|---|
| Completed at (UTC) | {row.get('timestamp_utc', 'unknown')} |
| Experiments | {total} total; {successful} successful; {failed} failed |
| Seasonal feature standardization | {bool(row.get('seasonally_standardize_features', False))} |
| Mean peak-event F1 | {_number(row, 'avg_peak_f1')} |
| Mean peak-event ROC AUC | {_number(row, 'avg_peak_auc')} |

| Held-out forecast | Mean MAE | Mean RMSE | Mean peak-event F1 |
|---|---:|---:|---:|
| Selected {model} | {_number(row, 'avg_mae')} | {_number(row, 'avg_rmse')} | {_number(row, 'avg_peak_f1')} |
| Persistence (`t=0`) | {_number(row, 'persistence_avg_mae')} | {_number(row, 'persistence_avg_rmse')} | {_number(row, 'persistence_avg_peak_f1')} |
| Seasonal climatology | {_number(row, 'climatology_avg_mae')} | {_number(row, 'climatology_avg_rmse')} | {_number(row, 'climatology_avg_peak_f1')} |
"""
    _replace_block(Path(readme_path), ABLATION_START, ABLATION_END, body)


def update_deployment_readme(
    *, model_row: dict[str, Any], target_column: str, origin_date: Any,
    future_dates: Iterable[Any], output_dir: str | Path, graph_paths: Iterable[str | Path],
    spatial_output_count: int, readme_path: str | Path = "README.md",
) -> None:
    dates = list(future_dates)
    graphs = [Path(value) for value in graph_paths]
    output = Path(output_dir)
    recent_plot = next(
        (path for path in graphs if path.name == "prediction_year_by_year_comparison.png"),
        graphs[0] if graphs else None,
    )
    spatial_gif = generate_spatial_prediction_gif(output)
    plot_markdown = "Not generated" if recent_plot is None else f"[{recent_plot.name}]({recent_plot.as_posix()})"
    plot_image = "" if recent_plot is None else f"\n![Latest temporal forecast]({recent_plot.as_posix()})"
    gif_markdown = "Not generated" if spatial_gif is None else f"[{spatial_gif.name}]({spatial_gif.as_posix()})"
    gif_image = "" if spatial_gif is None else f"\n![Spatial forecast by lead time]({spatial_gif.as_posix()})"
    body = f"""
| Latest deployed prediction | Value |
|---|---|
| Completed/retrained at (UTC) | {model_row.get('runtime_retrained_at', 'unknown')} |
| Selected experiment | `{model_row.get('run_id', 'unknown')}` ({model_row.get('model_type', 'unknown')}) |
| Target | `{target_column}` |
| Forecast origin | {origin_date} |
| Forecast range | {dates[0] if dates else 'n/a'} to {dates[-1] if dates else 'n/a'} |
| Prediction directory | [`{output.as_posix()}`]({output.as_posix()}) |
| Recent prediction plot | {plot_markdown} |
| Spatial lead-time animation | {gif_markdown} |
| Spatial outputs | {spatial_output_count} |
{plot_image}
{gif_image}
"""
    _replace_block(Path(readme_path), DEPLOYMENT_START, DEPLOYMENT_END, body)


def update_county_predictions_readme(
    *, trigger_map: str | Path, active_counties: Iterable[str],
    county_trigger_csv: str | Path, state_trigger_csv: str | Path,
    readme_path: str | Path = "README.md",
) -> None:
    """Publish the latest county trigger map and active-warning list."""
    map_path = Path(trigger_map).as_posix()
    county_csv = Path(county_trigger_csv).as_posix()
    state_csv = Path(state_trigger_csv).as_posix()
    active = sorted(str(value) for value in active_counties)
    warnings = "None." if not active else "\n".join(f"- {value}" for value in active)
    body = f"""
![Latest county early-action trigger map](<{map_path}>)

**Counties with active early warnings:**

{warnings}

Detailed trigger results: [county CSV](<{county_csv}>) and
[state CSV](<{state_csv}>).
"""
    _replace_block(
        Path(readme_path), COUNTY_PREDICTIONS_START, COUNTY_PREDICTIONS_END, body
    )


def update_spatial_validation_readme(
    rows: Iterable[dict[str, Any]], *, forecast_horizon_periods: int,
    csv_path: str | Path, json_path: str | Path,
    readme_path: str | Path = "README.md",
) -> None:
    """Publish overall four-state spatial change metrics to the README."""
    overall = [row for row in rows if str(row.get("lead_period")) == "overall"]
    labels = {
        "stable_dry": "Non-inundated → non-inundated",
        "dry_to_flood": "Non-inundated → inundated",
        "flood_to_dry": "Inundated → non-inundated",
        "stable_flood": "Inundated → inundated",
    }
    models = {
        "convlstm": "ConvLSTM",
        "persistence": "Persistence",
        "glofas_20_year": "GloFAS 20-year",
        "glofas_100_year": "GloFAS 100-year",
    }
    sections = []
    for transition, label in labels.items():
        table_rows = [
            f"#### {label}",
            "",
            "| Method | F1 | Accuracy | Precision | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in overall:
            if str(row.get("transition")) != transition:
                continue
            table_rows.append(
                f"| {models.get(str(row.get('model')), str(row.get('model')))} "
                f"| {_number(row, 'f1')} | {_number(row, 'accuracy')} "
                f"| {_number(row, 'precision')} | {_number(row, 'recall')} |"
            )
        sections.append("\n".join(table_rows))
    csv_value = Path(csv_path).as_posix()
    json_value = Path(json_path).as_posix()
    body = f"""
Latest held-out spatial change-detection comparison across
{forecast_horizon_periods} forecast periods. Metrics are calculated separately
for each origin-to-target state so stable pixels do not obscure onset and
recession performance.

{chr(10).join(chr(10) + section for section in sections).strip()}

Detailed results: [`{csv_value}`]({csv_value}) and
[`{json_value}`]({json_value}).
"""
    _replace_block(
        Path(readme_path), SPATIAL_VALIDATION_START, SPATIAL_VALIDATION_END, body
    )
