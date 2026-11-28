from pathlib import Path
from PIL import Image, ImageColor

from readme_status import (
    ABLATION_END,
    ABLATION_START,
    COUNTY_PREDICTIONS_END,
    COUNTY_PREDICTIONS_START,
    DEPLOYMENT_END,
    DEPLOYMENT_START,
    SPATIAL_VALIDATION_END,
    SPATIAL_VALIDATION_START,
    generate_spatial_prediction_gif,
    update_ablation_readme,
    update_county_predictions_readme,
    update_deployment_readme,
    update_spatial_validation_readme,
)


def test_spatial_prediction_gif_uses_ordered_lead_maps(tmp_path):
    for lead, colour in ((2, "blue"), (1, "red")):
        folder = tmp_path / "spatial_predictions" / f"lead_{lead:02d}_2026-01-{lead:02d}"
        folder.mkdir(parents=True)
        Image.new("RGB", (20, 10), colour).save(folder / "flood_change_map.png")

    output = generate_spatial_prediction_gif(tmp_path, duration_ms=50)

    assert output == tmp_path / "spatial_predictions" / "spatial_predictions_leads_1_to_6.gif"
    assert output.exists()
    with Image.open(output) as animation:
        assert animation.n_frames == 2


def test_spatial_prediction_gif_preserves_distinct_flood_category_colours(tmp_path):
    category_colours = (
        "#ffffff", "#d3d3d3", "#f4a261", "#e76f51",
        "#9b2226", "#a8dadc", "#457b9d", "#1d3557",
    )
    folder = tmp_path / "spatial_predictions" / "lead_01_2026-01-01"
    folder.mkdir(parents=True)
    image = Image.new("RGB", (80, 10))
    for index, colour in enumerate(category_colours):
        image.paste(ImageColor.getrgb(colour), (index * 10, 0, (index + 1) * 10, 10))
    image.save(folder / "flood_change_map.png")

    output = generate_spatial_prediction_gif(tmp_path)

    with Image.open(output) as animation:
        encoded_colours = set(animation.convert("RGB").getdata())
    expected_colours = {ImageColor.getrgb(colour) for colour in category_colours}
    assert expected_colours.issubset(encoded_colours)


def _readme(tmp_path: Path) -> Path:
    path = tmp_path / "README.md"
    path.write_text(
        f"before\n{ABLATION_START}\nold\n{ABLATION_END}\n"
        f"middle\n{DEPLOYMENT_START}\nold\n{DEPLOYMENT_END}\nafter\n"
        f"spatial\n{SPATIAL_VALIDATION_START}\nold\n{SPATIAL_VALIDATION_END}\nafter\n"
        f"county\n{COUNTY_PREDICTIONS_START}\nold\n{COUNTY_PREDICTIONS_END}\nafter\n",
        encoding="utf-8",
    )
    return path


def test_ablation_completion_updates_model_and_baseline_table(tmp_path):
    path = _readme(tmp_path)
    update_ablation_readme(
        {
            "model_type": "elastic_net", "target_type": "seasonally_differenced",
            "feature_block_name": "no_lags", "run_id": "run-1", "timestamp_utc": "now",
            "avg_mae": 0.1, "avg_rmse": 0.2, "avg_peak_f1": 0.8, "avg_peak_auc": 0.9,
            "persistence_avg_mae": 0.3, "persistence_avg_rmse": 0.4,
            "persistence_avg_peak_f1": 0.5, "climatology_avg_mae": 0.6,
            "climatology_avg_rmse": 0.7, "climatology_avg_peak_f1": 0.1,
        },
        total=12, successful=10, failed=2, readme_path=path,
    )
    text = path.read_text()
    assert "run-1" in text
    assert "12 total; 10 successful; 2 failed" in text
    assert "Persistence (`t=0`) | 0.3000" in text
    assert "rolling minimum observed during the preceding 365 days" in text
    assert "10% of training-set relative increases" in text
    assert "relative increases" in text
    assert "Held-out observations are" in text
    assert "not used to set this threshold" in text


def test_deployment_completion_links_latest_prediction_plot(tmp_path):
    path = _readme(tmp_path)
    update_deployment_readme(
        model_row={"run_id": "run-2", "model_type": "elastic_net", "runtime_retrained_at": "now"},
        target_column="percent_inundation_ssd", origin_date="2026-01-01",
        future_dates=["2026-01-11", "2026-01-21"], output_dir="predictions/latest",
        graph_paths=["predictions/latest/temporal_predictions/prediction_year_by_year_comparison.png"],
        spatial_output_count=6, readme_path=path,
    )
    text = path.read_text()
    assert "2026-01-11 to 2026-01-21" in text
    assert "![Latest temporal forecast](predictions/latest/temporal_predictions/prediction_year_by_year_comparison.png)" in text
    assert "Spatial outputs | 6" in text


def test_county_predictions_publish_map_and_active_warning_list(tmp_path):
    path = _readme(tmp_path)
    update_county_predictions_readme(
        trigger_map="predictions/run/county-level predictions/map.png",
        active_counties=["Juba (Central Equatoria)", "Bor South (Jonglei)"],
        county_trigger_csv="predictions/run/county-level predictions/counties.csv",
        state_trigger_csv="predictions/run/county-level predictions/states.csv",
        readme_path=path,
    )
    text = path.read_text()
    assert "![Latest county early-action trigger map](<predictions/run/county-level predictions/map.png>)" in text
    assert "- Juba (Central Equatoria)" in text
    assert "- Bor South (Jonglei)" in text
    assert "[state CSV](<predictions/run/county-level predictions/states.csv>)" in text


def test_spatial_validation_reports_four_requested_metrics(tmp_path):
    path = _readme(tmp_path)
    update_spatial_validation_readme(
        [{
            "model": "convlstm", "lead_period": "overall",
            "transition": "dry_to_flood", "f1": 0.4, "accuracy": 0.9,
            "precision": 0.3, "recall": 0.6,
        }],
        forecast_horizon_periods=6,
        csv_path="model/spatial/model.change_detection.validation.csv",
        json_path="model/spatial/model.validation.json",
        readme_path=path,
    )
    text = path.read_text()
    assert text.count("| Method | F1 | Accuracy | Precision | Recall |") == 4
    assert "#### Non-inundated → inundated" in text
    assert "| ConvLSTM | 0.4000 | 0.9000 | 0.3000 | 0.6000" in text
    assert "6 forecast periods" in text
