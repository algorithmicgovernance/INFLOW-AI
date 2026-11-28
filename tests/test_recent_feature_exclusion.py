import pandas as pd

import model.temporal.ablation.temporal_data_pipeline as pipeline


def test_temporal_feature_table_excludes_stale_data_products(tmp_path, monkeypatch):
    aligned = tmp_path / "viirs-aligned"
    aligned.mkdir()
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="MS")
    pd.DataFrame({
        "date": dates,
        "percent_inundation": range(len(dates)),
    }).to_csv(aligned / "inundation_viirs_temporal.csv", index=False)
    pd.DataFrame({
        "date": dates[:3],
        "stale_signal": [1.0, 2.0, 3.0],
    }).to_csv(aligned / "stale_product.csv", index=False)
    pd.DataFrame({
        "date": dates,
        "fresh_signal": range(len(dates)),
    }).to_csv(aligned / "fresh_product.csv", index=False)

    values = {
        "paths.historic.root": str(tmp_path),
        "runtime.operational_target_temporal_path": None,
        "ablation.pipeline.target_column": "percent_inundation",
        "ablation.pipeline.optional_feature_blocks.include_ecmwf_hindcasts": False,
    }
    monkeypatch.setattr(pipeline, "get_cfg", lambda key, default=None: values.get(key, default))

    table, _, source_files, _ = pipeline._build_temporal_feature_table(
        "viirs", include_ecmwf_hindcasts=False
    )

    assert "fresh_signal" in table.columns
    assert "stale_signal" not in table.columns
    assert "stale_product.csv" in source_files


def test_feature_availability_ranges_report_each_training_candidate():
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "complete": [1.0, 2.0, 3.0, 4.0],
        "partial": [None, 2.0, 3.0, None],
    })

    ranges = pipeline._feature_availability_ranges(frame)

    assert ranges["complete"] == {
        "start": "2024-01-01",
        "end": "2024-04-01",
        "available_rows": 4,
        "total_rows": 4,
    }
    assert ranges["partial"]["start"] == "2024-02-01"
    assert ranges["partial"]["end"] == "2024-03-01"
    assert ranges["partial"]["available_rows"] == 2
