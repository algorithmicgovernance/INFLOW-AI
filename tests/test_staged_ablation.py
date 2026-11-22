from model.ablation.temporal_ablation_runner import build_temporal_feature_ablation_configs


def test_feature_ablation_skips_fixed_and_effective_duplicates_for_non_ar_model():
    top = [{
        "model_type": "linear_regression",
        "training_cutoff_date": "2024-12-31",
        "autoregressive": False,
        "target_type": "raw",
        "inundation_product": "viirs",
        "seed": 42,
    }]
    fixed = {
        "include_ecmwf_hindcasts": True,
        "include_target_lag_features": True,
        "include_feature_lag_features": True,
    }
    blocks = [
        {"name": "no_target_lags", "include_ecmwf_hindcasts": True,
         "include_target_lag_features": False, "include_feature_lag_features": True},
        {"name": "no_ecmwf", "include_ecmwf_hindcasts": False,
         "include_target_lag_features": True, "include_feature_lag_features": True},
    ]
    configs = build_temporal_feature_ablation_configs(top, blocks, fixed)
    assert [config.feature_block_name for config in configs] == ["no_ecmwf"]
    assert configs[0].ablation_stage == "feature_blocks"


def test_feature_ablation_keeps_target_lag_removal_for_ar_model():
    top = [{
        "model_type": "random_forest",
        "training_cutoff_date": "2024-12-31",
        "autoregressive": True,
        "target_type": "raw",
        "inundation_product": "viirs",
        "seed": 7,
    }]
    fixed = {
        "include_ecmwf_hindcasts": True,
        "include_target_lag_features": True,
        "include_feature_lag_features": True,
    }
    blocks = [{"name": "no_target_lags", "include_ecmwf_hindcasts": True,
               "include_target_lag_features": False, "include_feature_lag_features": True}]
    configs = build_temporal_feature_ablation_configs(top, blocks, fixed)
    assert len(configs) == 1
    assert configs[0].include_target_lag_features is False
