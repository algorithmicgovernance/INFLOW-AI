from __future__ import annotations

import argparse
import logging

from .temporal_ablation_runner import (
    build_grid,
    run_temporal_ablation_grid,
    run_staged_temporal_ablation,
)
from processing.config import get_cfg
from model.data_preflight import audit_temporal_data


def _bool_values(csv: str) -> list[bool]:
    out = []
    for value in csv.split(","):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            out.append(True)
        elif normalized in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token '{value}' in --autoregressive-values")
    return out


def _run_data_preflight(configs) -> None:
    if not bool(get_cfg("data_preflight.enabled", True)):
        return
    configs = list(configs)
    for product in sorted({config.inundation_product for config in configs}):
        product_configs = [config for config in configs if config.inundation_product == product]
        include_ecmwf = any(config.include_ecmwf_hindcasts is not False for config in product_configs)
        audit_temporal_data(
            product,
            mode="ablation",
            include_ecmwf=include_ecmwf,
            fail_on_error=True,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run temporal inundation ablation studies.")
    parser.add_argument(
        "--models",
        default="random_forest,gradient_boosting,elastic_net,linear_regression,transformer",
        help="Comma-separated model types.",
    )
    parser.add_argument(
        "--cutoff-dates",
        default="2024-12-31",
        help="Comma-separated train/test cutoff dates (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--autoregressive-values",
        default="yes,no",
        help="Comma-separated values from {yes,no,true,false,1,0}.",
    )
    parser.add_argument(
        "--target-types",
        default="raw,first_differenced,deseasonalised,seasonally_differenced,differenced_anomaly",
        help="Comma-separated target transforms.",
    )
    parser.add_argument(
        "--inundation-products",
        default="viirs,modis",
        help="Comma-separated products (viirs/modis).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-path",
        default="model/ablation/ablation_experiment_log.csv",
        help="CSV file where all experiment metadata and metrics will be appended.",
    )

    args = parser.parse_args()

    if get_cfg("ablation.enabled", False):
        cfg_models = get_cfg("ablation.experiments.models", [])
        cfg_cutoffs = get_cfg("ablation.experiments.training_cutoff_dates", [])
        cfg_ar = get_cfg("ablation.experiments.autoregressive_values", [])
        cfg_targets = get_cfg("ablation.experiments.target_types", [])
        cfg_products = get_cfg("ablation.experiments.inundation_products", [])
        cfg_seed = get_cfg("ablation.experiments.seed", args.seed)
        cfg_log = get_cfg("ablation.experiments.log_path", args.log_path)

        if cfg_models and cfg_cutoffs and cfg_ar and cfg_targets and cfg_products:
            staged_enabled = bool(get_cfg("ablation.staged.enabled", True))
            fixed = get_cfg("ablation.staged.fixed_preprocessing", {})
            fixed = {
                "include_ecmwf_hindcasts": bool(fixed.get("include_ecmwf_hindcasts", True)),
                "include_target_lag_features": bool(fixed.get("include_target_lag_features", True)),
                "include_feature_lag_features": bool(fixed.get("include_feature_lag_features", True)),
            }
            configs = build_grid(
                model_types=[str(m).strip() for m in cfg_models],
                cutoff_dates=[str(d).strip() for d in cfg_cutoffs],
                autoregressive_values=[bool(v) for v in cfg_ar],
                target_types=[str(t).strip() for t in cfg_targets],
                inundation_products=[str(p).strip().lower() for p in cfg_products],
                seed=int(cfg_seed),
                include_ecmwf_hindcasts=fixed["include_ecmwf_hindcasts"],
                include_target_lag_features=fixed["include_target_lag_features"],
                include_feature_lag_features=fixed["include_feature_lag_features"],
            )
            _run_data_preflight(configs)
            if staged_enabled:
                feature_blocks = get_cfg("ablation.staged.feature_block_ablations", [])
                results = run_staged_temporal_ablation(
                    configs,
                    feature_blocks=list(feature_blocks),
                    fixed_preprocessing=fixed,
                    top_n=int(get_cfg("ablation.staged.top_n_model_configurations", 3)),
                    ranking_metric=str(get_cfg("ablation.staged.ranking_metric", "avg_peak_f1")),
                    log_csv_path=str(cfg_log),
                )
            else:
                results = run_temporal_ablation_grid(configs=configs, log_csv_path=str(cfg_log))
            success_count = sum(1 for r in results if r.get("status") == "success")
            fail_count = len(results) - success_count

            print(f"Ablation run complete (config). total={len(results)} success={success_count} failed={fail_count}")
            print(f"Log file: {cfg_log}")
            return

    configs = build_grid(
        model_types=[m.strip() for m in args.models.split(",") if m.strip()],
        cutoff_dates=[d.strip() for d in args.cutoff_dates.split(",") if d.strip()],
        autoregressive_values=_bool_values(args.autoregressive_values),
        target_types=[t.strip() for t in args.target_types.split(",") if t.strip()],
        inundation_products=[p.strip().lower() for p in args.inundation_products.split(",") if p.strip()],
        seed=args.seed,
    )
    _run_data_preflight(configs)

    results = run_temporal_ablation_grid(configs=configs, log_csv_path=args.log_path)
    success_count = sum(1 for r in results if r.get("status") == "success")
    fail_count = len(results) - success_count

    print(f"Ablation run complete. total={len(results)} success={success_count} failed={fail_count}")
    print(f"Log file: {args.log_path}")


if __name__ == "__main__":
    main()
