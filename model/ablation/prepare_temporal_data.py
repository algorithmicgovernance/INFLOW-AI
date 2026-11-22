"""Download and process all datasets required for temporal ablation experiments.

Run from the project root:

    python -m model.ablation.prepare_temporal_data

The target product is read from ``runtime.target_product`` in ``config.yaml``.
This command does not require an existing trained model or ablation result.
"""
from __future__ import annotations

import logging

from processing import cleaning_utils
from processing.data_cleaning import (
    process_albert,
    process_gridded_moisture,
    process_gridded_rainfall,
    process_gridded_rainfall_cumulative,
    process_inundation_modis,
    process_inundation_viirs,
    process_kyoga,
    process_rainfall,
    process_teleconnections,
    process_victoria,
)
from processing.data_cleaning.process_ecmwf import process_ecmwf_hindcasts

LOG = logging.getLogger(__name__)


def prepare_temporal_data(target_product: str | None = None) -> None:
    """Prepare temporal inundation observations and predictors for one target product."""
    product = cleaning_utils.resolve_target_product(target_product)
    LOG.info("Preparing all %s-aligned ablation data", product.upper())

    if product == "viirs":
        # Ablation preparation may use a complete, internally consistent local
        # archive when JASMIN is temporarily unavailable. Operational updates
        # remain strict and do not enable this fallback.
        process_inundation_viirs.update_inundation(allow_offline_existing=True)
    else:
        process_inundation_modis.update_inundation()

    process_victoria.update_victoria(product)
    process_albert.update_albert(product)
    process_kyoga.update_kyoga(product)
    process_rainfall.update_rainfall(product)
    process_teleconnections.update_teleconnections(product)
    process_gridded_rainfall.update_gridded_rainfall()
    process_gridded_rainfall_cumulative.update_gridded_rainfall_cumulative()
    process_gridded_moisture.update_gridded_moisture()
    process_ecmwf_hindcasts(target_product=product)

    LOG.info("Finished preparing data in data/historic/%s-aligned", product)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    prepare_temporal_data()


if __name__ == "__main__":
    main()
