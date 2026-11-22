"""Build date-aligned cumulative rainfall directly from the rainfall HDF5."""
from __future__ import annotations

import argparse
import logging
import os

import h5py
import numpy as np

from .. import cleaning_utils
from ..config import get_cfg
from .gridded_streaming import PeriodFiles, build_grid_masks, resolve_project_path, sync_product

LOG = logging.getLogger(__name__)
GR_RAINFALL_TEMPORAL_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall_temporal.csv")
GR_RAINFALL_H5_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall.h5")
GR_RAINFALL_CUM_TEMPORAL_PATH = cleaning_utils.get_target_historic_path(
    "gridded_rainfall_cumulative_temporal.csv"
)
GR_RAINFALL_CUM_H5_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall_cumulative.h5")
CATCHMENTS_PATH = get_cfg("paths.maps.catchments", "data/maps/inflow_catchments/INFLOW_all_cmts.shp")


def get_historic_dates(data_path=GR_RAINFALL_CUM_TEMPORAL_PATH):
    if not os.path.exists(data_path):
        return []
    import pandas as pd
    frame = pd.read_csv(data_path)
    return frame["date"].astype(str).tolist() if "date" in frame else []


def update_gridded_rainfall_cumulative(
    data_path=GR_RAINFALL_CUM_H5_PATH,
    temporal_data_path=GR_RAINFALL_CUM_TEMPORAL_PATH,
    rebuild=False,
):
    """Rebuild or append cumulative grids in exact rainfall-date order."""
    rainfall_hdf_path = resolve_project_path(GR_RAINFALL_H5_PATH)
    data_path = resolve_project_path(data_path)
    temporal_data_path = resolve_project_path(temporal_data_path)
    if not rainfall_hdf_path.exists():
        raise FileNotFoundError("Process gridded rainfall before cumulative rainfall")
    with h5py.File(rainfall_hdf_path, "r") as source:
        if "rainfall" not in source or "dates" not in source:
            raise ValueError("Rainfall HDF5 must contain rainfall and dates datasets")
        rainfall_dates = list(source["dates"].asstr()[:])
        if len(rainfall_dates) != source["rainfall"].shape[0]:
            raise ValueError("Rainfall HDF5 values/date lengths differ")

    periods = [PeriodFiles(date, tuple()) for date in rainfall_dates]
    masks = build_grid_masks(CATCHMENTS_PATH)
    cumulative = np.zeros((masks.height, masks.width), dtype=np.float32)
    if not rebuild and os.path.exists(data_path):
        try:
            with h5py.File(data_path, "r") as existing:
                existing_dates = list(existing["dates"].asstr()[:]) if "dates" in existing else []
                valid = (
                    "cumulative_rainfall" in existing
                    and len(existing_dates) == existing["cumulative_rainfall"].shape[0]
                    and rainfall_dates[:len(existing_dates)] == existing_dates
                )
                if valid and existing_dates:
                    cumulative = existing["cumulative_rainfall"][-1].astype(np.float32)
                elif not valid:
                    rebuild = True
        except (OSError, KeyError, ValueError):
            rebuild = True
    if rebuild:
        cumulative.fill(0.0)

    date_to_index = {date: index for index, date in enumerate(rainfall_dates)}

    with h5py.File(rainfall_hdf_path, "r") as source:
        rainfall_values = source["rainfall"]

        def load_cumulative(period):
            nonlocal cumulative
            rainfall = rainfall_values[date_to_index[period.start]]
            cumulative = cumulative + np.nan_to_num(rainfall, nan=0.0)
            cumulative[~masks.inside] = np.nan
            return cumulative.copy()

        status, count = sync_product(
            periods=periods,
            hdf_path=data_path,
            csv_path=temporal_data_path,
            dataset_name="cumulative_rainfall",
            prefix="cumulative_rainfall",
            target_product=cleaning_utils.resolve_target_product(None),
            masks=masks,
            loader=load_cumulative,
            rebuild=rebuild,
        )
    LOG.info("Cumulative rainfall outputs %s (%d periods written).", status, count)
    return data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    update_gridded_rainfall_cumulative(rebuild=arguments.rebuild)


if __name__ == "__main__":
    main()
