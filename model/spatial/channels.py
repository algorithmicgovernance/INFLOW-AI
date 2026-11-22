"""Shared spatial-channel construction for ConvLSTM training and inference."""
from __future__ import annotations

import h5py
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from processing import cleaning_utils


CHANNEL_NAMES = (
    "inundation",
    "rainfall",
    "moisture",
    "elevation",
    "basin_attributes",
    "ecmwf_days_1_10",
    "ecmwf_days_11_20",
    "ecmwf_days_21_30",
    "ecmwf_days_31_40",
    "glofas_10_year_return_period",
)


def load_glofas_grid(path, height: int, width: int) -> np.ndarray:
    """Load the configured GloFAS return-period raster on the target grid."""
    with rasterio.open(path) as source:
        data = source.read(1, out_shape=(height, width), resampling=Resampling.bilinear)
        expected_crs = str(cleaning_utils.MASK_REGIONS_REF_CRS)
        if source.shape != (height, width) or str(source.crs) != expected_crs:
            raise ValueError(
                f"GloFAS raster must use the target grid {(height, width)} / {expected_crs}: {path}"
            )
    data = np.asarray(data, dtype=np.float32)
    # The supplied raster encodes cells outside the river network as -9999
    # without declaring a GDAL nodata value.
    data[data <= -9990.0] = 0.0
    return np.nan_to_num(data)


def ecmwf_patch(
    dataset: h5py.Dataset,
    time_index: int,
    row: int,
    col: int,
    patch_size: int,
    source_transform,
    source_crs,
) -> np.ndarray:
    """Reproject four coarse ECMWF lead grids directly into one target-grid patch."""
    target_transform = rasterio.Affine(*cleaning_utils.MASK_REGIONS_REF_TRANSFORM)
    patch_transform = rasterio.windows.transform(
        Window(col, row, patch_size, patch_size), target_transform
    )
    # Rasterio supports multiband reprojection, avoiding four GDAL setup calls.
    output = np.zeros((4, patch_size, patch_size), dtype=np.float32)
    reproject(
        source=np.asarray(dataset[time_index], dtype=np.float32),
        destination=output,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=np.nan,
        dst_transform=patch_transform,
        dst_crs=cleaning_utils.MASK_REGIONS_REF_CRS,
        dst_nodata=0.0,
        resampling=Resampling.bilinear,
    )
    return np.nan_to_num(output.transpose(1, 2, 0), copy=False)

