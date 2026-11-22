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


def load_raster_on_target_grid(path, height: int, width: int) -> np.ndarray:
    """Read or reproject a raster onto the configured target reference grid."""
    destination = np.zeros((height, width), dtype=np.float32)
    target_transform = rasterio.Affine(*cleaning_utils.MASK_REGIONS_REF_TRANSFORM)
    with rasterio.open(path) as source:
        expected_crs = str(cleaning_utils.MASK_REGIONS_REF_CRS)
        expected_transform = tuple(target_transform)[:6]
        if (
            source.shape == (height, width)
            and str(source.crs) == expected_crs
            and np.allclose(tuple(source.transform)[:6], expected_transform)
        ):
            destination[:] = source.read(1)
        else:
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=target_transform,
                dst_crs=cleaning_utils.MASK_REGIONS_REF_CRS,
                dst_nodata=0.0,
                resampling=Resampling.bilinear,
            )
    data = destination
    # The supplied raster encodes cells outside the river network as -9999
    # without declaring a GDAL nodata value.
    data[data <= -9990.0] = 0.0
    return np.nan_to_num(data)


def load_glofas_grid(path, height: int, width: int) -> np.ndarray:
    """Load a configured GloFAS raster on the target grid."""
    return load_raster_on_target_grid(path, height, width)


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

