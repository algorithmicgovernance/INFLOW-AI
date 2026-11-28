import json

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from processing.data_cleaning import process_basin_attributes as basin


def test_basin_processing_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(basin, "get_cfg", lambda path, default=None: default)
    assert basin.process_basin_attributes() is None


def test_basin_processing_builds_reference_grid_pca_layers(monkeypatch, tmp_path):
    basin_path = tmp_path / "basins.gpkg"
    catchments_path = tmp_path / "catchments.gpkg"
    output_path = tmp_path / "basin_attributes.npy"
    metadata_path = tmp_path / "basin_attributes.metadata.json"
    preprocessing_path = tmp_path / "basin_attributes.preprocessing.npz"

    gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 102, 103],
            "DIST_SINK": [1.0, 2.0, 4.0],
            "DIST_MAIN": [5.0, 3.0, 1.0],
            "glc_cl_smj": [15, 3, 15],
        },
        geometry=[box(0, 0, 1, 2), box(1, 0, 2, 2), box(2, 0, 3, 2)],
        crs="EPSG:4326",
    ).to_file(basin_path, driver="GPKG")
    gpd.GeoDataFrame(
        {"name": ["study"]}, geometry=[box(0, 0, 3, 2)], crs="EPSG:4326"
    ).to_file(catchments_path, driver="GPKG")

    values = {
        "basin_attributes_processing.enabled": True,
        "basin_attributes_processing.overwrite_existing": True,
        "basin_attributes_processing.basin_path": str(basin_path),
        "basin_attributes_processing.catchments_path": str(catchments_path),
        "basin_attributes_processing.output_path": str(output_path),
        "basin_attributes_processing.metadata_path": str(metadata_path),
        "basin_attributes_processing.preprocessing_path": str(preprocessing_path),
        "basin_attributes_processing.id_column": "HYBAS_ID",
        "basin_attributes_processing.attributes": [
            "DIST_SINK", "DIST_MAIN", "glc_cl_smj_15", "glc_cl_smj_3",
        ],
        "basin_attributes_processing.pca_min_explained_variance_ratio": 0.0,
        "basin_attributes_processing.minimum_components": 1,
        "basin_attributes_processing.rasterize_all_touched": False,
        "basin_attributes_processing.numeric_imputation_strategy": "median",
    }
    monkeypatch.setattr(basin, "get_cfg", lambda path, default=None: values.get(path, default))
    monkeypatch.setattr(basin.cleaning_utils, "MASK_REGIONS_REF_CRS", "EPSG:4326")
    monkeypatch.setattr(
        basin.cleaning_utils, "MASK_REGIONS_REF_TRANSFORM",
        tuple(rasterio.transform.from_origin(0, 2, 1, 1))[:6],
    )
    monkeypatch.setattr(basin.cleaning_utils, "MASK_REGIONS_REF_SHAPE", (2, 3))

    assert basin.process_basin_attributes() == output_path
    result = np.load(output_path)
    assert result.shape == (3, 2, 3)
    assert np.isfinite(result).all()
    assert preprocessing_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["represented_basin_count"] == 3
    assert metadata["component_count"] == 3
