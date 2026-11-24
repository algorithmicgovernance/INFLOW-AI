"""Build reproducible HydroATLAS PCA layers on the shared spatial grid.

This processor replaces the original notebook workflow. It rasterizes basin
membership directly onto the configured reference grid, standardizes numeric
attributes, expands selected categorical levels, fits PCA to represented
basins, and saves all components above the configured variance threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .. import cleaning_utils
from ..config import get_cfg

LOG = logging.getLogger(__name__)


def _categorical_spec(attribute: str) -> tuple[str, str] | None:
    """Split a requested dummy name such as ``glc_cl_smj_15``."""
    if "_cl_" not in attribute or "_" not in attribute:
        return None
    source, category = attribute.rsplit("_", 1)
    return source, category


def _source_columns(attributes: list[str]) -> tuple[list[str], dict[str, tuple[str, str]]]:
    numeric = []
    categorical = {}
    for attribute in attributes:
        specification = _categorical_spec(attribute)
        if specification is None:
            numeric.append(attribute)
        else:
            categorical[attribute] = specification
    return numeric, categorical


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_numpy_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, values)
    temporary.replace(path)


def _atomic_npz_save(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def process_basin_attributes(force: bool = False) -> Path | None:
    """Create basin PCA rasters when enabled, returning the output path."""
    enabled = bool(get_cfg("basin_attributes_processing.enabled", False))
    if not enabled and not force:
        LOG.info("Basin-attribute processing is disabled in config.yaml")
        return None

    basin_path = Path(get_cfg(
        "basin_attributes_processing.basin_path",
        "data/maps/hydroatlas/inflow-basin-atlas.gpkg",
    ))
    catchments_path = Path(get_cfg(
        "basin_attributes_processing.catchments_path",
        get_cfg("paths.maps.catchments", "data/maps/inflow_catchments/INFLOW_all_cmts.shp"),
    ))
    output_path = Path(get_cfg(
        "basin_attributes_processing.output_path",
        get_cfg("spatial_training.basin_attributes_path", "data/maps/basin_attributes.npy"),
    ))
    metadata_path = Path(get_cfg(
        "basin_attributes_processing.metadata_path",
        output_path.with_suffix(".metadata.json"),
    ))
    preprocessing_path = Path(get_cfg(
        "basin_attributes_processing.preprocessing_path",
        output_path.with_suffix(".preprocessing.npz"),
    ))
    overwrite = bool(get_cfg("basin_attributes_processing.overwrite_existing", False))
    if output_path.exists() and not (overwrite or force):
        LOG.info("Basin attributes already exist at %s; skipping", output_path)
        return output_path
    if not basin_path.exists():
        raise FileNotFoundError(f"HydroATLAS basin source not found: {basin_path}")
    if not catchments_path.exists():
        raise FileNotFoundError(f"Catchments source not found: {catchments_path}")

    attributes = [str(value) for value in get_cfg("basin_attributes_processing.attributes", [])]
    if not attributes:
        raise ValueError("basin_attributes_processing.attributes must not be empty")
    id_column = str(get_cfg("basin_attributes_processing.id_column", "HYBAS_ID"))
    numeric_attributes, categorical_attributes = _source_columns(attributes)
    required = {id_column, *numeric_attributes,
                *(source for source, _ in categorical_attributes.values())}

    basins = gpd.read_file(basin_path)
    catchments = gpd.read_file(catchments_path)
    missing = required.difference(basins.columns)
    if missing:
        raise ValueError(f"HydroATLAS source is missing configured columns: {sorted(missing)}")
    if basins.crs is None or catchments.crs is None:
        raise ValueError("Basin and catchment files must declare a CRS")

    target_crs = cleaning_utils.MASK_REGIONS_REF_CRS
    target_transform = rasterio.Affine(*cleaning_utils.MASK_REGIONS_REF_TRANSFORM)
    target_shape = tuple(map(int, cleaning_utils.MASK_REGIONS_REF_SHAPE))
    basins = basins.to_crs(target_crs)
    catchments = catchments.to_crs(target_crs)
    study_geometry = catchments.geometry.union_all()
    basins = basins[basins.geometry.notna() & basins.geometry.intersects(study_geometry)].copy()
    basins = basins.drop_duplicates(id_column).reset_index(drop=True)
    if basins.empty:
        raise ValueError("No HydroATLAS basins intersect the configured catchments")

    all_touched = bool(get_cfg("basin_attributes_processing.rasterize_all_touched", False))
    basin_index = rasterize(
        ((geometry, index + 1) for index, geometry in enumerate(basins.geometry)),
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        all_touched=all_touched,
        dtype="int32",
    )
    catchment_mask = rasterize(
        ((geometry, 1) for geometry in catchments.geometry if geometry is not None),
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)
    basin_index[~catchment_mask] = 0
    represented = np.unique(basin_index)
    represented = represented[represented > 0]
    if not len(represented):
        raise ValueError("No basins rasterized inside the catchment mask")
    represented_rows = basins.iloc[represented - 1].copy()

    strategy = str(get_cfg("basin_attributes_processing.numeric_imputation_strategy", "median"))
    imputer = SimpleImputer(strategy=strategy)
    scaler = StandardScaler()
    numeric = represented_rows[numeric_attributes].apply(pd.to_numeric, errors="coerce")
    scaled_numeric = scaler.fit_transform(imputer.fit_transform(numeric))
    numeric_lookup = {name: scaled_numeric[:, index]
                      for index, name in enumerate(numeric_attributes)}

    feature_columns = []
    for attribute in attributes:
        if attribute in numeric_lookup:
            feature_columns.append(numeric_lookup[attribute])
            continue
        source, category = categorical_attributes[attribute]
        source_values = represented_rows[source]
        numeric_category = pd.to_numeric(pd.Series([category]), errors="coerce").iloc[0]
        if pd.notna(numeric_category):
            encoded = pd.to_numeric(source_values, errors="coerce").eq(numeric_category)
        else:
            encoded = source_values.astype(str).eq(category)
        feature_columns.append(encoded.to_numpy(dtype=np.float64))
    feature_matrix = np.column_stack(feature_columns)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Non-finite values remain after basin feature preprocessing")

    pca = PCA(n_components=min(feature_matrix.shape), svd_solver="full")
    principal_components = pca.fit_transform(feature_matrix)
    threshold = float(get_cfg(
        "basin_attributes_processing.pca_min_explained_variance_ratio", 0.01
    ))
    minimum_components = int(get_cfg("basin_attributes_processing.minimum_components", 1))
    selected = np.flatnonzero(pca.explained_variance_ratio_ >= threshold)
    if len(selected) < minimum_components:
        selected = np.arange(min(minimum_components, principal_components.shape[1]))
    selected_scores = principal_components[:, selected].astype(np.float32)

    lookup = np.zeros((len(basins) + 1, len(selected)), dtype=np.float32)
    lookup[represented, :] = selected_scores
    output = lookup[basin_index].transpose(2, 0, 1)
    output[:, ~catchment_mask] = 0.0
    _atomic_numpy_save(output_path, output)
    _atomic_npz_save(
        preprocessing_path,
        attributes=np.asarray(attributes),
        numeric_attributes=np.asarray(numeric_attributes),
        imputer_statistics=np.asarray(imputer.statistics_, dtype=np.float64),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        pca_components=np.asarray(pca.components_[selected], dtype=np.float64),
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_[selected]),
        selected_component_indices=selected,
    )
    metadata = {
        "schema_version": "hydroatlas-pca-reference-grid-v1",
        "basin_source": str(basin_path),
        "basin_source_sha256": _sha256(basin_path),
        "catchments_source": str(catchments_path),
        "catchments_source_sha256": _sha256(catchments_path),
        "id_column": id_column,
        "attributes": attributes,
        "represented_basin_count": int(len(represented)),
        "component_count": int(len(selected)),
        "selected_component_indices": selected.tolist(),
        "selected_explained_variance_ratio": pca.explained_variance_ratio_[selected].tolist(),
        "total_selected_explained_variance_ratio": float(
            pca.explained_variance_ratio_[selected].sum()
        ),
        "output_shape": list(output.shape),
        "crs": str(target_crs),
        "transform": list(tuple(target_transform)[:6]),
        "preprocessing_path": str(preprocessing_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOG.info("Saved %d basin PCA layers to %s", len(selected), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Run even when disabled and replace an existing output")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    process_basin_attributes(force=arguments.force)


if __name__ == "__main__":
    main()
