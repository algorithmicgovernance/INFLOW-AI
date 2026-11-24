import os
import numpy as np
import pandas as pd
import h5py
import logging
from collections import defaultdict
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResampleEnums
from shapely.geometry import Point, MultiPolygon
from matplotlib.patches import PathPatch
from pathlib import Path as FilePath
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from processing.data_cleaning import process_inundation_modis
from processing import cleaning_utils
from processing.config import get_cfg, resolve_forecast_horizon_periods
from model.spatial.channels import CHANNEL_NAMES, ecmwf_patch, load_glofas_grid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def check_file_lengths(*file_paths):
    lengths = []
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            key = list(f.keys())[0]
            lengths.append(f[key].shape[0])
    return lengths


def generate_deployment_sequences_in_memory(
    inundation_file_path,
    rainfall_file_path,
    moisture_file_path,
    ecmwf_file_path,
    glofas_file_path,
    static_dir=None,
    patch_size=None,
    stride=None,
    sequence_length=None,
    forecast_horizon=None):
    """
    Generates deployment-ready spatial sequences (in memory) for the last available time slices.

    Returns:
        X: np.ndarray of shape (N, sequence_length, 10, patch_size, patch_size)
        indices: np.ndarray of shape (N, 3) where each row is (origin, y, x)
    """

    static_dir = static_dir or get_cfg("spatial_prediction.static_dir", "data/maps")
    patch_size = int(patch_size or get_cfg("spatial_training.patch_size", 64))
    stride = int(stride or get_cfg("spatial_training.stride", 32))
    sequence_length = int(sequence_length or get_cfg("spatial_training.sequence_length", 6))
    inun_len, rain_len, moist_len = check_file_lengths(
        inundation_file_path, rainfall_file_path, moisture_file_path
    )
    with h5py.File(ecmwf_file_path, "r") as ecmwf_f:
        ecmwf_len = int(ecmwf_f["X"].shape[0])

    if not (inun_len == rain_len == moist_len == ecmwf_len):
        logger.warning("Time series lengths do not match:")
        logger.warning(f"  - Inundation: {inun_len}")
        logger.warning(f"  - Rainfall:   {rain_len}")
        logger.warning(f"  - Moisture:   {moist_len}")
        logger.warning(f"  - ECMWF:      {ecmwf_len}")
    
    T = min(inun_len, rain_len, moist_len, ecmwf_len)
    forecast_horizon = int(forecast_horizon or resolve_forecast_horizon_periods())
    
    if not (inun_len == rain_len == moist_len == ecmwf_len):
        logger.info(f"Using shortest time series ({T}) as reference.")

    # === Load static maps ===
    elevation_path = get_cfg("spatial_training.elevation_path", os.path.join(static_dir, "elevation.npy"))
    basin_path = get_cfg(
        "spatial_training.basin_attributes_path", os.path.join(static_dir, "basin_attributes.npy")
    )
    elevation = np.load(elevation_path)  # (1, H, W)
    basin_attributes = np.load(basin_path)[0:1]  # (1, H, W)

    # Rainfall and moisture use whole-timestep chunks. A cache large enough for
    # the six input frames prevents decompression for every spatial patch.
    cache_mb = int(get_cfg("spatial_training.hdf5_cache_mb", 64))
    cache_options = {"mode": "r", "rdcc_nbytes": cache_mb * 1024 * 1024, "rdcc_nslots": 10007}
    with h5py.File(inundation_file_path, **cache_options) as inun_f, \
         h5py.File(rainfall_file_path, **cache_options) as rain_f, \
         h5py.File(moisture_file_path, **cache_options) as moist_f, \
         h5py.File(ecmwf_file_path, **cache_options) as ecmwf_f:

        inundation = inun_f['inundation']
        rainfall = rain_f['rainfall']
        moisture = moist_f['moisture']
        ecmwf = ecmwf_f['X']
        ecmwf_transform = tuple(ecmwf_f.attrs['transform'])
        ecmwf_crs = ecmwf_f.attrs.get('crs', 'EPSG:4326')

        _, H, W = inundation.shape

        # Trim static maps
        elevation = elevation[:, :H, :W]
        basin_attributes = basin_attributes[:, :H, :W]
        glofas = load_glofas_grid(glofas_file_path, H, W)

        # Use all observations through the latest period. The model emits one
        # map for every configured future temporal-model horizon.
        start_t = T - sequence_length
        end_t = T

        y_positions = np.arange(0, H - patch_size + 1, stride)
        x_positions = np.arange(0, W - patch_size + 1, stride)

        num_spatial_patches = len(y_positions) * len(x_positions)
        total_samples = num_spatial_patches

        logger.info(f"Generating {total_samples} input samples from time {start_t} to {end_t}...")

        # Preallocate in-memory arrays
        X = np.zeros(
            (total_samples, sequence_length, len(CHANNEL_NAMES), patch_size, patch_size),
            dtype=np.float32,
        )
        indices = np.zeros((total_samples, 3), dtype=np.int32)

        sample_idx = 0

        origin = T - 1
        for y_start in y_positions:
            for x_start in x_positions:
                y_end = y_start + patch_size
                x_end = x_start + patch_size

                inun_seq = inundation[start_t:end_t, y_start:y_end, x_start:x_end]
                rain_seq = rainfall[start_t:end_t, y_start:y_end, x_start:x_end]
                moist_seq = moisture[start_t:end_t, y_start:y_end, x_start:x_end]

                elev_patch = elevation[:, y_start:y_end, x_start:x_end]
                basin_patch = basin_attributes[:, y_start:y_end, x_start:x_end]
                elev_seq = np.repeat(elev_patch, sequence_length, axis=0)
                basin_seq = np.repeat(basin_patch, sequence_length, axis=0)
                glofas_patch = glofas[y_start:y_end, x_start:x_end]
                glofas_seq = np.repeat(glofas_patch[np.newaxis, ...], sequence_length, axis=0)
                ecmwf_seq = np.stack([
                    ecmwf_patch(
                        ecmwf, time_index, y_start, x_start, patch_size,
                        ecmwf_transform, ecmwf_crs,
                    ).transpose(2, 0, 1)
                    for time_index in range(start_t, end_t)
                ])

                X_sample = np.zeros(
                    (sequence_length, len(CHANNEL_NAMES), patch_size, patch_size), dtype=np.float32
                )
                X_sample[:, 0:5] = np.stack(
                    [inun_seq, rain_seq, moist_seq, elev_seq, basin_seq], axis=1
                )
                X_sample[:, 5:9] = ecmwf_seq
                X_sample[:, 9] = glofas_seq
                X[sample_idx] = X_sample
                indices[sample_idx] = (origin, y_start, x_start)
                sample_idx += 1

    logger.info(f"✅ In-memory sequence generation complete: {X.shape} samples")

    X = X[:sample_idx]
    indices = indices[:sample_idx]

    return X, indices


def crop_borders(arr, border=4):
    if arr.ndim == 4:
        return arr[:, border:-border, border:-border, :]
    if arr.ndim == 5:
        return arr[:, :, border:-border, border:-border, :]
    raise ValueError(f"Expected 4D or 5D spatial predictions, got shape {arr.shape}.")


def load_trained_model(model_path):
    try:
        from keras.models import load_model
    except ImportError as exc:
        raise ImportError(
            "Spatial inference requires TensorFlow/Keras; use the spatial-training environment."
        ) from exc
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path, compile=False)
    logger.info("✅ Model loaded")
    return model


def validate_spatial_model_contract(model, forecast_horizon):
    """Fail before loading large grids when the saved model is incompatible."""
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    expected_channels = len(CHANNEL_NAMES)
    if len(input_shape) != 5 or int(input_shape[-1]) != expected_channels:
        raise ValueError(
            f"Spatial model input must be (batch, time, y, x, {expected_channels}); "
            f"got {input_shape}. Retrain with python3 -m model.spatial.train_convlstm."
        )
    if len(output_shape) != 5 or int(output_shape[1]) != int(forecast_horizon):
        raise ValueError(
            f"Spatial model must output {forecast_horizon} future maps with shape "
            f"(batch, horizon, y, x, 1); saved model output is {output_shape}. "
            "The configured model file is not the six-horizon checkpoint. Retrain or "
            "copy the correct checkpoint to spatial_training.model_path."
        )


def predict_with_model(model, X, batch_size=None):
    logger.info("Running model prediction...")
    X_input = X.transpose(0, 1, 3, 4, 2)  # To (N, T, H, W, C)
    expected_channels = int(model.input_shape[-1])
    if X_input.shape[-1] != expected_channels:
        raise ValueError(
            f"Spatial model expects {expected_channels} channels but the configured pipeline produced "
            f"{X_input.shape[-1]}. Retrain with python3 -m model.spatial.train_convlstm."
        )
    print("Making spatial predictions...")
    batch_size = int(batch_size or get_cfg("spatial_prediction.prediction_batch_size", 32))
    y_pred = model.predict(X_input, batch_size=batch_size, verbose=1)
    logger.info("✅ Prediction complete")
    return y_pred


def reconstruct_maps(y_pred, indices, full_shape, border=4):
    logger.info("Reconstructing full maps from predicted patches...")

    if y_pred.ndim == 4:
        y_pred = y_pred[:, np.newaxis, ...]
    if y_pred.ndim != 5:
        raise ValueError(f"Expected predictions shaped (patch, lead, y, x, channel), got {y_pred.shape}.")
    patch_size = y_pred.shape[2]
    horizon_count = y_pred.shape[1]
    y_pred_cropped = crop_borders(y_pred, border=border)
    cropped_size = patch_size - 2 * border

    lead_to_patches = defaultdict(list)
    for i, (_, y, x) in enumerate(indices):
        for lead in range(horizon_count):
            patch = y_pred_cropped[i, lead, :, :, 0]
            y0 = y + border
            x0 = x + border
            y1 = y0 + cropped_size
            x1 = x0 + cropped_size
            lead_to_patches[lead + 1].append((patch, y0, y1, x0, x1))

    time_maps = {}
    for lead, patches in lead_to_patches.items():
        sum_map = np.zeros(full_shape, dtype=np.float32)
        count_map = np.zeros(full_shape, dtype=np.uint8)

        for patch, y0, y1, x0, x1 in patches:
            sum_map[y0:y1, x0:x1] += patch
            count_map[y0:y1, x0:x1] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_map = np.true_divide(sum_map, count_map)
            avg_map[count_map == 0] = np.nan
        time_maps[lead] = avg_map

    logger.info(f"✅ Reconstructed {len(time_maps)} time-indexed maps")
    return time_maps


def run_inference_pipeline(
    model_path=None,
    inundation_file_path=None,
    rainfall_file_path=None,
    moisture_file_path=None,
    ecmwf_file_path=None,
    glofas_file_path=None,
    static_dir=None,
    full_shape=None,
    border=None
):
    target_product = cleaning_utils.resolve_target_product(get_cfg("runtime.target_product", "viirs"))
    aligned_dir = FilePath(get_cfg("paths.historic.root", "data/historic")) / f"{target_product}-aligned"
    inundation_file_path = inundation_file_path or str(
        aligned_dir / ("inundation_viirs.h5" if target_product == "viirs" else "inundation_modis.h5")
    )
    rainfall_file_path = rainfall_file_path or str(aligned_dir / "gridded_rainfall.h5")
    moisture_file_path = moisture_file_path or str(aligned_dir / "gridded_moisture.h5")
    ecmwf_file_path = ecmwf_file_path or str(aligned_dir / "ecmwf_hindcasts.h5")
    glofas_file_path = glofas_file_path or get_cfg(
        "spatial_training.glofas_path", "data/maps/flood_return_period/glofas_10_year_rp.tif"
    )
    model_path = model_path or get_cfg("spatial_training.model_path", "model/spatial_model.keras")
    static_dir = static_dir or get_cfg("spatial_prediction.static_dir", "data/maps")
    border = int(border if border is not None else get_cfg("spatial_prediction.patch_border", 4))
    with h5py.File(inundation_file_path, "r") as hdf:
        inferred_shape = tuple(map(int, hdf["inundation"].shape[-2:]))
    full_shape = tuple(full_shape or inferred_shape)
    model = load_trained_model(model_path)
    forecast_horizon = resolve_forecast_horizon_periods()
    validate_spatial_model_contract(model, forecast_horizon)

    X, indices = generate_deployment_sequences_in_memory(
        inundation_file_path,
        rainfall_file_path,
        moisture_file_path,
        ecmwf_file_path,
        glofas_file_path,
        static_dir=static_dir,
        forecast_horizon=forecast_horizon,
    )

    y_pred = predict_with_model(model, X)
    if y_pred.ndim != 5 or y_pred.shape[1] != forecast_horizon:
        raise ValueError(
            f"Spatial model must output {forecast_horizon} future maps; got {y_pred.shape}. "
            "Retrain it with python3 -m model.spatial.train_convlstm."
        )
    maps = reconstruct_maps(y_pred, indices, full_shape, border=border)
    return maps


def load_spatial_ref(inundation_path, catchments_path, download_path, inundation_file):
    # Process the new TIF files
    with rasterio.open(inundation_path) as src:
        inundation = src.read(1)  # First band
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

        catchments = process_inundation_modis.load_shapefile(catchments_path)
        catchments = process_inundation_modis.reproject_to_raster_crs(catchments, inundation_path)

        # Process rasters and gather new data
        inundation_clipped, _, _ = process_inundation_modis.process_and_clip_rasters([inundation_file], download_path, catchments)
        inundation = inundation_clipped[0]

        # Crop area to regions of interest
        regions_gdf = process_inundation_modis.cleaning_utils.extract_regions()

        import numpy as np
        from rasterio.features import rasterize

        # Assign each region a unique integer ID
        regions_gdf = regions_gdf.copy()
        regions_gdf['region_id'] = np.arange(1, len(regions_gdf) + 1)

        # Prepare list of (geometry, value) tuples
        shapes = list(zip(regions_gdf.geometry, regions_gdf['region_id']))

        # Get transform and shape from raster
        height, width = inundation.shape
        regions_gdf = regions_gdf.to_crs(crs)

        # Reproject regions_gdf to WGS84 (EPSG:4326)
        regions_gdf = regions_gdf.to_crs(epsg=4326)

        # Raster reprojection to EPSG:4326
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        dst_crs = 'EPSG:4326'
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, dst_crs, width, height, *src.bounds)

        # Create an empty array for reprojected raster
        reprojected_inundation = np.empty(shape=(height, width), dtype=inundation.dtype)
        reproject(
            source=inundation,
            destination=reprojected_inundation,
            src_transform=src.transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        # Return the new transform, CRS, and reprojected regions_gdf & raster array
        return dst_transform, dst_crs, regions_gdf, reprojected_inundation


def polygon_to_path_patch(polygon, hatch='///', **kwargs):
    """Convert a Shapely Polygon or MultiPolygon into a matplotlib PathPatch with hatching."""
    if isinstance(polygon, MultiPolygon):
        patches = []
        for poly in polygon.geoms:
            patches.append(polygon_to_path_patch(poly, hatch=hatch, **kwargs))
        return patches

    # Extract exterior and interior coordinates
    vertices = []
    codes = []

    # Exterior
    x, y = polygon.exterior.coords.xy
    verts = np.column_stack([x, y])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    vertices.extend(verts.tolist())

    # Interiors (holes)
    for interior in polygon.interiors:
        x, y = interior.coords.xy
        verts = np.column_stack([x, y])
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
        vertices.extend(verts.tolist())

    path = MplPath(vertices, codes)
    return PathPatch(path, facecolor='none', edgecolor='black', hatch=hatch, lw=0.0, alpha=0.4, **kwargs)


def align_mask(mask_array, src_transform, src_crs, metas, inundation_file):
    aligned = np.zeros((
        metas[inundation_file]["height"],
        metas[inundation_file]["width"]
        ), dtype=np.uint8)
    reproject(
        source=mask_array,
        destination=aligned,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=metas[inundation_file]["transform"],
        dst_crs=metas[inundation_file]['crs'],
        resampling=ResampleEnums.nearest
    )
    return aligned


def _resolve_latlon_columns(df):
    lat_cols = ['latitude', 'lat', 'Lat', 'Latitude', 'LAT', 'Y', 'y']
    lon_cols = ['longitude', 'lon', 'Lon', 'Longitude', 'LON', 'X', 'x']
    lat_col = next((c for c in lat_cols if c in df.columns), None)
    lon_col = next((c for c in lon_cols if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError("Could not identify latitude/longitude columns in exposure file")
    return lat_col, lon_col


def load_exposure_gdf(filepath, facility_type):
    if facility_type == 'school':
        df = pd.read_csv(filepath, encoding="latin1")
    else:
        df = pd.read_csv(filepath)

    lat_col, lon_col = _resolve_latlon_columns(df)
    name_col = next((c for c in ['name', 'Name', 'facility_name', 'Facility Name', 'Health Facility', 'SchName'] if c in df.columns), None)
    if name_col is None:
        df['name'] = f"unknown_{facility_type}"
        name_col = 'name'

    gdf = gpd.GeoDataFrame(
        df.assign(geometry=gpd.points_from_xy(df[lon_col], df[lat_col])),
        geometry='geometry',
        crs='EPSG:4326'
    )
    gdf = gdf.rename(columns={name_col: 'name'})
    keep_cols = ['name', lat_col, lon_col, 'geometry'] + [c for c in ['State', 'County', 'Payam'] if c in gdf.columns]
    gdf = gdf[keep_cols].copy()
    gdf['facility_type'] = facility_type
    gdf = gdf.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
    return gdf


def get_exposure_points(hospital_path=None, school_path=None):
    nodes = []
    for path, facility_type in [(hospital_path, 'hospital'), (school_path, 'school')]:
        if not path:
            continue
        if not os.path.exists(path):
            logger.warning(f"Exposure file not found; skipping: {path}")
            continue
        try:
            gdf = load_exposure_gdf(path, facility_type)
            nodes.append(gdf)
            logger.info(f"Loaded {len(gdf)} {facility_type} records from {path}")
        except Exception as e:
            logger.warning(f"Failed to load exposure data from {path}: {e}")

    if not nodes:
        logger.warning("No exposure points loaded; no impact CSV will be created.")
        return gpd.GeoDataFrame(columns=['name', 'latitude', 'longitude', 'facility_type', 'geometry'])

    all_points = pd.concat(nodes, ignore_index=True)
    return gpd.GeoDataFrame(all_points, geometry='geometry', crs='EPSG:4326')


def _classify_change(in_current, in_scenario):
    if in_current and not in_scenario:
        return "become_dry"
    if in_current and in_scenario:
        return "stay_wet"
    if not in_current and not in_scenario:
        return "stay_dry"
    return "become_wet"


def export_impacted_facilities(masks, current_extent, metas, inundation_file, points_gdf,
                               output_dir, filename, source_transform=None, source_crs=None):
    if points_gdf.empty:
        logger.warning("No exposure points found, skipping impact CSV generation.")
        return None

    target_crs = metas[inundation_file]["crs"]
    target_transform = metas[inundation_file]["transform"]
    source_transform = source_transform or target_transform
    source_crs = source_crs or target_crs

    points_projected = points_gdf.to_crs(target_crs)

    current_aligned = align_mask(
        (current_extent > 0).astype(np.uint8),
        source_transform,
        source_crs,
        metas,
        inundation_file
    )

    def _extract_status_row(row, scenario_mask):
        x = row.geometry.x
        y = row.geometry.y
        r, c = rasterio.transform.rowcol(target_transform, [x], [y], op=np.floor)
        r, c = int(r[0]), int(c[0])

        if 0 <= r < current_aligned.shape[0] and 0 <= c < current_aligned.shape[1]:
            in_current = current_aligned[r, c] == 1
            in_scenario = scenario_mask[r, c] == 1
            return _classify_change(in_current, in_scenario)

        return np.nan

    base_cols = [c for c in ["name", "latitude", "longitude", "State", "County", "Payam", "facility_type"] if c in points_projected.columns]
    result_base = points_projected[base_cols].copy()

    for scenario in ["Worst Case", "Average Case", "Best Case"]:
        scenario_aligned = align_mask(
            (masks[scenario] > 0).astype(np.uint8),
            source_transform,
            source_crs,
            metas,
            inundation_file
        )
        col_name = scenario.lower().replace(" ", "_")
        result_base[col_name] = points_projected.apply(
            lambda row: _extract_status_row(row, scenario_aligned),
            axis=1
        )

    csv_path = output_dir / filename
    result_base.to_csv(csv_path, index=False)
    logger.info(f"Saved impacted facility list: {csv_path}")
    return csv_path


def plot_flood_change_map(
    masks, current_extent, transform, crs, regions_gdf,
    inundation_clipped, metas, inundation_file='20250211.tif',
    title="Flood Change Map", region_name=None):

    # --- Align all masks ---
    current = align_mask((current_extent > 0).astype(np.uint8), transform, crs, metas, inundation_file)
    worst = align_mask((masks["Worst Case"] > 0).astype(np.uint8), transform, crs, metas, inundation_file)
    avg = align_mask((masks["Average Case"] > 0).astype(np.uint8), transform, crs, metas, inundation_file)
    best = align_mask((masks["Best Case"] > 0).astype(np.uint8), transform, crs, metas, inundation_file)

    # Use the transform from clipped raster metadata (not src!)
    out_meta = metas[inundation_file].copy()
    target_transform = out_meta["transform"]
    target_crs = out_meta["crs"]
    target_shape = inundation_clipped[0].shape

    # --- Change classification ---
    currently_flooded = current == 1
    currently_dry = current == 0

    no_change_dry = currently_dry & (worst == 0) & (avg == 0) & (best == 0)
    no_change_flooded = currently_flooded & (worst == 1) & (avg == 1) & (best == 1)
    very_likely_decrease = currently_flooded & (worst == 0) & (avg == 0) & (best == 0)
    likely_decrease = currently_flooded & (worst == 1) & (avg == 0) & (best == 0)
    possible_decrease = currently_flooded & (worst == 1) & (avg == 1) & (best == 0)
    very_likely_increase = currently_dry & (worst == 1) & (avg == 1) & (best == 1)
    likely_increase = currently_dry & (worst == 1) & (avg == 1) & (best == 0)
    possible_increase = currently_dry & (worst == 1) & (avg == 0) & (best == 0)

    # --- Color mapping ---
    colors = np.zeros((*current.shape, 4))
    colors[no_change_dry] = mcolors.to_rgba("#ffffff")
    colors[no_change_flooded] = mcolors.to_rgba("#d3d3d3")
    colors[possible_decrease] = mcolors.to_rgba("#f4a261")
    colors[likely_decrease] = mcolors.to_rgba("#e76f51")
    colors[very_likely_decrease] = mcolors.to_rgba("#9b2226")
    colors[possible_increase] = mcolors.to_rgba("#a8dadc")
    colors[likely_increase] = mcolors.to_rgba("#457b9d")
    colors[very_likely_increase] = mcolors.to_rgba("#1d3557")

    # --- Select region for boundary mask ---
    if region_name:
        selected_region = regions_gdf[regions_gdf['region'] == region_name]
        selected_region = selected_region.to_crs(target_crs)
        if selected_region.empty:
            raise ValueError(f"Region '{region_name}' not found in regions_gdf.")
        state_boundaries = selected_region
        country_boundary = selected_region
    else:
        state_boundaries = regions_gdf[regions_gdf["region_id"] != 1]
        country_boundary = regions_gdf[regions_gdf["region_id"] == 1]
    state_boundaries = state_boundaries.to_crs(target_crs)
    country_boundary = country_boundary.to_crs(target_crs)

    # --- Rasterize mask for region ---
    mask_geom = selected_region.geometry if region_name else country_boundary.geometry
    region_mask = rasterize(
        [(geom, 1) for geom in mask_geom],
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        dtype='uint8'
    )
    colors[region_mask == 0] = (1, 1, 1, 0)  # transparent

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 10))
    from rasterio.plot import plotting_extent

    extent = plotting_extent(
        np.zeros(target_shape),
        transform=target_transform
    )
    ax.imshow(colors, origin='upper', extent=extent)

    # --- Plot region boundaries ---
    state_boundaries.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.25)
    country_boundary.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.5)

    # --- Continue with legends, labels, points, hatching, etc. (unchanged) ---
    # Get Abyei geometry
    abyei_region = regions_gdf[regions_gdf["region"] == "Abyei Region"]
    abyei_region = abyei_region.to_crs(target_crs)

    # Only plot outline and hatching if full country or Abyei is selected
    show_abyei = region_name is None or region_name == "Abyei Region"

    if show_abyei:
        # Overlay dashed border
        for _, row in abyei_region.iterrows():
            g = row.geometry
            if g.geom_type == "MultiPolygon":
                for part in g.geoms:
                    x, y = part.exterior.xy
                    ax.plot(x, y, linestyle='--', color='white', linewidth=1.5)
            else:
                x, y = g.exterior.xy
                ax.plot(x, y, linestyle='--', color='white', linewidth=1.6)

        # Add hatching
        for _, row in abyei_region.iterrows():
            geom = row.geometry
            patch_or_patches = polygon_to_path_patch(geom, hatch='///')
            if isinstance(patch_or_patches, list):
                for patch in patch_or_patches:
                    ax.add_patch(patch)
            else:
                ax.add_patch(patch_or_patches)

    # --- Main legend for flood categories (bottom left) ---
    flood_legend_patches = [
        mpatches.Patch(color="#d3d3d3", label="No change (remains flooded)"),
        mpatches.Patch(color="#f4a261", label="Possible flood decrease (best-case only)"),
        mpatches.Patch(color="#e76f51", label="Likely flood decrease (average case)"),
        mpatches.Patch(color="#9b2226", label="Very likely flood decrease (all scenarios)"),
        mpatches.Patch(color="#a8dadc", label="Possible flood increase (worst-case only)"),
        mpatches.Patch(color="#457b9d", label="Likely flood increase (average case)"),
        mpatches.Patch(color="#1d3557", label="Very likely flood increase (all scenarios)")
    ]
    legend1 = ax.legend(
        handles=flood_legend_patches,
        loc="lower left",
        fontsize=10,
        frameon=False
    )
    ax.add_artist(legend1)  # ensure both legends show

    # --- Separate legend for population centres (bottom right) ---
    if region_name:
      pop_label = 'Population centre'
    else:
      minimum_population = int(get_cfg("spatial_prediction.minimum_population", 10000))
      pop_label = f'Population centre (≥{minimum_population:,} people)'

    population_legend = Line2D(
        [0], [0],
        marker='o',
        color='black',
        markerfacecolor='white',
        markersize=8,
        linewidth=0,
        label=pop_label
    )
    msf_legend = Line2D(
        [0], [0],
        marker='+',
        color='red',
        markersize=10,
        linewidth=1,
        label='MSF project location'
    )

    legend2 = ax.legend(
        handles=[population_legend, msf_legend],
        loc="lower right",
        fontsize=9,
        frameon=False
    )

    # Add region labels
    for _, row in state_boundaries.iterrows():
        point = row.geometry.representative_point()
        txt = ax.annotate(
            text=row['region'],
            xy=(point.x, point.y + 0.15),
            ha='center', va='center',
            fontsize=8,
            color='black',
            alpha=0.9,
            fontstyle='italic'
        )
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=2, foreground="white")
        ])

    # Plot population centres
    pop_centres = gpd.read_file(get_cfg(
        "spatial_prediction.population_centres_path",
        "data/maps/population_centres/hotosm_ssd_populated_places_points_shp.shp",
    ))
    pop_centres['name_en'] = np.where(pop_centres['name_en'].isna(), pop_centres['name'], pop_centres['name_en'])
    if pop_centres.crs != target_crs:
      pop_centres = pop_centres.to_crs(target_crs)
    pop_centres = pop_centres[pop_centres['population'].notna()]
    minimum_population = int(get_cfg("spatial_prediction.minimum_population", 10000))
    pop_centres = pop_centres[pop_centres['population'].astype(int) >= minimum_population]

    # Filter to selected region if applicable
    if region_name:
        pop_centres = gpd.sjoin(pop_centres, selected_region, how="inner", predicate="intersects")

    # Clip population centres to region
    if region_name:
        pop_centres_clipped = gpd.clip(pop_centres, selected_region)
    else:
        pop_centres_clipped = pop_centres

    # Plot only if non-empty
    if not pop_centres_clipped.empty:
        pop_centres_clipped.plot(ax=ax, markersize=20, color='white', edgecolor='black', linewidth=0.5, zorder=3)


    # --- Manually entered MSF project locations (as red plus signs, no labels) ---
    msf_locations = [
        ("Leer", 8.301, 30.107),
        ("Lankien", 8.519, 31.956),
        ("Bentiu", 9.239, 29.506),
        ("Renk", 11.746, 32.766),
        ("EGPAA (Boma & Maruwa)", 6.234, 34.193),
        ("Old Fangak", 9.0707, 30.8146),
        ("Aweil", 8.721, 27.276),
        ("Abyei", 9.628, 28.0616),
        ("Twic (Mayen Abun)", 9.1275, 28.11)
    ]

    # Plot MSF sites
    for _, lat, lon in msf_locations:
        point_geo = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(target_crs)
        x, y = point_geo.geometry.x.values[0], point_geo.geometry.y.values[0]

        if region_name:
            if not point_geo.intersects(selected_region.union_all()).iloc[0]:
                continue
        ax.plot(x, y, marker='+', color='red', markersize=7, zorder=5)

    # Add labels
    for _, row in pop_centres.iterrows():
        x, y = row.geometry.x, row.geometry.y
        label = row.get("name_en", "")
        if label:
            txt = ax.text(
                x + 0.02, y + 0.02,  # slight offset
                label,
                fontsize=6,
                color='black',
                zorder=4
            )
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=1.5, foreground="white")
            ])

    # Zoom to selected region
    if region_name:
        bounds = selected_region.total_bounds  # (minx, miny, maxx, maxy)
    else:
        bounds = country_boundary.total_bounds  # fallback to country-wide bounds

    # Compute width and height of bounds
    xmin, ymin, xmax, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin

    # Define padding as a percentage of width/height
    if region_name:
        x_pad_pct = 0.20
        y_pad_top_pct = 0.01
        y_pad_bottom_pct = 0.20
    else:
        x_pad_pct = 0.05
        y_pad_top_pct = 0.01
        y_pad_bottom_pct = 0.10

    # Convert percentages to absolute padding
    x_pad = width * x_pad_pct
    y_pad_top = height * y_pad_top_pct
    y_pad_bottom = height * y_pad_bottom_pct

    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad_bottom, ymax + y_pad_top)

    ax.axis('off')
    ax.set_title(title, size=16)
    ax.text(
    0.5, 1.01,  # x centered, y just under the main title
    f"*Forecast of {cleaning_utils.resolve_target_product(None).upper()} flood extent",
    transform=ax.transAxes,
    ha='center',
    fontsize=10,
    style='italic'
    )

    plt.tight_layout()

    return fig


def generate_flood_mask(prob_map, flood_percent):
    valid_mask = ~np.isnan(prob_map)
    valid_probs = prob_map[valid_mask]
    n_flood_pixels = int(np.round(flood_percent * valid_probs.size))
    sorted_probs = np.sort(valid_probs)[::-1]
    threshold = sorted_probs[n_flood_pixels - 1] if n_flood_pixels > 0 else np.inf
    binary_mask = np.zeros_like(prob_map, dtype=np.uint8)
    binary_mask[(prob_map >= threshold) & valid_mask] = 1
    return binary_mask


def run_operational_spatial_analysis(temporal_prediction_csv):
    """Generate one flood-change map and facility report for every forecast lead."""
    if not bool(get_cfg("spatial_prediction.enabled", True)):
        logger.info("Spatial prediction is disabled in config.yaml")
        return []

    target_product = cleaning_utils.resolve_target_product(None)
    historic_key = ("paths.historic.viirs_h5" if target_product == "viirs"
                    else "paths.historic.modis_h5")
    download_key = ("paths.downloads.inundation_viirs" if target_product == "viirs"
                    else "paths.downloads.inundation_modis")
    inundation_h5_path = FilePath(get_cfg(
        historic_key,
        f"data/historic/{target_product}-aligned/inundation_{target_product}.h5",
    ))
    inundation_download_path = FilePath(get_cfg(
        download_key, f"data/downloads/inundation_masks_{target_product}",
    ))
    catchments_path = get_cfg(
        "spatial_prediction.catchments_path",
        get_cfg("paths.maps.catchments", "data/maps/inflow_catchments/INFLOW_all_cmts.shp"),
    )
    raw_rasters = sorted(inundation_download_path.glob("*.tif"))
    if not inundation_h5_path.exists():
        raise FileNotFoundError(f"Spatial inundation HDF5 does not exist: {inundation_h5_path}")
    if not raw_rasters:
        raise FileNotFoundError(f"No target inundation rasters found in {inundation_download_path}")

    temporal_prediction_csv = FilePath(temporal_prediction_csv)
    predictions = pd.read_csv(temporal_prediction_csv)
    interval = int(get_cfg("forecast.interval_percent", 80))
    lower_column = f"lower_bound_{interval}_fraction"
    upper_column = f"upper_bound_{interval}_fraction"
    required = {"date", "predicted_fraction", lower_column, upper_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Temporal prediction CSV is missing columns: {sorted(missing)}")
    predictions = predictions[predictions["predicted_fraction"].notna()].copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    horizon = resolve_forecast_horizon_periods()
    if len(predictions) != horizon:
        raise ValueError(f"Expected {horizon} temporal forecast rows, found {len(predictions)}")

    maps = run_inference_pipeline(inundation_file_path=str(inundation_h5_path))
    if sorted(maps) != list(range(1, horizon + 1)):
        raise RuntimeError(f"Expected spatial leads 1..{horizon}, received {sorted(maps)}")

    latest_raster = raw_rasters[-1]
    inundation_file = latest_raster.name
    _, _, regions_gdf, _ = load_spatial_ref(
        str(latest_raster), catchments_path, str(inundation_download_path), inundation_file,
    )
    catchments = process_inundation_modis.load_shapefile(catchments_path)
    catchments = process_inundation_modis.reproject_to_raster_crs(catchments, str(latest_raster))
    inundation_clipped, _, metas = process_inundation_modis.process_and_clip_rasters(
        [inundation_file], str(inundation_download_path), catchments,
    )
    with h5py.File(inundation_h5_path, "r") as hdf:
        current_extent = np.asarray(hdf["inundation"][-1])

    hospital_points = get_exposure_points(
        get_cfg("spatial_prediction.hospitals_path", "data/maps/exposure/hospitals.csv"), None,
    )
    school_points = get_exposure_points(
        None, get_cfg("spatial_prediction.schools_path", "data/maps/exposure/schools.csv"),
    )
    output_root = temporal_prediction_csv.parent / get_cfg(
        "spatial_prediction.output_subdirectory", "spatial_predictions",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    dpi = int(get_cfg("spatial_prediction.map_dpi", 300))
    outputs = []

    source_transform = rasterio.Affine(*cleaning_utils.MASK_REGIONS_REF_TRANSFORM)
    source_crs = cleaning_utils.MASK_REGIONS_REF_CRS
    for lead, (_, forecast) in enumerate(predictions.iterrows(), start=1):
        forecast_date = forecast["date"].strftime("%Y-%m-%d")
        lead_dir = output_root / f"lead_{lead:02d}_{forecast_date}"
        facilities_dir = lead_dir / "impacted_facilities"
        facilities_dir.mkdir(parents=True, exist_ok=True)
        fractions = {
            "Worst Case": float(np.clip(forecast[upper_column], 0.0, 1.0)),
            "Average Case": float(np.clip(forecast["predicted_fraction"], 0.0, 1.0)),
            "Best Case": float(np.clip(forecast[lower_column], 0.0, 1.0)),
        }
        masks = {name: generate_flood_mask(maps[lead], value)
                 for name, value in fractions.items()}
        title = f"Predicted South Sudan Flood Extent Change: {forecast_date} (lead {lead})"
        figure = plot_flood_change_map(
            masks, current_extent, source_transform, source_crs, regions_gdf,
            inundation_clipped, metas, inundation_file=inundation_file, title=title,
        )
        map_path = lead_dir / "flood_change_map.png"
        figure.savefig(map_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        hospital_path = export_impacted_facilities(
            masks, current_extent, metas, inundation_file, hospital_points,
            facilities_dir, "impacted_hospitals.csv", source_transform, source_crs,
        )
        school_path = export_impacted_facilities(
            masks, current_extent, metas, inundation_file, school_points,
            facilities_dir, "impacted_schools.csv", source_transform, source_crs,
        )
        outputs.extend(path for path in (map_path, hospital_path, school_path) if path is not None)

    logger.info("Generated operational spatial outputs for %d forecast leads in %s",
                horizon, output_root)
    return outputs
