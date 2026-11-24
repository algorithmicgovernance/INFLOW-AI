# Import system libraries
import os
import re
import calendar
import logging
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.parse import urljoin

# Import data manipulation libraries
import numpy as np
import pandas as pd

# Import geospatial libraries
import geopandas as gpd
import rasterio

# Import client libraries
import requests

# Import compression libraries
import h5py

# Import progress bar libraries
from tqdm import tqdm

# Import cleaning utils
from .. import cleaning_utils
from ..config import get_cfg

# Import statistics
from data.stats import gridded_stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
# VIIRS configuration
# -----------------------------------------------------------------------------

VIIRS_BASE_URL = get_cfg(
    "sources.viirs_final_base_url",
    get_cfg(
        "sources.viirs_base_url",
        "https://gws-access.jasmin.ac.uk/public/tamsat/INFLOW/"
        "south-sudan_flood-maps/v2.0/final/",
    ),
)
VIIRS_PRELIM_BASE_URL = get_cfg(
    "sources.viirs_prelim_base_url",
    "https://gws-access.jasmin.ac.uk/public/tamsat/INFLOW/"
    "south-sudan_flood-maps/v2.0/prelim/",
)
VIIRS_PRELIM_CATALOGUE_URL = get_cfg(
    "sources.viirs_prelim_catalogue_url",
    urljoin(VIIRS_PRELIM_BASE_URL, "prelim_versions.csv"),
)

VIIRS_PERIOD_ORDER = {
    "D1": 0,
    "D2": 1,
    "D3": 2,
}

VIIRS_DATASET_VERSION = "2.0"
VIIRS_CADENCE = "dekadal"

VIIRS_DOWNLOAD_PATH = get_cfg("paths.downloads.inundation_viirs", "data/downloads/inundation_masks_viirs")
VIIRS_PRELIM_DOWNLOAD_PATH = get_cfg(
    "paths.downloads.inundation_viirs_prelim",
    "data/downloads/inundation_masks_viirs_prelim",
)
VIIRS_H5_PATH = get_cfg("paths.historic.viirs_h5", "data/historic/inundation_viirs.h5")
VIIRS_TEMPORAL_PATH = get_cfg(
    "paths.historic.viirs_temporal",
    "data/historic/inundation_viirs_temporal.csv",
)
VIIRS_DSET_NAME = "inundation"
INFLOW_CATCHMENTS_PATH = get_cfg(
    "paths.maps.catchments",
    "data/maps/inflow_catchments/INFLOW_all_cmts.shp",
)


def _request_with_retries(url, *, stream=False):
    retries = int(get_cfg("runtime.viirs.request_retries", 3))
    delay_seconds = float(get_cfg("runtime.viirs.retry_delay_seconds", 2))
    timeout = float(get_cfg("runtime.viirs.request_timeout_seconds", 120))
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                stream=stream,
                timeout=(20, timeout),
                headers={"User-Agent": "INFLOW-AI/1.0"},
            )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt < retries:
                time.sleep(delay_seconds * (2 ** (attempt - 1)))
    raise RuntimeError(f"Unable to download VIIRS resource {url}: {last_error}") from last_error


def parse_viirs_filename(file_name):
    """
    Parse a VIIRS inundation filename.

    The v2 archive uses dekadal dates. Dates embedded anywhere in a TIFF
    filename are accepted as YYYY-MM-DD or YYYYMMDD; YYYYMMd1/d2/d3 is also
    supported. Only dekad starts (days 1, 11 and 21) are valid.

    Returns:
        dict with year, month, period_type, period_order, period_id, and date fields.
    """
    basename = os.path.basename(file_name)
    date_match = re.search(
        r"(?<!\d)(?P<year>(?:19|20)\d{2})[-_]?(?P<month>0[1-9]|1[0-2])"
        r"[-_]?(?P<day>0[1-9]|[12]\d|3[01])(?!\d)",
        basename,
        flags=re.IGNORECASE,
    )
    dekad_match = re.search(
        r"(?<!\d)(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])"
        r"[Dd](?P<dekad>[123])(?!\d)",
        basename,
    )
    if date_match:
        year, month, day = (int(date_match.group(key)) for key in ("year", "month", "day"))
    elif dekad_match:
        year = int(dekad_match.group("year"))
        month = int(dekad_match.group("month"))
        day = {1: 1, 2: 11, 3: 21}[int(dekad_match.group("dekad"))]
    else:
        raise ValueError(f"Unrecognized VIIRS filename format: {file_name}")

    if day not in (1, 11, 21):
        raise ValueError(f"VIIRS v2 date is not a dekad start: {file_name}")
    _, last_day = calendar.monthrange(year, month)
    period_order = {1: 0, 11: 1, 21: 2}[day]
    period_type = f"D{period_order + 1}"
    period_start = datetime(year, month, day).date()
    period_end = datetime(year, month, {1: 10, 11: 20, 21: last_day}[day]).date()
    period_id = period_start.isoformat()

    return {
        "year": year,
        "month": month,
        "period_type": period_type,
        "period_order": period_order,
        "period_id": period_id,
        "period_start": period_start,
        "period_end": period_end,
    }


def viirs_target_filename(file_name):
    """
    Convert a remote VIIRS file name to the local target filename.

    Local convention uses period start date:
      FldYYYYMid_MM.tif -> YYYY-MM-01.tif
      FldYYYYShm_MM.tif -> YYYY-MM-16.tif

    The local convention is the dekad start date (YYYY-MM-DD.tif).
    """
    parsed = parse_viirs_filename(file_name)
    return f"{parsed['period_start'].isoformat()}.tif"


def normalize_viirs_identifier(identifier):
    """
    Normalize a VIIRS identifier to local filename convention when possible.
    """
    if not isinstance(identifier, str):
        return identifier

    if not identifier.lower().endswith(".tif"):
        return identifier

    try:
        target = viirs_target_filename(identifier)
        return target if target is not None else identifier
    except ValueError:
        return identifier


def sort_viirs_files(file_names):
    """
    Sort VIIRS files chronologically by year, month, and period type.
    """
    def _sort_key(file_name):
        parsed = parse_viirs_filename(file_name)
        return (parsed["period_start"], parsed["period_order"], file_name)

    return sorted(file_names, key=_sort_key)


def list_remote_tif_files(base_url=VIIRS_BASE_URL, max_retries=3, retry_delay_seconds=2):
    """
    List remote VIIRS TIFFs, retrying transient connection and TLS failures.

    A listing failure raises instead of returning an empty archive. Callers must
    not rebuild historic outputs unless the remote inventory was verified.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                base_url,
                timeout=(20, 60),
                headers={"User-Agent": "INFLOW-AI/1.0"},
            )
            response.raise_for_status()
            break
        except requests.RequestException as error:
            last_error = error
            if attempt < max_retries:
                delay = retry_delay_seconds * (2 ** (attempt - 1))
                logging.warning(
                    "VIIRS listing attempt %d/%d failed: %s; retrying in %ss",
                    attempt, max_retries, error, delay,
                )
                time.sleep(delay)
    else:
        raise RuntimeError(
            f"Unable to verify remote VIIRS archive at {base_url} after "
            f"{max_retries} attempts: {last_error}"
        ) from last_error

    # Try to parse direct links from the HTML directory listing.
    tif_files = re.findall(r'href=["\']?([^"\'>]+\.tif)["\']?', response.text, flags=re.IGNORECASE)
    tif_files = [os.path.basename(f) for f in tif_files if f.lower().endswith(".tif")]

    tif_files = list(dict.fromkeys(os.path.basename(f) for f in tif_files))

    # Keep only files containing a valid v2 dekadal date.
    filtered_files = []
    for file_name in tif_files:
        try:
            parse_viirs_filename(file_name)
            filtered_files.append(file_name)
        except ValueError:
            continue

    if not filtered_files:
        raise RuntimeError(
            f"Remote VIIRS listing at {base_url} contained no valid dekadal TIFF files"
        )
    return sort_viirs_files(filtered_files)


def download_inundation(file_list, download_path=VIIRS_DOWNLOAD_PATH, base_url=VIIRS_BASE_URL):
    """
    Download VIIRS inundation data for the specified files.

    Parameters:
        file_list (list): List of VIIRS TIF file names to download.
        download_path (str): Directory path to save downloaded TIF files.
        base_url (str): Base URL where the TIF files are hosted.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    max_retries = 3
    retry_delay_seconds = 2

    failures = []
    for file_name in tqdm(file_list, desc="Downloading VIIRS inundation data"):
        file_url = urljoin(base_url, file_name)
        target_name = viirs_target_filename(file_name)
        if target_name is None:
            continue
        file_path = os.path.join(download_path, target_name)
        temp_file_path = f"{file_path}.part"

        if os.path.exists(file_path):
            continue

        for attempt in range(1, max_retries + 1):
            try:
                with requests.get(file_url, stream=True, timeout=120) as response:
                    if response.status_code != 200:
                        raise requests.HTTPError(
                            f"HTTP {response.status_code} for {file_name}",
                            response=response,
                        )

                    with open(temp_file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                os.replace(temp_file_path, file_path)
                break

            except Exception as e:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                if attempt == max_retries:
                    logging.error(
                        f"Failed to download {file_name} after {max_retries} attempts: {e}"
                    )
                    failures.append((file_name, str(e)))
                else:
                    logging.warning(
                        f"Retry {attempt}/{max_retries} for {file_name} after error: {e}"
                    )
                    time.sleep(retry_delay_seconds)
    if failures:
        details = "; ".join(f"{name}: {error}" for name, error in failures[:5])
        raise RuntimeError(
            f"Failed to download {len(failures)} VIIRS files; archive is incomplete. {details}"
        )


def get_sorted_tif_files(folder_path):
    """
    Get a sorted list of TIF files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing TIF files.

    Returns:
        list: Sorted list of TIF file names.
    """
    try:
        tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tif")]
        return sort_viirs_files(tif_files)
    except FileNotFoundError:
        logging.error(f"Folder not found: {folder_path}")
        return []


def load_shapefile(path):
    """
    Load a shapefile as a GeoDataFrame.

    Parameters:
        path (str): Path to the shapefile.

    Returns:
        GeoDataFrame: Loaded shapefile as a GeoDataFrame.
    """
    try:
        return gpd.read_file(path)
    except FileNotFoundError:
        logging.error(f"Shapefile not found: {path}")
        return None


def reproject_to_raster_crs(shapefile, raster_path):
    """
    Reproject a GeoDataFrame to match the CRS of a raster file.

    Parameters:
        shapefile (GeoDataFrame): The GeoDataFrame to reproject.
        raster_path (str): Path to a raster file for CRS reference.

    Returns:
        GeoDataFrame: Reprojected GeoDataFrame.
    """
    try:
        with rasterio.open(raster_path) as src:
            return shapefile.to_crs(src.crs)
    except Exception as e:
        logging.error(f"Error in reprojecting shapefile: {e}")
        return None


def process_and_clip_rasters(tif_files, folder_path, catchments):
    """
    Align, mask, and collect metadata for each raster file in a folder.

    Parameters:
        tif_files (list): List of TIF file names.
        folder_path (str): Path to the folder containing TIF files.
        catchments (GeoDataFrame): GeoDataFrame of the catchment areas for clipping.

    Returns:
        tuple: Arrays of aligned rasters, list of file names, and metadata dictionary.
    """
    clipped_tif_files = []
    tif_file_names = []
    spatial_metadata = {}

    for file_name in tqdm(tif_files, desc="Processing TIF files"):
        file_path = os.path.join(folder_path, file_name)

        try:
            with rasterio.open(file_path) as src:
                aligned_masked = cleaning_utils.align_and_mask_raster_to_reference_grid(
                    src=src,
                    mask_gdf=catchments,
                    src_band=1,
                    dst_fill=0,
                )
                # VIIRS masks are binary. Some source footprints cause GDAL to
                # preserve the source nodata value (255) during reprojection;
                # treating any value except the inundated class (1) as dry
                # prevents nodata/burn artifacts from entering area totals.
                aligned_masked = (aligned_masked == 1).astype(np.uint8)

                clipped_tif_files.append(aligned_masked)
                tif_file_names.append(file_name)
                spatial_metadata[file_name] = {
                    "crs": cleaning_utils.MASK_REGIONS_REF_CRS,
                    "transform": cleaning_utils.MASK_REGIONS_REF_TRANSFORM,
                    "height": cleaning_utils.MASK_REGIONS_REF_SHAPE[0],
                    "width": cleaning_utils.MASK_REGIONS_REF_SHAPE[1],
                    "bounds": cleaning_utils.MASK_REGIONS_REF_BOUNDS,
                }
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")
            continue

    return clipped_tif_files, tif_file_names, spatial_metadata


def get_historic_dates(data_path=VIIRS_TEMPORAL_PATH):
    """
    Get list of historic VIIRS records from pre-downloaded data.

    Returns the most useful unique identifier available in the temporal CSV:
    file_name, period_id, or date.
    """
    try:
        inundation_temporal = pd.read_csv(data_path)

        if "file_name" in inundation_temporal.columns:
            return inundation_temporal["file_name"].tolist()
        if "period_id" in inundation_temporal.columns:
            return inundation_temporal["period_id"].tolist()
        if "date" in inundation_temporal.columns:
            return inundation_temporal["date"].astype(str).tolist()

        # Fallback: use the first column or index values if present.
        if inundation_temporal.shape[1] > 0:
            return inundation_temporal.iloc[:, 0].astype(str).tolist()

        return []
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        return []


def validate_local_viirs_v2(h5_file_path=VIIRS_H5_PATH,
                            temporal_data_path=VIIRS_TEMPORAL_PATH,
                            download_path=VIIRS_DOWNLOAD_PATH):
    """Validate that cached v2 TIFF, HDF5, and temporal records agree exactly."""
    if not all(os.path.exists(path) for path in (h5_file_path, temporal_data_path, download_path)):
        return False, "one or more local VIIRS products are missing"
    try:
        with h5py.File(h5_file_path, "r") as hdf:
            if VIIRS_DSET_NAME not in hdf:
                return False, f"HDF5 dataset '{VIIRS_DSET_NAME}' is missing"
            dataset = hdf[VIIRS_DSET_NAME]
            expected_shape = tuple(cleaning_utils.MASK_REGIONS_REF_SHAPE)
            if dataset.shape[1:] != expected_shape:
                return False, f"HDF5 grid {dataset.shape[1:]} differs from {expected_shape}"
            hdf_rows = dataset.shape[0]

        temporal = pd.read_csv(temporal_data_path)
        if "period_start" not in temporal or "file_name" not in temporal:
            return False, "temporal CSV lacks period_start or file_name"
        dates = pd.to_datetime(temporal["period_start"], errors="coerce")
        if dates.isna().any() or not dates.dt.day.isin([1, 11, 21]).all():
            return False, "temporal CSV contains invalid or non-dekadal dates"
        if dates.duplicated().any():
            return False, "temporal CSV contains duplicate periods"

        local_files = {
            normalize_viirs_identifier(name)
            for name in os.listdir(download_path)
            if name.lower().endswith(".tif")
        }
        temporal_files = {
            normalize_viirs_identifier(name) for name in temporal["file_name"].astype(str)
        }
        if hdf_rows != len(temporal) or local_files != temporal_files:
            return False, (
                f"cache counts differ: HDF5={hdf_rows}, CSV={len(temporal)}, "
                f"TIFFs={len(local_files)}"
            )
        return True, f"{hdf_rows} matching dekadal observations"
    except Exception as error:
        return False, str(error)


def download_new_inundation(download_path=VIIRS_DOWNLOAD_PATH, burn_in_steps=18):
    """
    Download VIIRS inundation data for the last `burn_in_steps` files (to refresh them)
    plus any new files available in the remote directory.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        burn_in_steps (int): Number of timesteps to always refresh.
    """
    remote_files = list_remote_tif_files()

    remote_items = [(f, viirs_target_filename(f)) for f in remote_files]
    remote_items = [(src, tgt) for src, tgt in remote_items if tgt is not None]

    local_ids = {
        normalize_viirs_identifier(name)
        for name in os.listdir(download_path)
        if name.lower().endswith(".tif")
    } if os.path.isdir(download_path) else set()
    if local_ids:
        if len(remote_items) >= burn_in_steps:
            refresh_files = [src for src, _ in remote_items[-burn_in_steps:]]
        else:
            refresh_files = [src for src, _ in remote_items]
    else:
        refresh_files = []

    new_files = [src for src, tgt in remote_items if tgt not in local_ids]
    files_to_download = []
    seen = set()
    for file_name in refresh_files + new_files:
        if file_name not in seen:
            files_to_download.append(file_name)
            seen.add(file_name)

    if files_to_download:
        logging.info(
            f"Downloading {len(files_to_download)} VIIRS files "
            f"(including last {burn_in_steps} for refresh)."
        )
        download_inundation(files_to_download, download_path)
    else:
        logging.info("No new VIIRS files to download.")

    expected_local = {target for _, target in remote_items}
    actual_local = {
        normalize_viirs_identifier(name)
        for name in os.listdir(download_path)
        if name.lower().endswith(".tif")
    } if os.path.isdir(download_path) else set()
    missing = sorted(expected_local.difference(actual_local))
    if missing:
        raise RuntimeError(
            f"Local VIIRS v2 cache is missing {len(missing)} files from the verified "
            f"remote archive (first missing: {missing[0]})"
        )
    return expected_local


def crop_historic_data(file_path, temporal_data_path):
    """
    Crop or recreate the historic inundation HDF5 dataset to match the temporal CSV length.
    """

    hist = pd.read_csv(temporal_data_path)
    new_len = len(hist)

    # --- Open HDF5 and check dataset length ---
    with h5py.File(file_path, "r+") as f:
        dset_name = list(f.keys())[0]
        dset = f[dset_name]
        current_len = dset.shape[0]

        # --- If HDF5 shorter, crop CSV to match ---
        if current_len < new_len:
            hist.iloc[:current_len].to_csv(temporal_data_path, index=False)
            print(f"Cropped CSV to {current_len} timesteps.")

        # --- If HDF5 longer, recreate file (truncate + rewrite) ---
        elif current_len > new_len:
            print(f"Cropping HDF5 from {current_len} → {new_len} timesteps...")

            # Read cropped data before removing file
            data = dset[:new_len]
            dtype, shape = dset.dtype, data.shape
            f.close()  # close handle before removing

            # Remove and recreate (truncate)
            os.remove(file_path)
            with h5py.File(file_path, "w") as newf:
                newf.create_dataset(
                    dset_name,
                    data=data,
                    maxshape=(None, *shape[1:]),
                    chunks=True,
                    dtype=dtype,
                )
            print("✅ HDF5 file truncated and recreated with cropped data.")

        else:
            print("✅ No cropping needed. Temporal lengths already match.")


def remove_burn_in_data(h5_file_path=VIIRS_H5_PATH,
                        temporal_data_path=VIIRS_TEMPORAL_PATH,
                        dset_name=VIIRS_DSET_NAME,
                        burn_in_steps=18):
    """
    Remove the last `burn_in_steps` timesteps from saved VIIRS data
    (spatio-temporal HDF5 dataset and temporal CSV).

    Parameters:
        h5_file_path (str): Path to spatio-temporal historic VIIRS HDF5 file.
        temporal_data_path (str): Path to temporal CSV file.
        dset_name (str): Name of dataset inside the HDF5 file.
        burn_in_steps (int): Number of timesteps (along axis 0) to drop from the end.
    """
    import pandas as pd
    import h5py

    # --- Process HDF5 file ---
    with h5py.File(h5_file_path, "r") as f:
        if dset_name not in f:
            raise KeyError(f"Dataset '{dset_name}' not found in {h5_file_path}.")
        data = f[dset_name][:]

    # Crop last axis (remove last `burn_in_steps` entries)
    if data.shape[0] <= burn_in_steps:
        raise ValueError("Not enough timesteps to remove burn-in data.")
    data_cropped = data[:-burn_in_steps]

    # Overwrite file with cropped dataset
    with h5py.File(h5_file_path, "w") as f:
        dset = f.create_dataset(
            dset_name,
            shape=data_cropped.shape,
            maxshape=(None, *data_cropped.shape[1:]),
            chunks=True,
            dtype=data_cropped.dtype,
        )
        dset[:] = data_cropped

    df = pd.read_csv(temporal_data_path)
    if len(df) <= burn_in_steps:
        raise ValueError(f"Not enough rows in {temporal_data_path} to remove burn-in data.")

    df_cropped = df.iloc[:-burn_in_steps].reset_index(drop=True)
    sort_cols = [c for c in ["year", "month", "period_order"] if c in df_cropped.columns]
    if sort_cols:
        df_cropped = df_cropped.sort_values(sort_cols).reset_index(drop=True)
    elif "date" in df_cropped.columns:
        df_cropped["date"] = pd.to_datetime(df_cropped["date"])
        df_cropped = df_cropped.sort_values("date").reset_index(drop=True)
    df_cropped.to_csv(temporal_data_path, index=False)

    print(f"Removed last {burn_in_steps} timesteps from HDF5 and temporal CSV.")


def build_viirs_temporal_dataframe(file_names, clipped_rasters, regions_gdf=None):
    """
    Build temporal dataframe from VIIRS rasters.
    """
    if len(clipped_rasters) == 0:
        raise ValueError("No clipped VIIRS rasters were provided.")

    parsed_records = [parse_viirs_filename(file_name) for file_name in file_names]
    total_cells = clipped_rasters[0].shape[0] * clipped_rasters[0].shape[1]

    inundation_temporal = pd.DataFrame({
        "file_name": file_names,
        "period_id": [rec["period_id"] for rec in parsed_records],
        "year": [rec["year"] for rec in parsed_records],
        "month": [rec["month"] for rec in parsed_records],
        "period_type": [rec["period_type"] for rec in parsed_records],
        "period_order": [rec["period_order"] for rec in parsed_records],
        "period_start": [rec["period_start"] for rec in parsed_records],
        "period_end": [rec["period_end"] for rec in parsed_records],
        "percent_inundation": np.sum(clipped_rasters, axis=(1, 2)) / total_cells,
    })

    if regions_gdf is not None and len(regions_gdf) > 0:
        for i in range(len(regions_gdf)):
            region_data = regions_gdf.iloc[[i]]
            region_code = gridded_stats.region_to_code_dict[region_data["region"].values[0]]
            region_area = cleaning_utils.mask_regions(region_data, np.array(clipped_rasters))
            region_series = np.nansum(region_area, axis=(1, 2)) / (
                total_cells - np.sum(np.isnan(region_area[0]))
            )
            inundation_temporal[f"percent_inundation_{region_code}"] = region_series

    sort_cols = [c for c in ["year", "month", "period_order"] if c in inundation_temporal.columns]
    if sort_cols:
        inundation_temporal = inundation_temporal.sort_values(sort_cols).reset_index(drop=True)

    return inundation_temporal


def _viirs_observation_dates(dataframe):
    """Return the observation dates used by the VIIRS temporal dataframe."""
    if "period_start" in dataframe.columns:
        return pd.to_datetime(dataframe["period_start"], errors="coerce")
    if "date" in dataframe.columns:
        return pd.to_datetime(dataframe["date"], errors="coerce")
    if {"year", "month"}.issubset(dataframe.columns):
        period_order = dataframe.get(
            "period_order",
            pd.Series(0, index=dataframe.index),
        )
        days = period_order.map({0: 1, 1: 11, 2: 21}).fillna(1).astype(int)
        return pd.to_datetime({
            "year": dataframe["year"],
            "month": dataframe["month"],
            "day": days,
        }, errors="coerce")
    raise ValueError("VIIRS temporal data must contain period_start, date, or year/month columns.")


def _remove_drying_period_increases(series, dates):
    """Interpolate anomalous increases within each December-April drying season."""
    cleaned = pd.to_numeric(series, errors="coerce").copy()
    dates = pd.to_datetime(dates, errors="coerce")

    # December belongs to the drying season ending in the following calendar year.
    season_year = dates.dt.year + (dates.dt.month == 12).astype(int)
    drying = dates.dt.month.isin([12, 1, 2, 3, 4])

    for year in season_year[drying & dates.notna()].unique():
        season_positions = np.flatnonzero((drying & season_year.eq(year)).to_numpy())
        if not len(season_positions):
            continue

        first = season_positions[0]
        preceding = np.flatnonzero(cleaned.iloc[:first].notna().to_numpy())
        if len(preceding):
            anchor = preceding[-1]
            cursor = 0
        else:
            # The historic record may begin part-way through a drying season.
            # In that case, treat its first observation as the initial anchor.
            anchor = first
            cursor = 1

        while cursor < len(season_positions):
            position = season_positions[cursor]
            value = cleaned.iloc[position]
            anchor_value = cleaned.iloc[anchor]
            if pd.isna(value):
                cursor += 1
                continue
            if value <= anchor_value:
                anchor = position
                cursor += 1
                continue

            # The increasing observations are artifacts. The recovery anchor
            # may fall just after April, so search all subsequent observations
            # for the first value that returns to or below the last valid one.
            endpoint = None
            for candidate in range(position + 1, len(cleaned)):
                candidate_value = cleaned.iloc[candidate]
                if pd.notna(candidate_value) and candidate_value <= anchor_value:
                    endpoint = candidate
                    break
            if endpoint is None:
                break

            span = endpoint - anchor
            end_value = cleaned.iloc[endpoint]
            for interpolated_position in range(anchor + 1, endpoint):
                fraction = (interpolated_position - anchor) / span
                cleaned.iloc[interpolated_position] = anchor_value + fraction * (
                    end_value - anchor_value
                )
            anchor = endpoint
            cursor = np.searchsorted(season_positions, endpoint, side="right")

    return cleaned


def clean_viirs_temporal_dataframe(dataframe):
    """Smooth and remove drying-season artifacts from VIIRS inundation series."""
    cleaned = dataframe.copy()
    dates = _viirs_observation_dates(cleaned)
    value_columns = [column for column in cleaned if column.startswith("percent_inundation")]

    for column in value_columns:
        smoothed = pd.to_numeric(cleaned[column], errors="coerce").rolling(
            window=3,
            center=True,
            min_periods=1,
        ).median()
        cleaned[column] = _remove_drying_period_increases(smoothed, dates)

    return cleaned


def rebuild_viirs_history(download_path=VIIRS_DOWNLOAD_PATH,
                          h5_file_path=VIIRS_H5_PATH,
                          temporal_data_path=VIIRS_TEMPORAL_PATH):
    """Rebuild the complete VIIRS HDF5 and temporal CSV from local TIFF files."""
    tif_files = get_sorted_tif_files(download_path)
    if not tif_files:
        raise FileNotFoundError(f"No VIIRS TIFF files found in {download_path}.")

    catchments = load_shapefile(INFLOW_CATCHMENTS_PATH)
    if catchments is None or catchments.empty:
        raise ValueError(f"No catchment geometries found in {INFLOW_CATCHMENTS_PATH}.")

    clipped_rasters, file_names, _ = process_and_clip_rasters(
        tif_files,
        download_path,
        catchments,
    )
    if len(clipped_rasters) != len(tif_files):
        raise RuntimeError(
            f"Processed {len(clipped_rasters)} of {len(tif_files)} VIIRS TIFF files; "
            "refusing to replace historic outputs with an incomplete rebuild."
        )

    regions_gdf = cleaning_utils.extract_regions()
    temporal = build_viirs_temporal_dataframe(
        file_names,
        clipped_rasters,
        regions_gdf=regions_gdf,
    )
    temporal = clean_viirs_temporal_dataframe(temporal)

    os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(temporal_data_path), exist_ok=True)
    temp_h5_path = f"{h5_file_path}.rebuild"
    temp_csv_path = f"{temporal_data_path}.rebuild"
    try:
        with h5py.File(temp_h5_path, "w") as hdf:
            dataset = hdf.create_dataset(
                VIIRS_DSET_NAME,
                data=np.asarray(clipped_rasters, dtype=np.uint8),
                maxshape=(None, *clipped_rasters[0].shape),
                chunks=True,
            )
            dataset.attrs["dataset_version"] = VIIRS_DATASET_VERSION
            dataset.attrs["cadence"] = VIIRS_CADENCE
            dataset.attrs["source_url"] = VIIRS_BASE_URL
        temporal.to_csv(temp_csv_path, index=False)
        os.replace(temp_h5_path, h5_file_path)
        os.replace(temp_csv_path, temporal_data_path)
    finally:
        for temp_path in (temp_h5_path, temp_csv_path):
            if os.path.exists(temp_path):
                os.remove(temp_path)

    logging.info(
        "Rebuilt VIIRS history: %d observations written to %s and %s.",
        len(temporal),
        h5_file_path,
        temporal_data_path,
    )
    return temporal


def update_inundation(h5_file_path=VIIRS_H5_PATH,
                      download_path=VIIRS_DOWNLOAD_PATH,
                      temporal_data_path=VIIRS_TEMPORAL_PATH,
                      allow_offline_existing=False):
    """
    Process newly downloaded VIIRS data and combine it with existing data.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        temporal_data_path (str): Directory path of pre-downloaded temporal data.
        temporal_data_path (str): Directory path of pre-downloaded temporal data.
    """
    try:
        has_existing_h5 = False
        requires_v2_rebuild = False
        if os.path.exists(h5_file_path):
            with h5py.File(h5_file_path, 'r') as f:
                if VIIRS_DSET_NAME in f:
                    viirs_historic = f[VIIRS_DSET_NAME]
                    logging.info(f"Existing VIIRS data shape: {viirs_historic.shape}")
                    has_existing_h5 = True
                    requires_v2_rebuild = (
                        viirs_historic.attrs.get("dataset_version") != VIIRS_DATASET_VERSION
                        or viirs_historic.attrs.get("cadence") != VIIRS_CADENCE
                    )

        # Download before migration so the replacement is built only after the
        # complete remote v2 archive is available locally.
        try:
            verified_remote_files = download_new_inundation(download_path)
        except RuntimeError as error:
            valid, details = validate_local_viirs_v2(
                h5_file_path=h5_file_path,
                temporal_data_path=temporal_data_path,
                download_path=download_path,
            )
            if not allow_offline_existing or not valid:
                raise
            logging.warning(
                "Remote VIIRS update unavailable; using validated local cache (%s). "
                "No freshness check was performed. Cause: %s",
                details,
                error,
            )
            with h5py.File(h5_file_path, "r+") as hdf:
                dataset = hdf[VIIRS_DSET_NAME]
                dataset.attrs["dataset_version"] = VIIRS_DATASET_VERSION
                dataset.attrs["cadence"] = VIIRS_CADENCE
                dataset.attrs["source_url"] = VIIRS_BASE_URL
            return
        if requires_v2_rebuild:
            logging.info(
                "Replacing legacy VIIRS history from %d verified v2 dekadal files.",
                len(verified_remote_files),
            )
            rebuild_viirs_history(download_path, h5_file_path, temporal_data_path)
            return

        if has_existing_h5 and os.path.exists(temporal_data_path):
            # Crop historic data if historic spatial and temporal data are not the same size
            crop_historic_data(
                file_path=h5_file_path,
                temporal_data_path=temporal_data_path,
            )

            # Remove burn-in data
            remove_burn_in_data(
                h5_file_path=h5_file_path,
                temporal_data_path=temporal_data_path,
            )
        elif has_existing_h5:
            logging.info(
                f"VIIRS temporal data not found at {temporal_data_path}; "
                "skipping historic alignment and burn-in removal."
            )
        else:
            logging.info(
                f"Historic VIIRS dataset not found at {h5_file_path}. "
                "Bootstrapping dataset from newly processed files."
            )

        # Get the sorted TIF files after the update
        sorted_files = get_sorted_tif_files(download_path)
        historic_ids = {normalize_viirs_identifier(x) for x in get_historic_dates()}

        # Identify new files to process (files not already processed)
        new_files = [file_name for file_name in sorted_files if file_name not in historic_ids]

        if not new_files:
            logging.info("No new files to process.")
            return

        # Process the new TIF files
        catchments = load_shapefile(INFLOW_CATCHMENTS_PATH)

        # Process rasters and gather new data
        new_clipped_tif_files, new_file_names, _ = process_and_clip_rasters(new_files, download_path, catchments)

        # Crop area to regions of interest
        regions_gdf = cleaning_utils.extract_regions()

        # Build temporal dataframes
        viirs_temporal = build_viirs_temporal_dataframe(
            new_file_names,
            new_clipped_tif_files,
            regions_gdf=regions_gdf
        )

        # Combine existing and new VIIRS data
        os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)
        with h5py.File(h5_file_path, 'a') as hdf:
            if VIIRS_DSET_NAME in hdf:
                dset = hdf[VIIRS_DSET_NAME]
                old_dataset_length = dset.shape[0]
                dset.resize(dset.shape[0] + len(new_clipped_tif_files), axis=0)
                dset[-len(new_clipped_tif_files):] = new_clipped_tif_files
            else:
                old_dataset_length = 0
                dset = hdf.create_dataset(
                    VIIRS_DSET_NAME,
                    data=np.array(new_clipped_tif_files),
                    maxshape=(None, *np.array(new_clipped_tif_files).shape[1:]),
                    chunks=True,
                )
                dset.attrs["dataset_version"] = VIIRS_DATASET_VERSION
                dset.attrs["cadence"] = VIIRS_CADENCE
                dset.attrs["source_url"] = VIIRS_BASE_URL
            logging.info(f"Updated VIIRS data shape: {dset.shape}")

        # Update temporal data
        if old_dataset_length > 0 and os.path.exists(temporal_data_path):
            viirs_temporal_historic = pd.read_csv(temporal_data_path)[:old_dataset_length]
        else:
            viirs_temporal_historic = pd.DataFrame(columns=viirs_temporal.columns)

        viirs_temporal_new = pd.concat([viirs_temporal_historic, viirs_temporal], ignore_index=True)

        # Save the updated temporal data
        sort_cols = [c for c in ["year", "month", "period_order"] if c in viirs_temporal_new.columns]
        if sort_cols:
            viirs_temporal_new = viirs_temporal_new.sort_values(sort_cols).reset_index(drop=True)
        elif "date" in viirs_temporal_new.columns:
            viirs_temporal_new["date"] = pd.to_datetime(viirs_temporal_new["date"], errors="coerce")
            viirs_temporal_new = viirs_temporal_new.sort_values("date").reset_index(drop=True)

        viirs_temporal_new = clean_viirs_temporal_dataframe(viirs_temporal_new)
        os.makedirs(os.path.dirname(temporal_data_path), exist_ok=True)
        viirs_temporal_new.to_csv(temporal_data_path, index=False)

    except Exception as e:
        logging.error(f"Error processing new VIIRS data: {e}")
        raise

    logging.info("VIIRS inundation processing complete.\\n")


def _latest_preliminary_catalogue():
    """Return the latest published preliminary version for every dekad."""
    response = _request_with_retries(VIIRS_PRELIM_CATALOGUE_URL)
    catalogue = pd.read_csv(StringIO(response.text))
    required = {"dekad", "version", "file"}
    missing = required.difference(catalogue.columns)
    if missing:
        raise ValueError(f"VIIRS preliminary catalogue is missing {sorted(missing)}")
    catalogue["dekad"] = pd.to_datetime(catalogue["dekad"], errors="coerce")
    catalogue["version"] = pd.to_numeric(catalogue["version"], errors="coerce")
    catalogue = catalogue.dropna(subset=["dekad", "version", "file"])
    latest = catalogue.sort_values(["dekad", "version"]).drop_duplicates("dekad", keep="last")
    return latest.sort_values("dekad").reset_index(drop=True)


def _download_latest_preliminary(rows, download_path):
    folder = Path(download_path)
    folder.mkdir(parents=True, exist_ok=True)
    local_files = []
    for row in rows.itertuples(index=False):
        remote_basename = Path(str(row.file)).name
        destination = folder / remote_basename
        if not destination.exists() or destination.stat().st_size == 0:
            response = _request_with_retries(
                urljoin(VIIRS_PRELIM_BASE_URL, remote_basename), stream=True
            )
            partial = destination.with_suffix(destination.suffix + ".part")
            try:
                with partial.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
                if partial.stat().st_size == 0:
                    raise RuntimeError(f"Empty VIIRS preliminary download: {remote_basename}")
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
        local_files.append(destination.name)
    return local_files


def prepare_operational_inundation(
    historic_h5_path=VIIRS_H5_PATH,
    historic_temporal_path=VIIRS_TEMPORAL_PATH,
    prelim_download_path=VIIRS_PRELIM_DOWNLOAD_PATH,
):
    """Build a final-history plus latest-preliminary operational VIIRS view.

    The immutable historical HDF5/CSV are never modified here. Preliminary
    maps whose dekads already exist in the final archive are excluded.
    """
    operational_h5 = Path(get_cfg(
        "paths.operational.viirs_h5", "data/operational/inundation_viirs.h5"
    ))
    prelim_h5 = Path(get_cfg(
        "paths.operational.viirs_prelim_h5", "data/operational/inundation_viirs_prelim.h5"
    ))
    operational_csv = Path(get_cfg(
        "paths.operational.viirs_temporal", "data/operational/inundation_viirs_temporal.csv"
    ))
    historic_h5 = Path(historic_h5_path).resolve()
    historic_csv = Path(historic_temporal_path)
    if not historic_h5.exists() or not historic_csv.exists():
        raise FileNotFoundError(
            "Final VIIRS HDF5 and temporal CSV must exist before operational overlay"
        )

    final_temporal = pd.read_csv(historic_csv)
    final_dates = set(pd.to_datetime(_viirs_observation_dates(final_temporal)).dt.normalize())
    latest = _latest_preliminary_catalogue()
    selected = latest[~latest["dekad"].dt.normalize().isin(final_dates)].copy()
    if selected.empty:
        logging.info("No non-final VIIRS preliminary dekads are currently available.")
        return str(historic_h5), str(historic_csv), str(VIIRS_DOWNLOAD_PATH)

    local_files = _download_latest_preliminary(selected, prelim_download_path)
    catchments = load_shapefile(INFLOW_CATCHMENTS_PATH)
    if catchments is None or catchments.empty:
        raise ValueError(f"No catchment geometries found in {INFLOW_CATCHMENTS_PATH}")
    arrays, names, _ = process_and_clip_rasters(
        local_files, prelim_download_path, catchments
    )
    if len(arrays) != len(local_files):
        raise RuntimeError("One or more operational VIIRS preliminary maps could not be processed")

    prelim_temporal = build_viirs_temporal_dataframe(
        names, arrays, regions_gdf=cleaning_utils.extract_regions()
    )
    prelim_temporal["map_status"] = "preliminary"
    version_by_date = dict(zip(
        selected["dekad"].dt.normalize(), selected["version"].astype(int)
    ))
    prelim_temporal["preliminary_version"] = pd.to_datetime(
        prelim_temporal["period_start"]
    ).dt.normalize().map(version_by_date)
    final_temporal["map_status"] = "final"
    final_temporal["preliminary_version"] = np.nan
    combined = pd.concat([final_temporal, prelim_temporal], ignore_index=True, sort=False)
    combined = combined.sort_values(["year", "month", "period_order"]).reset_index(drop=True)

    operational_h5.parent.mkdir(parents=True, exist_ok=True)
    operational_csv.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(prelim_h5, "w") as output:
        output.create_dataset(
            VIIRS_DSET_NAME, data=np.asarray(arrays, dtype=np.uint8), chunks=True
        )
    with h5py.File(historic_h5, "r") as source:
        historic_shape = source[VIIRS_DSET_NAME].shape
        historic_dtype = source[VIIRS_DSET_NAME].dtype
    layout = h5py.VirtualLayout(
        shape=(historic_shape[0] + len(arrays), *historic_shape[1:]),
        dtype=historic_dtype,
    )
    layout[:historic_shape[0]] = h5py.VirtualSource(
        str(historic_h5), VIIRS_DSET_NAME, shape=historic_shape
    )
    prelim_shape = (len(arrays), *historic_shape[1:])
    layout[historic_shape[0]:] = h5py.VirtualSource(
        str(prelim_h5.resolve()), VIIRS_DSET_NAME, shape=prelim_shape
    )
    with h5py.File(operational_h5, "w", libver="latest") as output:
        dataset = output.create_virtual_dataset(VIIRS_DSET_NAME, layout, fillvalue=0)
        dataset.attrs["historical_source"] = str(historic_h5)
        dataset.attrs["preliminary_source"] = str(prelim_h5.resolve())
        dataset.attrs["preliminary_rows"] = len(arrays)
    combined.to_csv(operational_csv, index=False)
    logging.info(
        "Operational VIIRS view contains %d final and %d latest preliminary dekads.",
        historic_shape[0], len(arrays),
    )
    return str(operational_h5), str(operational_csv), str(Path(prelim_download_path))
