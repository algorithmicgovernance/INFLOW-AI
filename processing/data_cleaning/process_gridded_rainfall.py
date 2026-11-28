# Import system libraries
import os
import glob
import tempfile
from pathlib import Path

# Import cleaning utils
from .. import cleaning_utils
from ..config import get_cfg

# Import statistics
from data.stats import gridded_stats

# Import TAMSAT API
from processing.data_cleaning.download_tamsat.tamsat_download_extract_api import download, extract

# Import data manipulation libraries
import numpy as np
import pandas as pd

# Import geospatial libraries
import rasterio
from rasterio.transform import from_origin
import netCDF4 as nc

# Import compression libraries
import h5py

# Import progress bar libraries
from tqdm import tqdm
from .gridded_streaming import (
    PeriodFiles, build_grid_masks, complete_periods, inventory_daily_files,
    quarantine_invalid_files, resolve_project_path, stream_period_grid,
    sync_product, validate_daily_files,
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOWNLOADS_ROOT = get_cfg("paths.downloads.root", "data/downloads")
GR_RAINFALL_DOWNLOAD_PATH = get_cfg("paths.downloads.tamsat_rfe_daily", "data/downloads/tamsat/rfe/data/v3.1/daily")
EXTRACTED_DOMAIN_PATH = get_cfg("paths.downloads.extracted_domain", "data/downloads/extracted_data/domain")
GR_RAINFALL_DEKADS_PATH = get_cfg("paths.downloads.tamsat_rfe_dekads", "data/downloads/tamsat/rfe/dekads")
GR_RAINFALL_TEMPORAL_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall_temporal.csv")
GR_RAINFALL_H5_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall.h5")
SAMPLE_TIF_FOLDER = get_cfg("paths.downloads.inundation_viirs", "data/downloads/inundation_masks_viirs_v2")
CATCHMENTS_PATH = get_cfg("paths.maps.catchments", "data/maps/inflow_catchments/INFLOW_all_cmts.shp")


def extract_date_from_filename(filename):
    """
    Extract the date from rainfall tif filename,

    Parameters:
        filname (str): Name of file for which the date is extracted.
    """
    return cleaning_utils.extract_date_from_tif_filename(filename)


def download_gridded_rainfall(dates_list, download_path):
    """
    Download gridded rainfall data for the specified dates.

    Parameters:
        dates_list (list): List of dates for which to download gridded rainfall data.
        download_path (str): Directory path to save downloaded NetCDF files.
    """
    try:
        download({
            "product": 'rfe',
            "timestep": 'daily',
            "resolution": 0.0375,
            "start_date": dates_list[0],
            "end_date": dates_list[-1],
            "version": 3.1,
            "localdata_dir": download_path
            })
    except Exception as e:
        print(f"Error occurred while downloading TAMSAT data: {e}")


def extract_gridded_rainfall(dates_list, download_path):
    """
    Extract gridded rainfall data for the specified dates into single file.

    Parameters:
        dates_list (list): List of dates for which to extract gridded rainfall data.
        download_path (str): Directory path to the downloaded NetCDF files.
    """
    try:
        extract({
            "product": 'rfe',
            "extract_type": 'domain',
            "N": 15.837321509670957,
            "S": -4.029166662242848,
            "W": 23.424907051000087,
            "E": 36.30367723700005,
            "timestep": 'daily',
            "resolution": 0.0375,
            "start_date": dates_list[0],
            "end_date": dates_list[-1],
            "version": 3.1,
            "localdata_dir": download_path
            })
    except Exception as e:
        raise RuntimeError(f"Failed to extract gridded rainfall data: {e}") from e


def _validate_rainfall_netcdf_file(netcdf_path):
    """Validate a domain rainfall NetCDF file and raise if unreadable/corrupt."""
    required_variables = {"lat", "lon", "rfe", "time"}
    try:
        with nc.Dataset(netcdf_path, mode='r') as ds:
            missing = required_variables - set(ds.variables.keys())
            if missing:
                raise RuntimeError(f"Missing required NetCDF variables: {sorted(missing)}")

            # Force a lightweight read to surface latent HDF issues early.
            _ = ds.variables['lat'][:1]
            _ = ds.variables['lon'][:1]
            _ = ds.variables['time'][:1]
            _ = ds.variables['rfe'][0:1, :, :]
    except Exception as e:
        raise RuntimeError(f"Invalid rainfall NetCDF file '{netcdf_path}': {e}") from e


def _rainfall_netcdf_date_range(netcdf_path):
    with nc.Dataset(netcdf_path, mode="r") as ds:
        time_var = ds.variables["time"]
        dates = nc.num2date(
            time_var[:], units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False, only_use_python_datetimes=False,
        )
    return pd.Timestamp(dates[0]).normalize(), pd.Timestamp(dates[-1]).normalize()


def _get_valid_rainfall_netcdf(extract_folder, required_start=None, required_end=None):
    """Return a readable rainfall NetCDF covering the requested local archive."""
    list_of_files = sorted(
        glob.glob(os.path.join(os.getcwd(), extract_folder, '*.nc')),
        key=os.path.getctime,
        reverse=True,
    )
    if not list_of_files:
        raise RuntimeError(f"No extracted NetCDF files found in '{extract_folder}'.")

    errors = []
    for path in list_of_files:
        try:
            _validate_rainfall_netcdf_file(path)
            file_start, file_end = _rainfall_netcdf_date_range(path)
            if required_start is not None and file_start > pd.Timestamp(required_start):
                raise RuntimeError(
                    f"coverage starts at {file_start.date()}, after required {pd.Timestamp(required_start).date()}"
                )
            if required_end is not None and file_end < pd.Timestamp(required_end):
                raise RuntimeError(
                    f"coverage ends at {file_end.date()}, before required {pd.Timestamp(required_end).date()}"
                )
            return path
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {e}")

    raise RuntimeError(
        "No valid extracted rainfall NetCDF file found. Validation errors: "
        + " | ".join(errors)
    )


def group_dates_by_decade(dates, target_product='viirs'):
    """
    Group dates into 10-day intervals (dekads).

    Parameters:
        dates (pd.DatetimeIndex): Array of datetime objects.

    Returns:
        tuple: Grouped dates and their indices.
    """
    return cleaning_utils.group_dates_by_target_period(dates, target_product=target_product)


def export_decadal_geotiffs(extract_folder, output_folder, target_product='viirs'):
    """
    Export rainfall data grouped by dekads into GeoTIFF files.

    Parameters:
        extract_folder (str): Path to folder where extracted rainfall data is saved.
        output_folder (str): Path to the folder to save GeoTIFFs.
    """
    # Ensure output directory exists before deleting/writing files.
    os.makedirs(output_folder, exist_ok=True)

    # Use glob to get all file paths in the folder
    files = glob.glob(os.path.join(output_folder, '*'))
    
    # Loop through the files and delete each one
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")
            
    # Select and validate latest extracted gridded rainfall file.
    local_dates = cleaning_utils.get_local_download_dates(
        os.path.join(os.getcwd(), GR_RAINFALL_DOWNLOAD_PATH)
    )
    required_start = min(local_dates) if local_dates else None
    required_end = max(local_dates) if local_dates else None
    latest_file = _get_valid_rainfall_netcdf(extract_folder, required_start, required_end)

    # Stream from NetCDF in small slices so the full time cube is never loaded.
    with nc.Dataset(latest_file, mode='r') as rainfall_grid:
        lats = rainfall_grid.variables['lat'][:]
        lons = rainfall_grid.variables['lon'][:]
        times = rainfall_grid.variables['time'][:]
        rainfall_var = rainfall_grid.variables['rfe']

        time_var = rainfall_grid.variables['time']
        decoded = nc.num2date(
            times, units=time_var.units,
            calendar=getattr(time_var, 'calendar', 'standard'),
            only_use_cftime_datetimes=False, only_use_python_datetimes=False,
        )
        dates = [pd.Timestamp(value).to_pydatetime() for value in decoded]

        print('--- Gridded rainfall data loaded ---')

        # Group the dates into target-product windows.
        date_groups, grouped_indices = group_dates_by_decade(dates, target_product=target_product)

        # Define transform once using the latitude and longitude arrays.
        lon_min = lons.min()
        lat_max = lats.max()
        pixel_size_x = abs(lons[1] - lons[0])
        pixel_size_y = abs(lats[1] - lats[0])
        transform = from_origin(lon_min, lat_max, pixel_size_x, pixel_size_y)

        # Export each target-period group as a GeoTIFF.
        for group, indices in tqdm(zip(date_groups, grouped_indices), total=len(date_groups), desc="Exporting decadal averages"):
            indices = list(indices)
            if not indices:
                continue

            # Incremental mean across slices for low-memory processing.
            sum_2d = None
            for idx in indices:
                arr_2d = np.array(rainfall_var[idx, :, :], dtype=np.float32, copy=False)
                if sum_2d is None:
                    sum_2d = np.zeros_like(arr_2d, dtype=np.float32)
                sum_2d += arr_2d

            decadal_avg = sum_2d / float(len(indices))

            first_date = group[0]
            first_dekad_str = first_date.strftime("%Y%m%d")
            output_file = os.path.join(output_folder, f'rainfall_decadal_{first_dekad_str}.tif')

            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=decadal_avg.shape[0],
                width=decadal_avg.shape[1],
                count=1,
                dtype=decadal_avg.dtype,
                crs='EPSG:4326',
                transform=transform,
            ) as dst:
                dst.write(decadal_avg, 1)

            period_label = f'{target_product.upper()} dekad'
            print(f'Exported {period_label} GeoTIFF for {first_dekad_str}')
        

def crop_historic_data(file_path, temporal_data_path):
    """Validate legacy outputs without deleting either side.

    Kept for compatibility with older callers. New code synchronizes by the
    explicit HDF5 ``dates`` dataset and recreates the CSV from HDF5 if needed.
    """
    hist = pd.read_csv(temporal_data_path)
    with h5py.File(file_path, "r") as hdf:
        if "rainfall" not in hdf:
            raise ValueError(f"Missing rainfall dataset in {file_path}")
        if hdf["rainfall"].shape[0] != len(hist):
            raise ValueError(
                "Rainfall HDF5 and temporal CSV lengths differ; use "
                "update_gridded_rainfall() to repair them safely."
            )


def _date_strings_from_hdf(hdf):
    if "dates" not in hdf:
        return []
    return list(hdf["dates"].asstr()[:])


def _rainfall_temporal_row(date_str, rainfall_2d, regions_gdf):
    total_cells = rainfall_2d.shape[0] * rainfall_2d.shape[1]
    row = {"date": date_str, "rainfall": float(np.nansum(rainfall_2d))}
    rainfall_3d = rainfall_2d[np.newaxis, :, :]
    for i in range(len(regions_gdf)):
        region_data = regions_gdf.iloc[[i]]
        region_code = gridded_stats.region_to_code_dict[region_data["region"].values[0]]
        region_area = cleaning_utils.mask_regions(region_data, rainfall_3d)
        valid_cells = total_cells - np.sum(np.isnan(region_area[0]))
        row[f"rainfall_{region_code}"] = (
            float(np.nansum(region_area)) / valid_cells if valid_cells > 0 else np.nan
        )
    return row


def _write_csv_atomic(frame, temporal_path):
    path = Path(temporal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _temporal_csv_from_hdf(hdf_path, temporal_path, regions_gdf):
    """Recreate temporal statistics from the authoritative HDF5 dates/grids."""
    rows = []
    with h5py.File(hdf_path, "r") as hdf:
        dates = _date_strings_from_hdf(hdf)
        if "rainfall" not in hdf or len(dates) != hdf["rainfall"].shape[0]:
            raise ValueError("Rainfall HDF5 has missing or inconsistent dates")
        for index, date_str in enumerate(tqdm(dates, desc="Rebuilding rainfall temporal CSV")):
            rows.append(_rainfall_temporal_row(date_str, hdf["rainfall"][index], regions_gdf))
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, temporal_path)
    return frame


def _validate_rainfall_outputs(hdf_path, temporal_path, expected_dates=None):
    frame = pd.read_csv(temporal_path)
    if "date" not in frame or frame.empty:
        raise ValueError("Rainfall temporal CSV is empty or missing date")
    csv_dates = pd.to_datetime(frame["date"], errors="coerce")
    if csv_dates.isna().any() or csv_dates.duplicated().any() or not csv_dates.is_monotonic_increasing:
        raise ValueError("Rainfall temporal CSV dates must be valid, unique, and sorted")
    rainfall_columns = [column for column in frame if column.startswith("rainfall")]
    if not rainfall_columns or frame[rainfall_columns].isna().all().any():
        raise ValueError("Rainfall temporal CSV contains an entirely empty rainfall feature")
    with h5py.File(hdf_path, "r") as hdf:
        if "rainfall" not in hdf or "dates" not in hdf:
            raise ValueError("Rainfall HDF5 must contain rainfall and dates datasets")
        hdf_dates = _date_strings_from_hdf(hdf)
        if hdf["rainfall"].shape[0] != len(hdf_dates):
            raise ValueError("Rainfall HDF5 grid/date lengths differ")
    csv_strings = list(csv_dates.dt.strftime("%Y-%m-%d"))
    if hdf_dates != csv_strings:
        raise ValueError("Rainfall HDF5 and temporal CSV dates differ")
    if expected_dates is not None and hdf_dates != list(expected_dates):
        raise ValueError("Rainfall outputs do not contain the expected complete target periods")
    return hdf_dates


def update_gridded_rainfall(
    download_folder=DOWNLOADS_ROOT,
    download_path=GR_RAINFALL_DOWNLOAD_PATH,
    extract_folder=EXTRACTED_DOMAIN_PATH,
    dekads_path=GR_RAINFALL_DEKADS_PATH,
    temporal_data_path=GR_RAINFALL_TEMPORAL_PATH,
    rebuild=False):
    """
    Combine newly downloaded gridded rainfall with existing data.

    Parameters:
        download_folder (str): Directory folder to save downloaded TIF files.
        download_path (str): Directory path to save downloaded TIF files.
        extract_folder (str): Directory folder to extracted TIF files.
        dekads_path (str): Directory path to export dekadal TIF files.
        temporal_data_path (str): Directory path to historic temporal data CSV.
    """
    try:
        target_product = cleaning_utils.resolve_target_product(None)
        daily_root = resolve_project_path(download_path)
        local = inventory_daily_files(daily_root)
        start = cleaning_utils.get_target_start_date(target_product)
        today = pd.Timestamp.today().normalize()
        if not local:
            download_gridded_rainfall(
                [start, today.strftime("%Y-%m-%d")], str(resolve_project_path(download_folder))
            )
        else:
            last = max(local)
            if last < today:
                download_gridded_rainfall(
                    [(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")],
                    str(resolve_project_path(download_folder)),
                )
        local = inventory_daily_files(daily_root)
        periods = complete_periods(local)
        invalid = validate_daily_files(periods, "rfe")
        if invalid:
            invalid_paths = {path for path, _ in invalid}
            invalid_dates = sorted(date for date, path in local.items() if path in invalid_paths)
            for path, reason in invalid:
                logging.error("Unreadable rainfall source %s: %s", path, reason)
            quarantine_invalid_files(invalid)
            if invalid_dates:
                download_gridded_rainfall(
                    [invalid_dates[0].strftime("%Y-%m-%d"), invalid_dates[-1].strftime("%Y-%m-%d")],
                    str(resolve_project_path(download_folder)),
                )
            local = inventory_daily_files(daily_root)
            missing_replacements = [date for date in invalid_dates if date not in local]
            if missing_replacements:
                raise RuntimeError(
                    "Unable to replace unreadable rainfall dates: "
                    + ", ".join(date.strftime("%Y-%m-%d") for date in missing_replacements)
                )
            replacement_errors = validate_daily_files(
                [PeriodFiles(date.strftime("%Y-%m-%d"), (local[date],)) for date in invalid_dates],
                "rfe",
            )
            if replacement_errors:
                raise RuntimeError(f"Replacement rainfall files are unreadable: {replacement_errors[:10]}")
            periods = complete_periods(local)
            remaining = validate_daily_files(periods, "rfe")
            if remaining:
                raise RuntimeError(f"Unreadable rainfall files remain after download attempt: {remaining[:10]}")
        masks = build_grid_masks(CATCHMENTS_PATH)
        status, count = sync_product(
            periods=periods,
            hdf_path=resolve_project_path(GR_RAINFALL_H5_PATH),
            csv_path=resolve_project_path(temporal_data_path),
            dataset_name="rainfall",
            prefix="rainfall",
            target_product=target_product,
            masks=masks,
            loader=lambda period: stream_period_grid(period.files, "rfe", masks),
            rebuild=rebuild,
        )
        logging.info("Rainfall outputs %s (%d periods written).", status, count)

    except Exception as e:
        logging.error(f"Error processing rainfall data: {e}")
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build or update target-aligned gridded rainfall data.")
    parser.add_argument("--rebuild", action="store_true", help="Recreate HDF5 and CSV from downloaded rainfall")
    args = parser.parse_args()
    update_gridded_rainfall(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
