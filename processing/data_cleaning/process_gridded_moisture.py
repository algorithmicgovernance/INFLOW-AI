import os
import glob

from .. import cleaning_utils
from ..config import get_cfg

from data.stats import gridded_stats
from processing.data_cleaning.download_tamsat.tamsat_download_extract_api import download, extract

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import netCDF4 as nc

import h5py

from tqdm import tqdm
from .gridded_streaming import (
	PeriodFiles, build_grid_masks, complete_periods, inventory_daily_files,
	quarantine_invalid_files, resolve_project_path, stream_period_grid,
	sync_product, validate_daily_files,
)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOWNLOADS_ROOT = get_cfg("paths.downloads.root", "data/downloads")
GR_MOISTURE_DOWNLOAD_PATH = get_cfg("paths.downloads.tamsat_sm_daily", "data/downloads/tamsat/soil_moisture/data/v2.3.1/daily")
EXTRACTED_DOMAIN_PATH = get_cfg("paths.downloads.extracted_domain", "data/downloads/extracted_data/domain")
GR_MOISTURE_DEKADS_PATH = get_cfg("paths.downloads.tamsat_sm_dekads", "data/downloads/tamsat/soil_moisture/dekads")
GR_MOISTURE_TEMPORAL_PATH = cleaning_utils.get_target_historic_path("gridded_moisture_temporal.csv")
GR_MOISTURE_H5_PATH = cleaning_utils.get_target_historic_path("gridded_moisture.h5")
CATCHMENTS_PATH = get_cfg("paths.maps.catchments", "data/maps/inflow_catchments/INFLOW_all_cmts.shp")
TAMSAT_SM_VERSION = str(get_cfg("runtime.tamsat.sm_version", "2.3.1"))


def reproject_moisture(moisture_ds, target_crs, target_transform, target_width, target_height):
	"""Reproject and resample moisture data."""
	moisture_data = moisture_ds.read(1)
	moisture_transform = moisture_ds.transform
	moisture_crs = moisture_ds.crs

	reprojected_moisture = np.empty((target_height, target_width), dtype=moisture_data.dtype)
	reproject(
		source=moisture_data,
		destination=reprojected_moisture,
		src_transform=moisture_transform,
		src_crs=moisture_crs,
		dst_transform=target_transform,
		dst_crs=target_crs,
		resampling=Resampling.bilinear,
	)
	return reprojected_moisture


def extract_date_from_filename(filename):
	"""Extract date from moisture tif filename."""
	return cleaning_utils.extract_date_from_tif_filename(filename)


def download_gridded_moisture(dates_list, download_path):
	"""Download gridded moisture data for the specified dates."""
	try:
		download(
			{
				"product": "sm",
				"timestep": "daily",
				"resolution": 0.25,
				"start_date": dates_list[0],
				"end_date": dates_list[-1],
				"version": TAMSAT_SM_VERSION,
				"localdata_dir": download_path,
			}
		)
	except Exception as e:
		print(f"Error occurred while downloading TAMSAT data: {e}")


def extract_gridded_moisture(dates_list, download_path):
	"""Extract gridded moisture data for the specified dates into one domain file."""
	try:
		extract(
			{
				"product": "sm",
				"extract_type": "domain",
				"N": 15.837321509670957,
				"S": -4.029166662242848,
				"W": 23.424907051000087,
				"E": 36.30367723700005,
				"timestep": "daily",
				"resolution": 0.25,
				"start_date": dates_list[0],
				"end_date": dates_list[-1],
				"version": TAMSAT_SM_VERSION,
				"localdata_dir": download_path,
			}
		)
	except Exception as e:
		raise RuntimeError(f"Failed to extract gridded moisture data: {e}") from e


def _validate_moisture_netcdf_file(netcdf_path):
	"""Validate a domain moisture NetCDF file and raise if unreadable/corrupt."""
	required_variables = {"lat", "lon", "sm_c4grass", "time"}
	try:
		with nc.Dataset(netcdf_path, mode="r") as ds:
			missing = required_variables - set(ds.variables.keys())
			if missing:
				raise RuntimeError(f"Missing required NetCDF variables: {sorted(missing)}")

			_ = ds.variables["lat"][:1]
			_ = ds.variables["lon"][:1]
			_ = ds.variables["time"][:1]
			_ = ds.variables["sm_c4grass"][0:1, :, :]
	except Exception as e:
		raise RuntimeError(f"Invalid moisture NetCDF file '{netcdf_path}': {e}") from e


def _moisture_netcdf_date_range(netcdf_path):
	with nc.Dataset(netcdf_path, mode="r") as ds:
		time_var = ds.variables["time"]
		dates = nc.num2date(
			time_var[:],
			units=time_var.units,
			calendar=getattr(time_var, "calendar", "standard"),
			only_use_cftime_datetimes=False,
			only_use_python_datetimes=False,
		)
	return pd.Timestamp(dates[0]).normalize(), pd.Timestamp(dates[-1]).normalize()


def _get_latest_valid_moisture_netcdf(extract_folder, required_start=None, required_end=None):
	"""Return most recent readable moisture NetCDF covering the required range."""
	list_of_files = sorted(
		glob.glob(os.path.join(os.getcwd(), extract_folder, "*.nc")),
		key=os.path.getctime,
		reverse=True,
	)
	if not list_of_files:
		raise RuntimeError(f"No extracted NetCDF files found in '{extract_folder}'.")

	errors = []
	for path in list_of_files:
		try:
			_validate_moisture_netcdf_file(path)
			file_start, file_end = _moisture_netcdf_date_range(path)
			if required_start is not None and file_start > pd.Timestamp(required_start):
				raise RuntimeError(f"coverage starts at {file_start.date()}, before required {pd.Timestamp(required_start).date()}")
			if required_end is not None and file_end < pd.Timestamp(required_end):
				raise RuntimeError(f"coverage ends at {file_end.date()}, before required {pd.Timestamp(required_end).date()}")
			return path
		except Exception as e:
			errors.append(f"{os.path.basename(path)}: {e}")

	raise RuntimeError(
		"No valid extracted moisture NetCDF file found. Validation errors: " + " | ".join(errors)
	)


def get_historic_dates(data_path=GR_MOISTURE_TEMPORAL_PATH):
	"""Get list of historic dates from temporal moisture data."""
	try:
		gridded_moisture_temporal = pd.read_csv(data_path)
		if "date" in gridded_moisture_temporal.columns:
			historic_dates = gridded_moisture_temporal["date"].astype(str).tolist()
		else:
			historic_dates = gridded_moisture_temporal.index.astype(str).tolist()
		return historic_dates
	except FileNotFoundError:
		logging.info(f"Historic moisture temporal data not found at {data_path}; bootstrapping.")
		return []


def download_new_gridded_moisture(download_folder, target_product=None):
	"""Download gridded moisture for dates not already downloaded."""
	target_product = cleaning_utils.resolve_target_product(target_product)
	download_path_full = os.path.join(os.getcwd(), download_folder)
	current_date_str = datetime.now().strftime("%Y-%m-%d")
	historic_dates = get_historic_dates()
	has_historic_temporal = bool(historic_dates)

	if has_historic_temporal:
		last_date = datetime.strptime(historic_dates[-1], "%Y-%m-%d").strftime("%Y-%m-%d")
	else:
		last_date = cleaning_utils.get_target_start_date(target_product=target_product)

	new_dates = cleaning_utils.get_dates_of_interest(
		start_date_str=last_date,
		end_date_str=current_date_str,
		target_product=target_product,
	)

	local_product_path = os.path.join(os.getcwd(), GR_MOISTURE_DOWNLOAD_PATH)
	local_dates = cleaning_utils.get_local_download_dates(local_product_path)
	missing_dates = [d for d in new_dates if d not in local_dates]
	required_start = min(local_dates) if local_dates else last_date
	required_end = max(local_dates) if local_dates else current_date_str

	if missing_dates:
		download_range = [missing_dates[0], missing_dates[-1]]
		download_gridded_moisture(download_range, download_path_full)
		# Extraction must include the full cached archive, not only newly missing days.
		extract_range = [required_start, required_end] if local_dates else download_range
		extract_gridded_moisture(extract_range, download_folder)
		_get_latest_valid_moisture_netcdf(EXTRACTED_DOMAIN_PATH, required_start, required_end)
	else:
		try:
			_get_latest_valid_moisture_netcdf(EXTRACTED_DOMAIN_PATH, required_start, required_end)
		except RuntimeError:
			if local_dates:
				extract_gridded_moisture([min(local_dates), max(local_dates)], download_folder)
				_get_latest_valid_moisture_netcdf(EXTRACTED_DOMAIN_PATH, required_start, required_end)
			else:
				raise


def group_dates_by_target_period(dates, target_product="modis"):
	"""Group dates into target-product windows."""
	return cleaning_utils.group_dates_by_target_period(dates, target_product=target_product)


def export_decadal_geotiffs(extract_folder, output_folder, target_product="modis"):
	"""Export moisture grouped by target period into GeoTIFF files."""
	os.makedirs(output_folder, exist_ok=True)

	files = glob.glob(os.path.join(output_folder, "*"))
	for file in files:
		try:
			os.remove(file)
		except Exception as e:
			print(f"Error deleting {file}: {e}")

	latest_file = _get_latest_valid_moisture_netcdf(extract_folder)

	# Stream from NetCDF in small slices so the full time cube is never loaded.
	with nc.Dataset(latest_file, mode="r") as moisture_grid:
		lats = moisture_grid.variables["lat"][:]
		lons = moisture_grid.variables["lon"][:]
		times = moisture_grid.variables["time"][:]
		moisture_var = moisture_grid.variables["sm_c4grass"]

		first_date = datetime.strptime(latest_file[-24:-14], "%Y-%m-%d")
		dates = [(first_date + timedelta(days=int(i))) for i in times]

		print("--- Gridded moisture data loaded ---")

		date_groups, grouped_indices = group_dates_by_target_period(dates, target_product=target_product)

		lon_min = lons.min()
		lat_max = lats.max()
		pixel_size_x = abs(lons[1] - lons[0])
		pixel_size_y = abs(lats[1] - lats[0])
		transform = from_origin(lon_min, lat_max, pixel_size_x, pixel_size_y)

		for group, indices in tqdm(
			zip(date_groups, grouped_indices),
			total=len(date_groups),
			desc="Exporting decadal averages",
		):
			indices = list(indices)
			if not indices:
				continue

			sum_2d = None
			for idx in indices:
				arr_2d = np.array(moisture_var[idx, :, :], dtype=np.float32, copy=False)
				if sum_2d is None:
					sum_2d = np.zeros_like(arr_2d, dtype=np.float32)
				sum_2d += arr_2d

			decadal_avg = sum_2d / float(len(indices))
			if lats[0] < lats[-1]:
				decadal_avg = np.flipud(decadal_avg)

			first_dekad_str = group[0].strftime("%Y%m%d")
			output_file = os.path.join(output_folder, f"moisture_decadal_{first_dekad_str}.tif")

			with rasterio.open(
				output_file,
				"w",
				driver="GTiff",
				height=decadal_avg.shape[0],
				width=decadal_avg.shape[1],
				count=1,
				dtype=decadal_avg.dtype,
				crs="EPSG:4326",
				transform=transform,
			) as dst:
				dst.write(decadal_avg, 1)

			period_label = f"{target_product.upper()} dekad"
			print(f"Exported {period_label} GeoTIFF for {first_dekad_str}")


def crop_historic_data(file_path, temporal_data_path):
	"""Crop temporal CSV length if HDF5 is shorter; rebuild HDF5 if longer."""
	hist = pd.read_csv(temporal_data_path)
	new_len = len(hist)

	with h5py.File(file_path, "r+") as f:
		dset_name = list(f.keys())[0]
		dset = f[dset_name]
		current_len = dset.shape[0]

		if current_len < new_len:
			hist.iloc[:current_len].to_csv(temporal_data_path, index=False)

		elif current_len > new_len:
			print(f"Cropping from {current_len} to {new_len} timesteps...")
			cropped = dset[:new_len]
			f.close()
			os.remove(file_path)

			with h5py.File(file_path, "w") as newf:
				newf.create_dataset(
					dset_name,
					data=cropped,
					maxshape=(None, *dset.shape[1:]),
					chunks=True,
					dtype=dset.dtype,
				)
			print("File truncated and recreated with cropped data.")
		else:
			print("No cropping needed. Temporal lengths already match.")


def process_new_gridded_moisture(moisture_dekads_folder):
	"""Create date-aligned moisture tif list for downstream streaming processing."""
	moisture_dekads_files = [f for f in os.listdir(moisture_dekads_folder) if f.endswith(".tif") and not f.endswith("(1).tif")]

	if not moisture_dekads_files:
		logging.info("No grouped moisture GeoTIFFs available to process yet.")
		return [], []

	moisture_dekads_files_new = glob.glob(os.path.join(moisture_dekads_folder, "*"))

	moisture_dates = [extract_date_from_filename(f) for f in moisture_dekads_files_new]
	moisture_dates = [d for d in moisture_dates if d is not None]

	dates = pd.to_datetime([date.strftime("%Y-%m-%d") for date in moisture_dates])
	dates_df = pd.DataFrame({"date": dates}).sort_values("date").reset_index()
	sorted_dates = list(dates_df["date"])
	moisture_df = pd.DataFrame({"moisture_file": moisture_dekads_files_new, "moisture_date": moisture_dates}).sort_values("moisture_date").reset_index()

	aligned_df = pd.merge(dates_df, moisture_df, left_on="date", right_on="moisture_date", how="left").sort_values("moisture_date").reset_index()

	missing_moisture_dates = aligned_df[aligned_df["moisture_file"].isna()]["date"]
	if not missing_moisture_dates.empty:
		print(f"Warning: Missing moisture data for {len(missing_moisture_dates)} dates.")

	aligned_moisture_files = aligned_df["moisture_file"].tolist()

	return aligned_moisture_files, sorted_dates


def process_single_moisture_tif(moisture_tif_path, catchments):
	"""Align and mask a single moisture GeoTIFF onto shared reference grid."""
	with rasterio.open(moisture_tif_path) as moisture_ds:
		resampled_moisture = cleaning_utils.align_and_mask_raster_to_reference_grid(
			src=moisture_ds,
			mask_gdf=catchments,
			src_band=1,
			dst_fill=0,
			resampling=Resampling.bilinear,
		)
	if resampled_moisture is None:
		raise RuntimeError(f"Reprojection returned None for file: {moisture_tif_path}")
	return resampled_moisture.astype(np.float32, copy=False)


def update_gridded_moisture(
	download_folder=DOWNLOADS_ROOT,
	download_path=GR_MOISTURE_DOWNLOAD_PATH,
	extract_folder=EXTRACTED_DOMAIN_PATH,
	dekads_path=GR_MOISTURE_DEKADS_PATH,
	temporal_data_path=GR_MOISTURE_TEMPORAL_PATH,
	rebuild=False,
):
	"""Stream downloaded daily moisture into date-indexed dekadal HDF5/CSV outputs."""
	try:
		target_product = cleaning_utils.resolve_target_product(None)
		daily_root = resolve_project_path(download_path)
		local = inventory_daily_files(daily_root)
		start = cleaning_utils.get_target_start_date(target_product)
		today = pd.Timestamp.today().normalize()
		if not local:
			download_gridded_moisture(
				[start, today.strftime("%Y-%m-%d")], str(resolve_project_path(download_folder))
			)
		else:
			last = max(local)
			if last < today:
				download_gridded_moisture(
					[(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")],
					str(resolve_project_path(download_folder)),
				)
		local = inventory_daily_files(daily_root)
		periods = complete_periods(local)
		invalid = validate_daily_files(periods, "sm_c4grass")
		if invalid:
			invalid_paths = {path for path, _ in invalid}
			invalid_dates = sorted(date for date, path in local.items() if path in invalid_paths)
			for path, reason in invalid:
				logging.error("Unreadable moisture source %s: %s", path, reason)
			quarantine_invalid_files(invalid)
			if invalid_dates:
				download_gridded_moisture(
					[invalid_dates[0].strftime("%Y-%m-%d"), invalid_dates[-1].strftime("%Y-%m-%d")],
					str(resolve_project_path(download_folder)),
				)
			local = inventory_daily_files(daily_root)
			missing_replacements = [date for date in invalid_dates if date not in local]
			if missing_replacements:
				raise RuntimeError(
					"Unable to replace unreadable moisture dates: "
					+ ", ".join(date.strftime("%Y-%m-%d") for date in missing_replacements)
				)
			replacement_errors = validate_daily_files(
				[PeriodFiles(date.strftime("%Y-%m-%d"), (local[date],)) for date in invalid_dates],
				"sm_c4grass",
			)
			if replacement_errors:
				raise RuntimeError(f"Replacement moisture files are unreadable: {replacement_errors[:10]}")
			periods = complete_periods(local)
			remaining = validate_daily_files(periods, "sm_c4grass")
			if remaining:
				raise RuntimeError(f"Unreadable moisture files remain after download attempt: {remaining[:10]}")
		masks = build_grid_masks(CATCHMENTS_PATH)
		status, count = sync_product(
			periods=periods,
			hdf_path=resolve_project_path(GR_MOISTURE_H5_PATH),
			csv_path=resolve_project_path(temporal_data_path),
			dataset_name="moisture",
			prefix="moisture",
			target_product=target_product,
			masks=masks,
			loader=lambda period: stream_period_grid(period.files, "sm_c4grass", masks),
			rebuild=rebuild,
		)
		logging.info("Moisture outputs %s (%d periods written).", status, count)

	except Exception as e:
		logging.error(f"Error processing moisture data: {e}")
		raise


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Build or update target-aligned gridded moisture data.")
	parser.add_argument("--rebuild", action="store_true")
	args = parser.parse_args()
	update_gridded_moisture(rebuild=args.rebuild)


if __name__ == "__main__":
	main()
