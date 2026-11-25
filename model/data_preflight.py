"""Fail-fast completeness audit shared by temporal ablation and deployment."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import h5py

from processing.config import get_cfg


LOG = logging.getLogger(__name__)
DATE_COLUMNS = ("date", "period_start", "measurement_date", "time")
HDF5_BY_CSV_STEM = {
    "ecmwf_hindcasts_temporal": "ecmwf_hindcasts.h5",
    "gridded_moisture_temporal": "gridded_moisture.h5",
    "gridded_rainfall_cumulative_temporal": "gridded_rainfall_cumulative.h5",
    "gridded_rainfall_temporal": "gridded_rainfall.h5",
    "inundation_viirs_temporal": "inundation_viirs.h5",
    "inundation_modis_temporal": "inundation_modis.h5",
}


def _date_series(frame: pd.DataFrame) -> pd.Series:
    for name in DATE_COLUMNS:
        if name in frame:
            return pd.to_datetime(frame[name], errors="coerce").dt.normalize()
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _trailing_false_count(values: np.ndarray) -> int:
    count = 0
    for value in values[::-1]:
        if bool(value):
            break
        count += 1
    return count


def _source_audit(path: Path, target_dates: pd.DatetimeIndex) -> dict[str, Any]:
    frame = pd.read_csv(path)
    dates = _date_series(frame)
    valid_dates = dates.dropna()
    numeric = frame.select_dtypes(include=[np.number])
    missing_by_column = {
        column: float(value)
        for column, value in numeric.isna().mean().items()
        if float(value) > 0
    }
    source_dates = pd.DatetimeIndex(valid_dates.drop_duplicates())
    date_coverage = float(target_dates.isin(source_dates).mean()) if len(target_dates) else 0.0
    result: dict[str, Any] = {
        "file": str(path),
        "rows": int(len(frame)),
        "date_start": str(valid_dates.min().date()) if len(valid_dates) else None,
        "date_end": str(valid_dates.max().date()) if len(valid_dates) else None,
        "invalid_dates": int(dates.isna().sum()),
        "duplicate_dates": int(valid_dates.duplicated().sum()),
        "target_date_coverage_fraction": date_coverage,
        "numeric_columns": int(len(numeric.columns)),
        "numeric_missing_fraction": (
            float(numeric.isna().to_numpy().mean()) if numeric.size else None
        ),
        "entirely_missing_columns": [
            column for column in numeric if numeric[column].isna().all()
        ],
        "missing_fraction_by_column": missing_by_column,
    }
    hdf_name = HDF5_BY_CSV_STEM.get(path.stem)
    if hdf_name:
        hdf_path = path.with_name(hdf_name)
        hdf_info: dict[str, Any] = {"file": str(hdf_path), "exists": hdf_path.exists()}
        if hdf_path.exists():
            try:
                with h5py.File(hdf_path, "r") as hdf:
                    datasets = {
                        name: list(value.shape)
                        for name, value in hdf.items()
                        if isinstance(value, h5py.Dataset)
                    }
                primary_rows = max(
                    (shape[0] for shape in datasets.values() if shape), default=0
                )
                hdf_info.update({"datasets": datasets, "primary_rows": int(primary_rows)})
            except OSError as error:
                hdf_info["error"] = str(error)
        result["hdf5"] = hdf_info

    if path.stem == "ecmwf_hindcasts_temporal":
        indexed = frame.assign(_audit_date=dates).dropna(subset=["_audit_date"])
        indexed = indexed.drop_duplicates("_audit_date", keep="last").set_index("_audit_date")
        if "ecmwf_has_forecast" in indexed:
            availability = indexed["ecmwf_has_forecast"].reindex(target_dates).fillna(False).astype(bool)
        else:
            rainfall_columns = [c for c in indexed if c.startswith("ecmwf_rainfall_")]
            availability = indexed[rainfall_columns].notna().any(axis=1).reindex(
                target_dates, fill_value=False
            )
        values = availability.to_numpy(dtype=bool)
        available_positions = np.flatnonzero(values)
        if len(available_positions):
            first = int(available_positions[0])
            after_first = values[first:]
            available_after_first = float(after_first.mean())
            first_available = str(target_dates[first].date())
        else:
            available_after_first = 0.0
            first_available = None
        result["ecmwf"] = {
            "available_periods": int(values.sum()),
            "missing_periods": int((~values).sum()),
            "available_fraction_all_target_periods": float(values.mean()) if len(values) else 0.0,
            "available_fraction_after_first": available_after_first,
            "first_available_period": first_available,
            "longest_missing_run_periods": _longest_false_run(values),
            "trailing_missing_periods": _trailing_false_count(values),
        }
    return result


def audit_temporal_data(
    target_product: str,
    *,
    mode: str,
    include_ecmwf: bool = True,
    aligned_dir: str | Path | None = None,
    target_path: str | Path | None = None,
    fail_on_error: bool = True,
) -> dict[str, Any]:
    """Report source completeness and optionally reject unsafe model execution."""
    product = str(target_product).strip().lower()
    root = Path(get_cfg("paths.historic.root", "data/historic"))
    aligned = Path(aligned_dir) if aligned_dir else root / f"{product}-aligned"
    if target_path is None:
        operational = get_cfg("runtime.operational_target_temporal_path")
        configured = get_cfg(
            f"paths.historic.{product}_temporal",
            aligned / f"inundation_{product}_temporal.csv",
        )
        target_path = operational if mode == "deployment" and product == "viirs" and operational else configured
    target_path = Path(target_path)

    errors: list[str] = []
    if not aligned.exists():
        errors.append(f"Aligned data directory is missing: {aligned}")
        csv_paths: list[Path] = []
    else:
        csv_paths = sorted(aligned.glob("*.csv"))
    if not target_path.exists():
        errors.append(f"Target temporal CSV is missing: {target_path}")
        target_dates = pd.DatetimeIndex([])
        target_frame = pd.DataFrame()
    else:
        target_frame = pd.read_csv(target_path)
        target_dates = pd.DatetimeIndex(_date_series(target_frame).dropna().drop_duplicates().sort_values())

    target_column = str(get_cfg("ablation.pipeline.target_column", "percent_inundation"))
    minimum_rows = int(get_cfg("data_preflight.minimum_target_rows", 100))
    if len(target_dates) < minimum_rows:
        errors.append(f"Target timeline has {len(target_dates)} periods; minimum is {minimum_rows}")
    if target_column not in target_frame:
        errors.append(f"Target column '{target_column}' is missing from {target_path}")
        target_missing = None
    else:
        target_missing = float(pd.to_numeric(target_frame[target_column], errors="coerce").isna().mean())
        if target_missing > float(get_cfg("data_preflight.maximum_target_missing_fraction", 0.0)):
            errors.append(f"Target '{target_column}' is {target_missing:.1%} missing")

    sources = [_source_audit(path, target_dates) for path in csv_paths]
    minimum_date_coverage = float(get_cfg("data_preflight.minimum_date_coverage_fraction", 0.95))
    maximum_missing = float(get_cfg("data_preflight.maximum_numeric_missing_fraction", 0.15))
    for source in sources:
        name = Path(source["file"]).name
        hdf_info = source.get("hdf5")
        if hdf_info:
            if not hdf_info["exists"]:
                errors.append(f"Required gridded source is missing: {hdf_info['file']}")
            elif hdf_info.get("error"):
                errors.append(f"Cannot read {hdf_info['file']}: {hdf_info['error']}")
            elif hdf_info.get("primary_rows") != source["rows"]:
                errors.append(
                    f"{name} has {source['rows']} rows but {Path(hdf_info['file']).name} "
                    f"has {hdf_info.get('primary_rows')}"
                )
        if source["invalid_dates"]:
            errors.append(f"{name} has {source['invalid_dates']} invalid dates")
        if source["duplicate_dates"]:
            errors.append(f"{name} has {source['duplicate_dates']} duplicate dates")
        if source["target_date_coverage_fraction"] < minimum_date_coverage:
            errors.append(
                f"{name} covers {source['target_date_coverage_fraction']:.1%} of target dates; "
                f"minimum is {minimum_date_coverage:.1%}"
            )
        if source["entirely_missing_columns"]:
            errors.append(f"{name} has entirely missing numeric columns: {source['entirely_missing_columns']}")
        missing = source["numeric_missing_fraction"]
        if missing is not None and missing > maximum_missing and name != "ecmwf_hindcasts_temporal.csv":
            errors.append(f"{name} numeric values are {missing:.1%} missing; maximum is {maximum_missing:.1%}")

        if include_ecmwf and "ecmwf" in source:
            ecmwf = source["ecmwf"]
            minimum_ecmwf = float(get_cfg(
                "data_preflight.ecmwf.minimum_available_fraction_after_first", 0.95
            ))
            maximum_trailing = int(get_cfg(
                "data_preflight.ecmwf.maximum_trailing_missing_periods", 1
            ))
            if ecmwf["available_fraction_after_first"] < minimum_ecmwf:
                errors.append(
                    "ECMWF availability after its first forecast is "
                    f"{ecmwf['available_fraction_after_first']:.1%}; minimum is {minimum_ecmwf:.1%}"
                )
            if ecmwf["trailing_missing_periods"] > maximum_trailing:
                errors.append(
                    f"ECMWF is missing for the latest {ecmwf['trailing_missing_periods']} target periods; "
                    f"maximum is {maximum_trailing}"
                )
    if include_ecmwf and not any("ecmwf" in source for source in sources):
        errors.append("ECMWF is enabled but ecmwf_hindcasts_temporal.csv is missing")

    report = {
        "mode": mode,
        "target_product": product,
        "target_file": str(target_path),
        "target_column": target_column,
        "target_periods": int(len(target_dates)),
        "target_date_start": str(target_dates.min().date()) if len(target_dates) else None,
        "target_date_end": str(target_dates.max().date()) if len(target_dates) else None,
        "target_missing_fraction": target_missing,
        "sources": sources,
        "errors": errors,
        "passed": not errors,
    }
    report_path = Path(str(get_cfg(
        f"data_preflight.report_paths.{mode}", f"model/data_audit_{mode}.json"
    )))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOG.info(
        "DATA PREFLIGHT [%s/%s]: target periods=%d, range=%s..%s, missing=%s",
        mode, product, len(target_dates), report["target_date_start"], report["target_date_end"],
        "unknown" if target_missing is None else f"{target_missing:.1%}",
    )
    for source in sources:
        suffix = ""
        if "ecmwf" in source:
            e = source["ecmwf"]
            suffix = (
                f", ECMWF available={e['available_periods']}/{len(target_dates)} "
                f"({e['available_fraction_after_first']:.1%} after first), "
                f"trailing missing={e['trailing_missing_periods']}"
            )
        LOG.info(
            "DATA SOURCE: %-42s rows=%d range=%s..%s date coverage=%.1f%% numeric missing=%s%s",
            Path(source["file"]).name, source["rows"], source["date_start"], source["date_end"],
            100 * source["target_date_coverage_fraction"],
            "n/a" if source["numeric_missing_fraction"] is None else f"{100 * source['numeric_missing_fraction']:.1f}%",
            suffix,
        )
    LOG.info("Detailed data preflight report: %s", report_path)
    if errors:
        for error in errors:
            if fail_on_error:
                LOG.error("DATA PREFLIGHT FAILURE: %s", error)
            else:
                LOG.warning("DATA PREFLIGHT ISSUE: %s", error)
        if fail_on_error:
            raise RuntimeError(
                f"Data preflight failed with {len(errors)} issue(s); see {report_path}"
            )
    return report
