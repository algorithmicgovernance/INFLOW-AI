import os
import sys

import matplotlib.pyplot as plt
import h5py
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.config import get_cfg
from processing.data_cleaning.process_inundation_viirs import (
    _remove_drying_period_increases,
    build_viirs_temporal_dataframe,
)
from processing import cleaning_utils


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "visual", "output", "viirs_cleaning")


def load_viirs_scaled_dataframe(path=None):
    """Reconstruct unsmoothed VIIRS temporal data from the binary HDF5 masks."""
    if path is None:
        path = get_cfg("paths.historic.viirs_temporal", "data/historic/viirs_inundation_temporal.csv")

    metadata = pd.read_csv(path)
    h5_path = get_cfg("paths.historic.viirs_h5", "data/historic/inundation_viirs.h5")
    with h5py.File(h5_path, "r") as hdf:
        rasters = hdf["inundation"][:]
    if len(metadata) != len(rasters):
        raise ValueError(
            f"VIIRS CSV/HDF5 length mismatch: {len(metadata)} rows versus {len(rasters)} masks."
        )

    df = build_viirs_temporal_dataframe(
        metadata["file_name"].tolist(),
        rasters,
        regions_gdf=cleaning_utils.extract_regions(),
    )

    if "period_start" in df.columns:
        df["date"] = pd.to_datetime(df["period_start"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Expected either 'period_start' or 'date' column in VIIRS CSV.")

    df = df.dropna(subset=["date"]).copy()

    # Plot the same target-style features used for inundation interpretation.
    plot_cols = [c for c in df.columns if c.startswith("percent_inundation")]
    if not plot_cols:
        raise ValueError("No 'percent_inundation*' columns found in VIIRS temporal CSV.")

    return df[["date"] + plot_cols], plot_cols


def build_viirs_cleaning_comparison(df, plot_cols):
    """Create raw, rolling-median, and drying-period-corrected series."""
    comparison = df[["date"]].copy()
    for column in plot_cols:
        raw = pd.to_numeric(df[column], errors="coerce")
        smoothed = raw.rolling(window=3, center=True, min_periods=1).median()
        comparison[f"{column}__raw"] = raw
        comparison[f"{column}__smoothed"] = smoothed
        comparison[f"{column}__corrected"] = _remove_drying_period_increases(
            smoothed,
            df["date"],
        )
    return comparison


def plot_viirs_cleaning_comparison(comparison, plot_cols, output_dir=OUTPUT_DIR):
    """Overlay all three VIIRS cleaning stages and save one plot per target."""
    os.makedirs(output_dir, exist_ok=True)

    for column in plot_cols:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            comparison["date"], comparison[f"{column}__raw"],
            color="0.65", linewidth=1, alpha=0.7, label="Raw",
        )
        ax.plot(
            comparison["date"], comparison[f"{column}__smoothed"],
            color="#1f77b4", linewidth=1.5, label="3-observation rolling median",
        )
        ax.plot(
            comparison["date"], comparison[f"{column}__corrected"],
            color="#d62728", linewidth=1.5, label="Drying-period corrected",
        )
        ax.set_title(f"VIIRS cleaning comparison: {column}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Percent inundation")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{column}_cleaning_comparison.png"), dpi=200)
        plt.close(fig)

    return output_dir


def run_visual_check():
    df, plot_cols = load_viirs_scaled_dataframe()
    comparison = build_viirs_cleaning_comparison(df, plot_cols)
    output_dir = plot_viirs_cleaning_comparison(comparison, plot_cols)
    print(
        f"Generated {len(plot_cols)} raw/smoothed/corrected VIIRS comparison "
        f"plot(s) in {output_dir}."
    )


if __name__ == "__main__":
    run_visual_check()
