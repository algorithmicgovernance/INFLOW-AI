"""Human-readable names used by explanation and feature-history plots."""

from __future__ import annotations

import re

from data.stats.gridded_stats import region_to_code_dict


VARIABLE_NAME_MAP = {
    "albert_water_level": "Lake Albert water level",
    "victoria_height_variation": "Lake Victoria water-level variation",
    "kyoga_height_variation": "Lake Kyoga water-level variation",
    "moisture": "Soil moisture",
    "rainfall": "Rainfall",
    "cumulative_rainfall": "Cumulative rainfall",
    # TAMSAT is retained as a model input name, but both lake rainfall products
    # use the requested display label.
    "CHIRPS": "Rainfall over Lake Victoria (CHIRPS)",
    "TAMSAT": "Rainfall over Lake Victoria (CHIRPS)",
    "CHIRPS_cumulative": "Cumulative rainfall over Lake Victoria (CHIRPS)",
    "TAMSAT_cumulative": "Cumulative rainfall over Lake Victoria (CHIRPS)",
    "CHIRPS_cumulative_scaled": "Scaled cumulative rainfall over Lake Victoria (CHIRPS)",
    "TAMSAT_cumulative_scaled": "Scaled cumulative rainfall over Lake Victoria (CHIRPS)",
    "CHIRPS_cumulative_cumulative_scaled": "Scaled cumulative rainfall over Lake Victoria (CHIRPS)",
    "TAMSAT_cumulative_cumulative_scaled": "Scaled cumulative rainfall over Lake Victoria (CHIRPS)",
    "oniTOTAL": "Oceanic Nino Index",
    "oniANOM": "Oceanic Nino Index anomaly",
    "soi_anom": "Southern Oscillation Index anomaly",
    "soi_sd": "Southern Oscillation Index standard deviation",
    "dmi": "Dipole Mode Index",
    "wtio": "Western Tropical Indian Ocean index",
    "setio": "Southeastern Tropical Indian Ocean index",
}

REGION_CODE_TO_NAME = {code: name for name, code in region_to_code_dict.items()}


def _lag_label(lag: int, target_product: str | None) -> str:
    if str(target_product).strip().lower() == "viirs":
        weeks = lag * 2
        if weeks % 4 == 0:
            months = weeks // 4
            return f"-{months} {'month' if months == 1 else 'months'}"
        return f"-{weeks} weeks"
    return f"-{lag} {'period' if lag == 1 else 'periods'}"


def human_readable_variable_name(name: str, target_product: str | None = None) -> str:
    """Convert a model feature key to a consistent plot label."""
    raw_name = str(name)
    lag_match = re.search(r"_lag_(\d+)$", raw_name)
    lag = int(lag_match.group(1)) if lag_match else None
    base_name = raw_name[: lag_match.start()] if lag_match else raw_name

    region_name = None
    for code, full_name in REGION_CODE_TO_NAME.items():
        suffix = f"_{code}"
        if base_name.lower().endswith(suffix):
            base_name = base_name[: -len(suffix)]
            region_name = full_name
            break

    label = VARIABLE_NAME_MAP.get(base_name)
    if label is None:
        if base_name.startswith("mjo"):
            label = f"Madden-Julian Oscillation ({base_name[3:]})"
        elif base_name.startswith("nino"):
            label = base_name.replace("nino", "Nino ", 1).replace("ANOM", "anomaly ").strip()
        else:
            label = base_name.replace("_", " ").strip().capitalize()

    if region_name:
        label = f"{label} in {region_name}"
    if lag is not None:
        label = f"{label} {_lag_label(lag, target_product)}"
    return label
