from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level config structure must be a mapping: {CONFIG_PATH}")

    return data


def get_cfg(path: str, default: Any = None) -> Any:
    current: Any = get_config()
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_forecast_horizon_periods(period_days: float | None = None) -> int:
    """Resolve the shared temporal/spatial forecast horizon from configuration."""
    raw_value = get_cfg(
        "forecast.horizon_periods",
        get_cfg("ablation.pipeline.forecast_horizon_periods", "auto_two_months"),
    )
    if isinstance(raw_value, int) and not isinstance(raw_value, bool):
        if raw_value < 1:
            raise ValueError("forecast.horizon_periods must be at least one.")
        return raw_value
    normalized = str(raw_value).strip().lower()
    try:
        periods = int(normalized)
    except ValueError:
        periods = 0
    if periods:
        if periods < 1:
            raise ValueError("forecast.horizon_periods must be at least one.")
        return periods
    if normalized not in {"auto", "auto_two_months", "2m", "two_months"}:
        raise ValueError(
            "forecast.horizon_periods must be an integer or one of "
            "{auto_two_months, auto, 2m, two_months}."
        )
    if period_days is None or period_days <= 0:
        raise ValueError("A positive period_days value is required for an automatic forecast horizon.")
    months = float(get_cfg(
        "forecast.horizon_months",
        get_cfg("ablation.pipeline.forecast_horizon_months", 2.0),
    ))
    return max(1, int(round(months * 30.4375 / float(period_days))))
