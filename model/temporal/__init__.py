"""Temporal inundation modelling package."""

from .ablation.temporal_ablation_runner import (
    TemporalAblationConfig,
    run_temporal_ablation_grid,
)

AblationConfig = TemporalAblationConfig
run_ablation_grid = run_temporal_ablation_grid

__all__ = [
    "TemporalAblationConfig",
    "run_temporal_ablation_grid",
    "AblationConfig",
    "run_ablation_grid",
]
