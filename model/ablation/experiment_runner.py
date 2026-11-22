"""Compatibility wrapper for :mod:`temporal_ablation_runner`."""

from . import temporal_ablation_runner as _implementation

globals().update({
    name: value for name, value in vars(_implementation).items()
    if not name.startswith("__")
})
