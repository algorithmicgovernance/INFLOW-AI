"""Compatibility wrapper for :mod:`temporal_transformer_pipeline`."""

from . import temporal_transformer_pipeline as _implementation

globals().update({
    name: value for name, value in vars(_implementation).items()
    if not name.startswith("__")
})

