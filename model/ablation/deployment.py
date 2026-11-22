"""Compatibility wrapper for :mod:`temporal_deployment`."""

from . import temporal_deployment as _implementation

globals().update({
    name: value for name, value in vars(_implementation).items()
    if not name.startswith("__")
})
