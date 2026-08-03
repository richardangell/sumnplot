"""Super simple summarisation and plotting."""

import importlib.metadata

from . import checks, discretisation, plot, summary

__version__ = importlib.metadata.version("sumnplot")

__all__ = [
    "__version__",
    "checks",
    "discretisation",
    "plot",
    "summary",
]
