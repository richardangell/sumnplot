"""Super simple summarisation and plotting."""

import importlib.metadata

from sumnplot import checks, discretisation, plot

__version__ = importlib.metadata.version("sumnplot")

__all__ = [
    "__version__",
    "checks",
    "discretisation",
    "plot",
]
