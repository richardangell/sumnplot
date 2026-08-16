"""Super simple summarisation and plotting."""

import importlib.metadata

from sumnplot import discretisation, plot, summarisation

__version__ = importlib.metadata.version("sumnplot")

__all__ = [
    "__version__",
    "discretisation",
    "plot",
    "summarisation",
]
