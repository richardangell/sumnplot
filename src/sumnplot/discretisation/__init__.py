"""Discretisation module.

Contains functions for discretising numeric variables using either equal-width or
equal-weight binning methods.

"""

from sumnplot.discretisation.equal_weight_discretiser import EqualWeightDiscretiser
from sumnplot.discretisation.equal_width_discretiser import EqualWidthDiscretiser

__all__ = [
    "EqualWeightDiscretiser",
    "EqualWidthDiscretiser",
]
