"""Helper functions for rounding."""

import polars as pl

from sumnplot.exceptions import SumNPlotError


class RoundingError(SumNPlotError):
    """Exception raised for errors in the rounding module."""


def _min_significance_to_differentiate_values(series: pl.Series) -> int:
    """Calculate the minimum significance to differentiate values.

    Args:
        series (pl.Series): The input series.

    Returns:
        int: The log10 of the minimum difference, floored and negated.

    """
    differences = series.sort().diff()
    return int(-1 * differences.log10().floor().cast(pl.Int32).min())  # type: ignore[reportOperatorIssue]


def determine_required_precision_to_retain_distinct_value(
    values: list[int | float],
) -> int | None:
    """Determine required precision to round values that keeps each distinct value.

    Function returns the required precision to round the provided values to such that
    rounded values are still distinct. The precision is returned as an  integer giving
    the number of decimals places to round to where 0 means rounding to the nearest
    integer and 1 means rounding to 1 decimal place.

    Also does not suggest rounding to a negative number of decimal places (i.e.
    rounding to 10s, 100s etc.) as this would not be a useful precision, in this case
    None will be returned from the function.

    Function returns None if a single value is provided in the input list.

    Args:
        values (list[int | float]): The list of values to determine precision for.

    Returns:
        int | None: The required precision. None means that no rounding is required.

    Raises:
        RoundingError: If the input values list is empty.

    """
    if not values:
        msg = "Input values list is empty."
        raise RoundingError(msg)

    if len(values) == 1:
        return None

    values_series = pl.Series("values", values)

    required_precision = _min_significance_to_differentiate_values(values_series)

    if required_precision < 0:
        return None

    # Check if lower precision can still retain unique values.
    for current_precision in range(required_precision, 0, -1):
        lower_precision = current_precision - 1
        rounded_values = values_series.round(lower_precision, mode="half_to_even")
        if rounded_values.n_unique() < len(values):
            return current_precision

    return 0
