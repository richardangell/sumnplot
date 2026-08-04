"""Function to calculate threshold crossings based on minimum weight.

This function is used to determine the cut points for discretisation based on a
minimum weight threshold. The buckets produced by these cut points will have at
least the specified minimum weight (except the upper bucket which may have less
is all or some of the others have more than the minimum weight or if the minimum
weight is not neatly divisible into the total weight).

"""

import polars as pl

from sumnplot.exceptions import MissingColumnError, SumNPlotError


class ThresholdCrossingError(SumNPlotError):
    """Raised when the minimum weight threshold is not met."""


def threshold_crossing_minimum_weight(
    df: pl.DataFrame,
    *,
    column: str,
    weights: str,
    min_weight: float,
) -> list[int]:
    """Return values of sorted column where cumulative weight rises by >= min_weight.

    Once the cumulative weight has risen by min_weight, the counter is reset and the
    next threshold is calculated from the current cumulative weight.

    Args:
        df : The input DataFrame.
        column : The name of the column to calculate thresholds for.
        weights : The name of the column containing weights.
        min_weight : The minimum weight threshold to trigger a crossing.

    Returns:
        A list of column values where the cumulative weight crosses the min_weight
        threshold.

    """
    if column not in df.columns:
        raise MissingColumnError(column)

    if weights not in df.columns:
        raise MissingColumnError(weights)

    if min_weight <= 0:
        msg = "min_weight must be greater than 0."
        raise ThresholdCrossingError(msg)

    cum_sum_weights = (
        df.group_by(column)
        .agg(pl.col(weights).sum())
        .sort(column, descending=False)
        .with_columns(pl.col(weights).cum_sum().alias("cumsum_weights"))
    )

    n = cum_sum_weights.height

    breaks = []
    baseline = 0.0
    pos = 0

    while pos < n:
        idx = cum_sum_weights.select(
            pl.col("cumsum_weights").search_sorted(
                element=baseline + min_weight,
                side="left",
                descending=False,
            ),
        ).item()

        if idx >= n:
            break

        baseline = cum_sum_weights.select(pl.col("cumsum_weights").get(idx)).item()

        column_value = cum_sum_weights.select(pl.col(column).get(idx)).item()
        breaks.append(column_value)

        pos = idx + 1

    return breaks
