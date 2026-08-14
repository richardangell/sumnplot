"""Helper function to calculate weighted average by groupby columns."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from sumnplot.exceptions import MissingColumnError, SumNPlotError
from sumnplot.summarisation.ensure_all_level_combinations_populated import (
    ensure_all_level_combinations_populated,
)
from sumnplot.summarisation.weighted_average import (
    get_sum_weight_expr,
    get_weighted_average_expr,
)


class GroupByWeightedAverageError(SumNPlotError):
    """Base class for errors in the group_by_weighted_average function."""


@dataclass(frozen=True)
class ResponseWeight:
    """Class to represent a response and weight column pair.

    Attributes:
        response (str): The name of the response column.
        weight (str): The name of the associated weight column.

    """

    response: str
    weight: str


def group_by_weighted_average(
    df: pl.DataFrame,
    *,
    groupby_columns: list[str],
    responses: list[ResponseWeight],
) -> pl.DataFrame:
    """Group by the specified columns and aggregate the specified column.

    This function ensures that all combinations of levels in the group by columns
    are represented in the output table. Level combinations that are not present in the
    input DataFrame will have their weight sums imputed with 0s, the weighted averages
    will be left as nulls.

    Args:
        df (pl.DataFrame): The DataFrame to group.
        groupby_columns (list[str]): The columns to group by.
        responses (list[ResponseWeight]): A list of ResponseWeight objects
            specifying response and weight column pairs.

    Returns:
        pl.DataFrame: Data grouped by the specified columns with aggregated responses.

    Raises:
        ExceptionGroup: If any of the groupby columns or response/weight columns are
            missing from the DataFrame, an ExceptionGroup is raised containing all
            MissingColumnError instances for the missing columns.

    """
    column_errors: list[MissingColumnError] = []
    for col in groupby_columns:
        if col not in df.columns:
            column_errors.append(MissingColumnError(col))
    for response in responses:
        if response.response not in df.columns:
            column_errors.append(MissingColumnError(response.response))
        if response.weight not in df.columns:
            column_errors.append(MissingColumnError(response.weight))

    if column_errors:
        exception_group_msg = "Missing columns"
        raise ExceptionGroup(exception_group_msg, column_errors)

    if len(set(responses)) != len(responses):
        msg = (
            "Duplicate responses found in the responses list. "
            "Please ensure all columns are unique."
        )
        raise GroupByWeightedAverageError(msg)

    deduped_weights = []
    for response in responses:
        if response.weight not in deduped_weights:
            deduped_weights.append(response.weight)

    agg_expressions: list[pl.Expr] = []

    for response_weight in responses:
        agg_expressions.append(
            get_weighted_average_expr(response_weight.response, response_weight.weight),
        )
    for weight in deduped_weights:
        agg_expressions.append(get_sum_weight_expr(weight))

    summary = df.group_by(groupby_columns).agg(agg_expressions)

    return ensure_all_level_combinations_populated(
        full_df=df,
        summary_df=summary,
        groupby_columns=groupby_columns,
        value_columns={response.weight: 0 for response in responses},
    )
