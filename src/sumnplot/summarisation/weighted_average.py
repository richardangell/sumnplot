"""Polars weighted average expression."""

import polars as pl


def get_weighted_average_expr(
    column: str,
    weights: str,
    new_name: str | None = None,
) -> pl.Expr:
    """Generate a polars expression to calculate the weighted average.

    Args:
        column (str): The name of the column for which to calculate the weighted
            average.
        weights (str): The name of the weights column.
        new_name (str | None): The name for the resulting expression. If None, no alias
            is applied and the expression will have the 'column' name.

    Returns:
        pl.Expr: Expression that can be used to calculate the weighted average.

    """
    expr = (pl.col(column) * pl.col(weights)).sum() / pl.col(weights).sum()

    if new_name:
        expr = expr.alias(new_name)

    return expr


def get_sum_weight_expr(weights: str, new_name: str | None = None) -> pl.Expr:
    """Generate a polars expression to calculate the sum of weights.

    Args:
        weights (str): The name of the weights column.
        new_name (str | None): The name for the resulting expression. If None, no alias
            is applied and the expression will have the 'weights' name.

    Returns:
        pl.Expr: Expression that can be used to calculate the sum of weights.

    """
    expr = pl.col(weights).sum()

    if new_name:
        expr = expr.alias(new_name)

    return expr
