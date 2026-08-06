"""Polars weighted average expression."""

import polars as pl


def get_weighted_average_expr(
    column: str,
    weights: str,
    new_name: str | None = None,
) -> pl.Expr:
    """Generate a polars expression to calculate the weighted average.

    Args:
        column : The name of the column for which to calculate the weighted average.
        weights : The name of the weights column.
        new_name : The name for the resulting expression. If None, no alias is applied
            and the expression will have the 'column' name.

    Returns:
        A polars expression that can be used to calculate the weighted average.

    """
    expr = (pl.col(column) * pl.col(weights)).sum() / pl.col(weights).sum()

    if new_name:
        expr = expr.alias(new_name)

    return expr


def get_weighted_average_expressions(
    column: str,
    weights: str,
    weighted_average_name: str | None = None,
    weight_sum_name: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """Get polars expressions to calculate weighted average and sum of weights.

    Args:
        column : The name of the column for which to calculate the weighted average.
        weights : The name of the weights column.
        weighted_average_name : The name for the resulting weighted average expression.
            If None, no alias is applied.
        weight_sum_name : The name for the resulting weight sum expression. If None, no
            alias is applied.

    Returns:
        A tuple of polars expressions: (weighted_average_expr, weight_sum_expr).

    """
    weighted_average_expr = get_weighted_average_expr(
        column=column,
        weights=weights,
        new_name=weighted_average_name,
    )

    weight_sum_expr = pl.col(weights).sum()
    if weight_sum_name:
        weight_sum_expr = weight_sum_expr.alias(weight_sum_name)

    return weighted_average_expr, weight_sum_expr
