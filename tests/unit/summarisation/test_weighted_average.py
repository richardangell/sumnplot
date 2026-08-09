"""Unit tests for the sumnplot.summarisation.weighted_average module."""

import polars as pl
import pytest

from sumnplot.summarisation.weighted_average import (
    get_sum_weight_expr,
    get_weighted_average_expr,
)


@pytest.fixture
def simple_data() -> pl.DataFrame:
    """Create a simple DataFrame for testing."""
    return pl.DataFrame(
        {
            "value": [1, 2, 3, 4],
            "weight": [1, 1, 1, 2],
        },
    )


def test_get_weighted_average_expr(simple_data: pl.DataFrame):
    """Test the weighted average expression generation."""
    expr = get_weighted_average_expr("value", "weight", new_name="weighted_avg")

    result = simple_data.select(expr).to_series().item()

    expected = (1 * 1 + 2 * 1 + 3 * 1 + 4 * 2) / (1 + 1 + 1 + 2)

    assert result == expected


def test_get_weighted_average_expressions(simple_data: pl.DataFrame):
    """Test the generation of both weighted average and weight sum expressions."""
    weighted_avg_expr = get_weighted_average_expr(
        column="value",
        weights="weight",
        new_name="weighted_avg",
    )
    weight_sum_expr = get_sum_weight_expr(
        weights="weight",
        new_name="weight_sum",
    )

    result = simple_data.select(weighted_avg_expr, weight_sum_expr)

    expected_weight_sum = 1 + 1 + 1 + 2
    expected_weighted_avg = (1 * 1 + 2 * 1 + 3 * 1 + 4 * 2) / expected_weight_sum

    assert result["weighted_avg"].item() == expected_weighted_avg
    assert result["weight_sum"].item() == expected_weight_sum
