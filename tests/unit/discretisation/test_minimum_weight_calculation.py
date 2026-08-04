"""Unit tests for threshold_crossing_minimum_weight function."""

import re

import polars as pl
import pytest

from sumnplot.discretisation.minimum_weight_calculation import (
    ThresholdCrossingError,
    threshold_crossing_minimum_weight,
)
from sumnplot.exceptions import (
    MissingColumnError,
    NullsInColumnError,
    NumericColumnError,
)


def test_missing_values_column_exception() -> None:
    """Test that MissingColumnError is raised when the values column is missing."""
    df = pl.DataFrame({"f0": [1, 2, 3], "w": [1, 1, 1]})

    with pytest.raises(
        MissingColumnError,
        match=re.escape("Column 'missing' not found in DataFrame."),
    ):
        threshold_crossing_minimum_weight(
            df,
            column="missing",
            weights="w",
            min_weight=1,
        )


def test_missing_weights_column_exception() -> None:
    """Test that MissingColumnError is raised when the weights column is missing."""
    df = pl.DataFrame({"f0": [1, 2, 3], "w": [1, 1, 1]})

    with pytest.raises(
        MissingColumnError,
        match=re.escape("Column 'missing_w' not found in DataFrame."),
    ):
        threshold_crossing_minimum_weight(
            df,
            column="f0",
            weights="missing_w",
            min_weight=1,
        )


def test_non_numeric_values_column_exception() -> None:
    """Test that NumericColumnError is raised when the values column is not numeric."""
    df = pl.DataFrame({"f0": ["a", "b", "c"], "w": [1, 1, 1]})

    with pytest.raises(
        NumericColumnError,
        match=re.escape("Column 'f0' must be a numeric dtype."),
    ):
        threshold_crossing_minimum_weight(
            df,
            column="f0",
            weights="w",
            min_weight=1,
        )


def test_non_numeric_weights_column_exception() -> None:
    """Test that NumericColumnError is raised when the weights column is not numeric."""
    df = pl.DataFrame({"f0": [1, 2, 3], "w": ["a", "b", "c"]})

    with pytest.raises(
        NumericColumnError,
        match=re.escape("Column 'w' must be a numeric dtype."),
    ):
        threshold_crossing_minimum_weight(
            df,
            column="f0",
            weights="w",
            min_weight=1,
        )


def test_nulls_in_values_column_exception() -> None:
    """Test that NullsInColumnError is raised when the values column contains nulls."""
    df = pl.DataFrame({"f0": [1, None, 3], "w": [1, 1, 1]})

    with pytest.raises(
        NullsInColumnError,
        match=re.escape("Column 'f0' contains null values."),
    ):
        threshold_crossing_minimum_weight(
            df,
            column="f0",
            weights="w",
            min_weight=1,
        )


def test_nulls_in_weights_column_exception() -> None:
    """Test that NullsInColumnError is raised when the weights column contains nulls."""
    df = pl.DataFrame({"f0": [1, 2, 3], "w": [1, None, 1]})

    with pytest.raises(
        NullsInColumnError,
        match=re.escape("Column 'w' contains null values."),
    ):
        threshold_crossing_minimum_weight(
            df,
            column="f0",
            weights="w",
            min_weight=1,
        )


@pytest.mark.parametrize(
    "min_weight",
    [0, -1, -10],
)
def test_min_weight_must_be_positive(min_weight: int) -> None:
    """Test that ThresholdCrossingError is raised when min_weight is not positive."""
    df = pl.DataFrame({"f0": [1, 2, 3], "w": [1, 1, 1]})

    with pytest.raises(
        ThresholdCrossingError,
        match=re.escape("min_weight must be greater than 0."),
    ):
        threshold_crossing_minimum_weight(
            df,
            column="f0",
            weights="w",
            min_weight=min_weight,
        )


@pytest.mark.parametrize(
    ("values", "weights"),
    [
        pytest.param([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], id="unique-values"),
        pytest.param([3, 5, 1, 4, 2], [1, 1, 1, 1, 1], id="unique-values-reordered"),
    ],
)
@pytest.mark.parametrize(
    ("min_weight", "expected"),
    [
        pytest.param(0.5, [1, 2, 3, 4, 5], id="all weights greater than min_weight"),
        pytest.param(1, [1, 2, 3, 4, 5], id="all weights equal to min_weight"),
        pytest.param(2, [2, 4], id="every 2nd value"),
        pytest.param(3, [3], id="every 3rd value"),
        pytest.param(4, [4], id="every 4th value"),
        pytest.param(5, [5], id="every 5th value"),
        pytest.param(6, [], id="min_weight greater than sum of weights"),
    ],
)
def test_equal_weight_by_values(
    values: list[int],
    weights: list[int],
    min_weight: float,
    expected: list[int],
) -> None:
    """Test threshold_crossing_minimum_weight with equal weight by value.

    This test uses different ordering of the same values to ensure the function
    correctly sorts by values.

    """
    df = pl.DataFrame({"f0": values, "w": weights})
    result = threshold_crossing_minimum_weight(
        df,
        column="f0",
        weights="w",
        min_weight=min_weight,
    )
    assert result == expected


@pytest.mark.parametrize(
    ("values", "weights"),
    [
        pytest.param([1, 2, 3, 4, 5], [1, 2, 1, 1, 2], id="unique-values"),
        pytest.param([5, 4, 1, 3, 2], [2, 1, 1, 1, 2], id="unique-values-reordered"),
        pytest.param(
            [1, 2, 2, 3, 4, 5, 5],
            [1, 1, 1, 1, 1, 1, 1],
            id="duplicate-values",
        ),
        pytest.param(
            [2, 1, 5, 2, 4, 3, 5],
            [1, 1, 1, 1, 1, 1, 1],
            id="duplicate-values-reordered",
        ),
    ],
)
@pytest.mark.parametrize(
    ("min_weight", "expected"),
    [
        pytest.param(0.5, [1, 2, 3, 4, 5], id="all weights greater than min_weight"),
        pytest.param(
            1,
            [1, 2, 3, 4, 5],
            id="all weights greater or equal to min_weight",
        ),
        pytest.param(2, [2, 4, 5], id="every 2nd value"),
        pytest.param(3, [2, 5], id="every 3rd value"),
        pytest.param(4, [3], id="every 4th value"),
        pytest.param(5, [4], id="every 5th value"),
        pytest.param(6, [5], id="min_weight greater than sum of weights"),
        pytest.param(7, [5], id="min_weight greater than sum of weights"),
        pytest.param(8, [], id="min_weight greater than sum of weights"),
    ],
)
def test_non_equal_weight_by_values(
    values: list[int],
    weights: list[int],
    min_weight: float,
    expected: list[int],
) -> None:
    """Test threshold_crossing_minimum_weight with non-equal weight by value.

    This test uses different ordering of the same values to ensure the function
    correctly sorts by values.

    It also uses duplicated and non-duplicated values to ensure the function
    correctly handles summarising weights by values before the calculation.

    """
    df = pl.DataFrame({"f0": values, "w": weights})

    result = threshold_crossing_minimum_weight(
        df,
        column="f0",
        weights="w",
        min_weight=min_weight,
    )

    assert result == expected
