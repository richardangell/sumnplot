"""Unit tests for rounding module."""

import re

import polars as pl
import pytest

from sumnplot.discretisation.rounding import (
    RoundingError,
    _min_significance_to_differentiate_values,
    determine_required_precision_to_retain_distinct_value,
)


def test_exception_raised_for_empty_list():
    """Test that RoundingError is raised for empty input list."""
    with pytest.raises(RoundingError, match=re.escape("Input values list is empty.")):
        determine_required_precision_to_retain_distinct_value([])


@pytest.mark.parametrize(
    ("values", "expected_precision"),
    [
        pytest.param([0, 1, 2], 0, id="distinct integers"),
        pytest.param(
            [112, 234, 345],
            None,
            id="could be rounded to positive precision",
        ),
        pytest.param(
            [0.0, 1.0, 2.1],
            0,
            id="floats that can be rounded to whole numbers",
        ),
        pytest.param(
            [0.0, 1.0, 2.145],
            0,
            id="floats that can be rounded to whole numbers (b)",
        ),
        pytest.param([0.0, 1.0, 1.1], 1, id="floats requiring one decimal place"),
        pytest.param([0.001, 0.002, 0.003], 3, id="small floats"),
        pytest.param([1.1, 1.12, 1.123], 3, id="incremental floats"),
        pytest.param([0.123456, 0.123457], 6, id="requiring high precision"),
        pytest.param(
            [1.1, 1.12, 1.1235678],
            3,
            id="one value much higher precision than others",
        ),
        pytest.param(
            [1.1, 1.12, 1.1235678, 1.1235679],
            7,
            id="two distinct groups of precision",
        ),
    ],
)
def test_determine_required_precision_to_retain_distinct_value(
    values: list[int | float],
    expected_precision: int | None,
):
    """Test determine_required_precision_to_retain_distinct_value output."""
    assert len({round(v, expected_precision) for v in values}) == len(values), (
        "Test case setup error: expected precision does not retain unique values."
    )

    assert (
        determine_required_precision_to_retain_distinct_value(values)
        == expected_precision
    )


def test_floating_point_inaccuracies_guarded_by_checking_lower_precision():
    """Test that floating point inaccuracies are handled correctly."""
    values = [1.234, 1.235, 1.236]
    required_precision = 3
    precision_due_to_floating_point_inaccuracies = 4

    assert (
        _min_significance_to_differentiate_values(pl.Series("values", values))
        == precision_due_to_floating_point_inaccuracies
    )

    assert (
        determine_required_precision_to_retain_distinct_value(values)
        == required_precision
    )


def test_zero_returned_if_lower_precision_check_falls_through():
    """Test that 0 is returned if lower precision check falls through."""
    values = [0.1, 0.9]
    required_precision = 0

    assert _min_significance_to_differentiate_values(pl.Series("values", values)) == 1

    assert (
        determine_required_precision_to_retain_distinct_value(values)
        == required_precision
    )


def test_none_returned_for_single_value():
    """Test that None is returned for a single value."""
    values = [1.234828393]

    assert determine_required_precision_to_retain_distinct_value(values) is None


@pytest.mark.parametrize(
    "values",
    [
        [100, 200, 300],
        [100.0, 200.0, 300.1342],
    ],
)
def test_negative_precision_returns_none(values: list[int | float]):
    """Test that None is returned for negative precision.

    Negative precision means that rounding to 10s, 100s etc.

    """
    assert determine_required_precision_to_retain_distinct_value(values) is None
