"""Unit tests for the one_way_summary module."""

import polars as pl
import pytest

from sumnplot.exceptions import MissingColumnError
from sumnplot.plot.altair.one_way_summary import produce_one_way_summary_plot


def test_missing_columns_raises_exception_group():
    """Test that an ExceptionGroup is raised when missing columns are specified."""
    df = pl.DataFrame()
    x_axis_column = "x"
    left_y_axis_column = "left_y"
    right_y_axis_columns = ["right_y1", "right_y2"]

    with pytest.raises(ExceptionGroup, match="Missing columns") as exc_info:
        produce_one_way_summary_plot(
            df,
            x_axis_column=x_axis_column,
            left_y_axis_column=left_y_axis_column,
            right_y_axis_columns=right_y_axis_columns,
        )
    assert len(exc_info.value.exceptions) == 4  # All four columns are missing

    expected_missing_columns = [
        x_axis_column,
        left_y_axis_column,
        *right_y_axis_columns,
    ]

    for raised_exception, expected_column in zip(
        exc_info.value.exceptions,
        expected_missing_columns,
        strict=True,
    ):
        assert isinstance(raised_exception, MissingColumnError)
        assert raised_exception.column == expected_column
        assert (
            raised_exception.message
            == f"Column '{expected_column}' not found in DataFrame."
        )
