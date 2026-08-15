"""Unit tests for the one_way_summary module."""

import polars as pl
import pytest

from sumnplot.exceptions import MissingColumnError
from sumnplot.plot.altair.one_way_summary import produce_one_way_summary_plot
from sumnplot.summarisation.summary_operation import SummaryOperation
from sumnplot.summarisation.summary_table import SummaryTable


def test_missing_columns_raises_exception_group():
    """Test that an ExceptionGroup is raised when missing columns are specified."""
    df = pl.DataFrame(
        schema=["x_b", "w_b", "left_y_b", "right_y1_b", "right_y2_b"],
    )

    summary_table = SummaryTable(
        df,
        groupby_columns=["x_b"],
        summarised_column_types={
            "left_y_b": SummaryOperation.SUM,
            "right_y1_b": SummaryOperation.WEIGHTED_AVERAGE,
            "right_y2_b": SummaryOperation.WEIGHTED_AVERAGE,
        },
    )

    # All the following columns are missging from the SummaryTable data.
    x_axis_column = "x"
    left_y_axis_column = "left_y"
    right_y_axis_columns = ["right_y1", "right_y2"]

    with pytest.raises(ExceptionGroup, match="Missing columns") as exc_info:
        produce_one_way_summary_plot(
            summary_table,
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
