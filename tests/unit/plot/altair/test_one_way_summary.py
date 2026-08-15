"""Unit tests for the one_way_summary module."""

import re

import polars as pl
import pytest

from sumnplot.exceptions import MissingColumnError
from sumnplot.plot.altair.exceptions import AltairPlotError
from sumnplot.plot.altair.one_way_summary import (
    OneWaySummaryColours,
    produce_one_way_summary_plot,
)
from sumnplot.summarisation.summary_operation import SummaryOperation
from sumnplot.summarisation.summary_table import SummaryTable


@pytest.fixture
def summary_table() -> SummaryTable:
    """Fixture to create a sample SummaryTable for testing."""
    df = pl.DataFrame(
        schema=["x", "f0", "f1", "f2"],
    )

    return SummaryTable(
        df,
        groupby_columns=["x"],
        summarised_column_types={
            "f0": SummaryOperation.SUM,
            "f1": SummaryOperation.WEIGHTED_AVERAGE,
            "f2": SummaryOperation.WEIGHTED_AVERAGE,
        },
    )


def test_missing_columns_raises_exception_group(summary_table: SummaryTable):
    """Test that an ExceptionGroup is raised when missing columns are specified."""
    # All the following columns are missging from the SummaryTable data.
    x_axis_column = "x_b"
    left_y_axis_column = "f0_b"
    right_y_axis_columns = ["f1_b", "f2_b"]

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


@pytest.mark.parametrize(
    ("colours", "expected_message"),
    [
        (
            ("#FF0000",),
            (
                "Not enough line colours specified for 2 lines. "
                "Only 1 line colours specified."
            ),
        ),
        (None, "No line colours specified."),
    ],
)
def test_too_many_columns_for_line_colours_raises_exception(
    summary_table: SummaryTable,
    colours: tuple[str, ...] | None,
    expected_message: str,
):
    """Test exception raised when more right y axis columns than line colours."""
    plot_colours = OneWaySummaryColours(
        bar_colour="#000000",
        line_colours=colours,
    )

    with pytest.raises(AltairPlotError, match=expected_message):
        produce_one_way_summary_plot(
            summary_table,
            x_axis_column="x",
            left_y_axis_column="f0",
            right_y_axis_columns=["f1", "f2"],
            colours=plot_colours,
        )


def test_multi_way_summary_exception():
    """Test that an exception is raised when a multi-way summary table is provided."""
    df = pl.DataFrame(
        schema=["x", "y", "f0", "f1", "f2"],
    )

    multi_way_summary_table = SummaryTable(
        df,
        groupby_columns=["x", "y"],
        summarised_column_types={
            "f0": SummaryOperation.SUM,
            "f1": SummaryOperation.WEIGHTED_AVERAGE,
            "f2": SummaryOperation.WEIGHTED_AVERAGE,
        },
    )

    with pytest.raises(
        AltairPlotError,
        match=re.escape(
            "summary contains data summarised by more than one groupby column.",
        ),
    ):
        produce_one_way_summary_plot(
            multi_way_summary_table,
            x_axis_column="x",
            left_y_axis_column="f0",
            right_y_axis_columns=["f1", "f2"],
        )
