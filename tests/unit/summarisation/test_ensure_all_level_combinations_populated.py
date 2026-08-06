"""Unit tests for sumnplot.summarisation.fill_out_summary_table."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sumnplot.exceptions import MissingColumnError
from sumnplot.summarisation.ensure_all_level_combinations_populated import (
    _cross,
    ensure_all_level_combinations_populated,
)


def test_cross():
    """Simple test of the _cross function."""
    series1 = pl.Series("A", [1, 2])
    series2 = pl.Series("B", ["x", "y"])
    series3 = pl.Series("C", [True, False])

    result = _cross([series1, series2, series3])

    expected = pl.DataFrame(
        {
            "A": [1, 1, 1, 1, 2, 2, 2, 2],
            "B": ["x", "x", "y", "y", "x", "x", "y", "y"],
            "C": [True, False, True, False, True, False, True, False],
        },
    )

    assert_frame_equal(result, expected)


class TestEnsureAllLevelCombinationsPopulated:
    """Tests for the ensure_all_level_combinations_populated function."""

    def test_column_missing_from_one_dataframe_inputs_exception(self):
        """Test ExceptionGroup is raised when groupby column missing from DataFrame."""
        full_df = pl.DataFrame({"A": [1, 2], "B": ["x", "y"], "C": [True, False]})
        summary_df = pl.DataFrame({"A": [1, 2], "B": ["x", "x"], "value": [10, 30]})
        groupby_columns = ["A", "C"]  # 'C' is missing
        value_columns = {"value": 0}

        with pytest.raises(ExceptionGroup) as exc_info:
            ensure_all_level_combinations_populated(
                full_df=full_df,
                summary_df=summary_df,
                groupby_columns=groupby_columns,
                value_columns=value_columns,
            )

        raised_exceptions = exc_info.value.exceptions

        expected_messages = [
            "Column 'C' not found in DataFrame (summary_df).",
        ]

        assert len(raised_exceptions) == 1
        raised_exception = raised_exceptions[0]
        assert isinstance(raised_exception, MissingColumnError)
        assert raised_exception.message == expected_messages[0]

    def test_column_missing_from_both_dataframe_inputs_exception(
        self,
        subtests: pytest.Subtests,
    ):
        """Test ExceptionGroup is raised when groupby column missing from DataFrames."""
        full_df = pl.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        summary_df = pl.DataFrame({"A": [1, 2], "B": ["x", "x"], "value": [10, 30]})
        groupby_columns = ["A", "C"]  # 'C' is missing
        value_columns = {"value": 0}

        with pytest.raises(ExceptionGroup) as exc_info:
            ensure_all_level_combinations_populated(
                full_df=full_df,
                summary_df=summary_df,
                groupby_columns=groupby_columns,
                value_columns=value_columns,
            )

        raised_exceptions = exc_info.value.exceptions

        expected_messages = [
            "Column 'C' not found in DataFrame (full_df).",
            "Column 'C' not found in DataFrame (summary_df).",
        ]

        assert len(raised_exceptions) == 2

        for i in range(2):
            with subtests.test(f"Exception {i}"):
                exception = raised_exceptions[i]
                assert isinstance(exception, MissingColumnError)
                assert exception.message == expected_messages[i]

    def test_multiple_columns_missing_from_both_dataframe_inputs_exception(
        self,
        subtests: pytest.Subtests,
    ):
        """Test ExceptionGroup is raised when groupby column missing from DataFrames."""
        full_df = pl.DataFrame({"A": [1, 2], "B": ["x", "y"], "D": [True, False]})
        summary_df = pl.DataFrame({"A": [1, 2], "B": ["x", "x"], "value": [10, 30]})
        groupby_columns = ["A", "C", "D"]  # 'C' and 'D' are missing
        value_columns = {"value": 0}

        with pytest.raises(ExceptionGroup) as exc_info:
            ensure_all_level_combinations_populated(
                full_df=full_df,
                summary_df=summary_df,
                groupby_columns=groupby_columns,
                value_columns=value_columns,
            )

        raised_exceptions = exc_info.value.exceptions

        expected_messages = [
            "Column 'C' not found in DataFrame (full_df).",
            "Column 'C' not found in DataFrame (summary_df).",
            "Column 'D' not found in DataFrame (summary_df).",
        ]

        assert len(raised_exceptions) == 3

        for i in range(3):
            with subtests.test(f"Exception {i}"):
                exception = raised_exceptions[i]
                assert isinstance(exception, MissingColumnError)
                assert exception.message == expected_messages[i]

    def test_all_levels_already_populated(self):
        """Test original summary_df returned when all combinations are present.

        The original summary_df is returned, but sorted by the groupby columns.

        """
        full_df = pl.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        summary_df = pl.DataFrame(
            {"A": [1, 1, 2, 2], "B": ["y", "x", "x", "y"], "value": [10, 20, 30, 40]},
        )
        groupby_columns = ["A", "B"]
        value_columns = {"value": 0}

        result = ensure_all_level_combinations_populated(
            full_df=full_df,
            summary_df=summary_df,
            groupby_columns=groupby_columns,
            value_columns=value_columns,
        )

        assert_frame_equal(result, summary_df.sort(by=groupby_columns))

    def test_missing_combinations_are_filled(self):
        """Test missing combinations are filled with default values.

        Also verifies that the returned DataFrame is sorted by the groupby columns.

        """
        full_df = pl.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        summary_df = pl.DataFrame({"A": [1, 2], "B": ["x", "x"], "value": [10, 30]})
        groupby_columns = ["A", "B"]
        value_columns = {"value": 0}

        result = ensure_all_level_combinations_populated(
            full_df=full_df,
            summary_df=summary_df,
            groupby_columns=groupby_columns,
            value_columns=value_columns,
        )

        expected = pl.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": ["x", "y", "x", "y"],
                "value": [10, 0, 30, 0],
            },
        )

        assert_frame_equal(result, expected)

    def test_only_some_value_columns_defaulted(self):
        """Test that only some value columns are defaulted when missing."""
        full_df = pl.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        summary_df = pl.DataFrame(
            {"A": [1, 2], "B": ["x", "x"], "value1": [10, 30], "value2": [-1, -1]},
        )
        groupby_columns = ["A", "B"]
        value_columns = {"value1": 0}

        result = ensure_all_level_combinations_populated(
            full_df=full_df,
            summary_df=summary_df,
            groupby_columns=groupby_columns,
            value_columns=value_columns,
        )

        expected = pl.DataFrame(
            {
                "A": [1, 1, 2, 2],
                "B": ["x", "y", "x", "y"],
                "value1": [10, 0, 30, 0],
                "value2": [-1, None, -1, None],
            },
        )

        assert_frame_equal(result, expected)
