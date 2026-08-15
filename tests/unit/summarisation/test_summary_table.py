"""Unit tests for the summary_table module."""

import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sumnplot.exceptions import MissingColumnError
from sumnplot.summarisation.summary_operation import SummaryOperation
from sumnplot.summarisation.summary_table import (
    SummaryTable,
    SummaryTableError,
    SummaryTableModificationError,
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Return a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "group": ["A", "B", "C"],
            "value": [10, 20, 30],
        },
    )


@pytest.fixture
def summary_table(sample_data: pl.DataFrame) -> SummaryTable:
    """Return a SummaryTable with single group column and summarised value column."""
    groupby_columns = ["group"]
    summarised_column_types = {"value": SummaryOperation.SUM}
    return SummaryTable(
        sample_data,
        groupby_columns=groupby_columns,
        summarised_column_types=summarised_column_types,
    )


@pytest.fixture
def sample_data_with_several_value_columns() -> pl.DataFrame:
    """Return a sample DataFrame with several value columns."""
    return pl.DataFrame(
        {
            "group": ["A", "B", "C"],
            "f0": [10, 20, 30],
            "f1": [100, 200, 300],
            "f2": [1000, 2000, 3000],
        },
    )


@pytest.fixture
def summary_table_with_several_value_columns(
    sample_data_with_several_value_columns: pl.DataFrame,
) -> SummaryTable:
    """Return a SummaryTable with single group column and summarised value column."""
    groupby_columns = ["group"]
    summarised_column_types = {
        "f0": SummaryOperation.WEIGHTED_AVERAGE,
        "f1": SummaryOperation.SUM,
        "f2": SummaryOperation.SUM,
    }
    return SummaryTable(
        sample_data_with_several_value_columns,
        groupby_columns=groupby_columns,
        summarised_column_types=summarised_column_types,
    )


@pytest.fixture
def sample_data_with_several_group_by_columns() -> pl.DataFrame:
    """Return a sample DataFrame with 2 groupby columns."""
    return pl.DataFrame(
        {
            "group_a": ["A", "B", "C", "A", "B", "C"],
            "group_b": ["X", "X", "X", "Y", "Y", "Y"],
            "f0": [10, 20, 30, 40, 50, 60],
        },
    )


def test_valid_initialisation(summary_table: SummaryTable) -> None:
    """Test that a SummaryTable instance can be created with valid inputs."""
    assert summary_table is not None
    assert summary_table.groupby_columns == ["group"]
    assert summary_table.summarised_column_types == {"value": SummaryOperation.SUM}

    expected_data = pl.DataFrame(
        {
            "group": ["A", "B", "C"],
            "value": [10, 20, 30],
        },
    )

    assert_frame_equal(
        summary_table.head(5),
        expected_data,
    )

    assert_frame_equal(
        summary_table.tail(5),
        expected_data,
    )


def test_empty_groupby_columns_raises_error(sample_data: pl.DataFrame) -> None:
    """Test that an error is raised when groupby_columns is empty."""
    with pytest.raises(
        SummaryTableError,
        match=re.escape("Groupby columns must not be empty."),
    ):
        SummaryTable(
            sample_data,
            groupby_columns=[],
            summarised_column_types={"value": SummaryOperation.SUM},
        )


def test_empty_summarised_column_types_raises_error(sample_data: pl.DataFrame) -> None:
    """Test that an error is raised when summarised_column_types is empty."""
    with pytest.raises(
        SummaryTableError,
        match=re.escape("Summarised column types must not be empty."),
    ):
        SummaryTable(sample_data, groupby_columns=["group"], summarised_column_types={})


def test_groupby_columns_not_unique_raises_error(sample_data: pl.DataFrame) -> None:
    """Test that an error is raised when groupby_columns are not unique."""
    with pytest.raises(
        SummaryTableError,
        match=re.escape("Groupby columns must be unique."),
    ):
        SummaryTable(
            sample_data,
            groupby_columns=["group", "group"],
            summarised_column_types={"value": SummaryOperation.SUM},
        )


def test_groupby_and_summarised_columns_overlap_raises_error(
    sample_data: pl.DataFrame,
) -> None:
    """Test if groupby_columns and summarised_column_types overlap error is raised."""
    with pytest.raises(
        SummaryTableError,
        match=re.escape(
            "Groupby columns and summarised columns must not overlap. "
            "Overlapping columns: group.",
        ),
    ):
        SummaryTable(
            sample_data,
            groupby_columns=["group"],
            summarised_column_types={"group": SummaryOperation.SUM},
        )


def test_missing_columns_raise_exception_group(sample_data: pl.DataFrame) -> None:
    """Test that an ExceptionGroup is raised when columns are missing from data."""
    with pytest.raises(
        ExceptionGroup,
        match=re.escape("Missing columns in SummaryTable input data."),
    ) as exc_info:
        SummaryTable(
            sample_data,
            groupby_columns=["missing_group"],
            summarised_column_types={"missing_value": SummaryOperation.SUM},
        )

    assert len(exc_info.value.exceptions) == 2

    expected_missing_columns = [
        "missing_group",
        "missing_value",
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


def test_modifying_groupby_columns_raises_error(summary_table: SummaryTable) -> None:
    """Test that modifying groupby_columns after initialisation raises an error."""
    with pytest.raises(
        SummaryTableModificationError,
        match=re.escape("Cannot modify 'groupby_columns' on SummaryTable instance."),
    ):
        summary_table.groupby_columns = ["new_group"]

    with pytest.raises(
        SummaryTableModificationError,
        match=re.escape("Cannot delete 'groupby_columns' on SummaryTable instance."),
    ):
        del summary_table.groupby_columns


def test_modifying_summarised_column_types_raises_error(
    summary_table: SummaryTable,
) -> None:
    """Test that modifying summarised_column_types raises an error."""
    with pytest.raises(
        SummaryTableModificationError,
        match=re.escape(
            "Cannot modify 'summarised_column_types' on SummaryTable instance.",
        ),
    ):
        summary_table.summarised_column_types = {"new_value": SummaryOperation.SUM}

    with pytest.raises(
        SummaryTableModificationError,
        match=re.escape(
            "Cannot delete 'summarised_column_types' on SummaryTable instance.",
        ),
    ):
        del summary_table.summarised_column_types


def test_n_property_returns_correct_row_count(summary_table: SummaryTable) -> None:
    """Test n property returns the correct number of rows in the summary table."""
    assert summary_table.n == 3


class TestSummaryTableEquality:
    """Test the equality operator for SummaryTable instances."""

    def test_different_types_not_equal(self, summary_table: SummaryTable) -> None:
        """Test that a SummaryTable is not equal to an object of a different type."""
        assert summary_table != "not_a_summary_table"

    def test_equal_summary_tables(self, summary_table: SummaryTable) -> None:
        """Test that two identical SummaryTable instances are equal."""
        identical_table = SummaryTable(
            summary_table.head(5),
            groupby_columns=summary_table.groupby_columns,
            summarised_column_types=summary_table.summarised_column_types,
        )
        assert summary_table == identical_table

    def test_unequal_groupby_columns(self, summary_table: SummaryTable) -> None:
        """Test that SummaryTables with different groupby_columns are not equal."""
        different_groupby = SummaryTable(
            summary_table.head(5).with_columns(
                pl.col("group").alias("different_group"),
            ),
            groupby_columns=["different_group"],
            summarised_column_types=summary_table.summarised_column_types,
        )
        assert summary_table != different_groupby

    def test_unequal_summarised_column_types(self, summary_table: SummaryTable) -> None:
        """Test SummaryTables with different summarised_column_types are not equal."""
        different_summarised = SummaryTable(
            summary_table.head(5),
            groupby_columns=summary_table.groupby_columns,
            summarised_column_types={"value": SummaryOperation.WEIGHTED_AVERAGE},
        )
        assert summary_table != different_summarised

    def test_unequal_data(self, summary_table: SummaryTable) -> None:
        """Test that SummaryTables with different data are not equal."""
        different_data = pl.DataFrame(
            {
                "group": ["A", "B", "C"],
                "value": [100, 200, 300],
            },
        )
        different_data_table = SummaryTable(
            different_data,
            groupby_columns=summary_table.groupby_columns,
            summarised_column_types=summary_table.summarised_column_types,
        )
        assert summary_table != different_data_table


class TestSummaryTableUnpivot:
    """Test the unpivot method of SummaryTable."""

    def test_unpivot_single_groupby_single_value(
        self,
        summary_table: SummaryTable,
    ) -> None:
        """Test that unpivot returns the expected DataFrame."""
        actual = summary_table.unpivot(
            on=["value"],
            index=["group"],
        )

        expected_df = pl.DataFrame(
            {
                "group": ["A", "B", "C"],
                "variable": ["value", "value", "value"],
                "value": [10, 20, 30],
            },
        )

        assert_frame_equal(actual, expected_df)

    def test_unpivot_single_groupby_single_value_rename(
        self,
        summary_table: SummaryTable,
    ) -> None:
        """Test that unpivot returns the expected DataFrame with renamed columns."""
        actual = summary_table.unpivot(
            on=["value"],
            index=["group"],
            variable_name="var",
            value_name="val",
        )

        expected_df = pl.DataFrame(
            {
                "group": ["A", "B", "C"],
                "var": ["value", "value", "value"],
                "val": [10, 20, 30],
            },
        )

        assert_frame_equal(actual, expected_df)

    def test_unpivot_multiple_value_columns(
        self,
        summary_table_with_several_value_columns: SummaryTable,
    ) -> None:
        """Test that unpivot works with multiple value columns."""
        actual = summary_table_with_several_value_columns.unpivot(
            on=["f0", "f1", "f2"],
            index=["group"],
        )

        expected_df = pl.DataFrame(
            {
                "group": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
                "variable": ["f0", "f0", "f0", "f1", "f1", "f1", "f2", "f2", "f2"],
                "value": [10, 20, 30, 100, 200, 300, 1000, 2000, 3000],
            },
        )

        assert_frame_equal(actual, expected_df)

    def test_unpivot_multiple_groupby_columns(
        self,
        sample_data_with_several_group_by_columns: pl.DataFrame,
    ) -> None:
        """Test that unpivot works with multiple groupby columns."""
        summary_table = SummaryTable(
            sample_data_with_several_group_by_columns,
            groupby_columns=["group_a", "group_b"],
            summarised_column_types={"f0": SummaryOperation.SUM},
        )

        actual = summary_table.unpivot(
            on=["f0"],
            index=["group_a", "group_b"],
        )

        expected_df = pl.DataFrame(
            {
                "group_a": ["A", "B", "C", "A", "B", "C"],
                "group_b": ["X", "X", "X", "Y", "Y", "Y"],
                "variable": ["f0"] * 6,
                "value": [10, 20, 30, 40, 50, 60],
            },
        )

        assert_frame_equal(actual, expected_df)
