"""Unit tests for the group_by_weighted_average function."""

import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sumnplot.exceptions import MissingColumnError
from sumnplot.summarisation.group_by_weighted_average import (
    GroupByWeightedAverageError,
    ResponseWeight,
    group_by_weighted_average,
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "group2": ["Y", "X", "Y", "X"],
            "value": [10, 20, 30, 40],
            "weight": [1, 2, 1, 3],
        },
    )


@pytest.fixture
def sample_data_with_enum() -> pl.DataFrame:
    """Create a sample DataFrame for testing where the group by columns are pl.Enums."""
    return pl.DataFrame(
        {
            "value": [10, 20, 30, 40],
            "weight": [1, 2, 1, 3],
        },
    ).with_columns(
        pl.Series(
            name="group",
            values=["A", "A", "B", "B"],
            dtype=pl.Enum(["A", "B", "C"]),
        ),
    )


@pytest.fixture
def sample_data_multiple_enums() -> pl.DataFrame:
    """Create a sample DataFrame for testing where the group by columns are pl.Enums."""
    return pl.DataFrame(
        {
            "value": [10, 20, 30, 40],
            "weight": [1, 2, 1, 3],
        },
    ).with_columns(
        [
            pl.Series(
                name="group",
                values=["A", "A", "B", "B"],
                dtype=pl.Enum(["A", "B", "C"]),
            ),
            pl.Series(
                name="group2",
                values=["X", "Y", "X", "Y"],
                dtype=pl.Enum(["X", "Y", "Z"]),
            ),
        ],
    )


@pytest.fixture
def sample_data_missing_combinations() -> pl.DataFrame:
    """Create a sample DataFrame with missing level combinations."""
    return pl.DataFrame(
        {
            "group": ["A", "A", "B", "C"],
            "group2": ["Y", "Y", "X", "X"],
            "value": [10, 20, 30, 40],
            "weight": [1, 2, 1, 3],
        },
    )


@pytest.fixture
def sample_data_with_mixed_group_by_columns() -> pl.DataFrame:
    """Return data with pl.Enum, pl.Categorical, and pl.String groupby columns."""
    return pl.DataFrame(
        {
            "value": [10, 20, 30, 40],
            "weight": [1, 2, 1, 3],
        },
    ).with_columns(
        [
            pl.Series(
                name="group",
                values=["A", "A", "B", "B"],
                dtype=pl.Enum(["A", "B", "C"]),
            ),
            pl.Series(name="group2", values=["X", "Y", "X", "Y"], dtype=pl.Categorical),
            pl.Series(
                name="group3",
                values=["foo", "foo", "foo", "bar"],
                dtype=pl.String,
            ),
        ],
    )


@pytest.fixture
def sample_data_with_multiple_responses() -> pl.DataFrame:
    """Create a sample DataFrame with multiple response columns.

    This data has 3 responses and 2 weights, with 2 groupby columns.

    """
    return pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value1": [10, 20, 30, 40],
            "weight1": [1, 2, 1, 3],
            "value2": [100, 200, 300, 400],
            "weight2": [2, 1, 3, 1],
            "value3": [10, 20, 30, 40],
        },
    )


def test_missing_group_by_columns_exception(sample_data: pl.DataFrame):
    """Test that ExceptionGroup is raised for missing groupby columns."""
    groupby_columns = ["group", "missing_col"]
    responses = [ResponseWeight(response="value", weight="weight")]

    with pytest.raises(ExceptionGroup) as exc_info:
        group_by_weighted_average(
            sample_data,
            groupby_columns=groupby_columns,
            responses=responses,
        )

    raised_exceptions = exc_info.value.exceptions

    assert len(raised_exceptions) == 1
    assert isinstance(raised_exceptions[0], MissingColumnError)
    assert (
        raised_exceptions[0].message == "Column 'missing_col' not found in DataFrame."
    )


def test_missing_response_weight_columns_exception(
    subtests: pytest.Subtests,
    sample_data: pl.DataFrame,
):
    """Test that ExceptionGroup is raised for missing response or weight columns."""
    groupby_columns = ["group"]
    responses = [
        ResponseWeight(response="missing_response", weight="weight"),
        ResponseWeight(response="value", weight="missing_weight"),
    ]

    with pytest.raises(ExceptionGroup) as exc_info:
        group_by_weighted_average(
            sample_data,
            groupby_columns=groupby_columns,
            responses=responses,
        )

    raised_exceptions = exc_info.value.exceptions

    expected_exception_messages = [
        "Column 'missing_response' not found in DataFrame.",
        "Column 'missing_weight' not found in DataFrame.",
    ]

    assert len(raised_exceptions) == 2

    for i in range(2):
        with subtests.test(f"Exception {i}"):
            exception = raised_exceptions[i]
            assert isinstance(exception, MissingColumnError)
            assert exception.message == expected_exception_messages[i]


def test_duplicate_responses_exception(sample_data: pl.DataFrame):
    """Test that GroupByWeightedAverageError is raised for duplicate responses."""
    groupby_columns = ["group"]
    responses = [
        ResponseWeight(response="value", weight="weight"),
        ResponseWeight(response="value", weight="weight"),
    ]

    with pytest.raises(
        GroupByWeightedAverageError,
        match=re.escape(
            "Duplicate responses found in the responses list. "
            "Please ensure all columns are unique.",
        ),
    ):
        group_by_weighted_average(
            sample_data,
            groupby_columns=groupby_columns,
            responses=responses,
        )


def test_output_single_groupby_column(sample_data: pl.DataFrame):
    """Test the output of group_by_weighted_average with a single groupby column.

    No level combinations missing from the input data.

    """
    groupby_columns = ["group"]
    responses = [ResponseWeight(response="value", weight="weight")]

    result = group_by_weighted_average(
        sample_data,
        groupby_columns=groupby_columns,
        responses=responses,
    )

    expected = pl.DataFrame(
        {
            "group": ["A", "B"],
            "value": [50 / 3, 150 / 4],
            "weight": [3, 4],
        },
    )

    assert_frame_equal(result, expected)


def test_output_multiple_groupby_columns(sample_data: pl.DataFrame):
    """Test the output of group_by_weighted_average with multiple groupby columns.

    No level combinations missing from the input data.

    """
    groupby_columns = ["group", "group2"]
    responses = [ResponseWeight(response="value", weight="weight")]

    result = group_by_weighted_average(
        sample_data,
        groupby_columns=groupby_columns,
        responses=responses,
    )

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "group2": ["X", "Y", "X", "Y"],
            "value": [20.0, 10.0, 40.0, 30.0],
            "weight": [2, 1, 3, 1],
        },
    )

    assert_frame_equal(result, expected)


def test_output_with_missing_level_in_single_enum_groupby_column(
    sample_data_with_enum: pl.DataFrame,
):
    """Test the output of group_by_weighted_average with pl.Enum groupby columns.

    In this example one of the Enum levels ("C") is not present in the input DataFrame,
    but it should be included in the output with a weight sum of 0 and a null weighted
    average.

    """
    groupby_columns = ["group"]
    responses = [ResponseWeight(response="value", weight="weight")]

    result = group_by_weighted_average(
        sample_data_with_enum,
        groupby_columns=groupby_columns,
        responses=responses,
    )

    expected = pl.DataFrame(
        [
            pl.Series(
                name="group",
                values=["A", "B", "C"],
                dtype=pl.Enum(["A", "B", "C"]),
            ),
            pl.Series(name="value", values=[50 / 3, 150 / 4, None]),
            pl.Series(name="weight", values=[3, 4, 0]),
        ],
    )

    assert_frame_equal(result, expected)


def test_output_with_missing_levels_in_multiple_enum_groupby_columns(
    sample_data_multiple_enums: pl.DataFrame,
):
    """Test group_by_weighted_average with multiple pl.Enum groupby columns.

    In this example one of the Enum levels ("C" in "group" and "Z" in "group2") is not
    present in the input DataFrame, but it should be included in the output with a
    weight sum of 0 and a null weighted average.

    """
    groupby_columns = ["group", "group2"]
    responses = [ResponseWeight(response="value", weight="weight")]

    result = group_by_weighted_average(
        sample_data_multiple_enums,
        groupby_columns=groupby_columns,
        responses=responses,
    )

    expected = pl.DataFrame(
        [
            pl.Series(
                name="group",
                values=["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                dtype=pl.Enum(["A", "B", "C"]),
            ),
            pl.Series(
                name="group2",
                values=["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"],
                dtype=pl.Enum(["X", "Y", "Z"]),
            ),
            pl.Series(
                name="value",
                values=[10.0, 20.0, None, 30.0, 40.0, None, None, None, None],
            ),
            pl.Series(name="weight", values=[1, 2, 0, 1, 3, 0, 0, 0, 0]),
        ],
    )

    assert_frame_equal(result, expected)


def test_output_with_missing_levels_in_multiple_groupby_columns(
    sample_data_missing_combinations: pl.DataFrame,
):
    """Test the output of group_by_weighted_average with missing level combinations.

    In this example the combination ("B", "Y") is not present in the input DataFrame,
    but it should be included in the output with a weight sum of 0 and a null weighted
    average.

    """
    groupby_columns = ["group", "group2"]
    responses = [ResponseWeight(response="value", weight="weight")]

    result = group_by_weighted_average(
        sample_data_missing_combinations,
        groupby_columns=groupby_columns,
        responses=responses,
    )

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C"],
            "group2": ["X", "Y", "X", "Y", "X", "Y"],
            "value": [None, 50 / 3, 30.0, None, 40.0, None],
            "weight": [0, 3, 1, 0, 3, 0],
        },
    )

    assert_frame_equal(result, expected)


def test_three_way_group_by(sample_data_with_mixed_group_by_columns: pl.DataFrame):
    """Test the output of group_by_weighted_average with three groupby columns.

    In this example the combination ("B", "Y", "foo") is not present in the input
    DataFrame, but it should be included in the output with a weight sum of 0 and a
    null weighted average.

    """
    groupby_columns = ["group", "group2", "group3"]
    responses = [ResponseWeight(response="value", weight="weight")]

    result = group_by_weighted_average(
        sample_data_with_mixed_group_by_columns,
        groupby_columns=groupby_columns,
        responses=responses,
    )

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "group2": ["X", "X", "Y", "Y", "X", "X", "Y", "Y", "X", "X", "Y", "Y"],
            "group3": [
                "bar",
                "foo",
                "bar",
                "foo",
                "bar",
                "foo",
                "bar",
                "foo",
                "bar",
                "foo",
                "bar",
                "foo",
            ],
            "value": [
                None,
                10.0,
                None,
                20.0,
                None,
                30.0,
                40.0,
                None,
                None,
                None,
                None,
                None,
            ],
            "weight": [0, 1, 0, 2, 0, 1, 3, 0, 0, 0, 0, 0],
        },
    ).with_columns(
        pl.col("group").cast(pl.Enum(["A", "B", "C"])),
        pl.col("group2").cast(pl.Categorical),
    )

    assert_frame_equal(result, expected)


def test_multiple_responses(sample_data_with_multiple_responses: pl.DataFrame):
    """Test the output of group_by_weighted_average with multiple response columns.

    In this example the combination ("B", "Y") is not present in the input DataFrame,
    but it should be included in the output with a weight sum of 0 and a null weighted
    average.

    """
    groupby_columns = ["group"]
    responses = [
        ResponseWeight(response="value1", weight="weight1"),
        ResponseWeight(response="value2", weight="weight2"),
        ResponseWeight(response="value3", weight="weight2"),
    ]

    result = group_by_weighted_average(
        sample_data_with_multiple_responses,
        groupby_columns=groupby_columns,
        responses=responses,
    )

    expected = pl.DataFrame(
        {
            "group": ["A", "B"],
            "value1": [50 / 3, 150 / 4],
            "value2": [400 / 3, 1300 / 4],
            "value3": [40 / 3, 130 / 4],
            "weight1": [3, 4],
            "weight2": [3, 4],
        },
    )

    assert_frame_equal(result, expected)
