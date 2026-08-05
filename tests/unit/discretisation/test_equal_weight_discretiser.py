"""Unit tests for the EqualWeightDiscretiser class."""

import re

import polars as pl
import pytest
from polars.testing import assert_series_equal

from sumnplot.discretisation.base_discretiser import BaseDiscretiserError
from sumnplot.discretisation.equal_weight_discretiser import EqualWeightDiscretiser
from sumnplot.exceptions import (
    InvalidArgumentError,
    MissingColumnError,
    NumericColumnError,
)


@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    )


@pytest.fixture
def sample_df_with_nulls():
    """Fixture to create a sample DataFrame for testing with nulls in the values.

    The null row has weight so without removing the output from discretisation would
    be different to the sample_df fixture.

    """
    return pl.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, None, 6, 7, 8, 9, 10],
            "weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    )


@pytest.fixture
def sample_df_larger_range():
    """Fixture to create a sample DataFrame for testing.

    Has wider range than sample_df but does not include every integer value in the
    range.

    """
    return pl.DataFrame(
        {
            "value": [-5, -4, -1, 0, 4, 5, 9, 10, 11, 12],
            "weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    )


class TestInitialisation:
    """Tests for the initialisation of the EqualWeightDiscretiser class."""

    def test_initialisation_with_valid_arguments(self):
        """Test that the EqualWeightDiscretiser can be initialised with valid args."""
        discretiser = EqualWeightDiscretiser(
            column="value",
            weights="weight",
            min_weight_proportion=0.3,
            new_name="value_bin",
        )
        assert discretiser.column == "value"
        assert discretiser.weights == "weight"
        assert discretiser.min_weight_proportion == 0.3
        assert discretiser.new_name == "value_bin"

        assert discretiser.breaks is None

    def test_initialisation_with_invalid_min_weight_proportion(self):
        """Test that an exception is raised for invalid min_weight_proportion."""
        expected_message = (
            "Invalid value for argument 'min_weight_proportion': 1.5. "
            "The following conditions were not met: > 0, <= 1."
        )
        with pytest.raises(InvalidArgumentError, match=expected_message):
            EqualWeightDiscretiser(
                column="value",
                weights="weight",
                min_weight_proportion=1.5,
                new_name="value_bin",
            )


class TestFit:
    """Tests for the fit method of the EqualWeightDiscretiser class."""

    def test_missing_values_column_exception(self) -> None:
        """Test that MissingColumnError is raised when the values column is missing."""
        df = pl.DataFrame({"f0": [1, 2, 3], "w": [1, 1, 1]})

        discretiser = EqualWeightDiscretiser(
            column="missing",
            weights="w",
            min_weight_proportion=0.3,
        )

        with pytest.raises(
            MissingColumnError,
            match=re.escape("Column 'missing' not found in DataFrame."),
        ):
            discretiser.fit(df)

    def test_missing_weights_column_exception(self) -> None:
        """Test that MissingColumnError is raised when the weights column is missing."""
        df = pl.DataFrame({"f0": [1, 2, 3], "w": [1, 1, 1]})

        discretiser = EqualWeightDiscretiser(
            column="f0",
            weights="missing_w",
            min_weight_proportion=0.3,
        )

        with pytest.raises(
            MissingColumnError,
            match=re.escape("Column 'missing_w' not found in DataFrame."),
        ):
            discretiser.fit(df)

    def test_non_numeric_values_column_exception(self) -> None:
        """Test that exception is raised when the values column is not numeric."""
        df = pl.DataFrame({"f0": ["a", "b", "c"], "w": [1, 1, 1]})

        discretiser = EqualWeightDiscretiser(
            column="f0",
            weights="w",
            min_weight_proportion=0.3,
        )

        with pytest.raises(
            NumericColumnError,
            match=re.escape("Column 'f0' must be a numeric dtype."),
        ):
            discretiser.fit(df)

    def test_non_numeric_weights_column_exception(self) -> None:
        """Test that exception is raised when the weights column is not numeric."""
        df = pl.DataFrame({"f0": [1, 2, 3], "w": ["a", "b", "c"]})

        discretiser = EqualWeightDiscretiser(
            column="f0",
            weights="w",
            min_weight_proportion=0.3,
        )

        with pytest.raises(
            NumericColumnError,
            match=re.escape("Column 'w' must be a numeric dtype."),
        ):
            discretiser.fit(df)

    def test_fit_with_valid_data(self, sample_df: pl.DataFrame):
        """Test that the fit method calculates the correct cut points for valid data."""
        discretiser = EqualWeightDiscretiser(
            column="value",
            weights="weight",
            min_weight_proportion=0.3,
        )

        discretiser.fit(sample_df)

        assert discretiser.breaks is not None
        assert discretiser.breaks == [3, 6, 9]

    def test_nulls_removed_before_calculating_breaks(
        self,
        sample_df_with_nulls: pl.DataFrame,
    ):
        """Test that nulls are removed before calculating breaks."""
        discretiser = EqualWeightDiscretiser(
            column="value",
            weights="weight",
            min_weight_proportion=0.3,
        )

        discretiser.fit(sample_df_with_nulls)

        assert discretiser.breaks is not None
        assert discretiser.breaks == [3, 6, 9]


class TestTransform:
    """Tests for the transform method of the EqualWeightDiscretiser class."""

    def test_transform_without_fit(self, sample_df: pl.DataFrame):
        """Test that exception is raised when transform is called before fit."""
        discretiser = EqualWeightDiscretiser(
            column="value",
            weights="weight",
            min_weight_proportion=0.3,
        )

        with pytest.raises(
            BaseDiscretiserError,
            match=re.escape(
                (
                    "Break points have not been calculated. "
                    "Please call the `fit` method first."
                ),
            ),
        ):
            discretiser.transform(sample_df)

    def test_transform_with_valid_data(self, sample_df: pl.DataFrame):
        """Test correct discretisation and overwrite of an existing column."""
        discretiser = EqualWeightDiscretiser(
            column="value",
            weights="weight",
            min_weight_proportion=0.3,
            new_name=None,
        )

        discretiser.fit(sample_df)
        transformed_df = discretiser.transform(sample_df)

        assert transformed_df.shape == (10, 2)

        expected_discretised_enum = pl.Enum(
            ["(-inf, 3]", "(3, 6]", "(6, 9]", "(9, inf]"],
        )
        assert transformed_df["value"].dtype == expected_discretised_enum

        actual_levels = transformed_df.get_column("value").to_physical()
        expected_levels = pl.Series(
            "value",
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            dtype=pl.UInt8,
        )

        assert_series_equal(actual_levels, expected_levels)

    def test_transform_with_valid_data_new_column(self, sample_df: pl.DataFrame):
        """Test correct discretisation and creation of a new column."""
        discretiser = EqualWeightDiscretiser(
            column="value",
            weights="weight",
            min_weight_proportion=0.3,
            new_name="value_bin",
        )

        discretiser.fit(sample_df)
        transformed_df = discretiser.transform(sample_df)

        assert transformed_df.shape == (10, 3)
        assert "value_bin" in transformed_df.columns

        expected_discretised_enum = pl.Enum(
            ["(-inf, 3]", "(3, 6]", "(6, 9]", "(9, inf]"],
        )
        assert transformed_df["value_bin"].dtype == expected_discretised_enum

        actual_levels = transformed_df.get_column("value_bin").to_physical()
        expected_levels = pl.Series(
            "value_bin",
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            dtype=pl.UInt8,
        )

        assert_series_equal(actual_levels, expected_levels)

    def test_transform_with_data_outside_fit_range(
        self,
        sample_df: pl.DataFrame,
        sample_df_larger_range: pl.DataFrame,
    ):
        """Test transform with values outside the fit range.

        Values outside the range seen during fit should be assigned to the first
        or last bin as appropriate.

        """
        discretiser = EqualWeightDiscretiser(
            column="value",
            weights="weight",
            min_weight_proportion=0.3,
            new_name=None,
        )

        discretiser.fit(sample_df)
        transformed_df = discretiser.transform(sample_df_larger_range)

        assert transformed_df.shape == (10, 2)

        expected_discretised_enum = pl.Enum(
            ["(-inf, 3]", "(3, 6]", "(6, 9]", "(9, inf]"],
        )
        assert transformed_df["value"].dtype == expected_discretised_enum

        actual_levels = transformed_df.get_column("value").to_physical()
        expected_levels = pl.Series(
            "value",
            [0, 0, 0, 0, 1, 1, 2, 3, 3, 3],
            dtype=pl.UInt8,
        )

        assert_series_equal(actual_levels, expected_levels)
