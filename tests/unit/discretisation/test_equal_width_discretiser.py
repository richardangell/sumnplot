"""Unit tests for the EqualWidthDiscretiser class."""

import re

import polars as pl
import pytest
from polars.testing import assert_series_equal

from sumnplot.discretisation.base_discretiser import BaseDiscretiserError
from sumnplot.discretisation.equal_width_discretiser import EqualWidthDiscretiser
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
def sample_df_with_float_values():
    """Fixture to create a sample DataFrame for testing.

    In this fixture the 'value' column is float.

    """
    return pl.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
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
def sample_df_with_nans():
    """Fixture to create a sample DataFrame for testing with NaNs in the values.

    The NaN row has weight so without removing the output from discretisation would
    be different to the sample_df fixture.

    """
    return pl.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, float("nan"), 6.0, 7.0, 8.0, 9.0, 10.0],
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
    """Tests for the initialisation of the EqualWidthDiscretiser class."""

    def test_initialisation_with_valid_arguments(self):
        """Test that the EqualWidthDiscretiser can be initialised with valid args."""
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=5,
            new_name="value_bin",
        )
        assert discretiser.column == "value"
        assert discretiser.weights == "weight"
        assert discretiser.n_bins == 5
        assert discretiser.new_name == "value_bin"

        assert discretiser.breaks is None

    def test_initialisation_with_invalid_n_bins(self):
        """Test that an exception is raised for invalid n_bins."""
        expected_message = (
            "Invalid value for argument 'n_bins': 0. "
            "The following conditions were not met: > 0."
        )
        with pytest.raises(InvalidArgumentError, match=expected_message):
            EqualWidthDiscretiser(
                column="value",
                weights="weight",
                n_bins=0,
                new_name="value_bin",
            )


class TestFit:
    """Tests for the fit method of the EqualWidthDiscretiser class."""

    def test_missing_values_column_exception(self) -> None:
        """Test that MissingColumnError is raised when the values column is missing."""
        df = pl.DataFrame({"f0": [1, 2, 3], "w": [1, 1, 1]})

        discretiser = EqualWidthDiscretiser(
            column="missing",
            weights="w",
            n_bins=5,
        )

        with pytest.raises(
            MissingColumnError,
            match=re.escape("Column 'missing' not found in DataFrame."),
        ):
            discretiser.fit(df)

    def test_missing_weights_column_exception(self) -> None:
        """Test that MissingColumnError is raised when the weights column is missing."""
        df = pl.DataFrame({"f0": [1, 2, 3], "w": [1, 1, 1]})

        discretiser = EqualWidthDiscretiser(
            column="f0",
            weights="missing_w",
            n_bins=5,
        )

        with pytest.raises(
            MissingColumnError,
            match=re.escape("Column 'missing_w' not found in DataFrame."),
        ):
            discretiser.fit(df)

    def test_non_numeric_values_column_exception(self) -> None:
        """Test that exception is raised when the values column is not numeric."""
        df = pl.DataFrame({"f0": ["a", "b", "c"], "w": [1, 1, 1]})

        discretiser = EqualWidthDiscretiser(
            column="f0",
            weights="w",
            n_bins=5,
        )

        with pytest.raises(
            NumericColumnError,
            match=re.escape("Column 'f0' must be a numeric dtype."),
        ):
            discretiser.fit(df)

    def test_non_numeric_weights_column_exception(self) -> None:
        """Test that exception is raised when the weights column is not numeric."""
        df = pl.DataFrame({"f0": [1, 2, 3], "w": ["a", "b", "c"]})

        discretiser = EqualWidthDiscretiser(
            column="f0",
            weights="w",
            n_bins=5,
        )

        with pytest.raises(
            NumericColumnError,
            match=re.escape("Column 'w' must be a numeric dtype."),
        ):
            discretiser.fit(df)

    def test_fit_with_valid_data(self, sample_df: pl.DataFrame):
        """Test that the fit method calculates the correct cut points for valid data."""
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=3,
        )

        discretiser.fit(sample_df)

        assert discretiser.breaks is not None
        assert discretiser.breaks == [4, 7]

    def test_nulls_do_not_affect_calculated_breaks(
        self,
        sample_df_with_nulls: pl.DataFrame,
    ):
        """Test that nulls are effectively removed when calculating breaks."""
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=3,
        )

        discretiser.fit(sample_df_with_nulls)

        assert discretiser.breaks is not None
        assert discretiser.breaks == [4, 7]

    def test_float_values_do_not_have_integer_breaks(
        self,
        sample_df_with_float_values: pl.DataFrame,
    ):
        """Test that break points calculated on a float column are float.

        This applies even if the break points are whole numbers.

        """
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=3,
        )

        discretiser.fit(sample_df_with_float_values)

        assert discretiser.breaks is not None
        assert discretiser.breaks == [4.0, 7.0]

    def test_nans_do_not_affect_calculated_breaks(
        self,
        sample_df_with_nans: pl.DataFrame,
    ):
        """Test that NaNs are effectively removed when calculating breaks."""
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=3,
        )

        discretiser.fit(sample_df_with_nans)

        assert discretiser.breaks is not None
        assert discretiser.breaks == [4.0, 7.0]

    def test_non_whole_number_breaks(self, sample_df: pl.DataFrame):
        """Test that break points from an integer column that are not whole numbers.

        In this case each bucket should have 2.25 (= (10 - 1) / 4) width.

        """
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=4,
        )

        discretiser.fit(sample_df)

        assert discretiser.breaks is not None
        assert discretiser.breaks == [3.25, 5.5, 7.75]


class TestTransform:
    """Tests for the transform method of the EqualWidthDiscretiser class."""

    def test_transform_without_fit(self, sample_df: pl.DataFrame):
        """Test that exception is raised when transform is called before fit."""
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=5,
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
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=3,
            new_name=None,
        )

        discretiser.fit(sample_df)
        transformed_df = discretiser.transform(sample_df)

        assert transformed_df.shape == (10, 2)

        expected_discretised_enum = pl.Enum(["(-inf, 4]", "(4, 7]", "(7, inf]"])
        assert transformed_df["value"].dtype == expected_discretised_enum

        actual_levels = transformed_df.get_column("value").to_physical()
        expected_levels = pl.Series(
            "value",
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            dtype=pl.UInt8,
        )

        assert_series_equal(actual_levels, expected_levels)

    def test_transform_with_valid_data_new_column(self, sample_df: pl.DataFrame):
        """Test correct discretisation and creation of a new column."""
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=3,
            new_name="value_bin",
        )

        discretiser.fit(sample_df)
        transformed_df = discretiser.transform(sample_df)

        assert transformed_df.shape == (10, 3)
        assert "value_bin" in transformed_df.columns

        expected_discretised_enum = pl.Enum(["(-inf, 4]", "(4, 7]", "(7, inf]"])
        assert transformed_df["value_bin"].dtype == expected_discretised_enum

        actual_levels = transformed_df.get_column("value_bin").to_physical()
        expected_levels = pl.Series(
            "value_bin",
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
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
        discretiser = EqualWidthDiscretiser(
            column="value",
            weights="weight",
            n_bins=3,
            new_name=None,
        )

        discretiser.fit(sample_df)
        transformed_df = discretiser.transform(sample_df_larger_range)

        assert transformed_df.shape == (10, 2)

        expected_discretised_enum = pl.Enum(["(-inf, 4]", "(4, 7]", "(7, inf]"])
        assert transformed_df["value"].dtype == expected_discretised_enum

        actual_levels = transformed_df.get_column("value").to_physical()
        expected_levels = pl.Series(
            "value",
            [0, 0, 0, 0, 0, 1, 2, 2, 2, 2],
            dtype=pl.UInt8,
        )

        assert_series_equal(actual_levels, expected_levels)
