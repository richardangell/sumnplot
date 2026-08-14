"""Equal width discretiser module."""

import polars as pl

from sumnplot.discretisation.base_discretiser import BaseDiscretiser
from sumnplot.exceptions import (
    InvalidArgumentError,
    MissingColumnError,
    NumericColumnError,
    SumNPlotError,
)


class EqualWidthDiscretiserError(SumNPlotError):
    """Exception raised for errors in the EqualWidthDiscretiser."""


class EqualWidthDiscretiser(BaseDiscretiser):
    """Discretiser that divides a column into bins of equal width."""

    def __init__(
        self,
        *,
        column: str,
        weights: str,
        n_bins: int,
        new_name: str | None = None,
        round_breaks_for_labels: bool = False,
    ) -> None:
        """Initialise the EqualWidthDiscretiser.

        Args:
            column (str): The name of the column to discretise.
            weights (str): The name of the weights column.
            n_bins (int): The number of bins to divide the column into.
            new_name (str | None): The name of the new column to be created. If None,
                the original column name will be used.
            round_breaks_for_labels (bool): Whether to round the break points for
                labels. If True, the break points will be rounded to the lowest
                precision to keep each unique value. If False, the break points will
                not be rounded.

        Raises:
            InvalidArgumentError: If n_bins is less than or equal to 0.

        """
        if n_bins <= 0:
            raise InvalidArgumentError(
                argument="n_bins",
                value=n_bins,
                conditions=["> 0"],
            )

        self.column = column
        self.weights = weights
        self.n_bins = n_bins
        self.new_name = new_name
        self.round_breaks_for_labels = round_breaks_for_labels

        self._breaks = None

    def fit(self, df: pl.DataFrame) -> "EqualWidthDiscretiser":
        """Calculate the cut points for discretisation based on equal width.

        Args:
            df (pl.DataFrame): The polars DataFrame containing the column to calculate
            cut points for.

        Returns:
            EqualWidthDiscretiser: The fitted EqualWidthDiscretiser instance.

        """
        if self.column not in df.columns:
            raise MissingColumnError(self.column)

        if self.weights not in df.columns:
            raise MissingColumnError(self.weights)

        if not df.get_column(self.column).dtype.is_numeric():
            raise NumericColumnError(self.column)

        if not df.get_column(self.weights).dtype.is_numeric():
            raise NumericColumnError(self.weights)

        min_value = df.get_column(self.column).min()
        max_value = df.get_column(self.column).max()

        breaks_inclusive_of_ends = pl.linear_space(
            start=min_value,
            end=max_value,
            num_samples=self.n_bins + 1,
            eager=True,
            closed="both",
        )

        # For integer try to convert break points to integer, if possible.
        if df.get_column(self.column).dtype.is_integer():
            breaks_integer = breaks_inclusive_of_ends.cast(pl.Int64)

            if (breaks_integer == breaks_inclusive_of_ends).all():
                breaks_inclusive_of_ends = breaks_integer

        self.breaks = breaks_inclusive_of_ends.to_list()[1:-1]

        return self
