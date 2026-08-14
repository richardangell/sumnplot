"""Equal weight discretiser module."""

import polars as pl

from sumnplot.discretisation.base_discretiser import BaseDiscretiser
from sumnplot.discretisation.minimum_weight_calculation import (
    threshold_crossing_minimum_weight,
)
from sumnplot.exceptions import (
    InvalidArgumentError,
    MissingColumnError,
    NumericColumnError,
    SumNPlotError,
)


class EqualWeightDiscretiserError(SumNPlotError):
    """Exception raised for errors in the EqualWeightDiscretiser."""


class EqualWeightDiscretiser(BaseDiscretiser):
    """Discretiser that divides a column into bins of equal weight."""

    def __init__(
        self,
        *,
        column: str,
        weights: str,
        min_weight_proportion: float,
        new_name: str | None = None,
        round_breaks_for_labels: bool = False,
    ) -> None:
        """Initialise the EqualWeightDiscretiser.

        Args:
            column : The name of the column to discretise.
            weights : The name of the weights column.
            min_weight_proportion : The minimum proportion of the total weight for
            each bin. Must be between 0 and 1 (exclusive).
            new_name : The name of the new column to be created. If None, the original
                column name will be used.
            round_breaks_for_labels : Whether to round the break points for labels. If
                True, the break points will be rounded to the lowest precision to
                keep each unique value. If False, the break points will not be rounded.

        """
        if not (0 < min_weight_proportion <= 1):
            raise InvalidArgumentError(
                argument="min_weight_proportion",
                value=min_weight_proportion,
                conditions=["> 0", "<= 1"],
            )

        self.column = column
        self.weights = weights
        self.min_weight_proportion = min_weight_proportion
        self.new_name = new_name
        self.round_breaks_for_labels = round_breaks_for_labels

        self._breaks = None

    def fit(self, df: pl.DataFrame) -> "EqualWeightDiscretiser":
        """Calculate the cut points for discretisation based on equal weight.

        Args:
            df : The polars DataFrame containing the column to calculate cut points for.

        Returns:
            self : The fitted EqualWeightDiscretiser instance.

        """
        if self.column not in df.columns:
            raise MissingColumnError(self.column)

        if self.weights not in df.columns:
            raise MissingColumnError(self.weights)

        if not df.get_column(self.column).dtype.is_numeric():
            raise NumericColumnError(self.column)

        if not df.get_column(self.weights).dtype.is_numeric():
            raise NumericColumnError(self.weights)

        has_nulls = df.get_column(self.column).has_nulls()

        df_without_nulls = (
            df.filter(pl.col(self.column).is_not_null()).select(
                self.column,
                self.weights,
            )
            if has_nulls
            else df
        )

        total_weight = df_without_nulls.get_column(self.weights).sum()

        self.breaks = threshold_crossing_minimum_weight(
            df=df_without_nulls,
            column=self.column,
            weights=self.weights,
            min_weight=self.min_weight_proportion * total_weight,  # pyright: ignore[reportOperatorIssue]
        )

        return self
