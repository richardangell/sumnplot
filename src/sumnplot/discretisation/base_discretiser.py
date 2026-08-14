"""Base discretiser abstract base class."""

from abc import ABC, abstractmethod
from typing import Sequence

import polars as pl

from sumnplot.discretisation.rounding import (
    determine_required_precision_to_keep_values_distinct,
)
from sumnplot.exceptions import MissingColumnError, NumericColumnError, SumNPlotError


class BaseDiscretiserError(SumNPlotError):
    """Exception raised for errors in the BaseDiscretiser."""


def generate_cut_expr(
    *,
    column: str,
    breaks: Sequence[int | float],
    labels: Sequence[str] | None = None,
    new_name: str | None = None,
) -> pl.Expr:
    """Generate a polars expression to discretise a column into bins.

    Args:
        column (str): The name of the column to discretise.
        breaks (Sequence[int | float]): The break points for discretisation.
        labels (Sequence[str] | None): The labels for the bins. If None, the bin
        indices will be used.
        new_name (str | None): The name of the new column to be created. If None, the
            original column name will be used.

    Returns:
        pl.Expr: The polars expression that can be used to discretise the column.

    """
    return (
        pl.col(column)
        .cut(breaks=breaks, labels=labels, left_closed=False, include_breaks=False)
        .alias(new_name if new_name is not None else column)
    )


def generate_bin_labels(
    breaks: Sequence[int | float],
    *,
    left_closed: bool = False,
) -> list[str]:
    """Generate labels for bins based on break points.

    Args:
        breaks (Sequence[int | float]): The break points for discretisation.
        left_closed (bool): Whether the intervals are left-closed or right-closed.

    Returns:
        list[str]: A list of labels for the bins.

    """
    if left_closed:
        left_bracket = "["
        right_bracket = ")"
    else:
        left_bracket = "("
        right_bracket = "]"

    first_label = f"{left_bracket}-inf, {breaks[0]}{right_bracket}"
    last_label = f"{left_bracket}{breaks[-1]}, inf{right_bracket}"
    labels = [first_label]
    for i in range(len(breaks) - 1):
        labels.append(f"{left_bracket}{breaks[i]}, {breaks[i + 1]}{right_bracket}")
    labels.append(last_label)
    return labels


class BaseDiscretiser(ABC):
    """Abstract base class for discretisers."""

    column: str
    weights: str
    min_weight_proportion: float
    new_name: str | None
    round_breaks_for_labels: bool
    _breaks: list[int | float] | None

    @property
    def breaks(self) -> list[int | float] | None:
        """Get the calculated cut points for discretisation."""
        return self._breaks

    @breaks.setter
    def breaks(self, value: list[int | float] | None) -> None:
        """Set the calculated cut points for discretisation.

        Remove duplicates and ensures the cut points are sorted in ascending order.

        """
        if isinstance(value, list):
            value = list(set(value))
            value = sorted(value)

        self._breaks = value

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> "BaseDiscretiser":
        """Calculate the cut points for discretisation.

        Cut points should be calculated using the provided data and stored in the
        instance for later use.

        Args:
            df (pl.DataFrame): The polars DataFrame containing the column to calculate
                cut points for.

        """
        ...

    def get_cut_expr(self) -> pl.Expr:
        """Generate a polars expression to discretise a column into bins.

        Constructs the expression using the column, breaks and new_name attributes of
        the instance.

        Returns:
            pl.Expr: A polars expression that can be used to discretise the column.

        Raises:
            BaseDiscretiserError : If the cut points have not been calculated.

        """
        if self.breaks is None:
            msg = (
                "Break points have not been calculated. "
                "Please call the `fit` method first."
            )
            raise BaseDiscretiserError(msg)

        if self.round_breaks_for_labels:
            required_precision = determine_required_precision_to_keep_values_distinct(
                self.breaks,
            )
            if required_precision is not None:
                rounded_breaks = [round(b, required_precision) for b in self.breaks]
                labels = generate_bin_labels(rounded_breaks)
            else:
                labels = None
        else:
            labels = None

        return generate_cut_expr(
            column=self.column,
            breaks=self.breaks,
            labels=labels,
            new_name=self.new_name,
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Discretise column in the DataFrame using the calculated cut points.

        Args:
            df (pl.DataFrame): The polars DataFrame containing the column to
                discretise.

        Returns:
            pl.DataFrame: A new polars DataFrame with the discretised column added.

        Raises:
            BaseDiscretiserError: If the cut points have not been calculated.
            MissingColumnError: If the specified column is not found in the DataFrame.
            NumericColumnError: If the specified column is not numeric.

        """
        if self.column not in df.columns:
            raise MissingColumnError(self.column)

        if not df.get_column(self.column).dtype.is_numeric():
            raise NumericColumnError(self.column)

        cut_expr = self.get_cut_expr()

        return df.with_columns(cut_expr)
