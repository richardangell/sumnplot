"""Exception to be raised in the sumnplot package."""

from typing import Any


class SumNPlotError(Exception):
    """Base class for exceptions in the sumnplot package."""


class MissingColumnError(SumNPlotError):
    """Raised when a required column is missing from the DataFrame."""

    def __init__(self, column: str) -> None:
        """Initialise the MissingColumnError with the missing column name."""
        self.column = column
        self.message = f"Column '{column}' not found in DataFrame."
        super().__init__(self.message)


class NumericColumnError(SumNPlotError):
    """Raised when a column is not numeric."""

    def __init__(self, column: str) -> None:
        """Initialise the NumericColumnError with the non-numeric column name."""
        self.column = column
        self.message = f"Column '{column}' must be a numeric dtype."
        super().__init__(self.message)


class NullsInColumnError(SumNPlotError):
    """Raised when a column contains null values."""

    def __init__(self, column: str) -> None:
        """Initialise the NullsInColumnError with the column name containing nulls."""
        self.column = column
        self.message = f"Column '{column}' contains null values."
        super().__init__(self.message)


class InvalidArgumentError(SumNPlotError):
    """Raised when an argument value is invalid."""

    def __init__(self, argument: str, value: Any, conditions: list[str]) -> None:  # noqa: ANN401
        """Initialise the InvalidArgumentError with the argument name and message."""
        self.argument = argument
        self.value = value
        self.conditions = conditions
        self.message = (
            f"Invalid value for argument '{argument}': {value}. "
            f"The following conditions were not met: {', '.join(conditions)}."
        )
        super().__init__(self.message)
