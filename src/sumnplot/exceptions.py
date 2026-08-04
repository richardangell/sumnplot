"""Exception to be raised in the sumnplot package."""


class SumNPlotError(Exception):
    """Base class for exceptions in the sumnplot package."""


class MissingColumnError(SumNPlotError):
    """Raised when a required column is missing from the DataFrame."""

    def __init__(self, column: str) -> None:
        """Initialise the MissingColumnError with the missing column name."""
        self.column = column
        self.message = f"Column '{column}' not found in DataFrame."
        super().__init__(self.message)
