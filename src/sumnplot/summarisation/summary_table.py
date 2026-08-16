"""Summary table class for holding results from a group by summary operation."""

from typing import TYPE_CHECKING, Any, Literal

from sumnplot.exceptions import MissingColumnError, SumNPlotError

if TYPE_CHECKING:
    import polars as pl

    from sumnplot.summarisation.summary_operation import SummaryOperation


class SummaryTableError(SumNPlotError):
    """Base class for errors in the SummaryTable class."""


class SummaryTableModificationError(SummaryTableError):
    """Exception to be raised when trying to modify an SummaryTable attribute."""

    def __init__(self, field: str, operation: Literal["modify", "delete"]) -> None:
        """Initialise a SummaryTableModificationError instance."""
        self.field = field
        self.operation = operation
        self.message = f"Cannot {operation} '{field}' on SummaryTable instance."
        super().__init__(self.message)


class SummaryTable:
    """Class to hold results from a group by summary operation."""

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        groupby_columns: list[str],
        summarised_column_types: dict[str, SummaryOperation],
    ) -> None:
        """Initialise a SummaryTable instance.

        Args:
            data (pl.DataFrame): The summary table as a DataFrame.
            groupby_columns (list[str]): The columns used for grouping in the summary
                table.
            summarised_column_types (dict[str, SummaryOperation]): A dictionary
                specifying the columns in the table that have been summarised and the
                type of summarisation applied to each.

        Raises:
            SummaryTableError: If groupby_columns or summarised_column_types are empty.
            SummaryTableError: If groupby_columns are not unique.
            SummaryTableError: If groupby_columns and summarised_column_types overlap.
            ExceptionGroup: If any of the groupby_columns or summarised_column_types
                are missing from the DataFrame, an ExceptionGroup is raised containing
                all MissingColumnError instances for the missing columns.
            SummaryTableModificationError: If there is an attempt to modify or delete
                the groupby_columns or summarised_column_types attributes after the
                SummaryTable instance has been initialised.

        """
        if not groupby_columns:
            msg = "Groupby columns must not be empty."
            raise SummaryTableError(msg)

        if not summarised_column_types:
            msg = "Summarised column types must not be empty."
            raise SummaryTableError(msg)

        if len(set(groupby_columns)) != len(groupby_columns):
            msg = "Groupby columns must be unique."
            raise SummaryTableError(msg)

        summarised_columns = list(summarised_column_types.keys())

        overlapping_columns = set(summarised_columns).intersection(set(groupby_columns))
        if overlapping_columns:
            msg = (
                f"Groupby columns and summarised columns must not overlap. "
                f"Overlapping columns: {', '.join(overlapping_columns)}."
            )
            raise SummaryTableError(msg)

        missing_columns_exceptions = []
        for col in groupby_columns:
            if col not in data.columns:
                missing_columns_exceptions.append(MissingColumnError(col))
        for col in summarised_column_types:
            if col not in data.columns:
                missing_columns_exceptions.append(MissingColumnError(col))
        if missing_columns_exceptions:
            msg = "Missing columns in SummaryTable input data."
            raise ExceptionGroup(
                msg,
                missing_columns_exceptions,
            )

        if data.width == len(groupby_columns) + len(summarised_column_types):
            self._data = data
        else:
            self._data = data.select(
                [*groupby_columns, *summarised_column_types.keys()],
            )
        self._groupby_columns = groupby_columns
        self._summarised_column_types = summarised_column_types

    def unpivot(
        self,
        on: list[str],
        index: list[str],
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> pl.DataFrame:
        """Unpivot the summary table data to a long format.

        Returns:
            pl.DataFrame: The unpivoted summary table in long format.

        """
        return self._data.unpivot(
            on=on,
            index=index,
            variable_name=variable_name,
            value_name=value_name,
        )

    def head(self, n: int = 5) -> pl.DataFrame:
        """Return the first n rows of the summary table."""
        return self._data.head(n)

    def tail(self, n: int = 5) -> pl.DataFrame:
        """Return the last n rows of the summary table."""
        return self._data.tail(n)

    @property
    def n(self) -> int:
        """Return the number of rows in the summary table."""
        return self._data.height

    @property
    def groupby_columns(self) -> list[str]:
        """The columns the results are grouped by."""
        return self._groupby_columns

    @groupby_columns.setter
    def groupby_columns(self, value: Any) -> None:  # noqa: ANN401, ARG002
        """Set the groupby_columns attribute."""
        raise SummaryTableModificationError(field="groupby_columns", operation="modify")

    @groupby_columns.deleter
    def groupby_columns(self) -> None:
        """Delete the groupby_columns attribute."""
        raise SummaryTableModificationError(field="groupby_columns", operation="delete")

    @property
    def summarised_column_types(self) -> dict[str, SummaryOperation]:
        """The columns that have been summarised and their types."""
        return self._summarised_column_types

    @summarised_column_types.setter
    def summarised_column_types(self, value: Any) -> None:  # noqa: ANN401, ARG002
        """Set the summarised_column_types attribute."""
        raise SummaryTableModificationError(
            field="summarised_column_types",
            operation="modify",
        )

    @summarised_column_types.deleter
    def summarised_column_types(self) -> None:
        """Delete the summarised_column_types attribute."""
        raise SummaryTableModificationError(
            field="summarised_column_types",
            operation="delete",
        )

    def __eq__(self, other: Any) -> bool:  # noqa: ANN401
        """Check if two SummaryTable instances are equal."""
        if not isinstance(other, SummaryTable):
            return False

        if self.groupby_columns != other.groupby_columns:
            return False

        if self.summarised_column_types != other.summarised_column_types:
            return False

        return self._data.equals(other.head(other.n))
