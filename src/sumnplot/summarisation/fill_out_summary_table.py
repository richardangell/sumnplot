"""Module for standardising groupby summary table outputs.

Specifically this means ensuring that all combinations of variable levels used in the
group by are represented in the table.

"""

from typing import Any

import polars as pl

from sumnplot.exceptions import MissingColumnError


def _cross(series_list: list[pl.Series]) -> pl.DataFrame:
    """Generate a DataFrame of all combinations of values in a list of Series."""
    lfs = [pl.LazyFrame(s) for s in series_list]
    out = lfs[0]
    for lf in lfs[1:]:
        out = out.join(lf, how="cross")
    return out.collect()


def ensure_all_level_combinations_populated(
    full_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    groupby_columns: list[str],
    value_columns: dict[str, Any],
) -> pl.DataFrame:
    """Ensure that all combinations of levels in the groupby columns are present.

    For pl.Enum types that set of unique levels is taken from the Enum categories, for
    all other types the unique levels are taken from what is available in the full_df.

    Args:
        full_df : The full DataFrame, provides the unique levels for each groupby
            column.
        summary_df : The summary DataFrame that may have missing combinations of
            levels.
        groupby_columns : The columns to group by.
        value_columns : A dictionary mapping value column names to their default fill
            values.

    Returns:
        A DataFrame with all combinations of levels populated, with missing values
        filled with values from the `value_columns` dictionary.

    Raises:
        ExceptionGroup : If any of the groupby columns are missing from either
            `full_df` or `summary_df`, an ExceptionGroup is raised containing all
            MissingColumnError instances for the missing columns.

    """
    column_errors: list[MissingColumnError] = []
    for col in groupby_columns:
        if col not in full_df.columns:
            column_errors.append(MissingColumnError(col, "full_df"))
        if col not in summary_df.columns:
            column_errors.append(MissingColumnError(col, "summary_df"))

    if column_errors:
        group_msg = "Missing columns"
        raise ExceptionGroup(group_msg, column_errors)

    groupby_cols_unique_levels = []
    for col in groupby_columns:
        col_dtype = full_df.get_column(col).dtype

        if isinstance(col_dtype, pl.Enum):
            unique_values = col_dtype.categories
        else:
            unique_values = full_df.select(col).unique().to_series()

        groupby_cols_unique_levels.append(unique_values)

    all_combinations = _cross(groupby_cols_unique_levels)

    df_filled = all_combinations.join(
        summary_df,
        on=groupby_columns,
        how="left",
        validate="1:1",
    )

    if df_filled.height == all_combinations.height == summary_df.height:
        return summary_df

    # Fill missing values in the value columns with their respective default values
    df_filled = df_filled.with_columns(
        pl.col(value_column).fill_null(fill_value)
        for value_column, fill_value in value_columns.items()
    )

    return df_filled.sort(by=groupby_columns)
