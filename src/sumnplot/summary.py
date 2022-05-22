import pandas as pd

from .checks import check_type, check_condition, check_columns_in_df

from typing import List


def dataset_values_summary(
    df: pd.DataFrame,
    columns: List = None,
    max_values: int = 50,
    summary_values: int = 5,
) -> pd.DataFrame:
    """Function to produce summaries of values in a DataFrame."""

    check_type(df, pd.DataFrame, "df")
    check_type(max_values, int, "max_values")
    check_type(summary_values, int, "summary_values")

    check_condition(max_values > 0, "max_values > 0")
    check_condition(summary_values > 0, "summary_values > 0")
    if columns is None:
        columns = df.columns.values

    check_columns_in_df(df, columns)

    columns_summary = [
        summarise_column_value_counts(df, col, max_values, summary_values)
        for col in columns
    ]

    columns_summary_all = pd.concat(columns_summary, axis=1)

    return columns_summary_all


def summarise_column_value_counts(
    df: pd.DataFrame, column: str, max_values: int, summary_values: int
) -> pd.DataFrame:
    """Function to return value_counts for a sinlge column in df resized to
    max_values rows.
    """

    value_counts = get_column_values(df[column])

    value_counts_resize = resize_column_value_counts(
        value_counts, max_values, summary_values
    )

    return value_counts_resize


def get_column_values(column: pd.Series, ascending: bool = True) -> pd.DataFrame:
    """Run a value_counts on pandas Series and return the results sorted by index
    with the index as a column in the output.
    """

    value_counts = (
        column.value_counts(dropna=False).sort_index(ascending=ascending).reset_index()
    )

    value_counts.columns = [column + "_value", column + "_count"]

    return value_counts


def resize_column_value_counts(
    df: pd.DataFrame, max_values: int, summary_values: int
) -> pd.DataFrame:
    """Function to resize the output the results of value_counts() to be
    max_values rows.

    If n (number rows of df) < max_values then df is padded with rows
    containing None. Otherwise if n > max_values then the first, middle and
    last summary_values rows are selected and similarly padded with None
    value rows.
    """

    n = df.shape[0]

    if n == max_values:

        return df.reset_index(drop=True)

    else:

        pad_row = pd.DataFrame({df.columns[0]: [None], df.columns[1]: [None]})

        if n < max_values:

            extra_rows = max_values - n

            dfs_to_concat = [pad_row] * extra_rows

            dfs_to_concat.insert(0, df)

        else:

            dfs_to_concat = []

            bottom_rows = df.loc[0 : (summary_values - 1)].copy()

            mid_row = n // 2
            below_mid_row = mid_row - (summary_values // 2)

            middle_rows = df.loc[
                below_mid_row : (below_mid_row + summary_values)
            ].copy()
            top_rows = df.loc[(n - summary_values) :].copy()

            extra_pad_rows = max_values - (3 * summary_values + 2)

            if extra_pad_rows > 0:

                dfs_to_concat = [pad_row] * extra_pad_rows

            else:

                dfs_to_concat = []

            dfs_to_concat.insert(0, top_rows)
            dfs_to_concat.insert(0, pad_row)
            dfs_to_concat.insert(0, middle_rows)
            dfs_to_concat.insert(0, pad_row)
            dfs_to_concat.insert(0, bottom_rows)

        return pd.concat(dfs_to_concat, axis=0).reset_index(drop=True)
