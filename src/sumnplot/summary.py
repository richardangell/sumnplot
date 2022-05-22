import pandas as pd
import abc
from copy import deepcopy
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_bool_dtype,
    is_categorical_dtype,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from .discretisation import Discretiser
from .checks import check_type, check_condition, check_columns_in_df

from typing import List


class ColumnSummariser:
    """"""

    def __init__(
        self,
        to_summarise_columns: List[str],
        by_columns: List[str],
        discretiser: Discretiser = None,
        discretiser_kwargs: dict = None,
        discretisers: List[Discretiser] = None,
        to_summarise_columns_labels: List[str] = None,
        to_summarise_divide_column: str = None,
    ):

        check_type(to_summarise_columns, list, "to_summarise_columns")
        check_type(by_columns, list, "by_columns")
        check_type(discretiser, abc.ABCMeta, "discretiser", none_allowed=True)
        check_type(discretiser_kwargs, dict, "discretiser_kwargs", none_allowed=True)
        check_type(discretisers, list, "discretisers", none_allowed=True)
        check_type(
            to_summarise_columns_labels,
            list,
            "to_summarise_columns_labels",
            none_allowed=True,
        )
        check_type(
            to_summarise_divide_column,
            str,
            "to_summarise_divide_column",
            none_allowed=True,
        )

        if to_summarise_columns_labels is not None:
            if len(to_summarise_columns_labels) != len(to_summarise_columns):
                raise ValueError(
                    "to_summarise_columns and to_summarise_columns_labels are different lengths"
                )

        if to_summarise_divide_column is not None:
            if to_summarise_divide_column not in to_summarise_columns:
                raise ValueError(
                    "to_summarise_divide_column not in to_summarise_columns"
                )

        if discretiser is None and discretisers is None:
            raise ValueError("either discretiser or discretisers")

        self.to_summarise_columns = to_summarise_columns
        self.by_columns = by_columns
        self.to_summarise_columns_labels = to_summarise_columns_labels
        self.to_summarise_divide_column = to_summarise_divide_column

        if type(discretisers) is list:

            if len(discretisers) != len(by_columns):
                raise ValueError("by_columns and discretisers have different lengths")

            for discretiser_no, (discretiser, by_column) in enumerate(
                zip(discretisers, by_columns)
            ):

                check_type(discretiser, Discretiser, f"discretisers[{discretiser_no}]")

                if not discretiser.variable == by_column:
                    raise ValueError(
                        f"mismatch in variable ordering at index {discretiser_no} between discretisers and by_columns"
                    )

            self.discretisers = discretisers

        elif discretiser is not None:

            self.discretiser = discretiser
            self.discretiser_kwargs = discretiser_kwargs

            initialised_discretisers = []

            for by_column in by_columns:

                if discretiser_kwargs is not None:
                    initialisation_kwargs = deepcopy(discretiser_kwargs)
                else:
                    initialisation_kwargs = {}
                initialisation_kwargs["variable"] = by_column

                initialised_discretiser = discretiser(**initialisation_kwargs)

                check_type(
                    initialised_discretiser,
                    Discretiser,
                    "initialised_discretiser for by_column",
                )

                initialised_discretisers.append(initialised_discretiser)

            self.discretisers = initialised_discretisers

    def summarise_columns(self, X, sample_weight=None):

        check_columns_in_df(X, self.to_summarise_columns)
        check_columns_in_df(X, self.by_columns)

        if self.to_summarise_divide_column is not None:
            check_columns_in_df(X, [self.to_summarise_divide_column])

        if sample_weight is not None:
            if len(X) != len(sample_weight):
                raise ValueError("X and sample_weight have different numbers of rows")
            if len(sample_weight.shape) == 2:
                if sample_weight.shape[1] > 1:
                    raise ValueError("sample_weight has more than one column")
            elif len(sample_weight.shape) > 2:
                raise ValueError("sample_weight has more than two dimensions")

        results = {}

        for by_column, discretiser in zip(self.by_columns, self.discretisers):

            results[by_column] = self._summarise_column(
                df=X,
                to_summarise_columns=self.to_summarise_columns,
                by_column=by_column,
                discretiser=discretiser,
                to_summarise_columns_labels=self.to_summarise_columns_labels,
                sample_weight=sample_weight,
            )

        return results

    @staticmethod
    def _summarise_column(
        df: pd.DataFrame,
        to_summarise_columns: List[str],
        by_column: str,
        discretiser: Discretiser,
        to_summarise_columns_labels: List[str] = None,
        to_summarise_divide_column: str = None,
        sample_weight=None,
    ):

        if (
            is_object_dtype(df[by_column])
            | is_bool_dtype(df[by_column])
            | is_categorical_dtype(df[by_column])
        ):

            groupby_column = df[by_column]

        elif is_numeric_dtype(df[by_column]):

            max_bins = discretiser._get_max_number_of_bins()

            if df[by_column].nunique(dropna=False) <= max_bins:

                if df[by_column].isnull().sum() > 0:

                    groupby_column = df[by_column].astype(str)

                else:

                    groupby_column = df[by_column]

            else:

                # if the discretiser is already fitted just run transform
                try:

                    check_is_fitted(discretiser, "cut_points")

                    groupby_column = discretiser.transform(X=df)

                # otherwise, if it is not fitted run both fit and transform
                except NotFittedError:

                    groupby_column = discretiser.fit_transform(
                        X=df, sample_weight=sample_weight
                    )

        else:

            raise TypeError(f"unexpected type for by_column; {df[by_column].dtype}")

        summary_functions = {column: ["sum"] for column in to_summarise_columns}

        summary_values = df.groupby(groupby_column).agg(summary_functions)

        # divide through other to_summarise_column by to_summarise_divide_column
        if to_summarise_divide_column is not None:

            non_divide_by_columns = [
                column
                for column in to_summarise_columns
                if column != to_summarise_divide_column
            ]

            for non_divide_by_column in non_divide_by_columns:

                summary_values[non_divide_by_column] = (
                    summary_values[non_divide_by_column]
                    / summary_values[to_summarise_divide_column]
                )

        # if no labels are available just use the column names
        if to_summarise_columns_labels is not None:

            renaming_dict = {
                old: new
                for old, new in zip(to_summarise_columns, to_summarise_columns_labels)
            }

            summary_values.rename(columns=renaming_dict, level=0, inplace=True)

        return summary_values


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
