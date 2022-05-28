"""Module for summarisation classes."""

import pandas as pd
import numpy as np
from abc import ABCMeta
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

from typing import List, Dict, Optional, Union


class ColumnSummariser:
    """Summarisation of columns in a DataFrame. The summarisation function
    is sum by default. Averages can be calculated if the user specifies the
    column to use as the divisor.

    This class allows multiple (to_summarise_columns) columns to be specified
    which will summarised by others columns (by_columns) or combinations of
    columns (if second_by_column is also specified).

    The user has the option to specify the columns to summarise by; either
    using column names (by_columns) and a discretiser class from the
    discretisation module (discretiser + discretiser_kwargs) or a list of
    discretisers or column names - if different discretisation is to be
    applied to different by columns.

    The user can specify one of the columns to summarise (to_summarise_columns)
    to divide the others by (to_summarise_divide_column).

    Parameters
    ----------
    to_summarise_columns : List[str]
        List of column name to summarise. These columns will be grouped by each
        of the by_columns (and second_by_column, if specified) in turn and
        summed.

    by_columns : Optional[List[str]], default = None
        List of columns to summarise by. If by_columns is not specified then
        the discretisers argument must be used. If by_columns is specified then
        discretiser (along with discretiser_kwargs optionally) must be
        specified in order to set the discretiation method to apply to these
        by columns.

    discretiser : Optional[Discretiser], default = None
        Discretiser class to use to bucket the columns to summarise by, if
        by_columns is specified. The discretiser is initialised with
        discretiser_kwargs if specified and with the current by column
        name as the variable argument.

    discretiser_kwargs : Optional[dict], default = None
        A dictionary of keyword args passed into the initialisation of
        discretiser for each by_column.

    discretisers : Optional[List[Union[Discretiser, str]]], default = None
        A list of column names (for categorical variables that do not need
        discretising only) or Discretiser objects (for numerical variables)
        that provide an alternative way to specify by_columns with different
        discretisation methods applied to differnet columns. The discretisers
        argument and by_columns/discretiser/discretiser_kwargs argument
        combination are mutually exclusive.

    second_by_column : Optional[Union[Discretiser, str]], default = None
        Second column to summarise by. Only one second by column can be
        specified. If it is, then for every by column, to_summarise_columns
        are summed by the given by column AND second_by_column.

    to_summarise_columns_labels : Optional[List[str]], default = None
        Optional labels to replace the names of to_summarise_columns in the
        summarised output.

    to_summarise_divide_column : Optional[str] = None
        One of the to_summarise_columns, the other variables in the
        to_summarise_columns set will be divided by this column. This allows
        averages to be calculated in the summary.

    """

    def __init__(
        self,
        to_summarise_columns: List[str],
        by_columns: Optional[List[str]] = None,
        discretiser: Optional[Discretiser] = None,
        discretiser_kwargs: Optional[dict] = None,
        discretisers: Optional[List[Union[Discretiser, str]]] = None,
        second_by_column: Optional[Union[Discretiser, str]] = None,
        to_summarise_columns_labels: Optional[List[str]] = None,
        to_summarise_divide_column: Optional[str] = None,
    ) -> None:

        check_type(to_summarise_columns, list, "to_summarise_columns")
        check_type(by_columns, list, "by_columns", none_allowed=True)
        check_type(discretiser, ABCMeta, "discretiser", none_allowed=True)
        check_type(discretiser_kwargs, dict, "discretiser_kwargs", none_allowed=True)
        check_type(discretisers, list, "discretisers", none_allowed=True)
        check_type(
            second_by_column, (str, Discretiser), "second_by_column", none_allowed=True
        )
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

        if discretisers is None and discretiser is None:
            raise ValueError(
                "either discretisers or discretiser (and by_columns) must be specified"
            )

        if discretisers is None and by_columns is None:
            raise ValueError(
                "either discretisers or by_columns (and discretiser) must be specified"
            )

        if by_columns is not None and discretiser is None:
            raise ValueError("by_columns and discretiser must be specified together")

        if by_columns is None and discretiser is not None:
            raise ValueError("by_columns and discretiser must be specified together")

        self.to_summarise_columns = to_summarise_columns
        self.to_summarise_columns_labels = to_summarise_columns_labels
        self.to_summarise_divide_column = to_summarise_divide_column

        self.discretiser = discretiser
        self.discretiser_kwargs = discretiser_kwargs

        self.second_by_column: Optional[Union[Discretiser, str]] = second_by_column

        if type(discretisers) is list:

            by_columns = []

            for discretiser_no, discretiser_ in enumerate(discretisers):

                check_type(
                    discretiser_, (str, Discretiser), f"discretisers[{discretiser_no}]"
                )

                if type(discretiser_) is str:
                    by_columns.append(discretiser_)
                elif isinstance(discretiser_, Discretiser):
                    by_columns.append(discretiser_.variable)

            self.discretisers = discretisers
            self.by_columns = by_columns

        elif discretiser is not None and by_columns is not None:

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
            self.by_columns = by_columns

    def summarise(
        self,
        X: pd.DataFrame,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Summarise columns in X.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing information to be summarised. Must contain
            variables specified in the to_summarise_columns, by_columns
            (if a column name), to_summarise_divide_column and second_by_column
            (if a column name).

        sample_weight : Optional[Union[pd.Series, np.ndarray]], default = None
            Optional weights for each row in X.

        Returns
        -------
        results : dict[str:pd.DataFrame]
            Summarised variables in a dict where each key is the by column
            name.

        """

        check_columns_in_df(X, self.to_summarise_columns)
        check_columns_in_df(X, self.by_columns)

        if self.to_summarise_divide_column is not None:
            check_columns_in_df(X, [self.to_summarise_divide_column])

        if self.second_by_column is not None:
            if type(self.second_by_column) is str:
                check_columns_in_df(X, [self.second_by_column])
            elif isinstance(self.second_by_column, Discretiser):
                check_columns_in_df(X, [self.second_by_column.variable])

        if sample_weight is not None:
            if len(X) != len(sample_weight):
                raise ValueError("X and sample_weight have different numbers of rows")
            if len(sample_weight.shape) == 2:
                if sample_weight.shape[1] > 1:
                    raise ValueError("sample_weight has more than one column")
            elif len(sample_weight.shape) > 2:
                raise ValueError("sample_weight has more than two dimensions")

        results = {}

        for by_column in self.discretisers:

            if type(by_column) is str:
                by_column_name = by_column
            elif isinstance(by_column, Discretiser):
                by_column_name = by_column.variable

            results[by_column_name] = self._summarise_column(
                df=X,
                to_summarise_columns=self.to_summarise_columns,
                by_column=by_column,
                to_summarise_columns_labels=self.to_summarise_columns_labels,
                to_summarise_divide_column=self.to_summarise_divide_column,
                sample_weight=sample_weight,
                second_by_column=self.second_by_column,  # type: ignore
            )

        return results

    @staticmethod
    def _summarise_column(
        df: pd.DataFrame,
        to_summarise_columns: List[str],
        by_column: Union[str, Discretiser],
        to_summarise_columns_labels: List[str] = None,
        to_summarise_divide_column: str = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        second_by_column: Optional[Union[Discretiser, str]] = None,
    ) -> pd.DataFrame:
        """Function to summarise to_summarise_columns in df by by_column and
        second_by_column, if specified.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with information to summarise.

        to_summarise_columns : List[str]
            List of column name to summarise. These columns will be grouped by each
            of the by_columns (and second_by_column, if specified) in turn and
            summed.

        by_column : str or Discretiser
            Either the column name to summarise by in the case of a categorical
            column or the Discretiser object to bucketed a numeric column.

        to_summarise_columns_labels : Optional[List[str]], default = None
            Optional labels to replace the names of to_summarise_columns in the
            summarised output.

        to_summarise_divide_column : Optional[str] = None
            One of the to_summarise_columns, the other variables in the
            to_summarise_columns set will be divided by this column. This allows
            averages to be calculated in the summary.

        sample_weight : Optional[Union[pd.Series, np.ndarray]], default = None
            Optional weights for each row in X.

        second_by_column : Optional[Union[Discretiser, str]], default = None
            Second column to summarise by. Only one second by column can be
            specified. If it is, then for every by column, to_summarise_columns
            are summed by the given by column AND second_by_column.

        Returns
        -------
        summary_values : pd.DataFrame
            The to_summarise_columns summarised by by_column (and optionally
            second_by_column).

        """

        check_type(by_column, (str, Discretiser), "by_column")

        groupby_column = ColumnSummariser._prepare_groupby_column(
            df, by_column, sample_weight
        )

        groupby_columns = [groupby_column]

        if second_by_column is not None:

            second_groupby_column = ColumnSummariser._prepare_groupby_column(
                df, second_by_column, sample_weight
            )

            groupby_columns.append(second_groupby_column)

        summary_functions = {column: ["sum"] for column in to_summarise_columns}

        summary_values = df.groupby(groupby_columns).agg(summary_functions)

        # divide through other to_summarise_column by to_summarise_divide_column
        if to_summarise_divide_column is not None:

            non_divide_by_columns = [
                column
                for column in to_summarise_columns
                if column != to_summarise_divide_column
            ]

            for column_no, column in enumerate(summary_values.columns):

                if column[0] in non_divide_by_columns:

                    summary_values[column] = (
                        summary_values[column]
                        / summary_values[(to_summarise_divide_column, "sum")]
                    )

                    summary_values.columns.values[column_no] = (
                        summary_values.columns.values[column_no][0],
                        "mean",
                    )

            # create new index on the DataFrame, otherwise changing the values directly
            # doesn't seem to flow through
            summary_values.columns = pd.MultiIndex.from_tuples(
                summary_values.columns.values.tolist()
            )

        if to_summarise_columns_labels is not None:

            renaming_dict = {
                old: new
                for old, new in zip(to_summarise_columns, to_summarise_columns_labels)
            }

            summary_values.rename(columns=renaming_dict, level=0, inplace=True)

        return summary_values

    @staticmethod
    def _prepare_groupby_column(
        df: pd.DataFrame,
        by_column: Union[str, Discretiser],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.Series:
        """Return column to group by given the input column type.

        If the input column is categorical then the original columns is
        returned. Otherwise if by_column is numeric then it is bucketed with
        discretiser.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing by_column to potentially discretise.

        by_column : Union[str, Discretiser]
            Either a categorical column name in df or a Discretiser object to
            bucket a numeric column.

        sample_weight : Optional[Union[pd.Series, np.ndarray]], default = None
            Optional weights for each record in df.

        Returns
        -------
        groupby_column : pd.Series
            A Series containing categorical data. May be a numeric columns that
            has been discretised or an input boolean, categorical or object
            dtype.

        """

        if type(by_column) is str:

            by_column_name = by_column
            discretiser = None

        elif isinstance(by_column, Discretiser):

            by_column_name = by_column.variable
            discretiser = by_column

        if (
            is_object_dtype(df[by_column_name])
            | is_bool_dtype(df[by_column_name])
            | is_categorical_dtype(df[by_column_name])
        ):

            groupby_column = df[by_column_name]

        elif is_numeric_dtype(df[by_column_name]):

            if discretiser is None:

                raise TypeError(
                    f"discretiser is None for {by_column_name} but column is numeric"
                )

            max_bins = discretiser._get_max_number_of_bins()

            if df[by_column_name].nunique(dropna=False) <= max_bins:

                if df[by_column_name].isnull().sum() > 0:

                    groupby_column = df[by_column_name].astype(str)

                else:

                    groupby_column = df[by_column_name]

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

            raise TypeError(
                f"unexpected type for by_column; {df[by_column_name].dtype}"
            )

        return groupby_column


class DataFrameValueCounter:
    """Summarisation of values in a DataFrame.

    A value_counts operation is peformed for each column of interest with a
    maximum number of values kept per column. If the number of unique values
    in a column exceeds this number then .

    Paremeters
    ----------
    columns : Optional[List], default = None
        Columns to summarise with value_counts. If not specified then all
        columns are used.

    max_values : int, default = 50
        Maximum number of value counts to keep per column.

    summary_values : int, default = 5
        If the number of unique values in a column exceeds max_values then only
        then top, middle and bottom summary_values are kept in the value_counts
        output.

    """

    def __init__(
        self,
        columns: Optional[List] = None,
        max_values: int = 50,
        summary_values: int = 5,
    ) -> None:

        check_type(columns, list, "columns", none_allowed=True)
        check_type(max_values, int, "max_values")
        check_type(summary_values, int, "summary_values")

        check_condition(max_values > 0, "max_values > 0")
        check_condition(summary_values > 0, "summary_values > 0")

        self.columns = columns
        self.max_values = max_values
        self.summary_values = summary_values

    def summarise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarise input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns specified in the columns attribute.

        Returns
        -------
        columns_summary_all : pd.DataFrame
            Value counts results for all columns concatenated along axis 1.

        """

        check_type(df, pd.DataFrame, "df")

        if self.columns is None:
            self.columns = list(df.columns.values)

        check_columns_in_df(df, self.columns)

        columns_summary = [
            self._summarise_column_value_counts(
                df, col, self.max_values, self.summary_values
            )
            for col in self.columns
        ]

        columns_summary_all = pd.concat(columns_summary, axis=1)

        return columns_summary_all

    def _summarise_column_value_counts(
        self, df: pd.DataFrame, column: str, max_values: int, summary_values: int
    ) -> pd.DataFrame:
        """Function to return value_counts for a sinlge column in df resized to
        max_values rows.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing column.

        column : str
            Column to summarise.

        max_values : int, default = 50
            Maximum number of value counts to keep per column.

        summary_values : int, default = 5
            If the number of unique values in a column exceeds max_values then only
            then top, middle and bottom summary_values are kept in the value_counts
            output.

        Returns
        -------
        value_counts_resize : pd.DataFrame
            Output from pd.Series.value_counts resized to have max_values,
            either by padding with null rows are taking top, middle and
            bottom summary_values of the value counts.

        """

        value_counts = self._get_column_values(df[column])

        value_counts_resize = self._resize_column_value_counts(
            value_counts, max_values, summary_values
        )

        return value_counts_resize

    def _get_column_values(
        self, column: pd.Series, ascending: Optional[bool] = True
    ) -> pd.DataFrame:
        """Run a value_counts on pandas Series and return the results sorted by index
        with the index as a column in the output.

        Parameters
        ----------
        column : pd.Series
            Column to summarise with value_counts.

        ascending : Optional[bool], default = True
            Order to sort value counts.

        Returns
        -------
        value_counts : pd.DataFrame
            Output from pd.Series.value_counts with columns renamed.

        """

        value_counts = (
            column.value_counts(dropna=False)
            .sort_index(ascending=ascending)
            .reset_index()
        )

        value_counts.columns = [column.name + "_value", column.name + "_count"]

        return value_counts

    def _resize_column_value_counts(
        self, df: pd.DataFrame, max_values: int, summary_values: int
    ) -> pd.DataFrame:
        """Function to resize the output the results of value_counts() to be
        max_values rows.

        If n (number rows of df) < max_values then df is padded with rows
        containing None. Otherwise if n > max_values then the first, middle and
        last summary_values rows are selected and similarly padded with None
        value rows.

        Parameters
        ----------
        df : pd.DataFrame
            Output from pd.Series.value_counts for a single column.

        max_values : int, default = 50
            Maximum number of value counts to keep per column.

        summary_values : int, default = 5
            If the number of unique values in a column exceeds max_values then only
            then top, middle and bottom summary_values are kept in the value_counts
            output.

        Returns
        -------
        df_resized : pd.DataFrame
            Resize value_counts output.

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

            df_resized = pd.concat(dfs_to_concat, axis=0).reset_index(drop=True)

            return df_resized
