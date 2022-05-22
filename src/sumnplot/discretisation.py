import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from pandas.api.types import is_categorical_dtype

from .checks import check_type, check_condition, check_columns_in_df


class Discretiser(ABC, TransformerMixin, BaseEstimator):
    """Class implementing different discretisation methods."""

    def __init__(self, variable):

        check_type(variable, str, "variable")
        self.variable = variable

    @abstractmethod
    def fit(self, X, y=None):

        pass

    def transform(self, X):
        """Cut variable in X at cut_points."""

        check_is_fitted(self, "cut_points")
        check_columns_in_df(X, [self.variable])

        variable_cut = pd.cut(
            x=X[self.variable],
            bins=self.cut_points,
            include_lowest=True,
            duplicates="drop",
        )

        variable_cut = self._add_null_category(variable_cut)

        return variable_cut

    @staticmethod
    def _clean_cut_points(cut_points):
        """Clean provided cut points for discretisation by removing null values
        and returning unique values.
        """

        cleaned_cut_points = np.unique(cut_points[~np.isnan(cut_points)])

        if len(cleaned_cut_points) <= 1:
            raise ValueError(
                f"only 1 cut point after cleaning {cleaned_cut_points} - before cleaning {cut_points}"
            )

        return cleaned_cut_points

    @staticmethod
    def _add_null_category(categorical_variable, null_category_name="Null"):
        """Function to add new categorical level to categorical variable and set NAs to this category.

        Parameters
        ----------
        categorical_variable : pd.Series
            Categorical variable to add null categorical level to.

        null_category_name : str, default = 'Null'
            The name of the categorical level for null values to add.

        Returns
        -------
        cat : pd.Series
            Categorical variable (pandas category type) with null categorical level added.
        """

        check_type(categorical_variable, pd.Series, "categorical_variable")
        check_type(null_category_name, str, "null_category_name")

        check_condition(
            is_categorical_dtype(categorical_variable),
            f"categorical_variable ({categorical_variable.name}) is categorical dtype",
        )

        check_condition(
            null_category_name not in categorical_variable.cat.categories,
            f"null_category_name ({null_category_name}) not already in categorical_variable ({categorical_variable.name}) categories",
        )

        cat = categorical_variable.cat.add_categories([null_category_name])

        cat.fillna(null_category_name, inplace=True)

        return cat

    @abstractmethod
    def _get_max_number_of_bins(self):
        """Method to return the maximum number of bins possible for the give
        variable.

        Note, the actual number may be lower once calculated on a given dataset
        because the cut points may not be unique.
        """

        pass


class EqualWidthDiscretiser(Discretiser):
    def __init__(self, variable, n=10):

        super().__init__(variable=variable)

        check_type(n, int, "n")
        self.n = n

    def fit(self, X, y=None):

        check_columns_in_df(X, [self.variable])

        variable_min = X[self.variable].min()
        variable_max = X[self.variable].max()

        cut_points = np.linspace(start=variable_min, stop=variable_max, num=self.n + 1)
        self.cut_points = self._clean_cut_points(cut_points)

        return self

    def _get_max_number_of_bins(self):

        return self.n


class EqualWeightDiscretiser(Discretiser):
    def __init__(self, variable, n=10):

        super().__init__(variable=variable)

        check_type(n, int, "n")
        self.n = n

    def fit(self, X, y=None, sample_weight=None):

        check_columns_in_df(X, [self.variable])

        cut_points = QuantileDiscretiser._compute_weighted_quantile(
            values=X[self.variable],
            quantiles=np.array(np.linspace(start=0, stop=1, num=self.n + 1)),
            sample_weight=sample_weight,
        )
        self.cut_points = self._clean_cut_points(cut_points)

        return self

    def _get_max_number_of_bins(self):

        return self.n


class QuantileDiscretiser(Discretiser):
    def __init__(self, variable, quantiles=tuple(np.linspace(0, 1, 11))):

        super().__init__(variable=variable)

        check_type(quantiles, tuple, "quantiles")
        self.quantiles = self._clean_quantiles(quantiles)

    def fit(self, X, y=None, sample_weight=None):

        check_columns_in_df(X, [self.variable])

        cut_points = self._compute_weighted_quantile(
            values=X[self.variable],
            quantiles=self.quantiles,
            sample_weight=sample_weight,
        )
        self.cut_points = self._clean_cut_points(cut_points)

        return self

    @staticmethod
    def _compute_weighted_quantile(
        values, quantiles, sample_weight=None, values_sorted=False
    ):
        """Funtion to calculate weighted percentiles.

        Code modified from the answer given by users Alleo & Max Ghenis on
        stackoverflow https://stackoverflow.com/a/29677616. Removed old_style
        arg and associated code from answer.

        See https://en.wikipedia.org/wiki/Percentile#The_weighted_percentile_method
        for description of method.

        Parameters
        ----------
        values : array-like
            Data of interest, must contain a column supplied in variable.

        quantiles : array-like
            Value(s) between 0 <= quantiles <= 1, the weighted quantile(s) to compute.

        sample_weight : array-like, default = None
            Array of weights, must be same length as values. Default value of None
            means each observation in values is equally weighted.

        values_sorted : bool
            Are the values and sample_weight arrays pre-sorted? If True arrays will not
            be sorted in function.

        Returns
        -------
        cat : np.array
            Computed (weighted) quantiles.
        """

        values = np.array(values)
        quantiles = np.array(quantiles)
        quantiles = np.unique(np.sort(np.append(quantiles, [0, 1])))
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)

        if not values_sorted:
            sorter = np.argsort(values)
            values = values[sorter]
            sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight

        weighted_quantiles /= np.sum(sample_weight)

        return np.interp(quantiles, weighted_quantiles, values)

    @staticmethod
    def _clean_quantiles(quantiles: tuple) -> tuple:
        """Quantiles are converted back and forth between tuple type - which works with
        sklearn estimators as an input argument.
        """

        quantiles_array = np.array(quantiles)
        quantiles_array = np.unique(np.sort(np.append(quantiles_array, [0, 1])))

        check_condition(all(quantiles_array >= 0), "all quantiles >= 0")
        check_condition(all(quantiles_array <= 1), "all quantiles <= 1")

        return tuple(quantiles_array)

    def _get_max_number_of_bins(self):

        return len(self.quantiles)
