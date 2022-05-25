"""Module for discretisation classes."""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from pandas.api.types import is_categorical_dtype

from typing import Optional, Union, Tuple

from .checks import check_type, check_condition, check_columns_in_df


class Discretiser(ABC, TransformerMixin, BaseEstimator):
    """Abstract base class for different discretisation methods.

    This abstract base class is a transformer compatible with
    scikit-learn.

    Parameters
    ----------
    variable : str
        Column to discretise in X, when the transform method is called.

    """

    def __init__(self, variable: str) -> None:

        check_type(variable, str, "variable")
        self.variable = variable

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """Calculate cut points for given discretisation approach.

        The cut_points attribute should be set by this method.
        """

        pass

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Cut variable in X at cut_points. This function uses the pd.cut
        method.

        A specific null category is added on the cut output.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing column to discretise. This column is defined
            by the variable attribute.

        Returns
        -------
        variable_cut : pd.Series
            Discretised variable.

        """

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
    def _clean_cut_points(cut_points: np.ndarray) -> np.ndarray:
        """Clean provided cut points for discretisation by removing null values
        and returning unique values.

        Parameters
        ----------
        cut_points : np.ndarray
            Array of cut points that define where a particular column should be
            split to discretise it.

        Returns
        -------
        cleaned_cut_points : np.ndarray
            Array of the unique cut points input to the function, with any null
            values also removed.

        """

        cleaned_cut_points = np.unique(cut_points[~np.isnan(cut_points)])

        if len(cleaned_cut_points) <= 1:
            raise ValueError(
                f"only 1 cut point after cleaning {cleaned_cut_points} - before cleaning {cut_points}"
            )

        return cleaned_cut_points

    @staticmethod
    def _add_null_category(
        categorical_variable: pd.Series, null_category_name: str = "Null"
    ) -> pd.Series:
        """Function to add new categorical level to categorical variable and
        set NAs to this category.

        Parameters
        ----------
        categorical_variable : pd.Series
            Categorical variable to add null categorical level to.

        null_category_name : str, default = 'Null'
            The name of the categorical level for null values to add.

        Returns
        -------
        cat : pd.Series
            Categorical variable (pandas category type) with null categorical
            level added.

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
        """Method to return the maximum number of bins possible for the given
        variable.

        Note, the actual number may be lower once calculated on a given dataset
        because the cut points may not be unique.
        """

        pass

    def _get_actual_number_of_bins(self) -> int:
        """Method to return the actual number of bins based off cut_points
        after the fit method has been run.

        Returns
        -------
        int
            Actual number of bins variable has been cut into.

        """

        check_is_fitted(self, "cut_points")

        return len(self.cut_points) - 1


class EqualWidthDiscretiser(Discretiser):
    """Equal width discretisation.

    This tansformer simply uses n+1 equally spaced cut points across the range
    of the variable.

    Parameters
    ----------
    variable : str
        Column to discretise in X, when the transform method is called.

    n : int, default = 10
        Number of bins to bucket variable into.

    """

    def __init__(self, variable: str, n: int = 10) -> None:

        super().__init__(variable=variable)

        check_type(n, int, "n")
        self.n = n

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        """Calculate cut points on the input data X.

        Cut points are equally spaced across the range of the variable. The
        attribute cut_points contains the calculate cut points.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing column to discretise. This column is defined
            by the variable attribute.

        y : pd.Series, default = None
            Response variable. Not used. Only implemented for compatibility
            with scikit-learn.

        sample_weight : pd.Series or np.ndarray, default = None
            Optional, sample weights for each record in X.

        """

        check_columns_in_df(X, [self.variable])

        variable_min = X[self.variable].min()
        variable_max = X[self.variable].max()

        cut_points = np.linspace(start=variable_min, stop=variable_max, num=self.n + 1)
        self.cut_points = self._clean_cut_points(cut_points)

        return self

    def _get_max_number_of_bins(self) -> int:
        """Return the maximum number of bins possible for the given
        variable.
        """

        return self.n


class EqualWeightDiscretiser(Discretiser):
    """Equal weight discretisation.

    This tansformer simply uses n+1 cut points across the range of the variable
    chosen such that each bucket contains an equal amount of weight.

    Parameters
    ----------
    variable : str
        Column to discretise in X, when the transform method is called.

    n : int, default = 10
        Number of bins to bucket variable into.

    """

    def __init__(self, variable: str, n: int = 10):

        super().__init__(variable=variable)

        check_type(n, int, "n")
        self.n = n

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        """Calculate cut points on the input data X.

        Cut points are chosen so each of the n buckets contains an equal amount
        of weight. The attribute cut_points contains the calculate cut points.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing column to discretise. This column is defined
            by the variable attribute.

        y : pd.Series, default = None
            Response variable. Not used. Only implemented for compatibility
            with scikit-learn.

        sample_weight : pd.Series or np.ndarray, default = None
            Optional, sample weights for each record in X.

        """

        check_columns_in_df(X, [self.variable])

        cut_points = QuantileDiscretiser._compute_weighted_quantile(
            values=X[self.variable],
            quantiles=np.array(np.linspace(start=0, stop=1, num=self.n + 1)),
            sample_weight=sample_weight,
        )
        self.cut_points = self._clean_cut_points(cut_points)

        return self

    def _get_max_number_of_bins(self) -> int:
        """Return the maximum number of bins possible for variable."""

        return self.n


class QuantileDiscretiser(Discretiser):
    """Quantile discretisation.

    This tansformer uses cut points defined by quantiles of the given variable.

    Note, this transformer handles weighted quantiles.

    Parameters
    ----------
    variable : str
        Column to discretise in X, when the transform method is called.

    quantiles : tuple, default = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        Quantiles defining the cut points to bucket variable at.

    """

    def __init__(
        self,
        variable,
        quantiles: Tuple[Union[int, float], ...] = tuple(np.linspace(0, 1, 11)),
    ) -> None:

        super().__init__(variable=variable)

        check_type(quantiles, tuple, "quantiles")
        self.quantiles = self._clean_quantiles(quantiles)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        """Calculate cut points on the input data X.

        Cut points are (potentially weighted) quantiles specified when
        initialising the transformer.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing column to discretise. This column is defined
            by the variable attribute.

        y : pd.Series, default = None
            Response variable. Not used. Only implemented for compatibility
            with scikit-learn.

        sample_weight : pd.Series or np.ndarray, default = None
            Optional, sample weights for each record in X.

        """

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
        values: np.ndarray,
        quantiles: tuple,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        values_sorted: bool = False,
    ):
        """Funtion to calculate weighted percentiles.

        Code modified from the answer given by users Alleo & Max Ghenis on
        stackoverflow https://stackoverflow.com/a/29677616. Removed old_style
        arg and associated code from answer.

        See https://en.wikipedia.org/wiki/Percentile#The_weighted_percentile_method
        for description of method.

        If no weights are passed then equal weighting per observation in values
        is applied.

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
        interpolated_quantiles : np.array
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

        interpolated_quantiles = np.interp(quantiles, weighted_quantiles, values)

        return interpolated_quantiles

    @staticmethod
    def _clean_quantiles(
        quantiles: Tuple[Union[int, float], ...]
    ) -> Tuple[Union[int, float], ...]:
        """Clean input quantiles by ensuring 0 and 1 are included, they are
        sorted and unique.

        Note, quantiles are converted back and forth between a tuple a
        np.ndarray. This is so the transformer is compatible with scikit-learn
        as the quantiles are set during init.

        Parameters
        ----------
        quantiles : tuple
            Quantiles within the range [0, 1].

        Returns
        -------
        cleaned_quantiles : tuple
            Sorted, unique quantiles.

        """

        quantiles_array = np.array(quantiles)
        quantiles_array = np.unique(np.sort(np.append(quantiles_array, [0, 1])))

        check_condition(all(quantiles_array >= 0), "all quantiles >= 0")
        check_condition(all(quantiles_array <= 1), "all quantiles <= 1")

        cleaned_quantiles = tuple(quantiles_array)

        return cleaned_quantiles

    def _get_max_number_of_bins(self) -> int:
        """Return the maximum number of bins possible for variable."""

        return len(self.quantiles)
