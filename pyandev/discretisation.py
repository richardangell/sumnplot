import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def discretise(df, variable, n = None, bucketing_type = 'equal_width', weights_column = None, quantiles = np.linspace(0, 1, 11)):
    '''Function to discretise a numeric variable with one of the following approaches; equal_width, 
    equal_weight, quantile or weighted_quantile buckets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data of interest, must contain a column supplied in variable.
        
    variable : str
        Column to bucket in df. Must be numeric type.

    n : int
        Number of buckets to use in case of "equal_width" or "equal_weight" bucketing.

    bucketing_type : str, default = "equal_width"
        One of the following "equal_width", "equal_weight", "quantile" or "weighted_quantile".
        The type of bucketing to use.

    weights_column : str, default = None
        Column of weights in df. Default value of None results in equal weighted observations. Only used
        for "equal_weight" or "weighted_quantile" buckets.

    quantiles : float or array-like, default = np.linspace(0, 1, 11)
        Value(s) between 0 <= quantiles <= 1, the quantile(s) to compute.

    Returns
    -------
    bucketed_variable : pd.Series 
        Categorical version (pandas category type) of df[variable] bucketed according to bucketing_type.
        
    '''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df must be a pandas DataFrame')

    type_valid_options = ['equal_width', 'equal_weight', 'quantile', 'weighted_quantile']

    if not isinstance(bucketing_type, str):

        raise TypeError('bucketing_type must be a str')

    if not bucketing_type in type_valid_options:

        raise ValueError('invalid value for bucketing_type; ' +  bucketing_type)

    if not isinstance(variable, str):

        raise ValueError('variable must be a str')

    if not variable in df.columns.values:
        
        raise ValueError('variable; ' + variable + ' not in df')

    if weights_column is not None:

        if not isinstance(weights_column, str):

            raise TypeError('weights_column must be a str')

        if not weights_column in df.columns.values:

            raise ValueError('weights_column; ' + weights_column + ' not in df')

    if quantiles is not None:

        quantiles = np.array(quantiles)

        quantiles = np.unique(np.sort(np.append(quantiles, [0, 1])))

        if not np.all(quantiles >= 0):

            raise ValueError('quantiles should be all greater than or equal to 0')

        if not np.all(quantiles <= 1):

            raise ValueError('quantiles should be all less than or equal to 1')

    if bucketing_type == 'equal_width':

        bucketed_variable = equal_width(
            df = df, 
            variable = variable, 
            n = n,
        )

    elif bucketing_type == 'equal_weight':

        bucketed_variable = equal_weight(
            df = df, 
            variable = variable,
            weights = weights_column,
            n = n,
        )

    elif bucketing_type == 'quantile':

        bucketed_variable = quantile(
            df = df, 
            variable = variable,
            quantiles = quantiles,
        )

    elif bucketing_type == 'weighted_quantile':

        bucketed_variable = weighted_quantile(
            df = df, 
            variable = variable,
            weights = weights_column,
            quantiles = quantiles,
        )

    return bucketed_variable



def equal_width(df, variable, n):
    '''Function to split variable into n buckets of equal width.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data of interest, must contain a column supplied in variable.
        
    variable : str
        Column to bucket in df. Must be numeric type.
    
    n : int
        Number of buckets.

    Returns
    -------
    cat : pd.Series 
        Categorical version (pandas category type) of df[variable] bucketed into n
        levels of equal width. 
    '''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pd.DataFrame')

    if not isinstance(variable, str):

        raise TypeError('variable must be a str')

    if not variable in df.columns.values:

        raise ValueError('variable; ' + variable + ' not in df')

    if not is_numeric_dtype(df[variable]):

        raise TypeError('df[' + variable + '] is not numeric type')

    if not isinstance(n, int):

        raise TypeError('n must be an int')

    if not n > 0:

        raise ValueError('n must be greater than 0')

    # see https://github.com/pandas-dev/pandas/issues/17047 for issues with include_lowest
    variable_cut = pd.cut(df[variable], n, include_lowest = True, duplicates = 'drop')

    variable_cut = add_null_category(variable_cut)

    return(variable_cut)



def equal_weight(df, variable, n, weights = None):
    '''Function to split variable into n buckets of equal weight.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data of interest, must contain a column supplied in variable.
        
    variable : str
        Column to bucket in df. Must be numeric type.

    weights : str
        Column of weights in df. Default value of None results in equal weighted observations.

    n : int
        Number of buckets.

    Returns
    -------
    cat : pd.Series 
        Categorical version (pandas category type) of df[variable] bucketed into n
        levels of equal weight. 
    '''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pd.DataFrame')

    if not isinstance(variable, str):

        raise TypeError('variable must be a str')

    if not variable in df.columns.values:

        raise ValueError('variable; ' + variable + ' not in df')

    if not is_numeric_dtype(df[variable]):

        raise TypeError('df[' + variable + '] is not numeric type')

    if not isinstance(n, int):

        raise TypeError('n must be an int')

    if not n > 0:

        raise ValueError('n must be greater than 0')

    if weights is None:

        weights = np.ones(df.shape[0])

    else:

        if not isinstance(weights, str):

            raise TypeError('weights must be a str')

        if not weights in df.columns.values:

            raise ValueError('weights; ' + weights + ' not in df')

        if not is_numeric_dtype(df[weights]):

            raise TypeError('df[' + weights + '] is not numeric type')

        weights = df[weights]

    weight_quantiles = compute_weighted_quantile(
        values = df[variable], 
        quantiles = np.array(np.linspace(start = 0, stop = 1, num = n + 1)), 
        sample_weight = weights, 
        values_sorted = False,
    )

    # remove null values from weightd quantiles array, o/w results in the following error;
    # ValueError: missing values must be missing in the same location both left and right sides
    # from line 452 in pandas/core/arrays/interval.py
    variable_cut = pd.cut(
        df[variable], 
        np.unique(weight_quantiles[~np.isnan(weight_quantiles)]), 
        include_lowest = True,
        duplicates = 'drop',
    )
    
    variable_cut = add_null_category(variable_cut)

    return variable_cut



def quantile(df, variable, quantiles):
    '''Function to split variable into buckets according to the supplied quantiles.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data of interest, must contain a column supplied in variable.
        
    variable : str
        Column to bucket in df. Must be numeric type.
    
    quantiles : float or array-like
        Value(s) between 0 <= quantiles <= 1, the quantile(s) to compute.

    Returns
    -------
    cat : pd.Series 
        Categorical version (pandas category type) of df[variable] bucketed according to
        the supplied quantiles. 
    '''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pd.DataFrame')

    if not isinstance(variable, str):

        raise TypeError('variable must be a str')

    if not variable in df.columns.values:

        raise ValueError('variable; ' + variable + ' not in df')

    if not is_numeric_dtype(df[variable]):

        raise TypeError('df[' + variable + '] is not numeric type')

    if not np.all(quantiles >= 0):

        raise ValueError('quantiles should be all greater than or equal to 0')

    if not np.all(quantiles <= 1):

        raise ValueError('quantiles should be all less than or equal to 1')

    quantile_values = df[variable].quantile(quantiles)

    variable_cut = pd.cut(df[variable], np.unique(quantile_values), include_lowest = True, duplicates = 'drop')

    variable_cut = add_null_category(variable_cut)

    return variable_cut    



def compute_weighted_quantile(values, quantiles, sample_weight = None, values_sorted = False):
    '''Funtion to calculate weighted percentiles.

    Code modified from the answer given by users Alleo & Max Ghenis on stackoverflow; https://stackoverflow.com/a/29677616.
    See https://en.wikipedia.org/wiki/Percentile#The_weighted_percentile_method for description of method.
    Removed the old_style arg and associated code from stackoverflow answer.

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
    '''

    if not np.all(quantiles >= 0) and np.all(quantiles <= 1):

        raise ValueError('quantiles must be in the range [0, 1]')

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



def weighted_quantile(df, variable, quantiles, weights = None):
    '''Function to split variable into buckets according to the supplied weighted quantiles.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data of interest, must contain a column supplied in variable.
        
    variable : str
        Column to bucket in df. Must be numeric type.
    
    quantiles : float or array-like
        Value(s) between 0 <= quantiles <= 1, the quantile(s) to compute.

    weights : str, default = None
        Column of weights in df. Default value of None results in observations treated as equally weighted.

    Returns
    -------
    cat : pd.Series 
        Categorical version (pandas category type) of df[variable] bucketed according to
        the supplied quantiles. 
    '''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pd.DataFrame')

    if not isinstance(variable, str):

        raise TypeError('variable must be a str')

    if not variable in df.columns.values:

        raise ValueError('variable; ' + variable + ' not in df')

    if not is_numeric_dtype(df[variable]):

        raise TypeError('df[' + variable + '] is not numeric type')

    if not np.all(quantiles >= 0):

        raise ValueError('quantiles should be all greater than or equal to 0')

    if not np.all(quantiles <= 1):

        raise ValueError('quantiles should be all less than or equal to 1')

    if not weights is None:

        if not isinstance(weights, str):

            raise TypeError('weights must be a str')

        if not weights in df.columns.values:

            raise ValueError('weights; ' + weights + ' not in df')

        if not is_numeric_dtype(df[weights]):

            raise TypeError('df[' + weights + '] is not numeric type')

        weights = df[weights]

    weight_quantiles = compute_weighted_quantile(
        values = df[variable], 
        quantiles = quantiles, 
        sample_weight = weights, 
        values_sorted = False,
    )

    # again drop nulls from weighted quantiles
    variable_cut = pd.cut(
        df[variable], 
        np.unique(weight_quantiles[~np.isnan(weight_quantiles)]),
        include_lowest = True, 
        duplicates = 'drop',
    )
    
    variable_cut = add_null_category(variable_cut)

    return variable_cut



def add_null_category(categorical_variable, null_category_name = 'Null'):
    '''Function to add new categorical level to categorical variable and set NAs to this category.
    
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
    '''

    if not isinstance(categorical_variable, pd.Series):

        raise TypeError('categorical_variable should be a pandas Series')

    if not categorical_variable.dtype.name == 'category':

        raise TypeError('categorical_variable should be category dtype')

    if not isinstance(null_category_name, str):

        raise TypeError('null_category_name must be a str')

    if null_category_name in categorical_variable.cat.categories:

        raise ValueError('null_category_name; ' + null_category_name + ' already in categorical_variable.cat.categories')

    cat = categorical_variable.cat.add_categories([null_category_name])

    cat.fillna('Null', inplace = True)

    return cat




