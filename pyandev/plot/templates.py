import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyandev.plot as p



def lift_curve(df,
               weights,
               observed,
               fitted,
               **kwargs,
               ):
    '''Function to plot a "lift curve", which is average observed and fitted ordered by fitted values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data of interest, must contain a column supplied in variable.
        
    variable : str
        Column to bucket in df. Must be numeric type.

    weights : str
        Column name of weights in df. 

    observed : str
        Column name of observed values in df. 

    fitted : str
        Column name of fitted (predicted) values in df. 

    **kwargs :
        Other arguments passed on to pyandev.plot.one_way.summary_plot().

    '''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pd.DataFrame')

    if not isinstance(weights, str):

        raise TypeError('weights should be a str')

    if not weights in df.columns.values:

        raise ValueError('weights; ' + weights + ' not in df')

    if not isinstance(observed, str):

        raise TypeError('observed should be a str')

    if not observed in df.columns.values:

        raise ValueError('observed; ' + observed + ' not in df')

    if not isinstance(fitted, str):

        raise TypeError('fitted should be a str')

    if not fitted in df.columns.values:

        raise ValueError('fitted; ' + fitted + ' not in df')

    p.one_way.summary_plot(
        df = df, 
        weights = weights,
        by_col = fitted, 
        observed = observed, 
        fitted = fitted, 
        **kwargs,
    )



