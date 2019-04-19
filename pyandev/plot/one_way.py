import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from matplotlib.backends.backend_pdf import PdfPages

import pyandev.discretisation as d







def summary_plot(df, 
                 weights,
                 by_col, 
                 observed = None, 
                 fitted = None, 
                 fitted2 = None, 
                 fitted3 = None,
                 fitted4 = None,
                 bins = 20, 
                 bucketing_type = 'equal_width',
                 title = None,
                 figsize_h = 14, 
                 figsize_w = 8,
                 legend = True,
                 pdf = None,
                 return_summary_values = False,
                 ):
    '''Function to plot a one way summary of the specified variable.

    The one way summary graph consists of the following;
    - sum of weights (yellow bars, left axis)
    - optionally, mean observed values (pink line, right axis)
    - optionally, mean fitted values i.e. model predictions (green line, right axis)
    - optionally, mean fitted 2 values i.e. model 2 predictions (light green line, right axis)
    - optionally, mean fitted 3 values i.e. model 3 predictions (orange line, right axis)
    - optionally, mean fitted 4 values i.e. model 4 predictions (blue line, right axis)    
    by another variable, specified in the by_col argument.

    Parameters
    ----------
    df : pd.DataFrame
        Data of interest. Must contain columns with names supplied in weights and by_col args.
        
    weights : str
        Column name of weights in df. 

    by_col : str
        Column name in df of variable to summarise by.

    observed : str, defualt = None
        Optional. Column name of observed values in df. If default value of None is passed
        then observed values are not plotted.

    fitted : str, defualt = None
        Optional. Column name of fitted (predicted) values in df. If default value of None is passed
        then fitted values are not plotted.

    fitted2 : str, defualt = None
        Optional. Column name of second set of fitted (predicted) values in df. If default value of 
        None is passed then fitted2 values are not plotted.

    fitted3 : str, defualt = None
        Optional. Column name of third set of fitted (predicted) values in df. If default value of 
        None is passed then fitted3 values are not plotted.

    fitted4 : str, defualt = None
        Optional. Column name of fourth set of fitted (predicted) values in df. If default value of 
        None is passed then fitted4 values are not plotted.

    bins : int, default = 20
        The number of bins to bucket by_col into if it is a numeric column, and has more than that many
        unique values.

    bucketing_type : str, default = 'equal_width'
        Type of bucketing to use to discretise by_col if it is numeric (and has more than bins unique values).
        Must be one of the values accepted by pyandev.discretisation.discretise; "equal_width", "equal_weight",
        "quantile" or "weighted_quantile".

    title : str, default = None
        Title of the plot. If None by_col is used as the title.

    figsize_h : int, default = 14
        Height of plot figure, used in matplotlib.pylot.subplots figsize arg.

    figsize_w : int, default = 8
        Width of plot figure, used in matplotlib.pylot.subplots figsize arg.

    legend : bool, default = True
        Should a legend be added to the plot?

    pdf : str, default = None
        Full fielpath of a pdf to output the plot to. If None not pdf saved.

    return_summary_values : bool, default = False
        If True the table of summarised values that is plotted is returned from the function.

    Returns
    -------
    summary_values : pd.DataFrame
        If return_summary_values is True then a dataframe containing the plotted summary values is returned.

    '''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pd.DataFrame')

    if not isinstance(weights, str):

        raise TypeError('weights should be a str')

    if not weights in df.columns.values:

        raise ValueError('weights; ' + weights + ' not in df')

    if not isinstance(by_col, str):

        raise TypeError('by_col should be a str')

    if not by_col in df.columns.values:

        raise ValueError('by_col; ' + by_col + ' not in df')

    if not observed is None:

        if not isinstance(observed, str):

            raise TypeError('observed should be a str')

        if not observed in df.columns.values:

            raise ValueError('observed; ' + observed + ' not in df')

    if not fitted is None:

        if not isinstance(fitted, str):

            raise TypeError('fitted should be a str')

        if not fitted in df.columns.values:

            raise ValueError('fitted; ' + fitted + ' not in df')

    if not fitted2 is None:

        if not isinstance(fitted2, str):

            raise TypeError('fitted2 should be a str')

        if not fitted2 in df.columns.values:

            raise ValueError('fitted2; ' + fitted2 + ' not in df')

    if not fitted3 is None:

        if not isinstance(fitted3, str):

            raise TypeError('fitted3 should be a str')

        if not fitted3 in df.columns.values:

            raise ValueError('fitted3; ' + fitted3 + ' not in df')

    if not fitted4 is None:

        if not isinstance(fitted4, str):

            raise TypeError('fitted4 should be a str')

        if not fitted4 in df.columns.values:

            raise ValueError('fitted4; ' + fitted4 + ' not in df')

    if df[by_col].dtype.name == ['object', 'category']:
        
        cut = by_col

    elif is_numeric_dtype(df[by_col]):

        if df[by_col].nunique(dropna = False) <= bins:

            if df[by_col].isnull().sum() > 0:

                cut = df[by_col].astype(str)

            else:

                cut = by_col

        else:

            cut = d.discretise(df = df,
                               bucketing_type = bucketing_type,
                               variable = by_col,
                               n = bins,
                               weights_column = weights)

    else:

        raise TypeError('unexpected type for column; ' + by_col)

    # function to apply to each variable
    f = {weights: ['sum']}

    weights_summary = weights + '__sum'

    if observed is not None:

        f[observed] = ['sum']

        observed_summary = observed + '__mean'

    else:

        observed_summary = observed

    if fitted is not None:

        f[fitted] = ['sum']

        fitted_summary = fitted + '__mean'

    else:

        fitted_summary = fitted

    if fitted2 is not None:

        f[fitted2] = ['sum']

        fitted2_summary = fitted2 + '__mean'

    else:

        fitted2_summary = fitted2

    if fitted3 is not None:

        f[fitted3] = ['sum']

        fitted3_summary = fitted3 + '__mean'

    else:

        fitted3_summary = fitted3

    if fitted4 is not None:

        f[fitted4] = ['sum']

        fitted4_summary = fitted4 + '__mean'

    else:

        fitted4_summary = fitted4

    summary_values = df.groupby(cut).agg(f)

    summary_values.index.name = by_col

    summary_values.columns = \
        [i + '__' + j for i, j in zip(summary_values.columns.get_level_values(0).values,
                                      summary_values.columns.get_level_values(1).values)]

    # for each column divide column sum by weights sum to get bucket averages
    for col in [observed, fitted, fitted2, fitted3, fitted4]:

        if not col is None:

            summary_values[col + '__sum'] = summary_values[col + '__sum'] / summary_values[weights_summary]

            summary_values.rename(columns = {col + '__sum': col + '__mean'}, inplace = True)

    plot_summarised_variable(summary_df = summary_values, 
                             weights = weights_summary, 
                             observed = observed_summary, 
                             fitted = fitted_summary, 
                             fitted2 = fitted2_summary, 
                             fitted3 = fitted3_summary,
                             fitted4 = fitted4_summary,
                             title = title,
                             figsize_h = figsize_h, 
                             figsize_w = figsize_w,
                             legend = legend,
                             pdf = pdf)

    if return_summary_values:

        return summary_values





def plot_summarised_variable(summary_df, 
                             weights, 
                             observed = None, 
                             fitted = None, 
                             fitted2 = None, 
                             fitted3 = None,
                             fitted4 = None,
                             title = None,
                             figsize_h = 14, 
                             figsize_w = 8,
                             legend = True,
                             pdf = None,
                             ):
    '''Plot a one way variable summary from summary stats.
    
    This function should be used once a variable has been summarised.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with summarised info to plot. Must contain a column with name specified by weights.
        
    weights : str
        Column name of weights in summary_df. 

    observed : str, defualt = None
        Optional. Column name of observed values in summary_df. If default value of None is passed
        then observed values are not plotted.

    fitted : str, defualt = None
        Optional. Column name of fitted (predicted) values in summary_df. If default value of None is passed
        then fitted values are not plotted.

    fitted2 : str, defualt = None
        Optional. Column name of second set of fitted (predicted) values in summary_df. If default value of 
        None is passed then fitted2 values are not plotted.

    fitted3 : str, defualt = None
        Optional. Column name of third set of fitted (predicted) values in summary_df. If default value of 
        None is passed then fitted3 values are not plotted.

    fitted4 : str, defualt = None
        Optional. Column name of fourth set of fitted (predicted) values in summary_df. If default value of 
        None is passed then fitted4 values are not plotted.

    title : str, default = None
        Title of the plot. If None summary_df.index.name is used as the title.

    figsize_h : int, default = 14
        Height of plot figure, used in matplotlib.pylot.subplots figsize arg.

    figsize_w : int, default = 8
        Width of plot figure, used in matplotlib.pylot.subplots figsize arg.

    legend : bool, default = True
        Should a legend be added to the plot?

    pdf : str, default = None
        Full fielpath of a pdf to output the plot to. If None not pdf saved.

    '''

    if not isinstance(summary_df, pd.DataFrame):

        raise ValueError('summary_df should be a pd.DataFrame')


    if title is None:
        
        title = summary_df.index.name

    fig, ax1 = plt.subplots(figsize=(figsize_h, figsize_w))
    
    # plot bin counts on 1st axis 
    ax1.bar(np.arange(summary_df.shape[0]), 
            summary_df.loc[:,weights].reset_index(drop = True),
            color = 'gold',
            label = weights)
    
    plt.xticks(np.arange(summary_df.shape[0]), summary_df.index, rotation = 270)
    
    ax2 = ax1.twinx()
    
    if observed is not None:

        # plot average observed on the 2nd axis in pink
        ax2.plot(summary_df.loc[:,observed].reset_index(drop = True).dropna().index,
                 summary_df.loc[:,observed].reset_index(drop = True).dropna(),
                 color = 'magenta', 
                 linestyle = '-',
                 marker = 'D')
    
    if fitted is not None:
    
        # plot average observed on the 2nd axis in green
        ax2.plot(summary_df.loc[:,fitted].reset_index(drop = True).dropna().index,
                 summary_df.loc[:,fitted].reset_index(drop = True).dropna(),
                 color = 'forestgreen', 
                 linestyle = '-',
                 marker = 'D')

    if fitted2 is not None:
        
        ax2.plot(summary_df.loc[:,fitted2].reset_index(drop = True).dropna().index,
                 summary_df.loc[:,fitted2].reset_index(drop = True).dropna(),
                 color = 'lime', 
                 linestyle = '-',
                 marker = 'D')
    
    if fitted3 is not None:
        
        ax2.plot(summary_df.loc[:,fitted3].reset_index(drop = True).dropna().index,
                 summary_df.loc[:,fitted3].reset_index(drop = True).dropna(),
                 color = 'orangered', 
                 linestyle = '-',
                 marker = 'D')

    if fitted4 is not None:
        
        ax2.plot(summary_df.loc[:,fitted4].reset_index(drop = True).dropna().index,
                 summary_df.loc[:,fitted4].reset_index(drop = True).dropna(),
                 color = 'dodgerblue', 
                 linestyle = '-',
                 marker = 'D')

    if legend:
    
        ax1.legend(bbox_to_anchor=(1.05, 1), loc = 2, borderaxespad = 0.)
        plt.legend(bbox_to_anchor=(1.05, 0.94), loc = 2, borderaxespad = 0.)
        
    plt.title(title, fontsize = 20)
    
    if pdf is not None:

        pdf.savefig()

        plt.close(fig)


