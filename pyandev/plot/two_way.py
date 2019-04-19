import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from matplotlib.backends.backend_pdf import PdfPages

import pyandev.discretisation as d






def summary_plot(df, 
                 weights,
                 by_col, 
                 split_by_col,
                 observed = None, 
                 fitted = None, 
                 fitted2 = None, 
                 bar_type = 'stacked',
                 bars_percent = False,
                 bins = 20, 
                 bucketing_type = 'equal_width',
                 title = None,
                 figsize_h = 14, 
                 figsize_w = 8,
                 legend = True,
                 pdf = None,
                 return_summary_values = False,
                 ):
    '''Function to plot a two way summary of the specified variable, split by another variable.

    The two way summary graph consists of the following;
    - sum of weights (yellow bars, left axis)
    - mean observed values (pink line(s), right axis)
    - optionally, mean fitted values i.e. model predictions (green line(s), right axis)
    - optionally, mean fitted 2 values i.e. model 2 predictions (light green line(s), right axis) 
    by another variable (x axis) specified in the by_col argument and split by the split_by_col variable (bars).

    Parameters
    ----------
    df : pd.DataFrame
        Data of interest. Must contain columns with names supplied in weights and by_col args.
        
    weights : str
        Column name of weights in df. 

    by_col : str
        Column name in df of variable to summarise by.

    split_by_col : str
        Second column name in df of variable to summarise by.

    observed : str, defualt = None
        Optional. Column name of observed values in df.

    fitted : str, defualt = None
        Optional. Column name of fitted (predicted) values in df. If default value of None is passed
        then fitted values are not plotted.

    fitted2 : str, defualt = None
        Optional. Column name of second set of fitted (predicted) values in df. If default value of 
        None is passed then fitted2 values are not plotted.

    bar_type : str, default = 'stacked'
        Must be one of 'stacked' or 'side_by_side'. Method to display bars visualising sum of weights 
        split by 2 variables. If 'stacked' then bars corresponding to the split of the second index on summary_df
        are plotted on top of each other, if 'side_by_side' then they are plotted side by side. 

    bars_percent : bool, default = False
        Should bars be rescaled to percentages instead of sum of values?

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

    if not isinstance(split_by_col, str):

        raise TypeError('split_by_col should be a str')

    if not split_by_col in df.columns.values:

        raise ValueError('split_by_col; ' + split_by_col + ' not in df')

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

    if not isinstance(bar_type, str):

        raise TypeError('bar_type must be a str')

    if not bar_type in ['stacked', 'side_by_side']:

        raise ValueError('unexpected bar_type; ' + bar_type)

    if df[split_by_col].nunique(dropna = False) > 7:

        raise ValueError('number of levels of split_by_col (' + split_by_col + ') is too large (greater than 7)')

    if df[by_col].dtype.name in ['object', 'category']:
        
        cut = by_col

    elif is_numeric_dtype(df[by_col]):

        if df[by_col].nunique(dropna = False) <= bins:
            
            # if there are nulls convert to string otherwise they will not be included in the 
            # plot (stemming from the groupby)
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

    if is_numeric_dtype(df[split_by_col]) and df[split_by_col].isnull().sum() > 0:

        by_col2 = df[split_by_col].astype(str)

    else:

        by_col2 = split_by_col

    f = {weights: ['sum']}

    weights_summary = weights + '__sum'

    if not observed is None:

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

    summary_values = df.groupby([cut, by_col2]).agg(f)

    summary_values.index.names = [by_col, split_by_col]

    summary_values.columns = \
        [i + '__' + j for i, j in zip(summary_values.columns.get_level_values(0).values,
                                      summary_values.columns.get_level_values(1).values)]

    for col in [observed, fitted, fitted2]:

        if not col is None:

            summary_values[col + '__sum'] = summary_values[col + '__sum'] / summary_values[weights_summary]

            summary_values.rename(columns = {col + '__sum': col + '__mean'}, inplace = True)

    plot_summarised_variable_2way(summary_df = summary_values, 
                                  weights = weights_summary, 
                                  observed = observed_summary, 
                                  fitted = fitted_summary, 
                                  fitted2 = fitted2_summary, 
                                  bar_type = bar_type,
                                  bars_percent = bars_percent,
                                  title = title,
                                  figsize_h = figsize_h, 
                                  figsize_w = figsize_w,
                                  legend = legend,
                                  pdf = pdf)

    if return_summary_values:

        return summary_values







def plot_summarised_variable_2way(summary_df, 
                                  weights, 
                                  observed = None, 
                                  fitted = None, 
                                  fitted2 = None, 
                                  bar_type = 'stacked',
                                  bars_percent = False,
                                  title = None,
                                  figsize_h = 14, 
                                  figsize_w = 8,
                                  legend = True,
                                  pdf = None,
                                  ):
    '''Plot a two way variable summary from summary stats.
    
    This function should be used once a variable has been summarised.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Data of interest. Must contain columns with names supplied in weights and by_col args.
        
    weights : str, defualt = None
        Optional. Column name of weights in summary_df. 

    observed : str, 
        Column name of observed values in summary_df.

    fitted : str, defualt = None
        Optional. Column name of fitted (predicted) values in summary_df. If default value of None is passed
        then fitted values are not plotted.

    fitted2 : str, defualt = None
        Optional. Column name of second set of fitted (predicted) values in summary_df. If default value of 
        None is passed then fitted2 values are not plotted.

    bar_type : str, default = 'stacked'
        Must be one of 'stacked' or 'side_by_side'. Method to display bars visualising sum of weights 
        split by 2 variables. If 'stacked' then bars corresponding to the split of the second index on summary_df
        are plotted on top of each other, if 'side_by_side' then they are plotted side by side. 

    bars_percent : bool, default = False
        Should bars be rescaled to percentages instead of sum of values?

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
    
    '''

    bin_colours = ['gold', 'khaki', 'goldenrod', 'darkkhaki', 'darkgoldenrod', 'olive', 'y']

    obs_colours = ['magenta', 'm', 'orchid', 'mediumvioletred', 'deeppink', 'darkmagenta', 'darkviolet']

    fit_colours = ['forestgreen', 'darkgreen', 'seagreen', 'green', 'darkseagreen', 'g', 'mediumseagreen']

    fit2_colours = ['lime', 'limegreen', 'greenyellow', 'lawngreen', 'chartreuse', 'lightgreen', 'springgreen']

    if title is None:
        
        title = summary_df.index.names[0] + ' by ' + summary_df.index.names[1]

    fig, ax1 = plt.subplots(figsize=(figsize_h, figsize_w))

    unstack_weights = summary_df[weights].unstack()

    if bars_percent:

        row_totals = unstack_weights.sum(axis = 1)

        for col in unstack_weights.columns.values:

            unstack_weights[col] = unstack_weights[col] / row_totals

    # fill in levels with no weight (i.e. nulls) with 0
    unstack_weights.fillna(0, inplace = True)

    split_levels = unstack_weights.columns.values

    if bar_type == 'stacked':

        top_bins = np.zeros(unstack_weights.shape[0])

        # plot bin counts on 1st axis 
        for i in range(0, len(split_levels)):

            heights = unstack_weights.loc[:,split_levels[i]].reset_index(drop = True)

            ax1.bar(x = np.arange(unstack_weights.shape[0]), 
                    height = heights,
                    color = bin_colours[i],
                    label = split_levels[i],
                    bottom = top_bins)

            top_bins = top_bins + heights

        plt.xticks(np.arange(unstack_weights.shape[0]), unstack_weights.index, rotation = 270)

        x_ticket_offset = 0

    elif bar_type == 'side_by_side':

        bar_width =  0.8 / unstack_weights.shape[1]

        x_offset = 0

        for i in range(0, len(split_levels)):

            ax1.bar(np.arange(unstack_weights.shape[0]) + x_offset, 
                    unstack_weights.loc[:,split_levels[i]].reset_index(drop = True),
                    color = bin_colours[i],
                    width = bar_width,
                    label = split_levels[i])

            x_offset += bar_width
        
        x_ticket_offset = (bar_width * (unstack_weights.shape[1] / 2)) - (bar_width * 0.5)

        plt.xticks(np.arange(unstack_weights.shape[0]) + x_ticket_offset, unstack_weights.index, rotation = 270)

    else:

        raise ValueError('unexpected value for bar_type; ' + bar_type)

    ax2 = ax1.twinx()

    if not observed is None:

        unstack_observed = summary_df[observed].unstack()

        for i in range(len(split_levels)):
            
            # plot average observed on the 2nd axis in pink
            ax2.plot(unstack_observed.loc[:,split_levels[i]].reset_index(drop = True).dropna().index + x_ticket_offset,
                    unstack_observed.loc[:,split_levels[i]].reset_index(drop = True).dropna(),
                    color = obs_colours[i], 
                    linestyle = '-',
                    marker = 'D')

    if fitted is not None:

        unstack_fitted = summary_df[fitted].unstack()

        for i in range(len(split_levels)):
            
            # plot average observed on the 2nd axis in pink
            ax2.plot(unstack_fitted.loc[:,split_levels[i]].reset_index(drop = True).dropna().index + x_ticket_offset,
                     unstack_fitted.loc[:,split_levels[i]].reset_index(drop = True).dropna(),
                     color = fit_colours[i], 
                     linestyle = '-',
                     marker = 'D')

    if fitted2 is not None:

        unstack_fitted2 = summary_df[fitted2].unstack()

        for i in range(len(split_levels)):
            
            # plot average observed on the 2nd axis in pink
            ax2.plot(unstack_fitted2.loc[:,split_levels[i]].reset_index(drop = True).dropna().index + x_ticket_offset,
                     unstack_fitted2.loc[:,split_levels[i]].reset_index(drop = True).dropna(),
                     color = fit2_colours[i], 
                     linestyle = '-',
                     marker = 'D')

    ax1.legend(bbox_to_anchor=(1.05, 1), loc = 2, borderaxespad = 0.)
    plt.legend(bbox_to_anchor=(1.05, (0.94 - (0.03 * len(split_levels)))), loc = 2, borderaxespad = 0.)

    plt.title(title, fontsize = 20)
    
    if pdf is not None:

        pdf.savefig()

        plt.close(fig)



