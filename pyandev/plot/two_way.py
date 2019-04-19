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
                 observed, 
                 fitted = None, 
                 fitted2 = None, 
                 bars_percent = False,
                 bins = 20, 
                 bucketing_type = 'equal_width',
                 title = None,
                 figsize_h = 14, 
                 figsize_w = 8,
                 legend = True,
                 pdf = None,
                 ):

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

    if df[split_by_col].nunique(dropna = False) > 7:

        raise ValueError('number of levels of split_by_col (' + split_by_col + ') is too large (greater than 7)')

    if df[by_col].dtype.name in ['object', 'category']:
        
        cut = by_col

    elif is_numeric_dtype(df[by_col]):

        if df[by_col].nunique(dropna = False) <= bins:

            cut = by_col

        else:

            cut = d.discretise(df = df,
                               bucketing_type = bucketing_type,
                               variable = by_col,
                               n = bins,
                               weights_column = weights)

    else:

        raise TypeError('unexpected type for column; ' + by_col)

    f = {weights: ['sum'], observed: ['mean']}

    if fitted is not None:

        f[fitted] = ['mean']

        fitted_summary = fitted + '_mean'

    else:

        fitted_summary = fitted

    if fitted2 is not None:

        f[fitted2] = ['mean']

        fitted2_summary = fitted2 + '_mean'

    else:

        fitted2_summary = fitted2

    summary_values = df.groupby([cut, df[split_by_col]]).agg(f)

    summary_values.index.names = [by_col, split_by_col]

    summary_values.columns = \
        [i + '_' + j for i, j in zip(summary_values.columns.get_level_values(0).values,
                                     summary_values.columns.get_level_values(1).values)]

    weights_summary = weights + '_sum'

    observed_summary = observed + '_mean'

    plot_summarised_variable_2way(summary_df = summary_values, 
                                  weights = weights_summary, 
                                  observed = observed_summary, 
                                  fitted = fitted_summary, 
                                  fitted2 = fitted2_summary, 
                                  bars_percent = bars_percent,
                                  title = title,
                                  figsize_h = figsize_h, 
                                  figsize_w = figsize_w,
                                  legend = legend,
                                  pdf = pdf)









def plot_summarised_variable_2way(summary_df, 
                                  weights, 
                                  observed, 
                                  fitted = None, 
                                  fitted2 = None, 
                                  bars_percent = False,
                                  title = None,
                                  figsize_h = 14, 
                                  figsize_w = 8,
                                  legend = True,
                                  pdf = None,
                                  ):
    '''Plot a variable after it has been 2-way summarised already'''

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

    # plot bin counts on 1st axis 
    ax1.bar(np.arange(unstack_weights.shape[0]), 
            unstack_weights.loc[:,split_levels[0]].reset_index(drop = True),
            color = bin_colours[0],
            label = split_levels[0])

    top_bins = unstack_weights.loc[:,split_levels[0]].reset_index(drop = True)

    for i in range(1, len(split_levels)):
        
        ax1.bar(np.arange(unstack_weights.shape[0]), 
                unstack_weights.loc[:,split_levels[i]].reset_index(drop = True),
                color = bin_colours[i],
                label = split_levels[i],
                bottom = top_bins)

        top_bins = top_bins + unstack_weights.loc[:,split_levels[i]].reset_index(drop = True)

    plt.xticks(np.arange(unstack_weights.shape[0]), unstack_weights.index, rotation = 270)

    ax2 = ax1.twinx()

    unstack_observed = summary_df[observed].unstack()

    for i in range(len(split_levels)):
        
        # plot average observed on the 2nd axis in pink
        ax2.plot(unstack_observed.loc[:,split_levels[i]].reset_index(drop = True).dropna().index,
                 unstack_observed.loc[:,split_levels[i]].reset_index(drop = True).dropna(),
                 color = obs_colours[i], 
                 linestyle = '-',
                 marker = 'D')

    if fitted is not None:

        unstack_fitted = summary_df[fitted].unstack()

        for i in range(len(split_levels)):
            
            # plot average observed on the 2nd axis in pink
            ax2.plot(unstack_fitted.loc[:,split_levels[i]].reset_index(drop = True).dropna().index,
                     unstack_fitted.loc[:,split_levels[i]].reset_index(drop = True).dropna(),
                     color = fit_colours[i], 
                     linestyle = '-',
                     marker = 'D')

    if fitted2 is not None:

        unstack_fitted2 = summary_df[fitted2].unstack()

        for i in range(len(split_levels)):
            
            # plot average observed on the 2nd axis in pink
            ax2.plot(unstack_fitted2.loc[:,split_levels[i]].reset_index(drop = True).dropna().index,
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



