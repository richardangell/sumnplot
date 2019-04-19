import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from matplotlib.backends.backend_pdf import PdfPages

import pyandev.discretisation as d









def plot_1way_summary(df, 
                      weights,
                      by_col, 
                      observed = None, 
                      fitted = None, 
                      fitted2 = None, 
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

    if df[by_col].dtype.name == ['object', 'category']:
        
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

    # function to apply to each variable
    f = {weights: ['sum']}

    weights_summary = weights + '_sum'

    if observed is not None:

        f[observed] = ['mean']

        observed_summary = observed + '_mean'

    else:

        observed_summary = observed

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

    summary_values = df.groupby(cut).agg(f)

    summary_values.index.name = by_col

    summary_values.columns = \
        [i + '_' + j for i, j in zip(summary_values.columns.get_level_values(0).values,
                                     summary_values.columns.get_level_values(1).values)]

    plot_summarised_variable(summary_df = summary_values, 
                             weights = weights_summary, 
                             observed = observed_summary, 
                             fitted = fitted_summary, 
                             fitted2 = fitted2_summary, 
                             title = title,
                             figsize_h = figsize_h, 
                             figsize_w = figsize_w,
                             legend = legend,
                             pdf = pdf)







def plot_summarised_variable(summary_df, 
                             weights, 
                             observed = None, 
                             fitted = None, 
                             fitted2 = None, 
                             title = None,
                             figsize_h = 14, 
                             figsize_w = 8,
                             legend = True,
                             pdf = None,
                             ):
    '''Plot a variable once it has already been summarised.'''

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
        
        ax2.plot(summary_df.iloc[:,fitted2].reset_index(drop = True).dropna().index,
                 summary_df.iloc[:,fitted2].reset_index(drop = True).dropna(),
                 color = 'lime', 
                 linestyle = '-',
                 marker = 'D')
    
    if legend:
    
        ax1.legend(bbox_to_anchor=(1.05, 1), loc = 2, borderaxespad = 0.)
        plt.legend(bbox_to_anchor=(1.05, 0.94), loc = 2, borderaxespad = 0.)
        
    plt.title(title, fontsize = 20)
    
    if pdf is not None:

        pdf.savefig()

        plt.close(fig)


