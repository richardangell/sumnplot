import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import discretisation as d




def plot_summarised_variable(summary_df, 
                             weights, 
                             observed, 
                             fitted = None, 
                             fitted2 = None, 
                             title = None,
                             figsize_h = 14, 
                             figsize_w = 8,
                             legend = True,
                             pdf = None):
    """Plot a variable after it has been summarised already"""

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









def plot_1way_summary(df, 
                      weights,
                      by_col, 
                      observed, 
                      fitted = None, 
                      fitted2 = None, 
                      bins = None, 
                      bucketing_type = None,
                      title = None,
                      figsize_h = 14, 
                      figsize_w = 8,
                      legend = True,
                      pdf = None):

    if df[by_col].dtype == 'O':
        
        cut = by_col

    elif df[by_col].dtype in ['int', 'int32', 'int64', 'float']:

        if df[by_col].nunique(dropna = False) <= bins:

            cut = by_col

        else:

            cut = d.discretise(data = df,
                               bucketing_type = bucketing_type,
                               variable = by_col,
                               n = bins,
                               weights = weights)

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

    summary_values = df.groupby(cut).agg(f)

    summary_values.columns = [i + '_' + j for i, j in zip(summary_values.columns.get_level_values(0).values,
                                                          summary_values.columns.get_level_values(1).values)]

    weights_summary = weights + '_sum'

    observed_summary = observed + '_mean'

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






def plot_2way_summary(df, 
                      weights,
                      by_col, 
                      split_by_col,
                      observed, 
                      fitted = None, 
                      fitted2 = None, 
                      bins = None, 
                      bucketing_type = None,
                      title = None,
                      figsize_h = 14, 
                      figsize_w = 8,
                      legend = True,
                      pdf = None):

    assert df[split_by_col].nunique(dropna = False) <= 7, \
        'number of levels of split_by_col (' + split_by_col + ') is too large (greater than 7)'

    if df[by_col].dtype == 'O':
        
        cut = by_col

    elif df[by_col].dtype in ['int', 'int32', 'int64', 'float']:

        if df[by_col].nunique(dropna = False) <= bins:

            cut = by_col

        else:

            cut = d.discretise(data = df,
                               bucketing_type = bucketing_type,
                               variable = by_col,
                               n = bins,
                               weights = weights)

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

    summary_values.columns = [i + '_' + j for i, j in zip(summary_values.columns.get_level_values(0).values,
                                                          summary_values.columns.get_level_values(1).values)]

    weights_summary = weights + '_sum'

    observed_summary = observed + '_mean'

    plot_summarised_variable_2way(summary_df = summary_values, 
                                  weights = weights_summary, 
                                  observed = observed_summary, 
                                  fitted = fitted_summary, 
                                  fitted2 = fitted2_summary, 
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
                                  title = None,
                                  figsize_h = 14, 
                                  figsize_w = 8,
                                  legend = True,
                                  pdf = None):
    """Plot a variable after it has been 2-way summarised already"""

    bin_colours = ['gold', 'khaki', 'goldenrod', 'darkkhaki', 'darkgoldenrod', 'olive', 'y']

    obs_colours = ['magenta', 'm', 'orchid', 'mediumvioletred', 'deeppink', 'darkmagenta', 'darkviolet']

    fit_colours = ['forestgreen', 'darkgreen', 'seagreen', 'green', 'darkseagreen', 'g', 'mediumseagreen']

    fit2_colours = ['lime', 'limegreen', 'greenyellow', 'lawngreen', 'chartreuse', 'lightgreen', 'springgreen']

    if title is None:
        
        title = summary_df.index.names[0] + ' by ' + summary_df.index.names[1]

    fig, ax1 = plt.subplots(figsize=(figsize_h, figsize_w))


    unstack_weights = summary_df[weights].unstack()

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

    unstack_observed = summary_df['income_mean'].unstack()

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





