import numpy as np
import matplotlib.pyplot as plt


def plot_summarised_variable(
    summary_df,
    axis_right,
    axis_left=None,
    title=None,
    figsize_h=14,
    figsize_w=8,
    legend=True,
):
    """Plot a one way variable summary from summary stats.

    This function should be used once a variable has been summarised.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with summarised info to plot. Must contain a column with name specified by weights.

    title : str, default = None
        Title of the plot. If None summary_df.index.name is used as the title.

    figsize_h : int, default = 14
        Height of plot figure, used in matplotlib.pylot.subplots figsize arg.

    figsize_w : int, default = 8
        Width of plot figure, used in matplotlib.pylot.subplots figsize arg.

    legend : bool, default = True
        Should a legend be added to the plot?

    """

    if axis_left is None and axis_right is None:
        raise ValueError("no columns to plot on either y axis specified")

    if title is None:
        title = summary_df.index.name

    fig, ax1 = plt.subplots(figsize=(figsize_h, figsize_w))

    # plot bin counts on 1st axis
    ax1.bar(
        np.arange(summary_df.shape[0]),
        summary_df[summary_df.columns[axis_right]].reset_index(drop=True),
        color="gold",
        label=summary_df.columns[axis_right],
    )

    plt.xticks(np.arange(summary_df.shape[0]), summary_df.index, rotation=270)

    ax2 = ax1.twinx()

    LEFT_Y_AXIS_COLOURS = ["magenta", "forestgreen", "lime", "orangered", "dodgerblue"]

    if axis_left is not None:

        for column_no, left_axis_column_index in enumerate(axis_left):

            ax2.plot(
                summary_df[summary_df.columns[left_axis_column_index]]
                .reset_index(drop=True)
                .dropna()
                .index,
                summary_df[summary_df.columns[left_axis_column_index]]
                .reset_index(drop=True)
                .dropna(),
                color=LEFT_Y_AXIS_COLOURS[column_no],
                linestyle="-",
                marker="D",
                label=summary_df.columns[left_axis_column_index],
            )

    if legend:

        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        if axis_left is not None:

            ax2.legend(bbox_to_anchor=(1.05, 0.94), loc=2, borderaxespad=0.0)

    plt.title(title, fontsize=20)
