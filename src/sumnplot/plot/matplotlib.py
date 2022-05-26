"""Module for plotting summarised data with matplotlib."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Optional

from ..checks import check_type, check_condition


def plot_summarised_variable(
    summary_df: pd.DataFrame,
    axis_right: int,
    axis_left: Optional[List[int]] = None,
    title: Optional[str] = None,
    figsize_h: int = 14,
    figsize_w: int = 8,
    legend: bool = True,
):
    """Produce one way summary plot from pre-summarised data.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with summarised info to plot.

    axis_right : int
        The index of the column in summary_df to plot on the right axis.
        Typically this would be a weights column.

    axis_left : Optional[List[int]], default = None
        The index of the columns in summary_df to plot on the left axis.
        Currently the maximum number of left axis columns supported is 5.

    title : str, default = None
        Title for the plot. If None summary_df.index.name is used as the title.

    figsize_h : int, default = 14
        Height of plot figure, used in matplotlib.pylot.subplots figsize arg.

    figsize_w : int, default = 8
        Width of plot figure, used in matplotlib.pylot.subplots figsize arg.

    legend : bool, default = True
        Should a legend be added to the plot?

    """

    LEFT_Y_AXIS_COLOURS = ["magenta", "forestgreen", "lime", "orangered", "dodgerblue"]

    check_type(summary_df, pd.DataFrame, "summary_df")
    check_type(axis_right, int, "axis_right")
    check_type(axis_left, list, "axis_left", none_allowed=True)
    check_type(title, str, "title", none_allowed=True)
    check_type(figsize_h, int, "figsize_h", none_allowed=True)
    check_type(figsize_w, int, "figsize_w", none_allowed=True)
    check_type(legend, bool, "legend")

    check_condition(
        axis_right <= summary_df.shape[1] - 1,
        f"only {summary_df.shape[1]} columns in summary_df but axis_right = {axis_right}",
    )

    if axis_left is not None:

        if axis_right in axis_left:
            raise ValueError(
                f"column index {axis_right} specified for both right and left axes"
            )

        if len(axis_left) > len(LEFT_Y_AXIS_COLOURS):
            raise ValueError(
                f"only {len(LEFT_Y_AXIS_COLOURS)} plots supports for the left axis but {len(axis_left)} given"
            )

        for axis_left_no, axis_left_index in enumerate(axis_left):
            check_type(axis_left_index, int, f"axis_left_index[{axis_left_no}]")
            check_condition(
                axis_left_index <= summary_df.shape[1] - 1,
                f"only {summary_df.shape[1]} columns in summary_df but axis_left[{axis_left_no}] = {axis_left_index}",
            )

    if title is None:
        title = summary_df.index.name

    _, ax1 = plt.subplots(figsize=(figsize_h, figsize_w))

    # plot bin counts on 1st axis
    ax1.bar(
        np.arange(summary_df.shape[0]),
        summary_df[summary_df.columns[axis_right]].reset_index(drop=True),
        color="gold",
        label=summary_df.columns[axis_right],
    )

    plt.xticks(np.arange(summary_df.shape[0]), summary_df.index, rotation=270)

    ax2 = ax1.twinx()

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


def plot_summarised_variable_2way(
    summary_df: pd.DataFrame,
    axis_right: int,
    axis_left: Optional[List[int]] = None,
    bar_type: Optional[str] = "stacked",
    bars_percent: Optional[bool] = False,
    title: Optional[str] = None,
    figsize_h: int = 14,
    figsize_w: int = 8,
    legend: bool = True,
):
    """Produce one way summary plot from pre-summarised data.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with summarised info to plot.

    axis_right : int
        The index of the column in summary_df to plot on the right axis.
        Typically this would be a weights column.

    axis_left : Optional[List[int]], default = None
        The index of the columns in summary_df to plot on the left axis.
        Currently only 3 left axis lines are supported.

    bar_type : Optional[str], default = "stacked"
        Type of bars to plot on the right axis. Must be either "stacked" or
        "side_by_side".

    bars_percent : Optional[bool], default = False
        Should bars on the right axis be plotted as percentage of total within
        each bar?

    title : str, default = None
        Title for the plot. If None summary_df.index.name is used as the title.

    figsize_h : int, default = 14
        Height of plot figure, used in matplotlib.pylot.subplots figsize arg.

    figsize_w : int, default = 8
        Width of plot figure, used in matplotlib.pylot.subplots figsize arg.

    legend : bool, default = True
        Should a legend be added to the plot?

    """

    BIN_COLOURS = [
        "gold",
        "khaki",
        "goldenrod",
        "darkkhaki",
        "darkgoldenrod",
        "olive",
        "y",
    ]

    LEFT_AXIS_COLOURS = [
        [
            "magenta",
            "m",
            "orchid",
            "mediumvioletred",
            "deeppink",
            "darkmagenta",
            "darkviolet",
        ],
        [
            "forestgreen",
            "darkgreen",
            "seagreen",
            "green",
            "darkseagreen",
            "g",
            "mediumseagreen",
        ],
        [
            "lime",
            "limegreen",
            "greenyellow",
            "lawngreen",
            "chartreuse",
            "lightgreen",
            "springgreen",
        ],
    ]

    check_type(summary_df, pd.DataFrame, "summary_df")
    check_type(axis_right, int, "axis_right")
    check_type(axis_left, list, "axis_left", none_allowed=True)
    check_type(bar_type, str, "bar_type", none_allowed=True)
    check_type(bars_percent, bool, "bars_percent", none_allowed=True)
    check_type(title, str, "title", none_allowed=True)
    check_type(figsize_h, int, "figsize_h", none_allowed=True)
    check_type(figsize_w, int, "figsize_w", none_allowed=True)
    check_type(legend, bool, "legend")

    check_condition(
        axis_right <= summary_df.shape[1] - 1,
        f"only {summary_df.shape[1]} columns in summary_df but axis_right = {axis_right}",
    )

    if axis_left is not None:

        if axis_right in axis_left:
            raise ValueError(
                f"column index {axis_right} specified for both right and left axes"
            )

        if len(axis_left) > len(LEFT_AXIS_COLOURS):
            raise ValueError(
                f"only {len(LEFT_AXIS_COLOURS)} plots supported for the left axis but {len(axis_left)} given"
            )

        for axis_left_no, axis_left_index in enumerate(axis_left):
            check_type(axis_left_index, int, f"axis_left_index[{axis_left_no}]")
            check_condition(
                axis_left_index <= summary_df.shape[1] - 1,
                f"only {summary_df.shape[1]} columns in summary_df but axis_left[{axis_left_no}] = {axis_left_index}",
            )

    if len(summary_df.index.levels[1]) > len(BIN_COLOURS):
        raise ValueError(
            f"only {len(BIN_COLOURS)} levels supported for the second groupby column but {len(summary_df.index.levels[1])} given in summary_df"
        )

    by_col = summary_df.index.names[0]
    split_by_col = summary_df.index.names[1]

    if title is None:
        title = f"{by_col} by {split_by_col}"

    _, ax1 = plt.subplots(figsize=(figsize_h, figsize_w))

    # turn data into by_col x split_by_col table and fill in levels
    # with no weight (i.e. nulls) with 0
    unstack_weights = summary_df[summary_df.columns[axis_right]].unstack()
    unstack_weights.fillna(0, inplace=True)

    if bars_percent:
        row_totals = unstack_weights.sum(axis=1)
        for col in unstack_weights.columns.values:
            unstack_weights[col] = unstack_weights[col] / row_totals

    split_levels = unstack_weights.columns.values

    unstack_weights.columns = pd.Index(
        [
            "("
            + split_by_col
            + " = "
            + str(x)
            + ") "
            + str(summary_df.columns[axis_right])
            for x in unstack_weights.columns.values
        ]
    )

    if bar_type == "stacked":

        top_bins = np.zeros(unstack_weights.shape[0])

        # plot bin counts on 1st axis
        for i in range(unstack_weights.shape[1]):

            heights = unstack_weights.loc[
                :, unstack_weights.columns.values[i]
            ].reset_index(drop=True)

            ax1.bar(
                x=np.arange(unstack_weights.shape[0]),
                height=heights,
                color=BIN_COLOURS[i],
                bottom=top_bins,
                label=unstack_weights.columns.values[i],
            )

            top_bins = top_bins + heights

        plt.xticks(
            np.arange(unstack_weights.shape[0]), unstack_weights.index, rotation=270
        )

        x_ticket_offset = 0

    elif bar_type == "side_by_side":

        bar_width = 0.8 / unstack_weights.shape[1]

        x_offset = 0

        for i in range(unstack_weights.shape[1]):

            ax1.bar(
                np.arange(unstack_weights.shape[0]) + x_offset,
                unstack_weights.loc[:, unstack_weights.columns.values[i]].reset_index(
                    drop=True
                ),
                color=BIN_COLOURS[i],
                width=bar_width,
                label=unstack_weights.columns.values[i],
            )

            x_offset += bar_width

        x_ticket_offset = (bar_width * (unstack_weights.shape[1] / 2)) - (
            bar_width * 0.5
        )

        plt.xticks(
            np.arange(unstack_weights.shape[0]) + x_ticket_offset,
            unstack_weights.index,
            rotation=270,
        )

    else:

        raise ValueError(f"unexpected value for bar_type; {bar_type}")

    ax2 = ax1.twinx()

    if axis_left is not None:

        for column_no, axis_left_column_index in enumerate(axis_left):

            unstacked_left_axis_column = summary_df[
                summary_df.columns[axis_left_column_index]
            ].unstack()

            unstacked_left_axis_column.columns = pd.Index(
                [
                    "("
                    + split_by_col
                    + " = "
                    + str(x)
                    + ") "
                    + str(summary_df.columns[axis_left_column_index])
                    for x in unstacked_left_axis_column.columns.values
                ]
            )

            for i in range(unstacked_left_axis_column.shape[1]):

                ax2.plot(
                    unstacked_left_axis_column.loc[
                        :, unstacked_left_axis_column.columns.values[i]
                    ]
                    .reset_index(drop=True)
                    .dropna()
                    .index
                    + x_ticket_offset,
                    unstacked_left_axis_column.loc[
                        :, unstacked_left_axis_column.columns.values[i]
                    ]
                    .reset_index(drop=True)
                    .dropna(),
                    color=LEFT_AXIS_COLOURS[column_no][i],
                    linestyle="-",
                    marker="D",
                    label=summary_df.columns[axis_left_column_index],
                )

    if legend:

        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        if axis_left is not None:

            plt.legend(
                bbox_to_anchor=(1.05, (0.94 - (0.03 * len(split_levels)))),
                loc=2,
                borderaxespad=0.0,
            )

    plt.title(title, fontsize=20)
