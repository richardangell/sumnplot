import numpy as np
import matplotlib.pyplot as plt


def plot_summarised_variable_2way(
    summary_df,
    weights,
    axis_left=None,
    bar_type="stacked",
    bars_percent=False,
    title=None,
    figsize_h=14,
    figsize_w=8,
    legend=True,
    pdf=None,
):
    """Plot a two way variable summary from summary stats.

    This function should be used once a variable has been summarised.
    """

    bin_colours = [
        "gold",
        "khaki",
        "goldenrod",
        "darkkhaki",
        "darkgoldenrod",
        "olive",
        "y",
    ]

    left_axis_colours = [
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

    by_col = summary_df.index.names[0]
    split_by_col = summary_df.index.names[1]

    if title is None:
        title = f"{by_col} by {split_by_col}"

    fig, ax1 = plt.subplots(figsize=(figsize_h, figsize_w))

    # turn data into by_col x split_by_col table and fill in levels
    # with no weight (i.e. nulls) with 0
    unstack_weights = summary_df[weights].unstack()
    unstack_weights.fillna(0, inplace=True)

    if bars_percent:
        row_totals = unstack_weights.sum(axis=1)
        for col in unstack_weights.columns.values:
            unstack_weights[col] = unstack_weights[col] / row_totals

    split_levels = unstack_weights.columns.values

    unstack_weights.columns = [
        "(" + split_by_col + " = " + str(x) + ") " + str(weights)
        for x in unstack_weights.columns.values
    ]

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
                color=bin_colours[i],
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
                color=bin_colours[i],
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

        raise ValueError("unexpected value for bar_type; " + bar_type)

    ax2 = ax1.twinx()

    if axis_left is not None:

        for column_no, axis_left_column_index in enumerate(axis_left):

            unstacked_left_axis_column = summary_df[
                summary_df.columns[axis_left_column_index]
            ].unstack()

            unstacked_left_axis_column.columns = [
                "("
                + split_by_col
                + " = "
                + str(x)
                + ") "
                + str(summary_df.columns[axis_left_column_index])
                for x in unstacked_left_axis_column.columns.values
            ]

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
                    color=left_axis_colours[column_no][i],
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
