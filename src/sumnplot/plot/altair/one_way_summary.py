"""One-way summary plots using Altair."""

import altair as alt
import polars as pl
from altair import FacetChart, LayerChart

BAR_COLOUR = "#ffa500"
LINE_COLOURS = [
    "#ff1493",
    "#00ff00",
    "#00ced1",
    "#0000ff",
]


def _determine_y_axis_range(
    df: pl.DataFrame,
    y_axis_columns: list[str],
    buffer: float = 0.1,
) -> tuple[int | float, int | float]:
    """Determine a suitable range y axis shared by multiple lines.

    The range between the min and max of all columns is extended by an amount equal to
    the buffer proportion of the range at both ends. If the buffer is 0 then the
    returned range will be the observed min and max across all columns in the input
    df argument.

    """
    min_all_y_columns = (
        df.select(pl.col(col).min() for col in y_axis_columns).min_horizontal().item()
    )

    max_all_y_columns = (
        df.select(pl.col(col).max() for col in y_axis_columns).max_horizontal().item()
    )

    range_ = max_all_y_columns - min_all_y_columns
    margin = buffer * range_

    min_with_buffer = min_all_y_columns - margin
    max_all_y_columns = max_all_y_columns + margin

    return min_with_buffer, max_all_y_columns


def produce_one_way_summary_plot(
    df: pl.DataFrame,
    *,
    x_axis_column: str,
    left_y_axis_column: str,
    right_y_axis_columns: list[str] | None = None,
    chart_width: int = 600,
) -> LayerChart | FacetChart:
    """Output an Altair chart one-way summary of pre-summarised data.

    Args:
        df : The pre-summarised data to plot.
        x_axis_column : the name of the column in df that contains the labels to plot
            along the x axis.
        left_y_axis_column : The name of the columns in df to plot on the left y axis,
            plotted as bars.
        right_y_axis_columns : The names of the columns in df to plot on the right y
            axis. Plotted as line is specified, if not specified then no lines are
            plotted on the chart.
        chart_width : The width of the chart in pixels.

    Returns:
        The Altair chart containing bars and optionally multiple lines, sharing an
        x axis with lines on the right y axis and bars on the left y axis.

    """
    tooltip_columns = (
        [*right_y_axis_columns, left_y_axis_column]
        if right_y_axis_columns
        else [left_y_axis_column]
    )

    x_axis = alt.Chart(df).encode(
        tooltip=tooltip_columns,
        x=alt.X(f"{x_axis_column}:O", axis=alt.Axis(labelAngle=0, title=x_axis_column)),
    )

    bar = x_axis.mark_bar(color=BAR_COLOUR, opacity=0.85).encode(
        y=alt.Y(
            f"{left_y_axis_column}:Q",
            axis=alt.Axis(grid=False, title=left_y_axis_column),
        ),
    )

    line_marks = []

    if right_y_axis_columns:
        right_y_min, right_y_max = _determine_y_axis_range(
            df=df,
            y_axis_columns=right_y_axis_columns,
        )

        for right_y_axis_index, right_y_axis_column in enumerate(right_y_axis_columns):
            line_colour = LINE_COLOURS[right_y_axis_index]

            y_values = alt.Y(
                f"{right_y_axis_column}:Q",
                axis=alt.Axis(grid=False, title="Response Scale"),
                scale=alt.Scale(domain=[right_y_min, right_y_max]),
            )

            line = x_axis.mark_line(color=line_colour).encode(y=y_values)
            line_points = x_axis.mark_point(color=line_colour).encode(y=y_values)

            line_marks.append(line)
            line_marks.append(line_points)

        right_axis_lines = alt.layer(*line_marks)

        chart = (
            alt.layer(bar, right_axis_lines)
            .resolve_scale(y="independent")
            .properties(width=chart_width)
            .interactive()
        )

    else:
        chart = alt.layer(bar).properties(width=chart_width).interactive()

    return chart
