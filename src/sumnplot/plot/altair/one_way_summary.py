"""One-way summary plots using Altair."""

from dataclasses import dataclass

import altair as alt
import polars as pl
from altair import FacetChart, LayerChart

from sumnplot.exceptions import MissingColumnError, SumNPlotError


class ColourError(SumNPlotError):
    """Raised when there is a problem with the colours specified for a plot."""


@dataclass(frozen=True)
class OneWaySummaryColours:
    """Colours used in one-way summary plots."""

    bar_colour: str
    line_colours: tuple[str, ...] | None = None

    def check_enough_line_colours(self, lines: list[str]) -> None:
        """Check that there are enough line colours for the number of lines to plot.

        Args:
            lines : The list of lines to plot.

        Raises:
            ColourError : If there are not enough line colours specified for the number
                of lines to plot.

        """
        if self.line_colours is None:
            msg = "No line colours specified."
            raise ColourError(msg)

        if len(lines) > len(self.line_colours):
            msg = (
                f"Not enough line colours specified for {len(lines)} lines. "
                f"Only {len(self.line_colours)} line colours specified."
            )
            raise ColourError(msg)


DEFAULT_COLOURS = OneWaySummaryColours(
    bar_colour="#ffa500",
    line_colours=(
        "#ff1493",
        "#00ff00",
        "#00ced1",
        "#0000ff",
        "#4b0082",
    ),
)


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
    colours: OneWaySummaryColours = DEFAULT_COLOURS,
    chart_width: int | None = 600,
    chart_height: int | None = 400,
    bar_opacity: float = 0.5,
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
        colours : The colours to use for the bars and lines in the chart. If not
            specified then default colours are used.
        chart_width : The width of the chart in pixels.
        chart_height : The height of the chart in pixels.
        bar_opacity : The opacity of the bars in the chart, between 0 and 1.

    Returns:
        The Altair chart containing bars and optionally multiple lines, sharing an
        x axis with lines on the right y axis and bars on the left y axis.

    Raises:
        ExceptionGroup : If any of the specified columns are not present in the input
            df argument then an ExceptionGroup is raised containing a
            MissingColumnError for each missing column.

    """
    column_errors = []
    if x_axis_column not in df.columns:
        column_errors.append(MissingColumnError(x_axis_column))
    if left_y_axis_column not in df.columns:
        column_errors.append(MissingColumnError(left_y_axis_column))
    if right_y_axis_columns:
        for col in right_y_axis_columns:
            if col not in df.columns:
                column_errors.append(MissingColumnError(col))

    if column_errors:
        msg = "Missing columns"
        raise ExceptionGroup(msg, column_errors)

    tooltip_columns = (
        [x_axis_column, *right_y_axis_columns, left_y_axis_column]
        if right_y_axis_columns
        else [x_axis_column, left_y_axis_column]
    )

    x_axis = alt.Chart(df).encode(
        tooltip=tooltip_columns,
        x=alt.X(f"{x_axis_column}:O", axis=alt.Axis(labelAngle=0, title=x_axis_column)),
    )

    bar = x_axis.mark_bar(color=colours.bar_colour, opacity=bar_opacity).encode(
        y=alt.Y(
            f"{left_y_axis_column}:Q",
            axis=alt.Axis(grid=False, title=left_y_axis_column),
        ),
    )

    line_marks = []

    if right_y_axis_columns:
        colours.check_enough_line_colours(right_y_axis_columns)

        right_y_min, right_y_max = _determine_y_axis_range(
            df=df,
            y_axis_columns=right_y_axis_columns,
        )

        for right_y_axis_index, right_y_axis_column in enumerate(right_y_axis_columns):
            line_colour = colours.line_colours[right_y_axis_index]

            y_values = alt.Y(
                f"{right_y_axis_column}:Q",
                axis=alt.Axis(grid=True, title="Response Scale"),
                scale=alt.Scale(domain=[right_y_min, right_y_max]),
            )

            line = x_axis.mark_line(color=line_colour).encode(y=y_values)
            line_points = x_axis.mark_point(color=line_colour).encode(y=y_values)

            line_marks.append(line)
            line_marks.append(line_points)

        right_axis_lines = alt.layer(*line_marks)

        chart = alt.layer(bar, right_axis_lines).resolve_scale(y="independent")

    else:
        chart = alt.layer(bar)

    properties = {}
    if chart_width is not None:
        properties["width"] = chart_width
    if chart_height is not None:
        properties["height"] = chart_height

    return chart.properties(**properties)
