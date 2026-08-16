"""Component tests for the one_way_summary module."""

from typing import Any, Literal

import altair as alt
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sumnplot.plot.altair.one_way_summary import (
    DEFAULT_ONE_WAY_PLOT_COLOURS,
    OneWaySummaryColours,
    produce_one_way_summary_plot,
)
from sumnplot.summarisation.summary_operation import SummaryOperation
from sumnplot.summarisation.summary_table import SummaryTable


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "x_var": ["a", "b", "c", "d", "e"],
            "w_col": [10, 12, 11, 9, 10],
            "f0": [0.5, 0.49, 0.48, 0.47, 0.46],
            "f1": [0.47, 0.48, 0.49, 0.5, 0.5],
            "f2": [0.48, 0.48, 0.48, 0.48, 0.48],
            "f3": [0.47, 0.48, 0.49, 0.48, 0.47],
            "f4": [0.49, 0.48, 0.485, 0.48, 0.49],
        },
    )


@pytest.fixture
def summary_table(sample_data: pl.DataFrame) -> SummaryTable:
    """Create a SummaryTable for testing."""
    groupby_columns = ["x_var"]
    summarised_column_types = {
        "w_col": SummaryOperation.SUM,
        "f0": SummaryOperation.WEIGHTED_AVERAGE,
        "f1": SummaryOperation.WEIGHTED_AVERAGE,
        "f2": SummaryOperation.WEIGHTED_AVERAGE,
        "f3": SummaryOperation.WEIGHTED_AVERAGE,
        "f4": SummaryOperation.WEIGHTED_AVERAGE,
    }
    return SummaryTable(
        sample_data,
        groupby_columns=groupby_columns,
        summarised_column_types=summarised_column_types,
    )


@pytest.fixture
def sample_data_with_extra_columns(sample_data: pl.DataFrame) -> pl.DataFrame:
    """Create a sample DataFrame with extra columns for testing."""
    return sample_data.with_columns(
        pl.col("f0").alias("f5"),
        pl.col("f1").alias("f6"),
        pl.col("f2").alias("f7"),
        pl.col("f3").alias("f8"),
        pl.col("f4").alias("f9"),
    )


@pytest.fixture
def summary_table_with_extra_columns(
    sample_data_with_extra_columns: pl.DataFrame,
) -> SummaryTable:
    """Create a SummaryTable with extra columns for testing."""
    groupby_columns = ["x_var"]
    summarised_column_types = {
        "w_col": SummaryOperation.SUM,
        "f0": SummaryOperation.WEIGHTED_AVERAGE,
        "f1": SummaryOperation.WEIGHTED_AVERAGE,
        "f2": SummaryOperation.WEIGHTED_AVERAGE,
        "f3": SummaryOperation.WEIGHTED_AVERAGE,
        "f4": SummaryOperation.WEIGHTED_AVERAGE,
        "f5": SummaryOperation.WEIGHTED_AVERAGE,
        "f6": SummaryOperation.WEIGHTED_AVERAGE,
        "f7": SummaryOperation.WEIGHTED_AVERAGE,
        "f8": SummaryOperation.WEIGHTED_AVERAGE,
        "f9": SummaryOperation.WEIGHTED_AVERAGE,
    }
    return SummaryTable(
        sample_data_with_extra_columns,
        groupby_columns=groupby_columns,
        summarised_column_types=summarised_column_types,
    )


def _assert_top_level_keys(
    chart_dict: dict,
    expected_fixed_keys: tuple[str, ...] = (
        "config",
        "layer",
        "data",
        "$schema",
        "datasets",
    ),
    **chart_specific_expected_keys: dict[str, Any],
) -> None:
    """Assert that the top-level keys of the chart dictionary match the expected keys.

    Tests that the user supplied keys exist in the top level keys of the chart dict
    and take the supplied values. Also checks for the existence of the fixed keys
    that are always present in the chart dict; the keys in `expected_fixed_keys`.

    """
    fixed_keys = list(expected_fixed_keys)
    expected_keys = fixed_keys + list(chart_specific_expected_keys.keys())
    assert set(chart_dict.keys()) == set(expected_keys)

    for expected_key, expected_value in chart_specific_expected_keys.items():
        assert chart_dict[expected_key] == expected_value


def _assert_chart_data(
    chart_dict: dict,
    dataset_key: str,
    expected_df: pl.DataFrame,
) -> None:
    """Assert that the chart data matches the expected DataFrame."""
    assert_frame_equal(
        expected_df,
        pl.DataFrame(chart_dict["datasets"][dataset_key]),
    )


def _construct_expected_tooltip(
    x_axis_name: str,
    y_axis_names: list[str],
) -> list[dict]:
    """Create tooltip structure with x axis and y axis names for testing.

    The x-axis field is expected to be nominal type and the y-axis fields are
    expected to be quantitative types. The x axis field is expected to be the
    first element in the tooltip list, followed by the y axis fields in the
    order they are provided in the y_axis_names list.

    """
    tooltip = [{"field": x_axis_name, "type": "nominal"}]
    tooltip.extend(
        [
            {"field": y_axis_name, "type": "quantitative"}
            for y_axis_name in y_axis_names
        ],
    )
    return tooltip


def _construct_expected_bar_layer(
    colour: str,
    opacity: float,
    x_axis_name: str,
    y_axis_name: str,
    expected_title: str,
    label_angle: int,
    extra_tooltip_fields: list[str] | None = None,
) -> dict:
    """Construct the expected bar layer dictionary.

    The bar layer has mark and encoding keys. The mark key contains the type of mark
    (bar), the colour and opacity.

    The encoding key contains the x and y axis encodings and the tooltip configuration.

    """
    tooltip_fields = (
        [*extra_tooltip_fields, y_axis_name] if extra_tooltip_fields else [y_axis_name]
    )
    expected_tooltip = _construct_expected_tooltip(
        x_axis_name=x_axis_name,
        y_axis_names=tooltip_fields,
    )

    expected_x_axis = {
        "axis": {"labelAngle": label_angle, "title": x_axis_name},
        "field": x_axis_name,
        "type": "ordinal",
        "sort": None,
    }
    expected_y_axis = {
        "axis": {"grid": False, "title": y_axis_name},
        "field": y_axis_name,
        "type": "quantitative",
    }

    expected_bar_encoding = {
        "tooltip": expected_tooltip,
        "x": expected_x_axis,
        "y": expected_y_axis,
    }

    expected_bar_mark = {"type": "bar", "color": colour, "opacity": opacity}

    expected_title_ = {"anchor": "middle", "text": expected_title}

    return {
        "mark": expected_bar_mark,
        "encoding": expected_bar_encoding,
        "title": expected_title_,
    }


def _construct_line_or_point_layer(
    type_: Literal["line", "point"],
    *,
    colours: list[str],
    x_axis_name: str,
    y_axis_names: list[str],
    y_axis_label: str,
    y_axis_range: tuple[float, float],
    x_axis_label_angle: int = 0,
    unpivoted_value_name: str = "value",
    unpivoted_variable_name: str = "variable",
    encoding_color_legend_is_none: bool = False,
) -> dict:
    """Construct either point or line layer for the right y axis."""
    expected_x_axis = {
        "axis": {"labelAngle": x_axis_label_angle, "title": x_axis_name},
        "field": x_axis_name,
        "type": "ordinal",
        "sort": None,
    }
    expected_y_axis = {
        "axis": {"grid": True, "title": y_axis_label},
        "field": unpivoted_value_name,
        "type": "quantitative",
        "scale": {"domain": list(y_axis_range)},
    }
    expected_color = {
        "field": unpivoted_variable_name,
        "type": "nominal",
        "scale": {"domain": y_axis_names, "range": colours},
        "legend": None,
    }
    if not encoding_color_legend_is_none:
        del expected_color["legend"]

    expected_line_encoding = {
        "x": expected_x_axis,
        "y": expected_y_axis,
        "color": expected_color,
    }

    expected_line_mark = {"type": type_}

    return {
        "mark": expected_line_mark,
        "encoding": expected_line_encoding,
    }


def test_bars_plot_only(sample_data: pl.DataFrame, summary_table: SummaryTable):
    """Test that the function produces a bar plot when only."""
    opacity = 0.6
    height = 90
    width = 120
    title = "Bars Only Summary Plot"
    label_angle = 90

    chart = produce_one_way_summary_plot(
        summary_table,
        x_axis_column="x_var",
        left_y_axis_column="w_col",
        right_y_axis_columns=None,
        title=title,
        chart_width=width,
        chart_height=height,
        bar_opacity=opacity,
        x_axis_label_angle=label_angle,
    )
    assert chart is not None
    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()

    _assert_top_level_keys(chart_dict, height=height, width=width)  # type: ignore[reportArgumentType]

    # One dataset for the entire chart as there is only one mark.
    _assert_chart_data(
        chart_dict=chart_dict,
        dataset_key=chart_dict["data"]["name"],
        expected_df=sample_data,
    )

    expected_bar_layer = _construct_expected_bar_layer(
        colour=DEFAULT_ONE_WAY_PLOT_COLOURS.bar_colour,
        opacity=opacity,
        x_axis_name="x_var",
        y_axis_name="w_col",
        label_angle=label_angle,
        expected_title=title,
    )

    assert len(chart_dict["layer"]) == 1
    assert chart_dict["layer"][0] == expected_bar_layer


def test_bar_and_single_line_plot(
    sample_data: pl.DataFrame,
    summary_table: SummaryTable,
):
    """Test that the function produces a bar plot and a single line plot."""
    opacity = 0.9
    height = 100
    width = 140

    chart = produce_one_way_summary_plot(
        summary_table,
        x_axis_column="x_var",
        left_y_axis_column="w_col",
        right_y_axis_columns=["f0"],
        title=None,
        chart_width=width,
        chart_height=height,
        bar_opacity=opacity,
    )
    assert chart is not None
    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()

    _assert_top_level_keys(
        chart_dict,
        expected_fixed_keys=("config", "layer", "$schema", "datasets"),
        height=height,  # type: ignore[reportArgumentType]
        width=width,  # type: ignore[reportArgumentType]
        resolve={"scale": {"y": "independent", "color": "independent"}},
    )

    assert chart_dict["layer"][0]["mark"]["type"] == "bar", (
        "Expecting layer 0 to be a bar mark."
    )
    _assert_chart_data(
        chart_dict=chart_dict,
        dataset_key=chart_dict["layer"][0]["data"]["name"],
        expected_df=sample_data,
    )

    assert chart_dict["layer"][1]["mark"]["type"] == "line", (
        "Expecting layer 1 to be a line mark."
    )
    expected_line_data_unpivoted = sample_data.with_columns(
        pl.lit("f0").alias("variable"),
        pl.col("f0").alias("value"),
    ).select(["x_var", "variable", "value"])
    _assert_chart_data(
        chart_dict=chart_dict,
        dataset_key=chart_dict["layer"][1]["data"]["name"],
        expected_df=expected_line_data_unpivoted,
    )

    assert chart_dict["layer"][2]["mark"]["type"] == "point", (
        "Expecting layer 2 to be a point mark."
    )
    _assert_chart_data(
        chart_dict=chart_dict,
        dataset_key=chart_dict["layer"][2]["data"]["name"],
        expected_df=expected_line_data_unpivoted,
    )

    expected_bar_layer = _construct_expected_bar_layer(
        colour=DEFAULT_ONE_WAY_PLOT_COLOURS.bar_colour,
        opacity=opacity,
        x_axis_name="x_var",
        y_axis_name="w_col",
        extra_tooltip_fields=["f0"],
        expected_title="x_var",
        label_angle=0,
    )
    # Use the actual dataset name from the chart dict.
    expected_bar_layer["data"] = {"name": chart_dict["layer"][0]["data"]["name"]}

    assert DEFAULT_ONE_WAY_PLOT_COLOURS.line_colours is not None

    expected_line_layer = _construct_line_or_point_layer(
        type_="line",
        colours=list(DEFAULT_ONE_WAY_PLOT_COLOURS.line_colours),
        x_axis_name="x_var",
        y_axis_names=["f0"],
        y_axis_range=(0.456, 0.504),  # range +-0.1 * (0.5 - 0.46)
        y_axis_label="Response Scale",
    )
    # Use the actual dataset name from the chart dict.
    expected_line_layer["data"] = {"name": chart_dict["layer"][1]["data"]["name"]}

    expected_points_layer = _construct_line_or_point_layer(
        type_="point",
        colours=list(DEFAULT_ONE_WAY_PLOT_COLOURS.line_colours),
        x_axis_name="x_var",
        y_axis_names=["f0"],
        y_axis_range=(0.456, 0.504),  # range +-0.1 * (0.5 - 0.46)
        y_axis_label="Response Scale",
        encoding_color_legend_is_none=True,
    )
    # Use the actual dataset name from the chart dict.
    expected_points_layer["data"] = {"name": chart_dict["layer"][2]["data"]["name"]}

    assert len(chart_dict["layer"]) == 3
    assert chart_dict["layer"][0] == expected_bar_layer
    assert chart_dict["layer"][1] == expected_line_layer
    assert chart_dict["layer"][2] == expected_points_layer


def test_bar_and_multiple_line_plot(
    sample_data: pl.DataFrame,
    summary_table: SummaryTable,
):
    """Test that the function produces a bar plot and multiple line plots."""
    opacity = 0.3
    height = 200
    width = 240
    title = "Bar and Multiple Lines Summary Plot"

    right_y_axis_columns = ["f0", "f1", "f2", "f3"]

    chart = produce_one_way_summary_plot(
        summary_table,
        x_axis_column="x_var",
        left_y_axis_column="w_col",
        right_y_axis_columns=right_y_axis_columns,
        title=title,
        chart_width=width,
        chart_height=height,
        bar_opacity=opacity,
    )
    assert chart is not None
    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()

    _assert_top_level_keys(
        chart_dict,
        expected_fixed_keys=("config", "layer", "$schema", "datasets"),
        height=height,  # type: ignore[reportArgumentType]
        width=width,  # type: ignore[reportArgumentType]
        resolve={"scale": {"y": "independent", "color": "independent"}},
    )
    assert chart_dict["layer"][0]["mark"]["type"] == "bar", (
        "Expecting layer 0 to be a bar mark."
    )
    _assert_chart_data(
        chart_dict=chart_dict,
        dataset_key=chart_dict["layer"][0]["data"]["name"],
        expected_df=sample_data,
    )

    assert chart_dict["layer"][1]["mark"]["type"] == "line", (
        "Expecting layer 1 to be a line mark."
    )
    # Manual unpivot of the original summary data.
    expected_line_data_unpivoted = pl.concat(
        [
            sample_data.with_columns(
                pl.lit(col).alias("variable"),
                pl.col(col).alias("value"),
            ).select(["x_var", "variable", "value"])
            for col in right_y_axis_columns
        ],
        how="vertical",
    )
    _assert_chart_data(
        chart_dict=chart_dict,
        dataset_key=chart_dict["layer"][1]["data"]["name"],
        expected_df=expected_line_data_unpivoted,
    )

    assert chart_dict["layer"][2]["mark"]["type"] == "point", (
        "Expecting layer 2 to be a point mark."
    )
    _assert_chart_data(
        chart_dict=chart_dict,
        dataset_key=chart_dict["layer"][2]["data"]["name"],
        expected_df=expected_line_data_unpivoted,
    )

    expected_bar_layer = _construct_expected_bar_layer(
        colour=DEFAULT_ONE_WAY_PLOT_COLOURS.bar_colour,
        opacity=opacity,
        x_axis_name="x_var",
        y_axis_name="w_col",
        extra_tooltip_fields=right_y_axis_columns,
        expected_title=title,
        label_angle=0,
    )
    # Use the actual dataset name from the chart dict.
    expected_bar_layer["data"] = {"name": chart_dict["layer"][0]["data"]["name"]}

    assert DEFAULT_ONE_WAY_PLOT_COLOURS.line_colours is not None

    expected_line_layer = _construct_line_or_point_layer(
        type_="line",
        colours=list(DEFAULT_ONE_WAY_PLOT_COLOURS.line_colours),
        x_axis_name="x_var",
        y_axis_names=right_y_axis_columns,
        y_axis_range=(0.456, 0.504),  # range +-0.1 * (0.5 - 0.46)
        y_axis_label="Response Scale",
    )
    # Use the actual dataset name from the chart dict.
    expected_line_layer["data"] = {"name": chart_dict["layer"][1]["data"]["name"]}

    expected_points_layer = _construct_line_or_point_layer(
        type_="point",
        colours=list(DEFAULT_ONE_WAY_PLOT_COLOURS.line_colours),
        x_axis_name="x_var",
        y_axis_names=right_y_axis_columns,
        y_axis_range=(0.456, 0.504),  # range +-0.1 * (0.5 - 0.46)
        y_axis_label="Response Scale",
        encoding_color_legend_is_none=True,
    )
    # Use the actual dataset name from the chart dict.
    expected_points_layer["data"] = {"name": chart_dict["layer"][2]["data"]["name"]}

    assert len(chart_dict["layer"]) == 3
    assert chart_dict["layer"][0] == expected_bar_layer
    assert chart_dict["layer"][1] == expected_line_layer
    assert chart_dict["layer"][2] == expected_points_layer


def test_many_lines_can_be_plot_as_long_as_colours_specified(
    summary_table_with_extra_columns: SummaryTable,
):
    """Test that the function produces a bar plot and multiple line plots."""
    colours = OneWaySummaryColours(
        bar_colour="#000000",
        line_colours=(
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FFFF00",
            "#FF00FF",
            "#00FFFF",
            "#800000",
            "#008000",
            "#000080",
            "#808000",
        ),
    )

    right_y_axis_columns = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]

    chart = produce_one_way_summary_plot(
        summary_table_with_extra_columns,
        x_axis_column="x_var",
        left_y_axis_column="w_col",
        right_y_axis_columns=right_y_axis_columns,
        colours=colours,
    )
    assert chart is not None
    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()
    assert len(chart_dict["layer"]) == 3
