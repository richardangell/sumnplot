"""Component tests for the one_way_summary module."""

from typing import Any, Literal

import altair as alt
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sumnplot.plot.altair.one_way_summary import (
    DEFAULT_ONE_WAY_PLOT_COLOURS,
    ColourError,
    OneWaySummaryColours,
    produce_one_way_summary_plot,
)


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
def sample_data_with_extra_columns(sample_data: pl.DataFrame) -> pl.DataFrame:
    """Create a sample DataFrame with extra columns for testing."""
    return sample_data.with_columns(
        pl.col("f0").alias("f5"),
        pl.col("f1").alias("f6"),
        pl.col("f2").alias("f7"),
        pl.col("f3").alias("f8"),
        pl.col("f4").alias("f9"),
    )


def _assert_top_level_keys(
    chart_dict: dict,
    **chart_specific_expected_keys: dict[str, Any],
) -> None:
    """Assert that the top-level keys of the chart dictionary match the expected keys.

    Tests that the user supplied keys exist in the top level keys of the chart dict
    and take the supplied values. Also checks for the existence of the fixed keys
    that are always present in the chart dict; 'config', 'layer', 'data', '$schema',
    'datasets'.

    """
    fixed_keys = ["config", "layer", "data", "$schema", "datasets"]
    expected_keys = fixed_keys + list(chart_specific_expected_keys.keys())
    assert set(chart_dict.keys()) == set(expected_keys)

    for expected_key, expected_value in chart_specific_expected_keys.items():
        assert chart_dict[expected_key] == expected_value


def _assert_chart_data(chart_dict: dict, expected_df: pl.DataFrame) -> None:
    """Assert that the chart data matches the expected DataFrame."""
    assert chart_dict["data"].keys() == {"name"}

    actual_data_name = chart_dict["data"]["name"]

    assert chart_dict["datasets"].keys() == {actual_data_name}
    assert len(chart_dict["datasets"][actual_data_name]) == len(expected_df)

    assert_frame_equal(
        expected_df,
        pl.DataFrame(chart_dict["datasets"][actual_data_name]),
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


def _construct_line_layer(
    type_: Literal["line", "point"],
    colour: str,
    x_axis_name: str,
    y_axis_name: str,
    left_y_axis_name: str,
    y_axis_label: str,
    y_axis_range: tuple[float, float],
    expected_title: str,
    extra_tooltip_fields: list[str] | None = None,
) -> dict:
    """Construct either point or line layer for the right y axis."""
    tooltip_fields = (
        [*extra_tooltip_fields, left_y_axis_name]
        if extra_tooltip_fields
        else [y_axis_name]
    )
    expected_tooltip = _construct_expected_tooltip(
        x_axis_name=x_axis_name,
        y_axis_names=tooltip_fields,
    )

    expected_x_axis = {
        "axis": {"labelAngle": 0, "title": x_axis_name},
        "field": x_axis_name,
        "type": "ordinal",
    }
    expected_y_axis = {
        "axis": {"grid": True, "title": y_axis_label},
        "field": y_axis_name,
        "type": "quantitative",
        "scale": {"domain": list(y_axis_range)},
    }

    expected_line_encoding = {
        "tooltip": expected_tooltip,
        "x": expected_x_axis,
        "y": expected_y_axis,
    }

    expected_line_mark = {"type": type_, "color": colour}

    expected_title_ = {"anchor": "middle", "text": expected_title}

    return {
        "mark": expected_line_mark,
        "encoding": expected_line_encoding,
        "title": expected_title_,
    }


def _construct_expected_line_layer(
    x_axis_name: str,
    left_y_axis_name: str,
    right_y_axis_names: list[str],
    right_y_axis_range: tuple[float, float],
    right_y_axis_label: str,
    expected_title: str,
) -> dict:
    """Construct entire line layer.

    This includes a line and point layer for each right y axis column. The left y
    axis column is included in the tooltip for each line and point layer.

    """
    lines = []

    for right_y_axis_index, right_y_axis_name in enumerate(right_y_axis_names):
        assert DEFAULT_ONE_WAY_PLOT_COLOURS.line_colours is not None
        line_colour = DEFAULT_ONE_WAY_PLOT_COLOURS.line_colours[right_y_axis_index]

        line = _construct_line_layer(
            type_="line",
            colour=line_colour,
            x_axis_name=x_axis_name,
            y_axis_name=right_y_axis_name,
            left_y_axis_name=left_y_axis_name,
            y_axis_label=right_y_axis_label,
            y_axis_range=right_y_axis_range,
            expected_title=expected_title,
            extra_tooltip_fields=right_y_axis_names,
        )

        line_points = _construct_line_layer(
            type_="point",
            colour=line_colour,
            x_axis_name=x_axis_name,
            y_axis_name=right_y_axis_name,
            left_y_axis_name=left_y_axis_name,
            y_axis_label=right_y_axis_label,
            y_axis_range=right_y_axis_range,
            expected_title=expected_title,
            extra_tooltip_fields=right_y_axis_names,
        )

        lines.append(line)
        lines.append(line_points)

    return {"layer": lines}


def test_too_many_columns_for_line_colours_raises_exception(sample_data: pl.DataFrame):
    """Test exception raised when more right y axis columns than line colours."""
    colours = OneWaySummaryColours(
        bar_colour="#000000",
        line_colours=("#FF0000", "#00FF00"),  # Only 2 line colours specified
    )

    right_y_axis_columns = ["f0", "f1", "f2"]  # 3 right y axis columns

    expected_message = (
        "Not enough line colours specified for 3 lines. Only 2 line colours specified."
    )

    with pytest.raises(ColourError, match=expected_message):
        produce_one_way_summary_plot(
            sample_data,
            x_axis_column="x_var",
            left_y_axis_column="w_col",
            right_y_axis_columns=right_y_axis_columns,
            colours=colours,
        )


def test_bars_plot_only(sample_data: pl.DataFrame):
    """Test that the function produces a bar plot when only."""
    opacity = 0.6
    height = 90
    width = 120
    title = "Bars Only Summary Plot"
    label_angle = 90

    chart = produce_one_way_summary_plot(
        sample_data,
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
    _assert_chart_data(chart_dict, expected_df=sample_data)

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


def test_bar_and_single_line_plot(sample_data: pl.DataFrame):
    """Test that the function produces a bar plot and a single line plot."""
    opacity = 0.9
    height = 100
    width = 140

    chart = produce_one_way_summary_plot(
        sample_data,
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
        height=height,  # type: ignore[reportArgumentType]
        width=width,  # type: ignore[reportArgumentType]
        resolve={"scale": {"y": "independent"}},
    )
    _assert_chart_data(chart_dict, expected_df=sample_data)

    expected_bar_layer = _construct_expected_bar_layer(
        colour=DEFAULT_ONE_WAY_PLOT_COLOURS.bar_colour,
        opacity=opacity,
        x_axis_name="x_var",
        y_axis_name="w_col",
        extra_tooltip_fields=["f0"],
        expected_title="x_var",
        label_angle=0,
    )

    expected_line_layer = _construct_expected_line_layer(
        x_axis_name="x_var",
        left_y_axis_name="w_col",
        right_y_axis_names=["f0"],
        right_y_axis_range=(0.456, 0.504),  # range +-0.1 * (0.5 - 0.46)
        right_y_axis_label="Response Scale",
        expected_title="x_var",
    )

    assert len(chart_dict["layer"]) == 2
    assert chart_dict["layer"][0] == expected_bar_layer
    assert chart_dict["layer"][1] == expected_line_layer


def test_bar_and_multiple_line_plot(sample_data: pl.DataFrame):
    """Test that the function produces a bar plot and multiple line plots."""
    opacity = 0.3
    height = 200
    width = 240
    title = "Bar and Multiple Lines Summary Plot"

    chart = produce_one_way_summary_plot(
        sample_data,
        x_axis_column="x_var",
        left_y_axis_column="w_col",
        right_y_axis_columns=["f0", "f1", "f2", "f3"],
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
        height=height,  # type: ignore[reportArgumentType]
        width=width,  # type: ignore[reportArgumentType]
        resolve={"scale": {"y": "independent"}},
    )
    _assert_chart_data(chart_dict, expected_df=sample_data)

    expected_bar_layer = _construct_expected_bar_layer(
        colour=DEFAULT_ONE_WAY_PLOT_COLOURS.bar_colour,
        opacity=opacity,
        x_axis_name="x_var",
        y_axis_name="w_col",
        extra_tooltip_fields=["f0", "f1", "f2", "f3"],
        expected_title=title,
        label_angle=0,
    )

    expected_line_layer = _construct_expected_line_layer(
        x_axis_name="x_var",
        left_y_axis_name="w_col",
        right_y_axis_names=["f0", "f1", "f2", "f3"],
        right_y_axis_range=(0.456, 0.504),  # range +-0.1 * (0.5 - 0.46)
        right_y_axis_label="Response Scale",
        expected_title=title,
    )

    assert len(chart_dict["layer"]) == 2
    assert chart_dict["layer"][0] == expected_bar_layer
    assert chart_dict["layer"][1] == expected_line_layer


def test_many_lines_can_be_plot_as_long_as_colours_specified(
    sample_data_with_extra_columns: pl.DataFrame,
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
        sample_data_with_extra_columns,
        x_axis_column="x_var",
        left_y_axis_column="w_col",
        right_y_axis_columns=right_y_axis_columns,
        colours=colours,
    )
    assert chart is not None
    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()

    assert len(chart_dict["layer"]) == 2
    assert chart_dict["layer"][1].keys() == {"layer"}
    assert len(chart_dict["layer"][1]["layer"]) == 20  # 10 lines and 10 points
