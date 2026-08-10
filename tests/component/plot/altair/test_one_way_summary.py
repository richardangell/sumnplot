"""Component tests for the one_way_summary module."""

from typing import Literal

import altair as alt
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sumnplot.plot.altair.one_way_summary import (
    DEFAULT_COLOURS,
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


def _assert_top_level_keys(
    chart_dict: dict,
    **chart_specific_expected_keys: dict,
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
    expected to be quantitative types.

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
    extra_tooltip_fields: list[str] | None = None,
) -> dict:
    """Construct the expected bar layer dictionary for testing."""
    tooltip_fields = (
        [*extra_tooltip_fields, y_axis_name] if extra_tooltip_fields else [y_axis_name]
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

    return {
        "mark": expected_bar_mark,
        "encoding": expected_bar_encoding,
    }


def _construct_line_layer(
    type_: Literal["line", "point"],
    colour: str,
    x_axis_name: str,
    y_axis_name: str,
    y_axis_label: str,
    y_axis_range: tuple[float, float],
    extra_tooltip_fields: list[str] | None = None,
) -> dict:
    """Construct the either point or line layer on the right y axis."""
    tooltip_fields = (
        [y_axis_name, *extra_tooltip_fields] if extra_tooltip_fields else [y_axis_name]
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

    return {
        "mark": expected_line_mark,
        "encoding": expected_line_encoding,
    }


def _construct_expected_line_layer(
    x_axis_name: str,
    left_y_axis_name: str,
    right_y_axis_names: list[str],
) -> dict:
    """Construct entire line layer.

    This includes a line and point layer for each right y axis column. The left y
    axis column is included in the tooltip for each line and point layer.

    """
    lines = []

    for right_y_axis_index, right_y_axis_name in enumerate(right_y_axis_names):
        line_colour = DEFAULT_COLOURS.line_colours[right_y_axis_index]

        line = _construct_line_layer(
            type_="line",
            colour=line_colour,
            x_axis_name=x_axis_name,
            y_axis_name=right_y_axis_name,
            y_axis_label="Response Scale",
            y_axis_range=(0.456, 0.504),
            extra_tooltip_fields=[left_y_axis_name],
        )

        line_points = _construct_line_layer(
            type_="point",
            colour=line_colour,
            x_axis_name=x_axis_name,
            y_axis_name=right_y_axis_name,
            y_axis_label="Response Scale",
            y_axis_range=(0.456, 0.504),
            extra_tooltip_fields=[left_y_axis_name],
        )

        lines.append(line)
        lines.append(line_points)

    return {"layer": lines}


def test_bars_plot_only(sample_data: pl.DataFrame):
    """Test that the function produces a bar plot when only."""
    opacity = 0.6
    height = 90
    width = 120

    chart = produce_one_way_summary_plot(
        sample_data,
        x_axis_column="x_var",
        left_y_axis_column="w_col",
        right_y_axis_columns=None,
        chart_width=width,
        chart_height=height,
        bar_opacity=opacity,
    )
    assert chart is not None
    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()

    _assert_top_level_keys(chart_dict, height=height, width=width)
    _assert_chart_data(chart_dict, expected_df=sample_data)

    expected_bar_layer = _construct_expected_bar_layer(
        colour=DEFAULT_COLOURS.bar_colour,
        opacity=opacity,
        x_axis_name="x_var",
        y_axis_name="w_col",
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
        chart_width=width,
        chart_height=height,
        bar_opacity=opacity,
    )
    assert chart is not None
    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()

    _assert_top_level_keys(
        chart_dict,
        height=height,
        width=width,
        resolve={"scale": {"y": "independent"}},
    )
    _assert_chart_data(chart_dict, expected_df=sample_data)

    expected_bar_layer = _construct_expected_bar_layer(
        colour=DEFAULT_COLOURS.bar_colour,
        opacity=opacity,
        x_axis_name="x_var",
        y_axis_name="w_col",
        extra_tooltip_fields=["f0"],
    )

    expected_line_layer = _construct_expected_line_layer(
        x_axis_name="x_var",
        left_y_axis_name="w_col",
        right_y_axis_names=["f0"],
    )

    assert len(chart_dict["layer"]) == 2
    assert chart_dict["layer"][0] == expected_bar_layer
    assert chart_dict["layer"][1] == expected_line_layer
