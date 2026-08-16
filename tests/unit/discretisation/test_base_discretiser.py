"""Unit tests for the generate_cut_expr function."""

import polars as pl
import pytest
from polars.testing import assert_series_equal

from sumnplot.discretisation.base_discretiser import (
    generate_bin_labels,
    generate_cut_expr,
)


class TestGenerateCutExpr:
    """Unit tests for the generate_cut_expr function."""

    @pytest.mark.parametrize(
        ("new_name", "expected_output_name"),
        [
            ("f0_group", "f0_group"),
            (None, "f0"),
        ],
    )
    def test_setting_new_name(self, new_name: str | None, expected_output_name: str):
        """Test new_name sets the output name for the expression."""
        column = "f0"
        breaks = [0, 18, 35, 50, 65, 100]

        expr = generate_cut_expr(
            column=column,
            breaks=breaks,
            labels=None,
            new_name=new_name,
        )
        assert expr.meta.output_name() == expected_output_name

    def test_predefined_labels(self):
        """Test that predefined labels are used in the expression."""
        column = "f0"
        breaks = [0, 10, 20]
        labels = ["a", "b", "c", "d"]

        df = pl.DataFrame({column: [-5, 5, 15, 25]})

        expr = generate_cut_expr(
            column=column,
            breaks=breaks,
            labels=labels,
            new_name=None,
        )

        df = df.with_columns(expr)
        cut_column = df.get_column(column)

        assert isinstance(cut_column.dtype, pl.Enum)
        assert cut_column.dtype.categories.to_list() == labels

        assert_series_equal(
            cut_column,
            pl.Series(column, labels, dtype=pl.Enum(categories=labels)),
        )


class TestGenerateBinLabels:
    """Unit tests for the generate_bin_labels function."""

    def test_left_closed_interval(self):
        """Test that generate_bin_labels with left closed intervals."""
        breaks = [0, 18, 35, 50, 65, 100]
        expected_labels = [
            "[-inf, 0)",
            "[0, 18)",
            "[18, 35)",
            "[35, 50)",
            "[50, 65)",
            "[65, 100)",
            "[100, inf)",
        ]
        assert generate_bin_labels(breaks, left_closed=True) == expected_labels

    def test_right_closed_interval(self):
        """Test that generate_bin_labels with right-closed intervals."""
        breaks = [0, 18, 35, 50, 65, 100]
        expected_labels = [
            "(-inf, 0]",
            "(0, 18]",
            "(18, 35]",
            "(35, 50]",
            "(50, 65]",
            "(65, 100]",
            "(100, inf]",
        ]
        assert generate_bin_labels(breaks, left_closed=False) == expected_labels
