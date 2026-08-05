"""Unit tests for the generate_cut_expr function."""

import pytest

from sumnplot.discretisation.base_discretiser import generate_cut_expr


@pytest.mark.parametrize(
    ("new_name", "expected_output_name"),
    [
        ("f0_group", "f0_group"),
        (None, "f0"),
    ],
)
def test_generate_cut_expr_new_name(new_name: str | None, expected_output_name: str):
    """Test new_name argument can be used to set the output name for the expression."""
    column = "f0"
    breaks = [0, 18, 35, 50, 65, 100]

    expr = generate_cut_expr(
        column=column,
        breaks=breaks,
        labels=None,
        new_name=new_name,
    )
    assert expr.meta.output_name() == expected_output_name
