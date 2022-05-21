"""Module for simple reusable checks."""

import pandas as pd

from typing import Any, Type, List


def check_type(obj: Any, expected_types: List[Type], obj_name: str) -> None:
    """Function to check object is of given types and raise a TypeError
    if not.
    """

    if type(expected_types) is not list:

        raise TypeError("expected_types must be a list")

    if not all([type(expected_type) is type for expected_type in expected_types]):

        raise TypeError("all elements in expected_types must be types")

    if type(obj) not in expected_types:

        raise TypeError(
            f"{obj_name} is not in expected types {expected_types}, got {type(obj)}"
        )


def check_condition(condition: bool, error_message_text: str):
    """Check that condition (which evaluates to a bool) is True and raise a
    ValueError if not.
    """

    check_type(condition, [bool], "condition")
    check_type(error_message_text, [str], "error_message_text")

    if not condition:

        raise ValueError(f"condition: [{error_message_text}] not met")


def check_columns_in_df(df: pd.DataFrame, columns: List) -> None:
    """Function to check that all specified columns are in a given DataFrame."""

    check_type(df, [pd.DataFrame], "df")
    check_type(columns, [list], "columns")

    missing_columns = [
        column_name for column_name in columns if column_name not in df.columns.values
    ]

    if len(missing_columns) > 0:

        raise ValueError(
            f"the following columns are missing from df; {missing_columns}"
        )
