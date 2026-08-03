"""Module for simple reusable checks."""

import abc
from typing import Any, List, Tuple, Type, Union

import pandas as pd


def check_type(
    obj: Any,
    expected_types: Union[Type, Tuple[Union[Type, Type[abc.ABCMeta]], ...]],
    obj_name: str,
    none_allowed: bool = False,
) -> None:
    """Check object is of given types and raise a TypeError if not.

    Parameters
    ----------
    obj : Any
        Any object to check the type of.

    expected_types : Union[Type, Tuple[Union[Type, Type[abc.ABCMeta]], ...]]
        Expected type or tuple of expected types of obj.

    none_allowed : bool = False
        Is None an allowed value for obj?

    """
    if type(expected_types) is tuple:
        if not all(
            type(expected_type) in [type, abc.ABCMeta]
            for expected_type in expected_types
        ):
            raise TypeError("all elements in expected_types must be types")

    else:
        if type(expected_types) not in [type, abc.ABCMeta]:
            raise TypeError("expected_types must be a type when passing a single type")

    if obj is None and not none_allowed:
        raise TypeError(f"{obj_name} is None and not is not allowed")

    if obj is not None and not isinstance(obj, expected_types):
        raise TypeError(
            f"{obj_name} is not in expected types {expected_types}, got {type(obj)}",
        )


def check_condition(condition: bool, error_message_text: str) -> None:
    """Check condition (which evaluates to a bool) is True.

    Raises a ValueError if the condition if not True.

    Parameters
    ----------
    condition : bool
        Condition that evaluates to bool, to check.

    error_message_text : str
        Message to print in ValueError if condition does not evalute to True.

    """
    check_type(condition, bool, "condition")
    check_type(error_message_text, str, "error_message_text")

    if not condition:
        raise ValueError(f"condition: [{error_message_text}] not met")


def check_columns_in_df(df: pd.DataFrame, columns: List) -> None:
    """Check that all specified columns are in a given DataFrame.

    Raises a ValueError if any columns missing from df.

    Parameters
    ----------
    df : pd.DataFrame
        Condition that evaluates to bool, to check.

    columns : List
        List of columns that must appear in df.

    """
    check_type(df, pd.DataFrame, "df")
    check_type(columns, list, "columns")

    if len(columns) == 0:
        raise ValueError("no columns specified in list")

    missing_columns = [
        column_name for column_name in columns if column_name not in df.columns.values
    ]

    if len(missing_columns) > 0:
        raise ValueError(
            f"the following columns are missing from df; {missing_columns}",
        )
