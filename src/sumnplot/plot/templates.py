import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pad.plot as p


def lift_curve(
    df,
    weights,
    observed,
    fitted,
    **kwargs,
):
    """Function to plot a "lift curve", which is average observed and fitted ordered by fitted values.

    Parameters
    ----------
    df : pd.DataFrame
        Data of interest. Must contain a columns with names supplied in weights, observed and fitted.

    weights : str
        Column name of weights in df.

    observed : str
        Column name of observed values in df.

    fitted : str
        Column name of fitted (predicted) values in df.

    **kwargs :
        Other arguments passed on to pyandev.plot.one_way.summary_plot().

    """

    if not isinstance(df, pd.DataFrame):

        raise TypeError("df should be a pd.DataFrame")

    if not isinstance(weights, str):

        raise TypeError("weights should be a str")

    if not weights in df.columns.values:

        raise ValueError("weights; " + weights + " not in df")

    if not isinstance(observed, str):

        raise TypeError("observed should be a str")

    if not observed in df.columns.values:

        raise ValueError("observed; " + observed + " not in df")

    if not isinstance(fitted, str):

        raise TypeError("fitted should be a str")

    if not fitted in df.columns.values:

        raise ValueError("fitted; " + fitted + " not in df")

    p.one_way.summary_plot(
        df=df,
        weights=weights,
        by_col=fitted,
        observed=observed,
        fitted=fitted,
        **kwargs,
    )


def lift_curve_model_ratio(
    df,
    weights,
    observed,
    fitted,
    fitted2,
    models_fitted_ratio=None,
    **kwargs,
):
    """Function to plot a "model ratio lift curve" which is; average observed and fitted from 2 different models, ordered
    by ratio of the model fitted values.

    Parameters
    ----------
    df : pd.DataFrame
        Data of interest. Must contain a columns with names supplied in weights, observed and fitted.

    weights : str
        Column name of weights in df.

    observed : str
        Column name of observed values in df.

    fitted : str
        Column name of fitted (predicted) values in df.

    fitted2 : str
        Column name of second fitted (predicted) values in df.

    fitted2 : str
        Column name of second fitted (predicted) values in df.

    models_fitted_ratio : str, default = None
        Column name of ratio of fitted and fitted2 columns. If None this ratio column
        is calculated.

    **kwargs :
        Other arguments passed on to pyandev.plot.one_way.summary_plot().

    """

    if not isinstance(df, pd.DataFrame):

        raise TypeError("df should be a pd.DataFrame")

    if not isinstance(weights, str):

        raise TypeError("weights should be a str")

    if not weights in df.columns.values:

        raise ValueError("weights; " + weights + " not in df")

    if not isinstance(observed, str):

        raise TypeError("observed should be a str")

    if not observed in df.columns.values:

        raise ValueError("observed; " + observed + " not in df")

    if not isinstance(fitted, str):

        raise TypeError("fitted should be a str")

    if not fitted in df.columns.values:

        raise ValueError("fitted; " + fitted + " not in df")

    if not isinstance(fitted2, str):

        raise TypeError("fitted2 should be a str")

    if not fitted2 in df.columns.values:

        raise ValueError("fitted2; " + fitted2 + " not in df")

    if not models_fitted_ratio is None:

        if not isinstance(models_fitted_ratio, str):

            raise TypeError("models_fitted_ratio should be a str")

        if not models_fitted_ratio in df.columns.values:

            raise ValueError(
                "models_fitted_ratio; " + models_fitted_ratio + " not in df"
            )

    if not models_fitted_ratio is None:

        p.one_way.summary_plot(
            df=df,
            weights=weights,
            by_col=models_fitted_ratio,
            observed=observed,
            fitted=fitted,
            fitted2=fitted2,
            **kwargs,
        )

    else:

        df = df[[weights, observed, fitted, fitted2]].copy()

        ratio_col = fitted + " over " + fitted2

        df[ratio_col] = df[fitted] / df[fitted2]

        p.one_way.summary_plot(
            df=df,
            weights=weights,
            by_col=ratio_col,
            observed=observed,
            fitted=fitted,
            fitted2=fitted2,
            **kwargs,
        )
