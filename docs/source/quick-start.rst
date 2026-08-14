Quick Start
====================

Welcome to the quick start guide for ``sumnplot``. 

``sumnplot`` provides functionality to produce summary plots like below. The code to produce this plot can be found at the bottom of this page.

   .. raw:: html
       :file: ../images/chart.html

Installation
--------------------

The easiest way to get ``sumnplot`` is to install directly from ``pypi``;

   .. code::

     pip install sumnplot

Discretisation
--------------------

``sumnplot.discretisation`` provides some simple ways to discretise numeric variables with either equal width or equal weight buckets.

Summarisation
--------------------

``sumnplot.summarisation`` provides a ``group_by_weighted_average`` function that works with ``polars.DataFrame`` objects. It is provided as a light wrapper around the ``polars.group_by`` functionality but ensures that all levels of the group by variables are present in the output.

Plotting
--------------------

``sumnplot.plot.altair`` provides a function to produce one-way summary plots using ``altair``.

The code to produce the plot at the top of this page is below;

   .. code::

     import polars as pl
     from sklearn.datasets import load_diabetes

     from sumnplot.discretisation import EqualWeightDiscretiser
     from sumnplot.summarisation import group_by_weighted_average, ResponseWeight
     from sumnplot.plot.altair.one_way_summary import produce_one_way_summary_plot

     X, _ = load_diabetes(return_X_y=True, as_frame=True)
     df = pl.DataFrame(X).with_columns(pl.lit(1).alias("w"))

     bp_discretiser = EqualWeightDiscretiser(
         column="bp",
         weights="w",
         min_weight_proportion=0.05,
         round_breaks_for_labels=True,
     )
     bp_discretiser.fit(df)
     df = bp_discretiser.transform(df)

     results = group_by_weighted_average(
         df,
         groupby_columns=["bp"],
         responses=[ResponseWeight("s1", "w"), ResponseWeight("s2", "w"), ResponseWeight("s3", "w")],
     )

     produce_one_way_summary_plot(
         results,
         x_axis_column="bp",
         left_y_axis_column="w",
         right_y_axis_columns=["s1", "s2", "s3"],
         title="One Way Summary Plot by 'bp'",
         x_axis_label_angle=90,
     )
