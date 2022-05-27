Quick Start
====================

Welcome to the quick start guide for `sumnplot``.


Installation
--------------------

The easiest way to get ``sumnplot`` is to install directly from ``pip``;

   .. code::

     pip install sumnplot

Methods Summary
--------------------

``pitci`` allows the user to generate intervals about predictions when using tree based models. 
Conformal intervals are the underlying technqiue that makes this possible. Here we use
*inductive* conformal intervals learn an expected interval width at a given confidence level 
(``alpha``) from a calibration dataset and then this interval is applied to new examples when 
making predictions.
