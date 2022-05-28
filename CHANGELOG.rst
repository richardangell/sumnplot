Changelog
=========

This changelog follows the great advice from https://keepachangelog.com/.

Each section will have a title of the format ``X.Y.Z (YYYY-MM-DD)`` giving the version of the package and the date of release of that version. Unreleased changes i.e. those that have been merged into master (e.g. with a .dev suffix) but which are not yet in a new release (on PyPI) are added to the changelog but with the title ``X.Y.Z (unreleased)``. Unreleased sections can be combined when they are released and the date of release added to the title.

Subsections for each version can be one of the following;

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.

Each individual change should have a link to the pull request after the description of the change.

0.3.0 (2022-05-28) Revamp and release package `#3 <https://github.com/richardangell/sumnplot/pull/3>`_
------------------------------------------------------------------------------------------------------

Added
^^^^^

- Add github actions pipelines
- Add ``.flake8``, ``mypy.ini`` and ``.pre-comit-config.yaml`` files for the project
- Add ``checks`` module
- Add new ``summary.ColumnSummariser`` class to summarise multiple columns
- Add new base class for discretisers; ``discretisation.Discretiser``
- Add new changelog in ``CHANGELOG.rst`` file

Changed
^^^^^^^

- Rename package to sumnplot
- Update documentation on readthedocs
- Update demo notebooks and rename ``examples`` folder to ``demo``
- Update project ``README.md``
- Change project to use ``flit`` as the package build tool
- Swap ``requirements.txt`` to ``pyproject.toml``
- Combine ``plot.one_way`` and ``plot.two_way`` modules into ``plot.matplotlib``
- Convert ``data_values_summary.data_values_summary`` function to ``summary.DataFrameValueCounter`` class
- Convert ``discretisation.discretise`` function into ``discretisation.EqualWidthDiscretiser``, ``discretisation.EqualWeightDiscretiser``, ``discretisation.QuantileDiscretiser`` classes

Removed
^^^^^^^

- Remove ``plot.helpers`` and ``plot.templates`` modules
- Remove ``plot.one_way.summary_plot`` and ``plot.two_way.summary_plot`` functions
- Remove ``tests`` folder
