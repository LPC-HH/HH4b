"""Local copy of argparse helpers, to avoid the import chain through
``HH4b.run_utils`` → ``HH4b.processors`` → xgboost (which is not installed
in every conda env we want to run from, e.g. ``hbb-tagger``).
"""

from __future__ import annotations

import argparse


def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    help: str,
    default: bool = False,
    no_name: str | None = None,
) -> None:
    """``--name`` / ``--no-name`` mutually exclusive boolean flag."""
    varname = "_".join(name.split("-"))
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})
