from __future__ import annotations

import importlib.metadata

import hh4b as m


def test_version():
    assert importlib.metadata.version("hh4b") == m.__version__
