from __future__ import annotations

import importlib.metadata

import HH4b as m


def test_version():
    assert importlib.metadata.version("HH4b") == m.__version__
