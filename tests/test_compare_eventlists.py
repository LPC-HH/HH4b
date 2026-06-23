"""Tests for compare_eventlists.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import uproot

from HH4b.overlap.compare_eventlists import compare_trees, load_event_set


def _write_tree(path: Path, tree_name: str, run, lumi, event):
    """Write a simple ROOT tree with run/lumi/event arrays."""
    data = {
        "run": np.array(run, dtype=np.int64),
        "luminosityBlock": np.array(lumi, dtype=np.int64),
        "event": np.array(event, dtype=np.int64),
    }
    if path.exists():
        with uproot.update(path) as f:
            f[tree_name] = data
    else:
        with uproot.recreate(path) as f:
            f[tree_name] = data


class TestLoadEventSet:
    def test_loads_events(self, tmp_path):
        path = tmp_path / "test.root"
        _write_tree(path, "hh4b", run=[1, 1, 2], lumi=[10, 20, 30], event=[100, 200, 300])
        result = load_event_set(path, "hh4b")
        assert result == {(1, 10, 100), (1, 20, 200), (2, 30, 300)}

    def test_missing_file_returns_none(self, tmp_path):
        result = load_event_set(tmp_path / "missing.root", "hh4b")
        assert result is None

    def test_missing_tree_returns_none(self, tmp_path):
        path = tmp_path / "test.root"
        _write_tree(path, "hh4b", run=[1], lumi=[1], event=[1])
        result = load_event_set(path, "nonexistent_tree")
        assert result is None


class TestCompareTrees:
    def test_identical_events(self, tmp_path):
        curr = tmp_path / "curr.root"
        prev = tmp_path / "prev.root"
        _write_tree(curr, "data", run=[1, 2], lumi=[10, 20], event=[100, 200])
        _write_tree(prev, "data", run=[1, 2], lumi=[10, 20], event=[100, 200])

        result = compare_trees(curr, prev, "data", "2022")
        assert result["status"] == "ok"
        assert result["in_both"] == 2
        assert result["only_in_current"] == 0
        assert result["only_in_previous"] == 0

    def test_events_only_in_current(self, tmp_path):
        curr = tmp_path / "curr.root"
        prev = tmp_path / "prev.root"
        _write_tree(curr, "data", run=[1, 2, 3], lumi=[10, 20, 30], event=[100, 200, 300])
        _write_tree(prev, "data", run=[1, 2], lumi=[10, 20], event=[100, 200])

        result = compare_trees(curr, prev, "data", "2022")
        assert result["status"] == "diff"
        assert result["only_in_current"] == 1
        assert result["in_both"] == 2

    def test_events_only_in_previous(self, tmp_path):
        curr = tmp_path / "curr.root"
        prev = tmp_path / "prev.root"
        _write_tree(curr, "data", run=[1], lumi=[10], event=[100])
        _write_tree(prev, "data", run=[1, 2], lumi=[10, 20], event=[100, 200])

        result = compare_trees(curr, prev, "data", "2022")
        assert result["status"] == "diff"
        assert result["only_in_previous"] == 1

    def test_both_missing(self, tmp_path):
        result = compare_trees(tmp_path / "a.root", tmp_path / "b.root", "tree", "2022")
        assert result["status"] == "both_missing"

    def test_current_missing(self, tmp_path):
        prev = tmp_path / "prev.root"
        _write_tree(prev, "data", run=[1], lumi=[1], event=[1])
        result = compare_trees(tmp_path / "missing.root", prev, "data", "2022")
        assert result["status"] == "current_missing"

    def test_previous_missing(self, tmp_path):
        curr = tmp_path / "curr.root"
        _write_tree(curr, "data", run=[1], lumi=[1], event=[1])
        result = compare_trees(curr, tmp_path / "missing.root", "data", "2022")
        assert result["status"] == "previous_missing"

    def test_verbose_includes_samples(self, tmp_path):
        curr = tmp_path / "curr.root"
        prev = tmp_path / "prev.root"
        _write_tree(curr, "data", run=[1, 2], lumi=[10, 20], event=[100, 200])
        _write_tree(prev, "data", run=[1], lumi=[10], event=[100])

        result = compare_trees(curr, prev, "data", "2022", verbose=True)
        assert "sample_only_current" in result
        assert len(result["sample_only_current"]) > 0

    def test_result_has_year(self, tmp_path):
        curr = tmp_path / "curr.root"
        prev = tmp_path / "prev.root"
        _write_tree(curr, "data", run=[1], lumi=[1], event=[1])
        _write_tree(prev, "data", run=[1], lumi=[1], event=[1])
        result = compare_trees(curr, prev, "data", "2023BPix")
        assert result["year"] == "2023BPix"
        assert result["tree"] == "data"
