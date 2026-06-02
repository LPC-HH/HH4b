"""Unit tests for the generic histogram helpers migrated out of
``CombineTemplates.ipynb`` (``rename_sample_axis`` and ``combine_hists``)."""

from __future__ import annotations

import hist
import numpy as np
import pytest
from hist import Hist

from HH4b.utils import align_sample_axis, combine_hists, rename_sample_axis


def _mk(samples, base=0.0, nbins=4):
    """Build a (Sample x mass) weight-storage hist with deterministic content.

    Sample ``i`` gets values ``base + i + arange(nbins)`` and variances at half
    that, so every sample/bin is distinguishable and sums are easy to predict.
    """
    h = Hist(
        hist.axis.StrCategory(samples, name="Sample"),
        hist.axis.Regular(nbins, 0, nbins, name="H2PNetMass"),
        storage="weight",
    )
    view = h.view()
    for i in range(len(samples)):
        vals = base + i + np.arange(nbins, dtype=float)
        view["value"][i, :] = vals
        view["variance"][i, :] = vals * 0.5
    return h


# --------------------------------------------------------------------------- #
# rename_sample_axis
# --------------------------------------------------------------------------- #
def test_rename_sample_axis_appends_suffix():
    h = _mk(["hh4b", "ttbar"], base=10)
    r = rename_sample_axis(h, "_JMS_up")
    assert list(r.axes[0]) == ["hh4b_JMS_up", "ttbar_JMS_up"]


def test_rename_sample_axis_preserves_values_and_variances():
    h = _mk(["hh4b", "ttbar"], base=10)
    r = rename_sample_axis(h, "_JMS_up")
    np.testing.assert_array_equal(r["hh4b_JMS_up", :].values(), h["hh4b", :].values())
    np.testing.assert_array_equal(r["ttbar_JMS_up", :].values(), h["ttbar", :].values())
    np.testing.assert_array_equal(r["hh4b_JMS_up", :].variances(), h["hh4b", :].variances())


def test_rename_sample_axis_keeps_other_axes_and_flow():
    h = _mk(["hh4b"], base=3)
    r = rename_sample_axis(h, "_x")
    assert r.axes[1].name == h.axes[1].name
    assert r.axes[1].size == h.axes[1].size
    # flow bins carried through unchanged
    assert r.view(flow=True).shape == h.view(flow=True).shape


def test_rename_sample_axis_does_not_mutate_input():
    h = _mk(["hh4b", "ttbar"], base=10)
    rename_sample_axis(h, "_JMS_up")
    assert list(h.axes[0]) == ["hh4b", "ttbar"]


# --------------------------------------------------------------------------- #
# combine_hists
# --------------------------------------------------------------------------- #
def test_combine_hists_concatenates_sample_axis_in_order():
    h1 = _mk(["hh4b", "ttbar"], base=0)
    h2 = _mk(["hh4b_JMS_up", "ttbar_JMS_up"], base=100)
    c = combine_hists(h1, h2)
    assert list(c.axes[0]) == ["hh4b", "ttbar", "hh4b_JMS_up", "ttbar_JMS_up"]


def test_combine_hists_copies_values_and_variances():
    h1 = _mk(["hh4b", "ttbar"], base=0)
    h2 = _mk(["hh4b_JMS_up"], base=100)
    c = combine_hists(h1, h2)
    np.testing.assert_array_equal(c["hh4b", :].values(), h1["hh4b", :].values())
    np.testing.assert_array_equal(c["ttbar", :].variances(), h1["ttbar", :].variances())
    np.testing.assert_array_equal(c["hh4b_JMS_up", :].values(), h2["hh4b_JMS_up", :].values())
    np.testing.assert_array_equal(c["hh4b_JMS_up", :].variances(), h2["hh4b_JMS_up", :].variances())


def test_combine_hists_preserves_mass_flow_bins():
    h1 = _mk(["hh4b"], base=0)
    # stuff the mass-overflow bin of sample 0 (axis-0 index 1 is the StrCategory
    # "other" overflow, which combine_hists intentionally does not carry per-sample)
    h1.view(flow=True)["value"][0, -1] = 42.0
    c = combine_hists(h1)
    assert c.view(flow=True)["value"][0, -1] == 42.0


def test_combine_hists_single_input_roundtrips():
    h1 = _mk(["hh4b", "ttbar"], base=5)
    c = combine_hists(h1)
    assert list(c.axes[0]) == ["hh4b", "ttbar"]
    np.testing.assert_array_equal(c["ttbar", :].values(), h1["ttbar", :].values())


def test_combine_hists_rejects_duplicate_samples():
    h1 = _mk(["hh4b", "ttbar"], base=0)
    h2 = _mk(["ttbar", "qcd"], base=1)  # 'ttbar' duplicated across inputs
    with pytest.raises(ValueError, match="duplicate"):
        combine_hists(h1, h2)


def test_combine_hists_rejects_mismatched_non_sample_axes():
    h1 = _mk(["hh4b"], base=0, nbins=4)
    h2 = _mk(["ttbar"], base=0, nbins=8)  # different mass binning
    with pytest.raises(ValueError, match="non-Sample axes"):
        combine_hists(h1, h2)


def test_combine_hists_requires_at_least_one_hist():
    with pytest.raises(ValueError, match="at least one"):
        combine_hists()


# --------------------------------------------------------------------------- #
# align_sample_axis
# --------------------------------------------------------------------------- #
def test_align_sample_axis_reorders_to_requested_order():
    h = _mk(["hh4b", "ttbar", "qcd"], base=10)
    r = align_sample_axis(h, ["qcd", "hh4b", "ttbar"])
    assert list(r.axes[0]) == ["qcd", "hh4b", "ttbar"]


def test_align_sample_axis_preserves_per_sample_content():
    h = _mk(["hh4b", "ttbar", "qcd"], base=10)
    r = align_sample_axis(h, ["qcd", "hh4b", "ttbar"])
    for sample in ["hh4b", "ttbar", "qcd"]:
        np.testing.assert_array_equal(r[sample, :].values(), h[sample, :].values())
        np.testing.assert_array_equal(r[sample, :].variances(), h[sample, :].variances())


def test_align_sample_axis_makes_reordered_hists_summable():
    h1 = _mk(["hh4b", "ttbar", "qcd"], base=0)
    h2 = _mk(["qcd", "hh4b", "ttbar"], base=100)  # same set, different order
    aligned = align_sample_axis(h2, list(h1.axes[0]))
    total = h1 + aligned  # raises without alignment: Sample axes in different order
    np.testing.assert_array_equal(
        total["hh4b", :].values(), h1["hh4b", :].values() + h2["hh4b", :].values()
    )


def test_align_sample_axis_rejects_missing_label():
    h = _mk(["hh4b", "ttbar"], base=0)
    with pytest.raises((KeyError, ValueError)):
        align_sample_axis(h, ["hh4b", "ttbar", "qcd"])


def test_align_sample_axis_fill_missing_zero_fills_absent_samples():
    h = _mk(["hh4b", "ttbar"], base=5)
    r = align_sample_axis(h, ["hh4b", "ttbar", "qcd"], fill_missing=True)
    assert list(r.axes[0]) == ["hh4b", "ttbar", "qcd"]
    np.testing.assert_array_equal(r["hh4b", :].values(), h["hh4b", :].values())
    np.testing.assert_array_equal(r["qcd", :].values(), np.zeros(4))
    np.testing.assert_array_equal(r["qcd", :].variances(), np.zeros(4))
