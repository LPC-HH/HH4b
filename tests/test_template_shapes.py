"""Unit tests for the shape-systematic infrastructure migrated out of
``CombineTemplates.ipynb``.

Covers the cross-year template combination + weight-shift/JEC-JMSR dispatch
(``combine_templates``), the systematic enumeration (``get_shape_systematics``),
the smorph JMS/JMR variation helper (``compute_jmsr_variations``), and the
plotting driver dispatch (``make_shape_plots``).
"""

from __future__ import annotations

import importlib.util

import hist
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hist import Hist

from HH4b import plotting
from HH4b.hh_vars import jmsr_values
from HH4b.postprocessing import (
    Region,
    combine_templates,
    compute_jmsr_variations,
    get_shape_systematics,
    get_weight_shifts,
)
from HH4b.postprocessing.PlotShapes import (
    _year_label,
    default_shape_plot_dir,
    make_shape_plots,
)
from HH4b.utils import Syst, align_sample_axis


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


# compute_jmsr_variations -> smorph -> rhalphalib, which is not installed in the CI extra
requires_rhalphalib = pytest.mark.skipif(
    not _module_available("rhalphalib"), reason="rhalphalib not installed"
)

YEARS = ["2022", "2022EE"]
BASES = {"2022": 100.0, "2022EE": 200.0}
REGION = "pass_bin1"


def _mk(samples, base=0.0, nbins=4):
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


@pytest.fixture
def templates():
    """Synthetic per-year templates mimicking the real pickle structure:

    - nominal region hist carries weight-shift variations as extra Sample
      categories ``{sample}_trigger_{up,down}``;
    - JEC/JMSR shifts live in separate ``{region}_{shift}_{up,down}`` hists.
    """
    t = {}
    for year in YEARS:
        b = BASES[year]
        t[year] = {
            REGION: _mk(["hh4b", "ttbar", "hh4b_trigger_up", "hh4b_trigger_down"], base=b),
            f"{REGION}_JMS_up": _mk(["hh4b", "ttbar"], base=b + 10),
            f"{REGION}_JMS_down": _mk(["hh4b", "ttbar"], base=b + 20),
            f"{REGION}_JER_up": _mk(["hh4b", "ttbar"], base=b + 30),
            f"{REGION}_JER_down": _mk(["hh4b", "ttbar"], base=b + 40),
            f"{REGION}_JES_up": _mk(["hh4b", "ttbar"], base=b + 50),
            f"{REGION}_JES_down": _mk(["hh4b", "ttbar"], base=b + 60),
            f"{REGION}_JMR_up": _mk(["hh4b", "ttbar"], base=b + 70),
            f"{REGION}_JMR_down": _mk(["hh4b", "ttbar"], base=b + 80),
        }
    return t


# --------------------------------------------------------------------------- #
# combine_templates
# --------------------------------------------------------------------------- #
def test_combine_templates_nominal_sums_over_years(templates):
    h = combine_templates(templates, YEARS, REGION)
    np.testing.assert_array_equal(h["hh4b", :].values(), [300.0, 302.0, 304.0, 306.0])
    np.testing.assert_array_equal(h["ttbar", :].values(), [302.0, 304.0, 306.0, 308.0])


def test_combine_templates_nominal_keeps_weight_shift_samples(templates):
    h = combine_templates(templates, YEARS, REGION)
    samples = list(h.axes[0])
    assert "hh4b_trigger_up" in samples
    assert "hh4b_trigger_down" in samples


def test_combine_templates_weight_shift_returns_summed_nominal(templates):
    # For weight-based systematics the up/down are baked into the nominal hist,
    # so the dispatch must NOT try to read separate {region}_trigger_{dir} hists.
    h = combine_templates(templates, YEARS, REGION, shift="trigger")
    np.testing.assert_array_equal(h["hh4b_trigger_up", :].values(), [304.0, 306.0, 308.0, 310.0])
    np.testing.assert_array_equal(h["hh4b", :].values(), [300.0, 302.0, 304.0, 306.0])


def test_combine_templates_jmsr_concatenates_shifted_samples(templates):
    h = combine_templates(templates, YEARS, REGION, shift="JMS")
    samples = list(h.axes[0])
    # nominal samples retained ...
    assert "hh4b" in samples
    assert "ttbar" in samples
    # ... plus the renamed JMS up/down categories sigErrRatioPlot looks for
    assert "hh4b_JMS_up" in samples
    assert "hh4b_JMS_down" in samples


def test_combine_templates_jmsr_values_are_cross_year_sums(templates):
    h = combine_templates(templates, YEARS, REGION, shift="JMS")
    np.testing.assert_array_equal(h["hh4b_JMS_up", :].values(), [320.0, 322.0, 324.0, 326.0])
    np.testing.assert_array_equal(h["hh4b_JMS_down", :].values(), [340.0, 342.0, 344.0, 346.0])
    # nominal untouched
    np.testing.assert_array_equal(h["hh4b", :].values(), [300.0, 302.0, 304.0, 306.0])


def test_combine_templates_jec_concatenates_shifted_samples(templates):
    h = combine_templates(templates, YEARS, REGION, shift="JER")
    samples = list(h.axes[0])
    assert "hh4b_JER_up" in samples
    assert "hh4b_JER_down" in samples
    np.testing.assert_array_equal(h["hh4b_JER_up", :].values(), [360.0, 362.0, 364.0, 366.0])


def test_combine_templates_handles_reordered_era(templates):
    # mimic a reprocessed era whose Sample axis has the same set in a different
    # order (real case: 2023BPix in the split2024MC templates)
    h = templates["2022EE"][REGION]
    reordered = align_sample_axis(h, list(h.axes[0])[::-1])
    templates["2022EE"][REGION] = reordered

    out = combine_templates(templates, YEARS, REGION)
    # summed values must be unchanged by the reordering
    np.testing.assert_array_equal(out["hh4b", :].values(), [300.0, 302.0, 304.0, 306.0])
    np.testing.assert_array_equal(out["ttbar", :].values(), [302.0, 304.0, 306.0, 308.0])


def test_combine_templates_unions_partial_sample_sets(templates):
    # mimic 2024/2025 missing 'qcd' relative to 2022-2023: one era lacks 'ttbar'
    templates["2022EE"][REGION] = _mk(["hh4b", "hh4b_trigger_up", "hh4b_trigger_down"], base=200)
    out = combine_templates(templates, YEARS, REGION)
    samples = list(out.axes[0])
    # the sample present in only one era is retained via the union ...
    assert "ttbar" in samples
    # ... summing only over the era(s) that have it (2022: base 100, idx 1)
    np.testing.assert_array_equal(out["ttbar", :].values(), [101.0, 102.0, 103.0, 104.0])
    # hh4b is present in both eras, so it sums over both
    np.testing.assert_array_equal(out["hh4b", :].values(), [300.0, 302.0, 304.0, 306.0])


def test_combine_templates_preserves_variances(templates):
    h = combine_templates(templates, YEARS, REGION, shift="JMS")
    # year-summed variances add: 0.5*(b+ ...) per year
    expected = 0.5 * np.array([110.0, 111.0, 112.0, 113.0]) + 0.5 * np.array(
        [210.0, 211.0, 212.0, 213.0]
    )
    np.testing.assert_allclose(h["hh4b_JMS_up", :].variances(), expected)


# --------------------------------------------------------------------------- #
# get_shape_systematics
# --------------------------------------------------------------------------- #
def test_get_shape_systematics_includes_signal_weight_shifts():
    ws = get_weight_shifts("glopart-v2", "25Feb5_v13_glopartv2_rawmass")
    shifts = get_shape_systematics(ws, "hh4b")
    assert "trigger" in shifts  # applies to signal
    assert "scale" in shifts  # applies to signal
    # ttbar-only shifts excluded for a signal sample
    assert "ttbarSF_pTjj" not in shifts


def test_get_shape_systematics_appends_jec_and_jmsr():
    ws = get_weight_shifts("glopart-v2", "25Feb5_v13_glopartv2_rawmass")
    shifts = get_shape_systematics(ws, "hh4b")
    for s in ("JES", "JER", "JMS", "JMR"):
        assert s in shifts


def test_get_shape_systematics_respects_sample_membership():
    ws = get_weight_shifts("glopart-v2", "25Feb5_v13_glopartv2_rawmass")
    tshifts = get_shape_systematics(ws, "ttbar")
    assert "ttbarSF_pTjj" in tshifts


def test_get_shape_systematics_can_disable_jec_jmsr():
    ws = get_weight_shifts("glopart-v2", "25Feb5_v13_glopartv2_rawmass")
    shifts = get_shape_systematics(ws, "hh4b", include_jec=False, include_jmsr=False)
    assert not any(s in shifts for s in ("JES", "JER", "JMS", "JMR"))


# --------------------------------------------------------------------------- #
# compute_jmsr_variations (smorph fold of notebook cells 11-14)
# --------------------------------------------------------------------------- #
@pytest.fixture
def signal_mass_hist():
    h = Hist(
        hist.axis.StrCategory(["hh4b"], name="Sample"),
        hist.axis.Regular(16, 60, 220, name="bbFatJetParTmassVis"),
        storage="weight",
    )
    rng = np.random.default_rng(0)
    h.fill(Sample="hh4b", bbFatJetParTmassVis=rng.normal(125, 15, 50000))
    return h["hh4b", :]


def _mean(h1d):
    centers = h1d.axes[0].centers
    vals = h1d.values()
    return np.sum(centers * vals) / np.sum(vals)


@requires_rhalphalib
def test_compute_jmsr_variations_returns_five_named_variations(signal_mass_hist):
    out = compute_jmsr_variations(signal_mass_hist, "hh4b", "2022")
    assert set(out) == {"nominal", "jms_up", "jms_down", "jmr_up", "jmr_down"}


@requires_rhalphalib
def test_compute_jmsr_variations_keep_binning_and_are_finite(signal_mass_hist):
    out = compute_jmsr_variations(signal_mass_hist, "hh4b", "2022")
    for v in out.values():
        assert v.axes[0].size == signal_mass_hist.axes[0].size
        assert np.all(np.isfinite(v.values()))


@requires_rhalphalib
def test_compute_jmsr_variations_jms_shifts_mean_monotonically(signal_mass_hist):
    out = compute_jmsr_variations(signal_mass_hist, "hh4b", "2022")
    # 2022 JMS: down=1.007 < up=1.014, so the up-scaled mean must exceed the down one
    assert _mean(out["jms_up"]) > _mean(out["jms_down"])


@requires_rhalphalib
def test_compute_jmsr_variations_conserves_yield(signal_mass_hist):
    out = compute_jmsr_variations(signal_mass_hist, "hh4b", "2022")
    nominal_integral = np.sum(signal_mass_hist.values())
    for v in out.values():
        np.testing.assert_allclose(np.sum(v.values()), nominal_integral, rtol=0.02)


@requires_rhalphalib
def test_jms_jmr_comparison_plot_writes_pdf(signal_mass_hist, tmp_path):
    plt.switch_backend("Agg")
    variations = compute_jmsr_variations(signal_mass_hist, "hh4b", "2022")
    jms = jmsr_values["bbFatJetParTmassVis"]["JMS"]["2022"]
    jmr = jmsr_values["bbFatJetParTmassVis"]["JMR"]["2022"]

    plotting.jmsJmrComparisonPlot(
        variations, jms, jmr, original=signal_mass_hist, plot_dir=tmp_path, name="cmp"
    )
    assert (tmp_path / "cmp.pdf").exists()


# --------------------------------------------------------------------------- #
# make_shape_plots (driver dispatch)
# --------------------------------------------------------------------------- #
def test_make_shape_plots_calls_plotter_per_region_and_shift(templates, tmp_path, monkeypatch):
    calls = []

    def _spy(*args, **_kwargs):
        h, _, wshift, _, title, _, name = args[:7]
        calls.append(
            {
                "samples": list(h.axes[0]),
                "wshift": wshift,
                "title": title,
                "name": name,
            }
        )

    monkeypatch.setattr(plotting, "sigErrRatioPlot", _spy)

    weight_shifts = {"trigger": Syst(samples=["hh4b"], label="Trigger")}
    selection_regions = {REGION: Region(cuts={"Category": [1, 2]}, label="Bin1")}

    make_shape_plots(
        templates,
        YEARS,
        selection_regions,
        "hh4b",
        weight_shifts,
        plot_dir=tmp_path,
        xlabel="$m_{reg}$ (GeV)",
        shifts=["trigger", "JMS"],
    )

    assert len(calls) == 2
    by_shift = {c["wshift"]: c for c in calls}

    # weight shift: baked-in up/down samples present, label drawn from Syst
    assert "hh4b_trigger_up" in by_shift["trigger"]["samples"]
    assert "Trigger" in by_shift["trigger"]["title"]
    assert by_shift["trigger"]["name"] == f"{REGION}_sig_trigger"

    # JMSR shift: renamed up/down samples present, raw shift name used as label
    assert "hh4b_JMS_up" in by_shift["JMS"]["samples"]
    assert "JMS" in by_shift["JMS"]["title"]
    assert by_shift["JMS"]["name"] == f"{REGION}_sig_JMS"


def test_make_shape_plots_skips_jec_jmsr_shift_absent_from_templates(
    templates, tmp_path, monkeypatch
):
    # real template sets do not always carry every JEC/JMSR shift (e.g. no JES)
    for year in YEARS:
        del templates[year][f"{REGION}_JES_up"]
        del templates[year][f"{REGION}_JES_down"]

    seen = []
    monkeypatch.setattr(plotting, "sigErrRatioPlot", lambda *a, **_: seen.append(a[2]))
    ws = {"trigger": Syst(samples=["hh4b"], label="Trigger")}
    sel = {REGION: Region(cuts={"Category": [1, 2]}, label="Bin1")}

    with pytest.warns(UserWarning, match="JES"):
        make_shape_plots(
            templates, YEARS, sel, "hh4b", ws, plot_dir=tmp_path, xlabel="m", shifts=["JES", "JMS"]
        )
    assert seen == ["JMS"]


def test_make_shape_plots_skips_weight_shift_without_baked_samples(
    templates, tmp_path, monkeypatch
):
    seen = []
    monkeypatch.setattr(plotting, "sigErrRatioPlot", lambda *a, **_: seen.append(a[2]))
    # 'pdf' is not baked into the synthetic nominal hist
    ws = {
        "pdf": Syst(samples=["hh4b"], label="PDF"),
        "trigger": Syst(samples=["hh4b"], label="Trigger"),
    }
    sel = {REGION: Region(cuts={"Category": [1, 2]}, label="Bin1")}

    with pytest.warns(UserWarning, match="pdf"):
        make_shape_plots(
            templates,
            YEARS,
            sel,
            "hh4b",
            ws,
            plot_dir=tmp_path,
            xlabel="m",
            shifts=["pdf", "trigger"],
        )
    assert seen == ["trigger"]


# --------------------------------------------------------------------------- #
# output-dir derivation
# --------------------------------------------------------------------------- #
def test_year_label_spans_min_to_max():
    assert _year_label(["2022", "2022EE", "2023", "2023BPix"]) == "2022-2023"
    assert _year_label(["2022", "2022EE", "2023", "2023BPix", "2024", "2025"]) == "2022-2025"


def test_year_label_single_year():
    assert _year_label(["2024"]) == "2024"


def test_default_shape_plot_dir_derives_from_tag_and_repo_root(tmp_path):
    # fake a standard checkout layout: <root>/src/HH4b + the templates dir
    root = tmp_path / "HH4b"
    (root / "src" / "HH4b").mkdir(parents=True)
    tdir = root / "src" / "HH4b" / "postprocessing" / "templates" / "26Mar10_split2024MC"
    tdir.mkdir(parents=True)

    out = default_shape_plot_dir(tdir, ["2022", "2022EE", "2023", "2023BPix", "2024", "2025"])
    expected = root / "plots/PostProcess/26Mar10_split2024MC/Templates/2022-2025/wshifts"
    assert out == expected


def test_make_shape_plots_defaults_to_all_applicable_shifts(templates, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        plotting, "sigErrRatioPlot", lambda *a, **_: seen.append(a[2])  # wshift positional arg
    )
    ws = {"trigger": Syst(samples=["hh4b"], label="Trigger")}
    selection_regions = {REGION: Region(cuts={"Category": [1, 2]}, label="Bin1")}

    make_shape_plots(
        templates,
        YEARS,
        selection_regions,
        "hh4b",
        ws,
        plot_dir=tmp_path,
        xlabel="m",
    )
    # default enumerates the trigger weight shift plus JEC/JMSR
    assert "trigger" in seen
    assert "JMS" in seen
    assert "JER" in seen
