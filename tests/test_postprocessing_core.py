from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hist")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from HH4b.hh_vars import (
    LUMI,
    ttbarsfs_decorr_ggfbdt_bins,
    ttbarsfs_decorr_txbb_bins,
    txbbsfs_decorr_pt_bins,
    txbbsfs_decorr_txbb_wps,
)
from HH4b.postprocessing import postprocessing as pp
from HH4b.postprocessing.PostProcess import (
    add_bdt_scores,
    fom_classic,
    fom_update,
    get_jets_for_txbb_sf,
    get_nevents_data,
    get_nevents_nosignal,
    get_nevents_signal,
)
from HH4b.utils import ShapeVar


# ---------------------------------------------------------------------------
# add_bdt_scores
# ---------------------------------------------------------------------------


def _make_events():
    return pd.DataFrame(index=range(3))


def test_add_bdt_scores_binary():
    events = _make_events()
    preds = np.array([[0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    add_bdt_scores(events, preds)
    np.testing.assert_array_almost_equal(events["bdt_score"].to_numpy(), preds[:, 1])
    assert "bdt_score_vbf" not in events.columns


def test_add_bdt_scores_3class():
    events = _make_events()
    preds = np.array([[0.8, 0.1, 0.1], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2]])
    add_bdt_scores(events, preds)
    np.testing.assert_array_almost_equal(events["bdt_score"].to_numpy(), preds[:, 0])
    assert "bdt_score_vbf" not in events.columns


def test_add_bdt_scores_4class_discriminant():
    """4-class: ggF disc = p0/(p0+p2+p3), VBF disc = p1/(p1+p2+w*p3)."""
    events = _make_events()
    preds = np.array(
        [[0.5, 0.2, 0.2, 0.1], [0.1, 0.7, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]],
        dtype=float,
    )
    w_ttbar = 2.0
    add_bdt_scores(events, preds, weight_ttbar=w_ttbar, bdt_disc=True)

    expected_ggf = preds[:, 0] / (preds[:, 0] + preds[:, 2] + preds[:, 3])
    expected_vbf = preds[:, 1] / (preds[:, 1] + preds[:, 2] + w_ttbar * preds[:, 3])
    np.testing.assert_array_almost_equal(events["bdt_score"].to_numpy(), expected_ggf)
    np.testing.assert_array_almost_equal(events["bdt_score_vbf"].to_numpy(), expected_vbf)


def test_add_bdt_scores_4class_raw():
    """4-class with bdt_disc=False returns raw class probabilities."""
    # Use a single-row DataFrame to match the single prediction row
    events = pd.DataFrame(index=range(1))
    preds = np.array([[0.5, 0.2, 0.2, 0.1]], dtype=float)
    add_bdt_scores(events, preds, bdt_disc=False)
    assert events["bdt_score"].iloc[0] == pytest.approx(0.5)
    assert events["bdt_score_vbf"].iloc[0] == pytest.approx(0.2)


def test_add_bdt_scores_5class_discriminant():
    """5-class: ggF disc = p0/(p0+p3+p4), VBF disc = (p1+p2)/(p1+p2+p3+w*p4)."""
    # Use a single-row DataFrame to match the single prediction row
    events = pd.DataFrame(index=range(1))
    preds = np.array([[0.4, 0.15, 0.15, 0.2, 0.1]], dtype=float)
    w_ttbar = 1.5
    add_bdt_scores(events, preds, weight_ttbar=w_ttbar, bdt_disc=True)

    bg_tot = preds[0, 3] + preds[0, 4]
    expected_ggf = preds[0, 0] / (preds[0, 0] + bg_tot)
    expected_vbf = (preds[0, 1] + preds[0, 2]) / (
        preds[0, 1] + preds[0, 2] + preds[0, 3] + w_ttbar * preds[0, 4]
    )
    assert events["bdt_score"].iloc[0] == pytest.approx(expected_ggf)
    assert events["bdt_score_vbf"].iloc[0] == pytest.approx(expected_vbf)


def test_add_bdt_scores_jshift_suffix():
    """A jshift label is appended to the score column names."""
    events = _make_events()
    preds = np.array([[0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    add_bdt_scores(events, preds, jshift="JES_up")
    assert "bdt_score_JES_up" in events.columns
    assert "bdt_score" not in events.columns


# ---------------------------------------------------------------------------
# get_jets_for_txbb_sf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("hh4b", [1, 2]),
        ("vbfhh4b", [1, 2]),
        ("hh4b-kl0", [1, 2]),
        ("vhtobb", [1, 2]),
        ("zz", [1, 2]),
        ("novhhtobb", [1]),
        ("tthtobb", [1]),
        ("vjets", [1]),
        ("nozzdiboson", [1]),
        ("data", []),
        ("qcd", []),
        ("ttbar", []),
    ],
)
def test_get_jets_for_txbb_sf(key, expected):
    assert get_jets_for_txbb_sf(key) == expected


# ---------------------------------------------------------------------------
# get_nevents_data  (sideband estimate)
# ---------------------------------------------------------------------------


def _mass_events(masses, cut=None):
    """Helper: build a DataFrame with a 'mass' column and an optional boolean cut."""
    df = pd.DataFrame({"mass": masses, "weight": np.ones(len(masses))})
    if cut is None:
        cut = np.ones(len(masses), dtype=bool)
    return df, cut


def test_get_nevents_data_counts_sidebands_not_window():
    # mass window: [100, 150], half-width = 25 → sidebands [75, 100) and (150, 175]
    masses = [80.0, 90.0, 125.0, 160.0, 170.0]
    df, cut = _mass_events(masses)
    n = get_nevents_data(df, cut, "mass", [100, 150])
    # 80, 90 in left sideband; 160, 170 in right sideband → 4 events
    assert n == 4


def test_get_nevents_data_respects_cut():
    masses = [80.0, 90.0, 160.0, 170.0]
    df, _ = _mass_events(masses)
    cut = np.array([True, False, True, False])
    n = get_nevents_data(df, cut, "mass", [100, 150])
    # only 80 (left) and 160 (right) pass the cut
    assert n == 2


def test_get_nevents_data_window_events_not_counted():
    masses = [110.0, 120.0, 130.0]  # all inside window
    df, cut = _mass_events(masses)
    n = get_nevents_data(df, cut, "mass", [100, 150])
    assert n == 0


# ---------------------------------------------------------------------------
# get_nevents_signal / get_nevents_nosignal
# ---------------------------------------------------------------------------


def test_get_nevents_signal_sums_weighted_yield_in_window():
    df = pd.DataFrame(
        {
            "mass": [110.0, 130.0, 90.0, 160.0],
            "weight": [2.0, 3.0, 5.0, 7.0],
        }
    )
    cut = np.array([True, True, True, True])
    # mass window [100, 150]: events at 110 (w=2) and 130 (w=3) → yield = 5
    y = get_nevents_signal(df, cut, "mass", [100, 150])
    assert y == pytest.approx(5.0)


def test_get_nevents_nosignal_sums_yield_outside_window():
    df = pd.DataFrame(
        {
            "mass": [65.0, 110.0, 130.0, 160.0],
            "weight": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cut = np.array([True, True, True, True])
    # outside window but in [60, 220]: 65 (w=1) and 160 (w=4) → 5
    y = get_nevents_nosignal(df, cut, "mass", [100, 150])
    assert y == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# fom_classic / fom_update
# ---------------------------------------------------------------------------


def test_fom_classic_formula():
    s, b = 4.0, 100.0
    assert fom_classic(s, b) == pytest.approx(2 * np.sqrt(b) / s)


def test_fom_classic_zero_signal_returns_nan():
    assert np.isnan(fom_classic(0.0, 100.0))


def test_fom_classic_zero_background_returns_nan():
    assert np.isnan(fom_classic(4.0, 0.0))


def test_fom_update_no_abcd_matches_classic():
    s, b = 4.0, 100.0
    assert fom_update(s, b) == pytest.approx(fom_classic(s, b))


def test_fom_update_with_abcd_larger_than_classic():
    """With ABCD systematics the figure-of-merit should be >= classic."""
    s, b = 4.0, 100.0
    abcd = [0, 50.0, 60.0, 70.0]  # element 0 unused by formula
    result = fom_update(s, b, abcd=abcd)
    assert result >= fom_classic(s, b)


# ---------------------------------------------------------------------------
# combine_run3_samples
# ---------------------------------------------------------------------------


def _yr_df(weight=1.0, n=5):
    return pd.DataFrame({"weight": np.full(n, weight), "x": np.arange(n, dtype=float)})


def test_combine_run3_samples_concatenates_all_years():
    events_dict_years = {
        "2022": {"qcd": _yr_df(1.0, 3), "hh4b": _yr_df(0.5, 2)},
        "2022EE": {"qcd": _yr_df(1.0, 4), "hh4b": _yr_df(0.5, 3)},
    }
    combined, _ = pp.combine_run3_samples(
        events_dict_years,
        processes=["qcd", "hh4b"],
        years_run3=["2022", "2022EE"],
    )
    assert len(combined["qcd"]) == 7
    assert len(combined["hh4b"]) == 5


def test_combine_run3_samples_skips_years_not_in_years_run3():
    events_dict_years = {
        "2022": {"qcd": _yr_df(1.0, 3)},
        "2022EE": {"qcd": _yr_df(1.0, 4)},
        "2023": {"qcd": _yr_df(1.0, 5)},
    }
    combined, _ = pp.combine_run3_samples(
        events_dict_years,
        processes=["qcd"],
        years_run3=["2022", "2022EE"],  # exclude 2023
    )
    assert len(combined["qcd"]) == 7  # only 2022 + 2022EE


def test_combine_run3_samples_scales_weight_for_partial_lumi():
    """scale_processes: a process available only in some years gets its weights upscaled."""
    years_run3 = ["2022", "2022EE"]
    lumi_total = LUMI["2022"] + LUMI["2022EE"]
    lumi_partial = LUMI["2022"]
    expected_scale = lumi_total / lumi_partial

    events_dict_years = {
        "2022": {"hh4b": _yr_df(1.0, 4)},
        "2022EE": {},  # not present
    }
    combined, scaled_by = pp.combine_run3_samples(
        events_dict_years,
        processes=["hh4b"],
        scale_processes={"hh4b": ["2022"]},
        years_run3=years_run3,
    )
    assert scaled_by["hh4b"] == pytest.approx(expected_scale, rel=1e-4)
    np.testing.assert_array_almost_equal(
        combined["hh4b"]["weight"].to_numpy(),
        np.full(4, expected_scale),
    )


def test_combine_run3_samples_merges_ttbar_and_ttlep():
    events_dict_years = {
        "2022": {"ttbar": _yr_df(1.0, 3), "ttlep": _yr_df(1.0, 2)},
    }
    combined, _ = pp.combine_run3_samples(
        events_dict_years,
        processes=["ttbar", "ttlep"],
        bg_keys=["ttbar", "ttlep"],
        years_run3=["2022"],
    )
    assert "ttbar" in combined
    assert "ttlep" not in combined
    assert len(combined["ttbar"]) == 5


def test_combine_run3_samples_skips_missing_process_in_year():
    """If a process is absent for a year, that year is simply skipped."""
    events_dict_years = {
        "2022": {"qcd": _yr_df(1.0, 3)},
        "2022EE": {},  # no qcd
    }
    combined, _ = pp.combine_run3_samples(
        events_dict_years,
        processes=["qcd"],
        years_run3=["2022", "2022EE"],
    )
    assert len(combined["qcd"]) == 3


# ---------------------------------------------------------------------------
# get_weight_shifts
# ---------------------------------------------------------------------------

_KNOWN_TXB = "glopart-v2"
_KNOWN_BDT = "25Feb5_v13_glopartv2_rawmass"


def test_get_weight_shifts_ttbar_xbb_bins_count():
    """Number of ttbarSF_Xbb_bin_* keys must match len(bins)-1."""
    bins = ttbarsfs_decorr_txbb_bins[_KNOWN_TXB]
    ws = pp.get_weight_shifts(_KNOWN_TXB, _KNOWN_BDT)
    xbb_keys = [k for k in ws if k.startswith("ttbarSF_Xbb_bin_")]
    assert len(xbb_keys) == len(bins) - 1


def test_get_weight_shifts_ggf_bdt_bins_count():
    """Number of ttbarSF_ggF_BDT_bin_* keys must match len(bins)-1."""
    bins = ttbarsfs_decorr_ggfbdt_bins[_KNOWN_BDT]
    ws = pp.get_weight_shifts(_KNOWN_TXB, _KNOWN_BDT)
    bdt_keys = [k for k in ws if k.startswith("ttbarSF_ggF_BDT_bin_")]
    assert len(bdt_keys) == len(bins) - 1


def test_get_weight_shifts_txbb_sf_keys_per_wp_and_pt_bin():
    """TXbbSF_uncorrelated keys exist for every (WP, pT bin) combination."""
    wps = txbbsfs_decorr_txbb_wps[_KNOWN_TXB]
    pt_bins = txbbsfs_decorr_pt_bins[_KNOWN_TXB]
    expected_count = sum(len(pt_bins[wp]) - 1 for wp in wps)

    ws = pp.get_weight_shifts(_KNOWN_TXB, _KNOWN_BDT)
    txbb_keys = [k for k in ws if k.startswith("TXbbSF_uncorrelated_")]
    assert len(txbb_keys) == expected_count


def test_get_weight_shifts_core_systematics_present():
    ws = pp.get_weight_shifts(_KNOWN_TXB, _KNOWN_BDT)
    for syst in ["pileup", "trigger", "ISRPartonShower", "FSRPartonShower", "scale", "pdf"]:
        assert syst in ws, f"Expected systematic '{syst}' not found in weight_shifts"


def test_get_weight_shifts_falls_back_for_unknown_txbb():
    """Unknown txbb version falls back to glopart-v2 bins without raising."""
    ws_known = pp.get_weight_shifts(_KNOWN_TXB, _KNOWN_BDT)
    ws_unknown = pp.get_weight_shifts("unknown-tagger", _KNOWN_BDT)
    # Both should have the same number of TXbb and ttbar Xbb keys
    assert len([k for k in ws_known if k.startswith("TXbbSF_uncorrelated_")]) == len(
        [k for k in ws_unknown if k.startswith("TXbbSF_uncorrelated_")]
    )


# ---------------------------------------------------------------------------
# get_templates
# ---------------------------------------------------------------------------


def _make_minimal_events(n=20, seed=0):
    """Build a minimal DataFrame usable in get_templates for a non-sig, non-syst sample."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Category": rng.integers(0, 5, size=n).astype(float),
            "weight": rng.uniform(0.5, 2.0, size=n),
        }
    )


def _fake_make_selection(var_cuts, events_dict, weight_key="weight", jshift="", **kwargs):
    """Minimal make_selection replacement that avoids the coffea PackedSelection dependency."""
    sel = {}
    for sample, events in events_dict.items():
        mask = np.ones(len(events), dtype=bool)
        for var, bounds in var_cuts.items():
            vals = events[var].to_numpy()
            lo, hi = bounds[0], bounds[1]
            mask = mask & (vals >= lo) & (vals < hi)
        sel[sample] = mask
    cutflow = {sample: {} for sample in events_dict}
    return sel, cutflow


def test_get_templates_returns_all_regions(monkeypatch):
    monkeypatch.setattr(pp, "plotting", _DummyPlotting())
    monkeypatch.setattr(pp.utils, "make_selection", _fake_make_selection)

    events_dict = {"qcd": _make_minimal_events(30), "ttbar": _make_minimal_events(20, seed=1)}
    shape_var = ShapeVar(var="Category", label="Category", bins=[5, 0, 5])
    regions = {
        "pass_bin1": pp.Region(cuts={"Category": [1, 2]}, label="Bin1"),
        "fail": pp.Region(cuts={"Category": [4, 5]}, label="Fail"),
    }

    templates = pp.get_templates(
        events_dict=events_dict,
        year="2022",
        sig_keys=[],
        selection_regions=regions,
        shape_vars=[shape_var],
        systematics={},
        bg_keys=["qcd", "ttbar"],
        blind=False,
    )

    assert "pass_bin1" in templates
    assert "fail" in templates


def test_get_templates_hist_contains_all_samples(monkeypatch):
    monkeypatch.setattr(pp, "plotting", _DummyPlotting())
    monkeypatch.setattr(pp.utils, "make_selection", _fake_make_selection)

    events_dict = {"qcd": _make_minimal_events(30), "ttbar": _make_minimal_events(20, seed=1)}
    shape_var = ShapeVar(var="Category", label="Category", bins=[5, 0, 5])
    regions = {"fail": pp.Region(cuts={"Category": [4, 5]}, label="Fail")}

    templates = pp.get_templates(
        events_dict=events_dict,
        year="2022",
        sig_keys=[],
        selection_regions=regions,
        shape_vars=[shape_var],
        systematics={},
        bg_keys=["qcd", "ttbar"],
        blind=False,
    )

    sample_axis = list(templates["fail"].axes["Sample"])
    assert "qcd" in sample_axis
    assert "ttbar" in sample_axis


def test_get_templates_jshift_appends_suffix(monkeypatch):
    monkeypatch.setattr(pp, "plotting", _DummyPlotting())
    monkeypatch.setattr(pp.utils, "make_selection", _fake_make_selection)

    events_dict = {"qcd": _make_minimal_events(30)}
    shape_var = ShapeVar(var="Category", label="Category", bins=[5, 0, 5])
    regions = {"fail": pp.Region(cuts={"Category": [4, 5]}, label="Fail")}

    templates = pp.get_templates(
        events_dict=events_dict,
        year="2022",
        sig_keys=[],
        selection_regions=regions,
        shape_vars=[shape_var],
        systematics={},
        bg_keys=["qcd"],
        jshift="JES_up",
        blind=False,
    )

    assert "fail_JES_up" in templates
    assert "fail" not in templates  # no un-shifted key when jshift is set


def test_get_templates_weight_shift_adds_up_down_samples(monkeypatch):
    from HH4b.utils import Syst

    monkeypatch.setattr(pp, "plotting", _DummyPlotting())
    monkeypatch.setattr(pp.utils, "make_selection", _fake_make_selection)

    events_dict = {
        "qcd": pd.DataFrame(
            {
                "Category": np.array([4.5] * 10),
                "weight": np.ones(10),
                "weight_pileupUp": np.ones(10) * 1.1,
                "weight_pileupDown": np.ones(10) * 0.9,
            }
        )
    }
    shape_var = ShapeVar(var="Category", label="Category", bins=[5, 0, 5])
    regions = {"fail": pp.Region(cuts={"Category": [4, 5]}, label="Fail")}
    weight_shifts = {"pileup": Syst(samples=["qcd"], label="Pileup", years=["2022"])}

    templates = pp.get_templates(
        events_dict=events_dict,
        year="2022",
        sig_keys=[],
        selection_regions=regions,
        shape_vars=[shape_var],
        systematics={},
        bg_keys=["qcd"],
        weight_shifts=weight_shifts,
        blind=False,
    )

    sample_axis = list(templates["fail"].axes["Sample"])
    assert "qcd_pileup_up" in sample_axis
    assert "qcd_pileup_down" in sample_axis


# ---------------------------------------------------------------------------
# scale_smear_mass
# ---------------------------------------------------------------------------


def _make_mass_events(key, mass_str, n=10, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(80, 180, size=(n, 2))
    df = pd.DataFrame(
        {(f"{mass_str}_raw", 0): raw[:, 0], (f"{mass_str}_raw", 1): raw[:, 1]},
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return {key: df}


def test_scale_smear_mass_modifies_mass_column():
    mass_str = "bbFatJetParTmassVis"
    year = "2022"
    key = "hh4b"  # a valid jmsr_keys entry
    events_dict = _make_mass_events(key, mass_str)
    raw_vals = events_dict[key][(f"{mass_str}_raw", 0)].to_numpy().copy()

    result = pp.scale_smear_mass(events_dict, year, mass_str)

    # The nominal mass should differ from raw due to JMS scaling
    scaled_vals = result[key][(mass_str, 0)].to_numpy()
    assert not np.allclose(scaled_vals, raw_vals), "Scaled mass should differ from raw mass"


def test_scale_smear_mass_produces_jms_jmr_shifts():
    mass_str = "bbFatJetParTmassVis"
    year = "2022"
    key = "hh4b"
    events_dict = _make_mass_events(key, mass_str)

    result = pp.scale_smear_mass(events_dict, year, mass_str)

    expected_cols = [
        (f"{mass_str}_JMS_up", 0),
        (f"{mass_str}_JMS_down", 0),
        (f"{mass_str}_JMR_up", 0),
        (f"{mass_str}_JMR_down", 0),
    ]
    for col in expected_cols:
        assert col in result[key].columns, f"Missing column {col}"


def test_scale_smear_mass_is_deterministic():
    """Two calls with the same seed=42 must produce identical results."""
    mass_str = "bbFatJetParTmassVis"
    year = "2022"
    key = "hh4b"

    events1 = _make_mass_events(key, mass_str, seed=7)
    events2 = _make_mass_events(key, mass_str, seed=7)

    r1 = pp.scale_smear_mass(events1, year, mass_str)
    r2 = pp.scale_smear_mass(events2, year, mass_str)

    np.testing.assert_array_equal(
        r1[key][(mass_str, 0)].to_numpy(),
        r2[key][(mass_str, 0)].to_numpy(),
    )


def test_scale_smear_mass_skips_non_jmsr_keys():
    """Samples not in jmsr_keys (e.g. 'qcd') should be returned unchanged."""
    mass_str = "bbFatJetParTmassVis"
    year = "2022"
    key = "qcd"  # NOT in jmsr_keys
    events_dict = _make_mass_events(key, mass_str)
    original_cols = set(events_dict[key].columns.tolist())

    result = pp.scale_smear_mass(events_dict, year, mass_str)

    assert set(result[key].columns.tolist()) == original_cols


# ---------------------------------------------------------------------------
# Minimal stub for plotting to avoid heavy deps in unit tests
# ---------------------------------------------------------------------------


class _DummyPlotting:
    """Drop-in replacement for the plotting module inside get_templates."""

    label_by_sample: dict = {}
    color_by_sample: dict = {}

    def ratioHistPlot(self, *args, **kwargs):  # noqa: N802
        pass
