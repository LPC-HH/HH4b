"""Unit tests for the private helper functions extracted from postprocess_run3.

These tests cover the pure / near-pure helpers that were pulled out of the
monolithic postprocess_run3 function.  They use only synthetic DataFrames and
tmp directories — no parquet files, no network, no GPU.

For end-to-end tests that exercise load_process_run3_samples against real
(or fixture) data, see test_postprocess_integration.py.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hist")
pytest.importorskip("pandas")

from HH4b.postprocessing.PostProcess import (
    _build_combined_cutflow,
    _combine_years,
    _get_mass_windows,
    _save_cutflows,
    _save_event_lists,
    _setup_shape_var,
)
from HH4b.utils import ShapeVar

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _minimal_args(**overrides):
    """Return an args Namespace with valid defaults for all helper functions."""
    base = Namespace(
        txbb="glopart-v2",
        mass="H2PNetMass",
        mass_bins=10,
        years=["2022", "2022EE"],
        sig_keys=["hh4b"],
        txbb_wps=[0.945, 0.85],
        bdt_wps=[0.94, 0.755, 0.03],
        vbf=False,
        vbf_priority=False,
        vbf_txbb_wp=0.8,
        vbf_bdt_wp=0.9825,
        fom_vbf_samples=["vbfhh4b-k2v0"],
        fom_ggf_samples=["hh4b"],
        method="abcd",
        event_list=False,
        event_list_dir="event_lists",
        fom_scan=False,
        fom_scan_vbf=False,
        fom_scan_bin1=False,
        fom_scan_bin2=False,
        bdt_model="25Feb5_v13_glopartv2_rawmass",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _make_events(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "H2PNetMass": rng.uniform(60, 220, n),
            "H2Msd": rng.uniform(60, 220, n),
            "bdt_score": rng.uniform(0, 1, n),
            "bdt_score_vbf": rng.uniform(0, 1, n),
            "H2TXbb": rng.uniform(0, 1, n),
            "H2DiscTXbb": rng.integers(1, 7, n).astype(float),
            "Category": rng.integers(0, 5, n).astype(float),
            "weight": rng.uniform(0.5, 2.0, n),
            "run": np.ones(n, dtype=int),
            "event": np.arange(n, dtype=int),
            "luminosityBlock": np.ones(n, dtype=int),
            "year": np.full(n, "2022"),
        }
    )


# ---------------------------------------------------------------------------
# _get_mass_windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("txbb", "expected_fom", "expected_blind"),
    [
        ("pnet-legacy", [105, 150], [110, 140]),
        ("pnet-v12", [120, 150], [120, 150]),
        ("glopart-v2", [110, 155], [110, 140]),
        ("glopart-v3", [110, 155], [110, 140]),
    ],
)
def test_get_mass_windows_pnetmass_windows(txbb, expected_fom, expected_blind):
    args = _minimal_args(txbb=txbb)
    fom, blind = _get_mass_windows(args)
    assert fom["H2PNetMass"] == expected_fom
    assert blind["H2PNetMass"] == expected_blind


def test_get_mass_windows_msd_always_present():
    """H2Msd windows must be set regardless of which tagger is used."""
    for txbb in ["pnet-legacy", "glopart-v2", "glopart-v3", "pnet-v12"]:
        fom, blind = _get_mass_windows(_minimal_args(txbb=txbb))
        assert "H2Msd" in fom
        assert "H2Msd" in blind
        assert fom["H2Msd"] == [110, 140]


def test_get_mass_windows_glopart_v3_matches_v2():
    """glopart-v3 reuses the glopart-v2 mass windows."""
    fom_v2, blind_v2 = _get_mass_windows(_minimal_args(txbb="glopart-v2"))
    fom_v3, blind_v3 = _get_mass_windows(_minimal_args(txbb="glopart-v3"))
    assert fom_v2["H2PNetMass"] == fom_v3["H2PNetMass"]
    assert blind_v2["H2PNetMass"] == blind_v3["H2PNetMass"]


def test_get_mass_windows_unknown_txbb_returns_no_pnetmass_key():
    """An unrecognised tagger should not populate H2PNetMass (caller handles it)."""
    fom, blind = _get_mass_windows(_minimal_args(txbb="future-tagger-v99"))
    assert "H2PNetMass" not in fom
    assert "H2PNetMass" not in blind


# ---------------------------------------------------------------------------
# _setup_shape_var
# ---------------------------------------------------------------------------


def test_setup_shape_var_bin_count():
    args = _minimal_args(mass="H2PNetMass", mass_bins=10)
    _, blind = _get_mass_windows(args)
    sv = _setup_shape_var(args, blind)
    assert isinstance(sv, ShapeVar)
    assert sv.bins[0] == int((220 - 60) / 10)  # 16


def test_setup_shape_var_blind_window_propagated():
    args = _minimal_args(txbb="glopart-v2", mass="H2PNetMass", mass_bins=10)
    _, blind = _get_mass_windows(args)
    sv = _setup_shape_var(args, blind)
    assert sv.blind_window == blind["H2PNetMass"]


def test_setup_shape_var_msd():
    args = _minimal_args(mass="H2Msd", mass_bins=10)
    _, blind = _get_mass_windows(args)
    sv = _setup_shape_var(args, blind)
    assert sv.var == "H2Msd"
    assert sv.blind_window == [110, 140]


# ---------------------------------------------------------------------------
# _combine_years
# ---------------------------------------------------------------------------


def _yr_df(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"weight": rng.uniform(0.5, 2.0, n), "x": np.arange(n, dtype=float)})


def test_combine_years_single_year_passthrough():
    """With one year, combine_years should return the dict unchanged."""
    events_dict = {"2022": {"hh4b": _yr_df(5), "qcd": _yr_df(8, seed=1)}}
    args = _minimal_args(years=["2022"], sig_keys=["hh4b"])
    combined, scaled_by, scaled_by_years = _combine_years(
        args, events_dict, ["hh4b", "qcd"], ["qcd"]
    )
    # _combine_years returns a shallow copy for single-year (not the same object)
    assert combined == events_dict["2022"]
    assert scaled_by == {}
    assert scaled_by_years == {}


def test_combine_years_multi_year_all_sig_keys_in_scaled_by_years():
    """Every sig_key should appear in scaled_by_years when combining multiple years."""
    events_dict = {
        "2022": {"hh4b": _yr_df(5), "vbfhh4b": _yr_df(3, seed=1)},
        "2022EE": {"hh4b": _yr_df(4, seed=2), "vbfhh4b": _yr_df(2, seed=3)},
    }
    args = _minimal_args(years=["2022", "2022EE"], sig_keys=["hh4b", "vbfhh4b"])

    # monkeypatch combine_run3_samples to avoid importing heavy deps at module level
    import HH4b.postprocessing.PostProcess as pp_mod

    def _fake_combine(
        events_dict_years, processes, *, bg_keys=None, scale_processes=None, years_run3=None
    ):
        combined = {}
        for proc in processes:
            parts = [v[proc] for v in events_dict_years.values() if proc in v]
            if parts:
                combined[proc] = pd.concat(parts, ignore_index=True)
        return combined, dict.fromkeys(scale_processes or {}, 1.0)

    original = pp_mod.combine_run3_samples
    pp_mod.combine_run3_samples = _fake_combine
    try:
        _, _, scaled_by_years = _combine_years(args, events_dict, ["hh4b", "vbfhh4b"], [])
    finally:
        pp_mod.combine_run3_samples = original

    assert "hh4b" in scaled_by_years
    assert "vbfhh4b" in scaled_by_years


# ---------------------------------------------------------------------------
# _build_combined_cutflow
# ---------------------------------------------------------------------------


def _make_cutflows(years, samples, cuts):
    """Minimal cutflows dict: each cut → Series of yield-per-sample."""
    cutflows = {}
    for year in years:
        cutflows[year] = {
            cut: pd.Series({s: float(i + 1) for i, s in enumerate(samples)}) for cut in cuts
        }
    return cutflows


def test_build_combined_cutflow_empty_years_returns_none():
    args = _minimal_args(years=[])
    result = _build_combined_cutflow(args, {}, {}, {}, {}, np.array([110, 140]), [])
    assert result is None


def test_build_combined_cutflow_returns_dataframe():
    samples = ["data", "qcd", "hh4b"]
    years = ["2022"]
    cuts = ["preselection"]
    events_combined = {s: _make_events(20) for s in samples}
    cutflows = _make_cutflows(years, samples, cuts)

    args = _minimal_args(years=years)

    import HH4b.postprocessing.PostProcess as pp_mod

    def _fake_abcd(
        events, get_cut, get_anti_cut, txbb_wp, bdt_wp, mass, mass_window, bg_keys, sig_keys
    ):
        return 2.0, 10.0, None

    original_abcd = pp_mod.abcd
    pp_mod.abcd = _fake_abcd
    try:
        result = _build_combined_cutflow(
            args, events_combined, cutflows, {}, {}, np.array([110, 140]), ["qcd"]
        )
    finally:
        pp_mod.abcd = original_abcd

    assert isinstance(result, pd.DataFrame)
    assert set(result.index) >= set(samples)


# ---------------------------------------------------------------------------
# _save_cutflows
# ---------------------------------------------------------------------------


def test_save_cutflows_creates_per_year_csv(tmp_path):
    args = _minimal_args(years=["2022", "2022EE"])
    cutflows = {
        "2022": pd.DataFrame({"cut_a": [1.0, 2.0]}, index=["qcd", "hh4b"]),
        "2022EE": pd.DataFrame({"cut_a": [3.0, 4.0]}, index=["qcd", "hh4b"]),
    }
    templ_dir = tmp_path / "templates" / "test_tag"
    (templ_dir / "cutflows").mkdir(parents=True)

    _save_cutflows(args, templ_dir, cutflows, cutflow_combined=None)

    assert (templ_dir / "cutflows" / "preselection_cutflow_2022.csv").exists()
    assert (templ_dir / "cutflows" / "preselection_cutflow_2022EE.csv").exists()
    assert not (templ_dir / "cutflows" / "preselection_cutflow_combined.csv").exists()


def test_save_cutflows_writes_combined_when_provided(tmp_path):
    args = _minimal_args(years=["2022"])
    cutflows = {"2022": pd.DataFrame({"cut_a": [1.0]}, index=["qcd"])}
    combined = pd.DataFrame({"cut_a": ["1.0000"]}, index=["qcd"])
    templ_dir = tmp_path / "templates" / "test_tag"
    (templ_dir / "cutflows").mkdir(parents=True)

    _save_cutflows(args, templ_dir, cutflows, cutflow_combined=combined)

    assert (templ_dir / "cutflows" / "preselection_cutflow_combined.csv").exists()


def test_save_cutflows_rounds_to_4_decimal_places(tmp_path):
    args = _minimal_args(years=["2022"])
    cutflows = {"2022": pd.DataFrame({"cut_a": [1.123456789]}, index=["qcd"])}
    templ_dir = tmp_path / "templates" / "test_tag"
    (templ_dir / "cutflows").mkdir(parents=True)

    _save_cutflows(args, templ_dir, cutflows, cutflow_combined=None)

    csv_path = templ_dir / "cutflows" / "preselection_cutflow_2022.csv"
    content = csv_path.read_text()
    assert "1.1235" in content
    assert "1.123456789" not in content


# ---------------------------------------------------------------------------
# _save_event_lists
# ---------------------------------------------------------------------------


def test_save_event_lists_creates_root_file(tmp_path, monkeypatch):
    uproot = pytest.importorskip("uproot")
    import HH4b.postprocessing.PostProcess as pp_mod

    monkeypatch.setattr(pp_mod, "write_eventlist_manifest", lambda *_, **__: None)

    events_dict = {
        "2022": {
            "data": _make_events(10),
            "hh4b": _make_events(5, seed=1),
            "qcd": _make_events(8, seed=2),  # should be skipped
        }
    }
    args = _minimal_args(event_list_dir=str(tmp_path / "eventlists"))

    _save_event_lists(args, events_dict, np.array([110, 140]))

    root_path = tmp_path / "eventlists" / "eventlist_boostedHH4b_2022.root"
    assert root_path.exists()

    with uproot.open(root_path) as f:
        keys = [k.rstrip(";1") for k in f]
    assert "data" in keys
    assert "hh4b" in keys
    assert "qcd" not in keys  # not data/hh4b/vbfhh4b


def test_save_event_lists_appends_to_existing_root_file(tmp_path, monkeypatch):
    uproot = pytest.importorskip("uproot")
    import HH4b.postprocessing.PostProcess as pp_mod

    monkeypatch.setattr(pp_mod, "write_eventlist_manifest", lambda *_, **__: None)

    # Two years → two calls that write / append to the same file per year
    events_dict = {
        "2022": {"data": _make_events(5)},
    }
    args = _minimal_args(event_list_dir=str(tmp_path / "eventlists"))

    _save_event_lists(args, events_dict, np.array([110, 140]))
    # Call again — simulates a second sample writing to the already-created file
    events_dict2 = {"2022": {"hh4b": _make_events(3, seed=1)}}
    _save_event_lists(args, events_dict2, np.array([110, 140]))

    root_path = tmp_path / "eventlists" / "eventlist_boostedHH4b_2022.root"
    with uproot.open(root_path) as f:
        keys = [k.rstrip(";1") for k in f]
    assert "data" in keys
    assert "hh4b" in keys
