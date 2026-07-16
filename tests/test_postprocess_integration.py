"""Integration tests for postprocess_run3 helpers using a small fixture dataset.

These tests exercise the full _load_samples → _combine_years →
_build_combined_cutflow pipeline against real (but tiny) parquet files.
They are skipped automatically when the fixture does not exist.

To create the fixture, run once:

    python scripts/make_test_data.py \\
        --data-dir /ceph/cms/store/user/<user>/bbbb/skimmer \\
        --tag 25May9_v12v2_private_signal \\
        --out-dir tests/fixtures/skimmer \\
        --years 2022 \\
        --samples hh4b qcd data

The fixture directory is intentionally not committed to git (it lives under
tests/fixtures/ which is gitignored).  Store a copy on EOS or ceph and
document the path in the group wiki so anyone can regenerate it.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hist")
pytest.importorskip("pandas")
pytest.importorskip("xgboost")

# ---------------------------------------------------------------------------
# Fixture location
# ---------------------------------------------------------------------------

FIXTURE_TAG = "25May9_v12v2_private_signal"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "skimmer"
FIXTURE_YEAR_DIR = FIXTURE_DIR / FIXTURE_TAG / "2022"

requires_fixture = pytest.mark.skipif(
    not FIXTURE_YEAR_DIR.exists(),
    reason=(
        f"Integration fixture not found at {FIXTURE_YEAR_DIR}. "
        "Run  python scripts/make_test_data.py --data-dir <skimmer-root> "
        f"--tag {FIXTURE_TAG} --out-dir tests/fixtures/skimmer  to create it."
    ),
)


# ---------------------------------------------------------------------------
# Shared args
# ---------------------------------------------------------------------------


def _integration_args(**overrides):
    base = Namespace(
        data_dir=str(FIXTURE_DIR),
        tag=FIXTURE_TAG,
        years=["2022"],
        txbb="glopart-v2",
        mass="H2PNetMass",
        mass_bins=10,
        sig_keys=["hh4b"],
        txbb_wps=[0.945, 0.85],
        bdt_wps=[0.94, 0.755, 0.03],
        bdt_model="25Feb5_v13_glopartv2_rawmass",
        bdt_config="v13_glopartv2",
        vbf=False,
        vbf_priority=False,
        vbf_txbb_wp=0.8,
        vbf_bdt_wp=0.9825,
        method="abcd",
        fom_scan=False,
        fom_scan_vbf=False,
        fom_scan_bin1=False,
        fom_scan_bin2=False,
        fom_vbf_samples=["vbfhh4b-k2v0"],
        fom_ggf_samples=["hh4b"],
        templates=False,
        control_plots=False,
        bdt_roc=False,
        blind=True,
        event_list=False,
        event_list_dir="event_lists",
        rerun_inference=False,
        correct_vbf_bdt_shape=False,
        scale_smear=False,
        dummy_txbb_sfs=True,  # skip real SFs that need correction files
        training_years=None,
        pt_first=300.0,
        pt_second=250.0,
        weight_ttbar_bdt=1.0,
        bdt_disc=True,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_fixture
def test_load_samples_returns_expected_keys(tmp_path):
    """_load_samples should return a dict keyed by year with the loaded samples."""
    from HH4b.postprocessing.PostProcess import _load_samples

    args = _integration_args()
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()

    events_dict, cutflows = _load_samples(args, plot_dir, np.array([110, 140]))

    assert "2022" in events_dict
    # At least one sample should be present
    assert len(events_dict["2022"]) > 0
    assert "2022" in cutflows


@requires_fixture
def test_load_samples_events_have_required_columns(tmp_path):
    """Every loaded sample must have at minimum a 'weight' and 'year' column."""
    from HH4b.postprocessing.PostProcess import _load_samples

    args = _integration_args()
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()

    events_dict, _ = _load_samples(args, plot_dir, np.array([110, 140]))

    for sample_key, df in events_dict["2022"].items():
        assert "weight" in df.columns, f"'weight' missing from {sample_key}"
        assert "year" in df.columns, f"'year' missing from {sample_key}"


@requires_fixture
def test_combine_years_single_year_is_passthrough(tmp_path):
    """With one year, _combine_years should return the loaded dict unchanged."""
    from HH4b.postprocessing.PostProcess import _combine_years, _load_samples

    args = _integration_args()
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()

    events_dict, _ = _load_samples(args, plot_dir, np.array([110, 140]))

    samples = list(events_dict["2022"].keys())
    combined, scaled_by, scaled_by_years = _combine_years(args, events_dict, samples, [])

    assert scaled_by == {}
    assert scaled_by_years == {}
    for key in events_dict["2022"]:
        assert key in combined


@requires_fixture
def test_build_combined_cutflow_produces_dataframe(tmp_path):
    """_build_combined_cutflow should return a non-empty DataFrame."""
    from HH4b.postprocessing.PostProcess import (
        _build_combined_cutflow,
        _combine_years,
        _load_samples,
    )

    args = _integration_args()
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()

    events_dict, cutflows = _load_samples(args, plot_dir, np.array([110, 140]))
    samples = list(events_dict["2022"].keys())
    events_combined, scaled_by, scaled_by_years = _combine_years(args, events_dict, samples, [])

    bg_keys = [k for k in samples if k not in args.sig_keys and k != "data"]
    cutflow_combined = _build_combined_cutflow(
        args,
        events_combined,
        cutflows,
        scaled_by,
        scaled_by_years,
        np.array([110, 140]),
        bg_keys,
    )

    assert cutflow_combined is not None
    assert len(cutflow_combined) > 0


@requires_fixture
def test_full_pipeline_no_templates_completes(tmp_path):
    """Run postprocess_run3 with --templates=False end-to-end; it should not raise."""
    import HH4b.hh_vars as hh_vars_mod
    from HH4b.postprocessing.PostProcess import postprocess_run3

    original_bg_keys = list(hh_vars_mod.bg_keys)
    try:
        args = _integration_args(
            templates=False,
            templates_tag="integration_test",
        )
        # Change to tmp_path so templates/ dir is written there
        import os

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            postprocess_run3(args)
        finally:
            os.chdir(cwd)
    finally:
        # Restore the global bg_keys list modified by postprocess_run3
        hh_vars_mod.bg_keys[:] = original_bg_keys

    assert (tmp_path / "templates" / "integration_test" / "args.txt").exists()
