from __future__ import annotations

from HH4b import hh_vars


def test_run3_years_are_present_in_samples_map():
    for year in hh_vars.years:
        assert year in hh_vars.samples_run3


def test_qcd_selector_supports_both_naming_conventions():
    qcd_selectors = set(hh_vars.common_samples_bg["qcd"])
    assert "QCD-4Jets_HT" in qcd_selectors
    assert "QCD_HT-" in qcd_selectors


def test_required_background_selectors_are_non_empty():
    required = ["qcd", "ttbar", "zz", "nozzdiboson", "vjets"]
    for key in required:
        assert key in hh_vars.common_samples_bg
        assert len(hh_vars.common_samples_bg[key]) > 0


def test_diboson_selector_contract():
    assert hh_vars.common_samples_bg["zz"] == ["ZZ?"]
    assert set(hh_vars.common_samples_bg["nozzdiboson"]) == {"WW?", "WZ?"}


def test_lumi_has_combined_2025_key():
    assert "2025" in hh_vars.LUMI
    assert hh_vars.LUMI["2025"] > 0


def test_2024_and_2025_haves_data_key():
    assert "data" in hh_vars.samples_run3["2024"]
    assert "data" in hh_vars.samples_run3["2025"]
    assert len(hh_vars.samples_run3["2024"]["data"]) > 0
    assert len(hh_vars.samples_run3["2025"]["data"]) > 0
