from __future__ import annotations

import pytest

pytest.importorskip("hist")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from HH4b.postprocessing import postprocessing as pp


def _base_samples_run3_for_year(year: str) -> dict[str, dict[str, list[str]]]:
    return {
        year: {
            "data": [f"JetMET_Run{year}X"],
            "qcd": ["QCD_HT-"],
            "ttbar": ["TTto4Q"],
            "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"],
            "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1"],
        }
    }


def test_load_run3_samples_partitions_and_attaches_year_hlts(monkeypatch):
    calls: list[dict] = []

    def fake_load_samples(_input_dir, samples, _year, **kwargs):
        calls.append({"samples": list(samples.keys()), "columns": kwargs.get("columns", [])})
        return {label: {"weight": [1.0]} for label in samples}

    monkeypatch.setattr(pp.utils, "load_samples", fake_load_samples)

    samples_run3 = _base_samples_run3_for_year("2024")
    events = pp.load_run3_samples(
        input_dir="/tmp/does_not_matter",
        year="2024",
        samples_run3=samples_run3,
        reorder_txbb=False,
        load_systematics=False,
        txbb_version="glopart-v2",
        scale_and_smear=False,
        mass_str="bbFatJetParTmassVis",
        bdt_version="unit-test-bdt",
    )

    assert set(events) == {"data", "qcd", "ttbar", "hh4b", "vbfhh4b"}
    assert len(calls) == 5
    for call in calls:
        for hlt in pp.HLTs["2024"]:
            assert any(hlt in col for col in call["columns"])


def test_load_run3_samples_glopart_v3_rewrites_systematic_mass_columns(monkeypatch):
    captured_columns: list[list[str]] = []

    def fake_load_samples(_input_dir, samples, _year, **kwargs):
        if samples:
            captured_columns.append(kwargs.get("columns", []))
        return {label: {"weight": [1.0]} for label in samples}

    monkeypatch.setattr(pp.utils, "load_samples", fake_load_samples)

    samples_run3 = _base_samples_run3_for_year("2024")
    pp.load_run3_samples(
        input_dir="/tmp/does_not_matter",
        year="2024",
        samples_run3=samples_run3,
        reorder_txbb=False,
        load_systematics=True,
        txbb_version="glopart-v3",
        scale_and_smear=False,
        mass_str="bbFatJetParT3massX2p",
        bdt_version="unit-test-bdt",
    )

    all_columns = [col for call_columns in captured_columns for col in call_columns]
    assert any("bbFatJetParT3massX2p_JMS_up" in col for col in all_columns)
    assert not any("bbFatJetParTmassVis_JMS_up" in col for col in all_columns)
