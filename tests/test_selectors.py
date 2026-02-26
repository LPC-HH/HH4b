from __future__ import annotations

import pytest

pytest.importorskip("hist")

from HH4b.utils import check_selector


@pytest.mark.parametrize(
    ("sample", "selector", "expected"),
    [
        ("ZZ", "ZZ?", True),
        ("ZZTo4B", "ZZ?", False),
        ("QCD-4Jets_HT-1000to1200", "QCD-4Jets_HT", True),
        ("QCD_HT-1000to1200", "QCD_HT-", True),
        ("VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV", "*TuneCP5*", True),
        ("JetMET_Run2024C", "*Run2024", False),
    ],
)
def test_check_selector_patterns(sample: str, selector: str, expected: bool):
    assert check_selector(sample, selector) is expected


def test_check_selector_list_uses_or_logic():
    selectors = ["QCD-4Jets_HT", "QCD_HT-"]
    assert check_selector("QCD-4Jets_HT-1200to1500", selectors)
    assert check_selector("QCD_HT-1200to1500", selectors)


def test_check_selector_mixed_list_patterns():
    selectors = ["ZZ?", "*TuneCP5*", "QCD_HT-"]
    assert check_selector("ZZ", selectors)
    assert check_selector("QCD_HT-800to1000", selectors)
    assert check_selector("GluGlutoHHto4B_TuneCP5_13p6TeV", selectors)
    assert not check_selector("CompletelyDifferentSample", selectors)
