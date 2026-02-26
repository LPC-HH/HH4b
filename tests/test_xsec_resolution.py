from __future__ import annotations

import pytest

pytest.importorskip("hist")

from HH4b.utils import _resolve_xsec_key


def test_resolve_xsec_key_exact_match():
    xsecs = {"QCD_HT-1000to1200": 1.0}
    assert _resolve_xsec_key("QCD_HT-1000to1200", xsecs) == "QCD_HT-1000to1200"


def test_resolve_xsec_key_qcd_alias():
    xsecs = {"QCD_HT-1000to1200": 1.0}
    assert _resolve_xsec_key("QCD-4Jets_HT-1000to1200", xsecs) == "QCD_HT-1000to1200"


def test_resolve_xsec_key_tth_alias():
    xsecs = {"ttHto2B_M-125": 1.0}
    assert _resolve_xsec_key("TTHto2B_M-125", xsecs) == "ttHto2B_M-125"


def test_resolve_xsec_key_case_insensitive_unique():
    xsecs = {"vbfhto2b_m-125_dipolerecoilon": 1.0}
    assert (
        _resolve_xsec_key("VBFHto2B_M-125_dipoleRecoilOn", xsecs)
        == "vbfhto2b_m-125_dipolerecoilon"
    )


def test_resolve_xsec_key_raises_for_ambiguous_case_insensitive_match():
    xsecs = {
        "sampleA": 1.0,
        "SAMPLEA": 2.0,
    }
    with pytest.raises(KeyError, match="No xsec entry"):
        _resolve_xsec_key("SampleA", xsecs)
