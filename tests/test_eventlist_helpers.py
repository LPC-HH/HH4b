"""Tests for event-list helper functions in PostProcess.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from HH4b.postprocessing.PostProcess import (
    EVENTLIST_BASE_COLUMNS,
    _add_ordered_gen_higgs_columns,
    _build_event_list_frame,
    _compute_all_hist_samples,
    _decode_coupling_value,
    _get_column_array,
    _get_indexed_column_array,
    _parse_signal_couplings,
)


# ---------------------------------------------------------------------------
# _decode_coupling_value
# ---------------------------------------------------------------------------
class TestDecodeCouplingValue:
    def test_positive_integer(self):
        assert _decode_coupling_value("5") == 5.0

    def test_positive_decimal(self):
        assert _decode_coupling_value("2p45") == pytest.approx(2.45)

    def test_negative_integer(self):
        assert _decode_coupling_value("m3") == -3.0

    def test_negative_decimal(self):
        assert _decode_coupling_value("m2p45") == pytest.approx(-2.45)

    def test_zero(self):
        assert _decode_coupling_value("0") == 0.0

    def test_multiple_decimal_points(self):
        # "1p00" should work
        assert _decode_coupling_value("1p00") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _parse_signal_couplings
# ---------------------------------------------------------------------------
class TestParseSignalCouplings:
    @pytest.mark.parametrize(
        ("key", "expected_kl", "expected_k2v"),
        [
            ("hh4b", 1.0, 1.0),
            ("hh4b-kl0", 0.0, 1.0),
            ("hh4b-kl2p45", 2.45, 1.0),
            ("hh4b-kl5", 5.0, 1.0),
        ],
    )
    def test_ggf_samples(self, key, expected_kl, expected_k2v):
        kl, k2v = _parse_signal_couplings(key)
        assert kl == pytest.approx(expected_kl)
        assert k2v == pytest.approx(expected_k2v)

    @pytest.mark.parametrize(
        ("key", "expected_kl", "expected_k2v"),
        [
            ("vbfhh4b", 1.0, 1.0),
            ("vbfhh4b-k2v0", 1.0, 0.0),
            ("vbfhh4b-kv1p74-k2v1p37-kl14p4", 14.4, 1.37),
            ("vbfhh4b-kvm0p012-k2v0p03-kl10p2", 10.2, 0.03),
            ("vbfhh4b-kvm0p758-k2v1p44-klm19p3", -19.3, 1.44),
            ("vbfhh4b-kvm1p83-k2v3p57-klm3p39", -3.39, 3.57),
            ("vbfhh4b-kvm2p12-k2v3p87-klm5p96", -5.96, 3.87),
        ],
    )
    def test_vbf_samples(self, key, expected_kl, expected_k2v):
        kl, k2v = _parse_signal_couplings(key)
        assert kl == pytest.approx(expected_kl)
        assert k2v == pytest.approx(expected_k2v)

    def test_unsupported_key_raises(self):
        with pytest.raises(ValueError, match="Unsupported signal key"):
            _parse_signal_couplings("ttbar")


# ---------------------------------------------------------------------------
# _get_column_array / _get_indexed_column_array
# ---------------------------------------------------------------------------
class TestGetColumnArray:
    def test_flat_column(self):
        df = pd.DataFrame({"foo": [1.0, 2.0, 3.0]})
        result = _get_column_array(df, "foo")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_tuple_column(self):
        df = pd.DataFrame({("bar", 0): [4.0, 5.0]})
        result = _get_column_array(df, "bar")
        np.testing.assert_array_equal(result, [4.0, 5.0])

    def test_missing_returns_none(self):
        df = pd.DataFrame({"foo": [1.0]})
        assert _get_column_array(df, "missing") is None


class TestGetIndexedColumnArray:
    def test_tuple_indexed(self):
        df = pd.DataFrame({("GenHiggsPt", 0): [100.0], ("GenHiggsPt", 1): [80.0]})
        result = _get_indexed_column_array(df, "GenHiggsPt", 0)
        np.testing.assert_array_equal(result, [100.0])

    def test_integer_suffix(self):
        df = pd.DataFrame({"GenHiggsPt1": [100.0], "GenHiggsPt2": [80.0]})
        result = _get_indexed_column_array(df, "GenHiggsPt", 0)
        np.testing.assert_array_equal(result, [100.0])

    def test_missing_returns_none(self):
        df = pd.DataFrame({"x": [1.0]})
        assert _get_indexed_column_array(df, "GenHiggsPt", 0) is None


# ---------------------------------------------------------------------------
# _add_ordered_gen_higgs_columns
# ---------------------------------------------------------------------------
class TestAddOrderedGenHiggsColumns:
    def _make_tree_df(self, pt1, pt2, eta1, eta2, phi1, phi2, m1, m2):
        """Create a tree_df with GenHiggs columns using tuple-index convention."""
        return pd.DataFrame(
            {
                ("GenHiggsPt", 0): pt1,
                ("GenHiggsPt", 1): pt2,
                ("GenHiggsEta", 0): eta1,
                ("GenHiggsEta", 1): eta2,
                ("GenHiggsPhi", 0): phi1,
                ("GenHiggsPhi", 1): phi2,
                ("GenHiggsMass", 0): m1,
                ("GenHiggsMass", 1): m2,
            }
        )

    def test_orders_by_pt(self):
        # Higgs 1 has higher pT in some events, Higgs 2 in others
        tree_df = self._make_tree_df(
            pt1=[200.0, 50.0],
            pt2=[100.0, 300.0],
            eta1=[1.0, 2.0],
            eta2=[3.0, 4.0],
            phi1=[0.1, 0.2],
            phi2=[0.3, 0.4],
            m1=[125.0, 125.0],
            m2=[125.0, 125.0],
        )
        event_list = pd.DataFrame(index=range(2))
        _add_ordered_gen_higgs_columns(event_list, tree_df)

        # Event 0: pt1=200 > pt2=100, so H1_FC = Higgs1
        assert event_list["genp_H1_FC_pt"].iloc[0] == pytest.approx(200.0)
        assert event_list["genp_H2_FC_pt"].iloc[0] == pytest.approx(100.0)
        assert event_list["genp_H1_FC_eta"].iloc[0] == pytest.approx(1.0)
        assert event_list["genp_H2_FC_eta"].iloc[0] == pytest.approx(3.0)

        # Event 1: pt1=50 < pt2=300, so H1_FC = Higgs2
        assert event_list["genp_H1_FC_pt"].iloc[1] == pytest.approx(300.0)
        assert event_list["genp_H2_FC_pt"].iloc[1] == pytest.approx(50.0)
        assert event_list["genp_H1_FC_eta"].iloc[1] == pytest.approx(4.0)
        assert event_list["genp_H2_FC_eta"].iloc[1] == pytest.approx(2.0)

    def test_all_eight_columns_present(self):
        tree_df = self._make_tree_df(
            pt1=[100.0],
            pt2=[90.0],
            eta1=[0.5],
            eta2=[0.6],
            phi1=[0.7],
            phi2=[0.8],
            m1=[125.0],
            m2=[125.0],
        )
        event_list = pd.DataFrame(index=range(1))
        _add_ordered_gen_higgs_columns(event_list, tree_df)
        expected = {
            "genp_H1_FC_pt",
            "genp_H1_FC_eta",
            "genp_H1_FC_phi",
            "genp_H1_FC_m",
            "genp_H2_FC_pt",
            "genp_H2_FC_eta",
            "genp_H2_FC_phi",
            "genp_H2_FC_m",
        }
        assert expected.issubset(set(event_list.columns))

    def test_no_op_when_columns_missing(self):
        tree_df = pd.DataFrame({"unrelated": [1.0]})
        event_list = pd.DataFrame(index=range(1))
        _add_ordered_gen_higgs_columns(event_list, tree_df)
        assert "genp_H1_FC_pt" not in event_list.columns


# ---------------------------------------------------------------------------
# _build_event_list_frame
# ---------------------------------------------------------------------------
class TestBuildEventListFrame:
    @pytest.fixture
    def base_tree_df(self):
        """Minimal DataFrame with all required base columns."""
        n = 3
        data = {col: np.arange(n, dtype=float) for col in EVENTLIST_BASE_COLUMNS}
        data["Category"] = np.array([1, 2, 0])  # GGF bin1, bin2, VBF
        return pd.DataFrame(data)

    def test_base_columns_present(self, base_tree_df):
        result = _build_event_list_frame(base_tree_df, key="data")
        for col in EVENTLIST_BASE_COLUMNS:
            assert col in result.columns

    def test_ggf_category_mapping(self, base_tree_df):
        result = _build_event_list_frame(base_tree_df, key="data")
        np.testing.assert_array_equal(result["ggf_category"].to_numpy(), [1, 2, 0])

    def test_vbf_category_flag(self, base_tree_df):
        result = _build_event_list_frame(base_tree_df, key="data")
        np.testing.assert_array_equal(result["VBF_CATEGORY"].to_numpy(), [False, False, True])

    def test_data_sample_has_no_signal_columns(self, base_tree_df):
        result = _build_event_list_frame(base_tree_df, key="data")
        for col in ["kl", "k2v", "lumiwgt", "xsecWeight", "genWeight"]:
            assert col not in result.columns

    def test_signal_sample_includes_weight_columns(self, base_tree_df):
        base_tree_df["lumiwgt"] = 59.8
        base_tree_df["xsecWeight"] = 0.01
        base_tree_df["genWeight"] = 1.0
        base_tree_df["kl"] = 1.0
        base_tree_df["k2v"] = 1.0
        result = _build_event_list_frame(base_tree_df, key="hh4b")
        for col in ["lumiwgt", "xsecWeight", "genWeight", "kl", "k2v"]:
            assert col in result.columns

    def test_signal_sample_includes_gen_higgs_if_available(self, base_tree_df):
        base_tree_df["lumiwgt"] = 59.8
        base_tree_df["xsecWeight"] = 0.01
        base_tree_df["genWeight"] = 1.0
        base_tree_df["kl"] = 1.0
        base_tree_df["k2v"] = 1.0
        for kin in ["Pt", "Eta", "Phi", "Mass"]:
            base_tree_df[("GenHiggs" + kin, 0)] = np.array([100.0, 200.0, 150.0])
            base_tree_df[("GenHiggs" + kin, 1)] = np.array([90.0, 80.0, 70.0])
        result = _build_event_list_frame(base_tree_df, key="hh4b")
        assert "genp_H1_FC_pt" in result.columns
        assert "GenHiggsMass1" in result.columns
        assert "GenHiggsMass2" in result.columns


# ---------------------------------------------------------------------------
# _compute_all_hist_samples
# ---------------------------------------------------------------------------
class TestComputeAllHistSamples:
    def test_basic(self):
        sample_keys = ["data", "hh4b", "ttbar"]
        sig_keys = ["hh4b"]

        class FakeShift:
            def __init__(self, samples):
                self.samples = samples

        weight_shifts = {"pileup": FakeShift(samples=["ttbar"])}
        result = _compute_all_hist_samples(sample_keys, sig_keys, weight_shifts)
        # Base samples
        assert "data" in result
        assert "hh4b" in result
        assert "ttbar" in result
        # TXbb shifts for signal
        assert "hh4b_txbb_down" in result
        assert "hh4b_txbb_up" in result
        # Weight shifts
        assert "ttbar_pileup_down" in result
        assert "ttbar_pileup_up" in result

    def test_no_shifts(self):
        result = _compute_all_hist_samples(["data", "hh4b"], ["hh4b"], {})
        assert result == ["data", "hh4b", "hh4b_txbb_down", "hh4b_txbb_up"]
