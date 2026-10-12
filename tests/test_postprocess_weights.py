"""Tests for weight calculation functions in PostProcess.py.

These tests verify the wrapper logic of calculate_trigger_weights and
calculate_txbb_weights by mocking the underlying corrections module.
This ensures that the refactoring of load_process_run3_samples (e.g.
early deletion of events_dict) does not break weight composition.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from HH4b.postprocessing.PostProcess import (
    calculate_trigger_weights,
    calculate_txbb_weights,
)


# ---------------------------------------------------------------------------
# calculate_trigger_weights
# ---------------------------------------------------------------------------
class TestCalculateTriggerWeights:
    def test_data_returns_ones(self):
        """Data samples should get unit trigger weights (no SF applied)."""
        n = 5
        w, w_up, w_dn = calculate_trigger_weights(
            events_dict={},
            key="data",
            year="2022",
            txbb="glopart-v2",
            trigger_region="QCD",
            n_events=n,
        )
        np.testing.assert_array_equal(w, np.ones(n))
        np.testing.assert_array_equal(w_up, np.ones(n))
        np.testing.assert_array_equal(w_dn, np.ones(n))

    @patch("HH4b.postprocessing.PostProcess.corrections.trigger_SF")
    def test_mc_applies_sf(self, mock_trigger_sf):
        """MC samples should call corrections.trigger_SF and propagate the uncertainty."""
        n = 3
        sf = np.array([1.1, 0.9, 1.0])
        total = 100.0
        total_err = 5.0
        mock_trigger_sf.return_value = (sf, None, total, total_err)

        w, w_up, w_dn = calculate_trigger_weights(
            events_dict=MagicMock(),
            key="ttbar",
            year="2022",
            txbb="glopart-v2",
            trigger_region="QCD",
            n_events=n,
        )

        np.testing.assert_array_equal(w, sf)
        expected_up = sf * (1 + total_err / total)
        expected_dn = sf * (1 - total_err / total)
        np.testing.assert_allclose(w_up, expected_up)
        np.testing.assert_allclose(w_dn, expected_dn)
        mock_trigger_sf.assert_called_once()

    @patch("HH4b.postprocessing.PostProcess.corrections.trigger_SF")
    def test_signal_also_applies_sf(self, mock_trigger_sf):
        """Signal samples (key != 'data') should also get trigger SF."""
        n = 2
        sf = np.array([1.05, 0.95])
        mock_trigger_sf.return_value = (sf, None, 50.0, 2.0)

        w, _, _ = calculate_trigger_weights(
            events_dict=MagicMock(),
            key="hh4b",
            year="2023",
            txbb="glopart-v2",
            trigger_region="QCD",
            n_events=n,
        )
        np.testing.assert_array_equal(w, sf)


# ---------------------------------------------------------------------------
# calculate_txbb_weights
# ---------------------------------------------------------------------------
class TestCalculateTxbbWeights:
    def _make_bdt_events(self, n, txbb1=0.95, txbb2=0.90, pt1=400.0, pt2=300.0):
        return pd.DataFrame(
            {
                "H1TXbb": np.full(n, txbb1),
                "H2TXbb": np.full(n, txbb2),
                "H1Pt": np.full(n, pt1),
                "H2Pt": np.full(n, pt2),
            }
        )

    @patch("HH4b.postprocessing.PostProcess.corrections.restrict_SF")
    def test_signal_applies_two_jet_sfs(self, mock_restrict_sf):
        """Signal samples (jets [1, 2]) should multiply two restrict_SF calls."""
        n = 4
        sf_jet1 = np.array([1.1, 1.0, 0.9, 1.05])
        sf_jet2 = np.array([1.0, 1.0, 1.1, 0.95])
        mock_restrict_sf.side_effect = [sf_jet1, sf_jet2]

        bdt_events = self._make_bdt_events(n)
        TXbb_wps = {"HP": [0.945, 1.0], "MP": [0.85, 0.945]}
        TXbb_pt_corr_bins = {"HP": [200, 400, 600], "MP": [200, 400, 600]}

        result = calculate_txbb_weights(
            bdt_events,
            key="hh4b",
            txbb_sf={"nominal": MagicMock()},
            TXbb_wps=TXbb_wps,
            TXbb_pt_corr_bins=TXbb_pt_corr_bins,
            n_events=n,
        )
        expected = sf_jet1 * sf_jet2
        np.testing.assert_allclose(result, expected)
        assert mock_restrict_sf.call_count == 2

    @patch("HH4b.postprocessing.PostProcess.corrections.restrict_SF")
    def test_data_returns_ones(self, mock_restrict_sf):
        """Data has no jets for TXbb SF (get_jets_for_txbb_sf returns []),
        so the weight should stay at 1.0."""
        n = 3
        bdt_events = self._make_bdt_events(n)
        TXbb_wps = {"HP": [0.945, 1.0]}
        TXbb_pt_corr_bins = {"HP": [200, 400, 600]}

        result = calculate_txbb_weights(
            bdt_events,
            key="data",
            txbb_sf={"nominal": MagicMock()},
            TXbb_wps=TXbb_wps,
            TXbb_pt_corr_bins=TXbb_pt_corr_bins,
            n_events=n,
        )
        np.testing.assert_array_equal(result, np.ones(n))
        mock_restrict_sf.assert_not_called()

    @patch("HH4b.postprocessing.PostProcess.corrections.restrict_SF")
    def test_single_jet_process(self, mock_restrict_sf):
        """Processes like novhhtobb only apply SF to jet 1."""
        n = 2
        sf_jet1 = np.array([1.2, 0.8])
        mock_restrict_sf.return_value = sf_jet1

        bdt_events = self._make_bdt_events(n)
        TXbb_wps = {"HP": [0.945, 1.0]}
        TXbb_pt_corr_bins = {"HP": [200, 600]}

        result = calculate_txbb_weights(
            bdt_events,
            key="novhhtobb",
            txbb_sf={"nominal": MagicMock()},
            TXbb_wps=TXbb_wps,
            TXbb_pt_corr_bins=TXbb_pt_corr_bins,
            n_events=n,
        )
        np.testing.assert_allclose(result, sf_jet1)
        assert mock_restrict_sf.call_count == 1
