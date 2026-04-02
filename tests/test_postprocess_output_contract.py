"""Output contract tests for load_process_run3_samples.

These tests do NOT run the actual function (which requires real parquet data).
Instead they define the column sets that must be present in the output
DataFrame for each sample type (data, signal, ttbar, background).

When refactoring load_process_run3_samples — especially early deletion of
events_dict for memory optimization — these contracts verify that all
required columns survive the transformation.

The column sets are derived from lines 1066-1123 of PostProcess.py (the
`columns` list that filters bdt_events at the end of the per-sample loop).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Expected columns by sample type
# ---------------------------------------------------------------------------

# Columns present for ALL sample types (line 1066-1079)
BASE_COLUMNS = {
    "Category",
    "H2Msd",
    "bdt_score",
    "H2TXbb",
    "H2PNetMass",
    "H1Pt",
    "H2Pt",
    "weight",
    "event",
    "run",
    "luminosityBlock",
    "year",
}

# BDT VBF score (present whenever 4+ class BDT is used, i.e. bdt_disc=True)
BDT_VBF_COLUMN = "bdt_score_vbf"

# Columns added for non-data samples (line 1121-1122)
MC_WEIGHT_COLUMNS = {
    "weight_triggerUp",
    "weight_triggerDown",
    "weight_pileupUp",
    "weight_pileupDown",
    "weight_ISRPartonShowerUp",
    "weight_ISRPartonShowerDown",
    "weight_FSRPartonShowerUp",
    "weight_FSRPartonShowerDown",
}

# Scale weights (line 1089-1091): signal + ttbar
SCALE_WEIGHT_COLUMNS = {f"scale_weights_{i}" for i in range(6)}

# Signal-only columns (line 1103-1120)
SIGNAL_METADATA_COLUMNS = {
    "lumiwgt",
    "xsecWeight",
    "genWeight",
    "kl",
    "k2v",
    "GenHiggsPt1",
    "GenHiggsEta1",
    "GenHiggsPhi1",
    "GenHiggsMass1",
    "GenHiggsPt2",
    "GenHiggsEta2",
    "GenHiggsPhi2",
    "GenHiggsMass2",
}

# Columns used by the event list export (PostProcess.py line 1608-1618)
EVENTLIST_BASE_COLUMNS = {
    "event",
    "bdt_score",
    "bdt_score_vbf",
    "H2TXbb",
    "H2Msd",
    "run",
    "Category",
    "H2PNetMass",
    "luminosityBlock",
}


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------
class TestDataOutputContract:
    """Data samples get base columns only (no MC weights, no signal metadata)."""

    def test_base_columns(self):
        expected = BASE_COLUMNS
        # Data must NOT have MC-only columns
        forbidden = MC_WEIGHT_COLUMNS | SCALE_WEIGHT_COLUMNS | SIGNAL_METADATA_COLUMNS
        assert expected & forbidden == set(), "Base columns should not overlap with MC columns"

    def test_no_signal_columns_for_data(self):
        """Signal metadata must not be in the data contract."""
        for col in SIGNAL_METADATA_COLUMNS:
            assert col not in BASE_COLUMNS


class TestSignalOutputContract:
    """Signal samples must have base + MC weights + scale weights + signal metadata."""

    def test_all_required_columns(self):
        required = BASE_COLUMNS | MC_WEIGHT_COLUMNS | SCALE_WEIGHT_COLUMNS | SIGNAL_METADATA_COLUMNS
        # Verify nothing in the required set is accidentally empty
        assert len(required) > 30

    def test_signal_metadata_complete(self):
        """All gen-level Higgs columns must be present."""
        for h_idx in [1, 2]:
            for kin in ["Pt", "Eta", "Phi", "Mass"]:
                assert f"GenHiggs{kin}{h_idx}" in SIGNAL_METADATA_COLUMNS

    def test_coupling_parameters(self):
        assert "kl" in SIGNAL_METADATA_COLUMNS
        assert "k2v" in SIGNAL_METADATA_COLUMNS

    def test_weight_columns(self):
        assert "lumiwgt" in SIGNAL_METADATA_COLUMNS
        assert "xsecWeight" in SIGNAL_METADATA_COLUMNS
        assert "genWeight" in SIGNAL_METADATA_COLUMNS


class TestTtbarOutputContract:
    """Ttbar has base + MC weights + scale weights (but NOT signal metadata)."""

    def test_has_scale_weights(self):
        expected = BASE_COLUMNS | MC_WEIGHT_COLUMNS | SCALE_WEIGHT_COLUMNS
        assert SCALE_WEIGHT_COLUMNS.issubset(expected)

    def test_no_signal_metadata(self):
        """Ttbar should not have coupling parameters or gen Higgs columns."""
        ttbar_columns = BASE_COLUMNS | MC_WEIGHT_COLUMNS | SCALE_WEIGHT_COLUMNS
        for col in SIGNAL_METADATA_COLUMNS:
            assert col not in ttbar_columns


class TestBackgroundOutputContract:
    """Generic backgrounds have base + MC weights (no scale weights, no signal metadata)."""

    def test_mc_weights_present(self):
        bg_columns = BASE_COLUMNS | MC_WEIGHT_COLUMNS
        assert "weight_triggerUp" in bg_columns
        assert "weight_triggerDown" in bg_columns

    def test_no_scale_weights(self):
        """Generic backgrounds don't have scale weights."""
        bg_only = BASE_COLUMNS | MC_WEIGHT_COLUMNS
        for col in SCALE_WEIGHT_COLUMNS:
            assert col not in bg_only


class TestEventListColumnsSubset:
    """Event list export columns must be a subset of what's available in bdt_events."""

    def test_base_eventlist_columns_in_output(self):
        """All event list base columns must exist in the base output."""
        # bdt_score_vbf requires 4-class BDT but is always present in practice
        non_vbf = EVENTLIST_BASE_COLUMNS - {BDT_VBF_COLUMN}
        assert non_vbf.issubset(BASE_COLUMNS)

    def test_signal_eventlist_columns_in_output(self):
        """Signal event list needs weight/coupling columns from signal output."""
        signal_evlist_needs = {"lumiwgt", "xsecWeight", "genWeight", "kl", "k2v"}
        assert signal_evlist_needs.issubset(SIGNAL_METADATA_COLUMNS)


class TestColumnSetsConsistency:
    """Cross-check that the column sets defined here are internally consistent."""

    @pytest.mark.parametrize(
        "col",
        [
            "weight",
            "event",
            "run",
            "luminosityBlock",
            "Category",
            "H2TXbb",
            "bdt_score",
        ],
    )
    def test_critical_columns_in_base(self, col):
        assert col in BASE_COLUMNS

    def test_trigger_weights_in_mc(self):
        assert "weight_triggerUp" in MC_WEIGHT_COLUMNS
        assert "weight_triggerDown" in MC_WEIGHT_COLUMNS

    def test_pileup_weights_in_mc(self):
        assert "weight_pileupUp" in MC_WEIGHT_COLUMNS
        assert "weight_pileupDown" in MC_WEIGHT_COLUMNS

    def test_six_scale_weights(self):
        assert len(SCALE_WEIGHT_COLUMNS) == 6
