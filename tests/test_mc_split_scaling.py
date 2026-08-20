"""Tests for sharing the 2024 MC between the 2024 and 2025 templates.

TEMPORARY(mc-sharing) -- delete once 2025 has MC of its own.

2025 has no dedicated MC, so under ``--split-shared-mc`` the 2024 sample is split 50/50 by
event parity: half builds the 2024 templates, half the 2025 templates, each scaled back up
to its year's luminosity. Keeping the halves disjoint is what lets the two years enter the
same fit without their MC statistical uncertainties being correlated.

When 2025 MC exists, ``grep -rn "TEMPORARY(mc-sharing)"`` lists every site to remove:
this file, the two helpers in PostProcess, the CLI flag, and its call site.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hist")
pytest.importorskip("mplhep")
pytest.importorskip("xgboost")

from HH4b import hh_vars
from HH4b.postprocessing import PostProcess as pp

# Read from hh_vars rather than hardcoded: 2025 is still accumulating luminosity, so a
# pinned ratio would turn a routine LUMI update into a test failure.
LUMI_RATIO = hh_vars.LUMI["2025"] / hh_vars.LUMI["2024"]

STANDARD_YEARS = ["2022", "2022EE", "2023", "2023BPix"]


class TestMCSplitConfig:
    """The (year, --split-shared-mc) -> (source MC, half, weight scale) table."""

    @pytest.mark.parametrize("split_shared_mc", [False, True])
    def test_standard_years_never_split_or_rescale(self, split_shared_mc):
        """Years with their own MC are untouched, and the flag must not reach them."""
        for year in STANDARD_YEARS:
            cfg = pp.get_mc_split_config(year, split_shared_mc=split_shared_mc)
            assert cfg.source_year == year
            assert cfg.split_half is None
            assert cfg.weight_scale == 1.0

    def test_2024_without_sharing_uses_full_mc(self):
        """Default: nothing is reserved for 2025, so 2024 keeps every event at face value."""
        cfg = pp.get_mc_split_config("2024", split_shared_mc=False)
        assert cfg.source_year == "2024"
        assert cfg.split_half is None
        assert cfg.weight_scale == 1.0

    def test_2024_with_sharing_takes_half_and_compensates(self):
        """Sharing on: 2024 keeps parity half 0, x2 to recover the full 2024 yield."""
        cfg = pp.get_mc_split_config("2024", split_shared_mc=True)
        assert cfg.source_year == "2024"
        assert cfg.split_half == 0
        assert cfg.weight_scale == pytest.approx(2.0)

    def test_2025_with_sharing_takes_other_half_and_rescales_to_2025_lumi(self):
        """Sharing on: 2025 keeps parity half 1, x2 for the split and x lumi ratio for the year.

        The x2 is the factor that was missing before: without it the 2025 templates came out
        at exactly half their correct normalization.
        """
        cfg = pp.get_mc_split_config("2025", split_shared_mc=True)
        assert cfg.source_year == "2024"
        assert cfg.split_half == 1
        assert cfg.weight_scale == pytest.approx(2.0 * LUMI_RATIO)
        # guard against a regression to the un-compensated value
        assert cfg.weight_scale != pytest.approx(LUMI_RATIO)

    def test_2025_without_sharing_is_an_error(self):
        """There is no 2025 MC. Refusing to run beats silently inventing a normalization."""
        with pytest.raises(ValueError, match="split-shared-mc"):
            pp.get_mc_split_config("2025", split_shared_mc=False)

    def test_2025_lumi_falls_back_to_sum_of_sub_eras(self, monkeypatch):
        """If LUMI has no aggregate '2025' key, the sub-era luminosities are summed."""
        lumi = {k: v for k, v in hh_vars.LUMI.items() if k != "2025"}
        monkeypatch.setattr(hh_vars, "LUMI", lumi)
        expected = sum(v for k, v in lumi.items() if k.startswith("2025"))
        cfg = pp.get_mc_split_config("2025", split_shared_mc=True)
        assert cfg.weight_scale == pytest.approx(2.0 * expected / lumi["2024"])


class TestMCSplitMask:
    """The parity mask that assigns each event to exactly one of the two years."""

    @staticmethod
    def _event_ids(n=10_000, seed=0):
        rng = np.random.default_rng(seed)
        return (
            rng.integers(1, 400_000, size=n, dtype=np.int64),
            rng.integers(1, 3_000, size=n, dtype=np.int64),
            rng.integers(1, 2**40, size=n, dtype=np.int64),
        )

    def test_split_mask_is_a_partition(self):
        """Every event lands in exactly one half: disjoint and covering."""
        run, lumi, event = self._event_ids()
        half_0 = pp.mc_split_mask(run, lumi, event, 0)
        half_1 = pp.mc_split_mask(run, lumi, event, 1)
        assert not np.any(half_0 & half_1), "an event was assigned to both years"
        assert np.all(half_0 | half_1), "an event was assigned to neither year"

    def test_split_mask_accepts_skimmer_dtypes(self):
        """The skimmer writes run/lumi as uint32 and event as uint64; no cast may be needed."""
        run = np.array([1, 2], dtype=np.uint32)
        lumi = np.array([3, 4], dtype=np.uint32)
        event = np.array([2**63 + 1, 2**63 + 2], dtype=np.uint64)  # odd, even
        # (1 + 3 + odd) % 2 == 1 ; (2 + 4 + even) % 2 == 0
        np.testing.assert_array_equal(
            pp.mc_split_mask(run, lumi, event, 1), np.array([True, False])
        )

    def test_split_mask_rejects_non_integer_ids(self):
        """A float column cannot represent these event numbers; fail loudly, never guess."""
        ids = np.array([1.0, 2.0])
        with pytest.raises(TypeError, match="integer"):
            pp.mc_split_mask(ids, ids, ids, 0)


class TestSplitClosure:
    """Config and mask together must reproduce the full sample's yield."""

    def test_split_halves_close_to_the_full_sample_yield(self):
        """Constructed with an exactly even split, so this is an identity, not a statistic."""
        n = 10_000
        run = np.zeros(n, dtype=np.int64)
        lumi = np.zeros(n, dtype=np.int64)
        event = np.arange(n, dtype=np.int64)  # alternating parity -> exactly half each
        weights = np.ones(n)

        cfg_2024 = pp.get_mc_split_config("2024", split_shared_mc=True)
        cfg_2025 = pp.get_mc_split_config("2025", split_shared_mc=True)
        half_2024 = pp.mc_split_mask(run, lumi, event, cfg_2024.split_half)
        half_2025 = pp.mc_split_mask(run, lumi, event, cfg_2025.split_half)

        total = weights.sum()
        # unscaled, the halves partition the sample exactly
        assert weights[half_2024].sum() + weights[half_2025].sum() == total
        # rescaled, 2024 recovers the full 2024 yield and 2025 lands at the 2025 lumi
        assert (weights[half_2024] * cfg_2024.weight_scale).sum() == pytest.approx(total)
        assert (weights[half_2025] * cfg_2025.weight_scale).sum() == pytest.approx(
            total * LUMI_RATIO
        )
