"""Structural sanity checks for combine datacards.

These tests run against the shipped fixture cards under
``tests/fixtures/cards/sm_only/``. They catch the kind of mistake that
silently produces wrong physics rather than a noisy error — most
prominently the BSM-cocktail regression where a card meant for vanilla
combine accidentally includes the 12 BSM coupling points alongside SM
HH, turning the limit on ``r`` into a BSM-weighted sum.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cards" / "sm_only"
COMBINED_CARD = FIXTURE_DIR / "combined.txt"
PER_BIN_CARDS = [
    FIXTURE_DIR / f"{name}.txt" for name in ("passbin1", "passbin2", "passbin3", "passvbf", "fail")
]
ALL_CARDS = [COMBINED_CARD, *PER_BIN_CARDS]

# Process names that should never appear as backgrounds (negative indices)
# in an SM-only card; any signal with a name outside this set is a regression.
EXPECTED_SIGNAL_NAMES = {"ggHH", "qqHH"}


def _read_card(path: Path) -> dict:
    """Parse a combine datacard into a small dict.

    We track only the bits the tests need: bins, observations, the process
    block (names + indices), the rate block, and the constrained-NP lines.
    Comments, the imax/jmax/kmax header, and shape directives are ignored.
    """
    obs_bins: list[str] = []
    observations: list[float] = []
    process_bins: list[str] = []
    proc_names: list[str] = []
    proc_indices: list[int] = []
    rates: list[float] = []
    np_lines: list[tuple[str, str, list[str]]] = []

    bin_seen = 0
    process_seen = 0

    for raw in path.read_text().splitlines():
        toks = raw.split()
        if not toks:
            continue
        head = toks[0]
        if head.startswith("---") or head in {"imax", "jmax", "kmax", "shapes"}:
            continue
        if head == "observation":
            observations = [float(x) for x in toks[1:]]
            continue
        if head == "bin":
            bin_seen += 1
            if bin_seen == 1:
                obs_bins = toks[1:]
            else:
                process_bins = toks[1:]
            continue
        if head == "process":
            process_seen += 1
            if process_seen == 1:
                proc_names = toks[1:]
            else:
                proc_indices = [int(x) for x in toks[1:]]
            continue
        if head == "rate":
            rates = [float(x) for x in toks[1:]]
            continue
        # Constrained-NP line: <name> <type> <values...>
        if len(toks) >= 2 and toks[1] in {"lnN", "shape", "shape?", "shapeN"}:
            np_lines.append((toks[0], toks[1], toks[2:]))

    return {
        "path": path,
        "obs_bins": obs_bins,
        "observations": observations,
        "process_bins": process_bins,
        "proc_names": proc_names,
        "proc_indices": proc_indices,
        "rates": rates,
        "np_lines": np_lines,
    }


def test_combined_card_has_only_sm_signals() -> None:
    """Combined card holds ggHH + qqHH as its only signals.

    Catches the BSM-cocktail regression: a card meant for vanilla combine
    should not carry the 12 BSM coupling points. If it does, the unique
    signal-name set grows from {ggHH, qqHH} to 14 entries, and the limit
    on `r` ceases to mean "SM HH signal strength".
    """
    card = _read_card(COMBINED_CARD)
    signal_names = {name for name, idx in zip(card["proc_names"], card["proc_indices"]) if idx <= 0}
    assert signal_names == EXPECTED_SIGNAL_NAMES


@pytest.mark.parametrize(
    ("card_path", "expected_obs"),
    [
        (FIXTURE_DIR / "passbin1.txt", 84),
        (FIXTURE_DIR / "passbin2.txt", 476),
        (FIXTURE_DIR / "passbin3.txt", 4779),
        (FIXTURE_DIR / "passvbf.txt", 115),
        (FIXTURE_DIR / "fail.txt", 5000000),
    ],
)
def test_per_bin_observation(card_path: Path, expected_obs: int) -> None:
    """Each per-bin card reports the expected observation."""
    card = _read_card(card_path)
    assert card["observations"] == [float(expected_obs)]


def test_combined_observation_matches_per_bin() -> None:
    """combined.txt's observation row must agree with the per-bin cards.

    A drift here means the combined card was regenerated from a different
    set of inputs than the per-bin cards in this directory — i.e. they're
    out of sync.
    """
    combined = _read_card(COMBINED_CARD)
    per_bin_obs: dict[str, float] = {}
    for path in PER_BIN_CARDS:
        bin_card = _read_card(path)
        per_bin_obs[bin_card["obs_bins"][0]] = bin_card["observations"][0]
    combined_obs = dict(zip(combined["obs_bins"], combined["observations"]))
    assert combined_obs == per_bin_obs


@pytest.mark.parametrize("card_path", ALL_CARDS)
def test_rate_and_process_columns_align(card_path: Path) -> None:
    """Rate row, process names, and process indices must all be the same width."""
    card = _read_card(card_path)
    assert len(card["rates"]) == len(card["proc_indices"]) == len(card["proc_names"])


@pytest.mark.parametrize("card_path", ALL_CARDS)
def test_np_columns_match_process_columns(card_path: Path) -> None:
    """Every NP line must have one value per process column."""
    card = _read_card(card_path)
    n_processes = len(card["proc_indices"])
    for name, _, values in card["np_lines"]:
        assert len(values) == n_processes, (
            f"{card_path.name}: NP {name!r} has {len(values)} value columns, "
            f"expected {n_processes}"
        )


@pytest.mark.parametrize("card_path", ALL_CARDS)
def test_no_duplicate_nps(card_path: Path) -> None:
    """Each constrained NP must be declared exactly once.

    Combine will accept duplicate declarations but downstream tooling
    (impacts, plotting) typically does not, and the resulting confusion
    is hard to diagnose. Catching it here is cheap.
    """
    card = _read_card(card_path)
    np_names = [name for name, _, _ in card["np_lines"]]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in np_names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    assert duplicates == set(), f"{card_path.name}: duplicate NPs {duplicates}"


def test_combined_has_expected_process_block_width() -> None:
    """combined.txt's process block is 5 bins x 6 processes = 30 columns."""
    combined = _read_card(COMBINED_CARD)
    assert len(combined["proc_indices"]) == 30


@pytest.mark.parametrize("card_path", PER_BIN_CARDS)
def test_per_bin_has_six_processes(card_path: Path) -> None:
    """Each per-bin card declares 2 signals + 4 backgrounds = 6 processes."""
    card = _read_card(card_path)
    assert len(card["proc_indices"]) == 6
