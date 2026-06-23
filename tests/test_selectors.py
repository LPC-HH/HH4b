"""Unit coverage for ``HH4b.utils.check_selector``.

``check_selector`` is the sample-name matcher used throughout postprocessing to
decide whether a sample belongs to a group (see ``hh_vars.common_samples_bg`` and
the trigger-study code). It is pure and string-only, but its matching rules are
easy to get subtly wrong, so they are pinned here:

* trailing ``?`` -> exact match   ("ZZ?"      matches only the sample "ZZ")
* leading  ``*`` -> substring      ("*TuneCP5" matches any sample containing "TuneCP5")
* otherwise      -> prefix match   ("QCD_HT-"  matches anything starting with "QCD_HT-")

A list/tuple of selectors matches if ANY entry matches.
"""

from __future__ import annotations

import pytest

from HH4b.utils import check_selector


@pytest.mark.parametrize(
    ("sample", "selector", "expected"),
    [
        # prefix match — the common case (QCD_HT- / QCD-4Jets_HT, ...)
        ("QCD_HT-100to200", "QCD_HT-", True),
        ("QCD-4Jets_HT-200to400", "QCD-4Jets_HT", True),
        ("TTto2L2Nu", "QCD_HT-", False),
        ("QCD_HT-100to200", "QCD_HT-100to200", True),  # whole name is a valid prefix
        ("QCD", "QCD_HT-", False),  # selector longer than the sample
        # exact match via a trailing "?"
        ("ZZ", "ZZ?", True),
        ("ZZto2L2Nu", "ZZ?", False),  # "?" demands equality, not a prefix
        ("ZZ", "ZZZ?", False),
        # without "?", the same text behaves as a prefix
        ("ZZto2L2Nu", "ZZ", True),
        # substring match via a leading "*"
        ("WJetsToLNu_TuneCP5_13TeV", "*TuneCP5", True),
        ("WJetsToLNu_TuneCP5_13TeV", "*TuneXYZ", False),
        ("TuneCP5_leading", "*TuneCP5", True),  # substring anywhere, including the start
    ],
)
def test_check_selector_single(sample: str, selector: str, expected: bool) -> None:
    assert check_selector(sample, selector) is expected


def test_check_selector_list_matches_if_any_entry_matches() -> None:
    assert check_selector("QCD_HT-100to200", ["TTbar", "QCD_HT-"])
    assert check_selector("QCD_HT-100to200", ("TTbar", "QCD_HT-"))
    assert not check_selector("DYJetsToLL", ["TTbar", "QCD_HT-"])


def test_check_selector_empty_selector_list_never_matches() -> None:
    assert not check_selector("anything", [])


def test_check_selector_mixes_pattern_kinds_in_one_list() -> None:
    selectors = ["QCD_HT-", "ZZ?", "*TuneCP5"]
    assert check_selector("QCD_HT-50to100", selectors)  # prefix
    assert check_selector("ZZ", selectors)  # exact
    assert check_selector("ggHH_TuneCP5", selectors)  # substring
    assert not check_selector("SingleMuon", selectors)
