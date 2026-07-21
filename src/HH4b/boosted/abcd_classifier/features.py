"""
Feature lists and extraction.  See notes/ABCDnn.md §1.4.

Rule for inclusion: no H1/H2 mass and no H1/H2 tagger score.
Otherwise the classifier could trivially shortcut the (TXbb, mass) ABCD
region from the inputs.

  - τ₃/τ₂ (T32) ratios are kept: substructure observables, not tagger
    outputs.
  - VBFjjMass is kept: VBF dijet mass is independent of the bb-jet system,
    so it does not leak the ABCD axis (h1.M / h1 TXbb).
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

# BDT config module providing ``bdt_dataframe(events, key_map)``.  Hard-coded
# for v0; if you need a different BDT config the import can be parametrized
# via a function argument.
_BDT_CONFIG = "HH4b.boosted.bdt_trainings_run3.v13_glopartv3"

# Strict feature set.  Excludes any H1/H2 mass or tagger score.
STRICT_FEATURES: list[str] = [
    "HHPt",
    "HHeta",
    "MET",
    "H1T32",
    "H2T32",
    "H1Pt",
    "H2Pt",
    "H1eta",
    "H1Pt_H2Pt",
    "VBFjjMass",
    "VBFjjDeltaEta",
    "H1AK4JetAway1dR",
    "H2AK4JetAway2dR",
    # Hidden — H{1,2} mass / tagger leaks of the ABCD axis (h1.M, H1Xbb).
    # "H2AK4JetAway2mass",   # M(h2 + AK4 away) — uses h2.M
]

# Literal feature set was originally STRICT + four mass-leak variables for
# ablation.  All four violate the no-H1/H2-mass rule and are hidden below;
# LITERAL therefore reduces to STRICT.  Kept here for record.
LITERAL_FEATURES: list[str] = STRICT_FEATURES + [
    # Hidden — all four pull in h1.M = bbFatJetParT3massX2p[0] (ABCD axis).
    # "HHmass",              # (h1+h2).M
    # "H1Pt_HHmass",         # h1.pt / HHmass
    # "H2Pt_HHmass",         # h2.pt / HHmass
    # "H1AK4JetAway1mass",   # M(h1 + AK4 away) — uses h1.M
]


# Era one-hot conditioning.  A ``*_era`` feature set appends one column per
# era so the classifier can be year-aware (shared backbone + year-dependent
# input).  Only meaningful for multi-era runs; for a single-era run the
# one-hot is constant (a dead input).  The per-event era is read from the
# ``("era", "")`` column written by prep_bdt_inference_pickles.py.
ERA_ORDER: list[str] = ["2022", "2022EE", "2023", "2023BPix"]
ERA_FEATURES: list[str] = [f"era_{e}" for e in ERA_ORDER]


def _uses_era(feature_set: str) -> bool:
    return feature_set.endswith("_era")


def _base_feature_set(feature_set: str) -> str:
    return feature_set[:-4] if _uses_era(feature_set) else feature_set


def feature_columns(feature_set: str) -> list[str]:
    """Return the feature list for 'strict'/'literal', optionally '*_era'.

    A trailing ``_era`` (e.g. ``strict_era``) appends one-hot era columns
    (``era_2022``, … in ``ERA_ORDER``) to the base set.
    """
    base = _base_feature_set(feature_set)
    if base == "strict":
        cols = list(STRICT_FEATURES)
    elif base == "literal":
        cols = list(LITERAL_FEATURES)
    else:
        raise ValueError(f"unknown feature_set: {feature_set!r}")
    if _uses_era(feature_set):
        cols = cols + ERA_FEATURES
    return cols


def _era_one_hot(events: pd.DataFrame) -> np.ndarray:
    """One-hot encode the per-event era (column order = ``ERA_ORDER``)."""
    if ("era", "") in events.columns:
        era = events[("era", "")]
    elif "era" in events.columns:
        era = events["era"]
    else:
        raise KeyError(
            "feature_set '*_era' requires an 'era' column in the events; "
            "build the pickle cache with prep_bdt_inference_pickles.py."
        )
    era = np.asarray(era).reshape(-1)
    oh = np.zeros((len(era), len(ERA_ORDER)), dtype=np.float32)
    for i, e in enumerate(ERA_ORDER):
        oh[:, i] = era == e
    return oh


# Magnitude beyond which a feature value is treated as a padding sentinel.
# Matches the threshold used in
# ``src/HH4b/neural_network/prepare_data.py:cleanup_events``.  Anything past
# this magnitude (or NaN/inf) is replaced via ``replace_pad_vals``.
#
# Why a threshold rather than ``== PAD_VAL``: derived features (ratios like
# ``H1Pt_HHmass``, boosted-frame masses, etc.) computed from a padded input
# may take large finite values that are not exactly -99999.  Threshold
# catches both the sentinel and its propagated descendants.
PAD_THRESHOLD = 50000.0


def replace_pad_vals(
    X: np.ndarray, replacement: float = 0.0, threshold: float = PAD_THRESHOLD
) -> np.ndarray:
    """
    Replace padding sentinels (NaN/inf or ``|x| > threshold``) with a finite
    value, suitable for feeding into an MLP that has no per-event mask.

    Defaults to ``replacement=0.0`` for training: zero is a benign value
    after standardization and avoids the MLP chasing ``-99999`` outliers.
    Diagnostic callers may pass a different replacement (e.g. ``np.nan``)
    to keep the missingness visible in plots.
    """
    bad = ~np.isfinite(X) | (np.abs(X) > threshold)
    if not bad.any():
        return X
    out = X.copy()
    out[bad] = replacement
    return out


def extract_features(events: pd.DataFrame, feature_set: str) -> np.ndarray:
    """
    Compute the ABCDnn classifier features for ``events``.

    Internally calls the BDT config's ``bdt_dataframe(events, key_map)``
    (defaulting to v13_glopartv3) to recompute the full set of BDT inputs,
    then selects the columns named in ``feature_columns(feature_set)``.

    Parameters
    ----------
    events
        Event-level DataFrame with multi-index columns as written by
        ``HH4b.postprocessing.load_run3_samples``.  No mutation.
    feature_set
        ``"strict"`` (no H1/H2 mass or tagger; default in CLIs) or
        ``"literal"`` (currently identical to ``strict`` — see the
        feature-list docstring above).

    Returns
    -------
    np.ndarray of shape ``(len(events), d)`` and dtype ``float32``, where
    ``d == len(feature_columns(feature_set))``.
    """
    # No-jshift identity key_map.  Avoids importing HH4b.utils.get_var_mapping
    # (which chains through coffea/numba and is brittle outside the analysis
    # env) — the v13_glopartv3 BDT config only uses key_map for column-name
    # remapping under JEC shifts, which we never apply here.
    bdt_dataframe = importlib.import_module(_BDT_CONFIG).bdt_dataframe
    df = bdt_dataframe(events, lambda x: x)

    # Base (kinematic) columns computed by the BDT config.
    base_cols = feature_columns(_base_feature_set(feature_set))
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns from bdt_dataframe output: {missing}")
    X = df[base_cols].to_numpy(dtype=np.float32)

    # Optionally append one-hot era (read from the original events, not df).
    if _uses_era(feature_set):
        X = np.concatenate([X, _era_one_hot(events)], axis=1)
    return X
