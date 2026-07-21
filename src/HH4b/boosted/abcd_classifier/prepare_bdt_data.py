"""
Prepare BDT training datasets restricted to ABCD Region A.

Produces, per ``--version``::

    <out-dir>/<version>/{train,val,test}.parquet

Each parquet has one row per training event with columns:

    {BDT input features}     # from v13_glopartv3
    label        int         # 0=hh4b, 1=vbfhh4b-k2v0, 2=ttbar, 3=qcd
    sample       str         # one of {hh4b, vbfhh4b-k2v0, ttbar, qcd}
    weight       float32     # finalWeight  (or w_QCD_A for ABCD-derived QCD)
    region       str         # always "A" — kept for downstream sanity checks

Two versions ("--version abcd" vs "--version mc") differ ONLY in the QCD
source for the SR (Region A):

  - **abcd**  : QCD events are Region-B *data*, each weighted by
                ``w_QCD_A(x) = TF(x) · purity(x)``.  No event lives in A
                physically; their `region` column is set to "A" because they
                are *labelled* as QCD-in-A.
  - **mc**    : QCD events are Region-A *MC*, weighted by ``finalWeight``.
                Standard baseline.

Signal (hh4b SM, vbfhh4b-k2v0) and ttbar are *always* taken from Region A
MC.  Following the canonical TrainBDT convention, the VBF signal class is
the BSM κ_{2V}=0 point (the kinematic outlier) rather than the SM point.

``H1Xbb`` is retained as a BDT input: the ABCD regions are now defined by
``H2Xbb``, so H1Xbb no longer trivially separates the regions.

Preselection: matches ``HH4b.boosted.TrainBDT.apply_cuts`` for
``glopart-v3``: pT₀,₁ > 250, TXbb₀ > 0.3, mSD cuts, m₀,₁ > 50; plus the HLT
OR.  Then Region-A cut (TXbb₀ ∈ [0.8,1.0), m₀ ∈ [100,150)) for
signal/ttbar/MC-QCD, and Region-B cut (TXbb₀ ∈ [0.3,0.8), m₀ ∈ [100,150))
for ABCD-derived QCD.

Train / val / test split: 70 / 15 / 15, stratified by ``label`` so each
split contains every class.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import logging.config
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from HH4b import hh_vars
from HH4b.hh_vars import txbb_strings
from HH4b.log_utils import log_config
from HH4b.postprocessing import HLTs

from . import dataset as ds_mod

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ABCDnn.prepare_bdt_data")


# Default training set (matches the user's analysis convention).
DEFAULT_SIGNAL_KEYS = ("hh4b", "vbfhh4b-k2v0")
DEFAULT_TTBAR_KEY = "ttbar"
DEFAULT_QCD_KEY = "qcd"
DEFAULT_DATA_KEY = "data"

# 4-class label encoding (matches the four-key training in TrainBDT.py).
DEFAULT_LABEL_MAP = {
    "hh4b": 0,
    "vbfhh4b-k2v0": 1,
    "ttbar": 2,
    "qcd": 3,
}

# Matches the preselection in HH4b.boosted.TrainBDT.apply_cuts for glopart-v3.
TXBB_PRESEL = 0.3
MSD1_PRESEL = 40
MSD2_PRESEL = 30
MASS_PRESEL = 50


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Region-A BDT training datasets, with either MC "
        "or ABCDnn-derived QCD.",
    )
    parser.add_argument(
        "--bdt-inference-dir",
        default="/ceph/cms/store/user/zichun/bbbb/signal_processed/bdt_inference",
        help="base directory of cached post-inference pickles.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="BDT training directory name; selects the cache subdir.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="ABCDnn run dir written by train.py / apply.py.  Used to find "
        "per_event_weights.parquet (for --version abcd) and to write the "
        "default output path under <run-dir>/bdt_datasets/.",
    )
    parser.add_argument(
        "--year",
        nargs="+",
        type=str,
        default=["2022"],
        choices=hh_vars.years + ["2022-2023", "2022-2023-2024"],
    )
    parser.add_argument(
        "--version",
        nargs="+",
        choices=["abcd", "mc", "full", "hybrid", "fullqcd", "base_regionA", "base_full"],
        default=["abcd", "mc"],
        help="which dataset version(s) to produce.  'abcd'/'mc' restrict to "
        "Region A with the two QCD definitions; 'full' is the canonical "
        "TrainBDT baseline (MC QCD, no ABCD region restriction — only "
        "preselection + trigger).  'hybrid' is full A+B+C+D for signals + "
        "ttbar, plus split QCD: ABCDnn-derived B-data with TF·purity → "
        "Region A, and QCD MC for B/C/D.  When producing 'full' or "
        "'hybrid', pair with --bdt-config v13_glopartv3 to include H1Xbb.",
    )
    parser.add_argument(
        "--txbb",
        choices=["pnet-v12", "pnet-legacy", "glopart-v2", "glopart-v3"],
        default="glopart-v3",
    )
    parser.add_argument(
        "--mass",
        default="bbFatJetParT3massX2p",
        help="leading-jet mass column name.",
    )
    parser.add_argument(
        "--txbb-bins",
        type=float,
        nargs=3,
        default=[0.3, 0.8, 1.0],
        metavar=("LOW", "SPLIT", "HIGH"),
        help="ABCD bins for the TXbb axis.  v1: [0.3, 0.8, 1.0] (jet 0). "
        "v2: [0.0, 0.65, 1.0] (jet 1).",
    )
    parser.add_argument(
        "--mass-bins",
        type=float,
        nargs=3,
        default=[50.0, 100.0, 150.0],
        metavar=("LOW", "SPLIT", "HIGH"),
    )
    parser.add_argument(
        "--txbb-jet-index",
        type=int,
        default=0,
        choices=[0, 1],
        help="Which FatJet's TXbb defines the ABCD TXbb axis.  v1: 0 (leading). "
        "v2: 1 (subleading).  When set to 1, pair with --bdt-config "
        "v13_glopartv3 so the freed H1Xbb is included as a BDT feature.",
    )
    parser.add_argument(
        "--mass-jet-index",
        type=int,
        default=0,
        choices=[0, 1],
        help="Which FatJet's mass defines the ABCD mass axis.  Kept at 0 "
        "across all current tags.",
    )
    parser.add_argument(
        "--bdt-config",
        default="v13_glopartv3",
        help="BDT input config module under HH4b.boosted.bdt_trainings_run3.",
    )
    parser.add_argument(
        "--abcdnn-weight-col",
        default="w_QCD_A",
        choices=["w_QCD_A", "w_QCD_A_raw", "TF"],
        help="Column from apply/per_event_weights.parquet to use as the "
        "B-data weight for --version abcd.  'w_QCD_A' = TF · purity, clamped "
        "to [0,100] (default); 'w_QCD_A_raw' = the same, UN-clamped (signed, "
        "for the TrainBDT-style |w| convention — train_bdt applies np.abs); "
        "'TF' = TF only, no purity correction.",
    )
    parser.add_argument(
        "--signal-keys",
        nargs="+",
        default=list(DEFAULT_SIGNAL_KEYS),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output base dir (default: <run-dir>/bdt_datasets/).",
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ------------------------------------------------------------------
# Preselection (mirror of TrainBDT.apply_cuts) + trigger selection
# ------------------------------------------------------------------


def _apply_preselection(df: pd.DataFrame, txbb_str: str, mass_str: str) -> np.ndarray:
    """Return a boolean mask matching TrainBDT.apply_cuts for glopart-v3."""
    pt1 = (
        df[(("bbFatJetPt"), 0)].to_numpy()  # noqa: RUF034
        if (("bbFatJetPt"), 0) in df.columns
        else df[("bbFatJetPt", 0)].to_numpy()
    )
    pt2 = df[("bbFatJetPt", 1)].to_numpy()
    txbb1 = df[(txbb_str, 0)].to_numpy()
    msd1 = df[("bbFatJetMsd", 0)].to_numpy()
    msd2 = df[("bbFatJetMsd", 1)].to_numpy()
    m1 = df[(mass_str, 0)].to_numpy()
    m2 = df[(mass_str, 1)].to_numpy()
    return (
        (pt1 > 250)
        & (pt2 > 250)
        & (txbb1 > TXBB_PRESEL)
        & (msd1 > MSD1_PRESEL)
        & (msd2 > MSD2_PRESEL)
        & (m1 > MASS_PRESEL)
        & (m2 > MASS_PRESEL)
    )


def _trigger_mask(df: pd.DataFrame, year: str) -> np.ndarray:
    # Multi-era caches (from prep_bdt_inference_pickles.py) already applied
    # the per-era HLT-OR and dropped the HLT columns, so there is nothing to
    # re-apply here.  Combined year-tags like "2022-2023" are not keys in
    # HLTs — treat any unknown tag (or HLT-less DataFrame) as a no-op.
    if year not in HLTs:
        return np.ones(len(df), dtype=bool)
    hlt_cols = [c for c in df.columns if c[0] in HLTs[year]]
    if not hlt_cols:
        return np.ones(len(df), dtype=bool)
    return df[hlt_cols].any(axis=1).to_numpy()


# ------------------------------------------------------------------
# Per-sample feature + weight extraction
# ------------------------------------------------------------------


def _features_and_weight(
    df: pd.DataFrame,
    mask: np.ndarray,
    bdt_dataframe_fn,
    weight_col: str = "finalWeight",
    weight_override: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    sub = df[mask].reset_index(drop=True)
    feats = bdt_dataframe_fn(sub, lambda x: x).reset_index(drop=True)
    if weight_override is not None:
        weight = weight_override.astype(np.float32)
    else:
        weight = sub[weight_col].to_numpy().astype(np.float32).reshape(-1)
    return feats, weight


def _build_per_sample(
    sample: str,
    df: pd.DataFrame,
    region_mask: np.ndarray,
    txbb_str: str,
    mass_str: str,
    year: str,
    bdt_dataframe_fn,
    weight_override: np.ndarray | None = None,
    region_label: str = "A",
) -> pd.DataFrame:
    presel = _apply_preselection(df, txbb_str, mass_str)
    trig = _trigger_mask(df, year)
    final_mask = presel & trig & region_mask
    n = int(final_mask.sum())
    logger.info(f"  {sample}: presel × trigger × region: {n} events")  # noqa: RUF001
    if n == 0:
        return pd.DataFrame()
    feats, weight = _features_and_weight(
        df,
        final_mask,
        bdt_dataframe_fn,
        weight_override=(weight_override[final_mask] if weight_override is not None else None),
    )
    out = feats.copy()
    out["weight"] = weight
    out["sample"] = sample
    out["region"] = region_label
    # Per-event year tag (from the cache 'era' column), so a combined-cache
    # base can carry real per-event year info for a categorical year feature.
    # era -> year by longest-prefix (2022EE->2022, 2023BPix->2023, 2024->2024).
    era_col = None
    if ("era", "") in df.columns:
        era_col = df[("era", "")]
    elif "era" in df.columns:
        era_col = df["era"]
    if era_col is not None:
        era = np.asarray(era_col)[final_mask]
        tags = ["2022", "2023", "2024", "2025"]
        out["year"] = [
            max((t for t in tags if str(e).startswith(t)), key=len, default=str(e)) for e in era
        ]
    return out


# ------------------------------------------------------------------
# Train / val / test split (stratified by label)
# ------------------------------------------------------------------


def _stratified_three_way_split(
    y: np.ndarray, train_frac: float, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_frac = 1.0 - train_frac - val_frac
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(np.zeros(len(y)), y))
    val_within = val_frac / (train_frac + val_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_within, random_state=seed + 1)
    y_tv = y[idx_trainval]
    rel_train, rel_val = next(sss2.split(np.zeros(len(y_tv)), y_tv))
    return idx_trainval[rel_train], idx_trainval[rel_val], idx_test


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def _data_driven_qcd_part(
    pickles: dict[str, pd.DataFrame],
    region_masks: dict[str, dict[str, np.ndarray]],
    abcdnn_run_dir: Path,
    txbb_str: str,
    mass_str: str,
    year: str,
    bdt_dataframe_fn,
    abcdnn_weight_col: str,
    sample_tag: str,
) -> pd.DataFrame:
    """Region-A QCD predicted from B-region DATA reweighted by the ABCDnn
    per-event weight (``abcdnn_weight_col`` from apply/per_event_weights.parquet).
    Tagged as ``sample_tag`` (e.g. 'qcd' for the abcd version, 'qcd_dd' in a
    base dataset)."""
    if DEFAULT_DATA_KEY not in pickles:
        raise RuntimeError("data sample missing — required for data-driven QCD")
    pew_path = abcdnn_run_dir / "apply" / "per_event_weights.parquet"
    if not pew_path.exists():
        raise FileNotFoundError(f"{pew_path} missing — run apply.py first.")
    pew = pd.read_parquet(pew_path)
    pew_B_data = pew[(pew["sample"] == "data") & (pew["region"] == "B")]
    df_data = pickles[DEFAULT_DATA_KEY]
    mask_B = region_masks[DEFAULT_DATA_KEY]["B"]
    idx_B = np.where(mask_B)[0]
    if len(idx_B) != len(pew_B_data):
        raise RuntimeError(
            f"row count mismatch: data B mask has {len(idx_B)} events, "
            f"per_event_weights.parquet has {len(pew_B_data)} rows for "
            f"(sample=data, region=B)."
        )
    if abcdnn_weight_col not in pew_B_data.columns:
        raise KeyError(
            f"column {abcdnn_weight_col!r} not found in {pew_path}; "
            f"available: {list(pew_B_data.columns)}"
        )
    weight_override = np.zeros(len(df_data), dtype=np.float32)
    weight_override[idx_B] = pew_B_data[abcdnn_weight_col].to_numpy(np.float32)
    logger.info(f"  data-driven QCD ({sample_tag}): weight column {abcdnn_weight_col!r}")
    return _build_per_sample(
        sample_tag,
        df_data,
        mask_B,
        txbb_str,
        mass_str,
        year,
        bdt_dataframe_fn,
        weight_override=weight_override,
        region_label="A",
    )


def _build_one_version(
    version: str,
    pickles: dict[str, pd.DataFrame],
    region_masks: dict[str, dict[str, np.ndarray]],
    bdt_dataframe_fn,
    abcdnn_run_dir: Path,
    year: str,
    txbb_str: str,
    mass_str: str,
    signal_keys: list[str],
    abcdnn_weight_col: str = "w_QCD_A",
) -> pd.DataFrame:
    """Assemble the unsplit DataFrame for one version (`abcd`, `mc`, `full`,
    `fullqcd`, `hybrid`) or a base dataset (`base_regionA`, `base_full`).

    Base datasets keep each event stream under a distinct ``sample`` tag (incl.
    'qcd' = MC QCD and 'qcd_dd' = data-driven QCD) so a version YAML can select
    per-class sources at train time — no per-version duplication on disk.  Their
    ``label`` column is just sample codes (for the stratified split); the real
    class label is assigned at composition time in train_bdt.
    """
    parts: list[pd.DataFrame] = []

    # ----- Base datasets (stream library; composed into versions later) -----
    if version in ("base_regionA", "base_full"):
        region_a = version == "base_regionA"
        for sample in list(signal_keys) + [DEFAULT_TTBAR_KEY, DEFAULT_QCD_KEY]:
            if sample not in pickles:
                logger.warning(f"  {sample} missing from cache; skipping")
                continue
            mask = (
                region_masks[sample]["A"] if region_a else np.ones(len(pickles[sample]), dtype=bool)
            )
            part = _build_per_sample(
                sample,
                pickles[sample],
                mask,
                txbb_str,
                mass_str,
                year,
                bdt_dataframe_fn,
                region_label="A" if region_a else "ABCD",
            )
            if not part.empty:
                parts.append(part)
        if region_a:
            # Data-driven QCD stream, tagged 'qcd_dd'.
            part = _data_driven_qcd_part(
                pickles,
                region_masks,
                abcdnn_run_dir,
                txbb_str,
                mass_str,
                year,
                bdt_dataframe_fn,
                abcdnn_weight_col,
                "qcd_dd",
            )
            if not part.empty:
                parts.append(part)
        if not parts:
            raise RuntimeError(f"no events for version={version}")
        combined = pd.concat(parts, axis=0, ignore_index=True)
        # label = per-stream code (stratification only); real label set later.
        combined["label"] = combined["sample"].astype("category").cat.codes.astype(np.int64)
        n_neg = int((combined["weight"] < 0).sum())
        if n_neg:
            logger.info(
                f"  parquet contains {n_neg} events with weight < 0 "
                f"({n_neg / len(combined):.3%}); train_bdt.py uses |w|"
            )
        return combined

    # Special-case the canonical baseline: no ABCD region restriction,
    # all four classes from MC.  Matches TrainBDT.apply_cuts (preselection
    # + trigger only).  Pair with --bdt-config v13_glopartv3 to keep
    # H1Xbb in the feature set as in the original training.
    if version == "full":
        for sample in list(signal_keys) + [DEFAULT_TTBAR_KEY, DEFAULT_QCD_KEY]:
            if sample not in pickles:
                logger.warning(f"  {sample} missing from cache; skipping")
                continue
            full_mask = np.ones(len(pickles[sample]), dtype=bool)
            part = _build_per_sample(
                sample,
                pickles[sample],
                full_mask,
                txbb_str,
                mass_str,
                year,
                bdt_dataframe_fn,
                region_label="ABCD",
            )
            if not part.empty:
                parts.append(part)

        if not parts:
            raise RuntimeError(f"no events for version={version}")
        combined = pd.concat(parts, axis=0, ignore_index=True)
        combined["label"] = combined["sample"].map(DEFAULT_LABEL_MAP).astype(np.int64)
        n_neg = int((combined["weight"] < 0).sum())
        if n_neg:
            logger.info(
                f"  parquet contains {n_neg} events with weight < 0 "
                f"({n_neg / len(combined):.3%}); train_bdt.py uses |w| per "
                f"TrainBDT.preprocess_data convention"
            )
        return combined

    # Hybrid: full A+B+C+D for signals + ttbar (same as `full` for those);
    # QCD source split by region — ABCDnn-derived B-data ·TF·purity for A,
    # QCD MC for B/C/D.  Region-A MC QCD is *excluded* so it doesn't
    # double-count with the ABCDnn prediction.
    if version == "hybrid":
        # Signals + ttbar: full preselected events, no region cut.
        for sample in list(signal_keys) + [DEFAULT_TTBAR_KEY]:
            if sample not in pickles:
                logger.warning(f"  {sample} missing from cache; skipping")
                continue
            full_mask = np.ones(len(pickles[sample]), dtype=bool)
            part = _build_per_sample(
                sample,
                pickles[sample],
                full_mask,
                txbb_str,
                mass_str,
                year,
                bdt_dataframe_fn,
                region_label="ABCD",
            )
            if not part.empty:
                parts.append(part)

        # QCD MC restricted to B/C/D (NOT Region A) to avoid double-counting
        # with the ABCDnn-derived A prediction.
        if DEFAULT_QCD_KEY not in pickles:
            raise RuntimeError("qcd sample missing — required for version=hybrid")
        not_a_mask = ~region_masks[DEFAULT_QCD_KEY]["A"]
        part = _build_per_sample(
            DEFAULT_QCD_KEY,
            pickles[DEFAULT_QCD_KEY],
            not_a_mask,
            txbb_str,
            mass_str,
            year,
            bdt_dataframe_fn,
            region_label="BCD",
        )
        if not part.empty:
            parts.append(part)

        # ABCDnn-derived QCD in Region A: B-data × (TF · purity or TF only).  # noqa: RUF003
        if DEFAULT_DATA_KEY not in pickles:
            raise RuntimeError("data sample missing — required for version=hybrid")
        pew_path = abcdnn_run_dir / "apply" / "per_event_weights.parquet"
        if not pew_path.exists():
            raise FileNotFoundError(f"{pew_path} missing — run apply.py first.")
        pew = pd.read_parquet(pew_path)
        pew_B_data = pew[(pew["sample"] == "data") & (pew["region"] == "B")]
        df_data = pickles[DEFAULT_DATA_KEY]
        mask_B = region_masks[DEFAULT_DATA_KEY]["B"]
        idx_B = np.where(mask_B)[0]
        if len(idx_B) != len(pew_B_data):
            raise RuntimeError(
                f"row count mismatch: data B mask has {len(idx_B)} events, "
                f"per_event_weights.parquet has {len(pew_B_data)} rows for "
                f"(sample=data, region=B)."
            )
        weight_override = np.zeros(len(df_data), dtype=np.float32)
        if abcdnn_weight_col not in pew_B_data.columns:
            raise KeyError(f"column {abcdnn_weight_col!r} not found in {pew_path}")
        weight_override[idx_B] = pew_B_data[abcdnn_weight_col].to_numpy(np.float32)
        logger.info(f"  hybrid: using ABCDnn weight column {abcdnn_weight_col!r}")
        part = _build_per_sample(
            DEFAULT_QCD_KEY,
            df_data,
            mask_B,
            txbb_str,
            mass_str,
            year,
            bdt_dataframe_fn,
            weight_override=weight_override,
            region_label="A",
        )
        if not part.empty:
            parts.append(part)

        if not parts:
            raise RuntimeError(f"no events for version={version}")
        combined = pd.concat(parts, axis=0, ignore_index=True)
        combined["label"] = combined["sample"].map(DEFAULT_LABEL_MAP).astype(np.int64)
        n_neg = int((combined["weight"] < 0).sum())
        if n_neg:
            logger.info(
                f"  parquet contains {n_neg} events with weight < 0 "
                f"({n_neg / len(combined):.3%}); train_bdt.py uses |w| per "
                f"TrainBDT.preprocess_data convention"
            )
        return combined

    # Signals + ttbar: always A-region MC, weight = finalWeight.
    for sample in list(signal_keys) + [DEFAULT_TTBAR_KEY]:
        if sample not in pickles:
            logger.warning(f"  {sample} missing from cache; skipping")
            continue
        part = _build_per_sample(
            sample,
            pickles[sample],
            region_masks[sample]["A"],
            txbb_str,
            mass_str,
            year,
            bdt_dataframe_fn,
            region_label="A",
        )
        if not part.empty:
            parts.append(part)

    # QCD: depends on version.
    if version == "mc":
        if DEFAULT_QCD_KEY not in pickles:
            raise RuntimeError("qcd sample missing — required for version=mc")
        part = _build_per_sample(
            DEFAULT_QCD_KEY,
            pickles[DEFAULT_QCD_KEY],
            region_masks[DEFAULT_QCD_KEY]["A"],
            txbb_str,
            mass_str,
            year,
            bdt_dataframe_fn,
            region_label="A",
        )
        if not part.empty:
            parts.append(part)
    elif version == "abcd":
        if DEFAULT_DATA_KEY not in pickles:
            raise RuntimeError("data sample missing — required for version=abcd")

        # Load w_QCD_A from apply step.
        pew_path = abcdnn_run_dir / "apply" / "per_event_weights.parquet"
        if not pew_path.exists():
            raise FileNotFoundError(f"{pew_path} missing — run apply.py first.")
        pew = pd.read_parquet(pew_path)
        pew_B_data = pew[(pew["sample"] == "data") & (pew["region"] == "B")]

        # Build weight_override aligned to the data pickle's row index, so it
        # filters through preselection × trigger × region together.  # noqa: RUF003
        df_data = pickles[DEFAULT_DATA_KEY]
        mask_B = region_masks[DEFAULT_DATA_KEY]["B"]
        idx_B = np.where(mask_B)[0]
        if len(idx_B) != len(pew_B_data):
            raise RuntimeError(
                f"row count mismatch: data B mask has {len(idx_B)} events, "
                f"per_event_weights.parquet has {len(pew_B_data)} rows for "
                f"(sample=data, region=B). Did the region cuts change between "
                f"apply.py and prepare_bdt_data.py?"
            )
        weight_override = np.zeros(len(df_data), dtype=np.float32)
        weight_col = abcdnn_weight_col
        if weight_col not in pew_B_data.columns:
            raise KeyError(
                f"column {weight_col!r} not found in {pew_path}; "
                f"available: {list(pew_B_data.columns)}"
            )
        weight_override[idx_B] = pew_B_data[weight_col].to_numpy(np.float32)
        logger.info(f"  abcd: using ABCDnn weight column {weight_col!r}")

        # Label this stream as "qcd" but use the B-data events, with the
        # ABCDnn weight standing in for finalWeight.  The "region" column
        # records that they are *predicted to look like* region A, even
        # though the underlying events sit physically in B.
        part = _build_per_sample(
            DEFAULT_QCD_KEY,
            df_data,
            mask_B,
            txbb_str,
            mass_str,
            year,
            bdt_dataframe_fn,
            weight_override=weight_override,
            region_label="A",
        )
        if not part.empty:
            parts.append(part)
    elif version == "fullqcd":
        # Region-A signal + ttbar (from the shared loop above) but QCD from
        # the FULL A+B+C+D region (MC).  Isolates the QCD-region effect with
        # ttbar held at Region A — pair with train_bdt --balance-bg so the
        # (much larger) full QCD doesn't swamp the Region-A ttbar.
        if DEFAULT_QCD_KEY not in pickles:
            raise RuntimeError("qcd sample missing — required for version=fullqcd")
        full_mask = np.ones(len(pickles[DEFAULT_QCD_KEY]), dtype=bool)
        part = _build_per_sample(
            DEFAULT_QCD_KEY,
            pickles[DEFAULT_QCD_KEY],
            full_mask,
            txbb_str,
            mass_str,
            year,
            bdt_dataframe_fn,
            region_label="ABCD",
        )
        if not part.empty:
            parts.append(part)
    else:
        raise ValueError(f"unknown version: {version!r}")

    if not parts:
        raise RuntimeError(f"no events for version={version}")

    combined = pd.concat(parts, axis=0, ignore_index=True)
    combined["label"] = combined["sample"].map(DEFAULT_LABEL_MAP).astype(np.int64)

    # Note: we intentionally keep weight ≤ 0 events.  They come from two
    # sources:
    #   (a) NLO MC book-keeping weights (mostly ttbar @ POWHEG, ~3–10% of  # noqa: RUF003
    #       events flip sign).
    #   (b) ABCDnn per-event QCD correction (B-data, ~0.1% negative).
    # Following HH4b.boosted.TrainBDT.preprocess_data (line 232), the
    # downstream training takes |weight| at training time.  We preserve the
    # signs in the parquet so other consumers can choose differently.
    n_neg = int((combined["weight"] < 0).sum())
    if n_neg:
        logger.info(
            f"  parquet contains {n_neg} events with weight < 0 "
            f"({n_neg / len(combined):.3%}); train_bdt.py uses |w| per "
            f"TrainBDT.preprocess_data convention"
        )
    return combined


def main() -> None:
    args = parse_args()

    # Year-tags are literal cache subdir names (multi-era combining done
    # upstream in prep_bdt_inference_pickles.py).  No era expansion.
    years = list(args.year)
    if len(years) != 1:
        raise NotImplementedError("multi-year prep not implemented yet")
    year = years[0]

    txbb_str = txbb_strings[args.txbb]
    mass_str = args.mass

    abcdnn_run_dir = Path(args.run_dir)
    out_base = Path(args.out_dir) if args.out_dir else (abcdnn_run_dir / "bdt_datasets")
    out_base.mkdir(parents=True, exist_ok=True)

    bdt_dataframe_fn = importlib.import_module(
        f"HH4b.boosted.bdt_trainings_run3.{args.bdt_config}"
    ).bdt_dataframe

    # Load every sample we might need.
    samples_to_load = tuple(
        s
        for s in (list(args.signal_keys) + [DEFAULT_TTBAR_KEY, DEFAULT_QCD_KEY, DEFAULT_DATA_KEY])
        if (Path(args.bdt_inference_dir) / args.model_name / year / f"{s}.pkl").exists()
    )
    logger.info(f"loading samples: {samples_to_load}")
    pickles = ds_mod.load_events(
        args.bdt_inference_dir,
        args.model_name,
        [year],
        samples=samples_to_load,
    )

    region_masks: dict[str, dict[str, np.ndarray]] = {}
    for sample, df in pickles.items():
        region_masks[sample] = ds_mod.build_region_masks(
            df,
            txbb_str,
            mass_str,
            tuple(args.txbb_bins),
            tuple(args.mass_bins),
            txbb_jet_index=args.txbb_jet_index,
            mass_jet_index=args.mass_jet_index,
        )
    logger.info(
        f"ABCD region defn: TXbb={txbb_str}[{args.txbb_jet_index}] split at "
        f"{args.txbb_bins}; mass={mass_str}[{args.mass_jet_index}] split at "
        f"{args.mass_bins}"
    )

    for version in args.version:
        logger.info(f"=== building version={version} ===")
        combined = _build_one_version(
            version,
            pickles,
            region_masks,
            bdt_dataframe_fn,
            abcdnn_run_dir,
            year,
            txbb_str,
            mass_str,
            args.signal_keys,
            abcdnn_weight_col=args.abcdnn_weight_col,
        )
        logger.info(f"  total: {len(combined)} events")
        for sample, n in combined["sample"].value_counts().items():
            sw = combined.loc[combined["sample"] == sample, "weight"].sum()
            logger.info(f"    {sample}: {n} events  Σw = {sw:.2f}")

        idx_train, idx_val, idx_test = _stratified_three_way_split(
            combined["label"].to_numpy(),
            args.train_frac,
            args.val_frac,
            args.seed,
        )

        out_dir = out_base / version
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
            df_split = combined.iloc[idx].reset_index(drop=True)
            path = out_dir / f"{split}.parquet"
            df_split.to_parquet(path)
            logger.info(f"  saved {path}  ({len(df_split)} events)")


if __name__ == "__main__":
    main()
