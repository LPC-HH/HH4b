"""
Build per-sample BDT-inference pickles for the ABCDnn pipeline, combining
one or more data-taking *eras* into a single *year-tag*.

The ABCDnn pipeline (``dataset.load_events``, ``prepare_bdt_data``,
``apply.py``, ``plot.py``) reads cached per-sample DataFrames at::

    <out-cache>/<year-tag>/<sample>.pkl

Each pickle is one sample's events with the raw skimmer multi-index
columns (``bbFatJetPt``, ``bbFatJetParT3TXbb``, …) + ``finalWeight`` +
``bdt_score``/``bdt_score_vbf`` (read straight from the skimmer parquet).

Only 2022 existed in the cache previously.  This script produces the
multi-era caches needed to train ABCDnn on, e.g., 2022+2022EE.

Era → year-tag mapping is arbitrary and set by ``--eras`` / ``--year``::

    --eras 2022 2022EE              --year 2022
    --eras 2023 2023BPix            --year 2023
    --eras 2022 2022EE 2023 2023BPix --year 2022-2023

Trigger handling
----------------
The skimmer (``bbbbSkimmer``) stores HLT flags as columns but does NOT
cut on them for Run3 (``apply_trigger = False``).  The trigger OR is
applied downstream.

**Each era applies its own HLT-OR** using the ``bbbbSkimmer`` signal
trigger menu for that era (``SKIMMER_HLTS`` below, copied verbatim from
``bbbbSkimmer.process``).  These are the authoritative per-year menus
(fuller than the year-varying ``postprocessing.HLTs`` used elsewhere).
We apply the era's OR while the era is known, then **drop the HLT
columns** so downstream ``prepare_bdt_data._trigger_mask`` finds none and
returns all-True (a no-op) — avoiding any second, inconsistent
re-application (important because a combined ``2022-2023`` tag has no
single ``postprocessing.HLTs`` key).

An ``era`` column is added for bookkeeping.
"""

from __future__ import annotations

import argparse
import copy
import logging
import logging.config
import pickle
from pathlib import Path

import pandas as pd

from HH4b import hh_vars
from HH4b.hh_vars import txbb_strings
from HH4b.log_utils import log_config
from HH4b.postprocessing import HLTs as PP_HLTS
from HH4b.postprocessing import load_run3_samples

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ABCDnn.prep_bdt_inference_pickles")


DEFAULT_SKIMMER_DIR = "/ceph/cms/store/user/zichun/bbbb/skimmer/nanov15_20251202_v15_signal"
DEFAULT_OUT_CACHE = (
    "/ceph/cms/store/user/zichun/bbbb/signal_processed/bdt_inference/"
    "nanov15_v15_glopartv3_rawmass"
)
DEFAULT_SAMPLES = ("data", "ttbar", "qcd", "hh4b", "vbfhh4b", "vbfhh4b-k2v0")

# Analysis preselection (mirrors HH4b.boosted.TrainBDT.apply_cuts /
# prepare_bdt_data._apply_preselection for glopart-v3).  The original
# 26Apr07 BDT-inference cache had these baked in, so we replicate them
# here to keep the ABCDnn event population identical.
PT_PRESEL = 250
TXBB0_PRESEL = 0.3
MSD0_PRESEL = 40
MSD1_PRESEL = 30
MASS_PRESEL = 50

# Per-era signal HLT menus, copied verbatim (deduplicated) from
# bbbbSkimmer.process (HLTs["signal"][year]).  Each era applies its OWN
# trigger OR — not a uniform set.  All listed triggers are present in the
# corresponding era's skimmer parquet output.
SKIMMER_HLTS = {
    "2022": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet425_SoftDropMass40",
    ],
    "2022EE": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet425_SoftDropMass40",
    ],
    "2023": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        "AK8PFJet230_SoftDropMass40",
        "AK8PFJet400_SoftDropMass40",
        "AK8PFJet425_SoftDropMass40",
        "AK8PFJet420_MassSD30",
    ],
    "2023BPix": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        "AK8PFJet230_SoftDropMass40",
        "AK8PFJet400_SoftDropMass40",
        "AK8PFJet425_SoftDropMass40",
        "AK8PFJet420_MassSD30",
    ],
    # bbbbSkimmer 'signal' menu for 2024 (verbatim from bbbbSkimmer.py).
    # AK8PFJet420_MassSD30 / AK8PFJet425_SoftDropMass40 are not in 2024.
    "2024": [
        "AK8PFJet500",
        "AK8PFJet400_SoftDropMass30",
        "AK8PFJet425_SoftDropMass30",
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
    ],
}

# QCD HT-bin expansion — mirrors HH4b.boosted.TrainBDT (the default
# samples_run3['qcd'] is a single placeholder and must be overridden).
QCD_SUBSAMPLES = [
    "QCD_HT-100to200",
    "QCD_HT-200to400",
    "QCD_HT-400to600",
    "QCD_HT-600to800",
    "QCD_HT-800to1000",
    "QCD_HT-1000to1200",
    "QCD_HT-1200to1500",
    "QCD_HT-1500to2000",
    "QCD_HT-2000",
    "QCD-4Jets_HT-100to200",
    "QCD-4Jets_HT-200to400",
    "QCD-4Jets_HT-400to600",
    "QCD-4Jets_HT-600to800",
    "QCD-4Jets_HT-800to1000",
    "QCD-4Jets_HT-1000to1200",
    "QCD-4Jets_HT-1200to1500",
    "QCD-4Jets_HT-1500to2000",
    "QCD-4Jets_HT-2000",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skimmer-dir",
        default=DEFAULT_SKIMMER_DIR,
        help="Skimmer base dir (contains <era>/<sample>/parquet/).",
    )
    p.add_argument(
        "--eras",
        nargs="+",
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix", "2024", "2025"],
        help="Data-taking eras to combine into one year-tag.",
    )
    p.add_argument(
        "--year",
        required=True,
        help="Output year-tag (subdir name under --out-cache).",
    )
    p.add_argument(
        "--out-cache",
        default=DEFAULT_OUT_CACHE,
        help="Cache root; pickles land in <out-cache>/<year>/<sample>.pkl.",
    )
    p.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    p.add_argument(
        "--txbb",
        choices=["pnet-v12", "pnet-legacy", "glopart-v2", "glopart-v3"],
        default="glopart-v3",
    )
    p.add_argument("--mass", default="bbFatJetParT3massX2p")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-sample pickles.",
    )
    return p.parse_args()


def _samples_dict_for_era(era: str, sample: str, skimmer_dir: str | None = None) -> dict:
    """Return ``{era: {sample: subsample-list}}`` with the QCD override.

    The QCD HT-bin list is filtered to subsamples that actually exist in the
    skimmer for this era (e.g. 2024 lacks the low-HT bins and the QCD-4Jets_*
    set), so the loader doesn't fail on missing directories.
    """
    sd = copy.deepcopy({era: dict(hh_vars.samples_run3[era])})
    if sample == "qcd":
        qcd = list(QCD_SUBSAMPLES)
        if skimmer_dir is not None:
            present = [s for s in qcd if (Path(skimmer_dir) / era / s).is_dir()]
            dropped = [s for s in qcd if s not in present]
            if dropped:
                logger.info(f"  {era}: QCD subsamples not in skimmer, skipping: {dropped}")
            qcd = present
        sd[era]["qcd"] = qcd
    return {era: {sample: sd[era][sample]}}


def _load_one_era_sample(
    skimmer_dir: str,
    era: str,
    sample: str,
    txbb: str,
    mass_str: str,
    model_tag: str,
) -> pd.DataFrame | None:
    """Load one sample for one era and apply that era's own HLT-OR.

    Returns the trigger-filtered DataFrame with HLT columns dropped and an
    ``era`` column added, or None if the sample has no events.
    """
    # load_run3_samples only loads the postprocessing.HLTs subset by default;
    # request the skimmer-menu triggers it would otherwise skip via
    # extra_columns so we can apply the full bbbbSkimmer per-era OR.
    extra_hlt_cols = [(h, 1) for h in SKIMMER_HLTS[era] if h not in PP_HLTS[era]]
    events = load_run3_samples(
        skimmer_dir,
        era,
        _samples_dict_for_era(era, sample, skimmer_dir),
        reorder_txbb=True,
        load_systematics=False,
        txbb_version=txbb,
        scale_and_smear=False,
        mass_str=mass_str,
        bdt_version=model_tag,
        extra_columns=extra_hlt_cols or None,
    ).get(sample)
    if events is None or len(events) == 0:
        logger.warning(f"  {era}/{sample}: no events")
        return None

    n_before = len(events)

    # --- analysis preselection (match the original 26Apr07 cache) ---
    txbb_str = txbb_strings[txbb]
    pt0 = events[("bbFatJetPt", 0)].to_numpy()
    pt1 = events[("bbFatJetPt", 1)].to_numpy()
    txbb0 = events[(txbb_str, 0)].to_numpy()
    msd0 = events[("bbFatJetMsd", 0)].to_numpy()
    msd1 = events[("bbFatJetMsd", 1)].to_numpy()
    m0 = events[(mass_str, 0)].to_numpy()
    m1 = events[(mass_str, 1)].to_numpy()
    presel = (
        (pt0 > PT_PRESEL)
        & (pt1 > PT_PRESEL)
        & (txbb0 > TXBB0_PRESEL)
        & (msd0 > MSD0_PRESEL)
        & (msd1 > MSD1_PRESEL)
        & (m0 > MASS_PRESEL)
        & (m1 > MASS_PRESEL)
    )

    # --- per-era HLT-OR ---
    era_hlts = SKIMMER_HLTS[era]
    hlt_cols = [c for c in events.columns if c[0] in era_hlts]
    missing = [h for h in era_hlts if not any(c[0] == h for c in events.columns)]
    if missing:
        raise RuntimeError(
            f"{era}/{sample}: HLT columns missing from skimmer output: "
            f"{missing}.  Cannot apply this era's trigger OR."
        )
    trig_mask = events[hlt_cols].any(axis=1).to_numpy()

    events = events[presel & trig_mask].copy()
    # Drop ALL HLT flag columns so no second, year-inconsistent
    # re-application can happen downstream.
    all_hlt_cols = [
        c for c in events.columns if isinstance(c[0], str) and c[0].startswith("AK8PFJet")
    ]
    events = events.drop(columns=all_hlt_cols)
    logger.info(
        f"  {era}/{sample}: {len(events)}/{n_before} pass presel × {era} HLT-OR "  # noqa: RUF001
        f"({len(hlt_cols)} triggers; {len(all_hlt_cols)} HLT cols dropped)"
    )
    # Bookkeeping column (single-level key; harmless to downstream which
    # selects multi-index kinematic columns explicitly).
    events[("era", "")] = era
    return events.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    skimmer_dir = args.skimmer_dir
    out_dir = Path(args.out_cache) / args.year
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(args.out_cache).name

    logger.info(
        f"building cache {out_dir}  from eras {args.eras}  " f"(txbb={args.txbb}, mass={args.mass})"
    )

    for sample in args.samples:
        out_path = out_dir / f"{sample}.pkl"
        if out_path.exists() and not args.force:
            logger.info(f"{sample}: exists, skipping (use --force to rebuild)")
            continue

        logger.info(f"=== {sample} ===")
        per_era = []
        for era in args.eras:
            df = _load_one_era_sample(skimmer_dir, era, sample, args.txbb, args.mass, model_tag)
            if df is not None:
                per_era.append(df)

        if not per_era:
            logger.warning(f"{sample}: no events in any era; not writing")
            continue

        # Align columns across eras (HLT already dropped; remaining schema
        # should match, but guard against stray differences).
        combined = pd.concat(per_era, axis=0, ignore_index=True)
        logger.info(f"{sample}: total {len(combined)} events across {len(per_era)} era(s)")
        with out_path.open("wb") as f:
            pickle.dump(combined, f)
        logger.info(f"saved {out_path}")


if __name__ == "__main__":
    main()
