from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from HH4b import utils
from HH4b.hh_vars import mreg_strings, samples_run3, txbb_strings
from HH4b.postprocessing import HLTs, load_run3_samples

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(name)-20s %(message)s"
)

# ── Sample overrides (consistent with TrainBDT.py) ──────────────────────────
# Keep a deep copy so we don't mutate the module-level dict for other imports
samples = copy.deepcopy(samples_run3)

# QCD_HT for 2024, QCD-4Jets_HT for 2022/2023
for _year in samples:
    samples[_year]["qcd"] = [
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
    samples[_year]["ttbar"] = [
        "TTto2L2Nu",
        "TTto4Q",
        "TTtoLNu2Q",
    ]
    samples[_year]["diboson"] = [
        "WW",
        "WZ",
        "ZZ",
    ]

# ── Extra columns needed for NN training (not in postprocessing defaults) ───
NN_EXTRA_COLUMNS = [
    ("ht", 1),
    ("nJets", 1),
    ("nFatJets", 1),
    ("bbFatJetMass", 2),
    ("bbFatJetPNetMass", 2),
    ("bbFatJetPNetMassRaw", 2),
    ("bbFatJetPNetMassLegacy", 2),
    ("bbFatJetParT3PTopbWev", 2),
    ("bbFatJetParT3PTopbWmv", 2),
    ("bbFatJetParT3PTopbWq", 2),
    ("bbFatJetParT3PTopbWqq", 2),
    ("bbFatJetParT3PTopbWtauhv", 2),
    ("bbFatJetParT3PXbb", 2),
    ("bbFatJetParT3PXcc", 2),
    ("bbFatJetParT3PXcs", 2),
    ("bbFatJetParT3PXqq", 2),
    ("AK4JetAwayrawFactor", 2),
    ("AK4JetAwaybtagDeepFlavB", 2),
    ("AK4JetAwaybtagPNetB", 2),
    ("AK4JetAwaybtagPNetCvB", 2),
    ("AK4JetAwaybtagPNetCvL", 2),
    ("AK4JetAwaybtagPNetQvG", 2),
    ("VBFJetrawFactor", 2),
    ("VBFJetbtagDeepFlavB", 2),
    ("VBFJetbtagPNetB", 2),
    ("VBFJetbtagPNetCvB", 2),
    ("VBFJetbtagPNetCvL", 2),
    ("VBFJetbtagPNetQvG", 2),
]

# ── Preselection cuts (consistent with TrainBDT.py) ─────────────────────────
txbb_preselection = {
    "bbFatJetPNetTXbb": 0.3,
    "bbFatJetPNetTXbbLegacy": 0.8,
    "bbFatJetParTTXbb": 0.3,
    "bbFatJetParT3TXbb": 0.3,
}
msd1_preselection = {
    "bbFatJetPNetTXbb": 40,
    "bbFatJetPNetTXbbLegacy": 40,
    "bbFatJetParTTXbb": 40,
    "bbFatJetParT3TXbb": 40,
}
msd2_preselection = {
    "bbFatJetPNetTXbb": 30,
    "bbFatJetPNetTXbbLegacy": 0,
    "bbFatJetParTTXbb": 30,
    "bbFatJetParT3TXbb": 30,
}


def apply_cuts(events_dict, txbb_str, mass_str):
    """
    Apply cuts consistent with TrainBDT.py.
    pT(1,2) > 250, mReg(1,2) > 50 and TXbb(1) and mSD(1,2) preselection.
    """
    for key in list(events_dict.keys()):
        events = events_dict[key]
        n_before = len(events)
        msd1 = events["bbFatJetMsd"][0]
        msd2 = events["bbFatJetMsd"][1]
        pt1 = events["bbFatJetPt"][0]
        pt2 = events["bbFatJetPt"][1]
        txbb1 = events[txbb_str][0]
        mass1 = events[mass_str][0]
        mass2 = events[mass_str][1]
        events_dict[key] = events[
            (pt1 > 250)
            & (pt2 > 250)
            & (txbb1 > txbb_preselection[txbb_str])
            & (msd1 > msd1_preselection[txbb_str])
            & (msd2 > msd2_preselection[txbb_str])
            & (mass1 > 50)
            & (mass2 > 50)
        ].copy()
        logger.info(f"    {key}: {len(events_dict[key])} / {n_before} events pass cuts")
    return events_dict


def parse_args():
    storage_dir = Path("/ceph/cms/store/user/zichun/bbbb")
    default_tag = "nanov15_20251202_v15_signal"

    parser = argparse.ArgumentParser(description="Prepare Neural Network training dataframes.")
    parser.add_argument("--tag", default=default_tag)
    parser.add_argument("--years", nargs="+", default=["2024"])
    parser.add_argument("--txbb", default="glopart-v3")
    parser.add_argument("--mass", default=None, help="Mass variable (default: from txbb version)")
    parser.add_argument("--input-dir", default=str(storage_dir / f"skimmer/{default_tag}"))
    parser.add_argument(
        "--output-dir", default=str(storage_dir / f"signal_processed/nn_training/{default_tag}")
    )
    parser.add_argument("--bdt-config", default="v13_glopartv3")
    parser.add_argument(
        "--sig-keys",
        nargs="+",
        default=["hh4b", "vbfhh4b-k2v0"],
        help="Signal sample keys",
    )
    parser.add_argument(
        "--bg-keys",
        nargs="+",
        default=["qcd", "ttbar"],
        help="Background sample keys",
    )
    parser.add_argument("--apply-cuts", action="store_true", default=True)
    parser.add_argument(
        "--train-frac", type=float, default=0.80, help="Fraction for training set (default: 0.80)"
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.10, help="Fraction for validation set (default: 0.10)"
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.10, help="Fraction for test set (default: 0.10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting (default: 42)"
    )
    return parser.parse_args()


def validate_fracs(args):
    total = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"--train-frac + --val-frac + --test-frac must sum to 1.0, got {total:.4f}"
        )


def cleanup_events(events: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN values and out-of-range values (|x| > 50000) with -99999."""
    for col in events.columns:
        if "weight" not in col[0].lower():
            events[col] = events[col].apply(lambda x: -99999 if pd.isna(x) or abs(x) > 50000 else x)
    return events


def train_val_test_split(
    events: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Shuffle and split events into train / val / test DataFrames."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(events))

    n_train = int(np.floor(train_frac * len(events)))
    n_val = int(np.floor(val_frac * len(events)))

    splits = {
        "train": events.iloc[idx[:n_train]],
        "val": events.iloc[idx[n_train : n_train + n_val]],
        "test": events.iloc[idx[n_train + n_val :]],
    }
    return splits


def main():
    args = parse_args()
    validate_fracs(args)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    txbb_str = txbb_strings[args.txbb]
    mass_str = args.mass if args.mass else mreg_strings[args.txbb]

    bdt_dataframe_fn = importlib.import_module(
        f".{args.bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
    ).bdt_dataframe

    # Determine which sample keys to load
    training_keys = args.sig_keys + args.bg_keys
    aux_keys = ["vhtobb", "tthtobb", "diboson", "vjets"]
    keys_to_load = training_keys + aux_keys

    # Filter samples to only include requested keys
    samples_to_load = copy.deepcopy(samples)
    for year in list(samples_to_load.keys()):
        if year not in args.years:
            for key in list(samples_to_load[year].keys()):
                del samples_to_load[year][key]
        else:
            for key in list(samples_to_load[year].keys()):
                if key not in keys_to_load:
                    del samples_to_load[year][key]

    logger.info(f"Samples to load: {samples_to_load}")

    for year in args.years:
        logger.info(f"Loading {year} with txbb={args.txbb}")

        # Use load_run3_samples consistent with TrainBDT.py
        events_dict = load_run3_samples(
            input_dir,
            year,
            samples_to_load,
            reorder_txbb=True,
            load_systematics=False,
            txbb_version=args.txbb,
            scale_and_smear=False,
            mass_str=mass_str,
            bdt_version=args.bdt_config,
            load_bdt_scores=False,
            extra_columns=NN_EXTRA_COLUMNS,
        )

        # Apply cuts consistent with TrainBDT.py
        if args.apply_cuts:
            logger.info("Applying kinematic cuts...")
            events_dict = apply_cuts(events_dict, txbb_str, mass_str)

        # Apply trigger selection (skimmer does not apply trigger for signal region MC)
        hlt_cols = [hlt for hlt in HLTs[year]]  # noqa: C416
        n_before_trig = sum(len(events_dict[k]) for k in events_dict)
        for key in list(events_dict.keys()):
            trigger_mask = events_dict[key][
                [col for col in events_dict[key].columns if col[0] in hlt_cols]
            ].any(axis=1)
            events_dict[key] = events_dict[key][trigger_mask]
        n_after_trig = sum(len(events_dict[k]) for k in events_dict)
        logger.info(f"    Trigger selection: {n_after_trig} / {n_before_trig} events pass")

        # Filter keys to only those that loaded
        sig_keys = [k for k in args.sig_keys if k in events_dict]
        bg_keys = [k for k in args.bg_keys if k in events_dict]

        # Log weights and equalization scaling factors
        weight_totals = {}
        for key in sig_keys + bg_keys:
            weight_totals[key] = np.sum(np.abs(events_dict[key]["finalWeight"].to_numpy()))
            logger.info(
                f"    {key}: total weight = {weight_totals[key]:.3f}, events = {len(events_dict[key])}"
            )

        bkg_total = sum(weight_totals[k] for k in bg_keys)
        sig_total = sum(weight_totals[k] for k in sig_keys)
        num_sigs = len(sig_keys)
        logger.info(f"    Total bkg weight: {bkg_total:.3f}")
        logger.info(f"    Total sig weight: {sig_total:.3f}")
        # Compute class weights (equalize signal = bkg)
        class_weights = {}
        for key in sig_keys:
            class_weights[key] = (bkg_total / weight_totals[key]) / num_sigs
        for key in bg_keys:
            class_weights[key] = 1.0

        logger.info("    Equalize-weights scaling factors (signal = bkg):")
        for key, scale in class_weights.items():
            logger.info(f"        {key}: x{scale:.3f}")

        # Save class weights json
        weights_path = output_dir / f"class_weights_{year}.json"
        with open(weights_path, "w") as f:  # noqa: PTH123
            json.dump(class_weights, f, indent=2)
        logger.info(f"    Class weights saved to {weights_path}")

        # Process each sample
        for sample, events in events_dict.items():
            if len(events) == 0:
                logger.warning(f"    {sample}: no events after cuts, skipping")
                continue

            logger.info(f"    {sample}: {len(events)} events")

            logger.info(f"    Computing BDT input variables for {sample}...")
            bdt_df = bdt_dataframe_fn(events, utils.get_var_mapping(""))
            bdt_df.columns = pd.MultiIndex.from_tuples([(col, 0) for col in bdt_df.columns])

            events = cleanup_events(pd.concat([events, bdt_df], axis=1))  # noqa: PLW2901

            # Split and save
            splits = train_val_test_split(events, args.train_frac, args.val_frac, args.seed)
            splits["all"] = events
            for split_name, split_df in splits.items():
                save_path = output_dir / f"{split_name}/{year}_{sample}.parquet"
                save_path.parent.mkdir(exist_ok=True, parents=True)
                logger.info(f"        [{split_name:5s}] {len(split_df):>7,} events -> {save_path}")
                split_df.to_parquet(save_path)


if __name__ == "__main__":
    main()
