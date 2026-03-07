from __future__ import annotations

import argparse
import importlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from HH4b import utils
from HH4b.postprocessing import filters_to_apply, txbb_strings
from HH4b.utils import get_var_mapping

warnings.filterwarnings("ignore")


SIGNAL_DICT = {
    "hh4b": ["GluGlutoHHto4B"],
    "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1"],
    "qcd": [
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
        "QCD_HT-2000",
    ],
    "ttbar": ["TTto4Q", "TTtoLNu2Q", "TTto2L2Nu"],
}

VAR_COLUMNS = [
    # event level
    ("MET_pt", 1),
    ("ht", 1),
    ("nJets", 1),
    ("nFatJets", 1),
    ("weight", 1),
    ("single_weight_genweight", 1),
    # triggers
    ("AK8PFJet500", 1),
    ("AK8PFJet400_SoftDropMass30", 1),
    ("AK8PFJet425_SoftDropMass30", 1),
    ("AK8PFJet230_SoftDropMass40_PNetBB0p06", 1),
    # AK4 jets
    ("AK4JetAwayEta", 2),
    ("AK4JetAwayPhi", 2),
    ("AK4JetAwayMass", 2),
    ("AK4JetAwayPt", 2),
    ("AK4JetAwayrawFactor", 2),
    ("AK4JetAwaybtagDeepFlavB", 2),
    ("AK4JetAwaybtagPNetB", 2),
    ("AK4JetAwaybtagPNetCvB", 2),
    ("AK4JetAwaybtagPNetCvL", 2),
    ("AK4JetAwaybtagPNetQvG", 2),
    # AK8 jets
    ("bbFatJetEta", 2),
    ("bbFatJetPhi", 2),
    ("bbFatJetMass", 2),
    ("bbFatJetPt", 2),
    ("bbFatJetMsd", 2),
    ("bbFatJetPNetTXbb", 2),
    ("bbFatJetPNetTXjj", 2),
    ("bbFatJetPNetTQCD", 2),
    ("bbFatJetPNetQCD1HF", 2),
    ("bbFatJetPNetQCD2HF", 2),
    ("bbFatJetPNetQCD0HF", 2),
    ("bbFatJetPNetMass", 2),
    ("bbFatJetPNetMassRaw", 2),
    ("bbFatJetTau3OverTau2", 2),
    ("bbFatJetrawFactor", 2),
    ("bbFatJetPNetMassLegacy", 2),
    ("bbFatJetPNetTXbbLegacy", 2),
    ("bbFatJetPNetPXbbLegacy", 2),
    ("bbFatJetPNetPQCDLegacy", 2),
    ("bbFatJetParT3PQCD", 2),
    ("bbFatJetParT3PTopbWev", 2),
    ("bbFatJetParT3PTopbWmv", 2),
    ("bbFatJetParT3PTopbWq", 2),
    ("bbFatJetParT3PTopbWqq", 2),
    ("bbFatJetParT3PTopbWtauhv", 2),
    ("bbFatJetParT3PXbb", 2),
    ("bbFatJetParT3PXcc", 2),
    ("bbFatJetParT3PXcs", 2),
    ("bbFatJetParT3PXqq", 2),
    ("bbFatJetParT3TXbb", 2),
    ("bbFatJetParT3massGeneric", 2),
    ("bbFatJetParT3massX2p", 2),
    # VBF jets
    ("VBFJetEta", 2),
    ("VBFJetPhi", 2),
    ("VBFJetMass", 2),
    ("VBFJetPt", 2),
    ("VBFJetrawFactor", 2),
    ("VBFJetbtagDeepFlavB", 2),
    ("VBFJetbtagPNetB", 2),
    ("VBFJetbtagPNetCvB", 2),
    ("VBFJetbtagPNetCvL", 2),
    ("VBFJetbtagPNetQvG", 2),
]


def parse_args():
    storage_dir = Path("/ceph/cms/store/user/zichun/bbbb")
    default_tag = "nanov15BDT_2024ggHHAndttbar_20260209_v15_signal"

    parser = argparse.ArgumentParser(description="Prepare Neural Network training dataframes.")
    parser.add_argument("--tag", default=default_tag)
    parser.add_argument("--years", nargs="+", default=["2024"])
    parser.add_argument("--txbb", default="glopart-v3")
    parser.add_argument("--input-dir", default=str(storage_dir / f"skimmer/{default_tag}"))
    parser.add_argument(
        "--output-dir", default=str(storage_dir / f"signal_processed/nn_training/{default_tag}")
    )
    parser.add_argument("--bdt-config", default="v13_glopartv3")
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
            # only apply to non-weight columns
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
    # test gets whatever remains so the total always equals len(events)

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

    bdt_dataframe_fn = importlib.import_module(
        f".{args.bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
    ).bdt_dataframe

    events_dict = {}
    for year in args.years:
        print(f"Processing year: {year}...")
        events_dict[year] = {}

        for sample, sample_list in SIGNAL_DICT.items():
            print(f"    Processing sample: {sample}")

            print("        Loading...")
            events = utils.load_samples(
                data_dir=str(input_dir),
                samples={sample: sample_list},
                year="2024" if (year == "2025" and sample != "data") else year,
                filters=filters_to_apply[args.txbb],
                columns=utils.format_columns(VAR_COLUMNS),
                reorder_txbb=False,
                txbb_str=txbb_strings[args.txbb],
                variations=False,
                weight_shifts=[""],
            )[sample]

            print("        Computing BDT input variables...")
            bdt_df = bdt_dataframe_fn(events, get_var_mapping(""))
            bdt_df.columns = pd.MultiIndex.from_tuples([(col, 0) for col in bdt_df.columns])

            events = cleanup_events(pd.concat([events, bdt_df], axis=1))

            # ── Split and save ────────────────────────────────────────────────
            splits = train_val_test_split(events, args.train_frac, args.val_frac, args.seed)
            splits["all"] = events
            for split_name, split_df in splits.items():
                save_path = output_dir / f"{split_name}/{year}_{sample}.parquet"
                save_path.parent.mkdir(exist_ok=True, parents=True)
                print(f"        [{split_name:5s}] {len(split_df):>7,} events  →  {save_path}")
                split_df.to_parquet(save_path)

            events_dict[year][sample] = events


if __name__ == "__main__":
    main()
