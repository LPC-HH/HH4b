"""
Validate BDT versions
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, roc_curve

import HH4b.utils as utils
from HH4b import hh_vars
from HH4b.log_utils import log_config
from HH4b.utils import get_var_mapping

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ValidateBDT")

jet_collection = "bbFatJet"  # ARG001
jet_index = 0  # ARG001


def load_events(path_to_dir, year, jet_coll_pnet, jet_coll_mass, bdt_models):
    logger.info(f"Load {year}")

    jet_collection = "bbFatJet"
    reorder_txbb = True
    txbb_str = "bbFatJet" + jet_coll_pnet
    mass_str = "bbFatJet" + jet_coll_mass

    txbb_preselection = {
        "bbFatJetPNetTXbb": 0.3,
        "bbFatJetPNetTXbbLegacy": 0.8,
        "bbFatJetParTTXbb": 0.3,
    }
    msd1_preselection = {
        "bbFatJetPNetTXbb": 40,
        "bbFatJetPNetTXbbLegacy": 40,
        "bbFatJetParTTXbb": 40,
    }
    msd2_preselection = {
        "bbFatJetPNetTXbb": 30,
        "bbFatJetPNetTXbbLegacy": 0,
        "bbFatJetParTTXbb": 30,
    }

    sample_dirs = {
        year: {
            "qcd": [
                "QCD_HT-1000to1200",
                # "QCD_HT-1200to1500",
                # "QCD_HT-1500to2000",
                # "QCD_HT-2000",
                # "QCD_HT-400to600",
                # "QCD_HT-600to800",
                # "QCD_HT-800to1000",
            ],
            "ttbar": [
                "TTto4Q",
            ],
        },
    }
    sample_dirs_sig = {
        year: {
            "hh4b": [
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"
            ],  # the ? enforces exact matching
        }
    }
    # columns (or branches) to load: (branch_name, number of columns)
    # e.g. to load 2 jets ("ak8FatJetPt", 2)
    num_jets = 2

    columns = [
        ("weight", 1),  # genweight * otherweights
        ("event", 1),
        ("MET_pt", 1),
        ("bbFatJetTau3OverTau2", 2),
        ("VBFJetPt", 2),
        ("VBFJetEta", 2),
        ("VBFJetPhi", 2),
        ("VBFJetMass", 2),
        ("AK4JetAwayPt", 2),
        ("AK4JetAwayEta", 2),
        ("AK4JetAwayPhi", 2),
        ("AK4JetAwayMass", 2),
        (f"{jet_collection}Pt", num_jets),
        (f"{jet_collection}Msd", num_jets),
        (f"{jet_collection}Eta", num_jets),
        (f"{jet_collection}Phi", num_jets),
        (f"{jet_collection}PNetPXbbLegacy", num_jets),  # Legacy PNet
        (f"{jet_collection}PNetPQCDbLegacy", num_jets),
        (f"{jet_collection}PNetPQCDbbLegacy", num_jets),
        (f"{jet_collection}PNetPQCD0HFLegacy", num_jets),
        (f"{jet_collection}PNetMassLegacy", num_jets),
        (f"{jet_collection}PNetTXbbLegacy", num_jets),
        (f"{jet_collection}PNetTXbb", num_jets),  # 103X PNet
        (f"{jet_collection}PNetMass", num_jets),
        (f"{jet_collection}PNetQCD0HF", num_jets),
        (f"{jet_collection}PNetQCD1HF", num_jets),
        (f"{jet_collection}PNetQCD2HF", num_jets),
        (f"{jet_collection}ParTmassVis", num_jets),  # GloParT
        (f"{jet_collection}ParTTXbb", num_jets),
        (f"{jet_collection}ParTPXbb", num_jets),
        (f"{jet_collection}ParTPQCD0HF", num_jets),
        (f"{jet_collection}ParTPQCD1HF", num_jets),
        (f"{jet_collection}ParTPQCD2HF", num_jets),
    ]
    signal_exclusive_columns = []
    # selection to apply
    filters = [
        [
            (f"('{jet_collection}Pt', '0')", ">=", 250),
            (f"('{jet_collection}Pt', '1')", ">=", 250),
        ],
    ]

    # dictionary that will contain all information (from all samples)
    events_dict = {
        # this function will load files (only the columns selected), apply filters and compute a weight per event
        **utils.load_samples(
            path_to_dir,
            sample_dirs_sig[year],
            year,
            filters=filters,
            columns=utils.format_columns(columns + signal_exclusive_columns),
            reorder_txbb=reorder_txbb,
            txbb_str=txbb_str,
            variations=False,
        ),
        **utils.load_samples(
            path_to_dir,  # input directory
            sample_dirs[year],  # process_name: datasets
            year,  # year (to find corresponding luminosity)
            filters=filters,  # do not apply filter
            columns=utils.format_columns(
                columns
            ),  # columns to load from parquet (to not load all columns), IMPORTANT columns must be formatted: ("column name", "idx")
            reorder_txbb=reorder_txbb,  # whether to reorder bbFatJet collection
            txbb_str=txbb_str,
            variations=False,  # do not load systematic variations of weights
        ),
    }

    def apply_cuts(events_dict, txbb_str, mass_str):
        for key in events_dict:
            msd1 = events_dict[key]["bbFatJetMsd"][0]
            msd2 = events_dict[key]["bbFatJetMsd"][1]
            pt1 = events_dict[key]["bbFatJetPt"][0]
            pt2 = events_dict[key]["bbFatJetPt"][1]
            txbb1 = events_dict[key][txbb_str][0]
            mass1 = events_dict[key][mass_str][0]
            mass2 = events_dict[key][mass_str][1]
            # add msd > 40 cut for the first jet FIXME: replace this by the trigobj matched jet
            events_dict[key] = events_dict[key][
                (pt1 > 250)
                & (pt2 > 250)
                & (txbb1 > txbb_preselection[txbb_str])
                & (msd1 > msd1_preselection[txbb_str])
                & (msd2 > msd2_preselection[txbb_str])
                & (mass1 > 50)
                & (mass2 > 50)
            ].copy()
        return events_dict

    def get_bdt(events_dict, bdt_model, bdt_model_name, bdt_config, jlabel=""):
        bdt_model = xgb.XGBClassifier()
        bdt_model.load_model(
            fname=f"../boosted/bdt_trainings_run3/{bdt_model_name}/trained_bdt.model"
        )
        make_bdt_dataframe = importlib.import_module(
            f".{bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
        )
        bdt_events = make_bdt_dataframe.bdt_dataframe(events_dict, get_var_mapping(jlabel))
        preds = bdt_model.predict_proba(bdt_events)

        bdt_score = None
        bdt_score_vbf = None
        if preds.shape[1] == 2:  # binary BDT only
            bdt_score = preds[:, 1]
        elif preds.shape[1] == 3:  # multi-class BDT with ggF HH, QCD, ttbar classes
            bdt_score = preds[:, 0]  # ggF HH
        elif preds.shape[1] == 4:  # multi-class BDT with ggF HH, VBF HH, QCD, ttbar classes
            bg_tot = np.sum(preds[:, 2:], axis=1)
            bdt_score = preds[:, 0] / (preds[:, 0] + bg_tot)
            bdt_score_vbf = preds[:, 1] / (preds[:, 1] + preds[:, 2] + preds[:, 3])
        return bdt_score, bdt_score_vbf

    events_dict = apply_cuts(events_dict, txbb_str, mass_str)

    bdt_scores = []
    for bdt_model in bdt_models:
        logger.info(f"Perform inference {bdt_model}")
        bdt_config = bdt_models[bdt_model]["config"]
        bdt_model_name = bdt_models[bdt_model]["model_name"]
        for key in events_dict:
            bdt_score, bdt_score_vbf = get_bdt(
                events_dict[key], bdt_model, bdt_model_name, bdt_config
            )
            events_dict[key][f"bdtscore_{bdt_model}"] = (
                bdt_score if bdt_score is not None else np.ones(events_dict[key]["weight"])
            )
            events_dict[key][f"bdtscoreVBF_{bdt_model}"] = (
                bdt_score if bdt_score is not None else np.ones(events_dict[key]["weight"])
            )
            bdt_scores.extend([f"bdtscore_{bdt_model}", f"bdtscoreVBF_{bdt_model}"])

    # Add finalWeight to the list of columns being retained
    bdt_scores.append("finalWeight")

    return {key: events_dict[key][bdt_scores] for key in events_dict}


def get_roc_inputs(
    events_dict,
    # jet_collection,
    discriminator_name,
    # jet_index,
):
    sig_key = "hh4b"
    bg_keys = ["qcd"]
    discriminator = f"{discriminator_name}"  # f"{jet_collection}{discriminator_name}"

    # 1 for signal, 0 for background
    y_true = np.concatenate(
        [
            np.ones(len(events_dict[sig_key])),
            np.zeros(sum(len(events_dict[bg_key]) for bg_key in bg_keys)),
        ]
    )
    # weights
    weights = np.concatenate(
        [events_dict[sig_key]["finalWeight"]]  # subst finalWeight->weight
        + [events_dict[bg_key]["finalWeight"] for bg_key in bg_keys],  # subst finalWeight->weight
    )
    # discriminator
    # print(events_dict[sig_key][discriminator])
    scores = np.concatenate(
        [
            events_dict[sig_key][discriminator].iloc[:, 0]
        ]  # flatten duplicated bdt scores (n_events,3) to (n_events,)
        + [events_dict[bg_key][discriminator].iloc[:, 0] for bg_key in bg_keys],
    )
    return y_true, scores, weights


def get_roc(
    events_dict,
    discriminator_name,
    discriminator_label,
    discriminator_color,
):
    y_true, scores, weights = get_roc_inputs(
        events_dict, jet_collection, discriminator_name, jet_index
    )
    fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)
    roc = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "label": discriminator_label + f" AUC ({auc(fpr, tpr):.4f})",
        "color": discriminator_color,
    }

    return roc


def plot_roc(rocs, out_dir):
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed

    # Plot each ROC curve
    for roc in rocs:
        plt.plot(roc["tpr"], roc["fpr"], label=roc["label"], color=roc["color"])

    # Set x and y axis labels
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")

    # Set y-axis to log scale and specify y-axis limits
    plt.yscale("log")
    plt.ylim([1e-5, 1e-1])

    # Set x-axis limits (no need to set x-scale, as it is linear by default)
    plt.xlim([0, 0.6])

    # Add legend and gridlines
    plt.legend(loc="best")
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Save the plot
    plt.savefig(out_dir / "rocs.png")

    # Close the figure to free memory
    plt.close()


def main(args):
    out_dir = Path(f"./bdt_comparison/{args.out_dir}/")
    out_dir.mkdir(exist_ok=True, parents=True)

    bdt_models = {
        "v5_PNetLegacy": {
            "config": "v5",
            "model_name": "24May31_lr_0p02_md_8_AK4Away",
        },
        "v5_ParT": {
            "config": "v5_glopartv2",
            "model_name": "24Sep27_v5_glopartv2",
        },
    }

    bdt_dict = {
        year: load_events(
            args.data_path,
            year,
            jet_coll_pnet="ParTTXbb",
            jet_coll_mass="ParTmassVis",
            bdt_models=bdt_models,
        )
        for year in args.year
    }
    processes = ["qcd", "ttbar", "hh4b"]
    bdt_dict_combined = {
        key: pd.concat([bdt_dict[year][key] for year in bdt_dict]) for key in processes
    }
    print("BDT_dict_combined")
    print(bdt_dict_combined)

    colors = ["blue", "red"]
    rocs = {}
    for i, bdt_model in enumerate(bdt_models):
        rocs[bdt_model] = get_roc(
            bdt_dict_combined,
            f"bdtscore_{bdt_model}",
            bdt_model,
            colors[i],
        )
    print("ROCS")
    print(rocs)
    plot_roc(rocs.values(), out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        nargs="+",
        type=str,
        default=["2022EE"],
        choices=hh_vars.years,
        help="years to train on",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="path to training data",
        type=str,
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="path to save plots",
        type=str,
    )
    args = parser.parse_args()
    sys.exit(main(args))
