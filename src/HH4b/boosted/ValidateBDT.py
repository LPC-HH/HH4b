"""
Validate BDT versions
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve

import HH4b.utils as utils
from HH4b import hh_vars
from HH4b.event_selection import EventSelection
from HH4b.log_utils import log_config
from HH4b.plotting import multiROCCurveGrey
from HH4b.utils import get_var_mapping

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ValidateBDT")

jet_collection = "bbFatJet"  # ARG001
jet_index = 0  # ARG001

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


def load_events(path_to_dir, year, jet_coll_pnet, jet_coll_mass, bdt_models):
    logger.info(f"Load {year}")

    jet_collection = "bbFatJet"
    reorder_txbb = True
    txbb_str = jet_collection + jet_coll_pnet
    mass_str = jet_collection + jet_coll_mass

    sample_dirs = {
        year: {
            "qcd": [
                "QCD_HT-1000to1200",
                "QCD_HT-1200to1500",
                "QCD_HT-1500to2000",
                "QCD_HT-2000",
                "QCD_HT-400to600",
                "QCD_HT-600to800",
                "QCD_HT-800to1000",
            ],
            "ttbar": [
                "TTto4Q",
            ],
            "diboson": [
                "WW",
                "WZ",
                "ZZ",
            ],
            "VBFHH": [
                "VBFHHTo4B_CV_1_C2V_1_C3_1_TuneCP5_13TeV-madgraph-pythia8",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8",
                "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8",
                "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8",
                "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8",
                "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8",
                "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8",
                "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8",
                "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8",
                "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8",
            ],
            "VBFH": [
                "VBFHto2B_M-125_dipoleRecoilOn",
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
            (f"('{jet_collection}Pt', '0')", ">=", 300),
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

    # apply boosted selection
    event_selector = EventSelection(
        jet_collection=jet_collection,
        txbb_preselection=txbb_preselection,
        msd1_preselection=msd1_preselection,
        msd2_preselection=msd2_preselection,
    )

    events_dict = event_selector.apply_boosted(events_dict, txbb_str, mass_str)

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
    discriminator_name,
    bg_keys,
):
    sig_key = "hh4b"
    discriminator = f"{discriminator_name}"
    print(events_dict["ttbar"])
    # 1 for signal, 0 for background
    y_true = np.concatenate(
        [
            np.ones(len(events_dict[sig_key])),
            np.zeros(sum(len(events_dict[bg_key]) for bg_key in bg_keys)),
        ]
    )
    # weights
    weights = np.concatenate(
        [events_dict[sig_key]["finalWeight"]]
        + [events_dict[bg_key]["finalWeight"] for bg_key in bg_keys],
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
    bg_keys,
):
    y_true, scores, weights = get_roc_inputs(events_dict, discriminator_name, bg_keys)
    fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)

    # make sure fpr is sorted
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]

    roc = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "label": discriminator_label,  # + f" AUC ({auc(fpr, tpr):.4f})",
        "color": discriminator_color,
    }

    return roc


def get_legtitle(txbb_str):
    # title = r"FatJet p$_T^{(0,1)}$ > 250 GeV"
    title = r"FatJet p$_T^{0}$ > 300 GeV" + "\n"
    title += r"FatJet p$_T^{1}$ > 250 GeV" + "\n"
    title += "\n" + "$GloParT_{Xbb}^{0}$ > 0.3"
    title += "\n" + r"m$_{reg}$ > 50 GeV"
    title += "\n" + r"m$_{SD}^{0}$ > " + f"{msd1_preselection[txbb_str]} GeV"
    title += "\n" + r"m$_{SD}^{1}$ > " + f"{msd2_preselection[txbb_str]} GeV"

    return title


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def compute_bkg_effs(rocs, sig_effs):
    """
    Computes background efficiencies corresponding to the given signal efficiencies for each ROC curve.

    Args:
        rocs (dict): Dictionary of ROC curves.
        sig_effs (list of float): List of signal efficiencies.

    Returns:
        dict: Dictionary mapping model names to lists of background efficiencies.
    """
    bkg_effs_dict = {}
    for model_name, roc_dict in rocs.items():
        for _, roc in roc_dict.items():
            bkg_effs = []
            for sig_eff in sig_effs:
                print(f"Type of roc['tpr']: {type(roc['tpr'])}")
                print(f"Shape of roc['tpr']: {roc['tpr'].shape}")
                print(f"roc['tpr']: {roc['tpr']}")
                idx = _find_nearest(roc["tpr"], sig_eff)
                bkg_eff = roc["fpr"][idx]
                bkg_effs.append(bkg_eff)
            bkg_effs_dict[model_name] = bkg_effs
    return bkg_effs_dict


def restructure_rocs(rocs):
    """
    Restructure the 'rocs' dictionary to include an extra layer of nesting,
    as expected by the 'multiROCCurveGrey' function.

    Args:
        rocs (dict): Original dictionary of ROC curves.

    Returns:
        dict: Restructured dictionary with an extra layer of nesting.
    """
    new_rocs = {}
    for model_name, roc_dict in rocs.items():
        new_rocs[model_name] = {model_name: roc_dict}
    return new_rocs


def main(args):
    out_dir = Path(f"./bdt_comparison/{args.out_dir}/")
    out_dir.mkdir(exist_ok=True, parents=True)

    bdt_models = {
        "v5_PNetLegacy": {
            "config": "v5",
            "model_name": "24May31_lr_0p02_md_8_AK4Away",
        },
        # "v5_ParT": {
        #    "config": "v5_glopartv2",
        #    "model_name": "24Sep27_v5_glopartv2",
        # },
        # "v5_PNetv12": {
        #    "config": "v5_PNetv12",
        #    "model_name": "24Jul29_v5_PNetv12",
        # },
        # "v6_ParT": {
        #    "config": "v6_glopartv2",
        #    "model_name": "24Oct17_v6_glopartv2",
        # },
        "v5_ParT_rawmass": {
            "config": "v5_glopartv2",
            "model_name": "24Nov7_v5_glopartv2_rawmass",
        },
    }

    # distinguish between main backgrounds
    bkgs = ["qcd", "ttbar"] if args.bkgs == "all" else [args.bkgs]

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
    processes = ["hh4b"] + args.processes
    bdt_dict_combined = {
        key: pd.concat([bdt_dict[year][key] for year in bdt_dict]) for key in processes
    }

    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "cyan",
    ]
    rocs = {}
    for i, bdt_model in enumerate(bdt_models):

        rocs[bdt_model] = get_roc(
            bdt_dict_combined, f"bdtscore_{bdt_model}", bdt_model, colors[i], bg_keys=args.processes
        )

    # Plot multi-ROC curve
    bkgprocess_key = "-".join(args.processes)
    years_key = "_".join(args.year)
    output_name = f"PNet-parT-comparison-{bkgprocess_key}-{years_key}"
    print(output_name)
    multiROCCurveGrey(
        restructure_rocs(rocs),
        # sig_effs=sig_effs,
        # bkg_effs=bkg_effs,
        plot_dir=out_dir,
        legtitle=get_legtitle("bbFatJetParTTXbb"),
        title="ggF HH4b BDT ROC",
        name=output_name,
        plot_thresholds={
            "v5_PNetLegacy": [0.98, 0.88, 0.03],
            "v5_ParT_rawmass": [0.91, 0.64, 0.03],
        },
        # find_from_sigeff={0.98: [0.98, 0.88, 0.03]},
        # add_cms_label=True,
        show=True,
        xlim=[0, 1],
        ylim=[1e-5, 0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        nargs="+",
        type=str,
        default=["2022EE"],
        choices=hh_vars.years,
        help="years to evaluate on",
    )
    parser.add_argument(
        "--processes",
        nargs="+",
        type=str,
        default=["qcd"],
        choices=["qcd", "ttbar"],
        help="bkg processes to evaluate",
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
    parser.add_argument(
        "--bkgs",
        required=False,
        default="all",
        choices=["all", "qcd", "ttbar"],
        type=str,
        help="Backgrounds to include in the ROC curve",
    )
    args = parser.parse_args()
    sys.exit(main(args))
