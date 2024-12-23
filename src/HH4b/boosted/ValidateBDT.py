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
from sklearn.metrics import auc, roc_curve

import HH4b.utils as utils
from HH4b import hh_vars
from HH4b.event_selection import EventSelection
from HH4b.log_utils import log_config
from HH4b.plotting import multiROCCurveGrey
from HH4b.utils import get_var_mapping

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ValidateBDT")


def load_events(path_to_dir, year, jet_coll_tagger, jet_coll_mass, bdt_models):
    logger.info(f"Load {year}")

    event_sel = EventSelection(year, jet_coll_tagger, jet_coll_mass)

    sample_dirs, sample_dirs_sig = event_sel.get_samples()
    columns, sig_exclusive_columns = event_sel.get_columns()

    # dictionary that will contain all information (from all samples)
    events_dict = {
        # this function will load files (only the columns selected), apply filters and compute a weight per event
        **utils.load_samples(
            path_to_dir,
            sample_dirs_sig[year],
            year,
            filters=event_sel.filters,
            columns=utils.format_columns(columns + sig_exclusive_columns),
            reorder_txbb=event_sel.reorder_txbb,
            txbb_str=event_sel.txbb_str,
            variations=False,
        ),
        **utils.load_samples(
            path_to_dir,  # input directory
            sample_dirs[year],  # process_name: datasets
            year,  # year (to find corresponding luminosity)
            filters=event_sel.filters,  # do not apply filter
            columns=utils.format_columns(
                columns
            ),  # columns to load from parquet (to not load all columns), IMPORTANT columns must be formatted: ("column name", "idx")
            reorder_txbb=event_sel.reorder_txbb,  # whether to reorder bbFatJet collection
            txbb_str=event_sel.txbb_str,
            variations=False,  # do not load systematic variations of weights
        ),
    }

    # apply boosted selection
    events_dict = event_sel.apply_boosted(events_dict)
    print(events_dict.keys())

    # remove once done in pre-processing!
    def correct_mass(events_dict, mass_str):
        for key in events_dict:
            events_dict[key][(mass_str, 0)] = events_dict[key][(mass_str, 0)] * (
                1 - events_dict[key][("bbFatJetrawFactor", 0)]
            )
            events_dict[key][(mass_str, 1)] = events_dict[key][(mass_str, 1)] * (
                1 - events_dict[key][("bbFatJetrawFactor", 1)]
            )

    correct_mass(events_dict, "bbFatJetParTmassVis")

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

    return {key: events_dict[key][bdt_scores] for key in events_dict}, event_sel


def get_roc_inputs(
    events_dict,
    # jet_collection,
    discriminator_name,
    # jet_index,
    bkgs,
):
    sig_key = "hh4b"

    # bg_keys = ["ttbar"]
    bg_keys = bkgs

    # bg_keys = ["ttbar"]
    bg_keys = bkgs
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
    bkgs,
):
    y_true, scores, weights = get_roc_inputs(events_dict, discriminator_name, bkgs)
    fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)
    # make sure fpr is sorted
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]

    roc = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "label": discriminator_label + f" AUC ({auc(fpr, tpr):.4f})",
        "color": discriminator_color,
    }

    return roc


def get_legtitle(event_sel: EventSelection):
    # title = r"FatJet p$_T^{(0,1)}$ > 250 GeV"
    txbb_str = event_sel.txbb_str
    msd1_preselection = event_sel.msd1_preselection
    msd2_preselection = event_sel.msd2_preselection

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


# TODO: obsolete?
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

    bdt_dict = {}
    event_sel = None
    for year in args.year:
        event_dict, event_sel = load_events(
            args.data_path,
            year,
            jet_coll_tagger="ParTTXbb",
            jet_coll_mass="ParTmassVis",
            bdt_models=bdt_models,
        )
        bdt_dict[year] = event_dict
    processes = ["qcd", "ttbar", "hh4b"]
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
            bdt_dict_combined,
            f"bdtscore_{bdt_model}",
            bdt_model,
            colors[i],
            bkgs,
        )

    # Plot multi-ROC curve

    multiROCCurveGrey(
        restructure_rocs(rocs),
        # sig_effs=sig_effs,
        # bkg_effs=bkg_effs,
        plot_dir=out_dir,
        legtitle=get_legtitle(event_sel),
        title="ggF HH4b BDT ROC",
        name=f"PNet-parT-comparison-{args.bkgs}",
        plot_thresholds={
            "v5_PNetLegacy": [0.98, 0.88, 0.03],
            "v5_ParT_rawmass": [0.91, 0.64, 0.03],
            "ParTTXbb": [0.7475, 0.775, 0.9375],
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
        help="years to train on",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        default="/home/users/dprimosc/data/24Sep25_v12v2_private_signal",
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
