"""
Validate AK8 Jet bb Taggers
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

import HH4b.utils as utils
from HH4b import plotting


def load_events(path_to_dir, year, jet_collection, pt_cut, msd_cut, jet_coll_pnet, match_higgs):
    check_mass = False
    add_ttbar = False

    sample_dirs = {
        year: {
            "qcd": [
                "QCD_HT-1000to1200",
                #'QCD_HT-100to200',
                "QCD_HT-1200to1500",
                "QCD_HT-1500to2000",
                "QCD_HT-2000",
                #'QCD_HT-200to400',
                "QCD_HT-400to600",
                "QCD_HT-600to800",
                "QCD_HT-800to1000",
            ],
        }
    }
    if add_ttbar:
        sample_dirs[year]["ttbar"] = ["TTTo4Q"]
    sample_dirs_sig = {
        year: {
            "hh4b": [
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"
            ],  # the ? enforces exact matching
            # "gghtobb": ["GluGluHto2B_PT-200_M-125"],
        }
    }
    # columns (or branches) to load: (branch_name, number of columns)
    # e.g. to load 2 jets ("ak8FatJetPt", 2)
    columns = [
        ("weight", 1),  # genweight * otherweights
        (f"{jet_collection}Pt", 2),
        (f"{jet_collection}Msd", 2),
        (f"{jet_collection}PNetMassLegacy", 2),  # ParticleNet Legacy regressed mass
    ]
    if check_mass:
        columns += [
            (f"{jet_collection}PNetMass", 2),  # ParticleNet 130x regressed mass
            (f"{jet_collection}ParTmassRes", 2),  # GloParTv2  regressed mass (resonant)
            (f"{jet_collection}ParTmassVis", 2),  # GloParTv2  regressed mass (particles)
        ]
    else:
        columns += [
            (f"{jet_collection}PNetTXbb", 2),  # ParticleNet 130x bb vs QCD discriminator
            (f"{jet_collection}PNetTXbbLegacy", 2),  # ParticleNet Legacy bb vs QCD discriminator
            (f"{jet_collection}ParTTXbb", 2),  # GloParTv2 bb vs QCD discriminator
        ]

    # additional columns for mathching hh4b jets
    signal_exclusive_columns = (
        [
            (f"{jet_collection}HiggsMatchIndex", 2),  # index of higgs matched to jet
            (f"{jet_collection}NumBMatchedH1", 2),  # number of bquarks matched to H1
            (f"{jet_collection}NumBMatchedH2", 2),  # number of bquarks matched to H2
        ]
        if match_higgs
        else []
    )

    # apply selection on jet index
    filters = [
        [
            (f"('{jet_collection}Pt', '0')", ">=", pt_cut[0]),
            (f"('{jet_collection}Pt', '0')", "<=", pt_cut[1]),
            (f"('{jet_collection}Msd', '0')", ">=", msd_cut[0]),
            (f"('{jet_collection}Msd', '0')", "<=", msd_cut[1]),
        ],
        [
            (f"('{jet_collection}Pt', '1')", ">=", pt_cut[0]),
            (f"('{jet_collection}Pt', '1')", "<=", pt_cut[1]),
            (f"('{jet_collection}Msd', '1')", ">=", msd_cut[0]),
            (f"('{jet_collection}Msd', '1')", "<=", msd_cut[1]),
        ],
    ]

    reorder_txbb = False if jet_collection == "ak8FatJet" else True
    txbb = "bbFatJet" + jet_coll_pnet

    # dictionary that will contain all information (from all samples)
    events_dict = {
        # this function will load files (only the columns selected), apply filters and compute a weight per event
        **utils.load_samples(
            path_to_dir,  # input directory
            sample_dirs[year],  # process_name: datasets
            year,  # year (to find corresponding luminosity)
            filters=filters,  # do not apply filter
            columns=utils.format_columns(
                columns
            ),  # columns to load from parquet (to not load all columns), IMPORTANT columns must be formatted: ("column name", "idx")
            reorder_txbb=reorder_txbb,  # whether to reorder bbFatJet collection
            txbb=txbb,
            variations=False,  # do not load systematic variations of weigths
        ),
        **utils.load_samples(
            path_to_dir,
            sample_dirs_sig[year],
            year,
            filters=filters,
            columns=utils.format_columns(columns + signal_exclusive_columns),
            reorder_txbb=reorder_txbb,
            txbb=txbb,
            variations=False,
        ),
    }

    def get_hh4bmatched(events_dict):
        events = events_dict["hh4b"]
        indexak8 = events[
            f"{jet_collection}HiggsMatchIndex"
        ].to_numpy()  # index of higgs matched to jet
        nbh1ak8 = events[
            f"{jet_collection}NumBMatchedH1"
        ].to_numpy()  # number of bquarks matched to H1
        nbh2ak8 = events[
            f"{jet_collection}NumBMatchedH2"
        ].to_numpy()  # number of bquarks matched to H2

        matched_to_h1 = (indexak8 == 0) & (nbh1ak8 == 2)
        matched_to_h2 = (indexak8 == 1) & (nbh2ak8 == 2)
        matchedak8 = matched_to_h1 | matched_to_h2
        return matchedak8

    if match_higgs:
        events_dict["hh4b"] = events_dict["hh4b"][get_hh4bmatched(events_dict)]

    return events_dict


def get_roc_inputs(
    events_dict, jet_collection, discriminator_name, jet_index, sig_key="hh4b", bg_keys=["qcd"]
):
    discriminator = f"{jet_collection}{discriminator_name}"

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
    scores = np.concatenate(
        [events_dict[sig_key][discriminator][jet_index]]
        + [events_dict[bg_key][discriminator][jet_index] for bg_key in bg_keys],
    )
    return y_true, scores, weights


def get_roc(
    events_dict,
    jet_collection,
    discriminator_name,
    discriminator_label,
    discriminator_color,
    jet_indices,
    pt_cut,
    msd_cut,
    mreg_cut=None,
):
    y_true_arr = []
    scores_arr = []
    weights_arr = []
    for jet_index in jet_indices:
        events_dict_masked = {
            key: events[
                (events[f"{jet_collection}Pt"][jet_index] >= pt_cut[0])
                & (events[f"{jet_collection}Pt"][jet_index] <= pt_cut[1])
                & (events[f"{jet_collection}Msd"][jet_index] >= msd_cut[0])
                & (events[f"{jet_collection}Msd"][jet_index] <= msd_cut[1])
            ]
            for key, events in events_dict.items()
        }
        if mreg_cut:
            events_dict_masked = {
                key: events[
                    (events[f"{jet_collection}PNetMassLegacy"][jet_index] >= mreg_cut[0])
                    & (events[f"{jet_collection}PNetMassLegacy"][jet_index] <= mreg_cut[1])
                ]
                for key, events in events_dict.items()
            }
        y_true_i, scores_i, weights_i = get_roc_inputs(
            events_dict_masked, jet_collection, discriminator_name, jet_index
        )
        y_true_arr.append(y_true_i)
        scores_arr.append(scores_i)
        weights_arr.append(weights_i)
    y_true = np.concatenate(y_true_arr)
    scores = np.concatenate(scores_arr)
    weights = np.concatenate(weights_arr)
    fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)
    roc = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "label": discriminator_label + f" AUC ({auc(fpr, tpr):.4f})",
        "color": discriminator_color,
    }

    return roc


def main(args):
    jet_collection = "ak8FatJet"
    jet_coll_pnet = ""
    match_higgs = True

    jet_collection = "bbFatJet"
    jet_coll_pnet = "PNetTXbb"
    match_higgs = False

    MAIN_DIR = "/eos/uscms/store/user/cmantill/bbbb/skimmer/"
    tag = "24Sep19_v12v2_private_pre-sel"
    year = "2022"
    outdir = "24Sep24"  # date of plotting
    plot_dir = f"/uscms/home/cmantill/nobackup/hh/HH4b/plots/PostProcessing/{outdir}/{year}"
    _ = os.system(f"mkdir -p {plot_dir}")
    path_to_dir = f"{MAIN_DIR}/{tag}/"

    pt_cut = [int(x) for x in args.pt_cut.split(",")]
    msd_cut = [int(x) for x in args.msd_cut.split(",")]
    cut_str = "pt" + "-".join(str(x) for x in pt_cut) + "msd" + "-".join(str(x) for x in msd_cut)
    legtitle = (
        rf"{pt_cut[0]} < $p_T$ < {pt_cut[1]} GeV"
        + "\n"
        + f" {msd_cut[0]}"
        + r" < $m_{SD}$ <"
        + f" {msd_cut[1]} GeV"
    )
    mreg_cut = None if args.mreg_cut == "" else [int(x) for x in args.mreg_cut.split(",")]
    if mreg_cut:
        legtitle += "\n " + f"{mreg_cut[0]} <" + r" $m_{reg}$ (Legacy) <" + f" {mreg_cut[1]} GeV"
        cut_str += "mregleg" + "-".join(str(x) for x in mreg_cut)

    events_dict = load_events(
        path_to_dir, year, jet_collection, pt_cut, msd_cut, jet_coll_pnet, match_higgs
    )

    for jet_indices in [[0, 1], [0], [1]]:
        rocs = {
            "PNetTXbbLegacy": get_roc(
                events_dict,
                jet_collection,
                "PNetTXbbLegacy",
                "ParticleNet Legacy Hbb vs QCD",
                "blue",
                jet_indices,
                pt_cut,
                msd_cut,
                mreg_cut,
            ),
            "PNetTXbb": get_roc(
                events_dict,
                jet_collection,
                "PNetTXbb",
                "ParticleNet 103X Hbb vs QCD",
                "orange",
                jet_indices,
                pt_cut,
                msd_cut,
                mreg_cut,
            ),
            "ParTTXbb": get_roc(
                events_dict,
                jet_collection,
                "ParTTXbb",
                "GloParTv2 Hbb vs QCD",
                "red",
                jet_indices,
                pt_cut,
                msd_cut,
                mreg_cut,
            ),
        }
        # thresholds on the discriminator, used to search for signal efficiency
        plot_thresholds = {
            "PNetTXbbLegacy": [0.8],
            "PNetTXbb": [0.7],
            # "ParTTXbb": [0.38],
        }
        # find what the threshold should be to achieve this signal efficiency
        find_from_sigeff = {
            "PNetTXbb": [0.85],
            "ParTTXbb": [0.85],
            # "PNetTXbb": [0.72],
            # "ParTTXbb": [0.72],
        }
        plotting.multiROCCurveGrey(
            {"bb": rocs},
            sig_effs=[0.6],
            xlim=[0, 1.0],
            ylim=[1e-4, 1],
            show=True,
            plot_dir=Path(plot_dir),
            name=f"{jet_collection}{jet_coll_pnet}ROC{''.join(str(x) for x in jet_indices)}_{cut_str}",
            title=(
                f"AK8 Jets {jet_indices}"
                if jet_collection == "ak8FatJet"
                else f"bb Jets {jet_indices}"
            ),
            legtitle=legtitle,
            plot_thresholds=plot_thresholds,
            find_from_sigeff=find_from_sigeff,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt-cut",
        help="pt cut split by ,",
        required=True,
    )
    parser.add_argument(
        "--msd-cut",
        help="msd cut split by ,",
        required=True,
    )
    parser.add_argument("--mreg-cut", help="mreg cut split by ,", default="")
    args = parser.parse_args()
    sys.exit(main(args))
