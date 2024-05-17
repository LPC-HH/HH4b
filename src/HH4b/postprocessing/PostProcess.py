from __future__ import annotations

import argparse
import importlib
import pprint
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b import hh_vars, plotting, postprocessing, run_utils
from HH4b.boosted.TrainBDT import get_legtitle
from HH4b.hh_vars import LUMI, bg_keys, samples_run3, years  # noqa: F401
from HH4b.postprocessing import (
    Region,
    combine_run3_samples,
    load_run3_samples,
)
from HH4b.utils import ShapeVar, singleVarHist

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

mpl.rcParams["font.size"] = 30
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["grid.color"] = "#CCCCCC"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["figure.edgecolor"] = "none"

# modify samples run3
for year in samples_run3:
    samples_run3[year]["qcd"] = [
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
        "QCD_HT-2000",
        # "QCD_HT-200to400",
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
    ]

selection_regions = {
    "pass_vbf": Region(
        cuts={
            "Category": [0, 1],
        },
        label="VBF",
    ),
    "pass_bin1": Region(
        cuts={
            "Category": [1, 2],
        },
        label="Bin1",
    ),
    "pass_bin2": Region(
        cuts={
            "Category": [2, 3],
        },
        label="Bin2",
    ),
    "pass_bin3": Region(
        cuts={
            "Category": [3, 4],
        },
        label="Bin3",
    ),
    "fail": Region(
        cuts={
            "Category": [4, 5],
        },
        label="Fail",
    ),
}


label_by_mass = {
    "H2Msd": r"$m^{2}_\mathrm{SD}$ (GeV)",
    "H2PNetMass": r"$m^{2}_\mathrm{reg}$ (GeV)",
}


def get_bdt_training_keys(bdt_model: str):
    inferences_dir = Path(f"../boosted/bdt_trainings_run3/{bdt_model}/inferences/2022EE")

    training_keys = []
    for child in inferences_dir.iterdir():
        if child.suffix == ".npy":
            training_keys.append(child.stem.split("evt_")[-1])

    print("Found BDT Training keys", training_keys)
    return training_keys


def get_key_map(jshift: str = ""):

    def key_map(variable: str):
        if jshift in hh_vars.jec_shifts and variable in hh_vars.jec_vars:
            return f"{variable}_{jshift}"
        elif jshift in hh_vars.jmsr_shfits and variable in hh_vars.jsmr_vars:
            return f"{variable}_{jshift}"
        return variable        

    return key_map


def add_bdt_scores(events: pd.DataFrame, preds: np.ArrayLike, jshift: str = ""):
    jshift_under = "_" + jshift if jshift != "" else ""

    if preds.shape[1] == 2:  # binary BDT only
        events[f"bdt_score{jshift_under}"] = preds[:, 1]
    elif preds.shape[1] == 3:  # multi-class BDT with ggF HH, QCD, ttbar classes
        events[f"bdt_score{jshift_under}"] = preds[:, 0]  # ggF HH
    elif preds.shape[1] == 4:  # multi-class BDT with ggF HH, VBF HH, QCD, ttbar classes
        bg_tot = np.sum(preds[:, 2:], axis=1)
        events[f"bdt_score{jshift_under}"] = preds[:, 0] / (preds[:, 0] + bg_tot)
        events[f"bdt_score_vbf{jshift_under}"] = preds[:, 1] / (preds[:, 1] + bg_tot)


def bdt_roc(events_combined: dict[str, pd.DataFrame], plot_dir: str, legacy: bool):
    sig_keys = ["hh4b", "vbfhh4b-k2v0"]
    scores_keys = {
        "hh4b": "bdt_score",
        "vbfhh4b-k2v0": "bdt_score_vbf",
    }
    bkg_keys = ["qcd", "ttbar"]
    legtitle = get_legtitle(legacy, pnet_xbb_str="Legacy")

    if "bdt_score_vbf" not in events_combined["ttbar"]:
        sig_keys.remove("vbfhh4b-k2v0")

    for sig_key in sig_keys:
        rocs = postprocessing.make_rocs(
            events_combined, scores_keys[sig_key], "weight", sig_key, bkg_keys
        )
        bkg_colors = {**plotting.color_by_sample, "merged": "orange"}
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        for bg_key in [*bkg_keys, "merged"]:
            ax.plot(
                rocs[bg_key]["tpr"],
                rocs[bg_key]["fpr"],
                linewidth=2,
                color=bkg_colors[bg_key],
                label=rocs[bg_key]["label"],
            )

        ax.set_xlim([0.0, 0.6])
        ax.set_ylim([1e-5, 1e-1])
        ax.set_yscale("log")

        ax.set_title(f"{plotting.label_by_sample[sig_key]} BDT ROC Curve")
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel("Background efficiency")

        ax.xaxis.grid(True, which="major")
        ax.yaxis.grid(True, which="major")
        ax.legend(
            title=legtitle,
            bbox_to_anchor=(1.03, 1),
            loc="upper left",
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{sig_key}_roc.png")
        fig.savefig(plot_dir / f"{sig_key}_roc.pdf", bbox_inches="tight")
        plt.close()


def load_process_run3_samples(args, year, bdt_training_keys, control_plots, plot_dir):
    legacy_label = "Legacy" if args.legacy else ""

    events_dict = load_run3_samples(
        f"{args.data_dir}/{args.tag}",
        year,
        args.legacy,
        samples_run3,
        reorder_txbb=True,
        txbb=f"bbFatJetPNetTXbb{legacy_label}",
    )

    cutflow = pd.DataFrame(index=list(events_dict.keys()))
    cutflow_print = pd.DataFrame(index=list(events_dict.keys()))
    cutflow_dict = {
        key: OrderedDict(
            [("Skimmer Preselection", np.sum(events_dict[key]["finalWeight"].to_numpy()))]
        )
        for key in events_dict
    }

    # define BDT model
    bdt_model = xgb.XGBClassifier()
    bdt_model.load_model(fname=f"../boosted/bdt_trainings_run3/{args.bdt_model}/trained_bdt.model")
    # get function
    make_bdt_dataframe = importlib.import_module(
        f".{args.bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
    )

    # inference and assign score
    events_dict_postprocess = {}
    for key in events_dict:
        if "hh4b" in key:
            jshifts = [""] + hh_vars.jec_shifts
        else:
            jshifts = [""]

        bdt_events = {}
        for jshift in jshifts:
            bdt_events[jshift] = make_bdt_dataframe.bdt_dataframe(
                events_dict[key], get_key_map(jshift)
            )
            preds = bdt_model.predict_proba(bdt_events[jshift])
            add_bdt_scores(bdt_events[jshift], preds, jshift)
        bdt_events = pd.concat([bdt_events[jshift] for jshift in jshifts], axis=1)
        print(bdt_events)
        print(bdt_events.columns)
        bdt_events["H1Pt"] = events_dict[key]["bbFatJetPt"].to_numpy()[:, 0]
        bdt_events["H2Pt"] = events_dict[key]["bbFatJetPt"].to_numpy()[:, 1]
        bdt_events["H1Msd"] = events_dict[key]["bbFatJetMsd"].to_numpy()[:, 0]
        bdt_events["H2Msd"] = events_dict[key]["bbFatJetMsd"].to_numpy()[:, 1]
        bdt_events["H1TXbb"] = events_dict[key][f"bbFatJetPNetTXbb{legacy_label}"].to_numpy()[:, 0]
        bdt_events["H2TXbb"] = events_dict[key][f"bbFatJetPNetTXbb{legacy_label}"].to_numpy()[:, 1]
        bdt_events["H1PNetMass"] = events_dict[key][f"bbFatJetPNetMass{legacy_label}"].to_numpy()[
            :, 0
        ]
        bdt_events["H2PNetMass"] = events_dict[key][f"bbFatJetPNetMass{legacy_label}"].to_numpy()[
            :, 1
        ]
        bdt_events["H1TXbbNoLeg"] = events_dict[key]["bbFatJetPNetTXbb"].to_numpy()[:, 0]
        bdt_events["H2TXbbNoLeg"] = events_dict[key]["bbFatJetPNetTXbb"].to_numpy()[:, 1]

        # add HLTs
        bdt_events["hlt"] = np.any(
            np.array(
                [
                    events_dict[key][trigger].to_numpy()[:, 0]
                    for trigger in postprocessing.HLTs[year]
                    if trigger in events_dict[key]
                ]
            ),
            axis=0,
        )

        # weights
        # finalWeight: includes genWeight, puWeight
        # FIXME: genWeight taken only as sign for HH sample...
        bdt_events["weight"] = events_dict[key]["finalWeight"].to_numpy()

        ## Add TTBar Weight here TODO: does this need to be re-measured for legacy PNet Mass?
        # if key == "ttbar" and not args.legacy:
        #    bdt_events["weight"] *= corrections.ttbar_pTjjSF(year, events_dict, "bbFatJetPNetMass")

        # add selection to testing events
        bdt_events["event"] = events_dict[key]["event"].to_numpy()[:, 0]
        if (
            args.training_years is not None
            and year in args.training_years
            and key in bdt_training_keys
        ):
            print(f"year {year} used in training")
            inferences_dir = Path(
                f"../boosted/bdt_trainings_run3/{args.bdt_model}/inferences/{year}"
            )

            evt_list = np.load(inferences_dir / f"evt_{key}.npy")
            bdt_events = bdt_events[bdt_events["event"].isin(evt_list)]
            bdt_events["weight"] *= 1 / 0.4  # divide by BDT test / train ratio

        # HLT selection
        mask_hlt = bdt_events["hlt"] == 1
        bdt_events = bdt_events[mask_hlt]
        cutflow_dict[key]["HLT"] = np.sum(bdt_events["weight"].to_numpy())

        mask_presel = (
            (bdt_events["H1Msd"] >= 40)  # FIXME: replace by jet matched to trigger object
            & (bdt_events["H1Pt"] >= 300)
            & (bdt_events["H2Pt"] >= args.pt_second)
            & (bdt_events["H1TXbb"] >= 0.8)
            & (bdt_events[args.mass] >= 60)
            & (bdt_events[args.mass] <= 220)
            & (bdt_events[args.mass.replace("H2", "H1")] >= 60)
            & (bdt_events[args.mass.replace("H2", "H1")] <= 220)
        )
        bdt_events = bdt_events[mask_presel]
        cutflow_dict[key][f"H1Msd > 40 & H2Pt > {args.pt_second}"] = np.sum(
            bdt_events["weight"].to_numpy()
        )

        cutflow_dict[key]["BDT > min"] = np.sum(
            bdt_events["weight"][bdt_events["bdt_score"] > args.bdt_wps[2]].to_numpy()
        )

        ###### FINISH pre-selection
        mass_window = [110, 140]
        mass_str = f"[{mass_window[0]}-{mass_window[1]}]"
        mask_mass = (bdt_events[args.mass] >= mass_window[0]) & (
            bdt_events[args.mass] <= mass_window[1]
        )

        # define category
        bdt_events["Category"] = 5  # all events
        if args.vbf:
            mask_vbf = (bdt_events["bdt_score_vbf"] > args.vbf_bdt_wp) & (
                bdt_events["H2TXbb"] > args.vbf_txbb_wp
            )
        else:
            # if no VBF region, set all events to "fail VBF"
            mask_vbf = np.zeros(len(bdt_events), dtype=bool)

        mask_bin1 = (
            (bdt_events["H2TXbb"] > args.txbb_wps[0])
            & (bdt_events["bdt_score"] > args.bdt_wps[0])
            # & ~(mask_vbf)
        )

        if args.vbf_priority:
            # prioritize VBF region i.e. veto events in bin1 that pass the VBF selection
            mask_bin1 = mask_bin1 & ~(mask_vbf)
        else:
            # prioritize bin 1 i.e. veto events in VBF region that pass the bin 1 selection
            mask_vbf = mask_vbf & ~(mask_bin1)

        bdt_events.loc[mask_vbf, "Category"] = 0
        cutflow_dict[key][f"Bin VBF {mass_str}"] = np.sum(
            bdt_events["weight"][mask_vbf & mask_mass].to_numpy()
        )
        cutflow_dict[key]["Bin VBF"] = np.sum(bdt_events["weight"][mask_vbf].to_numpy())
        cutflow_dict[key][f"Bin VBF {mass_str}"] = np.sum(
            bdt_events["weight"][mask_vbf & mask_mass].to_numpy()
        )

        bdt_events.loc[mask_bin1, "Category"] = 1
        cutflow_dict[key]["Bin 1"] = np.sum(bdt_events["weight"][mask_bin1].to_numpy())
        cutflow_dict[key][f"Bin 1 {mass_str}"] = np.sum(
            bdt_events["weight"][mask_bin1 & mask_mass].to_numpy()
        )

        cutflow_dict[key]["VBF & Bin 1 overlap"] = np.sum(
            bdt_events["weight"][
                (bdt_events["H2TXbb"] > args.txbb_wps[0])
                & (bdt_events["bdt_score"] > args.bdt_wps[0])
                & mask_vbf
            ].to_numpy()
        )

        mask_corner = (bdt_events["H2TXbb"] < args.txbb_wps[0]) & (
            bdt_events["bdt_score"] < args.bdt_wps[0]
        )
        mask_bin2 = (
            (bdt_events["H2TXbb"] > args.txbb_wps[1])
            & (bdt_events["bdt_score"] > args.bdt_wps[1])
            & ~(mask_bin1)
            & ~(mask_corner)
            & ~(mask_vbf)
        )
        bdt_events.loc[mask_bin2, "Category"] = 2
        cutflow_dict[key]["Bin 2"] = np.sum(bdt_events["weight"][mask_bin2].to_numpy())
        cutflow_dict[key][f"Bin 2 {mass_str}"] = np.sum(
            bdt_events["weight"][mask_bin2 & mask_mass].to_numpy()
        )

        mask_bin3 = (
            (bdt_events["H2TXbb"] > args.txbb_wps[1])
            & (bdt_events["bdt_score"] > args.bdt_wps[2])
            & ~(mask_bin1)
            & ~(mask_bin2)
            & ~(mask_vbf)
        )
        bdt_events.loc[mask_bin3, "Category"] = 3
        cutflow_dict[key]["Bin 3"] = np.sum(bdt_events["weight"][mask_bin3].to_numpy())
        cutflow_dict[key][f"Bin 3 {mass_str}"] = np.sum(
            bdt_events["weight"][mask_bin3 & mask_mass].to_numpy()
        )

        mask_fail = (bdt_events["H2TXbb"] < args.txbb_wps[1]) & (
            bdt_events["bdt_score"] > args.bdt_wps[2]
        )
        bdt_events.loc[mask_fail, "Category"] = 4

        # keep some (or all) columns
        columns = ["Category", "H2Msd", "bdt_score", "H2TXbb", "H2PNetMass", "weight"]
        if "bdt_score_vbf" in bdt_events:
            columns += ["bdt_score_vbf"]

        if control_plots:
            bdt_events["H1T32top"] = bdt_events["H1T32"]
            bdt_events["H2T32top"] = bdt_events["H2T32"]
            bdt_events["H1Pt_H2Pt"] = bdt_events["H1Pt/H2Pt"]
            events_dict_postprocess[key] = bdt_events
        else:
            events_dict_postprocess[key] = bdt_events[columns]

        # blind!!
        if key == "data":
            # get sideband estimate instead
            print(f"Data cutflow in {mass_str} is taken from sideband estimate!")
            cutflow_dict[key][f"Bin VBF {mass_str}"] = get_nevents_data(
                bdt_events, mask_vbf, args.mass, mass_window
            )

            cutflow_dict[key][f"Bin 1 {mass_str}"] = get_nevents_data(
                bdt_events, mask_bin1, args.mass, mass_window
            )
            cutflow_dict[key][f"Bin 2 {mass_str}"] = get_nevents_data(
                bdt_events, mask_bin2, args.mass, mass_window
            )
            cutflow_dict[key][f"Bin 3 {mass_str}"] = get_nevents_data(
                bdt_events, mask_bin3, args.mass, mass_window
            )

    if control_plots:
        make_control_plots(events_dict_postprocess, plot_dir, year, args.legacy)
        for key in events_dict_postprocess:
            events_dict_postprocess[key] = events_dict_postprocess[key][columns]

    for cut in cutflow_dict[key]:
        cutflow[cut] = [cutflow_dict[key][cut] for key in events_dict]
        cutflow_print[cut] = [f"{cutflow_dict[key][cut]:.2f}" for key in events_dict]

    print("\nCutflow")
    print(cutflow_print)
    return events_dict_postprocess, cutflow


def get_nevents_data(events, cut, mass, mass_window):
    mw_size = mass_window[1] - mass_window[0]

    # get yield in left sideband
    cut_mass_0 = (events[mass] < mass_window[0]) & (events[mass] > (mass_window[0] - mw_size / 2))

    # get yield in right sideband
    cut_mass_1 = (events[mass] < mass_window[1] + mw_size / 2) & (events[mass] > mass_window[1])

    return np.sum((cut_mass_0 | cut_mass_1) & cut)


def get_nevents_signal(events, cut, mass, mass_window):
    cut_mass = (events[mass] >= mass_window[0]) & (events[mass] <= mass_window[1])

    # get yield in Higgs mass window
    return np.sum(events["weight"][cut & cut_mass])


def get_nevents_nosignal(events, cut, mass, mass_window):
    cut_mass = (events[mass] >= mass_window[0]) & (events[mass] <= mass_window[1])

    # get yield NOT in Higgs mass window
    return np.sum(events["weight"][cut & ~cut_mass])


def scan_fom(
    method: str,
    events_combined: pd.DataFrame,
    get_cut: Callable,
    xbb_cuts: np.ArrayLike,
    bdt_cuts: np.ArrayLike,
    mass_window: list[float],
    plot_dir: str,
    plot_name: str,
    bg_keys: list[str],
    sig_key: str = "hh4b",
    fom: str = "2sqrt(b)/s",
    mass: str = "H2Msd",
):
    """Generic FoM scan for given region, defined in the ``get_cut`` function."""
    print(list(bdt_cuts) + [1.0])
    print(list(xbb_cuts) + [1.0])
    h_sb = hist.Hist(
        hist.axis.Variable(list(bdt_cuts) + [1.0], name="bdt_cut"),
        hist.axis.Variable(list(xbb_cuts) + [1.0], name="xbb_cut"),
    )

    print(f"Scanning {fom} with {method}")
    for xbb_cut in xbb_cuts:
        figure_of_merits = []
        cuts = []
        min_fom = 1000
        min_nevents = []

        for bdt_cut in bdt_cuts:
            if method == "abcd":
                nevents_sig, nevents_bkg, _ = abcd(
                    events_combined, get_cut, xbb_cut, bdt_cut, mass, mass_window, bg_keys, sig_key
                )
                # print("abcd ", nevents_sig, nevents_bkg)
            else:
                nevents_sig, nevents_bkg, _ = sideband(
                    events_combined, get_cut, xbb_cut, bdt_cut, mass, mass_window, sig_key
                )
                # print("sideband ",  nevents_sig, nevents_bkg)
                # print("\n")

            if fom == "s/sqrt(s+b)":
                figure_of_merit = nevents_sig / np.sqrt(nevents_sig + nevents_bkg)
            elif fom == "2sqrt(b)/s":
                figure_of_merit = 2 * np.sqrt(nevents_bkg) / nevents_sig
            else:
                raise ValueError("Invalid FOM")

            if nevents_sig > 0.5 and nevents_bkg >= 2:
                cuts.append(bdt_cut)
                figure_of_merits.append(figure_of_merit)
                h_sb.fill(bdt_cut, xbb_cut, weight=figure_of_merit)
                if figure_of_merit < min_fom:
                    min_fom = figure_of_merit
                    min_nevents = [nevents_bkg, nevents_sig]

        if len(cuts) > 0:
            cuts = np.array(cuts)
            figure_of_merits = np.array(figure_of_merits)
            smallest = np.argmin(figure_of_merits)

            print(
                f"{xbb_cut:.3f} {cuts[smallest]:.2f} FigureOfMerit: {figure_of_merits[smallest]:.2f} "
                f"BG: {min_nevents[0]:.2f} S: {min_nevents[1]:.2f} S/B: {min_nevents[1]/min_nevents[0]:.2f}"
            )

    name = f"{plot_name}_{args.method}_mass{mass_window[0]}-{mass_window[1]}"
    print(f"Plotting FOM scan: {plot_dir}/{name} \n")
    plotting.plot_fom(h_sb, plot_dir, name=name)


def get_cuts(args, region: str):
    # VBF region
    def get_cut_vbf(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2TXbb"] > xbb_cut
        cut_bdt = events["bdt_score_vbf"] > bdt_cut
        return cut_xbb & cut_bdt

    def get_cut_novbf(events, xbb_cut, bdt_cut):  # noqa: ARG001
        return np.zeros(len(events), dtype=bool)

    # bin 1 with VBF region veto
    def get_cut_bin1(events, xbb_cut, bdt_cut):
        vbf_cut = (events["bdt_score_vbf"] >= args.vbf_bdt_wp) & (
            events["H2TXbb"] >= args.vbf_txbb_wp
        )
        cut_xbb = events["H2TXbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        # print("Passing Bin 1", np.mean(cut_xbb & cut_bdt & (~vbf_cut)))
        # print("Passing Bin 1 without VBF veto", np.mean(cut_xbb & cut_bdt))
        return cut_xbb & cut_bdt & (~vbf_cut)

    # bin 1 without VBF region veto
    def get_cut_bin1_novbf(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2TXbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        return cut_xbb & cut_bdt

    xbb_cut_bin1 = args.txbb_wps[0]
    bdt_cut_bin1 = args.bdt_wps[0]

    # bin 2 with VBF region veto
    def get_cut_bin2(events, xbb_cut, bdt_cut):
        vbf_cut = (events["bdt_score_vbf"] >= args.vbf_bdt_wp) & (
            events["H2TXbb"] >= args.vbf_txbb_wp
        )
        cut_bin1 = (events["H2TXbb"] > xbb_cut_bin1) & (events["bdt_score"] > bdt_cut_bin1)
        cut_corner = (events["H2TXbb"] < xbb_cut_bin1) & (events["bdt_score"] < bdt_cut_bin1)
        cut_bin2 = (
            (events["H2TXbb"] > xbb_cut)
            & (events["bdt_score"] > bdt_cut)
            & ~(cut_bin1)
            & ~(cut_corner)
            & ~(vbf_cut)
        )

        return cut_bin2

    # bin 2 without VBF region veto
    def get_cut_bin2_novbf(events, xbb_cut, bdt_cut):
        cut_bin1 = (events["H2TXbb"] > xbb_cut_bin1) & (events["bdt_score"] > bdt_cut_bin1)
        cut_corner = (events["H2TXbb"] < xbb_cut_bin1) & (events["bdt_score"] < bdt_cut_bin1)
        cut_bin2 = (
            (events["H2TXbb"] > xbb_cut)
            & (events["bdt_score"] > bdt_cut)
            & ~(cut_bin1)
            & ~(cut_corner)
        )

        return cut_bin2

    if region == "vbf":
        # if no VBF region, set all events to "fail VBF"
        return get_cut_vbf if args.vbf else get_cut_novbf
    elif region == "bin1":
        return get_cut_bin1 if args.vbf else get_cut_bin1_novbf
    elif region == "bin2":
        return get_cut_bin2 if args.vbf else get_cut_bin2_novbf
    else:
        raise ValueError("Invalid region")


def make_control_plots(events_dict, plot_dir, year, legacy):
    legacy_label = "Legacy" if legacy else ""

    control_plot_vars = [
        ShapeVar(var="H1Msd", label=r"$m_{SD}^{1}$ (GeV)", bins=[30, 0, 300]),
        ShapeVar(var="H2Msd", label=r"$m_{SD}^{2}$ (GeV)", bins=[30, 0, 300]),
        ShapeVar(var="H1TXbb", label=r"Xbb$^{1}$ " + legacy_label, bins=[30, 0, 1]),
        ShapeVar(var="H2TXbb", label=r"Xbb$^{2}$ " + legacy_label, bins=[30, 0, 1]),
        ShapeVar(var="H1TXbbNoLeg", label=r"Xbb$^{1}$ v12", bins=[30, 0, 1]),
        ShapeVar(var="H2TXbbNoLeg", label=r"Xbb$^{2}$ v12", bins=[30, 0, 1]),
        ShapeVar(var="H1PNetMass", label=r"$m_{reg}^{1}$ (GeV) " + legacy_label, bins=[30, 0, 300]),
        ShapeVar(var="H2PNetMass", label=r"$m_{reg}^{2}$ (GeV) " + legacy_label, bins=[30, 0, 300]),
        ShapeVar(var="HHPt", label=r"HH $p_{T}$ (GeV)", bins=[30, 0, 4000]),
        ShapeVar(var="HHeta", label=r"HH $\eta$", bins=[30, -5, 5]),
        ShapeVar(var="HHmass", label=r"HH mass (GeV)", bins=[30, 0, 1500]),
        ShapeVar(var="MET", label=r"MET (GeV)", bins=[30, 0, 600]),
        ShapeVar(var="H1T32top", label=r"$\tau_{32}^{1}$", bins=[30, 0, 1]),
        ShapeVar(var="H2T32top", label=r"$\tau_{32}^{2}$", bins=[30, 0, 1]),
        ShapeVar(var="H1Pt", label=r"H $p_{T}^{1}$ (GeV)", bins=[30, 200, 1000]),
        ShapeVar(var="H2Pt", label=r"H $p_{T}^{2}$ (GeV)", bins=[30, 200, 1000]),
        ShapeVar(var="H1eta", label=r"H $\eta^{1}$", bins=[30, -4, 4]),
        ShapeVar(var="H1QCDb", label=r"QCDb$^{2}$", bins=[30, 0, 1]),
        ShapeVar(var="H1QCDbb", label=r"QCDbb$^{2}$", bins=[30, 0, 1]),
        ShapeVar(var="H1QCDothers", label=r"QCDothers$^{1}$", bins=[30, 0, 1]),
        ShapeVar(var="H1Pt_HHmass", label=r"H$^1$ $p_{T}/mass$", bins=[30, 0, 1]),
        ShapeVar(var="H2Pt_HHmass", label=r"H$^2$ $p_{T}/mass$", bins=[30, 0, 0.7]),
        ShapeVar(var="H1Pt_H2Pt", label=r"H$^1$/H$^2$ $p_{T}$ (GeV)", bins=[30, 0.5, 1]),
        ShapeVar(var="bdt_score", label=r"BDT score", bins=[30, 0, 1]),
    ]

    (plot_dir / f"control/{year}").mkdir(exist_ok=True, parents=True)

    hists = {}
    for shape_var in control_plot_vars:
        if shape_var.var not in hists:
            hists[shape_var.var] = singleVarHist(
                events_dict,
                shape_var,
                weight_key="weight",
            )

        plotting.ratioHistPlot(
            hists[shape_var.var],
            year,
            ["hh4b"] if year in ["2022EE", "2023"] else [],
            bg_keys,
            name=f"{plot_dir}/control/{year}/{shape_var.var}",
            show=False,
            log=True,
            plot_significance=False,
            significance_dir=shape_var.significance_dir,
            ratio_ylims=[0.2, 1.8],
            bg_err_mcstat=True,
            reweight_qcd=True,
            # ylim=ylims[year],
        )


def sideband(events_dict, get_cut, txbb_cut, bdt_cut, mass, mass_window, sig_key="hh4b"):
    nevents_bkg = get_nevents_data(
        events_dict["data"],
        get_cut(events_dict["data"], txbb_cut, bdt_cut),
        mass,
        mass_window,
    )
    nevents_sig = get_nevents_signal(
        events_dict[sig_key],
        get_cut(events_dict[sig_key], txbb_cut, bdt_cut),
        mass,
        mass_window,
    )
    return nevents_sig, nevents_bkg, {}


def abcd(events_dict, get_cut, txbb_cut, bdt_cut, mass, mass_window, bg_keys, sig_key="hh4b"):
    dicts = {"data": [], **{key: [] for key in bg_keys}}

    for key in [sig_key, "data"] + bg_keys:
        events = events_dict[key]
        cut = get_cut(events, txbb_cut, bdt_cut)

        if key == sig_key:
            s = get_nevents_signal(events, cut, mass, mass_window)
            continue

        # region A
        if key == "data":
            dicts[key].append(0)
        else:
            dicts[key].append(get_nevents_signal(events, cut, mass, mass_window))

        # region B
        dicts[key].append(get_nevents_nosignal(events, cut, mass, mass_window))

        cut = (events["bdt_score"] < 0.6) & (events["H2TXbb"] < 0.8)
        # region C
        dicts[key].append(get_nevents_signal(events, cut, mass, mass_window))
        # region D
        dicts[key].append(get_nevents_nosignal(events, cut, mass, mass_window))

    # other backgrounds
    bg_tots = np.sum([dicts[key] for key in bg_keys], axis=0)
    # subtract other backgrounds
    dmt = np.array(dicts["data"]) - bg_tots
    # C/D * B
    bqcd = dmt[2] * dmt[1] / dmt[3]

    # print("bqcd ",bqcd)
    # print("bg0 ",bg_tots != 0)
    # print("bg_tots ",bg_tots[0])

    background = bqcd + bg_tots[0] if len(bg_keys) else bqcd
    return s, background, dicts


def postprocess_run3(args):
    global bg_keys  # noqa: PLW0602

    # NOT Removing all MC backgrounds for FOM scan only to save time
    # if not args.templates and not args.bdt_roc and not args.control_plots:
    #     print("Not loading any backgrounds.")

    # for year in samples_run3:
    #     if not args.templates and not args.bdt_roc and not args.control_plots:
    #         for key in bg_keys:
    #             if key in samples_run3[year]:
    #                 samples_run3[year].pop(key)

    # if not args.templates and not args.bdt_roc and not args.control_plots:
    #     bg_keys = []

    window_by_mass = {
        "H2Msd": [110, 140],
        "H2PNetMass": [110, 140],
        # "H2PNetMass": [115, 135],
    }
    if not args.legacy:
        window_by_mass["H2PNetMass"] = [120, 150]

    mass_window = np.array(window_by_mass[args.mass])  # + np.array([-5, 5])

    # variable to fit
    fit_shape_var = ShapeVar(
        args.mass,
        label_by_mass[args.mass],
        [16, 60, 220],
        reg=True,
        blind_window=window_by_mass[args.mass],
    )

    plot_dir = Path(f"../../../plots/PostProcess/{args.templates_tag}")
    plot_dir.mkdir(exist_ok=True, parents=True)

    # load samples
    bdt_training_keys = get_bdt_training_keys(args.bdt_model)
    events_dict_postprocess = {}
    cutflows = {}
    for year in args.years:
        print(f"\n{year}")
        events_dict_postprocess[year], cutflows[year] = load_process_run3_samples(
            args,
            year,
            bdt_training_keys,
            args.control_plots,
            plot_dir,
        )

    print("Loaded all years")

    processes = ["data"] + args.sig_keys + bg_keys
    bg_keys_combined = bg_keys.copy()

    if len(args.years) > 1:
        events_combined, scaled_by = combine_run3_samples(
            events_dict_postprocess,
            processes,
            bg_keys=bg_keys_combined,
            scale_processes={
                # "hh4b": ["2022EE", "2023", "2023BPix"], # FIXED
                "vbfhh4b-k2v0": ["2022", "2022EE"],
            },
            years_run3=args.years,
        )
        print("Combined years")
    else:
        events_combined = events_dict_postprocess[args.years[0]]
        scaled_by = {}

    # combined cutflow
    cutflow_combined = None
    if len(args.years) > 0:
        cutflow_combined = pd.DataFrame(index=list(events_combined.keys()))

        # get ABCD (warning!: not considering VBF region veto)
        s_bin1, b_bin1, _ = abcd(
            events_combined,
            get_cuts(args, "bin1"),
            args.txbb_wps[0],
            args.bdt_wps[0],
            args.mass,
            mass_window,
            bg_keys,
            "hh4b",
        )

        # note: need to do this since not all the years have all the samples..
        year_0 = "2022EE" if "2022EE" in args.years else args.years[0]
        samples = list(events_combined.keys())
        for cut in cutflows[year_0]:
            yield_s = 0
            yield_b = 0
            for s in samples:
                cutflow_sample = np.sum(
                    [
                        cutflows[year][cut].loc[s] if s in cutflows[year][cut].index else 0.0
                        for year in args.years
                    ]
                )
                if s in scaled_by:
                    print(f"Scaling combined cutflow for {s} by {scaled_by[s]}")
                    cutflow_sample *= scaled_by[s]
                if s == "hh4b":
                    yield_s = cutflow_sample
                if s == "data":
                    yield_b = cutflow_sample
                cutflow_combined.loc[s, cut] = f"{cutflow_sample:.2f}"

            if "Bin 1 [" in cut and yield_b > 0:
                cutflow_combined.loc["S/B sideband", cut] = f"{yield_s/yield_b:.3f}"
                cutflow_combined.loc["S/B ABCD", cut] = f"{s_bin1/b_bin1:.3f}"

        print(f"\n Combined cutflow TXbb:{args.txbb_wps} BDT: {args.bdt_wps}")
        print(cutflow_combined)

    if args.fom_scan:

        if args.fom_scan_vbf and args.vbf:
            print("Scanning VBF WPs")
            scan_fom(
                args.method,
                events_combined,
                get_cuts(args, "vbf"),
                np.arange(0.9, 0.999, 0.01),
                np.arange(0.9, 0.999, 0.01),
                mass_window,
                plot_dir,
                "fom_vbf",
                bg_keys=bg_keys,
                sig_key="vbfhh4b-k2v0",
                mass=args.mass,
            )

        if args.fom_scan_bin1:
            if args.vbf:
                print(
                    f"Scanning Bin 1 with VBF TXbb WP: {args.vbf_txbb_wp} BDT WP: {args.vbf_bdt_wp}"
                )
            else:
                print("Scanning Bin 1, no VBF")

            scan_fom(
                args.method,
                events_combined,
                get_cuts(args, "bin1"),
                np.arange(0.95, 0.999, 0.005),
                np.arange(0.9, 0.999, 0.01),
                mass_window,
                plot_dir,
                "fom_bin1",
                bg_keys=bg_keys,
                mass=args.mass,
            )

        if args.fom_scan_bin2:
            if args.vbf:
                print(
                    f"Scanning Bin 2 with VBF TXbb WP: {args.vbf_txbb_wp} BDT WP: {args.vbf_bdt_wp}, bin 1 WP: {args.txbb_wps[0]} BDT WP: {args.bdt_wps[0]}"
                )
            else:
                print(f"Scanning Bin 2 with bin 1 WP: {args.txbb_wps[0]} BDT WP: {args.bdt_wps[0]}")
            scan_fom(
                args.method,
                events_combined,
                get_cuts(args, "bin2"),
                np.arange(0.8, args.txbb_wps[0], 0.02),
                np.arange(0.5, args.bdt_wps[0], 0.02),
                mass_window,
                plot_dir,
                "fom_bin2",
                bg_keys=bg_keys,
                mass=args.mass,
            )

    if args.bdt_roc:
        print("Making BDT ROC curve")
        bdt_roc(events_combined, plot_dir, args.legacy)

    templ_dir = Path("templates") / args.templates_tag
    year = "2022-2023"
    (templ_dir / "cutflows" / year).mkdir(parents=True, exist_ok=True)
    (templ_dir / year).mkdir(parents=True, exist_ok=True)

    # save args for posterity
    with (templ_dir / "args.txt").open("w") as f:
        pretty_printer = pprint.PrettyPrinter(stream=f, indent=4)
        pretty_printer.pprint(vars(args))

    for cyear in args.years:
        cutflows[cyear] = cutflows[cyear].round(2)
        cutflows[cyear].to_csv(templ_dir / "cutflows" / f"preselection_cutflow_{cyear}.csv")
    if cutflow_combined is not None:
        cutflow_combined = cutflow_combined.round(2)
        cutflow_combined.to_csv(templ_dir / "cutflows" / "preselection_cutflow_combined.csv")

    if not args.templates:
        return

    if not args.vbf:
        selection_regions.pop("pass_vbf")

    # individual templates per year
    templates = postprocessing.get_templates(
        events_combined,
        year=year,
        sig_keys=args.sig_keys,
        selection_regions=selection_regions,
        shape_vars=[fit_shape_var],
        systematics={},
        template_dir=templ_dir,
        bg_keys=bg_keys_combined,
        plot_dir=f"{templ_dir}/{year}",
        weight_key="weight",
        show=False,
        energy=13.6,
    )

    # save templates per year
    postprocessing.save_templates(templates, templ_dir / f"{year}_templates.pkl", fit_shape_var)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--templates-tag",
        type=str,
        required=True,
        help="output pickle directory of hist.Hist templates inside the ./templates dir",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/eos/uscms/store/user/cmantill/bbbb/skimmer/",
        help="tag for input ntuples",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="tag for input ntuples",
    )
    parser.add_argument(
        "--years",
        type=str,
        nargs="+",
        default=hh_vars.years,
        choices=hh_vars.years,
        help="years to postprocess",
    )
    parser.add_argument(
        "--training-years",
        nargs="+",
        choices=hh_vars.years,
        help="years used in training",
    )
    parser.add_argument(
        "--mass",
        type=str,
        default="H2Msd",
        choices=["H2Msd", "H2PNetMass"],
        help="mass variable to make template",
    )
    parser.add_argument(
        "--bdt-model",
        type=str,
        default="v1_msd30_nomulticlass",
        help="BDT model to load",
    )
    parser.add_argument(
        "--bdt-config",
        type=str,
        default="v1_msd30",
        help="BDT model to load",
    )

    parser.add_argument(
        "--txbb-wps",
        type=float,
        nargs=2,
        default=[0.985, 0.94],
        help="TXbb Bin 1, Bin 2 WPs",
    )

    parser.add_argument(
        "--bdt-wps",
        type=float,
        nargs=3,
        default=[0.95, 0.75, 0.03],
        help="BDT Bin 1, Bin 2, Fail WPs",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sideband",
        choices=["abcd", "sideband"],
        help="method for scanning",
    )

    parser.add_argument("--vbf-txbb-wp", type=float, default=0.97, help="TXbb VBF WP")
    parser.add_argument("--vbf-bdt-wp", type=float, default=0.97, help="BDT VBF WP")

    parser.add_argument(
        "--sig-keys",
        type=str,
        nargs="+",
        default=["hh4b", "vbfhh4b-k2v0"],
        help="sig keys for which to make templates",
    )

    parser.add_argument(
        "--pt-second", type=float, default=300, help="pt threshold for subleading jet"
    )

    run_utils.add_bool_arg(parser, "bdt-roc", default=False, help="make BDT ROC curve")
    run_utils.add_bool_arg(parser, "control-plots", default=False, help="make control plots")
    run_utils.add_bool_arg(parser, "fom-scan", default=False, help="run figure of merit scans")
    run_utils.add_bool_arg(parser, "fom-scan-bin1", default=True, help="FOM scan for bin 1")
    run_utils.add_bool_arg(parser, "fom-scan-bin2", default=True, help="FOM scan for bin 2")
    run_utils.add_bool_arg(parser, "fom-scan-vbf", default=False, help="FOM scan for VBF bin")
    run_utils.add_bool_arg(parser, "templates", default=True, help="make templates")
    run_utils.add_bool_arg(parser, "legacy", default=True, help="using legacy pnet txbb and mass")
    run_utils.add_bool_arg(parser, "vbf", default=False, help="Add VBF region")
    run_utils.add_bool_arg(
        parser, "vbf-priority", default=False, help="Prioritize the VBF region over ggF Cat 1"
    )

    args = parser.parse_args()

    print(args)
    postprocess_run3(args)
