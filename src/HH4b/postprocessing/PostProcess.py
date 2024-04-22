from __future__ import annotations

import argparse
import importlib
from collections import OrderedDict
from pathlib import Path

import hist

# temp
import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b import hh_vars, plotting, postprocessing, run_utils, utils
from HH4b.hh_vars import LUMI, bg_keys, samples_run3, years
from HH4b.postprocessing import (
    Region,
    corrections,
    filters_legacy,
    filters_v12,
    load_columns_legacy,
    load_columns_v12,
)
from HH4b.utils import ShapeVar, load_samples

# from .corrections import ttbar_pTjjSF

# TODO: can switch to this in the future to get cutflows for each cut.
# def get_selection_regions(txbb_wps: list[float], bdt_wps: list[float]):
#     return {
#         "pass_bin1": Region(
#             cuts={
#                 "H2Xbb": [txbb_wps[0], CUT_MAX_VAL],
#                 "bdt_score": [bdt_wps[0], CUT_MAX_VAL],
#             },
#             label="Bin1",
#         ),
#         "pass_bin2": Region(
#             cuts={
#                 "H2Xbb": [txbb_wps[1], CUT_MAX_VAL],
#                 "bdt_score": [bdt_wps[1], CUT_MAX_VAL],
#                 # veto events in Bin 1
#                 "H2Xbb+bdt_score": [[-CUT_MAX_VAL, txbb_wps[0]], [-CUT_MAX_VAL, bdt_wps[0]]],
#                 # veto events in "lower left corner"
#                 "H2Xbb+bdt_score": [[txbb_wps[0], CUT_MAX_VAL], [bdt_wps[0], CUT_MAX_VAL]],
#             },
#             label="Bin2",
#         ),
#         "pass_bin3": Region(
#             cuts={
#                 "H2Xbb": [txbb_wps[1], CUT_MAX_VAL],
#                 "bdt_score": [bdt_wps[2], bdt_wps[0]],
#                 # veto events in Bin 2
#                 "H2Xbb+bdt_score": [[-CUT_MAX_VAL, txbb_wps[0]], [-CUT_MAX_VAL, bdt_wps[1]]],
#             },
#             label="Bin3",
#         ),
#         "fail": Region(
#             cuts={
#                 "H2Xbb": [-CUT_MAX_VAL, txbb_wps[1]],
#             },
#             label="Fail",
#         ),
#     }


selection_regions = {
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


def _get_bdt_training_keys(bdt_model: str):
    inferences_dir = Path(f"../boosted/bdt_trainings_run3/{bdt_model}/inferences/2022EE")

    training_keys = []
    for child in inferences_dir.iterdir():
        if child.endswith(".npy"):
            training_keys.append(child.stem.split("evt_")[-1])

    print("Found BDT Training keys", training_keys)


def _add_bdt_scores(events: pd.DataFrame, preds: np.ArrayLike):
    if preds.shape[1] == 2:  # binary BDT only
        events["bdt_score"] = preds[:, 1]
    elif preds.shape[1] == 3:  # multi-class BDT with ggF HH, QCD, ttbar classes
        events["bdt_score"] = preds[:, 0]  # ggF HH
    elif preds.shape[1] == 4:  # multi-class BDT with ggF HH, VBF HH, QCD, ttbar classes
        bg_tot = np.sum(preds[:, 2:], axis=1)
        events["bdt_score"] = preds[:, 0] / (preds[:, 0] + bg_tot)
        events["bdt_score_vbf"] = preds[:, 1] / (preds[:, 1] + bg_tot)


def load_run3_samples(args, year, bdt_training_keys):
    # modify as needed
    input_dir = f"{args.data_dir}/{args.tag}"
    samples = samples_run3[year].copy()
    samples.pop("qcd")  # QCD is all data-driven so don't need it

    legacy_label = "Legacy" if args.legacy else ""
    filters = filters_legacy if args.legacy else filters_v12
    load_columns = load_columns_legacy if args.legacy else load_columns_v12

    # define BDT model
    bdt_model = xgb.XGBClassifier()
    bdt_model.load_model(fname=f"../boosted/bdt_trainings_run3/{args.bdt_model}/trained_bdt.model")
    # get function
    make_bdt_dataframe = importlib.import_module(
        f".{args.bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
    )

    if year == "2023":
        load_columns_year = load_columns + [
            ("AK8PFJet230_SoftDropMass40_PNetBB0p06", 1),
            ("AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35", 1),
        ]
    elif year == "2023BPix":
        load_columns_year = load_columns + [("AK8PFJet230_SoftDropMass40_PNetBB0p06", 1)]
    else:
        load_columns_year = load_columns + [
            ("AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35", 1)
        ]

    # pre-selection
    events_dict = load_samples(
        input_dir,
        samples_run3[year],
        year,
        filters=filters,
        columns=utils.format_columns(load_columns_year),
        variations=False,
    )

    HLTs = {
        "2022": ["AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35"],
        "2022EE": ["AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35"],
        "2023": [
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        ],
        "2023BPix": ["AK8PFJet230_SoftDropMass40_PNetBB0p06"],
    }

    cutflow = pd.DataFrame(index=list(events_dict.keys()))
    cutflow_dict = {
        key: OrderedDict(
            [("Skimmer Preselection", np.sum(events_dict[key]["finalWeight"].to_numpy()))]
        )
        for key in events_dict
    }
    # inference and assign score
    events_dict_postprocess = {}
    for key in events_dict:
        bdt_events = make_bdt_dataframe.bdt_dataframe(events_dict[key])
        preds = bdt_model.predict_proba(bdt_events)
        _add_bdt_scores(bdt_events, preds)

        bdt_events["H1Msd"] = events_dict[key]["bbFatJetMsd"].to_numpy()[:, 0]
        bdt_events["H2Msd"] = events_dict[key]["bbFatJetMsd"].to_numpy()[:, 1]
        bdt_events["H2Xbb"] = events_dict[key][f"bbFatJetPNetXbb{legacy_label}"].to_numpy()[:, 1]
        bdt_events["H2PNetMass"] = events_dict[key][f"bbFatJetPNetMass{legacy_label}"].to_numpy()[
            :, 1
        ]

        # add HLTs
        bdt_events["hlt"] = np.any(
            np.array(
                [
                    events_dict[key][trigger].to_numpy()[:, 0]
                    for trigger in HLTs[year]
                    if trigger in events_dict[key]
                ]
            ),
            axis=0,
        )

        # add more columns (e.g. (uncertainties etc)
        bdt_events["weight"] = events_dict[key]["finalWeight"].to_numpy()
        ## Add TTBar Weight here TODO: does this need to be re-measured for legacy PNet Mass?
        if key == "ttbar" and not args.legacy:
            bdt_events["weight"] *= corrections.ttbar_pTjjSF(year, events_dict, "bbFatJetPNetMass")

        # add selection to testing events
        bdt_events["event"] = events_dict[key]["event"].to_numpy()[:, 0]
        if year == "2022EE":
            inferences_dir = Path(f"../boosted/bdt_trainings_run3/{bdt_model}/inferences/2022EE")

            if key in bdt_training_keys:
                evt_list = np.load(inferences_dir / f"evt_{key}.npy")
                bdt_events = bdt_events[bdt_events["event"].isin(evt_list)]
                bdt_events["weight"] *= 1 / 0.4  # divide by BDT test / train ratio

        # extra selection
        bdt_events = bdt_events[bdt_events["hlt"] == 1]
        cutflow_dict[key]["HLT"] = np.sum(bdt_events["weight"].to_numpy())

        if not args.legacy:
            bdt_events = bdt_events[bdt_events["H1Msd"] > 30]
            cutflow_dict[key]["H1Msd > 30"] = np.sum(bdt_events["weight"].to_numpy())
            bdt_events = bdt_events[bdt_events["H2Msd"] > 30]
            cutflow_dict[key]["H2Msd > 30"] = np.sum(bdt_events["weight"].to_numpy())

        # define category
        bdt_events["Category"] = 5  # all events
        mask_bin1 = (bdt_events["H2Xbb"] > args.txbb_wps[0]) & (
            bdt_events["bdt_score"] > args.bdt_wps[0]
        )
        bdt_events.loc[mask_bin1, "Category"] = 1
        mask_corner = (bdt_events["H2Xbb"] < args.txbb_wps[0]) & (
            bdt_events["bdt_score"] < args.bdt_wps[0]
        )
        mask_bin2 = (
            (bdt_events["H2Xbb"] > args.txbb_wps[1])
            & (bdt_events["bdt_score"] > args.bdt_wps[1])
            & ~(mask_bin1)
            & ~(mask_corner)
        )
        bdt_events.loc[mask_bin2, "Category"] = 2
        mask_bin3 = ~(mask_bin1) & ~(mask_bin2) & (bdt_events["bdt_score"] > args.bdt_wps[2])
        bdt_events.loc[mask_bin3, "Category"] = 3
        bdt_events.loc[
            (bdt_events["H2Xbb"] < args.txbb_wps[1]) & (bdt_events["bdt_score"] > args.bdt_wps[2]),
            "Category",
        ] = 4

        columns = ["Category", "H2Msd", "bdt_score", "H2Xbb", "H2PNetMass", "weight"]
        events_dict_postprocess[key] = bdt_events[columns]

    for cut in cutflow_dict[key]:
        yields = [cutflow_dict[key][cut] for key in events_dict]
        cutflow[cut] = yields

    print("\nCutflow")
    print(cutflow)
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


def scan_fom(
    events_combined: pd.DataFrame,
    mass_window: list[float],
    plot_dir: str,
    fom="2sqrt(b)/s",
    mass="H2Msd",
):
    """
    Scan figure of merit
    """

    def get_cut(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2Xbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        return cut_xbb & cut_bdt

    xbb_cuts = np.arange(0.8, 1, 0.02)
    bdt_cuts = np.arange(0.8, 1, 0.01)
    h_sb = hist.Hist(
        hist.axis.Variable(bdt_cuts, name="bdt_cut"),
        hist.axis.Variable(xbb_cuts, name="xbb_cut"),
    )

    for xbb_cut in xbb_cuts:
        figure_of_merits = []
        cuts = []
        for bdt_cut in bdt_cuts:
            nevents_data = get_nevents_data(
                events_combined["data"],
                get_cut(events_combined["data"], xbb_cut, bdt_cut),
                mass,
                mass_window,
            )
            nevents_signal = get_nevents_signal(
                events_combined["hh4b"],
                get_cut(events_combined["hh4b"], xbb_cut, bdt_cut),
                mass,
                mass_window,
            )

            if fom == "s/sqrt(s+b)":
                figure_of_merit = nevents_signal / np.sqrt(nevents_signal + nevents_data)
            elif fom == "2sqrt(b)/s":
                figure_of_merit = 2 * np.sqrt(nevents_data) / nevents_signal
            else:
                raise ValueError("Invalid FOM")

            if nevents_signal > 0.5:
                cuts.append(bdt_cut)
                figure_of_merits.append(figure_of_merit)
                h_sb.fill(bdt_cut, xbb_cut, weight=figure_of_merit)

        if len(cuts) > 0:
            cuts = np.array(cuts)
            figure_of_merits = np.array(figure_of_merits)
            smallest = np.argmin(figure_of_merits)

            print(xbb_cut, cuts[smallest], figure_of_merits[smallest])

    print(f"Plotting FOM scan for Bin 1 in {plot_dir}")
    plotting.plot_fom(h_sb, plot_dir)


def scan_fom_bin2(
    events_combined: pd.DataFrame,
    mass_window: list[float],
    plot_dir: str,
    xbb_cut_bin1=0.92,
    bdt_cut_bin1=0.94,
    fom="2sqrt(b)/s",
    mass="H2Msd",
):
    """
    Scan figure of merit for bin2
    """

    def get_cut(events, xbb_cut, bdt_cut):
        cut_bin1 = (events["H2Xbb"] > xbb_cut_bin1) & (events["bdt_score"] > bdt_cut_bin1)
        cut_corner = (events["H2Xbb"] < xbb_cut_bin1) & (events["bdt_score"] < bdt_cut_bin1)
        cut_bin2 = (
            (events["H2Xbb"] > xbb_cut)
            & (events["bdt_score"] > bdt_cut)
            & ~(cut_bin1)
            & ~(cut_corner)
        )

        return cut_bin2

    xbb_cuts = np.arange(0.7, xbb_cut_bin1, 0.02)
    bdt_cuts = np.arange(0.65, bdt_cut_bin1, 0.01)
    h_sb = hist.Hist(
        hist.axis.Variable(bdt_cuts, name="bdt_cut"),
        hist.axis.Variable(xbb_cuts, name="xbb_cut"),
    )
    for xbb_cut in xbb_cuts:
        figure_of_merits = []
        cuts = []
        for bdt_cut in bdt_cuts:
            nevents_data = get_nevents_data(
                events_combined["data"],
                get_cut(events_combined["data"], xbb_cut, bdt_cut),
                mass,
                mass_window,
            )
            nevents_signal = get_nevents_signal(
                events_combined["hh4b"],
                get_cut(events_combined["hh4b"], xbb_cut, bdt_cut),
                mass,
                mass_window,
            )

            if fom == "s/sqrt(s+b)":
                figure_of_merit = nevents_signal / np.sqrt(nevents_signal + nevents_data)
            elif fom == "2sqrt(b)/s":
                figure_of_merit = 2 * np.sqrt(nevents_data) / nevents_signal
            else:
                raise ValueError("Invalid FOM")

            if nevents_signal > 0.5:
                cuts.append(bdt_cut)
                figure_of_merits.append(figure_of_merit)
                h_sb.fill(bdt_cut, xbb_cut, weight=figure_of_merit)

        if len(cuts) > 0:
            cuts = np.array(cuts)
            figure_of_merits = np.array(figure_of_merits)
            smallest = np.argmin(figure_of_merits)

            print(xbb_cut, cuts[smallest], figure_of_merits[smallest])

    print(f"Plotting FOM scan for Bin 2 in {plot_dir}")
    plotting.plot_fom(h_sb, plot_dir, name="figofmerit_bin2")


def postprocess_run3(args):
    label_by_mass = {
        "H2Msd": r"$m^{2}_\mathrm{SD}$ (GeV)",
        "H2PNetMass": r"$m^{2}_\mathrm{reg}$ (GeV)",
    }
    window_by_mass = {"H2Msd": [110, 140]}
    if not args.legacy:
        window_by_mass["H2PNetMass"] = [120, 150]
    else:
        window_by_mass["H2PNetMass"] = [110, 140]

    # variable to fit
    fit_shape_var = ShapeVar(
        args.mass,
        label_by_mass[args.mass],
        [16, 60, 220],
        reg=True,
        blind_window=window_by_mass[args.mass],
    )

    # load samples
    bdt_training_keys = _get_bdt_training_keys(args.bdt_model)
    events_dict_postprocess = {}
    cutflows = {}
    for year in args.years:
        events_dict_postprocess[year], cutflows[year] = load_run3_samples(
            args, year, bdt_training_keys
        )

    bg_keys.remove("qcd")
    processes = ["data"] + args.sig_keys + bg_keys

    # these processes are temporarily only in certain eras, so their weights have to be scaled up to full luminosity
    scale_processes = {
        "hh4b": ["2022EE", "2023", "2023BPix"],
        "vbfhh4b-k2v0": ["2022", "2022EE"],
    }

    # create combined datasets
    # temporarily used 2022EEMC and scale to full luminosity
    lumi_total = np.sum([LUMI[year] for year in years])

    events_combined = {}
    for key in processes:
        if key not in scale_processes:
            combined = pd.concat([events_dict_postprocess[year][key] for year in years])
        else:
            combined = pd.concat(
                [events_dict_postprocess[year][key].copy() for year in scale_processes[key]]
            )
            lumi_scale = lumi_total / np.sum([LUMI[year] for year in scale_processes[key]])
            combined["weight"] = combined["weight"] * lumi_scale

        events_combined[key] = combined

    # combine ttbar
    events_combined["ttbar"] = pd.concat([events_combined["ttbar"], events_combined["ttlep"]])
    events_combined["others"] = pd.concat(
        [events_combined["diboson"], events_combined["vjets"], events_combined["novhhtobb"]]
    )

    if args.fom_scan:
        plot_dir = Path(f"../../../plots/PostProcess/{args.templates_tag}")
        plot_dir.mkdir(exist_ok=True, parents=True)
        # todo: update to [-5, +5] for next round!
        shift_mass_window = np.array([-5, 5])
        scan_fom(
            events_combined,
            np.array(window_by_mass[args.mass]) + shift_mass_window,
            plot_dir,
            mass=args.mass,
        )
        scan_fom_bin2(
            events_combined,
            np.array(window_by_mass[args.mass]) + shift_mass_window,
            plot_dir,
            xbb_cut_bin1=args.txbb_wps[0],
            bdt_cut_bin1=args.bdt_wps[0],
            mass=args.mass,
        )

    if not args.templates:
        return

    templ_dir = Path("templates") / args.templates_tag
    year = "2022-2023"
    (templ_dir / "cutflows" / year).mkdir(parents=True, exist_ok=True)
    (templ_dir / year).mkdir(parents=True, exist_ok=True)

    # TODO: fix combine cutflow
    # if len(args.years) > 1:
    #     cutflow_combined = pd.DataFrame(index=list(events_combined.keys()))
    #     for cut in cutflows[args.years[0]]:
    #         cutflow_combined[cut] = np.sum(
    #             [cutflows[year][cut].to_numpy() for year in args.years], axis=0
    #         )
    #     print(cutflow_combined)
    #     cutflow_combined.to_csv(templ_dir / "cutflows" / "preselection_cutflow.csv")

    for cyear in args.years:
        cutflows[cyear].to_csv(templ_dir / "cutflows" / f"preselection_cutflow_{cyear}.csv")

    print(events_combined["data"].columns)

    # individual templates per year
    templates = postprocessing.get_templates(
        events_combined,
        year=year,
        sig_keys=args.sig_keys,
        selection_regions=selection_regions,
        shape_vars=[fit_shape_var],
        systematics={},
        template_dir=templ_dir,
        bg_keys=bg_keys,
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
        default=[0.92, 0.8],
        help="TXbb Bin 1, Bin 2 WPs",
    )

    parser.add_argument(
        "--bdt-wps",
        type=float,
        nargs=3,
        default=[0.94, 0.68, 0.03],
        help="BDT Bin 1, Bin 2, Fail WPs",
    )

    parser.add_argument(
        "--sig-keys",
        type=str,
        nargs="+",
        default=["hh4b"],
        help="sig keys for which to make templates",
    )

    run_utils.add_bool_arg(parser, "fom-scan", default=True, help="run figure of merit scans")
    run_utils.add_bool_arg(parser, "fom-scan-bin1", default=True, help="FOM scan for bin 1")
    run_utils.add_bool_arg(parser, "fom-scan-bin2", default=True, help="FOM scan for bin 2")
    run_utils.add_bool_arg(parser, "fom-scan-vbf", default=True, help="FOM scan for VBF bin")
    run_utils.add_bool_arg(parser, "templates", default=True, help="make templates")
    run_utils.add_bool_arg(parser, "legacy", default=False, help="using legacy pnet txbb and mass")
    args = parser.parse_args()

    print(args)
    postprocess_run3(args)
