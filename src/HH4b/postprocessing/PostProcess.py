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
from HH4b.postprocessing import Region, corrections
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


def load_run3_samples(args, year):
    # modify as needed
    input_dir = f"/eos/uscms/store/user/cmantill/bbbb/skimmer/{args.tag}"

    samples_run3 = {
        "2022": {
            "data": [
                "JetMET_Run2022C",
                "JetMET_Run2022C_single",
                "JetMET_Run2022D",
            ],
            "ttbar": [
                "TTto4Q",
            ],
            "ttlep": [
                "TTtoLNu2Q",
                "TTto2L2Nu",
            ],
            "diboson": [
                "WW",
                "WZ",
                "ZZ",
            ],
            "vjets": [
                "Wto2Q-3Jets_HT-200to400",
                "Wto2Q-3Jets_HT-400to600",
                "Wto2Q-3Jets_HT-600to800",
                "Wto2Q-3Jets_HT-800",
                "Zto2Q-4Jets_HT-200to400",
                "Zto2Q-4Jets_HT-400to600",
                "Zto2Q-4Jets_HT-600to800",
                "Zto2Q-4Jets_HT-800",
            ],
            "novhhtobb": [
                "GluGluHto2B_PT-200_M-125",
                "VBFHto2B_M-125_dipoleRecoilOn",
            ],
            "tthtobb": [
                "ttHto2B_M-125",
            ],
            "vhtobb": [
                "WminusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_Wto2Q_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2Q_M-125",
            ],
        },
        "2022EE": {
            "data": [
                "JetMET_Run2022E",
                "JetMET_Run2022F",
                "JetMET_Run2022G",
            ],
            "ttbar": [
                "TTto4Q",
            ],
            "ttlep": [
                "TTtoLNu2Q",
                "TTto2L2Nu",
            ],
            "hh4b": [
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV",
            ],
            "vbfhh4b": [
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8",
            ],
            "diboson": [
                "WW",
                "WZ",
                "ZZ",
            ],
            "vjets": [
                "Wto2Q-3Jets_HT-200to400",
                "Wto2Q-3Jets_HT-400to600",
                "Wto2Q-3Jets_HT-600to800",
                "Wto2Q-3Jets_HT-800",
                "Zto2Q-4Jets_HT-200to400",
                "Zto2Q-4Jets_HT-400to600",
                "Zto2Q-4Jets_HT-600to800",
                "Zto2Q-4Jets_HT-800",
            ],
            "novhhtobb": [
                "GluGluHto2B_PT-200_M-125",
                "VBFHto2B_M-125_dipoleRecoilOn",
            ],
            "tthtobb": [
                "ttHto2B_M-125",
            ],
            "vhtobb": [
                "WminusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_Wto2Q_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2Q_M-125",
            ],
            "qcd": [
                "QCD_HT-200to400",
                "QCD_HT-400to600",
                "QCD_HT-600to800",
                "QCD_HT-800to1000",
                "QCD_HT-1000to1200",
                "QCD_HT-1200to1500",
                "QCD_HT-1500to2000",
                "QCD_HT-2000",
            ],
        },
        "2023": {
            "data": [
                "JetMET_Run2023C",
            ],
            "ttbar": [
                "TTto4Q",
            ],
            "ttlep": [
                "TTtoLNu2Q",
                "TTto2L2Nu",
            ],
            "diboson": [
                "WW",
                "WZ",
                "ZZ",
            ],
            "vjets": [
                "Wto2Q-3Jets_HT-200to400",
                "Wto2Q-3Jets_HT-400to600",
                "Wto2Q-3Jets_HT-600to800",
                "Wto2Q-3Jets_HT-800",
                "Zto2Q-4Jets_HT-200to400",
                "Zto2Q-4Jets_HT-400to600",
                "Zto2Q-4Jets_HT-600to800",
                "Zto2Q-4Jets_HT-800",
            ],
            "novhhtobb": [
                "GluGluHto2B_PT-200_M-125",
                "VBFHto2B_M-125_dipoleRecoilOn",
            ],
            "tthtobb": [
                "ttHto2B_M-125",
            ],
            "vhtobb": [
                "WminusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_Wto2Q_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2Q_M-125",
            ],
        },
        "2023BPix": {
            "data": [
                "JetMET_Run2023D",
            ],
            "ttbar": [
                "TTto4Q",
            ],
            "ttlep": [
                "TTtoLNu2Q",
                "TTto2L2Nu",
            ],
            "diboson": [
                "WW",
                "WZ",
                "ZZ",
            ],
            "vjets": [
                "Wto2Q-3Jets_HT-200to400",
                "Wto2Q-3Jets_HT-400to600",
                "Wto2Q-3Jets_HT-600to800",
                "Wto2Q-3Jets_HT-800",
                "Zto2Q-4Jets_HT-200to400",
                "Zto2Q-4Jets_HT-400to600",
                "Zto2Q-4Jets_HT-600to800",
                "Zto2Q-4Jets_HT-800",
            ],
            "novhhtobb": [
                "GluGluHto2B_PT-200_M-125",
                "VBFHto2B_M-125_dipoleRecoilOn",
            ],
            "tthtobb": [
                "ttHto2B_M-125",
            ],
            "vhtobb": [
                "WminusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_Wto2Q_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
            ],
        },
    }

    legacy_label = "Legacy" if args.legacy else ""

    load_columns = [
        ("weight", 1),
        ("MET_pt", 1),
        ("event", 1),
        ("nFatJets", 1),
        ("bbFatJetPt", 2),
        ("bbFatJetEta", 2),
        ("bbFatJetPhi", 2),
        ("bbFatJetMsd", 2),
        (f"bbFatJetPNetMass{legacy_label}", 2),
        (f"bbFatJetPNetXbb{legacy_label}", 2),
        ("bbFatJetTau3OverTau2", 2),
        (f"bbFatJetPNetQCD0HF{legacy_label}", 2),
        (f"bbFatJetPNetQCD1HF{legacy_label}", 2),
        (f"bbFatJetPNetQCD2HF{legacy_label}", 2),
    ]

    filters = [
        [
            ("('bbFatJetPt', '0')", ">=", 300),
            ("('bbFatJetPt', '1')", ">=", 300),
        ],
    ]

    # define BDT model
    bdt_model = xgb.XGBClassifier()
    bdt_model.load_model(fname=f"../boosted/bdt_trainings_run3/{args.bdt_model}/trained_bdt.model")
    # get function
    config = args.bdt_model if args.bdt_model != "v1_msd30_nomulticlass" else "v1_msd30"
    make_bdt_dataframe = importlib.import_module(
        f".{config}", package="HH4b.boosted.bdt_trainings_run3"
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
        bdt_score = bdt_model.predict_proba(bdt_events)[:, 1]

        bdt_events["bdt_score"] = bdt_score
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
        ## Add TTBar Weigh)t here
        if key == "ttbar":
            bdt_events["weight"] *= corrections.ttbar_pTjjSF(year, events_dict, "bbFatJetPNetMass")

        # add selection to testing events
        bdt_events["event"] = events_dict[key]["event"].to_numpy()[:, 0]
        if year == "2022EE" and key in ["qcd", "ttbar", "hh4b"]:
            evt_list = np.load(
                f"../boosted/bdt_trainings_run3/{args.bdt_model}/inferences/2022EE/evt_{key}.npy"
            )
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

    xbb_cuts = np.arange(0.8, 0.98, 0.02)
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

    label_by_mass = {
        "H2Msd": r"$m^{2}_\mathrm{SD}$ (GeV)",
        "H2PNetMass": r"$m^{2}_\mathrm{reg}$ (GeV)",
    }
    window_by_mass = {"H2Msd": [110, 140]}
    if args.legacy:
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
    events_dict_postprocess = {}
    cutflows = {}
    for year in args.years:
        events_dict_postprocess[year], cutflows[year] = load_run3_samples(args, year)

    bkg_keys = ["qcd", "ttbar", "ttlep", "vhtobb", "vjets", "diboson", "novhhtobb"]
    processes = ["data", "hh4b"] + bkg_keys

    # create combined datasets
    # temporarily used 2022EEMC and scale to full luminosity
    lumi_weight_2022EEtoall = (7971.4 + 26337.0 + 17650.0 + 9451.0) / 26337.0
    events_combined = {}
    for key in processes:
        if key not in ["hh4b", "qcd"]:
            combined = pd.concat(
                [
                    events_dict_postprocess["2022"][key],
                    events_dict_postprocess["2022EE"][key],
                    events_dict_postprocess["2023"][key],
                    events_dict_postprocess["2023BPix"][key],
                ]
            )
        else:
            combined = events_dict_postprocess["2022EE"][key].copy()
            combined["weight"] = combined["weight"] * lumi_weight_2022EEtoall
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
        shift_mass_window = np.array([-15, -5])
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
        sig_keys=["hh4b"],
        selection_regions=selection_regions,
        shape_vars=[fit_shape_var],
        systematics={},
        template_dir=templ_dir,
        bg_keys=bkg_keys,
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

    run_utils.add_bool_arg(parser, "fom-scan", default=True, help="run figure of merit scan")
    run_utils.add_bool_arg(parser, "templates", default=True, help="make templates")
    run_utils.add_bool_arg(parser, "legacy", default=False, help="using legacy pnet txbb and mass")
    args = parser.parse_args()

    print(args)
    postprocess_run3(args)
