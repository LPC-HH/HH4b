from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep

# temp
import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b import hh_vars, postprocessing, run_utils, utils
from HH4b.postprocessing import Region
from HH4b.utils import ShapeVar, load_samples

sys.path.append("../boosted/bdt_trainings_run3/")


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
            "novhhtobb": [
                "GluGluHto2B_PT-200_M-125",
                "VBFHto2B_M-125_dipoleRecoilOn",
            ],
            "vhtobb": [
                "WminusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_Wto2Q_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2Q_M-125",
            ],
            "vbfhh4b": [
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8",
            ],
            "diboson": [
                "WW",
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
        },
    }

    load_columns = [
        ("weight", 1),
        ("MET_pt", 1),
        ("event", 1),
        ("nFatJets", 1),
        ("bbFatJetPt", 2),
        ("bbFatJetEta", 2),
        ("bbFatJetPhi", 2),
        ("bbFatJetMsd", 2),
        ("bbFatJetPNetMass", 2),
        ("bbFatJetPNetXbb", 2),
        ("bbFatJetTau3OverTau2", 2),
        ("bbFatJetPNetQCD0HF", 2),
        ("bbFatJetPNetQCD1HF", 2),
        ("bbFatJetPNetQCD2HF", 2),
    ]

    # to-do change this to msd>30
    filters = [
        [
            ("('bbFatJetPt', '0')", ">=", 300),
            ("('bbFatJetPt', '1')", ">=", 300),
            ("('bbFatJetMsd', '0')", "<=", 250),
            ("('bbFatJetMsd', '1')", "<=", 250),
            ("('bbFatJetMsd', '0')", ">=", 50),
            ("('bbFatJetMsd', '1')", ">=", 50),
        ],
    ]

    # define BDT model
    bdt_model = xgb.XGBClassifier()
    model_name = "v1_msd30_nomulticlass"
    bdt_model.load_model(fname=f"../boosted/bdt_trainings_run3/{model_name}/trained_bdt.model")
    # get function
    make_bdt_dataframe = importlib.import_module("v1_msd30")

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

    # inference and assign score
    events_dict_postprocess = {}
    for key in events_dict:
        bdt_events = make_bdt_dataframe.bdt_dataframe(events_dict[key])
        bdt_score = bdt_model.predict_proba(bdt_events)[:, 1]

        bdt_events["bdt_score"] = bdt_score
        bdt_events["H1Msd"] = events_dict[key]["bbFatJetMsd"].to_numpy()[:, 0]
        bdt_events["H2Msd"] = events_dict[key]["bbFatJetMsd"].to_numpy()[:, 1]
        bdt_events["H2Xbb"] = events_dict[key]["bbFatJetPNetXbb"].to_numpy()[:, 1]
        bdt_events["H2PNetMass"] = events_dict[key]["bbFatJetPNetMass"].to_numpy()[:, 1]

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

        # add more columns (e.g. uncertainties etc)
        bdt_events["weight"] = events_dict[key]["finalWeight"].to_numpy()

        # add selection to testing events
        bdt_events["event"] = events_dict[key]["event"].to_numpy()[:, 0]
        if year == "2022EE" and key in ["qcd", "ttbar", "hh4b"]:
            evt_list = np.load(
                f"../boosted/bdt_trainings_run3/{model_name}/inferences/2022EE/evt_{key}.npy"
            )
            bdt_events = bdt_events[bdt_events["event"].isin(evt_list)]
            bdt_events["weight"] *= 1 / 0.4  # divide by BDT test / train ratio

        # extra selection
        bdt_events = bdt_events[bdt_events["hlt"] == 1]
        bdt_events = bdt_events[bdt_events["H1Msd"] > 30]
        bdt_events = bdt_events[bdt_events["H2Msd"] > 30]

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

    return events_dict_postprocess


def scan_fom(events_combined, fom="2sqrt(b)/s", mass="H2Msd"):
    """
    Scan figure of merit
    """

    def get_nevents_data(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2Xbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        cut_msd = events["H2Msd"] > 30

        # get yield between 75-95
        cut_mass_0 = (events[mass] < 95) & (events[mass] > 75)

        # get yield between 135-155
        cut_mass_1 = (events[mass] < 155) & (events[mass] > 135)

        return np.sum(cut_mass_0 & cut_xbb & cut_bdt & cut_msd) + np.sum(
            cut_mass_1 & cut_xbb & cut_bdt & cut_msd
        )

    def get_nevents_signal(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2Xbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        cut_msd = events["H2Msd"] > 30
        cut_mass = (events[mass] > 95) & (events[mass] < 135)

        # get yield between 95 and 135
        return np.sum(events["weight"][cut_xbb & cut_bdt & cut_msd & cut_mass])

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
            nevents_data = get_nevents_data(events_combined["data"], xbb_cut, bdt_cut)
            nevents_signal = get_nevents_signal(events_combined["hh4b"], xbb_cut, bdt_cut)

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

    eff, bins_x, bins_y = h_sb.to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    cbar = hep.hist2dplot(h_sb, ax=ax, cmin=7, cmax=12, flow="none")
    # cbar = hep.hist2dplot(h_sb, ax=ax, cmin=3, cmax=9, flow="none")
    cbar.cbar.set_label(r"Fig Of Merit", size=18)
    cbar.cbar.ax.get_yaxis().labelpad = 15
    for i in range(len(bins_x) - 1):
        for j in range(len(bins_y) - 1):
            if eff[i, j] > 0:
                ax.text(
                    (bins_x[i] + bins_x[i + 1]) / 2,
                    (bins_y[j] + bins_y[j + 1]) / 2,
                    eff[i, j].round(2),
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
    fig.tight_layout()
    fig.savefig("figofmerit.png")
    fig.savefig("figofmerit.pdf", bbox_inches="tight")


def scan_fom_bin2(
    events_combined, xbb_cut_bin1=0.92, bdt_cut_bin1=0.94, fom="2sqrt(b)/s", mass="H2Msd"
):
    """
    Scan figure of merit for bin2
    """

    def get_nevents_data(events, xbb_cut, bdt_cut):
        cut_bin1 = (events["H2Xbb"] > xbb_cut_bin1) & (events["bdt_score"] > bdt_cut_bin1)
        cut_corner = (events["H2Xbb"] < xbb_cut_bin1) & (events["bdt_score"] < bdt_cut_bin1)
        cut_bin2 = (
            (events["H2Xbb"] > xbb_cut)
            & (events["bdt_score"] > bdt_cut)
            & ~(cut_bin1)
            & ~(cut_corner)
        )
        cut_msd = events["H2Msd"] > 30

        # get yield between 75-95
        cut_mass_0 = (events[mass] < 95) & (events[mass] > 75)

        # get yield between 135-155
        cut_mass_1 = (events[mass] < 155) & (events[mass] > 135)

        return np.sum(cut_mass_0 & cut_bin2 & cut_msd) + np.sum(cut_mass_1 & cut_bin2 & cut_msd)

    def get_nevents_signal(events, xbb_cut, bdt_cut):
        cut_bin1 = (events["H2Xbb"] > xbb_cut_bin1) & (events["bdt_score"] > bdt_cut_bin1)
        cut_corner = (events["H2Xbb"] < xbb_cut_bin1) & (events["bdt_score"] < bdt_cut_bin1)
        cut_bin2 = (
            (events["H2Xbb"] > xbb_cut)
            & (events["bdt_score"] > bdt_cut)
            & ~(cut_bin1)
            & ~(cut_corner)
        )
        cut_msd = events["H2Msd"] > 30
        cut_mass = (events[mass] > 95) & (events[mass] < 135)

        # get yield between 95 and 135
        return np.sum(events["weight"][cut_bin2 & cut_msd & cut_mass])

    xbb_cuts = np.arange(0.7, 0.86, 0.02)
    bdt_cuts = np.arange(0.65, 0.85, 0.01)
    h_sb = hist.Hist(
        hist.axis.Variable(bdt_cuts, name="bdt_cut"),
        hist.axis.Variable(xbb_cuts, name="xbb_cut"),
    )
    for xbb_cut in xbb_cuts:
        figure_of_merits = []
        cuts = []
        for bdt_cut in bdt_cuts:
            nevents_data = get_nevents_data(events_combined["data"], xbb_cut, bdt_cut)
            nevents_signal = get_nevents_signal(events_combined["hh4b"], xbb_cut, bdt_cut)

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

    eff, bins_x, bins_y = h_sb.to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    cbar = hep.hist2dplot(h_sb, ax=ax, cmin=18, cmax=25, flow="none")
    # cbar = hep.hist2dplot(h_sb, ax=ax, cmin=3, cmax=9, flow="none")
    cbar.cbar.set_label(r"Fig Of Merit", size=18)
    cbar.cbar.ax.get_yaxis().labelpad = 15
    for i in range(len(bins_x) - 1):
        for j in range(len(bins_y) - 1):
            if eff[i, j] > 0:
                ax.text(
                    (bins_x[i] + bins_x[i + 1]) / 2,
                    (bins_y[j] + bins_y[j + 1]) / 2,
                    eff[i, j].round(2),
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
    fig.tight_layout()
    fig.savefig("figofmerit_bin2.png")
    fig.savefig("figofmerit_bin2.pdf")


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
    window_by_mass = {"H2Msd": [110, 140], "H2PNetMass": [115, 145]}

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
    for year in args.years:
        events_dict_postprocess[year] = load_run3_samples(args, year)

    # create combined datasets
    # temporarily used 2022EEMC and scale to full luminosity
    lumi_weight_2022EEtoall = (7971.4 + 26337.0 + 17650.0 + 9451.0) / 26337.0
    events_combined = {}
    for key in ["data", "qcd", "hh4b", "ttbar", "ttlep", "vhtobb", "vjets", "diboson", "novhhtobb"]:
        if key in ["data", "ttbar", "ttlep"]:
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
    events_combined["ttbar"] = pd.concat([events_combined["ttbar"], events_combined["ttlep"]])

    if args.fom_scan:
        scan_fom(events_combined, mass=args.mass)
        scan_fom_bin2(
            events_combined,
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

    bkg_keys = ["qcd", "ttbar", "vhtobb", "vjets", "diboson", "novhhtobb"]

    print(events_combined["data"].columns)

    # individual templates per year
    templates = postprocessing.get_templates(
        events_combined,
        bb_masks=None,
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
    args = parser.parse_args()

    postprocess_run3(args)
