from __future__ import annotations

import argparse
import importlib
import os
import sys

# temp
import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b import postprocessing
from HH4b.postprocessing import Region
from HH4b.utils import ShapeVar, format_columns, load_samples

sys.path.append("../boosted/bdt_trainings_run3/")


def load_run3_samples(args, year):
    # modify as needed
    input_dir = f"/eos/uscms/store/user/cmantill/bbbb/skimmer/{args.tag}"

    samples_run3 = {
        "2022": {
            # "ttbar": [
            #     "TTto4Q",
            #     "TTtoLNu2Q",
            #     "TTto2L2Nu",
            # ],
            # "hh4b": [
            #     "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV",
            # ],
            "data": [
                "JetMET_Run2022C",
                "JetMET_Run2022C_single",
                "JetMET_Run2022D",
            ],
            # "novhhtobb": [
            #     "GluGluHto2B_PT-200_M-125",
            #     "VBFHto2B_M-125_dipoleRecoilOn",
            # ],
            # "vhtobb": [
            #     "WminusH_Hto2B_Wto2Q_M-125",
            #     "WplusH_Hto2B_Wto2Q_M-125",
            #     "ZH_Hto2B_Zto2Q_M-125",
            #     "ggZH_Hto2B_Zto2Q_M-125",
            # ],
            # "vbfhh4b": [
            #     "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8",
            # ],
            # "diboson": [
            #     "WW",
            #     "ZZ",
            # ],
            # "vjets": [
            #     "Wto2Q-3Jets_HT-200to400",
            #     "Wto2Q-3Jets_HT-400to600",
            #     "Wto2Q-3Jets_HT-600to800",
            #     "Wto2Q-3Jets_HT-800",
            #     "Zto2Q-4Jets_HT-200to400",
            #     "Zto2Q-4Jets_HT-400to600",
            #     "Zto2Q-4Jets_HT-600to800",
            #     "Zto2Q-4Jets_HT-800"
            # ],
        },
        "2022EE": {
            "ttbar": [
                "TTto4Q",
                "TTtoLNu2Q",
                "TTto2L2Nu",
            ],
            "hh4b": [
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV",
            ],
            "data": [
                "JetMET_Run2022E",
                "JetMET_Run2022F",
                "JetMET_Run2022G",
            ],
            "novhhtobb": [
                "GluGluHto2B_PT-200_M-125",
                "VBFHto2B_M-125_dipoleRecoilOn",
            ],
            # "qcd": [
            #     "QCD_HT-200to400",
            #     "QCD_HT-400to600",
            #     "QCD_HT-600to800",
            #     "QCD_HT-800to1000",
            #     "QCD_HT-1000to1200",
            #     "QCD_HT-1200to1500",
            #     "QCD_HT-1500to2000",
            #     "QCD_HT-2000",
            # ],
            "vhtobb": [
                "WminusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_Wto2Q_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2Q_M-125",
            ],
            # "vbfhh4b": [
            #     "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8",
            # ],
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
        },
        "2023-pre-BPix": {
            "data": [
                "JetMET_Run2023C",
            ],
        },
        "2023-BPix": {
            "data": [
                "JetMET_Run2023D",
            ],
        },
    }

    load_columns = [
        ("weight", 1),
        ("MET_pt", 1),
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
            ("('bbFatJetPt', '0')", ">=", 270),
            ("('bbFatJetPt', '1')", ">=", 270),
            ("('bbFatJetMsd', '0')", "<=", 250),
            ("('bbFatJetMsd', '1')", "<=", 250),
            ("('bbFatJetMsd', '0')", ">=", 50),
            ("('bbFatJetMsd', '1')", ">=", 50),
        ],
    ]

    # define BDT model
    bdt_model = xgb.XGBClassifier()
    model_name = "v0"
    bdt_model.load_model(fname=f"../boosted/bdt_trainings_run3/{model_name}/trained_bdt.model")
    # get function
    make_bdt_dataframe = importlib.import_module(f"{model_name}")

    # pre-selection
    events_dict = load_samples(
        input_dir,
        samples_run3[year],
        year,
        filters=filters,
        columns_mc=format_columns(load_columns),
    )

    # inference and assign score
    events_dict_postprocess = {}
    for key in events_dict:
        bdt_events = make_bdt_dataframe.bdt_dataframe(events_dict[key])
        bdt_score = bdt_model.predict_proba(bdt_events)[:, 1]

        bdt_events["bdt_score"] = bdt_score
        bdt_events["H2Msd"] = events_dict[key]["bbFatJetMsd"].to_numpy()[:, 1]
        bdt_events["H2Xbb"] = events_dict[key]["bbFatJetPNetXbb"].to_numpy()[:, 1]

        # add more columns (e.g. uncertainties etc)
        bdt_events["weight"] = 1
        if key != "data":
            bdt_events["weight"] = events_dict[key]["weight"].to_numpy()[:, 0]

        # define category
        bdt_events["Category"] = 5  # all events
        bdt_events.loc[
            (bdt_events["H2Xbb"] > 0.9) & (bdt_events["bdt_score"] > 0.93),
            "Category",
        ] = 1
        bdt_events.loc[
            (bdt_events["H2Xbb"] < 0.9) & (bdt_events["bdt_score"] > 0.03),
            "Category",
        ] = 4

        columns = ["Category", "H2Msd", "bdt_score", "H2Xbb", "weight"]
        events_dict_postprocess[key] = bdt_events[columns]

    return events_dict_postprocess


def scan_fom(events_combined):
    """
    Scan figure of merit
    """

    def get_nevents_data(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2Xbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        cut_msd = (events["H2Msd"] > 50) & (events["H2Msd"] < 220)

        # get yield between 75-95
        cut_msd_0 = (events["H2Msd"] < 95) & (events["H2Msd"] > 75)

        # get yield between 135-155
        cut_msd_1 = (events["H2Msd"] < 155) & (events["H2Msd"] > 135)

        return np.sum(cut_msd_0 & cut_xbb & cut_bdt & cut_msd) + np.sum(
            cut_msd_1 & cut_xbb & cut_bdt & cut_msd
        )

    def get_nevents_signal(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2Xbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        cut_msd = (events["H2Msd"] > 95) & (events["H2Msd"] < 135)

        # get yield between 95 and 135
        return np.sum(events["weight"][cut_xbb & cut_bdt & cut_msd])

    xbb_cuts = [0.8, 0.9, 0.95, 0.98]
    for xbb_cut in xbb_cuts:
        figure_of_merits = []
        cuts = []
        for bdt_cut in np.arange(0.01, 1, 0.01):
            nevents_data = get_nevents_data(events_combined["data"], xbb_cut, bdt_cut)
            nevents_signal = get_nevents_signal(events_combined["hh4b"], xbb_cut, bdt_cut)

            figure_of_merit = 2 * np.sqrt(nevents_data) / nevents_signal

            if nevents_signal > 0.5:
                cuts.append(bdt_cut)
                figure_of_merits.append(figure_of_merit)
                print(
                    f"Xbb_Cut: {xbb_cut}, BDT_Cut: {bdt_cut:.2f}, NBkg: {nevents_data}, NSig: {nevents_signal:.2f}, FigOfMerit: {figure_of_merit:.2f}"
                )

        if len(cuts) > 0:
            cuts = np.array(cuts)
            figure_of_merits = np.array(figure_of_merits)
            smallest = np.argmin(figure_of_merits)

            print(xbb_cut, cuts[smallest], figure_of_merits[smallest])


def postprocess_run3(args):
    selection_regions = {
        "pass_bin1": Region(
            cuts={
                "Category": [1, 2],
            },
            label="Bin1",
        ),
        "fail": Region(
            cuts={
                "Category": [4, 5],
            },
            label="Fail",
        ),
    }

    # variable to fit
    fit_shape_var = ShapeVar(
        "H2Msd",
        r"$m^{2}_\mathrm{SD}$ (GeV)",
        [17, 50, 220],
        reg=True,
        blind_window=[110, 140],
    )

    # load samples
    years = args.years.split(",")
    events_dict_postprocess = {}
    for year in years:
        events_dict_postprocess[year] = load_run3_samples(args, year)

    # create combined datasets
    # temporarily used 2022EEMC and scale to full luminosity
    lumi_weight_2022EEtoall = (7971.4 + 26337.0 + 17650.0 + 9451.0) / 26337.0
    events_combined = {}
    for key in ["data", "hh4b", "ttbar", "vhtobb", "vjets", "diboson", "novhhtobb"]:
        if key == "data":
            combined = pd.concat(
                [
                    events_dict_postprocess["2022"][key],
                    events_dict_postprocess["2022EE"][key],
                    events_dict_postprocess["2023-pre-BPix"][key],
                    events_dict_postprocess["2023-BPix"][key],
                ]
            )
        else:
            combined = events_dict_postprocess["2022EE"][key].copy()
            combined["weight"] = combined["weight"] * lumi_weight_2022EEtoall
        events_combined[key] = combined

    scan_fom(events_combined)

    templ_dir = f"./templates/{args.template_dir}"
    year = "2022-2023"
    os.system(f"mkdir -p {templ_dir}/cutflows/{year}")
    os.system(f"mkdir -p {templ_dir}/{year}")

    bkg_keys = ["ttbar", "vhtobb", "vjets", "diboson", "novhhtobb"]

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
        show=True,
        energy=13.6,
    )

    # save templates per year
    postprocessing.save_templates(templates, f"{templ_dir}/{year}_templates.pkl", fit_shape_var)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template-dir",
        type=str,
        required=True,
        help="output pickle directory of hist.Hist templates",
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
        default="2022,2022EE,2023-pre-BPix,2023-BPix",
        help="years to postprocess",
    )
    args = parser.parse_args()

    postprocess_run3(args)
