from __future__ import annotations

import argparse
import importlib
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb
from correctionlib import schemav2
from hist.intervals import ratio_uncertainty

from HH4b import hh_vars, postprocessing, plotting, run_utils
from HH4b.hh_vars import (
    mreg_strings,
    txbb_strings,
    years,
)
from HH4b.postprocessing import (
    PostProcess,
    combine_run3_samples,
    corrections,
    load_run3_samples,
)
from HH4b.utils import ShapeVar, get_var_mapping, singleVarHist

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

label_by_mass = {
    "H2Msd": r"$m^{2}_\mathrm{SD}$ (GeV)",
    "H2PNetMass": r"$m^{2}_\mathrm{reg}$ (GeV)",
}

HLTs = {
    "2022": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet425_SoftDropMass40",
    ],
    "2022EE": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet425_SoftDropMass40",
    ],
    "2023": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        # "AK8PFJet400_SoftDropMass40", #TODO: add to ntuples
        "AK8PFJet425_SoftDropMass40",
    ],
    "2023BPix": [
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        # "AK8PFJet400_SoftDropMass40", #TODO: add to ntuples
        "AK8PFJet425_SoftDropMass40",
    ],
}

samples_run3 = {
    year: {
        "qcd": ["QCD_HT"],
        "data": [f"{key}_Run" for key in ["JetMET"]],
        "ttbar": ["TTto4Q", "TTto2L2Nu", "TTtoLNu2Q"],
        "diboson": ["ZZ", "WW", "WZ"],
        "vjets": ["Wto2Q-3Jets_HT", "Zto2Q-4Jets_HT"],
    }
    for year in years
}


def load_process_run3_samples(args, year, control_plots, plot_dir):  # noqa: ARG001

    # define BDT model
    bdt_model = xgb.XGBClassifier()
    bdt_model.load_model(fname=f"../boosted/bdt_trainings_run3/{args.bdt_model}/trained_bdt.model")

    # get function
    make_bdt_dataframe = importlib.import_module(
        f".{args.bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
    )

    # define cutflows
    samples_year = list(samples_run3[year].keys())
    cutflow = pd.DataFrame(index=samples_year)
    cutflow_dict = {}

    events_dict_postprocess = {}
    for key in samples_year:
        samples_to_process = {year: {key: samples_run3[year][key]}}

        events_dict = load_run3_samples(
            f"{args.data_dir}/{args.tag}",
            year,
            samples_to_process,
            reorder_txbb=True,
            load_systematics=False,
            txbb_version=args.txbb,
            scale_and_smear=True,
            mass_str=mreg_strings[args.txbb],
        )[key]

        cutflow_dict[key] = OrderedDict(
            [("Skimmer Preselection", np.sum(events_dict["finalWeight"].to_numpy()))]
        )

        bdt_events = make_bdt_dataframe.bdt_dataframe(events_dict, get_var_mapping(""))
        preds = bdt_model.predict_proba(bdt_events)
        PostProcess.add_bdt_scores(bdt_events, preds, "")

        # remove duplicates
        bdt_events = bdt_events.loc[:, ~bdt_events.columns.duplicated()].copy()

        # add more variables for control plots
        bdt_events["H1Pt"] = events_dict["bbFatJetPt"][0]
        bdt_events["H2Pt"] = events_dict["bbFatJetPt"][1]
        bdt_events["H1Msd"] = events_dict["bbFatJetMsd"][0]
        bdt_events["H2Msd"] = events_dict["bbFatJetMsd"][1]
        bdt_events["H1TXbb"] = events_dict[txbb_strings[args.txbb]][0]
        bdt_events["H2TXbb"] = events_dict[txbb_strings[args.txbb]][1]
        bdt_events["H1PNetMass"] = events_dict[mreg_strings[args.txbb]][0]
        bdt_events["H2PNetMass"] = events_dict[mreg_strings[args.txbb]][1]
        bdt_events["bdt_score_finebin"] = bdt_events["bdt_score"]
        bdt_events["bdt_score_coarsebin"] = bdt_events["bdt_score"]

        if key in hh_vars.jmsr_keys:
            for jshift in hh_vars.jmsr_shifts:
                bdt_events[f"H1PNetMass_{jshift}"] = events_dict[
                    f"{mreg_strings[args.txbb]}_{jshift}"
                ][0]
                bdt_events[f"H2PNetMass_{jshift}"] = events_dict[
                    f"{mreg_strings[args.txbb]}_{jshift}"
                ][1]

        # add HLTs
        bdt_events["hlt"] = np.any(
            np.array(
                [
                    events_dict[trigger].to_numpy()[:, 0]
                    for trigger in postprocessing.HLTs[year]
                    if trigger in events_dict
                ]
            ),
            axis=0,
        )

        # weights
        # finalWeight: includes genWeight, puWeight
        nominal_weight = events_dict["finalWeight"].to_numpy()

        # trigger weight

        # tt corrections

        # Unused! code copied from PostProcess
        # if args.bdt_model == "24May31_lr_0p02_md_8_AK4Away":
        #     tt_bdtshape_sf = corrections._load_ttbar_bdtshape_sfs("cat5", args.bdt_model)
        # else:
        #     tt_bdtshape_sf = corrections._load_ttbar_bdtshape_sfs("dummy", "dummy")

        nevents = len(events_dict["bbFatJetPt"][0])
        ttbar_weight = np.ones(nevents)
        if key == "ttbar":
            ptjjsf = corrections._load_ttbar_sfs(year, "PTJJ")
            tau32sf = corrections._load_ttbar_sfs(year, "Tau3OverTau2")
            if args.txbb == "pnet-legacy":
                txbbsf = corrections._load_ttbar_sfs(year, f"{args.txbb}_Xbb")
            else:
                txbbsf = corrections._load_ttbar_sfs(year, "dummy_Xbb")

            ttbar_weight = ptjjsf * txbbsf * tau32sf
        bdt_events["weight_ttbar"] = ttbar_weight

        bdt_events["weight_nottbar"] = nominal_weight
        bdt_events["weight"] = nominal_weight * ttbar_weight

        # HLT selection
        mask_hlt = bdt_events["hlt"] == 1
        bdt_events = bdt_events[mask_hlt]
        cutflow_dict[key]["HLT"] = np.sum(bdt_events["weight"].to_numpy())

        mask_presel = (
            (bdt_events["H1Msd"] > 40) & (bdt_events["H1Pt"] > 300) & (bdt_events["H2Pt"] > 300)
        )
        bdt_events = bdt_events[mask_presel]

        bdt_events["Category"] = 6  # all events
        """
        mask_bin1 = (
            (bdt_events["H1TXbb"] >= 0.9)
            & (bdt_events["H1PNetMass"] > 150)
            & (bdt_events["H1PNetMass"] < 200)
            & (bdt_events["H2PNetMass"] > 50)
            & (bdt_events["H2PNetMass"] < 250)
        )
        bdt_events.loc[mask_bin1, "Category"] = 1
        """
        mask_bin2 = (
            (bdt_events["H1TXbb"] > 0.1)
            & (bdt_events["H2TXbb"] > 0.1)
            & (bdt_events["H1T32"] < 0.46)
            & (bdt_events["H1PNetMass"] > 150)
            & (bdt_events["H1PNetMass"] < 200)
            & (bdt_events["H2PNetMass"] > 50)
            & (bdt_events["H2PNetMass"] < 250)
        )
        bdt_events.loc[mask_bin2, "Category"] = 2

        """
        mask_bin3 = (
            (bdt_events["H1TXbb"] >= 0.9)
            # & (bdt_events["H2TXbb"] > 0.1)
            & (bdt_events["H1T32"] < 0.6)
            & (bdt_events["H1PNetMass"] > 160)
            & (bdt_events["H1PNetMass"] < 200)
            & (bdt_events["H2PNetMass"] > 50)
            & (bdt_events["H2PNetMass"] < 250)
        )
        bdt_events.loc[mask_bin3, "Category"] = 3

        mask_bin4 = (
            (bdt_events["H1TXbb"] > 0.8)
            & (bdt_events["H2TXbb"] > 0.1)
            & (bdt_events["H1PNetMass"] > 150)
            & (bdt_events["H1PNetMass"] < 200)
            & (bdt_events["H2PNetMass"] > 50)
            & (bdt_events["H2PNetMass"] < 250)
        )
        bdt_events.loc[mask_bin4, "Category"] = 4
        """
        mask_bin5 = (
            (bdt_events["H1TXbb"] > 0.1)
            & (bdt_events["H2TXbb"] > 0.1)
            & (bdt_events["H1T32"] < 0.6)
            & (bdt_events["H1PNetMass"] > 150)
            & (bdt_events["H1PNetMass"] < 200)
            & (bdt_events["H2PNetMass"] > 50)
            & (bdt_events["H2PNetMass"] < 250)
        )
        bdt_events.loc[mask_bin5, "Category"] = 5

        # save cutflows for nominal variables
        cutflow_dict[key]["H1Msd > 40 & Pt>300"] = np.sum(bdt_events["weight"].to_numpy())
        # cutflow_dict[key]["H1TXbb>0.9, H1M:[150-200]"] = np.sum(
        #    bdt_events["weight"][mask_bin1].to_numpy()
        # )
        cutflow_dict[key]["H1TXbb>0.1,H2TXbb>0.1,H1T32<0.46, H1M:[150-200]"] = np.sum(
            bdt_events["weight"][mask_bin2].to_numpy()
        )
        """
        cutflow_dict[key]["H1TXbb>0.9,H1T32<0.46, H1M:[160-200]"] = np.sum(
            bdt_events["weight"][mask_bin3].to_numpy()
        )
        cutflow_dict[key]["H1TXbb>0.8,H2TXbb>0.1,H1M:[150-200]"] = np.sum(
            bdt_events["weight"][mask_bin4].to_numpy()
        )
        """
        cutflow_dict[key]["H1TXbb>0.1,H2TXbb>0.1,H1T32<0.6, H1M:[160-200]"] = np.sum(
            bdt_events["weight"][mask_bin5].to_numpy()
        )

        # keep some (or all) columns
        columns = [
            "H2TXbb",
            "weight",
            "Category",
            "bdt_score",
            "H2PNetMass",
            "HHPt",
            "H1PNetMass",
            "bdt_score_finebin",
            "bdt_score_coarsebin",
        ]
        columns = list(set(columns))

        if control_plots:
            bdt_events = bdt_events.rename(
                columns={"H1T32": "H1T32top", "H2T32": "H2T32top", "H1Pt/H2Pt": "H1Pt_H2Pt"}
            )
            events_dict_postprocess[key] = bdt_events[columns]
        else:
            events_dict_postprocess[key] = bdt_events[columns]

    # end of loop over samples

    """
    if control_plots:
        for i in range(1,6):
            events_to_plot = {
                key: events[events["Category"] == i] for key, events in events_dict_postprocess.items()
            }
            #print(events_to_plot)
            make_control_plots(
                events_to_plot,
                plot_dir,
                year,
                args.legacy,
                f"cat{i}",
                ["diboson", "vjets", "qcd", "ttbar"],
                model=args.bdt_model
            )
    """

    for cut in cutflow_dict["data"]:
        cutflow[cut] = [cutflow_dict[key][cut].round(2) for key in events_dict_postprocess]

    print("\nCutflow")
    print(cutflow)
    return events_dict_postprocess, cutflow


def get_corr(corr_key, eff, eff_unc_up, eff_unc_dn, year, edges):
    def singlebinning(eff):
        return schemav2.Binning(
            nodetype="binning",
            input=corr_key,
            edges=list(edges),
            content=list(eff.flatten()),
            flow=1.0,  # SET FLOW TO 1.0
        )

    corr = schemav2.Correction(
        name=f"ttbar_corr_{corr_key}_{year}",
        description=f"ttbar correction {corr_key} for {year}",
        version=1,
        inputs=[
            schemav2.Variable(
                name=corr_key,
                type="real",
                description=corr_key,
            ),
            schemav2.Variable(
                name="systematic",
                type="string",
                description="Systematic variation",
            ),
        ],
        output=schemav2.Variable(name="weight", type="real", description="ttbar efficiency"),
        data=schemav2.Category(
            nodetype="category",
            input="systematic",
            content=[
                {"key": "nominal", "value": singlebinning(eff)},
                {"key": "stat_up", "value": singlebinning(eff_unc_up)},
                {"key": "stat_dn", "value": singlebinning(eff_unc_dn)},
            ],
        ),
    )
    return corr


def make_control_plots(events_dict, plot_dir, year, txbb_version, tag, bgorder, model):
    if txbb_version == "pnet-legacy":
        txbb_label = "PNet Legacy"
    elif txbb_version == "pnet-v12":
        txbb_label = "PNet 103X"
    elif txbb_version == "glopart-v2":
        txbb_label = "GloParTv2"

    control_plot_vars = [
        # ShapeVar(var="H1Msd", label=r"$m_{SD}^{1}$ (GeV)", bins=[30, 0, 300]),
        # ShapeVar(var="H2Msd", label=r"$m_{SD}^{2}$ (GeV)", bins=[30, 0, 300]),
        # ShapeVar(var="H1TXbb", label=r"Xbb$^{1}$ " + legacy_label, bins=[30, 0, 1]),
        # ShapeVar(var="H2TXbb", label=r"Xbb$^{2}$ " + legacy_label, bins=[30, 0, 1]),
        # ShapeVar(var="H1TXbbNoLeg", label=r"Xbb$^{1}$ v12", bins=[30, 0, 1]),
        # ShapeVar(var="H2TXbbNoLeg", label=r"Xbb$^{2}$ v12", bins=[30, 0, 1]),
        ShapeVar(var="H1PNetMass", label=r"$m_{reg}^{1}$ (GeV) " + txbb_label, bins=[30, 0, 300]),
        ShapeVar(var="H2PNetMass", label=r"$m_{reg}^{2}$ (GeV) " + txbb_label, bins=[30, 0, 300]),
        ShapeVar(
            var="HHPt",
            label=r"HH $p_{T}$ (GeV)",
            bins=[0, 40, 80, 120, 200, 350, 500, 700, 1000],
            reg=False,
        ),
        # ShapeVar(var="HHeta", label=r"HH $\eta$", bins=[30, -5, 5]),
        # ShapeVar(var="HHmass", label=r"HH mass (GeV)", bins=[30, 0, 1500]),
        # ShapeVar(var="MET", label=r"MET (GeV)", bins=[30, 0, 600]),
        # ShapeVar(var="H1T32top", label=r"$\tau_{32}^{1}$", bins=[30, 0, 1]),
        # ShapeVar(var="H2T32top", label=r"$\tau_{32}^{2}$", bins=[30, 0, 1]),
        # ShapeVar(var="H1Pt", label=r"H $p_{T}^{1}$ (GeV)", bins=[30, 200, 1000]),
        # ShapeVar(var="H2Pt", label=r"H $p_{T}^{2}$ (GeV)", bins=[30, 200, 1000]),
        # ShapeVar(var="H1eta", label=r"H $\eta^{1}$", bins=[30, -4, 4]),
        # ShapeVar(var="H1QCDb", label=r"QCDb$^{2}$", bins=[30, 0, 1]),
        # ShapeVar(var="H1QCDbb", label=r"QCDbb$^{2}$", bins=[30, 0, 1]),
        # ShapeVar(var="H1QCDothers", label=r"QCDothers$^{1}$", bins=[30, 0, 1]),
        # ShapeVar(var="H1Pt_HHmass", label=r"H$^1$ $p_{T}/mass$", bins=[30, 0, 1]),
        # ShapeVar(var="H2Pt_HHmass", label=r"H$^2$ $p_{T}/mass$", bins=[30, 0, 0.7]),
        # ShapeVar(var="H1Pt_H2Pt", label=r"H$^1$/H$^2$ $p_{T}$ (GeV)", bins=[30, 0.5, 1]),
        ShapeVar(var="bdt_score", label=r"BDT score", bins=[30, 0, 1]),
        ShapeVar(
            var="bdt_score_coarsebin",
            label=r"BDT score",
            bins=[0, 0.3, 0.68, 1],
            reg=False,
        ),
        ShapeVar(
            var="bdt_score_finebin",
            label=r"BDT score",
            # bins=[0, 0.03, 0.3, 0.68, 0.9, 1],
            bins=[0, 0.03, 0.3, 0.5, 0.7, 0.93, 1],  # if I move to 0.92 I get disagreement
            reg=False,
        ),
    ]

    odir = f"control_{tag}/{year}"
    (plot_dir / odir).mkdir(exist_ok=True, parents=True)

    os.system(f"mkdir -p ../corrections/data/{model}")

    hists = {}
    for shape_var in control_plot_vars:
        if shape_var.var not in hists:
            hists[shape_var.var] = singleVarHist(
                events_dict,
                shape_var,
                weight_key="weight",
            )
            # print(shape_var.var, hists[shape_var.var]["data", :])

            plotting.ratioHistPlot(
                hists[shape_var.var],
                year,
                [],
                ["qcd", "ttbar", "vjets", "diboson"],
                name=f"{plot_dir}/{odir}/{shape_var.var}",
                show=False,
                log=True,
                bg_order=bgorder,
                plot_significance=False,
                significance_dir=shape_var.significance_dir,
                ratio_ylims=[0.2, 1.8],
                bg_err_mcstat=True,
                reweight_qcd=False,
                save_pdf=False,
            )

        if (shape_var.var == "bdt_score_coarsebin" and tag == "cat3") or (
            shape_var.var == "bdt_score_finebin" and tag != "cat3"
        ):
            # save correction
            num = (
                hists[shape_var.var]["data", :].values()
                - sum(
                    [hists[shape_var.var][bkg, :] for bkg in ["qcd", "vjets", "diboson"]]
                ).values()
            )
            den = hists[shape_var.var]["ttbar", :].values()
            tt = hists[shape_var.var]["ttbar", :]
            sfhist_values = num / den
            yerr = ratio_uncertainty(num, den, "poisson")
            edges = tt.axes.edges[0]
            corr = get_corr(
                "bdtshape",
                sfhist_values,
                yerr[1],
                yerr[0],
                year,
                edges,
            )

            cset = schemav2.CorrectionSet(
                schema_version=2,
                corrections=[corr],
            )
            path = Path(f"../corrections/data/{model}/ttbar_bdtshape{tag}_{year}.json")
            with path.open("w") as fout:
                fout.write(cset.json(exclude_unset=True))


def postprocess_run3(args):
    global bg_keys  # noqa: PLW0602

    plot_dir = Path(f"../../../plots/PostProcess/{args.templates_tag}")
    plot_dir.mkdir(exist_ok=True, parents=True)

    # load samples
    events_dict_postprocess = {}
    cutflows = {}
    for year in args.years:
        print(f"\n{year}")
        events_dict_postprocess[year], cutflows[year] = load_process_run3_samples(
            args,
            year,
            args.control_plots,
            plot_dir,
        )

    print("Loaded all years ", args.years)

    if len(args.years) > 1:
        events_combined, scaled_by = combine_run3_samples(
            events_dict_postprocess,
            ["data", "ttbar", "qcd", "vjets", "diboson"],
            years_run3=args.years,
        )
        print("Combined years")
    else:
        events_combined = events_dict_postprocess[args.years[0]]

    # for i in range(1,6):
    for i in [2, 5]:
        events_to_plot = {
            key: events[events["Category"] == i] for key, events in events_combined.items()
        }
        # print(i, events_to_plot["data"])
        make_control_plots(
            events_to_plot,
            plot_dir,
            "2022-2023",
            args.txbb,
            f"cat{i}",
            # ["diboson", "vjets", "qcd", "ttbar"],
            ["qcd"],
            model=args.bdt_model,
        )


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
        default="24Apr21_legacy_vbf_vars",
        help="BDT model to load",
    )
    parser.add_argument(
        "--bdt-config",
        type=str,
        default="24Apr21_legacy_vbf_vars",
        help="BDT model to load",
    )
    parser.add_argument(
        "--txbb",
        type=str,
        default="",
        choices=["pnet-legacy", "pnet-v12", "glopart-v2"],
        help="version of TXbb tagger/mass regression to use",
    )
    run_utils.add_bool_arg(parser, "control-plots", default=False, help="make control plots")
    args = parser.parse_args()

    print(args)
    postprocess_run3(args)
