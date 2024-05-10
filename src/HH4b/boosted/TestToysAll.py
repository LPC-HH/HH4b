from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b import postprocessing
from HH4b.hh_vars import bg_keys, samples_run3
from HH4b.utils import ShapeVar

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 12})
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.color"] = "#CCCCCC"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["figure.edgecolor"] = "none"

mass_axis = hist.axis.Regular(20, 50, 250, name="mass")
bdt_axis = hist.axis.Regular(60, 0, 1, name="bdt")
diff_axis = hist.axis.Regular(100, -2, 2, name="diff")
cut_axis = hist.axis.StrCategory([], name="cut", growth=True)
xbbcut_axis = hist.axis.StrCategory([], name="xbbcut", growth=True)

legacy_label = "Legacy"

CUT_MAX_VAL = 9999.0


def get_toy_from_hist(h_hist):
    """
    Get values drawn from histogram
    """

    h, bins = h_hist.to_numpy()
    integral = int(np.sum(h_hist.values()))

    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(h)
    cdf = cdf / cdf[-1]
    values = np.random.rand(integral)  # noqa: NPY002
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    return random_from_cdf


def get_dataframe(events_dict, year, bdt_model_name, bdt_config):
    bdt_model = xgb.XGBClassifier()
    bdt_model.load_model(fname=f"../boosted/bdt_trainings_run3/{bdt_model_name}/trained_bdt.model")
    make_bdt_dataframe = importlib.import_module(
        f".{bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
    )

    bdt_events_dict = {}
    for key in events_dict:
        events = events_dict[key]
        bdt_events = make_bdt_dataframe.bdt_dataframe(events)
        preds = bdt_model.predict_proba(bdt_events)
        # inference
        bdt_events["bdt_score"] = preds[:, 0]

        # extra variables
        bdt_events["H2PNetMass"] = events["bbFatJetPNetMassLegacy"][1]
        bdt_events["H1Msd"] = events["bbFatJetMsd"][0]
        bdt_events["H1TXbb"] = events[f"bbFatJetPNetTXbb{legacy_label}"][0]
        bdt_events["H2TXbb"] = events[f"bbFatJetPNetTXbb{legacy_label}"][1]
        bdt_events["weight"] = events["finalWeight"].to_numpy()

        bdt_events["hlt"] = np.any(
            np.array(
                [events[trigger][0] for trigger in postprocessing.HLTs[year] if trigger in events]
            ),
            axis=0,
        )
        mask_hlt = bdt_events["hlt"] == 1

        # masks
        mask_presel = (
            (bdt_events["H1Msd"] > 30)
            & (bdt_events["H1Pt"] > 300)
            & (bdt_events["H2Pt"] > 300)
            & (bdt_events["H1TXbb"] > 0.8)
        )
        mask_mass = (bdt_events["H2PNetMass"] > 50) & (bdt_events["H2PNetMass"] < 250)
        bdt_events = bdt_events[(mask_mass) & (mask_hlt) & (mask_presel)]

        columns = ["bdt_score", "H2TXbb", "H2PNetMass", "weight"]
        bdt_events_dict[key] = bdt_events[columns]
    return bdt_events_dict


def sideband_fom(mass_data, mass_sig, cut_data, cut_sig, weight_data, weight_signal, mass_window):
    mw_size = mass_window[1] - mass_window[0]

    # get yield in left sideband (half the size of the mass window)
    cut_mass_0 = (mass_data < mass_window[0]) & (mass_data > (mass_window[0] - mw_size / 2))
    # get yield in right sideband (half the size of the mass window)
    cut_mass_1 = (mass_data < mass_window[1] + mw_size / 2) & (mass_data > mass_window[1])
    # data yield in sideband
    nevents_bkg = np.sum(weight_data[(cut_mass_0 | cut_mass_1) & cut_data])

    cut_mass = (mass_sig >= mass_window[0]) & (mass_sig <= mass_window[1])
    # signal yield in Higgs mass window
    nevents_sig = np.sum(weight_signal[cut_sig & cut_mass])

    return nevents_sig, nevents_bkg


def abcd_fom(
    mass_data,
    mass_sig,
    mass_others,
    cut_data,
    cut_sig,
    cut_others,
    weight_data,
    weight_signal,
    weight_others,
    # definitions of mass values, BDT cut and weights for inverted Xbb regions (C,D)
    mass_inv_data,
    mass_inv_others,
    invcut_data,
    invcut_others,
    weight_inv_data,
    weight_inv_others,
    mass_window,
):
    # get A,B,C,D
    dicts = {"data": [], "others": []}

    cut_mass = (mass_data >= mass_window[0]) & (mass_data <= mass_window[1])
    cut_mass_invregion = (mass_inv_data >= mass_window[0]) & (mass_inv_data <= mass_window[1])
    dicts["data"] = [
        0,  # A
        np.sum(weight_data[~cut_mass & cut_data]),  # B,
        np.sum(weight_inv_data[cut_mass_invregion & invcut_data]),  # C
        np.sum(weight_inv_data[~cut_mass_invregion & invcut_data]),  # D
    ]

    if mass_others is not None:
        cut_mass = (mass_others >= mass_window[0]) & (mass_others <= mass_window[1])
        cut_mass_invregion = (mass_inv_others >= mass_window[0]) & (
            mass_inv_others <= mass_window[1]
        )

        dicts["others"] = [
            np.sum(weight_others[cut_mass & cut_others]),  # A
            np.sum(weight_others[~cut_mass & cut_others]),  # B
            np.sum(weight_inv_others[cut_mass_invregion & invcut_others]),  # C
            np.sum(weight_inv_others[~cut_mass_invregion & invcut_others]),  # D
        ]
        # subtract other backgrounds
        dmt = np.array(dicts["data"]) - np.array(dicts["others"])
    else:
        dmt = np.array(dicts["data"])

    cut_mass = (mass_sig >= mass_window[0]) & (mass_sig <= mass_window[1])
    nevents_sig = np.sum(weight_signal[cut_sig & cut_mass])

    # C/D * B
    bqcd = dmt[2] * dmt[1] / dmt[3]
    nevents_bkg = bqcd + dicts["others"][0] if mass_others is not None else bqcd
    return nevents_sig, nevents_bkg


def main(args):

    # do not load any QCD samples
    # for year in samples_run3:
    #     for key in ["qcd"]:
    #         if key in samples_run3[year]:
    #             samples_run3[year].pop(key)

    data_dir = "24Apr23LegacyLowerThresholds_v12_private_signal"
    # input_dir = f"/eos/uscms/store/user/cmantill/bbbb/skimmer/{data_dir}"
    input_dir = f"/ceph/cms/store/user/rkansal/bbbb/skimmer/{data_dir}"
    year = "2022EE"

    events_dict = postprocessing.load_run3_samples(
        input_dir=input_dir,
        year=year,
        legacy=True,
        samples_run3=samples_run3,
        reorder_txbb=True,
        txbb=f"bbFatJetPNetTXbb{legacy_label}",
    )

    mass_var = "H2PNetMass"
    bdt_events_dict = get_dataframe(events_dict, year, args.bdt_model_name, args.bdt_config)

    # get other backgrounds
    bdt_events_dict["others"] = pd.concat([bdt_events_dict[key] for key in bg_keys if key != "qcd"])

    ntoys = args.ntoys
    print(f"Number of toys {ntoys}")

    mass_window = [110, 140]

    all_bdt_cuts = 0.01 * np.arange(80, 100)
    bdt_cuts = all_bdt_cuts
    # bdt_cuts = [0.8, 0.9, 0.95]
    # xbb_cuts = [0.8, 0.95]
    all_xbb_cuts = 0.01 * np.arange(80, 100)
    xbb_cuts = all_xbb_cuts

    # define fail region for ABCD
    bdt_fail = 0.03
    xbb_cut_fail = 0.8

    h_pull = hist.Hist(diff_axis)

    # fixed signal k-factor
    kfactor_signal = 80
    print(f"Fixed factor by which to scale signal: {kfactor_signal}")

    # variable to fit
    fit_shape_var = ShapeVar(
        mass_var,
        r"$m^{2}_\mathrm{reg}$ (GeV)",
        [16, 60, 220],
        reg=True,
        # blind or not?
        blind_window=None,
        # blind_window=window_by_mass[args.mass],
    )

    ####################################################################
    # Evaluate expected sensitivity for each combination of Xbb, BDT cut
    ####################################################################
    expected_soverb_by_cut = {}
    cuts = []
    figure_of_merits = []

    bdt_events_data = bdt_events_dict["data"]
    bdt_events_sig = bdt_events_dict["hh4b"]
    bdt_events_others = bdt_events_dict["others"]

    for xbb_cut in xbb_cuts:
        bdt_events_data_invertedXbb = bdt_events_dict["data"][
            bdt_events_dict["data"]["H2TXbb"] < xbb_cut_fail
        ]
        bdt_events_others_invertedXbb = bdt_events_dict["others"][
            bdt_events_dict["others"]["H2TXbb"] < xbb_cut_fail
        ]

        ################################################
        # Evaluate expected sensitivity for each BDT cut
        ################################################
        expected_soverb_by_cut[xbb_cut] = {}

        for bdt_cut in bdt_cuts:
            if args.method == "sideband":
                nevents_sig, nevents_bkg = sideband_fom(
                    bdt_events_data[mass_var],
                    bdt_events_sig[mass_var],
                    (bdt_events_data["bdt_score"] >= bdt_cut)
                    & (bdt_events_data["H2TXbb"] >= xbb_cut),
                    (bdt_events_sig["bdt_score"] >= bdt_cut)
                    & (bdt_events_sig["H2TXbb"] >= xbb_cut),
                    bdt_events_data["weight"],
                    bdt_events_sig["weight"],
                    mass_window,
                )
            else:
                nevents_sig, nevents_bkg = abcd_fom(
                    mass_data=bdt_events_data[mass_var],
                    mass_sig=bdt_events_sig[mass_var],
                    mass_others=bdt_events_others[mass_var],
                    cut_data=(bdt_events_data["bdt_score"] >= bdt_cut)
                    & (bdt_events_data["H2TXbb"] >= xbb_cut),
                    cut_sig=(bdt_events_sig["bdt_score"] >= bdt_cut)
                    & (bdt_events_sig["H2TXbb"] >= xbb_cut),
                    cut_others=(bdt_events_others["bdt_score"] >= bdt_cut)
                    & (bdt_events_others["H2TXbb"] >= xbb_cut),
                    weight_data=bdt_events_data["weight"],
                    weight_signal=bdt_events_sig["weight"],
                    weight_others=bdt_events_others["weight"],
                    # INVERTED stuff
                    mass_inv_data=bdt_events_data_invertedXbb[mass_var],
                    mass_inv_others=bdt_events_others_invertedXbb[mass_var],
                    invcut_data=(
                        bdt_events_data_invertedXbb["bdt_score"] < bdt_fail
                    ),  # fail region
                    invcut_others=(bdt_events_others_invertedXbb["bdt_score"] < bdt_fail),
                    weight_inv_data=bdt_events_data_invertedXbb["weight"],
                    weight_inv_others=bdt_events_others_invertedXbb["weight"],
                    mass_window=mass_window,
                )
            nevents_sig_scaled = nevents_sig * kfactor_signal
            soversb = nevents_sig_scaled / np.sqrt(nevents_bkg + nevents_sig_scaled)
            expected_soverb_by_cut[xbb_cut][bdt_cut] = soversb

            if nevents_sig > 0.5 and nevents_bkg >= 2:
                cuts.append(bdt_cut)
                figure_of_merits.append(soversb)

    # compute the optimal cut for all possible combinations
    sensitivity_data = 0
    if len(cuts) > 0:
        cuts = np.array(cuts)
        figure_of_merits = np.array(figure_of_merits)
        biggest = np.argmax(figure_of_merits)
        optimal_cut_data = cuts[biggest]
        sensitivity_data = figure_of_merits[biggest]
        print(
            f"From Data: Xbb:{xbb_cut:.3f} BDT:{optimal_cut_data:.2f} S/(S+B):{sensitivity_data:.2f}"
        )
        print(f"Expected sensitivity for optimal Xbb and BDT cut {sensitivity_data}")

    ###################
    # TOYS
    ###################

    # create toy from data mass distribution (before xbb or BDT cut)
    h_mass = hist.Hist(mass_axis)
    h_mass.fill(bdt_events_data[mass_var])

    print("Xbb BDT Index-BDT S/(S+B) Difference Expected")
    for itoy in range(ntoys):
        print("itoy ", itoy)
        templ_dir = Path(f"templates/alltoys_{args.tag}/toy_{itoy}")
        (templ_dir / year).mkdir(parents=True, exist_ok=True)
        (templ_dir / "cutflows" / year).mkdir(parents=True, exist_ok=True)

        random_mass = get_toy_from_hist(h_mass)
        print("build random")

        # build toy = data + injected signal
        mass_toy = np.concatenate([random_mass, bdt_events_sig[mass_var]])
        bdt_toy = np.concatenate([bdt_events_data["bdt_score"], bdt_events_sig["bdt_score"]])
        xbb_toy = np.concatenate([bdt_events_data["H2TXbb"], bdt_events_sig["H2TXbb"]])
        # sum weights together, but scale weight of signal
        weight_toy = np.concatenate(
            [bdt_events_data["weight"], bdt_events_sig["weight"] * kfactor_signal]
        )

        cuts_xbb = []
        cuts_bdt = []
        figure_of_merits = []
        for xbb_cut in xbb_cuts:
            for bdt_cut in bdt_cuts:
                if args.method == "sideband":
                    nevents_sig_bdt_cut, nevents_bkg_bdt_cut = sideband_fom(
                        mass_toy,
                        bdt_events_sig[mass_var],
                        (bdt_toy >= bdt_cut) & (xbb_toy >= xbb_cut),
                        (bdt_events_sig["bdt_score"] >= bdt_cut)
                        & (bdt_events_sig["H2TXbb"] >= xbb_cut),
                        weight_toy,
                        bdt_events_sig["weight"] * kfactor_signal,
                        mass_window,
                    )
                else:
                    nevents_sig_bdt_cut, nevents_bkg_bdt_cut = abcd_fom(
                        mass_toy,
                        bdt_events_sig[mass_var],
                        bdt_events_others[mass_var],
                        (bdt_toy >= bdt_cut) & (xbb_toy >= xbb_cut),
                        (bdt_events_sig["bdt_score"] >= bdt_cut)
                        & (bdt_events_sig["H2TXbb"] >= xbb_cut),
                        (bdt_events_others["bdt_score"] >= bdt_cut)
                        & (bdt_events_others["H2TXbb"] >= xbb_cut),
                        weight_toy,
                        bdt_events_sig["weight"] * kfactor_signal,
                        bdt_events_others["weight"],
                        # INVERTED stuff (w/o signal)
                        random_mass,
                        bdt_events_others[mass_var],
                        (bdt_events_data["bdt_score"] < bdt_fail)
                        & (bdt_events_data["H2TXbb"] < xbb_cut_fail),
                        (bdt_events_others["bdt_score"] < bdt_fail)
                        & (bdt_events_others["H2TXbb"] < xbb_cut_fail),
                        bdt_events_data["weight"],
                        bdt_events_others["weight"],
                        mass_window,
                    )

                soversb = nevents_sig_bdt_cut / np.sqrt(nevents_bkg_bdt_cut + nevents_sig_bdt_cut)

                # NOTE: here optimizing by soversb but can change the figure of merit...
                if nevents_sig_bdt_cut > 0.5 and nevents_bkg_bdt_cut >= 2:
                    cuts_xbb.append(xbb_cut)
                    cuts_bdt.append(bdt_cut)
                    figure_of_merits.append(soversb)

        print("choosing optimal cuts")
        # choose "optimal" cuts, check if they give the expected sensitivity
        optimal_bdt_cut = 0
        optimal_xbb_cut = 0
        if len(cuts_xbb) > 1:
            cuts_xbb = np.array(cuts_xbb)
            cuts_bdt = np.array(cuts_bdt)
            figure_of_merits = np.array(figure_of_merits)
            biggest = np.argmax(figure_of_merits)
            optimal_xbb_cut = cuts_xbb[biggest]
            optimal_bdt_cut = cuts_bdt[biggest]
            expected = expected_soverb_by_cut[optimal_xbb_cut][optimal_bdt_cut]
            print(
                f"{optimal_xbb_cut:.3f} {optimal_bdt_cut:.2f} {biggest} {figure_of_merits[biggest]:.2f} {(figure_of_merits[biggest]-expected):.2f} {expected:.2f}"
            )
            h_pull.fill(figure_of_merits[biggest] - expected)

        # for this toy the BDT and Xbb cut are
        print("Xbb ", optimal_xbb_cut, "BDT ", optimal_bdt_cut)

        # define regions with optimal cut
        selection_regions = {
            "pass_bin1": postprocessing.Region(
                cuts={
                    "H2TXbb": [optimal_xbb_cut, CUT_MAX_VAL],
                    "bdt_score": [optimal_bdt_cut, CUT_MAX_VAL],
                },
                label="Bin1",
            ),
            "fail": postprocessing.Region(
                cuts={
                    "H2TXbb": [-CUT_MAX_VAL, xbb_cut],
                },
                label="Fail",
            ),
        }

        print("replace w toy")
        # replace data distribution with toy!!
        bdt_events_for_templates = bdt_events_dict.copy()
        bdt_events_for_templates["data"][mass_var] = random_mass

        print("Get templates")
        templates = postprocessing.get_templates(
            bdt_events_dict,
            year=year,
            sig_keys=["hh4b"],
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

        # save toys!
        postprocessing.save_templates(templates, templ_dir / f"{year}_templates.pkl", fit_shape_var)

    # plot pull
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    hep.histplot(h_pull, ax=ax, label="Xbb and BDT cut optimization")
    ax.set_xlabel("Difference w.r.t expected " + r"S/$\sqrt{S+B}$")
    ax.set_title(r"Injected S, S $\times$ " + f"{kfactor_signal}, 2022EE")
    ax.legend(title=f"{ntoys} toys")
    fig.savefig(f"toytest_all_{args.method}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bdt-model-name",
        help="model name",
        type=str,
        default="24Apr20_legacy_fix",
    )
    parser.add_argument(
        "--bdt-config",
        default="24Apr20_legacy_fix",
        help="config name in case model name is different",
        type=str,
    )
    parser.add_argument(
        "--method", required=True, choices=["sideband", "abcd"], help="method to test"
    )
    parser.add_argument(
        "--ntoys",
        help="number of toys",
        type=int,
        default=1,
    )
    parser.add_argument("--tag", help="tag test", required=True, type=str)
    args = parser.parse_args()
    main(args)
