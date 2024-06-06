from __future__ import annotations

import argparse
import importlib

import hist
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b import postprocessing
from HH4b.hh_vars import bg_keys

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 12})
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.color"] = "#CCCCCC"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["figure.edgecolor"] = "none"

mass_axis = hist.axis.Regular(20, 50, 250, name="mass")
bdt_axis = hist.axis.Regular(100, 0, 1, name="bdt")
xbb_axis = hist.axis.Regular(100, 0, 1, name="xbb")
diff_axis = hist.axis.Regular(50, -2, 2, name="diff")
cut_axis = hist.axis.StrCategory([], name="cut", growth=True)

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

    # data_dir = "24Apr23LegacyLowerThresholds_v12_private_signal"
    data_dir = "24May24_v12_private_signal"
    input_dir = f"/ceph/cms/store/user/cmantill/bbbb/skimmer/{data_dir}"
    years = ["2022"]  # , "2022EE"] #, "2023", "2023BPix"]

    samples_run3 = {
        year: {
            "data": [f"{key}_Run" for key in ["JetMET"]],
            "ttbar": ["TTto4Q", "TTto2L2Nu", "TTtoLNu2Q"],
            "diboson": ["ZZ", "WW", "WZ"],
            "vjets": ["Wto2Q-3Jets_HT", "Zto2Q-4Jets_HT"],
            "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        }
        for year in years
    }

    mass_var = "H2PNetMass"

    bdt_events_per_year = {}
    for year in years:
        bdt_events_per_year[year] = {}
        events_dict = postprocessing.load_run3_samples(
            input_dir=input_dir,
            year=year,
            legacy=True,
            samples_run3=samples_run3,
            reorder_txbb=True,
            txbb=f"bbFatJetPNetTXbb{legacy_label}",
        )
        bdt_events = get_dataframe(events_dict, year, args.bdt_model_name, args.bdt_config)
        columns = ["H2TXbb", "bdt_score", mass_var]
        for key in events_dict:
            bdt_events_per_year[year][key] = bdt_events[key][columns]

    print(bdt_events_per_year)
    # concatenate
    bdt_events_dict, _ = postprocessing.combine_run3_samples(
        bdt_events_per_year,
        ["data", "ttbar", "vjets", "diboson", "hh4b"],
        years_run3=years,
    )

    # get other backgrounds
    bdt_events_dict["others"] = pd.concat([bdt_events_dict[key] for key in bg_keys if key != "qcd"])

    ntoys = args.ntoys
    print(f"Number of toys {ntoys}")

    # mass_window = [110, 140]
    mass_window = [100, 150]

    all_bdt_cuts = 0.01 * np.arange(80, 100)
    bdt_cuts = all_bdt_cuts
    # bdt_cuts = [0.94]

    all_xbb_cuts = 0.01 * np.arange(90, 100)
    xbb_cuts = all_xbb_cuts
    # xbb_cuts = [0.98]

    # define fail region for ABCD
    bdt_fail = 0.03

    # fixed signal k-factor
    # kfactor_signal = 80
    # kfactor_signal = 40
    kfactor_signal = 20
    print(f"Fixed factor by which to scale signal: {kfactor_signal}")

    # get "truth" optimization and sensitivity per cut
    #########################################
    # Evaluate expected sensitivity from data
    #########################################
    xbb_cuts_data = []
    bdt_cuts_data = []
    figure_of_merits = []

    for xbb_cut in xbb_cuts:

        bdt_events_dict_xbb_cut = {}
        for key in bdt_events_dict:
            bdt_events_dict_xbb_cut[key] = bdt_events_dict[key][
                bdt_events_dict[key]["H2TXbb"] > xbb_cut
            ]

        # print(f"\n Xbb cut:{xbb_cut}")
        bdt_events_data = bdt_events_dict_xbb_cut["data"]
        bdt_events_sig = bdt_events_dict_xbb_cut["hh4b"]
        bdt_events_others = bdt_events_dict_xbb_cut["others"]
        bdt_events_data_invertedXbb = bdt_events_dict["data"][
            bdt_events_dict["data"]["H2TXbb"] < xbb_cut
        ]
        bdt_events_others_invertedXbb = bdt_events_dict["others"][
            bdt_events_dict["others"]["H2TXbb"] < xbb_cut
        ]

        for bdt_cut in bdt_cuts:
            nevents_sig, nevents_bkg = abcd_fom(
                mass_data=bdt_events_data[mass_var],
                mass_sig=bdt_events_sig[mass_var],
                mass_others=bdt_events_others[mass_var],
                cut_data=(bdt_events_data["bdt_score"] >= bdt_cut),
                cut_sig=(bdt_events_sig["bdt_score"] >= bdt_cut),
                cut_others=(bdt_events_others["bdt_score"] >= bdt_cut),
                weight_data=bdt_events_data["weight"],
                weight_signal=bdt_events_sig["weight"] * kfactor_signal,
                weight_others=bdt_events_others["weight"],
                # INVERTED stuff
                mass_inv_data=bdt_events_data_invertedXbb[mass_var],
                mass_inv_others=bdt_events_others_invertedXbb[mass_var],
                invcut_data=(bdt_events_data_invertedXbb["bdt_score"] < bdt_fail),  # fail region
                invcut_others=(bdt_events_others_invertedXbb["bdt_score"] < bdt_fail),
                weight_inv_data=bdt_events_data_invertedXbb["weight"],
                weight_inv_others=bdt_events_others_invertedXbb["weight"],
                mass_window=mass_window,
            )
            soversb = nevents_sig / np.sqrt(nevents_bkg + nevents_sig)

            nevents_sideband = np.sum(
                (
                    (bdt_events_data[mass_var] < mass_window[0])
                    | (bdt_events_data[mass_var] > mass_window[1])
                )
                & (bdt_events_data["bdt_score"] >= bdt_cut)
            )

            print("nevents_sideband ", nevents_sideband)
            if nevents_sig > 0.5 and nevents_bkg >= 2 and nevents_sideband >= 12:
                bdt_cuts_data.append(bdt_cut)
                xbb_cuts_data.append(xbb_cut)
                figure_of_merits.append(soversb)

    sensitivity_data = 0
    if len(bdt_cuts_data) > 0:
        bdt_cuts_data = np.array(bdt_cuts_data)
        xbb_cuts_data = np.array(xbb_cuts_data)
        figure_of_merits = np.array(figure_of_merits)
        biggest = np.argmax(figure_of_merits)

        optimal_xbb_cut_data = xbb_cuts_data[biggest]
        optimal_bdt_cut_data = bdt_cuts_data[biggest]
        sensitivity_data = figure_of_merits[biggest]

        print(
            f"From Data: Xbb:{optimal_xbb_cut_data:.3f} BDT:{optimal_bdt_cut_data:.2f} S/(S+B):{sensitivity_data:.2f}"
        )

    ###################
    # TOYS
    ###################

    h_pull = hist.Hist(diff_axis)
    h_pull_s = hist.Hist(diff_axis)
    h_diff = hist.Hist(diff_axis)
    h_diff_s = hist.Hist(diff_axis)

    pull_array = []
    pull_s_array = []
    diff_array = []
    diff_s_array = []

    # create toy from data mass distribution
    h_mass = hist.Hist(mass_axis)
    h_mass.fill(bdt_events_dict["data"][mass_var])
    h_xbb = hist.Hist(xbb_axis)
    h_xbb.fill(bdt_events_dict["data"]["H2TXbb"])
    h_bdt = hist.Hist(bdt_axis)
    h_bdt.fill(bdt_events_dict["data"]["bdt_score"])

    bdt_events_data = bdt_events_dict["data"]
    bdt_events_sig = bdt_events_dict["hh4b"]
    bdt_events_others = bdt_events_dict["others"]

    print("nevents data ", bdt_events_data[mass_var].shape[0])
    print("nevents sig ", np.sum(bdt_events_sig["weight"] * kfactor_signal))

    h_mass_sig = hist.Hist(mass_axis)
    h_mass_sig.fill(bdt_events_sig[mass_var])
    h_xbb_sig = hist.Hist(xbb_axis)
    h_xbb_sig.fill(bdt_events_sig["H2TXbb"])

    print("Xbb BDT Index-BDT S/(S+B) Difference Expected")
    for itoy in range(ntoys):
        # templ_dir = Path(f"templates/toys_{args.tag}/xbb_cut_{xbb_cut:.3f}/toy_{itoy}")
        # (templ_dir / year).mkdir(parents=True, exist_ok=True)
        # (templ_dir / "cutflows" / year).mkdir(parents=True, exist_ok=True)

        random_xbb = get_toy_from_hist(h_xbb)
        random_mass = get_toy_from_hist(h_mass)
        random_bdt = get_toy_from_hist(h_bdt)

        # print("xbb ", bdt_events_dict["data"]["H2TXbb"])
        # print("random ", random_xbb)
        # print("bdt ", bdt_events_dict["data"]["bdt_score"])
        # print("random ", random_bdt)
        # print("mass ", bdt_events_dict["data"][mass_var])
        # print("random ", random_mass)

        # build toy = data + injected signal
        mass_toy = np.concatenate([random_mass, bdt_events_sig[mass_var]])
        xbb_toy = np.concatenate([random_xbb, bdt_events_sig["H2TXbb"]])
        bdt_toy = np.concatenate([random_bdt, bdt_events_sig["bdt_score"]])

        # sum weights together, but scale weight of signal
        weight_toy = np.concatenate(
            [bdt_events_data["weight"], bdt_events_sig["weight"] * kfactor_signal]
        )

        # perform optimization for toy
        bdt_cuts_toys = []
        xbb_cuts_toys = []
        figure_of_merit_toys = []
        figure_of_merit_abcd_toys = []

        signal_toys = []
        background_toys = []
        truesignal_toys = []
        truebackground_toys = []

        sbleft_toys = []
        sbright_toys = []

        signal_abcd_toys = []

        for xbb_cut in xbb_cuts:
            for bdt_cut in bdt_cuts:

                # number of events in data in signal mass window, bdt cut and xbb cut
                cut_mass_toy = (mass_toy >= mass_window[0]) & (mass_toy <= mass_window[1])
                nevents_toy_bdt_cut = np.sum(
                    weight_toy[(cut_mass_toy) & (bdt_toy >= bdt_cut) & (xbb_toy >= xbb_cut)]
                )

                # true number of signal events
                cut_mass_sig = (bdt_events_sig[mass_var] >= mass_window[0]) & (
                    bdt_events_sig[mass_var] <= mass_window[1]
                )
                nevents_sig_true = np.sum(
                    (bdt_events_sig["weight"] * kfactor_signal)[
                        (bdt_events_sig["bdt_score"] >= bdt_cut)
                        & (bdt_events_sig["H2TXbb"] >= xbb_cut)
                        & cut_mass_sig
                    ]
                )

                # true number of bkg events
                cut_mass_data = (random_mass >= mass_window[0]) & (random_mass <= mass_window[1])
                nevents_bkg_true = np.sum(
                    bdt_events_data["weight"][
                        (random_bdt >= bdt_cut) & (random_xbb >= xbb_cut) & cut_mass_data
                    ]
                )

                # signal events in sideband
                nevents_sig_left = np.sum(
                    (bdt_events_sig["weight"] * kfactor_signal)[
                        (bdt_events_sig["bdt_score"] >= bdt_cut)
                        & (bdt_events_sig["H2TXbb"] >= xbb_cut)
                        & (bdt_events_sig[mass_var] < mass_window[0])
                    ]
                )
                nevents_sig_right = np.sum(
                    (bdt_events_sig["weight"] * kfactor_signal)[
                        (bdt_events_sig["bdt_score"] >= bdt_cut)
                        & (bdt_events_sig["H2TXbb"] >= xbb_cut)
                        & (bdt_events_sig[mass_var] > mass_window[1])
                    ]
                )
                nevents_data_left = np.sum(
                    [
                        (random_bdt >= bdt_cut)
                        & (random_xbb >= xbb_cut)
                        & (random_mass < mass_window[0])
                    ]
                )
                nevents_data_right = np.sum(
                    [
                        (random_bdt >= bdt_cut)
                        & (random_xbb >= xbb_cut)
                        & (random_mass > mass_window[1])
                    ]
                )
                # print(xbb_cut,bdt_cut)
                # print("nsig left right ",nevents_sig_left,nevents_sig_right)
                # print("ndata left right ",nevents_data_left, nevents_data_right)

                nevents_sig_bdt_cut_sb, nevents_bkg_bdt_cut_sb = sideband_fom(
                    mass_data=mass_toy,
                    mass_sig=bdt_events_sig[mass_var],
                    cut_data=(bdt_toy >= bdt_cut) & (xbb_toy >= xbb_cut),
                    cut_sig=(bdt_events_sig["bdt_score"] >= bdt_cut)
                    & (bdt_events_sig["H2TXbb"] >= xbb_cut),
                    weight_data=weight_toy,
                    weight_signal=bdt_events_sig["weight"] * kfactor_signal,
                    mass_window=mass_window,
                )

                nevents_sig_bdt_cut, nevents_bkg_bdt_cut = abcd_fom(
                    mass_data=mass_toy,
                    mass_sig=bdt_events_sig[mass_var],
                    mass_others=bdt_events_others[mass_var],
                    cut_data=(bdt_toy >= bdt_cut) & (xbb_toy >= xbb_cut),
                    cut_sig=(bdt_events_sig["bdt_score"] >= bdt_cut)
                    & (bdt_events_sig["H2TXbb"] >= xbb_cut),
                    cut_others=(bdt_events_others["bdt_score"] >= bdt_cut)
                    & (bdt_events_others["H2TXbb"] >= xbb_cut),
                    weight_data=weight_toy,
                    weight_signal=bdt_events_sig["weight"] * kfactor_signal,
                    weight_others=bdt_events_others["weight"],
                    # definitions of mass values, BDT cut and weights for inverted Xbb regions (C,D)
                    mass_inv_data=mass_toy,
                    mass_inv_others=bdt_events_others[mass_var],
                    invcut_data=(xbb_toy < xbb_cut) & (bdt_toy < bdt_fail),
                    invcut_others=(bdt_events_others["H2TXbb"] < xbb_cut)
                    & (bdt_events_others["bdt_score"] < bdt_fail),
                    weight_inv_data=weight_toy,
                    weight_inv_others=bdt_events_others["weight"],
                    mass_window=mass_window,
                )

                # B_toy
                b_from_toy = nevents_bkg_bdt_cut
                # S_mc
                # s_from_mc = nevents_sig_bdt_cut
                # S_toy
                # abcd already takes into account others
                # s_from_toy = nevents_toy_bdt_cut - nevents_bkg_bdt_cut - nevents_others_bdt_cut
                s_from_toy = nevents_toy_bdt_cut - nevents_bkg_bdt_cut

                # sig = s_from_mc
                sig = s_from_toy

                soversb = sig / np.sqrt(sig + b_from_toy)
                # print(xbb_cut, bdt_cut, soversb, " sfromtoy ", s_from_toy, " sfrommc ",s_from_mc)
                soversb_abcd = soversb

                if args.method == "sideband":
                    # print("abcd ",soversb)
                    soversb = (nevents_toy_bdt_cut - nevents_sig_bdt_cut_sb) / np.sqrt(
                        nevents_sig_bdt_cut_sb + nevents_bkg_bdt_cut_sb
                    )
                    sig = nevents_toy_bdt_cut - nevents_sig_bdt_cut_sb
                    # print("sideband ", soversb)

                nevents_sideband = np.sum(
                    weight_toy[
                        ((mass_toy < mass_window[0]) | (mass_toy > mass_window[1]))
                        & (bdt_toy >= bdt_cut)
                        & (xbb_toy >= xbb_cut)
                    ]
                )
                # print("n sideband ",nevents_sideband)

                # NOTE: here optimizing by soversb but can change the figure of merit...
                if (
                    nevents_sig_bdt_cut > 0.5 and nevents_bkg_bdt_cut >= 2
                ):  # and nevents_sideband >= 12:
                    bdt_cuts_toys.append(bdt_cut)
                    xbb_cuts_toys.append(xbb_cut)

                    figure_of_merit_toys.append(soversb)
                    figure_of_merit_abcd_toys.append(soversb_abcd)

                    signal_toys.append(sig)
                    background_toys.append(b_from_toy)
                    truebackground_toys.append(nevents_bkg_true)
                    truesignal_toys.append(nevents_sig_true)

                    sbleft_toys.append(nevents_sig_left / nevents_data_left)
                    sbright_toys.append(nevents_sig_right / nevents_data_right)
                    signal_abcd_toys.append(nevents_toy_bdt_cut - nevents_bkg_bdt_cut)

        # choose "optimal" bdt and xbb cut, check if it gives the expected sensitivity
        optimal_bdt_cut = 0
        optimal_xbb_cut = 0
        if len(bdt_cuts_toys) > 0:
            bdt_cuts_toys = np.array(bdt_cuts_toys)
            xbb_cuts_toys = np.array(xbb_cuts_toys)
            figure_of_merit_toys = np.array(figure_of_merit_toys)
            biggest = np.argmax(figure_of_merit_toys)
            truesignal_toys = np.array(truesignal_toys)

            optimal_bdt_cut = bdt_cuts_toys[biggest]
            optimal_xbb_cut = xbb_cuts_toys[biggest]

            # compare_to = expected_soverb_by_xbb_cut_by_bdt_cut[optimal_xbb_cut][optimal_bdt_cut]
            # compare_to_s = expected_s_by_xbb_cut_by_bdt_cut[optimal_xbb_cut][optimal_bdt_cut]
            compare_to_s = truesignal_toys[biggest]
            # compare_to = truesignal_toys[biggest]/np.sqrt(truesignal_toys[biggest]+background_toys[biggest])
            compare_to = truesignal_toys[biggest] / np.sqrt(
                truesignal_toys[biggest] + truebackground_toys[biggest]
            )

            print(
                f" Optimal {optimal_xbb_cut:.3f} {optimal_bdt_cut:.2f}, FOM: {figure_of_merit_toys[biggest]:.2f} \n"
                + f" S from-Method: {signal_toys[biggest]:.3f}, S from-ABCD: {signal_abcd_toys[biggest]:.3f}, S True: {truesignal_toys[biggest]:.3f} \n"
                + f" B from-Method: {background_toys[biggest]:.2f}, B True: {truebackground_toys[biggest]:.3f} \n"
                # +f" S/B left: {sbleft_toys[biggest]} right sideband: {sbright_toys[biggest]}"
                # Difference: {(figure_of_merit_toys[biggest]-compare_to):.2f},
                + f" S/S+B: {figure_of_merit_abcd_toys[biggest]:.2f} VS {compare_to:.2f} \n"
                + f" S: {signal_abcd_toys[biggest]:.2f} VS {compare_to_s:.2f}"
            )

            pull = (figure_of_merit_toys[biggest] - compare_to) / compare_to
            diff = figure_of_merit_toys[biggest] - compare_to

            # pull_s = (signal_toys[biggest] - truesignal_toys[biggest]) / truesignal_toys[biggest]
            # diff_s = signal_toys[biggest] - truesignal_toys[biggest]

            pull_s = (signal_abcd_toys[biggest] - truesignal_toys[biggest]) / truesignal_toys[
                biggest
            ]
            diff_s = signal_abcd_toys[biggest] - truesignal_toys[biggest]

            # print("t ",truesignal_toys[biggest])
            # print(f" Signal-From-Toy: {signal_toys[biggest]}, Signa-From-MC: {truesignal_toys[biggest]}")

            pull_array.append(pull)
            pull_s_array.append(pull_s)
            diff_array.append(diff)
            diff_s_array.append(diff_s)

            h_pull.fill(pull)
            h_diff.fill(diff)

            h_pull_s.fill(pull_s)
            h_diff_s.fill(diff_s)

            # for this toy the BDT and Xbb cut are
            print(f"{itoy} Optimal cuts: Xbb ", optimal_xbb_cut, "BDT ", optimal_bdt_cut)

    # plot diff
    pull_array = np.array(pull_array)
    pull_s_array = np.array(pull_s_array)
    diff_array = np.array(diff_array)
    diff_s_array = np.array(diff_s_array)

    gaus_fit = {
        "pull": [np.mean(pull_array), np.std(pull_array)],
        "pull_s": [np.mean(pull_s_array), np.std(pull_s_array)],
        "diff": [np.mean(diff_array), np.std(diff_array)],
        "diff_s": [np.mean(diff_s_array), np.std(diff_s_array)],
    }

    def plot_h(h_hist, xlabel, plot_name, xlim, gaus_label):
        # optimal = expected_soverb_by_xbb_cut_by_bdt_cut[optimal_xbb_cut_data][optimal_bdt_cut]

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        mu, sigma = gaus_fit[gaus_label]
        hep.histplot(
            h_hist,
            ax=ax,
            label=f"Mean: {mu:.2f}",
            # "#r"Expected S/$\sqrt{S+B}$ from optimal cut in data: "
            # + f" {optimal:.1f}, Mean: {mu:.2f}",
        )
        ax.set_xlabel(xlabel)
        plot_title = r"Injected S, S $\times$ " + f"{kfactor_signal}, 2022EE"
        legend_title = f"{ntoys} toys"
        ax.set_title(plot_title)
        ax.set_ylabel("Density")
        ax.set_xlim(xlim)
        ax.legend(title=legend_title)
        fig.savefig(f"templates/toybyxbb_{args.method}_{args.tag}_{plot_name}.png")

    plot_h(
        h_diff,
        r"(S$_{t}/\sqrt{S_{t}+B_{t}}$ - S/$\sqrt{S+B}$)",
        "soverb_diff",
        [-2, 2],
        "diff",
    )
    plot_h(
        h_diff_s,
        r"S$_{t}$ - S (ABCD)",
        "s_diff",
        [-2, 2],
        "diff_s",
    )

    plot_h(
        h_pull,
        r"(S$_{t}/\sqrt{S_{t}+B_{t}}$ - S/$\sqrt{S+B}$) / S/$\sqrt{S+B}$",
        "soverb",
        [-1.5, 1.5],
        "pull",
    )
    plot_h(
        h_pull_s,
        r"(S$_{t}$ - S)/S",
        "s",
        [-1.5, 1.5],
        "pull_s",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bdt-model-name",
        help="model name",
        type=str,
        default="24Apr21_legacy_vbf_vars",
    )
    parser.add_argument(
        "--bdt-config",
        default="24Apr21_legacy_vbf_vars",
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
