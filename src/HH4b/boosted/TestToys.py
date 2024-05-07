from __future__ import annotations

import argparse
import importlib

import hist
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import xgboost as xgb

from HH4b import utils
from HH4b.postprocessing import load_columns_legacy

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


def get_toy_from_hist(h_hist):
    """
    Get values drawn from histogram
    """

    h, bins = h_hist.to_numpy()
    integral = int(np.sum(h_hist.values()))

    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(h)
    cdf = cdf / cdf[-1]
    values = np.random.rand(integral) # noqa: NPY002
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    return random_from_cdf


def get_dataframe(events_dict, year, bdt_model_name, bdt_config):
    bdt_model = xgb.XGBClassifier()
    bdt_model.load_model(fname=f"../boosted/bdt_trainings_run3/{bdt_model_name}/trained_bdt.model")
    make_bdt_dataframe = importlib.import_module(
        f".{bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
    )

    HLTs = {
        "2022EE": [
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        ],
    }

    legacy_label = "Legacy"
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
            np.array([events[trigger][0] for trigger in HLTs[year] if trigger in events]),
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

def abcd_fom(mass_data, mass_sig, mass_others,
             cut_data, cut_sig, cut_others,
             weight_data, weight_signal, weight_others,
             # definitions of mass values, BDT cut and weights for inverted Xbb regions (C,D)
             mass_inv_data, mass_inv_others,
             invcut_data, invcut_others,
             weight_inv_data, weight_inv_others,
             mass_window):
    # get A,B,C,D
    dicts = {"data": [], "others": []}

    cut_mass = (mass_data >= mass_window[0]) & (mass_data <= mass_window[1])
    cut_mass_invregion = (mass_inv_data >= mass_window[0]) & (mass_inv_data <= mass_window[1])
    dicts["data"] = [
        0, # A
        np.sum(weight_data[~cut_mass & cut_data]), #B,
        np.sum(weight_inv_data[cut_mass_invregion & invcut_data]), #C
        np.sum(weight_inv_data[~cut_mass_invregion & invcut_data]), #D
    ]

    if mass_others is not None:
        cut_mass = (mass_others >= mass_window[0]) & (mass_others <= mass_window[1])
        cut_mass_invregion = (mass_inv_others >= mass_window[0]) & (mass_inv_others <= mass_window[1])

        dicts["others"] = [
            np.sum(weight_others[cut_mass & cut_others]), #A
       	    np.sum(weight_others[~cut_mass & cut_others]), #B
            np.sum(weight_inv_others[cut_mass_invregion & invcut_others]), #C
            np.sum(weight_inv_others[~cut_mass_invregion & invcut_others]), #D
        ]
        # subtract other backgrounds
        dmt = np.array(dicts["data"]) - np.array(dicts["others"])
    else:
        dmt = np.array(dicts["data"])

    cut_mass = (mass_sig >= mass_window[0]) & (mass_sig <= mass_window[1])
    nevents_sig = np.sum(weight_signal[cut_sig & cut_mass])

    # C/D * B
    bqcd = dmt[2] * dmt[1] / dmt[3]
    if mass_others is not None:
        nevents_bkg = bqcd + dicts["others"][0]
    else:
        nevents_bkg = bqcd
    return nevents_sig, nevents_bkg


def main(args):

    # FIXME: Load samples for all years..
    samples_run3 = {
        "2022EE": {
            "data": ["JetMET_Run"],
            "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
        },
    }
    data_dir = "24Apr23LegacyLowerThresholds_v12_private_signal"
    input_dir = f"/eos/uscms/store/user/cmantill/bbbb/skimmer/{data_dir}"
    year = "2022EE"

    events_dict = utils.load_samples(
        input_dir,
        samples_run3[year],
        year,
        filters=None,
        columns=utils.format_columns(
            load_columns_legacy + [("AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35", 1)]
        ),
        reorder_txbb=True,
        txbb="bbFatJetPNetTXbbLegacy",
        variations=False,
    )
    mass_var = "H2PNetMass"
    bdt_events_dict = get_dataframe(events_dict, year, args.bdt_model_name, args.bdt_config)
    ntoys = 200

    mass_window = [110, 140]
    # bdt_cuts = [0.8, 0.85, 0.9, 0.95]
    all_bdt_cuts = 0.01 * np.arange(80, 100)
    bdt_cuts = all_bdt_cuts
    # xbb_cuts = [0.8]
    xbb_cuts = [0.8, 0.95]

    bdt_fail = 0.6
    
    h_pull = hist.Hist(diff_axis, cut_axis)

    # fixed signal k-factor
    kfactor_signal = 80
    print(f"Fixed factor by which to scale signal: {kfactor_signal}")
    
    expected_by_xbb = {}
    for xbb_cut in xbb_cuts:
        print(f"\n Xbb cut:{xbb_cut}")
        bdt_events_data = bdt_events_dict["data"][bdt_events_dict["data"]["H2TXbb"] > xbb_cut]
        bdt_events_sig = bdt_events_dict["hh4b"][bdt_events_dict["hh4b"]["H2TXbb"] > xbb_cut]
        bdt_events_data_invertedXbb = bdt_events_dict["data"][bdt_events_dict["data"]["H2TXbb"] < 0.8]

        # get expected sensitivity for each bdt cut
        expected_soverb_by_bdt_cut = {}
        cuts = []
        figure_of_merits = []
        for bdt_cut in bdt_cuts:
            if args.method == "sideband":
                nevents_sig, nevents_bkg = sideband_fom(
                    bdt_events_data[mass_var],
                    bdt_events_sig[mass_var],
                    (bdt_events_data["bdt_score"] >= bdt_cut),
                    (bdt_events_sig["bdt_score"] >= bdt_cut),
                    bdt_events_data["weight"],
                    bdt_events_sig["weight"],
                    mass_window,
                )
            else:
                nevents_sig, nevents_bkg = abcd_fom(
                    mass_data = bdt_events_data[mass_var],
		    mass_sig = bdt_events_sig[mass_var],
                    mass_others = None, # no contribution from other backgrounds
                    cut_data = (bdt_events_data["bdt_score"] >= bdt_cut),
                    cut_sig = (bdt_events_sig["bdt_score"] >= bdt_cut),
                    cut_others = None,
                    weight_data = bdt_events_data["weight"],
                    weight_signal = bdt_events_sig["weight"],
                    weight_others = None,
                    # INVERTED stuff
                    mass_inv_data = bdt_events_data_invertedXbb[mass_var],
                    mass_inv_others = None,
                    invcut_data = (bdt_events_data_invertedXbb["bdt_score"] < bdt_fail), # fail region
                    invcut_others = None,
                    weight_inv_data = bdt_events_data_invertedXbb["weight"],
                    weight_inv_others = None,
                    mass_window = mass_window,
                )

            nevents_sig_scaled = nevents_sig * kfactor_signal
            soversb = nevents_sig_scaled / np.sqrt(nevents_bkg + nevents_sig_scaled)
            expected_soverb_by_bdt_cut[bdt_cut] = soversb
            
            if nevents_sig > 0.5 and nevents_bkg >= 2:
                cuts.append(bdt_cut)
                figure_of_merits.append(soversb)

        sensitivity_data = 0
        if len(cuts) > 0:
            cuts = np.array(cuts)
            figure_of_merits = np.array(figure_of_merits)
            biggest = np.argmax(figure_of_merits)
            optimal_bdt_cut_data = cuts[biggest]
            sensitivity_data = figure_of_merits[biggest]
            print(
                f"From Data: Xbb:{xbb_cut:.3f} BDT:{optimal_bdt_cut_data:.2f} S/(S+B):{sensitivity_data:.2f}"
            )
        expected_by_xbb[xbb_cut] = sensitivity_data
        
        print(f"Expected sensitivity by BDT cut {expected_soverb_by_bdt_cut}")
            
        # create toy from data mass distribution (with xbb cut)
        h_mass = hist.Hist(mass_axis)
        h_mass.fill(bdt_events_data[mass_var])

        h_mass_invertedXbb = hist.Hist(mass_axis)
        h_mass_invertedXbb.fill(bdt_events_data_invertedXbb[mass_var])
        
        print("Xbb BDT Index-BDT S/(S+B) Difference Expected")
        expected_sensitivities = []
        for i in range(ntoys): # noqa: B007
            random_mass = get_toy_from_hist(h_mass)

            # build toy = data + injected signal
            mass_toy = np.concatenate([random_mass, bdt_events_sig[mass_var]])
            bdt_toy = np.concatenate([bdt_events_data["bdt_score"], bdt_events_sig["bdt_score"]])
            # sum weights together, but scale weight of signal
            weight_toy = np.concatenate(
                [bdt_events_data["weight"], bdt_events_sig["weight"] * kfactor_signal]
            )

            # same thing but for mass distribution with inverted Xbb cut
            random_mass_invertedXbb = get_toy_from_hist(h_mass_invertedXbb)
            # do not concatenate toys
            mass_toy_invertedXbb = np.concatenate([random_mass_invertedXbb])
            bdt_toy_invertedXbb = np.concatenate([bdt_events_data_invertedXbb["bdt_score"]])
            weight_toy_invertedXbb = np.concatenate([bdt_events_data_invertedXbb["weight"]])
            
            max_soversb = 0
            cuts = []
            figure_of_merits = []
            for bdt_cut in bdt_cuts:
                if args.method == "sideband":
                    nevents_sig_bdt_cut, nevents_bkg_bdt_cut = sideband_fom(
                        mass_toy,
                        bdt_events_sig[mass_var],
                        (bdt_toy >= bdt_cut),
                        (bdt_events_sig["bdt_score"] >= bdt_cut),
                        weight_toy,
                        bdt_events_sig["weight"] * kfactor_signal,
                        mass_window,
                    )
                else:
                    nevents_sig_bdt_cut, nevents_bkg_bdt_cut = abcd_fom(
                        mass_toy,
                        bdt_events_sig[mass_var],
                        None,
                        (bdt_toy >= bdt_cut),
                        (bdt_events_sig["bdt_score"] >= bdt_cut),
                        None,
                        weight_toy,
                        bdt_events_sig["weight"] * kfactor_signal,
                        None,
                        # INVERTED stuff
                        mass_toy_invertedXbb,
                        None,
                        (bdt_toy_invertedXbb < bdt_fail),
                        None,
                        weight_toy_invertedXbb,
                        None,
                        mass_window,
                    )

                soversb = nevents_sig_bdt_cut / np.sqrt(nevents_bkg_bdt_cut + nevents_sig_bdt_cut)

                # NOTE: here optimizing by soversb but can change the figure of merit...
                if (
                    nevents_sig_bdt_cut > 0.5
                    and nevents_data_bdt_cut >= 2
                    and soversb > max_soversb
                ):
                    cuts.append(bdt_cut)
                    figure_of_merits.append(soversb)

            # choose "optimal" bdt cut, check if it gives the expected sensitivity
            if len(cuts) > 0:
                cuts = np.array(cuts)
                figure_of_merits = np.array(figure_of_merits)
                biggest = np.argmax(figure_of_merits)
                optimal_bdt_cut = cuts[biggest]
                print(
                    f"{xbb_cut:.3f} {optimal_bdt_cut:.2f} {biggest} {figure_of_merits[biggest]:.2f} {(figure_of_merits[biggest]-expected_soverb_by_bdt_cut[optimal_bdt_cut]):.2f} {expected_soverb_by_bdt_cut[optimal_bdt_cut]:.2f}"
                )
                h_pull.fill(
                    figure_of_merits[biggest] - expected_soverb_by_bdt_cut[optimal_bdt_cut],
                    cut=str(xbb_cut),
                )

    # plot pull
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for xbb_cut in xbb_cuts:
        hep.histplot(
            h_pull[{"cut": f"{xbb_cut}"}],
            ax=ax,
            label=f"Xbb > {xbb_cut}, Expected: {expected_by_xbb[xbb_cut]:.2f}",
        )
    ax.set_xlabel("Difference w.r.t expected " + r"S/$\sqrt{S+B}$")
    ax.set_title(r"Injected S, S $\times$ "+f"{kfactor_signal}, 2022EE")
    ax.legend(title=f"{ntoys} toys")
    fig.savefig(f"toytest_{args.method}.png")


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
        default	= "24Apr20_legacy_fix",
        help="config name in case model name is different",
        type=str,
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["sideband", "abcd"],
        help="method to test"
    )
    args = parser.parse_args()
    main(args)
