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

    # xbb_cuts = [0.8, 0.95, 0.975]
    xbb_cuts = [0.95, 0.975]

    # define fail region for ABCD
    bdt_fail = 0.03

    h_pull = hist.Hist(diff_axis, cut_axis)
    h_pull_s = hist.Hist(diff_axis, cut_axis)

    h_diff = hist.Hist(diff_axis, cut_axis)
    h_diff_s = hist.Hist(diff_axis, cut_axis)

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

    expected_by_xbb = {}
    gaus_fit = {}
    for xbb_cut in xbb_cuts:
        bdt_events_dict_xbb_cut = {}
        for key in bdt_events_dict:
            bdt_events_dict_xbb_cut[key] = bdt_events_dict[key][
                bdt_events_dict[key]["H2TXbb"] > xbb_cut
            ]

        print(f"\n Xbb cut:{xbb_cut}")
        bdt_events_data = bdt_events_dict_xbb_cut["data"]
        bdt_events_sig = bdt_events_dict_xbb_cut["hh4b"]
        bdt_events_others = bdt_events_dict_xbb_cut["others"]
        bdt_events_data_invertedXbb = bdt_events_dict["data"][
            bdt_events_dict["data"]["H2TXbb"] < xbb_cut
        ]
        bdt_events_others_invertedXbb = bdt_events_dict["others"][
            bdt_events_dict["others"]["H2TXbb"] < xbb_cut
        ]

        ################################################
        # Evaluate expected sensitivity for each BDT cut
        ################################################
        expected_soverb_by_bdt_cut = {}
        expected_s_by_bdt_cut = {}

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
                    mass_data=bdt_events_data[mass_var],
                    mass_sig=bdt_events_sig[mass_var],
                    mass_others=bdt_events_others[mass_var],
                    cut_data=(bdt_events_data["bdt_score"] >= bdt_cut),
                    cut_sig=(bdt_events_sig["bdt_score"] >= bdt_cut),
                    cut_others=(bdt_events_others["bdt_score"] >= bdt_cut),
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
            expected_soverb_by_bdt_cut[bdt_cut] = soversb
            expected_s_by_bdt_cut[bdt_cut] = nevents_sig_scaled

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

        ###################
        # TOYS
        ###################

        # create toy from data mass distribution (with xbb cut)
        h_mass = hist.Hist(mass_axis)
        h_mass.fill(bdt_events_data[mass_var])

        h_mass_invertedXbb = hist.Hist(mass_axis)
        h_mass_invertedXbb.fill(bdt_events_data_invertedXbb[mass_var])

        print("Xbb BDT Index-BDT S/(S+B) Difference Expected")
        for itoy in range(ntoys):
            templ_dir = Path(f"templates/toys_{args.tag}/xbb_cut_{xbb_cut:.3f}/toy_{itoy}")
            (templ_dir / year).mkdir(parents=True, exist_ok=True)
            (templ_dir / "cutflows" / year).mkdir(parents=True, exist_ok=True)

            random_mass = get_toy_from_hist(h_mass)

            # build toy = data + injected signal
            mass_toy = np.concatenate([random_mass, bdt_events_sig[mass_var]])
            xbb_toy = np.concatenate([bdt_events_data["H2TXbb"], bdt_events_sig["H2TXbb"]])
            bdt_toy = np.concatenate([bdt_events_data["bdt_score"], bdt_events_sig["bdt_score"]])
            # sum weights together, but scale weight of signal
            weight_toy = np.concatenate(
                [bdt_events_data["weight"], bdt_events_sig["weight"] * kfactor_signal]
            )

            # same thing but for mass distribution with inverted Xbb cut
            random_mass_invertedXbb = get_toy_from_hist(h_mass_invertedXbb)

            cuts = []
            figure_of_merit_toys = []
            signal_toys = []
            for bdt_cut in bdt_cuts:

                # number of events in data in signal mass window
                cut_mass_toy = (mass_toy >= mass_window[0]) & (mass_toy <= mass_window[1])
                nevents_toy_bdt_cut = np.sum(weight_toy[cut_mass_toy & (bdt_toy >= bdt_cut)])

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
                        bdt_events_others[mass_var],
                        (bdt_toy >= bdt_cut),
                        (bdt_events_sig["bdt_score"] >= bdt_cut),
                        (bdt_events_others["bdt_score"] >= bdt_cut),
                        weight_toy,
                        bdt_events_sig["weight"] * kfactor_signal,
                        bdt_events_others["weight"],
                        # INVERTED stuff (w/o signal)
                        random_mass_invertedXbb,
                        bdt_events_others_invertedXbb[mass_var],
                        (bdt_events_data_invertedXbb["bdt_score"] < bdt_fail),
                        (bdt_events_others_invertedXbb["bdt_score"] < bdt_fail),
                        bdt_events_data_invertedXbb["weight"],
                        bdt_events_others_invertedXbb["weight"],
                        mass_window,
                    )

                # B_toy
                b_from_toy = nevents_bkg_bdt_cut
                # S_mc
                # s_from_mc = nevents_sig_bdt_cut
                # S_toy
                s_from_toy = nevents_toy_bdt_cut - nevents_bkg_bdt_cut

                # soversb = s_from_mc / np.sqrt(b_from_toy + s_from_mc)
                soversb = s_from_toy / np.sqrt(s_from_toy + b_from_toy)

                # NOTE: here optimizing by soversb but can change the figure of merit...
                if nevents_sig_bdt_cut > 0.5 and nevents_bkg_bdt_cut >= 2:
                    cuts.append(bdt_cut)
                    figure_of_merit_toys.append(soversb)
                    signal_toys.append(s_from_toy)

            # choose "optimal" bdt cut, check if it gives the expected sensitivity
            optimal_bdt_cut = 0
            if len(cuts) > 0:
                cuts = np.array(cuts)
                figure_of_merit_toys = np.array(figure_of_merit_toys)
                biggest = np.argmax(figure_of_merit_toys)
                optimal_bdt_cut = cuts[biggest]
                print(
                    f"{xbb_cut:.3f} {optimal_bdt_cut:.2f} {biggest} {figure_of_merit_toys[biggest]:.2f} {(figure_of_merit_toys[biggest]-expected_soverb_by_bdt_cut[optimal_bdt_cut]):.2f} {expected_soverb_by_bdt_cut[optimal_bdt_cut]:.2f}"
                )
                pull = (
                    figure_of_merit_toys[biggest] - expected_soverb_by_bdt_cut[optimal_bdt_cut]
                ) / expected_soverb_by_bdt_cut[optimal_bdt_cut]
                pull_s = (
                    signal_toys[biggest] - expected_s_by_bdt_cut[optimal_bdt_cut]
                ) / expected_s_by_bdt_cut[optimal_bdt_cut]
                diff = figure_of_merit_toys[biggest] - expected_soverb_by_bdt_cut[optimal_bdt_cut]
                diff_s = signal_toys[biggest] - expected_s_by_bdt_cut[optimal_bdt_cut]

                # fit
                gaus_fit[xbb_cut] = {
                    "pull": [np.mean(pull), np.std(pull)],  # norm.fit(pull),
                    "pull_s": [np.mean(pull_s), np.std(pull_s)],
                    "diff": [np.mean(diff), np.std(diff)],
                    "diff_s": [np.mean(diff_s), np.std(diff_s)],
                }

                h_pull.fill(pull, str(xbb_cut))
                h_pull_s.fill(pull_s, str(xbb_cut))
                h_diff.fill(diff, str(xbb_cut))
                h_diff_s.fill(diff_s, str(xbb_cut))

            # for this toy the BDT and Xbb cut are
            print("Xbb ", xbb_cut, "BDT ", optimal_bdt_cut)

            save_templates = False
            if save_templates:

                # define regions with optimal cut
                selection_regions = {
                    "pass_bin1": postprocessing.Region(
                        cuts={
                            "H2TXbb": [xbb_cut, CUT_MAX_VAL],
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

                # replace data distribution with toy!!
                # + signal
                inject_signal = True
                bdt_events_for_templates = bdt_events_dict_xbb_cut.copy()
                if inject_signal:
                    bdt_events_for_templates["data"] = pd.DataFrame(
                        {
                            "bdt_score": bdt_toy,
                            "H2TXbb": xbb_toy,
                            "H2PNetMass": mass_toy,
                            "weight": weight_toy,
                        }
                    )
                else:
                    bdt_events_for_templates["data"][mass_var] = random_mass

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
                postprocessing.save_templates(
                    templates, templ_dir / f"{year}_templates.pkl", fit_shape_var
                )

    # plot diff
    colours = {
        "darkblue": "#1f78b4",
        "lightblue": "#a6cee3",
        "lightred": "#FF502E",
        "red": "#e31a1c",
        "darkred": "#A21315",
        "orange": "#ff7f00",
        "green": "#7CB518",
        "mantis": "#81C14B",
        "forestgreen": "#2E933C",
        "darkgreen": "#064635",
        "purple": "#9381FF",
        "slategray": "#63768D",
        "deeppurple": "#36213E",
        "ashgrey": "#ACBFA4",
        "canary": "#FFE51F",
        "arylideyellow": "#E3C567",
        "earthyellow": "#D9AE61",
        "satinsheengold": "#C8963E",
        "flax": "#EDD382",
        "vanilla": "#F2F3AE",
        "dutchwhite": "#F5E5B8",
    }
    colors_by_xbb = {xbb_cut: list(colours.values())[i] for i, xbb_cut in enumerate(xbb_cuts)}
    print(gaus_fit)

    def plot_h(h_hist, xlabel, plot_name, xlim, gaus_label):
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        for xbb_cut in xbb_cuts:
            mu, sigma = gaus_fit[xbb_cut][gaus_label]
            hep.histplot(
                h_hist[{"cut": f"{xbb_cut}"}],
                ax=ax,
                label=f"Xbb > {xbb_cut}, Expected "
                + r"S/$\sqrt{S+B}$:"
                + f" {expected_by_xbb[xbb_cut]:.1f}, Mean: {mu:.2f}",
                color=colors_by_xbb[xbb_cut],
            )
            # n, bins = h_hist[{"cut": f"{xbb_cut}"}].to_numpy()
            # print(bins)
            # y = norm.pdf(bins, mu, sigma)
            # print("y ", y)
            # l = ax.plot(bins, y, linestyle='dashed', linewidth=2, color=colors_by_xbb[xbb_cut])
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
        r"S$_{t}$ - S",
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
