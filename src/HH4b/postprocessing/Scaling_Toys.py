from __future__ import annotations

import pickle
from argparse import Namespace
from pathlib import Path
from typing import Callable

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from HH4b.hh_vars import (
    bg_keys,
    samples_run3,
)

xbb_cuts = np.arange(0.8, 0.999, 0.0025)
bdt_cuts = np.arange(0.9, 0.999, 0.0025)


# @nb.njit
def get_nevents_signal(events, cut, mass, mass_window):
    cut_mass = (events[mass] >= mass_window[0]) & (events[mass] <= mass_window[1])

    # get yield in Higgs mass window
    return np.sum(events["weight"][cut & cut_mass])


# @nb.njit
def get_nevents_nosignal(events, cut, mass, mass_window):
    cut_mass = ((events[mass] >= 60) & (events[mass] <= mass_window[0])) | (
        (events[mass] >= mass_window[1]) & (events[mass] <= 220)
    )

    # get yield NOT in Higgs mass window
    return np.sum(events["weight"][cut & cut_mass])


# @nb.njit
def abcd(
    events_dict,
    get_cut,
    get_anti_cut,
    txbb_cut,
    bdt_cut,
    mass,
    mass_window,
    bg_keys_all,
    sig_keys,
):
    bg_keys = bg_keys_all.copy()
    if "qcd" in bg_keys:
        bg_keys.remove("qcd")

    dicts = {"data": [], **{key: [] for key in bg_keys}}

    s = 0
    for key in sig_keys + ["data"] + bg_keys:
        events = events_dict[key]
        cut = get_cut(events, txbb_cut, bdt_cut)

        if key in sig_keys:
            s += get_nevents_signal(events, cut, mass, mass_window)
            continue

        # region A
        if key == "data":
            dicts[key].append(0)
        else:
            dicts[key].append(get_nevents_signal(events, cut, mass, mass_window))

        # region B
        dicts[key].append(get_nevents_nosignal(events, cut, mass, mass_window))

        cut = get_anti_cut(events)

        # region C
        dicts[key].append(get_nevents_signal(events, cut, mass, mass_window))
        # region D
        dicts[key].append(get_nevents_nosignal(events, cut, mass, mass_window))

    # other backgrounds (bg_tots[0] is in region A)
    bg_tots = np.sum([dicts[key] for key in bg_keys], axis=0)
    # subtract other backgrounds
    dmt = np.array(dicts["data"]) - bg_tots
    # A = B * C / D
    bqcd = dmt[1] * dmt[2] / dmt[3]

    background = bqcd + bg_tots[0] if len(bg_keys) else bqcd
    return s, background, dicts


def get_toy_from_hist(h_hist, n_samples, rng):
    """
    Get random values drawn from histogram
    """
    h, bins = h_hist.to_numpy()

    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(h)
    cdf = cdf / cdf[-1]
    values = rng.random(n_samples)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    return random_from_cdf


def get_toy_from_3d_hist(h_hist, n_samples, rng):
    """
    Get random values drawn from histogram
    """
    h, x_bins, y_bins, z_bins = h_hist.to_numpy()

    x_bin_midpoints = x_bins[:-1] + np.diff(x_bins) / 2
    y_bin_midpoints = y_bins[:-1] + np.diff(y_bins) / 2
    z_bin_midpoints = z_bins[:-1] + np.diff(z_bins) / 2
    cdf = np.cumsum(h.ravel())
    cdf = cdf / cdf[-1]
    values = rng.random(n_samples)
    value_bins = np.searchsorted(cdf, values)
    x_idx, y_idx, z_idx = np.unravel_index(
        value_bins, (len(x_bin_midpoints), len(y_bin_midpoints), len(z_bin_midpoints))
    )
    random_from_cdf = np.column_stack(
        (x_bin_midpoints[x_idx], y_bin_midpoints[y_idx], z_bin_midpoints[z_idx])
    )

    return random_from_cdf


args = Namespace(
    templates_tag="25June2ReRunBDTZbbSFs384Check",
    data_dir="/ceph/cms/store/user/dprimosc/bbbb/skimmer/",
    mass_bins=10,
    tag="25May9_v12v2_private_signal",
    years=["2022", "2022EE", "2023", "2023BPix"],
    training_years=None,
    mass="H2PNetMass",
    bdt_model="25Feb5_v13_glopartv2_rawmass",
    bdt_config="v13_glopartv2",
    txbb="glopart-v2",
    txbb_wps=[0.945, 0.85],
    bdt_wps=[0.94, 0.755, 0.03],
    method="abcd",
    vbf_txbb_wp=0.8,
    vbf_bdt_wp=0.9825,
    weight_ttbar_bdt=1.0,
    # sig_keys=['hh4b', 'hh4b-kl0', 'hh4b-kl2p45', 'hh4b-kl5', 'vbfhh4b', 'vbfhh4b-k2v0', 'vbfhh4b-kv1p74-k2v1p37-kl14p4', 'vbfhh4b-kvm0p012-k2v0p03-kl10p2', 'vbfhh4b-kvm0p758-k2v1p44-klm19p3', 'vbfhh4b-kvm0p962-k2v0p959-klm1p43', 'vbfhh4b-kvm1p21-k2v1p94-klm0p94', 'vbfhh4b-kvm1p6-k2v2p72-klm1p36', 'vbfhh4b-kvm1p83-k2v3p57-klm3p39', 'vbfhh4b-kvm2p12-k2v3p87-klm5p96'],
    sig_keys=["hh4b"],
    pt_first=300.0,
    pt_second=250.0,
    fom_vbf_samples=["vbfhh4b-k2v0"],
    fom_ggf_samples=["hh4b"],
    bdt_disc=True,
    event_list=False,
    event_list_dir="event_lists",
    bdt_roc=False,
    control_plots=False,
    fom_scan=False,
    fom_scan_bin1=True,
    fom_scan_bin2=True,
    fom_scan_vbf=False,
    templates=False,
    vbf=True,
    vbf_priority=False,
    correct_vbf_bdt_shape=True,
    blind=True,
    rerun_inference=True,
    scale_smear=False,
    dummy_txbb_sfs=False,
)

fom_window_by_mass = {"H2PNetMass": [110, 155]}
blind_window_by_mass = {"H2PNetMass": [110, 140]}
mass_window = np.array(fom_window_by_mass[args.mass])
n_mass_bins = int((220 - 60) / args.mass_bins)


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
    for key in list(samples_run3[year]):
        if "hh4b" in key and key != "hh4b":
            del samples_run3[year][key]

# get top-level HH4b directory
HH4B_DIR = "/home/users/woodson/HH4b/"
plot_dir = Path(f"{HH4B_DIR}/plots/Scaling_Toys/")
plot_dir.mkdir(exist_ok=True, parents=True)


def make_histograms(mass_array, xbb_array, bdt_array):
    mass_axis = hist.axis.Regular(16, 60, 220, name="mass")
    bdt_bins = 100
    bdt_axis = hist.axis.Regular(bdt_bins, 0, 1, name="bdt")
    xbb_bins = 100
    xbb_axis = hist.axis.Regular(xbb_bins, 0, 1, name="xbb")

    # mask = (mass_array > 150) | (mass_array < 110)
    mask = mass_array > 0  # no masking for now

    h_mass = hist.Hist(mass_axis)
    h_mass.fill(mass_array[mask])
    h_xbb = hist.Hist(xbb_axis)
    h_xbb.fill(xbb_array[mask])
    h_bdt = hist.Hist(bdt_axis)
    h_bdt.fill(bdt_array[mask])

    # sample toys from 3D distribution
    h_mass_xbb_bdt = hist.Hist(mass_axis, xbb_axis, bdt_axis)
    h_mass_xbb_bdt.fill(
        mass=mass_array[mask],
        xbb=xbb_array[mask],
        bdt=bdt_array[mask],
    )

    # make 2D histograms
    h_mass_xbb = hist.Hist(mass_axis, xbb_axis)
    h_mass_xbb.fill(
        mass=mass_array[mask],
        xbb=xbb_array[mask],
    )
    h_mass_bdt = hist.Hist(mass_axis, bdt_axis)
    h_mass_bdt.fill(
        mass=mass_array[mask],
        bdt=bdt_array[mask],
    )
    h_xbb_bdt = hist.Hist(xbb_axis, bdt_axis)
    h_xbb_bdt.fill(
        xbb=xbb_array[mask],
        bdt=bdt_array[mask],
    )

    return h_mass, h_xbb, h_bdt, h_mass_xbb, h_mass_bdt, h_xbb_bdt, h_mass_xbb_bdt


def plot_corner(h_mass, h_xbb, h_bdt, h_mass_xbb, h_mass_bdt, h_xbb_bdt):
    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    hep.histplot(h_mass, ax=ax[0, 0])
    hep.histplot(h_xbb, ax=ax[1, 1])
    hep.histplot(h_bdt, ax=ax[2, 2])
    ax[0, 0].set_xlim(60, 220)
    ax[1, 1].set_xlim(0, 1)
    ax[1, 1].set_ylim(5e-1, 5e6)
    ax[2, 2].set_xlim(0, 1)
    ax[1, 1].set_yscale("log")
    ax[2, 2].set_yscale("log")
    hep.hist2dplot(h_mass_xbb, ax=ax[1, 0], norm=mpl.colors.LogNorm())
    hep.hist2dplot(h_mass_bdt, ax=ax[2, 0], norm=mpl.colors.LogNorm())
    hep.hist2dplot(h_xbb_bdt, ax=ax[2, 1], norm=mpl.colors.LogNorm())
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[1, 2].axis("off")
    # tight layout
    plt.tight_layout()
    plt.savefig(plot_dir / "corners.pdf", bbox_inches="tight")
    return fig


def logit(x):
    """Logit function."""
    return np.log(x / (1 - x))


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def minuit_transform(x, xmin=0, xmax=1):
    """Minuit transform: See https://root.cern.ch/download/minuit.pdf#page=8"""
    return np.arcsin(2 * (x - xmin) / (xmax - xmin) - 1)


def minuit_inverse_transform(x, xmin=0, xmax=1):
    """Inverse Minuit transform: See https://root.cern.ch/download/minuit.pdf#page=8"""
    return (np.sin(x) + 1) * (xmax - xmin) / 2 + xmin


# @nb.njit
def scan_fom(
    method: str,
    events_combined: pd.DataFrame,
    get_cut: Callable,
    get_anti_cut: Callable,
    xbb_cuts: np.ArrayLike,
    bdt_cuts: np.ArrayLike,
    mass_window: list[float],
    bg_keys: list[str],
    sig_keys: list[str],
    fom: str = "2sqrt(b)/s",
    mass: str = "H2Msd",
):
    """Generic FoM scan for given region, defined in the ``get_cut`` function."""

    print(f"Scanning {fom} with {method}")
    all_s = []
    all_b = []
    all_sideband_events = []
    all_xbb_cuts = []
    all_bdt_cuts = []
    all_fom = []
    for xbb_cut in xbb_cuts:
        for bdt_cut in bdt_cuts:
            nevents_sig, nevents_bkg, _ = abcd(
                events_combined,
                get_cut,
                get_anti_cut,
                xbb_cut,
                bdt_cut,
                mass,
                mass_window,
                bg_keys,
                sig_keys,
            )

            # number of events in data in sideband
            cut = get_cut(events_combined["data"], xbb_cut, bdt_cut)
            nevents_sideband = get_nevents_nosignal(events_combined["data"], cut, mass, mass_window)

            if fom == "s/sqrt(s+b)":
                if nevents_sig + nevents_bkg > 0:
                    figure_of_merit = nevents_sig / np.sqrt(nevents_sig + nevents_bkg)
                else:
                    figure_of_merit = np.nan
            elif fom == "2sqrt(b)/s":
                if nevents_bkg > 0 and nevents_sig > 0:
                    figure_of_merit = 2 * np.sqrt(nevents_bkg) / nevents_sig
                else:
                    figure_of_merit = np.nan
            else:
                raise ValueError("Invalid FOM")

            all_b.append(nevents_bkg)
            all_s.append(nevents_sig)
            all_sideband_events.append(nevents_sideband)
            all_xbb_cuts.append(xbb_cut)
            all_bdt_cuts.append(bdt_cut)
            all_fom.append(figure_of_merit)

    all_fom = np.array(all_fom)
    all_b = np.array(all_b)
    all_s = np.array(all_s)
    all_sideband_events = np.array(all_sideband_events)
    all_xbb_cuts = np.array(all_xbb_cuts)
    all_bdt_cuts = np.array(all_bdt_cuts)

    return all_fom, all_b, all_s, all_sideband_events, all_xbb_cuts, all_bdt_cuts


def get_optimal_cuts(all_fom, all_b, all_s, all_sideband_events, all_xbb_cuts, all_bdt_cuts):

    bdt_cuts = np.sort(np.unique(all_bdt_cuts))
    xbb_cuts = np.sort(np.unique(all_xbb_cuts))

    h_sb = hist.Hist(
        hist.axis.Variable(list(bdt_cuts), name="bdt_cut"),
        hist.axis.Variable(list(xbb_cuts), name="xbb_cut"),
    )
    h_s = hist.Hist(
        hist.axis.Variable(list(bdt_cuts), name="bdt_cut"),
        hist.axis.Variable(list(xbb_cuts), name="xbb_cut"),
    )
    h_b = hist.Hist(
        hist.axis.Variable(list(bdt_cuts), name="bdt_cut"),
        hist.axis.Variable(list(xbb_cuts), name="xbb_cut"),
    )

    for xbb_cut in xbb_cuts:
        for bdt_cut in bdt_cuts:
            # find index of this cut
            idx = np.where((all_bdt_cuts == bdt_cut) & (all_xbb_cuts == xbb_cut))[0][0]
            if all_s[idx] > 0.5 and all_b[idx] >= 2 and all_sideband_events[idx] >= 12:
                h_sb.fill(bdt_cut, xbb_cut, weight=all_fom[idx])
                h_b.fill(bdt_cut, xbb_cut, weight=all_b[idx])
                h_s.fill(bdt_cut, xbb_cut, weight=all_s[idx])

    masked_h_sb = np.ma.masked_equal(h_sb.values(), 0)

    global_min = np.min(masked_h_sb)

    if np.ma.is_masked(global_min):
        return None, None, None, None, None, None

    masked_h_sb_min_diff = np.abs(masked_h_sb - global_min)

    argmin_axis0 = np.argmin(masked_h_sb_min_diff, axis=0)
    min_axis0 = np.min(masked_h_sb_min_diff, axis=0)

    argmin_axis1 = np.argmin(masked_h_sb_min_diff, axis=1)
    min_axis1 = np.min(masked_h_sb_min_diff, axis=1)

    bdt_cut = h_sb.axes[0].edges[argmin_axis0[min_axis0 == 0]][0]
    xbb_cut = h_sb.axes[1].edges[argmin_axis1[min_axis1 == 0]][0]

    b = h_b.values()[argmin_axis0[min_axis0 == 0], argmin_axis1[min_axis1 == 0]][0]
    s = h_s.values()[argmin_axis0[min_axis0 == 0], argmin_axis1[min_axis1 == 0]][0]

    return global_min, bdt_cut, xbb_cut, h_sb, b, s


def get_anti_cuts():

    def anti_cut_ggf(events):
        cut_xbb = events["H2TXbb"] < 0.3
        cut_bdt = events["bdt_score"] < 0.6
        return cut_xbb & cut_bdt

    return anti_cut_ggf


def get_cuts():

    # bin 1 region
    def get_cut_bin1(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2TXbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        return cut_xbb & cut_bdt

    return get_cut_bin1


# @nb.njit
def run_toys(
    ntoys, lumi_scale=1.0, method="2dkde", optimize=True, xbb_cut_data=0.945, bdt_cut_data=0.935
):

    bdt_cut_toys = []
    xbb_cut_toys = []
    s_toys = []
    b_toys = []
    fom_toys = []

    # get numpy generator for reproducibility
    rng = np.random.default_rng(42)

    for itoy in range(ntoys):
        n_samples = rng.poisson(integral * lumi_scale)

        if method == "3dhist":
            mass_xbb_bdt_toy = get_toy_from_3d_hist(h_mass_xbb_bdt, n_samples, rng)
            mass_toy = mass_xbb_bdt_toy[:, 0]
            xbb_toy = mass_xbb_bdt_toy[:, 1]
            bdt_toy = mass_xbb_bdt_toy[:, 2]
        elif method == "1dhist":
            mass_toy = get_toy_from_hist(h_mass, n_samples, rng)
            xbb_toy = get_toy_from_hist(h_xbb, n_samples, rng)
            bdt_toy = get_toy_from_hist(h_bdt, n_samples, rng)
        elif method == "3dkde":
            sampled_transformed_data = kde_3d_mass_xbb_bdt.resample(n_samples, seed=rng).T
            mass_toy = minuit_inverse_transform(sampled_transformed_data[:, 0], xmin=60, xmax=220)
            xbb_toy = sigmoid(sampled_transformed_data[:, 1])
            bdt_toy = sigmoid(sampled_transformed_data[:, 2])
        elif method == "2dkde":
            mass_toy = minuit_inverse_transform(
                kde_1d_mass.resample(n_samples, seed=rng)[0], xmin=60, xmax=220
            )
            sampled_transformed_data = kde_2d_xbb_bdt.resample(n_samples, seed=rng).T
            xbb_toy = sigmoid(sampled_transformed_data[:, 0])
            bdt_toy = sigmoid(sampled_transformed_data[:, 1])
        elif method == "1dkde":
            mass_toy = minuit_inverse_transform(
                kde_1d_mass.resample(n_samples, seed=rng)[0], xmin=60, xmax=220
            )
            xbb_toy = sigmoid(kde_1d_xbb.resample(n_samples, seed=rng)[0])
            bdt_toy = sigmoid(kde_1d_bdt.resample(n_samples, seed=rng)[0])
        else:
            raise ValueError(f"Unknown method: {method}")

        events_toy = {}
        for key in events_combined:
            if key != "data":
                events_toy[key] = events_combined[key][
                    ["H2PNetMass", "bdt_score", "H2TXbb", "weight"]
                ].copy()
                events_toy[key]["weight"] *= lumi_scale  # scale by lumi
            else:
                events_toy["data"] = pd.DataFrame(
                    {
                        "H2PNetMass": mass_toy,
                        "bdt_score": bdt_toy,
                        "H2TXbb": xbb_toy,
                        "weight": np.ones_like(mass_toy),
                    }
                )

        if optimize:
            all_fom, all_b, all_s, all_sideband_events, all_xbb_cuts, all_bdt_cuts = scan_fom(
                args.method,
                events_toy,
                get_cuts(),
                get_anti_cuts(),
                xbb_cuts,
                bdt_cuts,
                mass_window,
                bg_keys=bg_keys,
                sig_keys=args.fom_ggf_samples,
                mass=args.mass,
            )

            global_min, bdt_cut, xbb_cut, h_sb, b, s = get_optimal_cuts(
                all_fom, all_b, all_s, all_sideband_events, all_xbb_cuts, all_bdt_cuts
            )
            if global_min is None:
                print(f"Skipping toy {itoy} due to no valid cuts found.")
                continue
        else:
            # use fixed cuts
            bdt_cut = bdt_cut_data
            xbb_cut = xbb_cut_data
            s, b, _ = abcd(
                events_toy,
                get_cuts(),
                get_anti_cuts(),
                xbb_cut,
                bdt_cut,
                args.mass,
                mass_window,
                bg_keys,
                args.fom_ggf_samples,
            )

            global_min = 2 * np.sqrt(b) / s if s > 0 else np.nan

        bdt_cut_toys.append(bdt_cut)
        xbb_cut_toys.append(xbb_cut)
        s_toys.append(s)
        b_toys.append(b)
        fom_toys.append(global_min)

        print(f"Lumi scale: {lumi_scale}, Toy: {itoy + 1}")
        print(f"Optimal cuts: bdt_cut={bdt_cut:.4f}, xbb_cut={xbb_cut:.4f}")
        print(
            f"2sqrt(b)/s={global_min:.4f}, b={b:.4f}, s={s:.4f}, s/b={s/b:.4f}, s/sqrt(b)={s/np.sqrt(b):.4f}"
        )

    return bdt_cut_toys, xbb_cut_toys, s_toys, b_toys, fom_toys


if __name__ == "__main__":
    # events_dict_postprocess = {}
    # cutflows = {}
    # for year in args.years:
    #     print(f"\n{year}")
    #     events, cutflow = load_process_run3_samples(
    #         args,
    #         year,
    #         [],
    #         args.control_plots,
    #         plot_dir,
    #         mass_window,
    #         args.rerun_inference,
    #     )
    #     events_dict_postprocess[year] = events
    #     cutflows[year] = cutflow

    # processes = ["data"] + args.sig_keys + bg_keys
    # bg_keys_combined = bg_keys.copy()
    # if not args.control_plots and not args.bdt_roc:
    #     if "qcd" in processes:
    #         processes.remove("qcd")
    #     if "qcd" in bg_keys:
    #         bg_keys.remove("qcd")
    #     if "qcd" in bg_keys_combined:
    #         bg_keys_combined.remove("qcd")

    # if len(args.years) > 1:
    #     # list of years available for a given process to scale to full lumi,
    #     scaled_by_years = {
    #         # "zz": ["2022", "2022EE", "2023"],
    #     }
    #     events_combined, scaled_by = combine_run3_samples(
    #         events_dict_postprocess,
    #         processes,
    #         bg_keys=bg_keys_combined,
    #         scale_processes=scaled_by_years,
    #         years_run3=args.years,
    #     )
    #     print("Combined years")
    # else:
    #     events_combined = events_dict_postprocess[args.years[0]]
    #     scaled_by = {}
    # with open(f"{HH4B_DIR}/data/events_combined_{args.templates_tag}.pkl", "wb") as f:
    #     pickle.dump(events_combined, f)

    # open the pickle file
    with open(f"{HH4B_DIR}/data/events_combined_{args.templates_tag}.pkl", "rb") as f:  # noqa: PTH123
        events_combined = pickle.load(f)

    integral = len(events_combined["data"])

    h_mass, h_xbb, h_bdt, h_mass_xbb, h_mass_bdt, h_xbb_bdt, h_mass_xbb_bdt = make_histograms(
        events_combined["data"]["H2PNetMass"],
        events_combined["data"]["H2TXbb"],
        events_combined["data"]["bdt_score"],
    )

    # fig = plot_corner(h_mass, h_xbb, h_bdt, h_mass_xbb, h_mass_bdt, h_xbb_bdt)

    data_array = events_combined["data"][["H2PNetMass", "H2TXbb", "bdt_score"]].to_numpy()
    transformed_data_array = np.column_stack(
        (
            minuit_transform(data_array[:, 0], xmin=60, xmax=220),
            logit(data_array[:, 1]),
            logit(data_array[:, 2]),
        )
    )
    kde_3d_mass_xbb_bdt = gaussian_kde(transformed_data_array.T, bw_method="silverman")
    kde_2d_xbb_bdt = gaussian_kde(transformed_data_array[:, 1:].T, bw_method="silverman")
    kde_1d_mass = gaussian_kde(transformed_data_array[:, 0], bw_method="silverman")
    kde_1d_xbb = gaussian_kde(transformed_data_array[:, 1], bw_method="silverman")
    kde_1d_bdt = gaussian_kde(transformed_data_array[:, 2], bw_method="silverman")

    # lumi_scale = 138.0 / 62.0
    lumi_scale = 1
    ntoys = 100
    # ntoys = -1
    method = "2dkde"
    optimize = False

    if ntoys > 0:
        bdt_cut_toys, xbb_cut_toys, s_toys, b_toys, fom_toys = run_toys(
            ntoys, lumi_scale=lumi_scale, method=method, optimize=optimize
        )
    else:
        all_fom, all_b, all_s, all_sideband_events, all_xbb_cuts, all_bdt_cuts = scan_fom(
            args.method,
            events_combined,
            get_cuts(),
            get_anti_cuts(),
            xbb_cuts,
            bdt_cuts,
            mass_window,
            bg_keys=bg_keys,
            sig_keys=args.fom_ggf_samples,
            mass=args.mass,
        )

        fom_data, bdt_cut_data, xbb_cut_data, h_sb, b_data, s_data = get_optimal_cuts(
            all_fom, all_b, all_s, all_sideband_events, all_xbb_cuts, all_bdt_cuts
        )
        print("Data")
        print(f"Optimal cuts: bdt_cut={bdt_cut_data:.4f}, xbb_cut={xbb_cut_data:.4f}")
        print(
            f"2sqrt(b)/s={fom_data:.4f}, b={b_data:.4f}, s={s_data:.4f}, s/b={s_data/b_data:.4f}, s/sqrt(b)={s_data/np.sqrt(b_data):.4f}"
        )

    # save all arrays to a pickle file
    if optimize:
        optimize_str = f"{xbb_cuts[0]:.4f}_{xbb_cuts[-1]:.4f}_{xbb_cuts[1]-xbb_cuts[0]:.4f}_{bdt_cuts[0]:.4f}_{bdt_cuts[-1]:.4f}_{bdt_cuts[1]-bdt_cuts[0]:.4f}"
    else:
        optimize_str = "noopt"
    if ntoys > 0:
        with open(  # noqa: PTH123
            plot_dir
            / f"fom_toys_{method}_{ntoys}_{lumi_scale:4f}_{optimize_str}"
            ".pkl",
            "wb",
        ) as f:
            pickle.dump(
                {
                    "bdt_cut_toys": bdt_cut_toys,
                    "xbb_cut_toys": xbb_cut_toys,
                    "s_toys": s_toys,
                    "b_toys": b_toys,
                    "fom_toys": fom_toys,
                    "lumi_scale": lumi_scale,
                },
                f,
            )
    else:
        with open(  # noqa: PTH123
            plot_dir
            / f"fom_data_{optimize_str}"
            ".pkl",
            "wb",
        ) as f:
            pickle.dump(
                {
                    "bdt_cut_data": bdt_cut_data,
                    "xbb_cut_data": xbb_cut_data,
                    "s_data": s_data,
                    "b_data": b_data,
                    "fom_data": fom_data,
                },
                f,
            )
