from __future__ import annotations

import math
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from hist import Hist
from hist.intervals import ratio_uncertainty
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike
from tqdm import tqdm

from .hh_vars import LUMI, data_key, hbb_bg_keys, sig_keys

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

ps = {
    "hatch": "xxx",
    "facecolor": "none",
    "lw": 0,
    "color": "k",
    "edgecolor": (0, 0, 0, 0.5),
    "linewidth": 0,
    "alpha": 0.4,
}

data_err_opts = {
    "linestyle": "none",
    "marker": ".",
    "markersize": 10.0,
    "elinewidth": 1,
}

color_by_sample = {
    "novhhtobb": "aquamarine",
    "gghtobb": "aquamarine",
    "vbfhtobb": "teal",
    "tthtobb": "cadetblue",
    "vhtobb": "tab:cyan",
    "others": "aquamarine",
    "hh4b": colours["red"],
    "hh4b-kl0": "fuchsia",
    "hh4b-kl2p45": "brown",
    "hh4b-kl5": "cyan",
    "vbfhh4b": "fuchsia",
    "vbfhh4b-k2v0": "purple",
    "ttbar": colours["darkblue"],
    "ttlep": "cadetblue",
    "qcd": colours["canary"],
    "qcd-ht": colours["canary"],
    "qcdb-ht": colours["canary"],
    "diboson": "orchid",
    "dibosonvjets": "orchid",
    "vjets": colours["green"],
    "vjetslnu": colours["orange"],
    "top_matched": "cornflowerblue",
    "W_matched": "royalblue",
    "unmatched": "lightsteelblue",
    "singletop": "cadetblue",
}

label_by_sample = {
    "novhhtobb": "ggH+VBF+ttH H(bb)",
    "gghtobb": "ggH(bb)",
    "vbfhtobb": "VBFH(bb)",
    "tthtobb": "ttH(bb)",
    "vhtobb": "VH(bb)",
    "others": "Others",
    "qcd": "Multijet",
    "qcd-ht": "Multijet HT bin",
    "qcdb-ht": "Multijet B HT bin",
    "hh4b": r"ggF HH4b",
    "hh4b-kl2p45": r"HH4b ($\kappa_{\lambda}=2.45$)",
    "hh4b-kl5": r"HH4b ($\kappa_{\lambda}=5$)",
    "hh4b-kl0": r"HH4b ($\kappa_{\lambda}=0$)",
    "vbfhh4b": r"VBF HH4b",
    "vbfhh4b-k2v0": r"VBF HH4b ($\kappa_{2V}=0$)",
    "diboson": "VV",
    "dibosonvjets": "VV+VJets",
    "ttbar": r"$t\bar{t}$ + Jets",
    "ttlep": r"$t\bar{t}$ + Jets (Lep)",
    "vjets": r"W/Z$(qq)$ + Jets",
    "vjetslnu": r"W/Z$(\ell\nu/\ell\ell)$ + Jets",
    "data": "Data",
    "top_matched": "Top Matched",
    "W_matched": "W Matched",
    "unmatched": "Unmatched",
    "singletop": "Single Top",
}

bg_order_default = [
    "vbfhtobb",
    "vhtobb",
    "tthtobb",
    "gghtobb",
    "diboson",
    "vjets",
    "vjetslnu",
    "ttbar",
    "qcd",
]


def plot_hists(
    hists,
    vars_to_plot,
    luminosity=None,  # float (fb)
    add_data=True,
    add_data_over_mc=True,
    mult_factor=1,  # multiplicative factor for signal
    logy=True,
    density=False,
    stack=True,
    show=False,
    bbox_to_anchor=(1.05, 1),
    energy=13.6,
):
    if add_data_over_mc and not add_data:
        add_data_over_mc = False
    if density:
        add_data_over_mc = False

    for var in vars_to_plot:
        if var not in hists:
            print(f"{var} not stored in hists")
            continue

        print(f"Will plot {var} histogram")
        h = hists[var]

        samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]

        signal_labels = [label for label in samples if label in sig_keys]
        signal = [h[{"Sample": label}] for label in signal_labels]
        signal_mult = [s * mult_factor for s in signal]

        bkg_labels = [
            label
            for label in samples
            if (label and label not in signal_labels and (label not in ["data"]))
        ]
        bkg = [h[{"Sample": label}] for label in bkg_labels]

        if add_data_over_mc:
            fig, (ax, rax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(8, 8),
                gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07},
                sharex=True,
            )
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            rax = None
        plt.subplots_adjust(hspace=0)

        # plot data
        if add_data:
            data = h[{"Sample": "data"}]
            hep.histplot(
                data,
                ax=ax,
                histtype="errorbar",
                color="k",
                capsize=4,
                yerr=True,
                label="Data",
                **data_err_opts,
            )

        # plot bkg
        # accumulate values and bins (little trick to avoid having error bars at the end)
        bkg_hists = []
        bkg_bins = []
        for h in bkg:
            hist_values, bins = h.to_numpy()
            bkg_hists.append(hist_values)
            bkg_bins.append(bins)

        if stack:
            bkg_args = {
                "histtype": "fill",
                "edgecolor": "black",
            }
        else:
            bkg_args = {
                "histtype": "step",
            }

        hep.histplot(
            bkg,
            ax=ax,
            stack=stack,
            edges=True,
            sort="yield",
            w2method=None,
            linewidth=1,
            density=density,
            label=[label_by_sample[bkg_label] for bkg_label in bkg_labels],
            color=[color_by_sample[bkg_label] for bkg_label in bkg_labels],
            **bkg_args,
        )

        # sum all the background
        tot = bkg[0].copy()
        for i, b in enumerate(bkg):
            if i > 0:
                tot = tot + b

        tot_val = tot.values()
        tot_val_zero_mask = tot_val == 0
        tot_val[tot_val_zero_mask] = 1

        tot_err = np.sqrt(tot_val)
        tot_err[tot_val_zero_mask] = 0

        # plot bkg uncertainty
        # print(tot.values().shape)
        # print(tot.axes[0].edges.shape)
        if not density:
            ax.stairs(
                values=tot.values() + tot_err,
                baseline=tot.values() - tot_err,
                edges=tot.axes[0].edges,
                **ps,
                label="Stat. unc.",
            )

        # plot signal
        if len(signal) > 0:
            # tot_signal = None

            for i, sig in enumerate(signal_mult):
                lab_sig_mult = f"{mult_factor} * {label_by_sample[signal_labels[i]]}"
                if mult_factor == 1:
                    lab_sig_mult = f"{label_by_sample[signal_labels[i]]}"
                # print(lab_sig_mult)
                hep.histplot(
                    sig,
                    ax=ax,
                    label=lab_sig_mult,
                    linewidth=1,
                    density=density,
                    color=color_by_sample[signal_labels[i]],
                )

                # if tot_signal is None:
                #     tot_signal = signal[i].copy()
                # else:
                #     tot_signal = tot_signal + signal[i]

            # plot the total signal (w/o scaling)
            # hep.histplot(tot_signal, ax=ax, label="Total signal", linewidth=3, color="tab:red")

            # add MC stat errors for total signal
            # ax.stairs(
            #    values=tot_signal.values() + np.sqrt(tot_signal.values()),
            #     baseline=tot_signal.values() - np.sqrt(tot_signal.values()),
            #    edges=sig.axes[0].edges,
            #   **errps,
            # )

        # plot data/mc ratio
        if add_data_over_mc:
            data_val = data.values()
            data_val[tot_val_zero_mask] = 1
            yerr = ratio_uncertainty(data_val, tot_val, "poisson")

            hep.histplot(
                data_val / tot_val,
                tot.axes[0].edges,
                yerr=yerr,
                ax=rax,
                histtype="errorbar",
                color="k",
                capsize=4,
            )
            rax.grid()

        ax.set_ylabel("Events")
        ax.set_xlabel("")

        if rax is not None:
            rax.set_xlabel(
                f"{h.axes[-1].label}"
            )  # assumes the variable to be plotted is at the last axis
            rax.set_ylabel("Data/MC", fontsize=20)
        else:
            ax.set_xlabel(f"{h.axes[-1].label}")

        if luminosity:
            hep.cms.lumitext(
                "%.1f " % luminosity + r"fb$^{-1}$" + f"({energy} TeV)", ax=ax, fontsize=20
            )
            hep.cms.text("Internal", ax=ax, fontsize=15)

        # add legend
        handles, labels = ax.get_legend_handles_labels()

        # get total yield of backgrounds per label
        first_key = next(iter(hists.keys()))
        # (sort by yield after pre-sel)
        order_dic = {}
        for bkg_label in bkg_labels:
            bkg_yield = hists[first_key][{"Sample": bkg_label}].sum().value
            order_dic[label_by_sample[bkg_label]] = bkg_yield

        summ = [order_dic[label] for label in labels[: len(bkg_labels)]]

        # get indices of labels arranged by yield
        order = []
        for _ in range(len(summ)):
            order.append(np.argmax(np.array(summ)))
            summ[np.argmax(np.array(summ))] = -100

        # print(labels)
        # print(labels[-1])
        if add_data:
            legend_handles = [handles[-1]] + [handles[i] for i in order] + handles[len(bkg) : -1]
            legend_labels = [labels[-1]] + [labels[i] for i in order] + labels[len(bkg) : -1]
            loc = "upper left"
        else:
            legend_handles = [handles[i] for i in order] + handles[len(bkg) :]
            legend_labels = [labels[i] for i in order] + labels[len(bkg) :]
            loc = "best"

        ax.legend(
            [legend_handles[idx] for idx in range(len(legend_handles))],
            [legend_labels[idx] for idx in range(len(legend_labels))],
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
        )

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-1)

        outpath = Path("plots")
        if not outpath.exists():
            outpath.mkdir(parents=True)

        if show:
            plt.show()
        else:
            plt.close()

        plt.savefig(f"{outpath}/{var}.pdf", bbox_inches="tight")


def _combine_hbb_bgs(hists, bg_keys):
    # skip this if no hbb bg keys specified
    if len(set(bg_keys) & set(hbb_bg_keys)) == 0:
        return hists, bg_keys

    # combine all hbb backgrounds into a single "Hbb" background for plotting
    hbb_hists = []
    for key in hbb_bg_keys:
        if key in bg_keys:
            hbb_hists.append(hists[key, ...])
            bg_keys.remove(key)

    if "Hbb" not in bg_keys:
        bg_keys.append("Hbb")

    hbb_hist = sum(hbb_hists)

    # have to recreate hist with "Hbb" sample included
    h = Hist(
        hist.axis.StrCategory(list(hists.axes[0]) + ["Hbb"], name="Sample"),
        *hists.axes[1:],
        storage="weight",
    )

    for i, sample in enumerate(hists.axes[0]):
        h.view()[i] = hists[sample, ...].view()

    h.view()[-1] = hbb_hist

    return h, bg_keys


def sigErrRatioPlot(
    h: Hist,
    sig_key: str,
    wshift: str,
    xlabel: str,
    title: str = None,
    plot_dir: str = None,
    name: str = None,
    show: bool = False,
    ylim: list = None,
):
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    nom = h[f"{sig_key}", :].values()
    hep.histplot(
        h[f"{sig_key}", :],
        histtype="step",
        label=sig_key,
        yerr=False,
        color="k",
        ax=ax,
        linewidth=2,
    )

    for skey, shift in [("Up", "up"), ("Down", "down")]:
        if f"{sig_key}_{wshift}_{shift}" not in h.axes[0]:
            continue

        colour = {"up": "#81C14B", "down": "#1f78b4"}[shift]
        hep.histplot(
            h[f"{sig_key}_{wshift}_{shift}", :],
            histtype="step",
            yerr=False,
            label=f"{sig_key} {skey}",
            color=colour,
            ax=ax,
            linewidth=2,
        )

        hep.histplot(
            h[f"{sig_key}_{wshift}_{shift}", :] / nom,
            histtype="step",
            label=f"{sig_key} {skey}",
            color=colour,
            ax=rax,
        )

    ax.legend()
    ax.set_ylim(0)
    ax.set_ylabel("Events")
    ax.set_title(title, y=1.08)

    rax.set_ylim([0, 2])
    if ylim is not None:
        rax.set_ylim(ylim)
    rax.set_xlabel(xlabel)
    rax.legend()
    rax.set_ylabel("Variation / Nominal")
    rax.grid(axis="y")

    plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def _process_samples(sig_keys, bg_keys, sig_scale_dict, syst, variation, bg_order):
    # set up samples, colours and labels
    bg_keys = [key for key in bg_order if key in bg_keys]
    bg_colours = [color_by_sample[sample] for sample in bg_keys]
    bg_labels = [label_by_sample[sample] for sample in bg_keys]

    if sig_scale_dict is None:
        sig_scale_dict = OrderedDict([(sig_key, 1.0) for sig_key in sig_keys])
    else:
        sig_scale_dict = {key: val for key, val in sig_scale_dict.items() if key in sig_keys}

    sig_colours = [color_by_sample[sig_key] for sig_key in sig_keys]
    sig_labels = OrderedDict()
    for sig_key, sig_scale in sig_scale_dict.items():
        label = label_by_sample.get(sig_key, sig_key)

        if sig_scale == 1:
            label = label  # noqa: PLW0127
        elif sig_scale <= 100:
            label = f"{label} $\\times$ {sig_scale:.2f}"
        else:
            label = f"{label} $\\times$ {sig_scale:.2e}"

        sig_labels[sig_key] = label

    # set up systematic variations if needed
    if syst is not None and variation is not None:
        wshift, wsamples = syst
        shift = variation
        skey = {"up": " Up", "down": " Down"}[shift]

        for i, key in enumerate(bg_keys):
            if key in wsamples:
                bg_keys[i] += f"_{wshift}_{shift}"
                bg_labels[i] += skey

        for sig_key in list(sig_scale_dict.keys()):
            if sig_key in wsamples:
                new_key = f"{sig_key}_{wshift}_{shift}"
                sig_scale_dict[new_key] = sig_scale_dict[sig_key]
                sig_labels[new_key] = sig_labels[sig_key] + skey
                del sig_scale_dict[sig_key], sig_labels[sig_key]

    return bg_keys, bg_colours, bg_labels, sig_colours, sig_scale_dict, sig_labels


def _fill_error(ax, edges, down, up, scale=1):
    ax.fill_between(
        np.repeat(edges, 2)[1:-1],
        np.repeat(down, 2) * scale,
        np.repeat(up, 2) * scale,
        color="black",
        alpha=0.2,
        hatch="//",
        linewidth=0,
    )


def _asimov_significance(s, b):
    """Asimov estimate of discovery significance (with no systematic uncertainties).
    See e.g. https://www.pp.rhul.ac.uk/~cowan/atlas/cowan_atlas_15feb11.pdf.
    Or for more explanation: https://www.pp.rhul.ac.uk/~cowan/stat/cowan_munich16.pdf
    """
    return np.sqrt(2 * ((s + b) * np.log(1 + (s / b)) - s))


def ratioHistPlot(
    hists: Hist,
    year: str,
    sig_keys: list[str],
    bg_keys: list[str],
    sig_err: ArrayLike | str | None = None,
    bg_err: ArrayLike = None,
    data_err: ArrayLike | bool | None = None,
    sortyield: bool = False,
    title: str | None = None,
    name: str = "",
    sig_scale_dict=None,
    xlim: int | None = None,
    xlim_low: int | None = None,
    ylim: int | None = None,
    ylim_low: int | None = None,
    show: bool = True,
    syst: tuple = None,
    variation: str = None,
    bg_err_type: str = "shaded",
    bg_err_mcstat: bool = False,
    exclude_qcd_mcstat: bool = True,
    plot_data: bool = True,
    bg_order=None,
    log: bool = False,
    logx: bool = False,
    ratio_ylims: list[float] | None = None,
    plot_significance: bool = False,
    significance_dir: str = "right",
    axrax: tuple | None = None,
    energy: str = "13.6",
    add_pull: bool = False,
    reweight_qcd: bool = False,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below

    Args:
        hists (Hist): input histograms per sample to plot
        year (str): datataking year
        sig_keys (List[str]): signal keys
        bg_keys (List[str]): background keys
        sig_err (Union[ArrayLike, str], optional): plot error on signal.
          if string, will take up down shapes from the histograms (assuming they're saved as "{sig_key}_{sig_err}_{up/down}")
          if 1D Array, will take as error per bin
        data_err (Union[ArrayLike, bool, None], optional): plot error on data.
          if True, will plot poisson error per bin
          if array, will plot given errors per bin
        title (str, optional): plot title. Defaults to None.
        blind_region (list): [min, max] range of values which should be blinded in the plot
          i.e. Data set to 0 in those bins
        name (str): name of file to save plot
        sig_scale_dict (Dict[str, float]): if scaling signals in the plot, dictionary of factors
          by which to scale each signal
        xlim_low (optional): x-limit low on plot
        ylim (optional): y-limit on plot
        show (bool): show plots or not
        syst (Tuple): Tuple of (wshift: name of systematic e.g. pileup,  wsamples: list of samples which are affected by this),
          to plot variations of this systematic.
        variation (str): options:
          "up" or "down", to plot only one wshift variation (if syst is not None).
          Defaults to None i.e. plotting both variations.
        plot_data (bool): plot data
        bg_order (List[str]): order in which to plot backgrounds
        ratio_ylims (List[float]): y limits on the ratio plots
        plot_significance (bool): plot Asimov significance below ratio plot
        significance_dir (str): "Direction" for significance. i.e. a > cut ("right"), a < cut ("left"), or per-bin ("bin").
        axrax (Tuple): optionally input ax and rax instead of creating new ones
    """

    # copy hists and bg_keys so input objects are not changed
    hists, bg_keys = deepcopy(hists), deepcopy(bg_keys)
    # hists, bg_keys = _combine_hbb_bgs(hists, bg_keys)

    if bg_order is None:
        bg_order = bg_order_default

    bg_keys, bg_colours, bg_labels, sig_colours, sig_scale_dict, sig_labels = _process_samples(
        sig_keys, bg_keys, sig_scale_dict, syst, variation, bg_order
    )

    if syst is not None and variation is None:
        # plot up/down variations
        wshift, wsamples = syst
        sig_err = wshift  # will plot sig variations below
        if len(bg_keys) > 0:
            bg_err = []
            for shift in ["down", "up"]:
                bg_sums = []
                for sample in bg_keys:
                    if sample in wsamples and f"{sample}_{wshift}_{shift}" in hists.axes[0]:
                        bg_sums.append(hists[f"{sample}_{wshift}_{shift}", :].values())
                    elif sample != "Hbb":
                        bg_sums.append(hists[sample, :].values())
                bg_err.append(np.sum(bg_sums, axis=0))

    if add_pull and bg_err is None:
        add_pull = False

    # set up plots
    if axrax is not None:
        if plot_significance:
            raise RuntimeError("Significance plots with input axes not implemented yet.")

        ax, rax = axrax
        ax.sharex(rax)
    elif add_pull or plot_significance:
        fig, (ax, rax, sax) = plt.subplots(
            3,
            1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.15},
            sharex=True,
        )
    else:
        fig, (ax, rax) = plt.subplots(
            2,
            1,
            figsize=(12, 12),
            gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.18},
            sharex=True,
        )

    # only use integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.rcParams.update({"font.size": 30})

    # plot histograms
    ax.set_ylabel("Events")

    # re-weight qcd
    kfactor = {sample: 1 for sample in bg_keys}
    if reweight_qcd:
        bg_yield = np.sum(sum([hists[sample, :] for sample in bg_keys]).values())
        data_yield = np.sum(hists[data_key, :].values())
        if bg_yield > 0:
            kfactor["qcd"] = data_yield / bg_yield

    # background samples
    if len(bg_keys) > 0:
        hep.histplot(
            [hists[sample, :] * kfactor[sample] for sample in bg_keys],
            ax=ax,
            # yerr=[np.sqrt(hists[sample, :].variances()) for sample in bg_keys],
            histtype="fill",
            sort="yield" if sortyield else None,
            stack=True,
            edgecolor="black",
            linewidth=2,
            label=bg_labels,
            color=bg_colours,
            # flow="none",
        )

    # signal samples
    if len(sig_scale_dict):
        hep.histplot(
            [hists[sig_key, :] * sig_scale for sig_key, sig_scale in sig_scale_dict.items()],
            ax=ax,
            histtype="step",
            linewidth=2,
            label=list(sig_labels.values()),
            color=sig_colours,
            # flow="none",
        )

        # plot signal errors
        if isinstance(sig_err, str):
            for skey, shift in [("Up", "up"), ("Down", "down")]:
                hep.histplot(
                    [
                        hists[f"{sig_key}_{sig_err}_{shift}", :] * sig_scale
                        for sig_key, sig_scale in sig_scale_dict.items()
                    ],
                    yerr=0,
                    ax=ax,
                    histtype="step",
                    label=[f"{sig_label} {skey}" for sig_label in sig_labels.values()],
                    alpha=0.6,
                    color=sig_colours[: len(sig_keys)],
                )
        elif sig_err is not None:
            for sig_key, sig_scale in sig_scale_dict.items():
                _fill_error(
                    ax,
                    hists.axes[1].edges,
                    hists[sig_key, :].values() * (1 - sig_err),
                    hists[sig_key, :].values() * (1 + sig_err),
                    sig_scale,
                )

    # plot background errors
    # if bg_err is None:
    # get background error from variances
    #    bg_tot = sum([hists[sample, :] for sample in bg_keys])
    #    bg_err = np.sqrt(bg_tot.variances())

    if bg_err is not None:
        bg_tot = sum([hists[sample, :] * kfactor[sample] for sample in bg_keys])
        if len(np.array(bg_err).shape) == 1:
            bg_err = [bg_tot - bg_err, bg_tot + bg_err]

        if bg_err_type == "shaded":
            ax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat(bg_err[0].values(), 2),
                np.repeat(bg_err[1].values(), 2),
                color="black",
                alpha=0.2,
                hatch="//",
                linewidth=0,
                label="Bkg. Unc.",
            )
        else:
            ax.stairs(
                bg_tot.values(),
                hists.axes[1].edges,
                color="black",
                linewidth=3,
                label="Bkg. Total",
                baseline=bg_tot.values(),
            )

            ax.stairs(
                bg_err[0],
                hists.axes[1].edges,
                color="red",
                linewidth=3,
                label="Bkg. Down",
                baseline=bg_err[0],
            )

            ax.stairs(
                bg_err[1],
                hists.axes[1].edges,
                color="#7F2CCB",
                linewidth=3,
                label="Bkg. Up",
                baseline=bg_err[1],
            )

    # plot bkg statistical uncertainty (excludes QCD)
    def get_variances(bg_hist):
        if bg_hist.variances() is None:
            return np.sqrt(bg_hist)
        else:
            return np.sqrt(bg_hist.variances())

    # print(hists.axes[1].widths)

    if bg_err_mcstat:
        bg_err_label = (
            "Stat. MC Uncertainty (excl. Multijet)"
            if exclude_qcd_mcstat
            else "Stat. MC Uncertainty"
        )

        plot_shaded = False

        mcstat_up = {}
        mcstat_dn = {}
        stack = None
        for isam, sample in enumerate(bg_keys):
            if exclude_qcd_mcstat and sample == "qcd":
                continue
            bg_yield = hists[sample, :] * kfactor[sample]
            sample_bg_err = get_variances(bg_yield)
            yerr = sample_bg_err
            if stack is None:
                sample_bg_err = [bg_yield - sample_bg_err, bg_yield + sample_bg_err]
                stack = bg_yield
            else:
                stack += bg_yield
                sample_bg_err = [stack - sample_bg_err, stack + sample_bg_err]

            mcstat_up[sample] = sample_bg_err[0].values()
            mcstat_dn[sample] = sample_bg_err[1].values()

            if not plot_shaded:
                if isam == 0:
                    hep.histplot(
                        stack,
                        ax=ax,
                        yerr=yerr,
                        histtype="errorbar",
                        markersize=0,
                        color="gray",
                        label=bg_err_label,
                    )
                else:
                    hep.histplot(
                        stack,
                        ax=ax,
                        yerr=yerr,
                        histtype="errorbar",
                        markersize=0,
                        color="gray",
                    )

        if plot_shaded:
            for isam, sample in enumerate(bg_keys):
                if exclude_qcd_mcstat and sample == "qcd":
                    continue

                if isam == 0:
                    ax.fill_between(
                        np.repeat(hists.axes[1].edges, 2)[1:-1],
                        np.repeat(mcstat_up[sample], 2),
                        np.repeat(mcstat_dn[sample], 2),
                        hatch="x",
                        linewidth=0,
                        edgecolor="k",
                        facecolor="none",
                        label=bg_err_label,
                    )
                else:
                    ax.fill_between(
                        np.repeat(hists.axes[1].edges, 2)[1:-1],
                        np.repeat(mcstat_up[sample], 2),
                        np.repeat(mcstat_dn[sample], 2),
                        hatch="x",
                        linewidth=0,
                        edgecolor="k",
                        facecolor="none",
                    )

    # plot data
    if plot_data:
        hep.histplot(
            hists[data_key, :],
            ax=ax,
            yerr=data_err,
            histtype="errorbar",
            label=label_by_sample[data_key],
            markersize=20,
            color="black",
            # flow="none",
        )

    if log:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-1:] + handles[len(bg_keys) : -1] + handles[: len(bg_keys)][::-1]
    labels = labels[-1:] + labels[len(bg_keys) : -1] + labels[: len(bg_keys)][::-1]
    ax.legend(handles, labels, bbox_to_anchor=(1.03, 1), loc="upper left")
    if "qcd" in kfactor and kfactor["qcd"] != 1:
        ax.get_legend().set_title(r"Multijet $\times$ " + f"{kfactor['qcd']:.2f}")

    if xlim_low is not None:
        if xlim is not None:
            ax.set_xlim(xlim_low, xlim)
        else:
            ax.set_xlim(xlim_low, None)

    y_lowlim = ylim_low if ylim_low is not None else 0 if not log else 0.001

    if ylim is not None:
        ax.set_ylim([y_lowlim, ylim])
    else:
        ax.set_ylim(y_lowlim)

    ax.set_xlabel("")

    # plot ratio below
    if plot_data and len(bg_keys) > 0:
        bg_tot = sum([hists[sample, :] * kfactor[sample] for sample in bg_keys])

        tot_val = bg_tot.values()
        tot_val_zero_mask = tot_val == 0
        tot_val[tot_val_zero_mask] = 1
        data_val = hists[data_key, :].values()
        data_val[tot_val_zero_mask] = 1
        yerr = ratio_uncertainty(data_val, tot_val, "poisson")
        yvalue = data_val / tot_val

        hep.histplot(
            yvalue,
            bg_tot.axes[0].edges,
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            markersize=20,
            color="black",
            capsize=0,
        )
        rax.set_xlabel(hists.axes[1].label)

        # fill error band of background
        if bg_err is not None:
            # (bkg + err) / bkg
            rax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat((bg_err[0].values()) / tot_val, 2),
                np.repeat((bg_err[1].values()) / tot_val, 2),
                color="black",
                alpha=0.1,
                hatch="//",
                linewidth=0,
            )
    else:
        rax.set_xlabel(hists.axes[1].label)

    rax.set_ylabel("Data/pred.")
    rax.set_ylim(ratio_ylims)
    minor_locator = mticker.AutoMinorLocator(2)
    rax.yaxis.set_minor_locator(minor_locator)
    rax.grid(axis="y", linestyle="-", linewidth=2, which="both")

    if plot_significance:
        bg_tot = sum([hists[sample, :] * kfactor[sample] for sample in bg_keys]).values()
        sigs = [hists[sig_key, :].values() for sig_key in sig_scale_dict]

        if significance_dir == "left":
            bg_tot = np.cumsum(bg_tot[::-1])[::-1]
            sigs = [np.cumsum(sig[::-1])[::-1] for sig in sigs]
            sax.set_ylabel(r"Asimov Sign. for $\leq$ Cuts")
        elif significance_dir == "right":
            bg_tot = np.cumsum(bg_tot)
            sigs = [np.cumsum(sig) for sig in sigs]
            sax.set_ylabel(r"Asimov Sign. for $\geq$ Cuts")
        elif significance_dir == "bin":
            sax.set_ylabel("Asimov Sign. per Bin")
        else:
            raise RuntimeError(
                'Invalid value for ``significance_dir``. Options are ["left", "right", "bin"].'
            )

        edges = hists.axes[1].edges
        hep.histplot(
            [(_asimov_significance(sig, bg_tot), edges) for sig in sigs],
            ax=sax,
            histtype="step",
            label=[label_by_sample.get(sig_key, sig_key) for sig_key in sig_scale_dict],
            color=sig_colours[: len(sig_keys)],
        )

        sax.legend(fontsize=12)
        sax.set_yscale("log")
        sax.set_ylim([1e-3, 0.1])
        sax.set_xlabel(hists.axes[1].label)

    if add_pull:
        # set title of 2nd panel empty
        rax.set_xlabel("")

        # (data -bkg )/unc_bkg
        bg_tot = sum([hists[sample, :] * kfactor[sample] for sample in bg_keys])
        tot_val = bg_tot.values()
        tot_val_zero_mask = tot_val == 0
        tot_val[tot_val_zero_mask] = 1
        data_val = hists[data_key, :].values()
        data_val[tot_val_zero_mask] = 1

        dataerr = np.sqrt(hists[data_key, :].variances())
        # replace dataerr of 0 by 1
        dataerr[dataerr == 0] = 1

        yhist = (hists[data_key, :] - bg_tot) / dataerr
        # yerr is not used, can be nan
        # yerr = ratio_uncertainty(hists[data_key, :] - bg_tot, dataerr, "poisson")

        # if math.isinf(yhist[5]):
        # blind!
        yhist[5] = 0
        yhist[6] = 0
        yhist[7] = 0

        hep.histplot(
            yhist,
            ax=sax,
            # yerr=yerr,
            histtype="fill",
            facecolor="gray",
            edgecolor="k",
        )
        sax.set_ylim([-2, 2])
        sax.set_xlabel(hists.axes[1].label)
        sax.set_ylabel(r"$\frac{Data - bkg}{\sigma(data)}$")

        minor_locator = mticker.AutoMinorLocator(2)
        sax.yaxis.set_minor_locator(minor_locator)
        sax.grid(axis="y", linestyle="-", linewidth=2, which="both")

    if title is not None:
        ax.set_title(title, y=1.08)

    if year == "all":
        hep.cms.label(
            "Work in Progress",
            data=True,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}",
            year=None,
            ax=ax,
            com=energy,
        )
    else:
        hep.cms.label(
            "Work in Progress",
            fontsize=24,
            data=True,
            lumi=f"{LUMI[year] / 1e3:.0f}",
            year=year,
            ax=ax,
            com=energy,
        )

    if axrax is None and len(name):
        if not name.endswith((".pdf", ".png")):
            plt.savefig(f"{name}.pdf", bbox_inches="tight")
            plt.savefig(f"{name}.png")
        else:
            plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def mesh2d(
    xbins: ArrayLike,
    ybins: ArrayLike,
    vals: ArrayLike,
    year: str,
    print_vals: bool = True,
    vmax: float = 1,
    ax: plt.axis.Axes = None,
    title: str = None,
    title_params: dict = None,
    xlabel: str = "AK8 Jet SD Mass [GeV]",
    ylabel: str = r"AK8 Jet $p_T$ [GeV]",
    plot_dir: str = "",
    name: str = "",
    show: bool = False,
    data: bool = True,
    fontsize: float = 28,
):
    """2D histogram with values printed in bins for e.g. trigger efficiencies and SFs"""
    if ax is None:
        in_ax = False
        fig, ax = plt.subplots(1, 1, figsize=(18, 17))
    else:
        in_ax = True

    mesh = ax.pcolormesh(xbins, ybins, vals.T, cmap="turbo", vmin=0, vmax=vmax)
    if print_vals:
        for i in range(len(ybins) - 1):
            for j in range(len(xbins) - 1):
                if not math.isnan(vals[j, i]):
                    ax.text(
                        (xbins[j] + xbins[j + 1]) / 2,
                        (ybins[i] + ybins[i + 1]) / 2,
                        vals[j, i].round(2),
                        color="black" if 0.1 * vmax < vals[j, i] < 0.9 * vmax else "white",
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                    )

    title_params = {"x": 0.35, "y": 1.005} if title_params is None else title_params
    ax.set_title(title, **title_params)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    hep.cms.label(ax=ax, data=data, year=year, lumi=round(LUMI[year] / 1e3), com="13.6")

    if in_ax:
        return mesh
    else:
        fig.colorbar(mesh, ax=ax, pad=0.01)

        if len(name):
            plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


def multiROCCurveGrey(
    rocs: dict,
    sig_effs: list[float],
    xlim=None,
    ylim=None,
    plot_dir: Path = None,
    name: str = "",
    show: bool = False,
):
    """Plot multiple ROC curves (e.g. train and test) + multiple signals"""
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    line_style = {"colors": "lightgrey", "linestyles": "dashed"}

    plt.figure(figsize=(12, 12))
    for roc_sigs in rocs.values():
        for roc in roc_sigs.values():
            plt.plot(
                roc["tpr"],
                roc["fpr"],
                label=roc["label"],
                linewidth=2,
            )

            for sig_eff in sig_effs:
                y = roc["fpr"][np.searchsorted(roc["tpr"], sig_eff)]
                plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
                plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

    hep.cms.label(data=False, rlabel="")
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc="upper left")

    if len(name):
        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def ROCCurve(
    roc: dict,
    xlim=None,
    ylim=None,
    thresholds=None,
    plot_dir: Path = None,
    name: str = "",
    show: bool = False,
):
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    if thresholds is None:
        thresholds = []

    th_colours = [
        # "#36213E",
        # "#9381FF",
        # "#1f78b4",
        # "#a6cee3",
        # "#32965D",
        "#7CB518",
        # "#EDB458",
        # "#ff7f00",
        "#a70000",
    ]

    plt.figure(figsize=(12, 12))
    plt.plot(
        roc["tpr"],
        roc["fpr"],
        linewidth=2,
    )

    pths = {th: [[], []] for th in thresholds}
    for th in thresholds:
        idx = _find_nearest(roc["thresholds"], th)
        pths[th][0].append(roc["tpr"][idx])
        pths[th][1].append(roc["fpr"][idx])

    for k, th in enumerate(thresholds):
        plt.scatter(
            *pths[th],
            marker="o",
            s=80,
            label=(rf"$T_{{Xbb}}$ > {th}"),
            color=th_colours[k],
            zorder=100,
        )

        plt.vlines(
            x=pths[th][0],
            ymin=0,
            ymax=pths[th][1],
            color=th_colours[k],
            linestyles="dashed",
            alpha=0.5,
        )

        plt.hlines(
            y=pths[th][1],
            xmin=0,
            xmax=pths[th][0],
            color=th_colours[k],
            linestyles="dashed",
            alpha=0.5,
        )

    hep.cms.label(
        data=False,
        rlabel="",
    )
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()

    if len(name):
        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fom(h_sb, plot_dir, name="figofmerit", show=False):
    """Plot FoM scan"""

    eff, bins_x, bins_y = h_sb.to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plt.rcParams.update({"font.size": 18})

    cbar = hep.hist2dplot(
        h_sb, ax=ax, cmin=np.min(eff[eff > 0]), cmax=np.max(eff[eff > 0]), flow="none"
    )
    cbar.cbar.set_label(r"Fig Of Merit", size=18)
    cbar.cbar.ax.get_yaxis().labelpad = 15
    for i in tqdm(range(len(bins_x) - 1)):
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

    ax.set_xlabel("BDT Cut")
    ax.set_ylabel(r"$T_{Xbb}$ Cut")
    ax.set_ylim(bins_y[0], bins_y[-1])
    ax.set_xlim(bins_x[0], bins_x[-1])
    fig.tight_layout()
    plt.savefig(f"{plot_dir}/{name}.png")
    plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
