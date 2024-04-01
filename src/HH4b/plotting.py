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
    "hh4b": colours["red"],
    "hh4b-kl0": "fuchsia",
    "hh4b-kl2p45": "brown",
    "hh4b-kl5": "cyan",
    "ttbar": colours["darkblue"],
    "qcd": colours["canary"],
    "qcd-ht": colours["canary"],
    "qcdb-ht": colours["canary"],
    "diboson": "orchid",
    "dibosonvjets": "orchid",
    "vjets": colours["green"],
    "vjetslnu": "seagreen",
}

label_by_sample = {
    "novhhtobb": "ggH+VBF+ttH H(bb)",
    "gghtobb": "ggH(bb)",
    "vbfhtobb": "VBFH(bb)",
    "tthtobb": "ttH(bb)",
    "vhtobb": "VH(bb)",
    "qcd": "Multijet",
    "qcd-ht": "Multijet HT bin",
    "qcdb-ht": "Multijet B HT bin",
    "hh4b": r"HH 4b ($\kappa_{\lambda}=1$)",
    "hh4b-kl2p45": r"HH 4b ($\kappa_{\lambda}=2.45$)",
    "hh4b-kl5": r"HH 4b ($\kappa_{\lambda}=5$)",
    "hh4b-kl0": r"HH 4b ($\kappa_{\lambda}=0$)",
    "diboson": "VV",
    "dibosonvjets": "VV+VJets",
    "ttbar": r"$t\bar{t}$ + Jets",
    "vjets": r"W/Z$(qq)$ + Jets",
    "vjetslnu": r"W/Z$(\ell\nu/\ell\ell)$ + Jets",
    "data": "Data",
}

bg_order_default = ["vbfhtobb", "vhtobb", "tthtobb", "gghtobb", "diboson", "vjetslnu", "vjets", "ttbar", "qcd"]


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


def _process_samples(sig_keys, bg_keys, sig_scale_dict, variation, bg_order):
    # set up samples, colours and labels
    bg_keys = [key for key in bg_order if key in bg_keys]
    bg_colours = [color_by_sample[sample] for sample in bg_keys]
    bg_labels = [label_by_sample[sample] for sample in bg_keys]

    if sig_scale_dict is None:
        sig_scale_dict = OrderedDict([(sig_key, 1.0) for sig_key in sig_keys])
    else:
        sig_scale_dict = deepcopy(sig_scale_dict)

    sig_colours = [color_by_sample[sig_key] for sig_key in sig_keys]
    sig_labels = OrderedDict()
    for sig_key, sig_scale in sig_scale_dict.items():
        label = sig_key if sig_key not in label_by_sample else label_by_sample[sig_key]

        if sig_scale == 1:
            label = label  # noqa: PLW0127
        elif sig_scale <= 100:
            label = f"{label} $\\times$ {sig_scale:.2f}"
        else:
            label = f"{label} $\\times$ {sig_scale:.2e}"

        sig_labels[sig_key] = label

    # set up systematic variations if needed
    if variation is not None:
        wshift, shift, wsamples = variation
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
    variation: tuple | None = None,
    plot_data: bool = True,
    bg_order=None,
    log: bool = False,
    logx: bool = False,
    ratio_ylims: list[float] | None = None,
    plot_significance: bool = False,
    significance_dir: str = "right",
    axrax: tuple | None = None,
    energy: str = "13.6",
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
        variation (Tuple): Tuple of
          (wshift: name of systematic e.g. pileup, shift: up or down, wsamples: list of samples which are affected by this)
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
        sig_keys, bg_keys, sig_scale_dict, variation, bg_order
    )

    # set up plots
    if axrax is not None:
        if plot_significance:
            raise RuntimeError("Significance plots with input axes not implemented yet.")

        ax, rax = axrax
        ax.sharex(rax)
    elif plot_significance:
        fig, (ax, rax, sax) = plt.subplots(
            3,
            1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0},
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

    # background samples
    if len(bg_keys) > 0:
        hep.histplot(
            [hists[sample, :] for sample in bg_keys],
            ax=ax,
            histtype="fill",
            sort="yield" if sortyield else None,
            stack=True,
            edgecolor="black",
            linewidth=2,
            label=bg_labels,
            color=bg_colours,
            flow="none",
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
            flow="none",
        )

    # plot signal errors
    if isinstance(sig_err, str):
        # scolours = {"down": colours["lightred"], "up": colours["darkred"]}
        for skey, shift in [("Up", "up"), ("Down", "down")]:
            hep.histplot(
                [
                    hists[f"{sig_key}_{sig_err}_{shift}", :] * sig_scale
                    for sig_key, sig_scale in sig_scale_dict.items()
                ],
                yerr=0,
                ax=ax,
                histtype="step",
                label=[f"{sig_key} {skey:.2f}".format(skey) for sig_key in sig_scale_dict],
                alpha=0.6,
                color=sig_colours[: len(sig_keys)],
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
            flow="none",
        )

    if log:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-1:] + handles[len(bg_keys) : -1] + handles[: len(bg_keys)][::-1]
    labels = labels[-1:] + labels[len(bg_keys) : -1] + labels[: len(bg_keys)][::-1]
    ax.legend(handles, labels, bbox_to_anchor=(1.03, 1), loc="upper left")

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
        bg_tot = sum([hists[sample, :] for sample in bg_keys])

        tot_val = bg_tot.values()
        tot_val_zero_mask = tot_val == 0
        tot_val[tot_val_zero_mask] = 1
        data_val = hists[data_key, :].values()
        data_val[tot_val_zero_mask] = 1
        yerr = ratio_uncertainty(data_val, tot_val, "poisson")

        hep.histplot(
            data_val / tot_val,
            bg_tot.axes[0].edges,
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            markersize=20,
            color="black",
            capsize=0,
        )

        rax.set_xlabel(hists.axes[1].label)
        # print(hists[data_key, :] / (bg_tot.values() + 1e-5))
    else:
        rax.set_xlabel(hists.axes[1].label)

    rax.set_ylabel("Data/pred.")
    rax.set_ylim(ratio_ylims)
    minor_locator = mticker.AutoMinorLocator(2)
    rax.yaxis.set_minor_locator(minor_locator)
    rax.grid(axis="y", linestyle="-", linewidth=2, which="both")

    if plot_significance:
        bg_tot = sum([hists[sample, :] for sample in bg_keys]).values()
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
            label=[
                sig_key if sig_key not in label_by_sample else label_by_sample[sig_key]
                for sig_key in sig_scale_dict
            ],
            color=sig_colours[: len(sig_keys)],
        )

        sax.legend(fontsize=12)
        sax.set_yscale("log")
        sax.set_ylim([1e-3, 0.1])
        sax.set_xlabel(hists.axes[1].label)

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

    if axrax is None:
        if len(name):
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
