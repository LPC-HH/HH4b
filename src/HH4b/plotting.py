from __future__ import annotations

import math
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from hist import Hist
from hist.intervals import poisson_interval, ratio_uncertainty
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike
from pyparsing import Any
from tqdm import tqdm
from typing_extensions import Literal

from .hh_vars import LUMI, data_key

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
    "tthtobb": "#b9ac70",
    "vhtobb": "#94a4a2",
    "vhtthtobb": "#94a4a2",
    "others": "aquamarine",
    "hh4b": "#FF9933",
    "hh4b-kl0": "#FF9933",
    "hh4b-kl2p45": "#FF9933",
    "hh4b-kl5": "#FF9933",
    "vbfhh4b": "#FF0000",
    "vbfhh4b-k2v0": "#FF0000",
    "vbfhh4b-k2v2": "#FF0000",
    "vbfhh4b-kl2": "#FF0000",
    "ttbar": "#832db6",
    "ttlep": "cadetblue",
    "qcd": "#3f90da",
    "qcd-ht": colours["canary"],
    "qcdb-ht": colours["canary"],
    "zz": "#717581",
    "nozzdiboson": "#a96b59",
    "diboson": "orchid",
    "dibosonvjets": "#92dadd",
    "vjets": "#92dadd",
    "vjetslnu": colours["orange"],
    "top_matched": "cornflowerblue",
    "W_matched": "royalblue",
    "unmatched": "lightsteelblue",
    "singletop": "cadetblue",
}

label_by_sample = {
    "novhhtobb": r"ggH+VBF+$t\bar{t}$H",
    "gghtobb": "ggH",
    "vbfhtobb": "VBFH",
    "tthtobb": r"$t\bar{t}$H",
    "vhtobb": "VH",
    "vhtthtobb": r"VH, $t\bar{t}$H",
    "others": "Others",
    "qcd": "QCD multijet",
    "qcd-ht": "QCD multijet HT bin",
    "qcdb-ht": "QCD multijet b-enriched HT bin",
    "hh4b": r"ggHH",
    "hh4b-kl2p45": r"ggHH, $\kappa_{\lambda}=2.45$",
    "hh4b-kl5": r"ggHH, $\kappa_{\lambda}=5$",
    "hh4b-kl0": r"ggHH, $\kappa_{\lambda}=0$",
    "vbfhh4b": r"qqHH",
    "vbfhh4b-k2v0": r"qqHH, $\kappa_{2V}=0$",
    "vbfhh4b-k2v2": r"qqHH, $\kappa_{2V}=2$",
    "vbfhh4b-kl2": r"qqHH, $\kappa_{\lambda}=2$)",
    "zz": "ZZ",
    "nozzdiboson": "Other VV",
    "diboson": "VV",
    "dibosonvjets": "V+jets, VV",
    "ttbar": r"$t\bar{t}$+jets",
    "ttlep": r"$t\bar{t}$+jets (lep)",
    "vjets": r"$V$+jets",
    "vjetslnu": r"$V(\ell\nu/\ell\ell)$+jets",
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
    "zz",
    "nozzdiboson",
    "vjets",
    "vjetslnu",
    "ttbar",
    "qcd",
]


def ratio_uncertainty_fix_zeros(
    num: np.typing.NDArray[Any],
    denom: np.typing.NDArray[Any],
    uncertainty_type: Literal["poisson", "poisson-ratio", "efficiency"] = "poisson",
) -> Any:
    # compute ratio uncertainty, handling zero numerator case

    ratio_uncert = ratio_uncertainty(num, denom)

    if uncertainty_type == "poisson":
        ratio_uncert[:, num == 0] = np.abs(poisson_interval(num[num == 0]) / denom[num == 0])

    return ratio_uncert


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
    h_uncorr: Hist = None,
):
    _, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    nom = h[f"{sig_key}", :].values()
    hep.histplot(
        h[f"{sig_key}", :],
        histtype="step",
        label=sig_key,
        yerr=np.sqrt(h[f"{sig_key}", :].variances()),
        color="k",
        ax=ax,
        linewidth=2,
    )

    if h_uncorr:
        hep.histplot(
            h_uncorr[f"{sig_key}", :],
            histtype="step",
            label=f"{sig_key} No corr.",
            yerr=False,
            color="r",
            linestyle="--",
            ax=ax,
            linewidth=1,
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
            linewidth=1,
        )

        hep.histplot(
            h[f"{sig_key}_{wshift}_{shift}", :] / nom,
            yerr=False,
            histtype="step",
            label=f"{sig_key} {skey}",
            color=colour,
            ax=rax,
        )

    if h_uncorr:
        hep.histplot(
            h[f"{sig_key}", :] / nom,
            yerr=False,
            histtype="step",
            label=f"{sig_key}",
            color="k",
            ax=rax,
        )
        hep.histplot(
            h_uncorr[f"{sig_key}", :] / nom,
            yerr=np.sqrt(h_uncorr[f"{sig_key}", :].variances()) / nom,
            histtype="step",
            label=f"{sig_key} No corr.",
            color="r",
            linestyle="--",
            ax=rax,
            linewidth=1,
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
    sig_ls = OrderedDict()
    for sig_key, sig_scale in sig_scale_dict.items():
        label = label_by_sample.get(sig_key, sig_key)

        if sig_scale == 1:
            label = label  # noqa: PLW0127
        elif "vbfhh4b" in sig_key:
            label = f"{label}, $\\mu_{{qqHH}} = {sig_scale:.0f}$"
        else:
            label = f"{label}, $\\mu_{{ggHH}} = {sig_scale:.0f}$"

        sig_labels[sig_key] = label
        sig_ls[sig_key] = "--" if sig_key == "vbfhh4b-k2v0" else "-"

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

    return bg_keys, bg_colours, bg_labels, sig_colours, sig_scale_dict, sig_labels, sig_ls


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
    ylim: float | None = None,
    ylim_low: float | None = None,
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
    prefit_hists: Hist | None = None,
    reweight_qcd: bool = False,
    qcd_norm: float = None,
    save_pdf: bool = True,
    unblinded: bool = False,
    ratio: Hist | None = None,
    ratio_err: ArrayLike | None = None,
    ratio_label: str = "Data/Pred",
    cms_label: str | None = None,
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
        xlim (optional): x-limit on plot
        xlim_low (optional): x-limit low on plot
        ylim (optional): y-limit on plot
        ylim_low (optional): y-limit low on plot
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
        reweight_qcd (bool): reweight qcd process to agree with data-othermc
        qcd_norm (float): normalization to reweight qcd process, if not None
    """

    # copy hists and bg_keys so input objects are not changed
    hists, bg_keys = deepcopy(hists), deepcopy(bg_keys)
    prefit_hists = deepcopy(prefit_hists) if prefit_hists is not None else None

    if bg_order is None:
        bg_order = bg_order_default

    bg_keys, bg_colours, bg_labels, sig_colours, sig_scale_dict, sig_labels, sig_ls = (
        _process_samples(sig_keys, bg_keys, sig_scale_dict, syst, variation, bg_order)
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
        _fig, (ax, rax, sax) = plt.subplots(
            3,
            1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.15},
            sharex=True,
        )
    else:
        _fig, (ax, rax) = plt.subplots(
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
    if np.allclose(hists.axes[1].widths, hists.axes[1].widths[0]):
        ax.set_ylabel(f"Events / {hists.axes[1].edges[1] - hists.axes[1].edges[0]:.0f} GeV")
    else:
        ax.set_ylabel("Events / bin")

    # re-weight qcd
    kfactor = dict.fromkeys(bg_keys, 1)
    if reweight_qcd and qcd_norm is None:
        non_qcd_hists = [hists[sample, :] for sample in bg_keys if sample != "qcd"]
        non_qcd_yield = np.sum(sum(non_qcd_hists).values()) if len(non_qcd_hists) else 0.0
        qcd_yield = np.sum(hists["qcd", :].values()) if "qcd" in hists.axes[0] else 0.0
        data_yield = np.sum(hists[data_key, :].values())
        if qcd_yield > 0:
            kfactor["qcd"] = (data_yield - non_qcd_yield) / qcd_yield
        print("kfactor ", kfactor["qcd"], qcd_norm)
    elif reweight_qcd:
        kfactor["qcd"] = qcd_norm
    else:
        kfactor["qcd"] = 1.0

    # background samples
    if len(bg_keys) > 0:
        hep.histplot(
            [hists[sample, :] * kfactor[sample] for sample in bg_keys],
            ax=ax,
            # yerr=[np.sqrt(hists[sample, :].variances()) for sample in bg_keys],
            histtype="fill",
            sort="yield" if sortyield else None,
            stack=True,
            linewidth=2,
            label=bg_labels,
            color=bg_colours,
            # flow="none",
        )

    # signal samples
    if len(sig_scale_dict) and sum(np.abs(list(sig_scale_dict.values()))) > 0:
        # use prefit hists for signal if provided
        sig_hists = prefit_hists if prefit_hists is not None else hists
        hep.histplot(
            [sig_hists[sig_key, :] * sig_scale for sig_key, sig_scale in sig_scale_dict.items()],
            ax=ax,
            histtype="step",
            linewidth=3,
            label=list(sig_labels.values()),
            color=sig_colours,
            ls=list(sig_ls.values()),
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
                label=r"$\sigma_{Pred}$",
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

    bg_err_tot_mcstat = None
    if bg_err_mcstat:
        bg_err_label = (
            "Stat. MC Uncertainty (excl. Multijet)"
            if exclude_qcd_mcstat
            else "Stat. MC Uncertainty"
        )

        # this version has an issue:
        # bg_tot no longer weighted, returns None for variances
        # bg_tot = sum([hists[sample, :] for sample in bg_keys])
        # bg_err_tot_mcstat = np.sqrt(bg_tot.variances())
        # compute summed variance manually
        bg_err_tot_mcstat = np.sqrt(sum([hists[sample, :].variances() for sample in bg_keys]))
        # print("mcstat ",bg_err_tot_mcstat)

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
                        color="black",
                        label=bg_err_label,
                        xerr=True,
                    )
                else:
                    hep.histplot(
                        stack,
                        ax=ax,
                        yerr=yerr,
                        histtype="errorbar",
                        markersize=0,
                        color="black",
                        xerr=True,
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
    ax.legend(handles, labels, loc="upper right", fontsize=28, ncol=1)
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

        if ratio is not None and ratio_err is not None:
            yvalue = ratio
            yerr = ratio_err
        else:
            tot_val = bg_tot.values()
            tot_val_zero_mask = tot_val == 0
            tot_val[tot_val_zero_mask] = 1
            data_val = hists[data_key, :].values()
            data_val[tot_val_zero_mask] = 1
            yerr = ratio_uncertainty_fix_zeros(data_val, tot_val, "poisson")
            yvalue = data_val / tot_val

        if prefit_hists:
            bg_tot_prefit = sum([prefit_hists[sample, :] * kfactor[sample] for sample in bg_keys])

            tot_val_prefit = bg_tot_prefit.values()
            tot_val_zero_mask_prefit = tot_val_prefit == 0
            tot_val_prefit[tot_val_zero_mask_prefit] = 1
            yerr_prefit = ratio_uncertainty_fix_zeros(data_val, tot_val_prefit, "poisson")
            yvalue_prefit = data_val / tot_val_prefit

            hep.histplot(
                yvalue_prefit,
                bg_tot.axes[0].edges,
                yerr=yerr_prefit,
                ax=rax,
                histtype="errorbar",
                markeredgecolor="red",
                markersize=12,
                markerfacecolor="none",
                marker="s",
                color="red",
                xerr=False,
                elinewidth=2,
                capsize=0,
                label="Prefit",
            )

        hep.histplot(
            yvalue,
            bg_tot.axes[0].edges,
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            markersize=20,
            color="black",
            xerr=False,
            capsize=0,
            label="Postfit",
        )
        rax.set_xlabel(hists.axes[1].label)

        if prefit_hists:
            rax.legend(loc="best", fontsize=20, ncol=2)

        # fill error band of background
        if bg_err is not None:
            # (bkg + err) / bkg
            rax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat((bg_err[0].values()) / tot_val, 2),
                np.repeat((bg_err[1].values()) / tot_val, 2),
                color="#cccccc",
                alpha=1,
                hatch="//",
                linewidth=0,
            )
        if bg_err_tot_mcstat is not None:
            ax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat((bg_err_tot_mcstat) / tot_val, 2),
                np.repeat((bg_err_tot_mcstat) / tot_val, 2),
                color="#cccccc",
                alpha=1,
                hatch="//",
                linewidth=0,
            )
    else:
        rax.set_xlabel(hists.axes[1].label)

    rax.set_ylabel(ratio_label)
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

        if not unblinded:
            # blind!
            yhist[5] = 0
            yhist[6] = 0
            yhist[7] = 0

        hep.histplot(
            yhist,
            ax=sax,
            # yerr=yerr,
            histtype="fill",
            facecolor="#cccccc",
            edgecolor="k",
        )
        sax.set_ylim([-2, 2])
        sax.set_xlabel(hists.axes[1].label)
        sax.set_ylabel(r"$\frac{Data - Pred}{\sigma_{Data}}$")

        minor_locator = mticker.AutoMinorLocator(2)
        sax.yaxis.set_minor_locator(minor_locator)
        sax.grid(axis="y", linestyle="-", linewidth=2, which="both")

    # if title is not None:
    #     ax.set_title(title, y=1.08)

    if year == "all":
        hep.cms.label(
            cms_label,
            data=True,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}",
            year=None,
            ax=ax,
            com=energy,
            loc=1,
        )
    else:
        hep.cms.label(
            cms_label,
            fontsize=24,
            data=True,
            lumi=f"{LUMI[year] / 1e3:.0f}",
            year=None,
            ax=ax,
            com=energy,
            loc=1,
        )

    # add title (region label) below the CMS label
    if title is not None:
        x_text = 0.02
        y_text = 0.78
        ax.text(x_text + 0.03, y_text + 0.06, title, fontsize=24, transform=ax.transAxes)

    if axrax is None and len(name):
        if not name.endswith((".pdf", ".png")):
            if save_pdf:
                plt.savefig(f"{name}.pdf", bbox_inches="tight")
            plt.savefig(f"{name}.png", bbox_inches="tight")
        else:
            plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return kfactor.get("qcd", 1.0)


def subtractedHistPlot(
    hists: Hist,
    hists_fail: Hist,
    year: str,
    bg_keys: list[str],
    bg_err: ArrayLike = None,
    sortyield: bool = False,
    title: str | None = None,
    name: str = "",
    xlim: int | None = None,
    xlim_low: int | None = None,
    ylim: int | None = None,
    ylim_low: int | None = None,
    show: bool = True,
    bg_err_type: str = "shaded",
    plot_data: bool = True,
    bg_order=None,
    log: bool = False,
    logx: bool = False,
    ratio_ylims: list[float] | None = None,
    energy: str = "13.6",
    cms_label: str | None = None,
):
    """
    Makes and saves subtracted histogram plot, to show QCD transfer factor
    with a data/mc ratio plot below

    Args:
        hists (Hist): input histograms per sample in pass region to plot
        hists_fail (Hist): input histograms per sample in fail region to plot
        year (str): datataking year
        bg_keys (List[str]): background keys
        title (str, optional): plot title. Defaults to None.
        name (str): name of file to save plot
        sig_scale_dict (Dict[str, float]): if scaling signals in the plot, dictionary of factors
          by which to scale each signal
        xlim_low (optional): x-limit low on plot
        ylim (optional): y-limit on plot
        show (bool): show plots or not
        bg_order (List[str]): order in which to plot backgrounds
        ratio_ylims (List[float]): y limits on the ratio plots
        plot_significance (bool): plot Asimov significance below ratio plot
    """

    # copy hists and bg_keys so input objects are not changed
    hists, bg_keys = deepcopy(hists), deepcopy(bg_keys)

    if bg_order is None:
        bg_order = bg_order_default

    bg_keys, bg_colours, _bg_labels, _, _, _, _ = _process_samples(
        [], bg_keys, {}, None, None, bg_order
    )

    _fig, (ax, rax) = plt.subplots(
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
    ax.set_ylabel("SR / QCD multijet in CR")

    # background samples
    hep.histplot(
        hists["qcd", :] / hists_fail["qcd", :],
        ax=ax,
        histtype="fill",
        sort="yield" if sortyield else None,
        stack=True,
        edgecolor="black",
        linewidth=2,
        label="QCD multijet",
        color=bg_colours[-1],
    )

    if bg_err is not None:
        bg_tot = hists["qcd", :] / hists_fail["qcd", :]
        if len(np.array(bg_err).shape) == 1:
            bg_errs = [
                bg_tot - bg_err / hists_fail["qcd", :],
                bg_tot + bg_err / hists_fail["qcd", :],
            ]

        if bg_err_type == "shaded":
            ax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat(bg_errs[0].values(), 2),
                np.repeat(bg_errs[1].values(), 2),
                color="black",
                alpha=0.2,
                hatch="//",
                linewidth=0,
                label=r"$\sigma_{Pred}$",
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
                bg_errs[0],
                hists.axes[1].edges,
                color="red",
                linewidth=3,
                label="Bkg. Down",
                baseline=bg_errs[0],
            )

            ax.stairs(
                bg_err[1],
                hists.axes[1].edges,
                color="#7F2CCB",
                linewidth=3,
                label="Bkg. Up",
                baseline=bg_err[1],
            )

    # plot data
    if plot_data:
        data_val = hists[data_key, :].values()
        qcd_fail_val = hists_fail["qcd", :].values()
        yerr = ratio_uncertainty_fix_zeros(data_val, qcd_fail_val, "poisson")
        all_mc = sum(hists[bg_key, :] for bg_key in bg_keys if bg_key != "qcd")
        yvalue = (hists[data_key, :] - all_mc) / hists_fail["qcd", :]
        hep.histplot(
            yvalue,
            ax=ax,
            yerr=yerr,
            histtype="errorbar",
            label=r"$Data-Others$",
            markersize=20,
            color="black",
        )

    if log:
        ax.set_yscale("log")
        rax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
        rax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-1:] + handles[:-1]
    labels = labels[-1:] + labels[:-1]
    ax.legend(handles, labels, loc="upper right", fontsize=28, ncol=1)

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
        qcd_val = hists["qcd", :].values()
        hep.histplot(
            yvalue / (qcd_val / qcd_fail_val),
            yerr=ratio_uncertainty_fix_zeros(data_val, qcd_val, "poisson"),
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
                np.repeat((bg_errs[0].values()) / (qcd_val / qcd_fail_val), 2),
                np.repeat((bg_errs[1].values()) / (qcd_val / qcd_fail_val), 2),
                color="black",
                alpha=0.1,
                hatch="//",
                linewidth=0,
            )
    else:
        rax.set_xlabel(hists.axes[1].label)

    rax.set_ylabel("Ratio")
    rax.set_ylim(ratio_ylims)
    # minor_locator = mticker.AutoMinorLocator(2)
    # rax.yaxis.set_minor_locator(minor_locator)
    # rax.grid(axis="y", linestyle="-", linewidth=2, which="both")

    # add title (region label) below the CMS label
    if title is not None:
        x_text = 0.02
        y_text = 0.78
        ax.text(x_text + 0.03, y_text + 0.06, title, fontsize=24, transform=ax.transAxes)

    if year == "all":
        hep.cms.label(
            cms_label,
            loc=1,
            data=True,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}",
            year=None,
            ax=ax,
            com=energy,
        )
    else:
        hep.cms.label(
            cms_label,
            loc=1,
            fontsize=24,
            data=True,
            lumi=f"{LUMI[year] / 1e3:.0f}",
            year=None,
            ax=ax,
            com=energy,
        )

    if len(name):
        if not name.endswith((".pdf", ".png")):
            plt.savefig(f"{name}.pdf", bbox_inches="tight")
            plt.savefig(f"{name}.png", bbox_inches="tight")
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
    sig_effs: list[float] = None,
    bkg_effs: list[float] = None,
    xlim=None,
    ylim=None,
    plot_dir: Path = None,
    name: str = "",
    show: bool = False,
    add_cms_label=False,
    legtitle: str = None,
    title: str = None,
    plot_thresholds: dict = None,  # plot signal and bkg efficiency for a given discriminator threshold
    find_from_sigeff: dict = None,  # find discriminator threshold that matches signal efficiency
):
    """Plot multiple ROC curves (e.g. train and test) + multiple signals"""
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    line_style = {"colors": "lightgrey", "linestyles": "dashed"}
    th_colours = [
        "cornflowerblue",
        "deepskyblue",
        "mediumblue",
        "cyan",
        "cadetblue",
        "plum",
        "purple",
        "palevioletred",
    ]
    eff_colours = ["lime", "aquamarine", "greenyellow"]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    for roc_sigs in rocs.values():

        # plots roc curves for each type of signal
        for roc in roc_sigs.values():

            plt.plot(
                roc["tpr"],
                roc["fpr"],
                label=roc["label"],
                color=roc["color"],
                linewidth=2,
            )

            # determines the point on the ROC curve that corresponds to the signal efficiency
            # plots a vertical and horizontal line to the point
            if sig_effs is not None:
                for sig_eff in sig_effs:
                    y = roc["fpr"][np.searchsorted(roc["tpr"], sig_eff)]
                    plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
                    plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

            # determines the point on the ROC curve that corresponds to the background efficiency
            # plots a vertical and horizontal line to the point
            if bkg_effs is not None:
                for bkg_eff in bkg_effs:
                    x = roc["tpr"][np.searchsorted(roc["fpr"], bkg_eff)]
                    plt.vlines(x=x, ymin=0, ymax=bkg_eff, **line_style)
                    plt.hlines(y=bkg_eff, xmin=0, xmax=x, **line_style)

    # plots points and lines on plot corresponding to classifier thresholds
    for roc_sigs in rocs.values():
        i_sigeff = 0
        i_th = 0
        for rockey, roc in roc_sigs.items():
            if rockey in plot_thresholds:
                pths = {th: [[], []] for th in plot_thresholds[rockey]}
                for th in plot_thresholds[rockey]:
                    idx = _find_nearest(roc["thresholds"], th)
                    pths[th][0].append(roc["tpr"][idx])
                    pths[th][1].append(roc["fpr"][idx])
                for th in plot_thresholds[rockey]:
                    plt.scatter(
                        *pths[th],
                        marker="o",
                        s=40,
                        label=rf"{rockey} > {th:.2f}",
                        zorder=100,
                        color=th_colours[i_th],
                    )
                    plt.vlines(
                        x=pths[th][0],
                        ymin=0,
                        ymax=pths[th][1],
                        color=th_colours[i_th],
                        linestyles="dashed",
                        alpha=0.5,
                    )
                    plt.hlines(
                        y=pths[th][1],
                        xmin=0,
                        xmax=pths[th][0],
                        color=th_colours[i_th],
                        linestyles="dashed",
                        alpha=0.5,
                    )
                    i_th += 1

            if find_from_sigeff is not None and rockey in find_from_sigeff:
                pths = {sig_eff: [[], []] for sig_eff in find_from_sigeff[rockey]}
                thrs = {}
                for sig_eff in find_from_sigeff[rockey]:
                    idx = _find_nearest(roc["tpr"], sig_eff)
                    thrs[sig_eff] = roc["thresholds"][idx]
                    pths[sig_eff][0].append(roc["tpr"][idx])
                    pths[sig_eff][1].append(roc["fpr"][idx])
                for sig_eff in find_from_sigeff[rockey]:
                    plt.scatter(
                        *pths[sig_eff],
                        marker="o",
                        s=40,
                        label=rf"{rockey} > {thrs[sig_eff]:.2f}",
                        zorder=100,
                        color=eff_colours[i_sigeff],
                    )
                    plt.vlines(
                        x=pths[sig_eff][0],
                        ymin=0,
                        ymax=pths[sig_eff][1],
                        color=eff_colours[i_sigeff],
                        linestyles="dashed",
                        alpha=0.5,
                    )
                    plt.hlines(
                        y=pths[sig_eff][1],
                        xmin=0,
                        xmax=pths[sig_eff][0],
                        color=eff_colours[i_sigeff],
                        linestyles="dashed",
                        alpha=0.5,
                    )
                    i_sigeff += 1

    if add_cms_label:
        hep.cms.label(data=False, rlabel="")
    if title:
        plt.title(title)
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    if legtitle:
        plt.legend(title=legtitle, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if len(name):
        plt.savefig(plot_dir / f"{name}.png", bbox_inches="tight")

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
        plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fom(h_sb, plot_dir, name="figofmerit", show=False, fontsize=3.5, label="Fig Of Merit"):
    """Plot FoM scan"""

    eff, bins_x, bins_y = h_sb.to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plt.rcParams.update({"font.size": 18})

    cbar = hep.hist2dplot(
        h_sb, ax=ax, cmin=np.min(eff[eff > 0]) * 0.75, cmax=np.max(eff[eff > 0]) * 1.25, flow="none"
    )
    cbar.cbar.set_label(label, size=18)
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
                    fontsize=fontsize,
                )

    ax.set_xlabel("BDT Cut")
    ax.set_ylabel(r"$T_{Xbb}$ Cut")
    ax.set_ylim(bins_y[0], bins_y[-1])
    ax.set_xlim(bins_x[0], bins_x[-1])
    fig.tight_layout()
    plt.savefig(f"{plot_dir}/{name}.png", bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
