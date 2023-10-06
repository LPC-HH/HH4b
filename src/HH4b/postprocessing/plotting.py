
from hist.intervals import ratio_uncertainty

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

import matplotlib.ticker as mticker

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

import matplotlib as mpl

mpl.rcParams["font.size"] = 20
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["grid.color"] = "#CCCCCC"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["figure.edgecolor"] = "none"

import os

errps = {
    "hatch": "////",
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
    "gghtobb": "aquamarine",
    "vbfhtobb": "teal",
    "tthtobb": "cadetblue",
    "vhtobb": "tab:cyan",
    "ttbar": "royalblue",
    "qcd": "yellow",
    "diboson": "orchid",
    "vjets": "tab:green",
}

label_by_sample = {
    "gghtobb": "ggH(bb)",
    "vbfhtobb": "VBFH(bb)",
    "tthtobb": "ttH(bb)",
    "vhtobb": "VH(bb)",
    "qcd": "Multijet",
    "diboson": "VV",
    "ttbar": r"$t\bar{t}$+jets",
    "vjets": r"W/Z$(qq)$",
}


def plot_hists(
    year,
    hists,
    vars_to_plot,
):
    # TODO: add data as boolean

    # TODO: get as input (fb)
    luminosity = 20.66

    for var in vars_to_plot:
        if var not in hists.keys():
            print(f"{var} not stored in hists")
            continue

        print(f"Will plot {var} histogram")
        h = hists[var]

        samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]

        data = h[{"Sample": "data"}]

        signals = ["hh4b"]
        signal_labels = [label for label in samples if label in signals]
        bkg_labels = [
            label
            for label in samples
            if (label and label not in signal_labels and (label not in ["data"]))
        ]
        bkg = [h[{"Sample": label}] for label in bkg_labels]

        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07},
            sharex=True,
        )

        # plot data
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
        hep.histplot(
            bkg,
            ax=ax,
            stack=True,
            sort="yield",
            edgecolor="black",
            linewidth=1,
            histtype="fill",
            label=[label_by_sample[bkg_label] for bkg_label in bkg_labels],
            color=[color_by_sample[bkg_label] for bkg_label in bkg_labels],
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

        # plot data/mc ratio
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

        # plot bkg uncertainty
        ax.stairs(
            values=tot.values() + tot_err,
            baseline=tot.values() - tot_err,
            edges=tot.axes[0].edges,
            **errps,
            label="Stat. unc.",
        )

        ax.set_ylabel("Events")
        ax.set_xlabel("")
        rax.set_xlabel(
            f"{h.axes[-1].label}"
        )  # assumes the variable to be plotted is at the last axis
        rax.set_ylabel("Data/MC", fontsize=20)
        hep.cms.lumitext("%.1f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
        hep.cms.text("Work in Progress", ax=ax, fontsize=15)

        # add legend
        handles, labels = ax.get_legend_handles_labels()

        # get total yield of backgrounds per label
        first_key = list(hists.keys())[0]
        # (sort by yield after pre-sel)
        order_dic = {}
        for bkg_label in bkg_labels:
            bkg_yield = hists[first_key][{"Sample": bkg_label}].sum().value
            order_dic[label_by_sample[bkg_label]] = bkg_yield
        print(order_dic)

        summ = [order_dic[label] for label in labels[: len(bkg_labels)]]

        # get indices of labels arranged by yield
        order = []
        for i in range(len(summ)):
            order.append(np.argmax(np.array(summ)))
            summ[np.argmax(np.array(summ))] = -100

        legend_handles = [handles[-1]] + [handles[i] for i in order] + handles[len(bkg) : -1]
        legend_labels = [labels[-1]] + [labels[i] for i in order] + labels[len(bkg) : -1]

        ax.legend(
            [legend_handles[idx] for idx in range(len(legend_handles))],
            [legend_labels[idx] for idx in range(len(legend_labels))],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            # title=
        )

        logy = True
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-1)

        outpath = "plots"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        plt.savefig(f"{outpath}/{var}.pdf", bbox_inches="tight")
