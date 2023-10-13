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
    "gghtobb": "aquamarine",
    "vbfhtobb": "teal",
    "tthtobb": "cadetblue",
    "vhtobb": "tab:cyan",
    "hh4b": "lime",
    "hh4b-kl0": "salmon",
    "hh4b-kl2p45": "brown",
    "hh4b-kl5": "chocolate",
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
    "hh4b": r"HH 4b ($\kappa_{\lambda}=1$)",
    "hh4b-kl2p45": r"HH 4b ($\kappa_{\lambda}=2.45$)",
    "hh4b-kl5": r"HH 4b ($\kappa_{\lambda}=5$)",
    "hh4b-kl0": r"HH 4b ($\kappa_{\lambda}=0$)",
    "diboson": "VV",
    "ttbar": r"$t\bar{t}$+jets",
    "vjets": r"W/Z$(qq)$",
}


def plot_hists(
    year,
    hists,
    vars_to_plot,
    luminosity,  # float (fb)
    add_data=True,
    add_data_over_mc=True,
    mult_factor=1,  # multiplicative factor for signal
    logy=True,
):
    if add_data_over_mc and not add_data:
        add_data_over_mc = False

    for var in vars_to_plot:
        if var not in hists.keys():
            print(f"{var} not stored in hists")
            continue

        print(f"Will plot {var} histogram")
        h = hists[var]

        samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]

        signals = ["hh4b", "hh4b-kl0", "hh4b-kl2p45", "hh4b-kl5"]
        signal_labels = [label for label in samples if label in signals]
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

        hep.histplot(
            # bkg_hists,
            # bkg_bins[0],
            bkg,
            ax=ax,
            stack=True,
            edges=True,
            sort="yield",
            w2method=None,
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

        # plot bkg uncertainty
        # print(tot.values().shape)
        # print(tot.axes[0].edges.shape)
        ax.stairs(
            values=tot.values() + tot_err,
            baseline=tot.values() - tot_err,
            edges=tot.axes[0].edges,
            **errps,
            label="Stat. unc.",
        )

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

        # plot signal
        if len(signal) > 0:
            # tot_signal = None

            for i, sig in enumerate(signal_mult):
                lab_sig_mult = f"{mult_factor} * {label_by_sample[signal_labels[i]]}"
                if mult_factor == 1:
                    lab_sig_mult = f"{label_by_sample[signal_labels[i]]}"
                hep.histplot(
                    sig,
                    ax=ax,
                    label=lab_sig_mult,
                    linewidth=3,
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

        ax.set_ylabel("Events")
        ax.set_xlabel("")

        if rax is not None:
            rax.set_xlabel(
                f"{h.axes[-1].label}"
            )  # assumes the variable to be plotted is at the last axis
            rax.set_ylabel("Data/MC", fontsize=20)
        else:
            ax.set_xlabel(f"{h.axes[-1].label}")

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

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-1)

        outpath = "plots"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        plt.savefig(f"{outpath}/{var}.pdf", bbox_inches="tight")
