"""Draw the analysis category layout in the (TXbb, BDT) plane.

Two panels: the ggF plane (x = ggHH-vs-bkg BDT, y = TXbb(H2)) with the SR1 corner,
the L-shaped SR2, the SR3 staircase and the fail / QCD CR band below the TXbb floor,
and the VBF plane (x = qqHH-vs-bkg BDT) with the qqHH SR strip. The fail region is
TXbb < floor in either plane, so the QCD CR band (and its upper boundary at the
floor) is drawn in both. The BDT axes are compressed below a configurable zoom
start, with a break mark, so the signal-region cuts stay readable.

The working-point flags mirror PostProcess.py, so the diagram can be produced from
the same values used for template production, e.g. for the re-optimized selection:

    python PlotCategories.py --txbb-wps 0.9425 0.81 --bdt-wps 0.9875 0.89 0.03 \
        --vbf-txbb-wp 0.8975 --vbf-bdt-wp 0.995 --out categories_reopt.png
"""

from __future__ import annotations

import argparse

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

COLORS = {
    "sr1": "#C8380D",
    "sr2": "#3C77B4",
    "sr3": "#F5A623",
    "cr": "#C9CBCD",
    "crlight": "#E4E6E8",
    "vbf": "#8E5BA6",
}


def _tf(xlo, xbrk):
    """Piecewise x transform: [xbrk, xlo] compressed into the left 10% of the axis."""

    def f(x):
        if x <= xlo:
            return 0.10 * (x - xbrk) / (xlo - xbrk)
        return 0.10 + 0.90 * (x - xlo) / (1.0 - xlo)

    return f


def _rect(ax, f, x0, x1, y0, y1, color, label=None, tcol="white", fs=14, rot=None):
    ax.add_patch(
        Rectangle((f(x0), y0), f(x1) - f(x0), y1 - y0, facecolor=color, edgecolor="white", lw=1.6)
    )
    if label:
        if rot is None:
            rot = 90 if (f(x1) - f(x0)) < 0.12 else 0
        ax.text(
            (f(x0) + f(x1)) / 2,
            (y0 + y1) / 2,
            label,
            ha="center",
            va="center",
            color=tcol,
            fontsize=fs,
            fontweight="bold",
            rotation=rot,
        )


def _style(ax, f, ylo, xticks, yticks, xlabel, bold_x, bold_y):
    # drop the trailing 1.0 tick when it would collide with a cut tick
    xticks = [x for x in xticks if x != 1.0 or f(1.0) - max(f(v) for v in xticks if v < 1.0) > 0.08]
    yticks = sorted(set(yticks))
    ax.set_xlim(0, 1)
    ax.set_ylim(ylo, 1)
    ax.set_xticks([f(x) for x in xticks])
    ax.set_xticklabels([f"{x:g}" for x in xticks], fontsize=12)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}" for y in yticks], fontsize=12)
    for lab, x in zip(ax.get_xticklabels(), xticks):
        if x in bold_x:
            lab.set_fontweight("bold")
    for lab, y in zip(ax.get_yticklabels(), yticks):
        if y in bold_y:
            lab.set_fontweight("bold")
    ax.set_xlabel(xlabel, fontsize=14)
    for dx in (-0.008, 0.008):  # axis break mark where the compressed span ends
        ax.plot(
            [0.10 + dx - 0.006, 0.10 + dx + 0.006],
            [-0.018, 0.018],
            transform=ax.get_xaxis_transform(),
            color="black",
            lw=1.2,
            clip_on=False,
        )
    ax.tick_params(direction="out", length=4)


def _ggf_panel(ax, args, zoom):
    t1, t2 = args.txbb_wps
    b1, b2, xbrk = args.bdt_wps
    f = _tf(zoom, xbrk)
    ylo = args.txbb_lo
    _rect(ax, f, xbrk, 1, ylo, t2, COLORS["cr"], "QCD CR (fail)", tcol="#444444")
    # SR3: TXbb > floor, BDT > fail WP, minus SR1/SR2 (staircase polygon)
    pts = [(xbrk, t2), (b1, t2), (b1, t1), (b2, t1), (b2, 1), (xbrk, 1)]
    ax.add_patch(
        Polygon([(f(x), y) for x, y in pts], facecolor=COLORS["sr3"], edgecolor="white", lw=1.6)
    )
    ax.text(
        f(b2) * 0.5,
        (t2 + 1) / 2,
        "SR 3",
        ha="center",
        va="center",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    _rect(ax, f, b2, b1, t1, 1, COLORS["sr2"], "SR 2")  # top strip left of SR1
    _rect(ax, f, b1, 1, t2, t1, COLORS["sr2"], "SR 2")  # right column below SR1
    _rect(ax, f, b1, 1, t1, 1, COLORS["sr1"], "SR 1")
    _style(
        ax,
        f,
        ylo,
        [xbrk, zoom, b2, b1, 1.0],
        [ylo, t2, t1, 1.0],
        "BDT (ggHH vs bkg)",
        {b2, b1},
        {t2, t1},
    )


def _vbf_panel(ax, args, zoom):
    vt, vb = args.vbf_txbb_wp, args.vbf_bdt_wp
    t2 = args.txbb_wps[1]  # TXbb fail floor, shared with the ggF plane
    xbrk = args.bdt_wps[2]
    f = _tf(zoom, xbrk)
    ylo = args.txbb_lo
    _rect(ax, f, xbrk, 1, ylo, t2, COLORS["cr"], "QCD CR (fail)", tcol="#444444")
    _rect(ax, f, xbrk, 1, t2, 1, COLORS["crlight"])
    ax.text(
        f(zoom + (vb - zoom) * 0.5),
        (t2 + 1) / 2,
        "ggHH SR 1-3\n(assigned in the ggF plane)",
        ha="center",
        va="center",
        color="#666666",
        fontsize=12,
        style="italic",
    )
    _rect(ax, f, vb, 1, vt, 1, COLORS["vbf"], "qqHH SR", fs=12)
    _style(
        ax,
        f,
        ylo,
        [xbrk, zoom, vb, 1.0],
        [ylo, t2, vt, 1.0],
        "BDT$_{\\mathrm{VBF}}$ (qqHH vs bkg)",
        {vb},
        {t2, vt},
    )


def plot_categories(args):
    zoom = (
        args.zoom
        if args.zoom is not None
        else max(args.bdt_wps[2], round(args.bdt_wps[1] - 0.04, 3))
    )
    vzoom = (
        args.vbf_zoom
        if args.vbf_zoom is not None
        else round(args.vbf_bdt_wp - 2.5 * (1 - args.vbf_bdt_wp), 3)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), width_ratios=[1.35, 1])
    _ggf_panel(axes[0], args, zoom)
    _vbf_panel(axes[1], args, vzoom)
    axes[0].set_ylabel("T$_{\\mathrm{Xbb}}$(H2)", fontsize=14)
    axes[0].set_title("ggF categories", fontsize=15, pad=8)
    axes[1].set_title("VBF category", fontsize=15, pad=8)
    if args.title:
        fig.suptitle(args.title, fontsize=16, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--txbb-wps",
        type=float,
        nargs=2,
        required=True,
        help="TXbb Bin 1, Bin 2 WPs (Bin 2 is also the SR3 / fail floor)",
    )
    parser.add_argument(
        "--bdt-wps",
        type=float,
        nargs=3,
        required=True,
        help="BDT Bin 1, Bin 2, Fail WPs (the Fail WP is the lower BDT edge of SR3 and the CR)",
    )
    parser.add_argument("--vbf-txbb-wp", type=float, required=True, help="TXbb VBF WP")
    parser.add_argument("--vbf-bdt-wp", type=float, required=True, help="BDT VBF WP")
    parser.add_argument(
        "--zoom",
        type=float,
        default=None,
        help="ggF BDT value where the linear axis starts (default: Bin 2 WP - 0.04)",
    )
    parser.add_argument(
        "--vbf-zoom",
        type=float,
        default=None,
        help="VBF BDT value where the linear axis starts (default: sized so the qqHH "
        "strip spans about a quarter of the panel)",
    )
    parser.add_argument("--txbb-lo", type=float, default=0.75, help="lower edge of the TXbb axis")
    parser.add_argument("--title", default=None, help="optional figure title")
    parser.add_argument("--out", default="categories.png", help="output image path")
    parser.add_argument("--dpi", type=int, default=300)
    plot_categories(parser.parse_args())


if __name__ == "__main__":
    main()
