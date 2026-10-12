"""Plot per-systematic template shape variations (signal up/down vs nominal).

Migrates the driver logic from ``CombineTemplates.ipynb`` into a reusable
function (:func:`make_shape_plots`) plus a script entry point. For each region and
systematic shift it combines the per-year templates with
``postprocessing.combine_templates`` and draws the variation with
``plotting.sigErrRatioPlot``.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

from HH4b import plotting, run_utils
from HH4b.hh_vars import jecs, jmsr
from HH4b.postprocessing import (
    Region,
    combine_templates,
    get_shape_systematics,
    get_weight_shifts,
    shift_available,
)

# repo root of this checkout (…/src/HH4b/postprocessing/PlotShapes.py -> root)
HH4B_DIR = Path(__file__).resolve().parents[3]


def _year_label(years: list[str]) -> str:
    """Compact label for a set of years, e.g. ``2022-2025`` (single year -> itself)."""
    prefixes = sorted({str(year)[:4] for year in years})
    return prefixes[0] if len(prefixes) == 1 else f"{prefixes[0]}-{prefixes[-1]}"


def _templates_repo_root(templates_dir: str | Path) -> Path:
    """Repo root the templates belong to (the ancestor containing ``src/HH4b``), so
    plots land alongside the templates regardless of where the code runs from."""
    for ancestor in Path(templates_dir).resolve().parents:
        if (ancestor / "src" / "HH4b").is_dir():
            return ancestor
    return HH4B_DIR


def default_shape_plot_dir(templates_dir: str | Path, years: list[str]) -> Path:
    """Standard output dir for shape-systematic plots, derived from the template-set
    name: ``<repo>/plots/PostProcess/<tag>/Templates/<year_label>/wshifts``."""
    tag = Path(templates_dir).name
    return (
        _templates_repo_root(templates_dir)
        / "plots"
        / "PostProcess"
        / tag
        / "Templates"
        / _year_label(years)
        / "wshifts"
    )


def make_shape_plots(
    templates: dict[str, dict],
    years: list[str],
    selection_regions: dict[str, Region],
    sig_key: str,
    weight_shifts: dict,
    plot_dir: str | Path,
    xlabel: str,
    shifts: list[str] | None = None,
    ylim: list | None = None,
    show: bool = False,
):
    """Draw nominal vs up/down shape variations for each region and shift.

    Args:
        templates: ``{year: {region_key: Hist}}`` as loaded from the per-year pickles.
        years: years to sum over.
        selection_regions: ``{region_name: Region}`` to plot.
        sig_key: sample whose variations are drawn (e.g. ``"hh4b"``).
        weight_shifts: output of ``get_weight_shifts`` (used for labels and, when
            ``shifts`` is ``None``, to enumerate applicable systematics).
        plot_dir: output directory (created if missing).
        xlabel: x-axis label for the mass variable.
        shifts: explicit list of shift names; if ``None`` they are enumerated with
            ``get_shape_systematics`` for ``sig_key``.
        ylim: y-limits for the ratio panel.
        show: whether to display each figure.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if shifts is None:
        shifts = get_shape_systematics(weight_shifts, sig_key)

    for rname, region in selection_regions.items():
        for shift in shifts:
            if not shift_available(templates, years, rname, shift, sig_key):
                warnings.warn(
                    f"skipping '{shift}' for region '{rname}': shifted templates not found",
                    stacklevel=2,
                )
                continue
            h = combine_templates(templates, years, rname, shift)
            is_jshift = shift in jecs or shift in jmsr
            label = shift if is_jshift else weight_shifts[shift].label
            plotting.sigErrRatioPlot(
                h,
                sig_key,
                shift,
                xlabel,
                f"{region.label} Region {label} Variations",
                plot_dir,
                f"{rname}_sig_{shift}",
                show=show,
                ylim=ylim,
            )


def _default_selection_regions() -> dict[str, Region]:
    return {
        "pass_bin1": Region(cuts={"Category": [1, 2]}, label="Bin1"),
        "pass_bin2": Region(cuts={"Category": [2, 3]}, label="Bin2"),
        "pass_bin3": Region(cuts={"Category": [3, 4]}, label="Bin3"),
        "fail": Region(cuts={"Category": [4, 5]}, label="Fail"),
    }


def plot_shapes(args):
    templates_path = Path(args.templates_dir)
    templates = {}
    for year in args.years:
        with (templates_path / f"{year}_templates.pkl").open("rb") as f:
            templates[year] = pickle.load(f)

    weight_shifts = get_weight_shifts(args.txbb_version, args.bdt_version)

    selection_regions = _default_selection_regions()
    if args.regions != "all":
        selection_regions = {args.regions: selection_regions[args.regions]}

    plots_dir = (
        Path(args.plots_dir)
        if args.plots_dir
        else default_shape_plot_dir(args.templates_dir, args.years)
    )
    print(f"Saving shape plots to {plots_dir}")

    make_shape_plots(
        templates,
        args.years,
        selection_regions,
        args.sig_key,
        weight_shifts,
        plot_dir=plots_dir,
        xlabel=args.xlabel,
        shifts=args.shifts,
        ylim=args.ratio_ylims,
        show=args.show,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates-dir", required=True, type=str, help="per-year templates dir")
    parser.add_argument(
        "--plots-dir",
        default=None,
        type=str,
        help="output plots dir (default: <repo>/plots/PostProcess/<tag>/Templates/<years>/wshifts)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
        help="years to combine",
    )
    parser.add_argument(
        "--regions",
        default="all",
        choices=["pass_bin1", "pass_bin2", "pass_bin3", "fail", "all"],
        help="region(s) to plot",
    )
    parser.add_argument("--sig-key", default="hh4b", type=str, help="signal sample to plot")
    parser.add_argument("--txbb-version", default="glopart-v2", type=str)
    parser.add_argument("--bdt-version", default="25Feb5_v13_glopartv2_rawmass", type=str)
    parser.add_argument(
        "--shifts",
        nargs="+",
        default=None,
        help="explicit shift names; default enumerates all applicable to --sig-key",
    )
    parser.add_argument(
        "--xlabel", default=r"$m^\mathrm{reg}_{2}$ (GeV)", type=str, help="mass axis label"
    )
    parser.add_argument(
        "--ratio-ylims", nargs=2, default=[0, 2.5], type=float, help="ratio panel y-limits"
    )
    run_utils.add_bool_arg(parser, "show", default=False, help="show figures")

    args = parser.parse_args()
    plot_shapes(args)
