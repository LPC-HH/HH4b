from __future__ import annotations

import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import click
import hist
import pandas as pd
from hist import Hist

from HH4b import plotting, utils
from HH4b.hh_vars import (
    bg_keys,
    data_key,
    samples,
    sig_keys,
)

# define ShapeVar (label and bins for a given variable)
from HH4b.utils import CUT_MAX_VAL, ShapeVar, Syst


@dataclass
class Region:
    cuts: dict = None
    label: str = None


weight_shifts = {
    "pileup": Syst(samples=sig_keys + bg_keys, label="Pileup"),
    # "PDFalphaS": Syst(samples=sig_keys, label="PDF"),
    # "QCDscale": Syst(samples=sig_keys, label="QCDscale"),
    # "ISRPartonShower": Syst(samples=sig_keys_ggf + ["vjets"], label="ISR Parton Shower"),
    # "FSRPartonShower": Syst(samples=sig_keys_ggf + ["vjets"], label="FSR Parton Shower"),
    # "top_pt": ["ttbar"],
}

# {label: {cutvar: [min, max], ...}, ...}
txbb_cut = 0.985

selection_regions = {
    "pass": Region(
        cuts={
            "bb0FatJetPNetXbb": [txbb_cut, CUT_MAX_VAL],
            "bb1FatJetPNetXbb": [txbb_cut, CUT_MAX_VAL],
            "bb0FatJetPNetMass": [100, 150],
        },
        label="Pass",
    ),
    "fail": Region(
        cuts={
            "bb0FatJetPNetXbb": [-CUT_MAX_VAL, txbb_cut],
            "bb1FatJetPNetXbb": [-CUT_MAX_VAL, txbb_cut],
            "bb0FatJetPNetMass": [100, 150],
        },
        label="Fail",
    ),
}

fit_shape_var = ShapeVar(
    "bb1FatJetPNetMass",
    r"$m^{j2}_\mathrm{Reg}$ (GeV)",
    [19, 60, 250],
    reg=True,
    blind_window=[100, 150],
)


var_to_shapevar = {
    # var must match key in events dictionary (i.e. as saved in parquet file)
    "DijetMass": ShapeVar(var="DijetMass", label=r"$m^{jj}$ (GeV)", bins=[30, 600, 4000]),
    "ak8FatJetPt0": ShapeVar(
        var="ak8FatJetPt0", label=r"$p_T^0$ (GeV)", bins=[30, 300, 1500], significance_dir="right"
    ),
    "ak8FatJetPt1": ShapeVar(
        var="ak8FatJetPt1", label=r"$p_T^1$ (GeV)", bins=[30, 300, 1500], significance_dir="right"
    ),
    "ak8FatJetPNetMass0": ShapeVar(
        var="ak8FatJetPNetMass0", label=r"$m_{reg}^{0}$ (GeV)", bins=[20, 50, 250]
    ),
    "ak8FatJetPNetXbb0": ShapeVar(
        var="ak8FatJetPNetXbb0",
        label=r"$TX_{bb}^{0}$",
        bins=[50, 0.0, 1],
    ),
}


@click.command()
@click.option(
    "--year",
    "years",
    required=True,
    multiple=True,
    type=click.Choice(["2022", "2022EE", "2023", "2018"], case_sensitive=False),
    help="year",
)
def postprocess(years):
    # TODO: set this as a yaml file
    dirs = {"/eos/uscms/store/user/cmantill/bbbb/skimmer/Oct2/": samples}

    vars_to_plot = [
        "ak8FatJetPt0",
        "ak8FatJetPt1",
        "DijetMass",
        "ak8FatJetPNetXbb0",
    ]

    # weight to apply to histograms
    weight_key = ["finalWeight"]

    # filters are sequences of strings that can be used to place a selection
    # e.g. https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/postprocessing.py#L80
    filters = [
        [
            # [
            #    ("('HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35', '0')", "==", 1),
            #    ("('HLT_AK8PFJet425_SoftDropMass40', '0')", "==", 1),
            # ],
            # ("('HLT_AK8PFJet425_SoftDropMass40', '0')", "==", 1),
            ("('ak8FatJetPt', '0')", ">=", 300),
            ("('ak8FatJetPt', '1')", ">=", 250),
            ("('ak8FatJetMsd', '0')", ">=", 60),
            ("('ak8FatJetMsd', '1')", ">=", 60),
            # ("('ak8FatJetPNetXbb', '0')", ">=", 0.8),
            # ("('ak8FatJetPNetXbb', '1')", ">=", 0.8),
        ],
    ]

    # columns to load
    load_columns = [
        ("weight", 1),
        ("DijetMass", 1),
        ("ak8FatJetPt", 2),
        ("ak8FatJetPNetXbb", 2),
        # "single_weight_trigsf_2jet"
        # ("ak8FatJetPNetMass", 2),
    ]
    # reformat into ("column name", "idx") format for reading multiindex columns
    columns = []
    for key, num_columns in load_columns:
        for i in range(num_columns):
            columns.append(f"('{key}', '{i}')")

    for year in years:
        # load all samples, apply filters if needed
        events_dict = {}
        for input_dir, in_samples in dirs.items():
            events_dict = {
                **events_dict,
                **utils.load_samples(
                    input_dir,
                    in_samples,
                    year,
                    filters,
                    columns,
                    variations=True,
                    weight_shifts=weight_shifts,
                ),
            }

        # samples_loaded = list(events_dict.keys())
        # keys_loaded = list(events_dict[samples_loaded[0]].keys())
        # print(f"Keys in events_dict {keys_loaded}")

        # make a histogram
        hists = {}
        for var in vars_to_plot:
            shape_var = var_to_shapevar[var]
            if shape_var.var not in hists:
                hists[shape_var.var] = utils.singleVarHist(
                    events_dict,
                    shape_var,
                    weight_key=weight_key,
                    selection=None,
                )

        # make a stacked plot
        plotting.plot_hists(
            year,
            hists,
            vars_to_plot,
        )


def _get_fill_data(
    events: pd.DataFrame,
    bb_mask: dict[str, pd.DataFrame],
    shape_vars: list[ShapeVar],
    jshift: str = "",
):
    return {
        shape_var.var: utils.get_feat(
            events,
            shape_var.var if jshift == "" else utils.check_get_jec_var(shape_var.var, jshift),
            bb_mask,
        )
        for shape_var in shape_vars
    }


def bb_assignment(events_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Creates a dataframe of masks for getting leading and sub-leading jets in Txbb score.

    Returns:
        Dict[str, pd.DataFrame]: ``bb_masks`` dict of boolean masks for each sample,
          of shape ``[num_events, 2]``.

    """
    bb_masks = {}

    for sample, events in events_dict.items():
        txbb = events["ak8FatJetPNetXbb"]
        bb_mask = txbb[0] >= txbb[1]
        bb_masks[sample] = pd.concat((bb_mask, ~bb_mask), axis=1)

    return bb_masks


def get_templates(
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    year: str,
    sig_keys: list[str],
    selection_regions: dict[str, Region],
    shape_vars: list[ShapeVar],
    systematics: dict,  # noqa: ARG001
    template_dir: str = "",
    bg_keys: list[str] = bg_keys,
    plot_dir: str = "",
    prev_cutflow: pd.DataFrame | None = None,
    weight_key: str = "weight",
    plot_sig_keys: list[str] | None = None,
    sig_scale_dict: dict | None = None,
    weight_shifts: dict | None = None,
    jshift: str = "",
    plot_shifts: bool = False,
    pass_ylim: int | None = None,
    fail_ylim: int | None = None,
    blind_pass: bool = False,
    show: bool = False,
    energy=13.6,
) -> dict[str, Hist]:
    """
    (1) Makes histograms for each region in the ``selection_regions`` dictionary,
    (2) TODO: Applies the Txbb scale factor in the pass region,
    (3) TODO: Calculates trigger uncertainty,
    (4) Calculates weight variations if ``weight_shifts`` is not empty (and ``jshift`` is ""),
    (5) Takes JEC / JSMR shift into account if ``jshift`` is not empty,
    (6) Saves a plot of each (if ``plot_dir`` is not "").

    Args:
        selection_region (Dict[str, Dict]): Dictionary of ``Region``s including cuts and labels.
        bg_keys (list[str]): background keys to plot.

    Returns:
        Dict[str, Hist]: dictionary of templates, saved as hist.Hist objects.

    """
    do_jshift = jshift != ""
    jlabel = "" if not do_jshift else "_" + jshift
    templates = {}

    if weight_shifts is None:
        weight_shifts = {}

    for rname, region in selection_regions.items():
        pass_region = rname.startswith("pass")

        if not do_jshift:
            print(rname)

        # make selection, taking JEC/JMC variations into account
        sel, cf = utils.make_selection(
            region.cuts,
            events_dict,
            bb_masks,
            weight_key=weight_key,
            prev_cutflow=prev_cutflow,
            jshift=jshift,
        )

        if template_dir != "":
            cf = cf.round(2)
            cf.to_csv(f"{template_dir}/cutflows/{year}/{rname}_cutflow{jlabel}.csv")

        # # TODO: trigger uncertainties
        # if not do_jshift:
        #     systematics[year][rname] = {}
        #     total, total_err = corrections.get_uncorr_trig_eff_unc(events_dict, bb_masks, year, sel)
        #     systematics[year][rname]["trig_total"] = total
        #     systematics[year][rname]["trig_total_err"] = total_err
        #     print(f"Trigger SF Unc.: {total_err / total:.3f}\n")

        sig_events = {}
        for sig_key in sig_keys:
            sig_events[sig_key] = deepcopy(events_dict[sig_key][sel[sig_key]])

            # # TODO: ParticleNetMD Txbb
            # if pass_region:
            #     corrections.apply_txbb_sfs(sig_events[sig_key], year, weight_key)

        # set up samples
        hist_samples = list(events_dict.keys())

        if not do_jshift:
            # set up weight-based variations
            for shift in ["down", "up"]:
                if pass_region:
                    for sig_key in sig_keys:
                        hist_samples.append(f"{sig_key}_txbb_{shift}")

                for wshift, wsyst in weight_shifts.items():
                    # add to the axis even if not applied to this year to make it easier to sum later
                    for wsample in wsyst.samples:
                        if wsample in events_dict:
                            hist_samples.append(f"{wsample}_{wshift}_{shift}")

        # histograms
        h = Hist(
            hist.axis.StrCategory(hist_samples, name="Sample"),
            *[shape_var.axis for shape_var in shape_vars],
            storage="weight",
        )

        # fill histograms
        for sample in events_dict:
            events = sig_events[sample] if sample in sig_keys else events_dict[sample][sel[sample]]
            if not len(events):
                continue

            bb_mask = bb_masks[sample][sel[sample]] if bb_masks is not None else None
            fill_data = _get_fill_data(
                events, bb_mask, shape_vars, jshift=jshift if sample != data_key else None
            )
            weight = events[weight_key].to_numpy().squeeze()
            h.fill(Sample=sample, **fill_data, weight=weight)

            if not do_jshift:
                # add weight variations
                for wshift, wsyst in weight_shifts.items():
                    if sample in wsyst.samples and year in wsyst.years:
                        # print(wshift)
                        for skey, shift in [("Down", "down"), ("Up", "up")]:
                            # reweight based on diff between up/down and nominal weights
                            h.fill(
                                Sample=f"{sample}_{wshift}_{shift}",
                                **fill_data,
                                weight=events[f"weight_{wshift}{skey}"].to_numpy().squeeze(),
                            )

        if pass_region:
            # blind signal mass windows in pass region in data
            for i, shape_var in enumerate(shape_vars):
                if shape_var.blind_window is not None:
                    utils.blindBins(h, shape_var.blind_window, data_key, axis=i)

        # if pass_region and not do_jshift:
        #     for sig_key in sig_keys:
        #         if not len(sig_events[sig_key]):
        #             continue

        #         # ParticleNetMD Txbb SFs
        #         fill_data = _get_fill_data(
        #             sig_events[sig_key], bb_masks[sig_key][sel[sig_key]], shape_vars
        #         )
        #         for shift in ["down", "up"]:
        #             h.fill(
        #                 Sample=f"{sig_key}_txbb_{shift}",
        #                 **fill_data,
        #                 weight=sig_events[sig_key][f"{weight_key}_txbb_{shift}"],
        #             )

        templates[rname + jlabel] = h

        # plot templates incl variations
        if plot_dir != "" and (not do_jshift or plot_shifts):
            for shape_var in shape_vars:
                title = (
                    f"{region.label} Region Pre-Fit Shapes"
                    if not do_jshift
                    else f"{region.label} Region {jshift} Shapes"
                )

                plot_params = {
                    "hists": h,
                    "sig_keys": sig_keys if plot_sig_keys is None else plot_sig_keys,
                    "bg_keys": bg_keys,
                    "sig_scale_dict": sig_scale_dict if pass_region else None,
                    "show": show,
                    "year": year,
                    "ylim": pass_ylim if pass_region else fail_ylim,
                    "plot_data": not (rname == "pass" and blind_pass),
                }

                plot_name = (
                    f"{plot_dir}/"
                    f"{'jshifts/' if do_jshift else ''}"
                    f"{rname}_region_{shape_var.var}"
                )

                plotting.ratioHistPlot(
                    **plot_params,
                    title=title,
                    name=f"{plot_name}{jlabel}.pdf",
                    energy=energy,
                )

            if not do_jshift and plot_shifts:
                plot_name = f"{plot_dir}/wshifts/" f"{rname}_region_{shape_var.var}"

                for wshift, wsyst in weight_shifts.items():
                    if wsyst.samples == [sig_key]:
                        plotting.ratioHistPlot(
                            **plot_params,
                            sig_err=wshift,
                            title=f"{region.label} Region {wsyst.label} Unc. Shapes",
                            name=f"{plot_name}_{wshift}.pdf",
                        )
                    else:
                        for skey, shift in [("Down", "down"), ("Up", "up")]:
                            plotting.ratioHistPlot(
                                **plot_params,
                                variation=(wshift, shift, wsyst.samples),
                                title=f"{region.label} Region {wsyst.label} Unc. {skey} Shapes",
                                name=f"{plot_name}_{wshift}_{shift}.pdf",
                            )

                if pass_region:
                    plotting.ratioHistPlot(
                        **plot_params,
                        sig_err="txbb",
                        title=rf"{region.label} Region $T_{{Xbb}}$ Shapes",
                        name=f"{plot_name}_txbb.pdf",
                    )

    return templates


def save_templates(templates: dict[str, Hist], template_file: Path, shape_var: ShapeVar):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    from copy import deepcopy

    blind_window = shape_var.blind_window

    if blind_window is not None:
        for label, template in list(templates.items()):
            blinded_template = deepcopy(template)
            utils.blindBins(blinded_template, blind_window)
            templates[f"{label}MCBlinded"] = blinded_template

    with template_file.open("wb") as f:
        pickle.dump(templates, f)

    print("Saved templates to", template_file)


if __name__ == "__main__":
    sys.exit(postprocess())
