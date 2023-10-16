#!/usr/bin/python3

from typing import Dict, List
import click

import pickle, json
import utils
import plotting
from hh_vars import samples, data_key, bg_keys, sig_keys

from copy import deepcopy
import logging
import sys
import numpy as np
import pandas as pd

import hist
from hist import Hist

# define ShapeVar (label and bins for a given variable)
from utils import ShapeVar

from dataclasses import dataclass, field


@dataclass
class Region:
    cuts: Dict = None
    label: str = None


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
    dirs = {
        "/eos/uscms/store/user/cmantill/bbbb/skimmer/Oct2/": {
            "qcd": [
                "QCD_PT-120to170",
                "QCD_PT-170to300",
                "QCD_PT-470to600",
                "QCD_PT-600to800",
                "QCD_PT-800to1000",
                "QCD_PT-1000to1400",
                "QCD_PT-1400to1800",
                "QCD_PT-1800to2400",
                "QCD_PT-2400to3200",
                "QCD_PT-3200",
            ],
            "data": [
                "Run2022F",
                "Run2022G",
            ],
            "ttbar": [
                "TTtoLNu2Q",
                "TTto4Q",
                "TTto2L2Nu",
            ],
            "gghtobb": [
                "GluGluHto2B_PT-200_M-125",
            ],
            "vbfhtobb": [
                "VBFHto2B_M-125_dipoleRecoilOn",
            ],
            "vhtobb": [
                "WplusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_WtoLNu_M-125",
                "WminusH_Hto2B_Wto2Q_M-125",
                "WminusH_Hto2B_WtoLNu_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2L_M-125",
                "ggZH_Hto2B_Zto2Nu_M-125",
            ],
            "tthtobb": [
                "ttHto2B_M-125",
            ],
            "diboson": [
                "ZZ",
                "WW",
                "WZ",
            ],
            "vjets": [
                "Wto2Q-3Jets_HT-200to400",
                "Wto2Q-3Jets_HT-400to600",
                "Wto2Q-3Jets_HT-600to800",
                "Wto2Q-3Jets_HT-800",
                "Zto2Q-4Jets_HT-200to400",
                "Zto2Q-4Jets_HT-400to600",
                "Zto2Q-4Jets_HT-600to800",
                "Zto2Q-4Jets_HT-800",
            ],
        }
    }
    samples_to_fill = [
        "data",
        "qcd",
    ]
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
        for input_dir, samples in dirs.items():
            events_dict = {
                **events_dict,
                **utils.load_samples(input_dir, samples, year, filters, columns),
            }

        samples_loaded = list(events_dict.keys())
        keys_loaded = list(events_dict[samples_loaded[0]].keys())
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


def _get_fill_data(events: pd.DataFrame, shape_vars: List[ShapeVar], jshift: str = ""):
    return {
        shape_var.var: utils.get_feat(
            events,
            shape_var.var if jshift == "" else utils.check_get_jec_var(shape_var.var, jshift),
        )
        for shape_var in shape_vars
    }


def get_templates(
    events_dict: Dict[str, pd.DataFrame],
    year: str,
    sig_keys: List[str],
    selection_regions: Dict[str, Region],
    shape_vars: List[ShapeVar],
    systematics: Dict,
    template_dir: str = "",
    bg_keys: List[str] = bg_keys,
    plot_dir: str = "",
    prev_cutflow: pd.DataFrame = None,
    weight_key: str = "weight",
    plot_sig_keys: List[str] = None,
    sig_scale_dict: Dict = None,
    weight_shifts: Dict = {},
    jshift: str = "",
    plot_shifts: bool = False,
    pass_ylim: int = None,
    fail_ylim: int = None,
    blind_pass: bool = False,
    show: bool = False,
) -> Dict[str, Hist]:
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

    for rname, region in selection_regions.items():
        pass_region = rname.startswith("pass")

        if not do_jshift:
            print(rname)

        # make selection, taking JEC/JMC variations into account
        sel, cf = utils.make_selection(
            region.cuts, events_dict, prev_cutflow=prev_cutflow, jshift=jshift
        )

        if template_dir != "":
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

            fill_data = _get_fill_data(
                events, shape_vars, jshift=jshift if sample != data_key else None
            )
            weight = events[weight_key].values.squeeze()
            h.fill(Sample=sample, **fill_data, weight=weight)

            if not do_jshift:
                # add weight variations
                for wshift, wsyst in weight_shifts.items():
                    if sample in wsyst.samples and year in wsyst.years:
                        # print(wshift)
                        for skey, shift in [("Down", "down"), ("Up", "up")]:
                            if "QCDscale" in wshift:
                                # QCDscale7pt/QCDscale4
                                # https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L263-L288
                                sweight = (
                                    weight
                                    * (
                                        events[f"weight_QCDscale7pt{skey}"][0]
                                        / events["weight_QCDscale4"]
                                    ).values.squeeze()
                                )
                            else:
                                # reweight based on diff between up/down and nominal weights
                                sweight = (
                                    weight
                                    * (
                                        events[f"weight_{wshift}{skey}"][0]
                                        / events["weight_nonorm"]
                                    ).values.squeeze()
                                )
                            h.fill(Sample=f"{sample}_{wshift}_{shift}", **fill_data, weight=sweight)

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


def save_templates(templates: Dict[str, Hist], template_file: str, shape_var: ShapeVar):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    from copy import deepcopy

    blind_window = shape_var.blind_window

    for label, template in list(templates.items()):
        blinded_template = deepcopy(template)
        utils.blindBins(blinded_template, blind_window)
        templates[f"{label}MCBlinded"] = blinded_template

    with open(template_file, "wb") as f:
        pickle.dump(templates, f)

    print("Saved templates to", template_file)


if __name__ == "__main__":
    sys.exit(postprocess())
