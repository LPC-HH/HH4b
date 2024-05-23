from __future__ import annotations

import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import hist
import numpy as np
import pandas as pd
from hist import Hist
from sklearn.metrics import roc_curve

from HH4b import plotting, utils
from HH4b.hh_vars import (
    LUMI,
    bg_keys,
    data_key,
    sig_keys,
    syst_keys,
    years,
)

# define ShapeVar (label and bins for a given variable)
from HH4b.utils import ShapeVar, Syst


@dataclass
class Region:
    cuts: dict = None
    label: str = None


mass_key = "bbFatJetPNetMassLegacy"
filters_legacy = [
    [
        ("('bbFatJetPt', '0')", ">=", 250),
        ("('bbFatJetPt', '1')", ">=", 250),
        (f"('{mass_key}', '0')", "<=", 250),
        (f"('{mass_key}', '1')", "<=", 250),
        (f"('{mass_key}', '0')", ">=", 60),
        (f"('{mass_key}', '1')", ">=", 60),
    ],
]

filters_v12 = [
    [
        ("('bbFatJetPt', '0')", ">=", 300),
        ("('bbFatJetPt', '1')", ">=", 300),
        ("('bbFatJetMsd', '0')", "<=", 250),
        ("('bbFatJetMsd', '1')", "<=", 250),
        ("('bbFatJetMsd', '0')", ">=", 30),
        ("('bbFatJetMsd', '1')", ">=", 30),
    ],
]


HLTs = {
    "2022": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet425_SoftDropMass40",
    ],
    "2022EE": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet425_SoftDropMass40",
    ],
    "2023": [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        # "AK8PFJet400_SoftDropMass40", #TODO: add to ntuples
        "AK8PFJet425_SoftDropMass40",
    ],
    "2023BPix": [
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        # "AK8PFJet400_SoftDropMass40", #TODO: add to ntuples
        "AK8PFJet425_SoftDropMass40",
    ],
}


load_columns = [
    ("weight", 1),
    ("event", 1),
    ("MET_pt", 1),
    ("bbFatJetPt", 2),
    ("bbFatJetEta", 2),
    ("bbFatJetPhi", 2),
    ("bbFatJetMsd", 2),
    ("bbFatJetTau3OverTau2", 2),
    ("VBFJetPt", 2),
    ("VBFJetEta", 2),
    ("VBFJetPhi", 2),
    ("VBFJetMass", 2),
]

load_columns_legacy = load_columns + [
    ("bbFatJetPNetTXbbLegacy", 2),
    ("bbFatJetPNetPXbbLegacy", 2),
    ("bbFatJetPNetPQCDbLegacy", 2),
    ("bbFatJetPNetPQCDbbLegacy", 2),
    ("bbFatJetPNetPQCDothersLegacy", 2),
    ("bbFatJetPNetMassLegacy", 2),
    ("bbFatJetPNetTXbb", 2),
    ("bbFatJetPNetMass", 2),
    ("bbFatJetPNetQCD0HF", 2),
    ("bbFatJetPNetQCD1HF", 2),
    ("bbFatJetPNetQCD2HF", 2),
]

load_columns_v12 = load_columns + [
    ("bbFatJetPNetTXbb", 2),
    # ("bbFatJetPNetXbb", 2),
    ("bbFatJetPNetMass", 2),
    ("bbFatJetPNetQCD0HF", 2),
    ("bbFatJetPNetQCD1HF", 2),
    ("bbFatJetPNetQCD2HF", 2),
]

load_columns_syst = [
    ("bbFatJetPt_JES_up", 2),
    ("bbFatJetPt_JES_down", 2),
    ("VBFJetPt_JES_up", 2),
    ("VBFJetPt_JES_down", 2),
    # ("bbFatJetPt_JER_up", 2),  # TODO: load once present
    # ("bbFatJetPt_JER_down", 2),  # TODO: load once present
    # ("VBFJetPt_JER_up", 2),  # TODO: load once present
    # ("VBFJetPt_JER_down", 2),  # TODO: load once present
    # ("bbFatJetMsd_JMS_up", 2),  # TODO: load once present
    # ("bbFatJetMsd_JMS_down", 2),  # TODO: load once present
    # ("bbFatJetPNetMass_JMS_up", 2),  # TODO: load once present
    # ("bbFatJetPNetMass_JMS_down", 2),  # TODO: load once present
    # ("bbFatJetMsd_JMR_up", 2),  # TODO: load once present
    # ("bbFatJetMsd_JMR_down", 2),  # TODO: load once present
    # ("bbFatJetPNetMass_JMR_up", 2),  # TODO: load once present
    # ("bbFatJetPNetMass_JMR_down", 2),  # TODO: load once present
]

weight_shifts = {
    "ttbarSF_pTjj": Syst(samples=["ttbar"], label="ttbar SF pTjj", years=years + ["2022-2023"]),
    "ttbarSF_tau32": Syst(samples=["ttbar"], label="ttbar SF tau32", years=years + ["2022-2023"]),
    "trigger": Syst(samples=sig_keys + bg_keys, label="Trigger", years=years + ["2022-2023"]),
    # "pileup": Syst(samples=sig_keys + bg_keys, label="Pileup"),
    # "PDFalphaS": Syst(samples=sig_keys, label="PDF"),
    # "QCDscale": Syst(samples=sig_keys, label="QCDscale"),
    # "ISRPartonShower": Syst(samples=sig_keys_ggf + ["vjets"], label="ISR Parton Shower"),
    # "FSRPartonShower": Syst(samples=sig_keys_ggf + ["vjets"], label="FSR Parton Shower"),
}

decorr_txbb_bins = [0, 0.8, 0.94, 0.99, 1]

for i in range(len(decorr_txbb_bins) - 1):
    weight_shifts[f"ttbarSF_Xbb_bin_{decorr_txbb_bins[i]}_{decorr_txbb_bins[i+1]}"] = Syst(
        samples=["ttbar"],
        label=f"ttbar SF Xbb bin [{decorr_txbb_bins[i]}, {decorr_txbb_bins[i+1]}]",
        years=years + ["2022-2023"],
    )


def load_run3_samples(
    input_dir: str,
    year: str,
    legacy: bool,
    samples_run3: dict[str, list[str]],
    reorder_txbb: bool,
    txbb: str,
):
    filters = filters_legacy if legacy else filters_v12
    load_columns = load_columns_legacy if legacy else load_columns_v12

    # add HLTs to load columns
    load_columns_year = load_columns + [(hlt, 1) for hlt in HLTs[year]]

    samples_syst = {
        sample: samples_run3[year][sample] for sample in samples_run3[year] if sample in syst_keys
    }
    samples_nosyst = {
        sample: samples_run3[year][sample]
        for sample in samples_run3[year]
        if sample not in syst_keys
    }

    # pre-selection
    events_dict = {
        **utils.load_samples(
            input_dir,
            samples_nosyst,
            year,
            filters=filters,
            columns=utils.format_columns(load_columns_year),
            reorder_txbb=reorder_txbb,
            txbb=txbb,
            variations=False,
        ),
        **utils.load_samples(
            input_dir,
            samples_syst,
            year,
            filters=filters,
            columns=utils.format_columns(load_columns_year + load_columns_syst),
            reorder_txbb=reorder_txbb,
            txbb=txbb,
            variations=False,
        ),
    }
    return events_dict


def get_evt_testing(inferences_dir, key):
    evt_file = Path(f"{inferences_dir}/evt_{key}.npy")
    if evt_file.is_file():
        evt_list = np.load(f"{inferences_dir}/evt_{key}.npy")
    else:
        evt_list = None
    return evt_list


def combine_run3_samples(
    events_dict_years: dict[str, dict[str, pd.DataFrame]],
    processes: list[str],
    bg_keys: list[str] = None,
    weight_key: str = "weight",
    scale_processes: dict = None,  # processes which are temporarily on certain eras, e.g {"hh4b": ["2022EE", "2023"]}
    years_run3: list[str] = years,
):
    # create combined datasets
    lumi_total = np.sum([LUMI[year] for year in years_run3])

    events_combined = {}
    scaled_by = {}
    for key in processes:
        if key not in scale_processes:
            combined = pd.concat(
                [
                    events_dict_years[year][key]
                    for year in years_run3
                    if key in events_dict_years[year]
                ]
            )
        else:
            combined = pd.concat(
                [
                    events_dict_years[year][key].copy()
                    for year in scale_processes[key]
                    if year in years_run3
                ]
            )
            lumi_scale = lumi_total / np.sum(
                [LUMI[year] for year in scale_processes[key] if year in years_run3]
            )
            scaled_by[key] = lumi_scale
            print(f"Concatenate {scale_processes[key]}, scaling {key} by {lumi_scale:.2f}")
            combined[weight_key] = combined[weight_key] * lumi_scale

        events_combined[key] = combined

    # combine ttbar
    if "ttbar" in processes and "ttlep" in processes:
        events_combined["ttbar"] = pd.concat([events_combined["ttbar"], events_combined["ttlep"]])
        events_combined.pop("ttlep")
        if bg_keys:
            bg_keys.remove("ttlep")

    # combine others
    # others = ["diboson", "vjets", "gghtobb", "vbfhtobb"]
    # if np.all([key in processes for key in others]):
    #     events_combined["others"] = pd.concat([events_combined[key] for key in others])
    #     for key in others:
    #         events_combined.pop(key)
    #         if bg_keys:
    #             bg_keys.remove(key)
    #     if bg_keys:
    #         bg_keys.append("others")

    return events_combined, scaled_by


def make_rocs(
    events_dict: dict[str, pd.DataFrame],
    scores_key: str,
    weight_key: str,
    sig_key: str,
    bg_keys: list[str],
):
    rocs = {}
    for bkg in [*bg_keys, "merged"]:
        if bkg != "merged":
            scores_roc = np.concatenate(
                [events_dict[sig_key][scores_key], events_dict[bkg][scores_key]]
            )
            scores_true = np.concatenate(
                [
                    np.ones(len(events_dict[sig_key])),
                    np.zeros(len(events_dict[bkg])),
                ]
            )
            scores_weights = np.concatenate(
                [events_dict[sig_key][weight_key], events_dict[bkg][weight_key]]
            )
            fpr, tpr, thresholds = roc_curve(scores_true, scores_roc, sample_weight=scores_weights)
        else:
            scores_roc = np.concatenate(
                [events_dict[sig_key][scores_key]]
                + [events_dict[bg_key][scores_key] for bg_key in bg_keys]
            )
            scores_true = np.concatenate(
                [
                    np.ones(len(events_dict[sig_key])),
                    np.zeros(np.sum([len(events_dict[bg_key]) for bg_key in bg_keys])),
                ]
            )
            scores_weights = np.concatenate(
                [events_dict[sig_key][weight_key]]
                + [events_dict[bg_key][weight_key] for bg_key in bg_keys]
            )
            fpr, tpr, thresholds = roc_curve(scores_true, scores_roc, sample_weight=scores_weights)

        rocs[bkg] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "label": plotting.label_by_sample[bkg] if bkg != "merged" else "Combined",
        }

    return rocs


def _get_fill_data(
    events: pd.DataFrame,
    shape_vars: list[ShapeVar],
    jshift: str = "",
):
    return {
        shape_var.var: utils.get_feat(
            events,
            shape_var.var if jshift == "" else utils.check_get_jec_var(shape_var.var, jshift),
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
    year: str,
    sig_keys: list[str],
    selection_regions: dict[str, Region],
    shape_vars: list[ShapeVar],
    systematics: dict,  # noqa: ARG001
    template_dir: str = "",
    bg_keys: list[str] = bg_keys,
    plot_dir: Path = "",
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
            weight_key=weight_key,
            prev_cutflow=prev_cutflow,
            jshift=jshift,
        )

        if template_dir != "":
            cf = cf.round(2)
            print("cutflow ", rname, cf)
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
            weight = events[weight_key].to_numpy().squeeze()
            h.fill(Sample=sample, **fill_data, weight=weight)

            if not do_jshift:
                # add weight variations
                for wshift, wsyst in weight_shifts.items():
                    if sample in wsyst.samples and year in wsyst.years:
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
                    "bg_err_mcstat": True,
                    "reweight_qcd": False,
                }
                if do_jshift:
                    plot_dir_jshifts = plot_dir / "jshifts"
                    plot_dir_jshifts.mkdir(exist_ok=True, parents=True)
                    plot_name = plot_dir_jshifts / f"{rname}_region_{shape_var.var}"
                else:
                    plot_name = plot_dir / f"{rname}_region_{shape_var.var}"

                plotting.ratioHistPlot(
                    **plot_params,
                    title=title,
                    name=f"{plot_name}{jlabel}.pdf",
                    energy=energy,
                )

            if not do_jshift and plot_shifts:
                plot_dir_wshifts = plot_dir / "wshifts"
                plot_dir_wshifts.mkdir(exist_ok=True, parents=True)

                plot_name = plot_dir_wshifts / f"{rname}_region_{shape_var.var}"

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
    sys.exit()
