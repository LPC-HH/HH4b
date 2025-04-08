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
    jec_shifts,
    jmsr,
    jmsr_keys,
    jmsr_res,
    jmsr_shifts,
    jmsr_values,
    sig_keys,
    syst_keys,
    ttbarsfs_decorr_ggfbdt_bins,
    ttbarsfs_decorr_txbb_bins,
    ttbarsfs_decorr_vbfbdt_bins,
    txbb_strings,
    txbbsfs_decorr_pt_bins,
    txbbsfs_decorr_txbb_wps,
    years,
)

# define ShapeVar (label and bins for a given variable)
from HH4b.utils import ShapeVar, Syst, check_get_jec_var


@dataclass
class Region:
    cuts: dict = None
    label: str = None


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
        "AK8PFJet400_SoftDropMass40",
        "AK8PFJet425_SoftDropMass40",
    ],
    "2023BPix": [
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        "AK8PFJet400_SoftDropMass40",
        "AK8PFJet425_SoftDropMass40",
    ],
}


columns_to_load_default = [
    ("weight", 1),
    ("event", 1),
    ("run", 1),
    ("luminosityBlock", 1),
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
    ("AK4JetAwayPt", 2),
    ("AK4JetAwayEta", 2),
    ("AK4JetAwayPhi", 2),
    ("AK4JetAwayMass", 2),
    ("bdt_score", 1),
    ("bdt_score_vbf", 1),
]

columns_to_load = {
    "pnet-legacy": columns_to_load_default
    + [
        ("bbFatJetPNetTXbbLegacy", 2),
        ("bbFatJetPNetPXbbLegacy", 2),
        ("bbFatJetPNetPQCD0HFLegacy", 2),
        ("bbFatJetPNetPQCD1HFLegacy", 2),
        ("bbFatJetPNetPQCD2HFLegacy", 2),
        ("bbFatJetPNetMassLegacy", 2),
        ("bbFatJetPNetTXbb", 2),
        ("bbFatJetPNetMass", 2),
    ],
    "pnet-v12": columns_to_load_default
    + [
        ("bbFatJetPNetTXbb", 2),
        ("bbFatJetPNetMass", 2),
        ("bbFatJetPNetQCD0HF", 2),
        ("bbFatJetPNetQCD1HF", 2),
        ("bbFatJetPNetQCD2HF", 2),
    ],
    "glopart-v2": columns_to_load_default
    + [
        ("bbFatJetParTTXbb", 2),
        ("bbFatJetParTPXbb", 2),
        ("bbFatJetParTmassVis", 2),
        ("bbFatJetParTPQCD0HF", 2),
        ("bbFatJetParTPQCD1HF", 2),
        ("bbFatJetParTPQCD2HF", 2),
        ("bbFatJetrawFactor", 2),
    ],
}

filters_to_apply = {
    "pnet-legacy": [
        [
            ("('bbFatJetPt', '0')", ">=", 250),
            ("('bbFatJetPt', '1')", ">=", 250),
            ("('bbFatJetPNetMassLegacy', '0')", "<=", 250),
            ("('bbFatJetPNetMassLegacy', '1')", "<=", 250),
            ("('bbFatJetPNetMassLegacy', '0')", ">=", 60),
            ("('bbFatJetPNetMassLegacy', '1')", ">=", 60),
        ],
    ],
    "pnet-v12": [
        [
            ("('bbFatJetPt', '0')", ">=", 250),
            ("('bbFatJetPt', '1')", ">=", 250),
            ("('bbFatJetMsd', '0')", "<=", 250),
            ("('bbFatJetMsd', '1')", "<=", 250),
            ("('bbFatJetMsd', '0')", ">=", 30),
            ("('bbFatJetMsd', '1')", ">=", 30),
        ],
    ],
    "glopart-v2": [
        [
            ("('bbFatJetPt', '0')", ">=", 250),
            ("('bbFatJetPt', '1')", ">=", 250),
        ],
    ],
}

load_columns_syst = []

for jshift in jec_shifts:
    load_columns_syst += [
        (f"bbFatJetPt_{jshift}", 2),
        (f"VBFJetPt_{jshift}", 2),
        (f"bdt_score_{jshift}", 1),
        (f"bdt_score_vbf_{jshift}", 1),
    ]

for jshift in jmsr_shifts:
    load_columns_syst += [
        (f"bbFatJetParTmassVis_{jshift}", 2),
        (f"bdt_score_{jshift}", 1),
        (f"bdt_score_vbf_{jshift}", 1),
    ]
load_columns_syst += [("bbFatJetParTmassVis_raw", 2)]

# load scale and pdf weights
load_columns_thy = [
    ("scale_weights", 6),
    ("pdf_weights", 103),
]

# only the BG MC samples that are used in the fits
fit_bgs = ["ttbar", "vhtobb", "zz", "nozzdiboson", "vjets", "tthtobb"]
fit_mcs = sig_keys + fit_bgs


def get_weight_shifts(txbb_version: str, bdt_version: str):
    """Get weight shifts for systematics"""

    weight_shifts = {
        "ttbarSF_pTjj": Syst(samples=["ttbar"], label="ttbar SF pTjj", years=years + ["2022-2023"]),
        "ttbarSF_tau32": Syst(
            samples=["ttbar"], label="ttbar SF tau32", years=years + ["2022-2023"]
        ),
        "trigger": Syst(samples=sig_keys + bg_keys, label="Trigger", years=years + ["2022-2023"]),
        "TXbbSF_correlated": Syst(
            samples=sig_keys, label="TXbb SF correlated", years=years + ["2022-2023"]
        ),
        "pileup": Syst(samples=fit_mcs, label="Pileup", years=years + ["2022-2023"]),
        "scale": Syst(
            samples=sig_keys + ["ttbar"],
            label="QCDScaleAcc",
            years=years + ["2022-2023"],
        ),
        "pdf": Syst(samples=sig_keys, label="PDFAcc", years=years + ["2022-2023"]),
        "ISRPartonShower": Syst(
            samples=sig_keys, label="ISR Parton Shower", years=years + ["2022-2023"]
        ),
        "FSRPartonShower": Syst(
            samples=sig_keys, label="FSR Parton Shower", years=years + ["2022-2023"]
        ),
    }

    ttsf_xbb_bins = ttbarsfs_decorr_txbb_bins.get(txbb_version, "glopart-v2")
    ttsf_ggfbdtshape_bins = ttbarsfs_decorr_ggfbdt_bins.get(
        bdt_version, "25Feb5_v13_glopartv2_rawmass"
    )
    TXbb_pt_corr_bins = txbbsfs_decorr_pt_bins.get(txbb_version, "glopart-v2")
    TXbb_wps = txbbsfs_decorr_txbb_wps.get(txbb_version, "glopart-v2")

    for i in range(len(ttsf_xbb_bins) - 1):
        weight_shifts[f"ttbarSF_Xbb_bin_{ttsf_xbb_bins[i]}_{ttsf_xbb_bins[i+1]}"] = Syst(
            samples=["ttbar"],
            label=f"ttbar SF Xbb bin [{ttsf_xbb_bins[i]}, {ttsf_xbb_bins[i+1]}]",
            years=years + ["2022-2023"],
        )

    for i in range(len(ttsf_ggfbdtshape_bins) - 1):
        weight_shifts[
            f"ttbarSF_ggF_BDT_bin_{ttsf_ggfbdtshape_bins[i]}_{ttsf_ggfbdtshape_bins[i+1]}"
        ] = Syst(
            samples=["ttbar"],
            label=f"ttbar SF ggF BDT bin [{ttsf_ggfbdtshape_bins[i]}, {ttsf_ggfbdtshape_bins[i+1]}]",
            years=years + ["2022-2023"],
        )

    if bdt_version in ttbarsfs_decorr_vbfbdt_bins:
        ttsf_vbfbdtshape_bins = ttbarsfs_decorr_vbfbdt_bins[bdt_version]
        for i in range(len(ttsf_vbfbdtshape_bins) - 1):
            weight_shifts[
                f"ttbarSF_VBF_BDT_bin_{ttsf_vbfbdtshape_bins[i]}_{ttsf_vbfbdtshape_bins[i+1]}"
            ] = Syst(
                samples=["ttbar"],
                label=f"ttbar SF VBF BDT bin [{ttsf_vbfbdtshape_bins[i]}, {ttsf_vbfbdtshape_bins[i+1]}]",
                years=years + ["2022-2023"],
            )

    for wp in TXbb_wps:
        for j in range(len(TXbb_pt_corr_bins[wp]) - 1):
            weight_shifts[
                f"TXbbSF_uncorrelated_{wp}_pT_bin_{TXbb_pt_corr_bins[wp][j]}_{TXbb_pt_corr_bins[wp][j+1]}"
            ] = Syst(
                samples=sig_keys,
                label=f"TXbb SF uncorrelated {wp}, pT bin [{TXbb_pt_corr_bins[wp][j]}, {TXbb_pt_corr_bins[wp][j+1]}]",
                years=years + ["2022-2023"],
            )
    return weight_shifts


def load_run3_samples(
    input_dir: str,
    year: str,
    samples_run3: dict[str, list[str]],
    reorder_txbb: bool,
    load_systematics: bool,
    txbb_version: str,
    scale_and_smear: bool,
    mass_str: str,
):
    assert txbb_version in [
        "pnet-v12",
        "pnet-legacy",
        "glopart-v2",
    ], "txbb_version parameter must be pnet-v12, pnet-legacy, glopart-v2"

    txbb_str = txbb_strings[txbb_version]
    filters = filters_to_apply[txbb_version]
    load_columns = columns_to_load[txbb_version]

    # add HLTs to load columns
    load_columns_year = load_columns + [(hlt, 1) for hlt in HLTs[year]]

    samples_syst_sig = {
        sample: samples_run3[year][sample]
        for sample in samples_run3[year]
        if (sample in syst_keys and sample in sig_keys)
    }
    samples_syst_bg = {
        sample: samples_run3[year][sample]
        for sample in samples_run3[year]
        if (sample in syst_keys and sample not in sig_keys)
    }
    samples_nosyst = {
        sample: samples_run3[year][sample]
        for sample in samples_run3[year]
        if sample not in syst_keys
    }

    # load sig samples that need more systematics
    events_dict_syst_sig = {
        **utils.load_samples(
            input_dir,
            samples_syst_sig,
            year,
            filters=filters,
            columns=utils.format_columns(
                load_columns_year + load_columns_syst + load_columns_thy
                if load_systematics
                else load_columns_year
            ),
            reorder_txbb=reorder_txbb,
            txbb_str=txbb_str,
            variations=True,
            weight_shifts={},
        ),
    }

    # load bkg samples that need systematics
    events_dict_syst_bg = {
        **utils.load_samples(
            input_dir,
            samples_syst_bg,
            year,
            filters=filters,
            columns=utils.format_columns(
                load_columns_year + load_columns_syst if load_systematics else load_columns_year
            ),
            reorder_txbb=reorder_txbb,
            txbb_str=txbb_str,
            variations=False,
        ),
    }

    # load samples that do not need systematics (e.g. data)
    events_dict_nosyst = {
        **utils.load_samples(
            input_dir,
            samples_nosyst,
            year,
            filters=filters,
            columns=utils.format_columns(load_columns_year),
            reorder_txbb=reorder_txbb,
            txbb_str=txbb_str,
            variations=False,
        ),
    }

    if scale_and_smear:
        # re-run scaling and smearing of mass variables
        events_dict_syst_bg = scale_smear_mass(events_dict_syst_bg, year, mass_str)
        events_dict_syst_sig = scale_smear_mass(events_dict_syst_sig, year, mass_str)

    events_dict = {**events_dict_nosyst, **events_dict_syst_bg, **events_dict_syst_sig}

    return events_dict


def scale_smear_mass(
    events_dict: dict[str, pd.DataFrame], year: str, mass_str: str, morphing_formula: bool = False
):
    jms_nom = jmsr_values[mass_str]["JMS"][year]["nom"]
    jmr_nom = jmsr_values[mass_str]["JMR"][year]["nom"]
    rng = np.random.default_rng(seed=42)

    # formula for smearing and scaling
    for key in events_dict:
        if key in jmsr_keys:
            print(f"scaling and smearing mass for {key} {year}")
            x = events_dict[key][f"{mass_str}_raw"].to_numpy(copy=True)
            random_smear = rng.standard_normal(size=x.shape)
            x_smear = x * jms_nom
            x_smear *= (
                (1 + random_smear * np.sqrt(jmr_nom * jmr_nom - 1) * jmsr_res[mass_str][key] / x)
                if morphing_formula
                else (1 + random_smear * max(jmr_nom - 1, 0))
            )

            for i in range(2):
                events_dict[key][(mass_str, i)] = x_smear[:, i]
            for skey in jmsr:
                for shift in ["up", "down"]:
                    if skey == "JMS":
                        jms = jmsr_values[mass_str]["JMS"][year][shift]
                        jmr = jmr_nom
                    else:
                        jms = jms_nom
                        jmr = jmsr_values[mass_str]["JMR"][year][shift]
                    x_smear = x * jms
                    x_smear *= (
                        (1 + random_smear * np.sqrt(jmr * jmr - 1) * jmsr_res[mass_str][key] / x)
                        if morphing_formula
                        else (1 + random_smear * max(jmr - 1, 0))
                    )
                    for i in range(2):
                        events_dict[key][(f"{mass_str}_{skey}_{shift}", i)] = x_smear[:, i]
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

    if scale_processes is None:
        scale_processes = {}

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

    return events_combined, scaled_by


def make_rocs(
    events_dict: dict[str, pd.DataFrame],
    scores_key: str,
    weight_key: str,
    sig_key: str,
    bg_keys: list[str],
    jshift: str = "",
):
    rocs = {}
    for bkg in [*bg_keys, "merged"]:
        if bkg != "merged":
            scores_roc = np.concatenate(
                [
                    events_dict[sig_key][check_get_jec_var(scores_key, jshift)],
                    events_dict[bkg][scores_key],
                ]
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
                [events_dict[sig_key][check_get_jec_var(scores_key, jshift)]]
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


def _get_qcdvar_hists(
    events: pd.DataFrame, shape_vars: list[ShapeVar], fill_data: dict, wshift: str
):
    """Get histograms for QCD scale and PDF variations"""
    wkey = f"{wshift}_weights"
    cols = sorted([int(col.split("_")[-1]) for col in events.columns if wkey in col])
    h = Hist(
        hist.axis.StrCategory([str(i) for i in cols], name="Sample"),
        *[shape_var.axis for shape_var in shape_vars],
        storage="weight",
    )

    for i in cols:
        h.fill(
            Sample=str(i),
            **fill_data,
            weight=events[f"{wkey}_{i}"],
        )
    return h


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
    energy: float = 13.6,
    blind: bool = True,
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

        sig_events = {}
        for sig_key in sig_keys:
            sig_events[sig_key] = deepcopy(events_dict[sig_key][sel[sig_key]])

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
                events, shape_vars, jshift=jshift if sample in jmsr_keys else None
            )
            weight = events[weight_key].to_numpy().squeeze()
            h.fill(Sample=sample, **fill_data, weight=weight)

            if not do_jshift:
                # add weight variations
                for wshift, wsyst in weight_shifts.items():
                    if sample in wsyst.samples and year in wsyst.years:
                        if wshift not in ["scale", "pdf"]:
                            # fill histogram with weight variations
                            for skey, shift in [("Down", "down"), ("Up", "up")]:
                                h.fill(
                                    Sample=f"{sample}_{wshift}_{shift}",
                                    **fill_data,
                                    weight=events[f"weight_{wshift}{skey}"].to_numpy().squeeze(),
                                )
                        else:
                            # get histograms for all QCD scale and PDF variations
                            whists = _get_qcdvar_hists(events, shape_vars, fill_data, wshift)

                            if wshift == "scale":
                                # renormalization / factorization scale uncertainty is the max/min envelope of the variations
                                shape_up = np.max(whists.values(), axis=0)
                                shape_down = np.min(whists.values(), axis=0)
                            else:
                                # pdf uncertainty is the norm of each variation (corresponding to 103 eigenvectors) - nominal
                                nom_vals = h[sample, :].values()
                                abs_unc = np.linalg.norm((whists.values() - nom_vals), axis=0)
                                # cap at 100% uncertainty
                                rel_unc = np.clip(abs_unc / nom_vals, 0, 1)
                                shape_up = nom_vals * (1 + rel_unc)
                                shape_down = nom_vals * (1 - rel_unc)

                            h.values()[
                                utils.get_key_index(h, f"{sample}_{wshift}_up"), :
                            ] = shape_up
                            h.values()[
                                utils.get_key_index(h, f"{sample}_{wshift}_down"), :
                            ] = shape_down

        if pass_region and blind:
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
                    "bg_order": [
                        "vbfhtobb",
                        "gghtobb",
                        "tthtobb",
                        "vhtobb",
                        "singletop",
                        "zz",
                        "nozzdiboson",
                        "vjets",
                        "vjetslnu",
                        "ttbar",
                        "qcd",
                    ],
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


def save_templates(
    templates: dict[str, Hist], template_file: Path, shape_var: ShapeVar, blind: bool = True
):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    from copy import deepcopy

    blind_window = shape_var.blind_window

    if blind_window is not None and blind:
        for label, template in list(templates.items()):
            blinded_template = deepcopy(template)
            utils.blindBins(blinded_template, blind_window)
            templates[f"{label}MCBlinded"] = blinded_template

    with template_file.open("wb") as f:
        pickle.dump(templates, f)

    print("Saved templates to", template_file)


if __name__ == "__main__":
    sys.exit()
