from __future__ import annotations

import argparse
import itertools
from collections import OrderedDict
from pathlib import Path

import correctionlib
import numpy as np
import pandas as pd
import uproot

from HH4b import postprocessing, utils

YEARS = ["2022", "2022EE", "2023", "2023BPix"]
YEARS_COMBINED_DICT = {
    "2022All": ["2022", "2022EE"],
    "2023All": ["2023", "2023BPix"],
}
SCRIPT_DIR = Path(__file__).resolve().parent

PROCESSED_PATH: Path = Path("Zbb_events_combined.pkl")
PROCESSED_PATH_ERAS: Path = Path("Zbb_events_eras.pkl")
APPLY_Zto2Q_CORR: bool = True
APPLY_TRIGGER_SF: bool = True

SAMPLES_DICT = {
    "data": [f"{key}_Run" for key in ["JetMET"]],
    "ttbar": ["TTto4Q", "TTtoLNu2Q"],
    "qcd": ["QCD_HT"],
    "hbb": ["GluGluHto2B_M-125"],
    "Zto2Q": ["Zto2Q-4Jets"],
    "Wto2Q": ["Wto2Q-3Jets"],
}
MC_SAMPLES_LIST = [sample for sample in SAMPLES_DICT if sample != "data"]

trigger_sf_dir = SCRIPT_DIR.parent / "corrections/data/trigger_sfs"


def parse_args():
    parser = argparse.ArgumentParser(description="Analysis script for Zbb events")

    parser.add_argument(
        "--txbb-bins",
        type=float,
        nargs="+",
        default=[0.95, 0.975, 0.99, 1.0],
        help="TXbb bins (default: [0.95, 0.975, 0.99, 1.0])",
    )

    parser.add_argument(
        "--pt-bins",
        type=float,
        nargs="+",
        default=[350, 450, 550, 10000],
        help="pT bins (default: [350, 450, 550, 10000])",
    )

    # m_low, m_high = 50, 150
    # bins = 5
    # n_mass_bins = int((m_high - m_low) / bins)

    parser.add_argument(
        "--mass-bin-size", type=int, default=5, help="Size of mass bins (default: 5 GeV)"
    )
    parser.add_argument(
        "--m-low", type=float, default=50.0, help="Lower bound of mass range (default: 50 GeV)"
    )
    parser.add_argument(
        "--m-high", type=float, default=150.0, help="Upper bound of mass range (default: 150 GeV)"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="reprocess the data from the ntuples (default: False)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Columns to load from the ntuples
    sys_vars = ["FSRPartonShower", "ISRPartonShower", "pileup"]

    fatjet_vars = [
        "bbFatJetPt",
        "bbFatJetEta",
        "bbFatJetMsd",
        "bbFatJetParTmassVis",
        "bbFatJetPNetMassLegacy",
        "bbFatJetParTTXbb",
    ]

    pt_variations = []
    for jesr, ud in itertools.product(["JES", "JER"], ["up", "down"]):
        pt_variations.append(f"bbFatJetPt_{jesr}_{ud}")

    mass_variations = []
    for jmsr, ud in itertools.product(["JMS", "JMR"], ["up", "down"]):
        mass_variations.append(f"bbFatJetMsd_{jmsr}_{ud}")
        mass_variations.append(f"bbFatJetParTmassVis_{jmsr}_{ud}")
        mass_variations.append(f"bbFatJetPNetMassLegacy_{jmsr}_{ud}")

    base_columns = [(var, 2) for var in fatjet_vars] + [("weight", 1)]
    print(f"Base columns: {base_columns}")

    triggers = {
        "2022": [
            "AK8PFJet500",
            "AK8PFJet420_MassSD30",
            "AK8PFJet425_SoftDropMass40",
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        ],
        "2022EE": [
            "AK8PFJet500",
            "AK8PFJet420_MassSD30",
            "AK8PFJet425_SoftDropMass40",
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        ],
        "2023": [
            "AK8PFJet500",
            "AK8PFJet420_MassSD30",
            "AK8PFJet425_SoftDropMass40",
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        ],
        "2023BPix": [
            "AK8PFJet500",
            "AK8PFJet420_MassSD30",
            "AK8PFJet425_SoftDropMass40",
            "AK8PFJet230_SoftDropMass40_PNetBB0p06",
        ],
    }

    load_columns_pt_var = []
    for pt_var in pt_variations:
        load_columns_pt_var.append((pt_var, 2))

    load_columns_mass_var = []
    for mass_var in mass_variations:
        load_columns_mass_var.append((mass_var, 2))

    load_weight_shifts = []
    for var, ud in itertools.product(sys_vars, ["Up", "Down"]):
        load_weight_shifts.append((f"weight_{var}{ud}", 1))

    MC_common_extra_columns = load_columns_mass_var + load_columns_pt_var + load_weight_shifts

    ZQQ_extra_columns = [("GenZPt", 1), ("GenZBB", 1), ("GenZCC", 1), ("bbFatJetVQQMatch", 2)]
    WQQ_extra_columns = [("GenWPt", 1), ("GenWCS", 1), ("GenWUD", 1), ("bbFatJetVQQMatch", 2)]

    extra_columns_dict = {
        "data": [],
        "qcd": load_weight_shifts,
        "ttbar": MC_common_extra_columns,
        "hbb": MC_common_extra_columns,
        "Zto2Q": MC_common_extra_columns + ZQQ_extra_columns,
        "Wto2Q": MC_common_extra_columns + WQQ_extra_columns,
    }
    print(f"Extra columns: {extra_columns_dict}")

    # Trigger efficiency corrections
    trigger_eff_txbb = {
        year: correctionlib.CorrectionSet.from_file(
            str(trigger_sf_dir / f"fatjet_triggereff_{year}_txbbGloParT_QCD.json")
        )
        for year in YEARS
    }
    trigger_eff_ptmsd = {
        year: correctionlib.CorrectionSet.from_file(
            str(trigger_sf_dir / f"fatjet_triggereff_{year}_ptmsd_QCD.json")
        )
        for year in YEARS
    }

    def _compute_SF(mc_eff_set, data_eff_set, *args):
        """Helper function to compute scale factor and error for a given efficiency set."""
        # Evaluate MC efficiencies
        mc_eff_nom = mc_eff_set.evaluate(*args, "nominal")
        mc_eff_err_up = mc_eff_set.evaluate(*args, "stat_up")
        mc_eff_err_down = mc_eff_set.evaluate(*args, "stat_dn")
        mc_eff_err = np.maximum(np.abs(mc_eff_err_up), np.abs(mc_eff_err_down))

        # Evaluate data efficiencies
        data_eff_nom = data_eff_set.evaluate(*args, "nominal")
        data_eff_up = data_eff_set.evaluate(*args, "stat_up")
        data_eff_down = data_eff_set.evaluate(*args, "stat_dn")
        data_eff_err = np.maximum(np.abs(data_eff_up), np.abs(data_eff_down))

        # Compute scale factor and propagate errors
        with np.errstate(divide="ignore", invalid="ignore"):
            sf_nom = data_eff_nom / mc_eff_nom
            sf_err = sf_nom * np.sqrt(
                (data_eff_err / data_eff_nom) ** 2 + (mc_eff_err / mc_eff_nom) ** 2
            )

        # set sf to 1 if mc_eff_nom is zero to avoid division by zero
        sf_nom = np.where(mc_eff_nom == 0, 1.0, sf_nom)
        sf_err = np.where(mc_eff_nom == 0, 0.0, sf_err)
        sf_err = np.where(data_eff_nom == 0, 0.0, sf_err)
        # sf_nom = np.where(sf_nom > 2.0, 1.0, sf_nom)  # restrict scale factor to a maximum of 2.0
        # sf_err = np.where(sf_nom > 2.0, 0.0, sf_err)  # restrict scale factor error to a maximum of 2.0

        return sf_nom, sf_err

    def eval_trigger_sf(
        txbb: np.ndarray, pt: np.ndarray, msd: np.ndarray, year: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate trigger scale factors with error propagation."""

        # txbb scale factor
        mc_eff_set = trigger_eff_txbb[year][f"fatjet_triggereffmc_{year}_txbbGloParT"]
        data_eff_set = trigger_eff_txbb[year][f"fatjet_triggereffdata_{year}_txbbGloParT"]
        sf_txbb_nom, sf_txbb_err = _compute_SF(mc_eff_set, data_eff_set, txbb)

        # ptmsd scale factor
        mc_eff_set = trigger_eff_ptmsd[year][f"fatjet_triggereffmc_{year}_ptmsd"]
        data_eff_set = trigger_eff_ptmsd[year][f"fatjet_triggereffdata_{year}_ptmsd"]
        sf_ptmsd_nom, sf_ptmsd_err = _compute_SF(mc_eff_set, data_eff_set, pt, msd)

        # Combine scale factors
        sf = sf_txbb_nom * sf_ptmsd_nom
        sf_err = sf * np.sqrt((sf_txbb_err / sf_txbb_nom) ** 2 + (sf_ptmsd_err / sf_ptmsd_nom) ** 2)

        sf_up = sf + sf_err
        sf_down = sf - sf_err

        return sf, sf_up, sf_down

    # if True, apply the Z->2Q corrections from ZMuMu measurement
    if APPLY_Zto2Q_CORR:
        print("Applying Z->2Q corrections from ZMuMu measurement...")
        corr_dir = Path("ZMuMu_corrs")
        corr_dict = {}

        for year in ["2022", "2023"]:
            corr_file = corr_dir / f"corr_{year}.json"
            if not corr_file.exists():
                raise FileNotFoundError(f"Correction file {corr_file} does not exist.")

            # Load the correction
            corr = correctionlib.CorrectionSet.from_file(str(corr_file))
            corr_dict[year] = corr
            print(f"Loaded correction for {year} from {corr_file}")
    else:
        corr_dict = None
        print("Z->2Q corrections are not applied.")

    if args.reprocess or not PROCESSED_PATH.exists():
        path_dir = "/ceph/cms/store/user/zichun/bbbb/skimmer/ZbbHT25May28_v12v2_private_zbb/"

        events_dict = {}
        for year in YEARS:
            events_dict[year] = {}

            # Have to load the samples separately because branches vary
            for sample, sample_list in SAMPLES_DICT.items():
                print(f"Loading {sample} for {year}...")
                triggers_cols = [(trigger, 1) for trigger in triggers[year]]

                # append the event dictionary for each year
                columns = triggers_cols + base_columns + extra_columns_dict.get(sample, [])
                dataframes = {
                    **utils.load_samples(
                        data_dir=path_dir,
                        samples={sample: sample_list},
                        year=year,
                        columns=utils.format_columns(columns),
                        variations=True,
                        weight_shifts=["FSRPartonShower", "ISRPartonShower", "pileup"],
                    )
                }
                # concatenate all dataframes in this sample
                events_dict[year][sample] = []
                for _, df in dataframes.items():
                    # if pT variations are not present, set them to pT
                    for pt_var in ["bbFatJetPt"] + pt_variations:
                        if pt_var not in df.columns:
                            for i in range(2):
                                df[f"{pt_var}{i}"] = df[("bbFatJetPt", i)].copy()

                    # if mass variations are not present, set them to mass
                    for mass_var in [
                        "bbFatJetMsd",
                        "bbFatJetParTmassVis",
                        "bbFatJetPNetMassLegacy",
                    ] + mass_variations:
                        if mass_var not in df.columns:
                            for i in range(2):
                                df[f"{mass_var}{i}"] = df[(mass_var.split("_")[0], i)].copy()

                    if sample != "data":
                        # evaluate trigger scale factors
                        sf, sf_up, sf_down = eval_trigger_sf(
                            txbb=df[("bbFatJetParTTXbb", 0)].values,
                            pt=df[("bbFatJetPt", 0)].values,
                            msd=df[("bbFatJetMsd", 0)].values,
                            year=year,
                        )
                        df["SF_trigger"] = sf
                        df["SF_trigger_up"] = sf_up
                        df["SF_trigger_down"] = sf_down

                    events_dict[year][sample].append(df)

                # concatenate all dataframes for this sample
                events_dict[year][sample] = pd.concat(events_dict[year][sample], ignore_index=True)

        # Combine events from different years into a single dictionary
        print("Combining events from different years...")
        events_combined = {year: {} for year in YEARS_COMBINED_DICT}
        for sample in SAMPLES_DICT:
            for combined_year, year_list in YEARS_COMBINED_DICT.items():
                events_combined[combined_year][sample] = pd.concat(
                    [events_dict[year][sample] for year in year_list if sample in events_dict[year]]
                )

        # Store events_combined as a pickle file
        with PROCESSED_PATH.open("wb") as f:
            pd.to_pickle(events_combined, f)
        print(f"Events combined and saved to {PROCESSED_PATH}")

        with PROCESSED_PATH_ERAS.open("wb") as f:
            pd.to_pickle(events_dict, f)
        print(f"Events by eras and saved to {PROCESSED_PATH_ERAS}")

        del events_dict  # Free memory

    else:
        # Directly load the processed file
        print(f"Loading events from {PROCESSED_PATH}...")
        with PROCESSED_PATH.open("rb") as f:
            events_combined = pd.read_pickle(f)
        print(f"Loaded events from {PROCESSED_PATH}")

    # apply ZQQ corrections if needed
    if APPLY_Zto2Q_CORR:
        print("Applying Zto2Q corrections...")
        for year in YEARS_COMBINED_DICT:
            # apply corrections to the events
            corr = corr_dict[year.replace("All", "")]["GenZPtWeight"]
            GenZ_pt = events_combined[year]["Zto2Q"]["GenZPt"].to_numpy()[:, 0]
            sf_nom = corr.evaluate(GenZ_pt, "nominal")
            sf_up = corr.evaluate(GenZ_pt, "stat_up")
            sf_down = corr.evaluate(GenZ_pt, "stat_down")
            events_combined[year]["Zto2Q"]["SF_GenZPt"] = sf_nom
            events_combined[year]["Zto2Q"]["SF_GenZPt_up"] = sf_up
            events_combined[year]["Zto2Q"]["SF_GenZPt_down"] = sf_down

            # apply the scale factors to the final weight
            weight = events_combined[year]["Zto2Q"]["finalWeight"]
            events_combined[year]["Zto2Q"]["finalWeight"] = weight * sf_nom
            events_combined[year]["Zto2Q"]["weight_GenZPtUp"] = weight * sf_up
            events_combined[year]["Zto2Q"]["weight_GenZPtDown"] = weight * sf_down
        print("Zto2Q corrections applied")

    # further split Zto2Q and Wto2Q events into different categories
    for year in YEARS_COMBINED_DICT:
        Zto2Q = events_combined[year]["Zto2Q"]
        matched = Zto2Q[("bbFatJetVQQMatch", 0)] == 1
        is_ZBB = Zto2Q[("GenZBB", 0)]
        is_ZCC = Zto2Q[("GenZCC", 0)]
        is_ZQQ = ~(is_ZBB | is_ZCC)  # u, d, s quarks
        ZtoBB = is_ZBB & matched
        ZtoCC = is_ZCC & matched
        ZtoQQ = is_ZQQ & matched
        Z_unmatched = ~matched
        events_combined[year]["Zto2Q_BB"] = Zto2Q[ZtoBB]
        events_combined[year]["Zto2Q_CC"] = Zto2Q[ZtoCC]
        events_combined[year]["Zto2Q_QQ"] = Zto2Q[ZtoQQ]
        events_combined[year]["Zto2Q_unmatched"] = Zto2Q[Z_unmatched]

    MC_SAMPLES_FINAL_LIST = MC_SAMPLES_LIST + [
        "Zto2Q_BB",
        "Zto2Q_CC",
        "Zto2Q_QQ",
        "Zto2Q_unmatched",
    ]

    # Pass and fail regions - use argparse values
    txbb_bins = args.txbb_bins
    min_txbb = txbb_bins[0]
    pt_bins = args.pt_bins

    print(f"Using TXbb bins: {txbb_bins}")
    print(f"Using pT bins: {pt_bins}")

    txbb_bins = list(zip(txbb_bins[:-1], txbb_bins[1:]))
    pt_bins = list(zip(pt_bins[:-1], pt_bins[1:]))

    # Mass bins
    m_low = args.m_low
    m_high = args.m_high
    bins = args.mass_bin_size
    n_mass_bins = int((m_high - m_low) / bins)

    def save_to_root(outfile: Path, templates: dict):
        with uproot.recreate(str(outfile)) as f_out:
            for category, template in templates.items():
                hist = templates[category]
                categories, _ = hist.axes
                for sample in list(categories):
                    h = template[{"Sample": sample}]
                    f_out[f"{sample}_{category}"] = h

    def select_triggers(events, trigger_list):
        print(f"Selecting events with triggers: {trigger_list}")
        events_filtered = {}

        for year in events:
            events_filtered[year] = {}
            for sample in events[year]:
                sample_df = events[year][sample].copy()
                mask = np.zeros(len(sample_df), dtype=bool)
                for trigger in trigger_list:
                    if trigger in sample_df.columns:
                        mask = mask | (sample_df[trigger].to_numpy().reshape(-1) == 1)
                events_filtered[year][sample] = sample_df[mask].copy()
                num_sel = mask.sum()
                num_total = len(sample_df)
                print(
                    f"Year: {year}, Sample: {sample}, Selected: {num_sel}, Total: {num_total}, Efficiency: {num_sel / num_total:.2%}"
                )
        return events_filtered

    trigger_list_high_pt = [
        "AK8PFJet500",
        "AK8PFJet420_MassSD30",
        "AK8PFJet425_SoftDropMass40",
    ]

    trigger_list_PNet = [
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
    ]

    events_high_pt = select_triggers(events_combined, trigger_list_high_pt)
    events_PNet = select_triggers(events_combined, trigger_list_PNet)

    # apply trigger sf to events_PNet
    if APPLY_TRIGGER_SF:
        print("Applying trigger scale factors to events_PNet...")
        for year in events_PNet:
            for sample in events_PNet[year]:
                sample_df = events_PNet[year][sample]
                if sample != "data":
                    sf_nom = sample_df["SF_trigger"]
                    sf_up = sample_df["SF_trigger_up"]
                    sf_down = sample_df["SF_trigger_down"]

                    # apply the scale factors to the final weight
                    weight = sample_df["finalWeight"]
                    sample_df["finalWeight"] = sample_df["finalWeight"] * sf_nom
                    sample_df["weight_TriggerUp"] = weight * sf_up
                    sample_df["weight_TriggerDown"] = weight * sf_down
                events_PNet[year][sample] = sample_df
    else:
        print("Trigger scale factors are not applied to events_PNet.")

    # bkg_keys = ["Zto2Q_CC", "Zto2Q_QQ", "Zto2Q_unmatched", "Wto2Q", "hbb", "ttbar", "qcd"]
    # sig_keys = ["Zto2Q_BB"]
    # use this if you want to include Zto2Q_BB in the stack plot
    bkg_keys = [
        "Zto2Q_BB",
        "Zto2Q_CC",
        "Zto2Q_QQ",
        "Zto2Q_unmatched",
        "Wto2Q",
        "hbb",
        "ttbar",
        "qcd",
    ]
    sig_keys = []
    bg_order = list(reversed(bkg_keys))

    jshift_keys = [""]
    for var, ud in itertools.product(["JES", "JER", "JMS", "JMR"], ["up", "down"]):
        jshift_keys.append(f"{var}_{ud}")

    weight_shifts = {
        "pileup": postprocessing.Syst(
            samples=MC_SAMPLES_FINAL_LIST, label="Pileup", years=list(YEARS_COMBINED_DICT.keys())
        ),
        # "pdf": postprocessing.Syst(samples=sig_keys, label="PDFAcc", years=list(YEARS_COMBINED_DICT.keys())),
        "ISRPartonShower": postprocessing.Syst(
            samples=MC_SAMPLES_FINAL_LIST,
            label="ISR Parton Shower",
            years=list(YEARS_COMBINED_DICT.keys()),
        ),
        "FSRPartonShower": postprocessing.Syst(
            samples=MC_SAMPLES_FINAL_LIST,
            label="FSR Parton Shower",
            years=list(YEARS_COMBINED_DICT.keys()),
        ),
    }

    if APPLY_Zto2Q_CORR:
        weight_shifts["GenZPt"] = postprocessing.Syst(
            samples=["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Zto2Q_unmatched"],
            label="Gen Z pT correction derived from ZMuMu",
            years=list(YEARS_COMBINED_DICT.keys()),
        )

    if APPLY_TRIGGER_SF:
        weight_shifts_trig_sf = {
            "Trigger": postprocessing.Syst(
                samples=MC_SAMPLES_FINAL_LIST,
                label="Trigger SF of the PNet trigger",
                years=list(YEARS_COMBINED_DICT.keys()),
            ),
        }
    else:
        weight_shifts_trig_sf = {}

    print("Making templates...")
    out_dir = SCRIPT_DIR / "templates_zbb"
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in YEARS_COMBINED_DICT:

        cutflows_dir = Path(f"{out_dir}/cutflows/{year}")
        cutflows_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = Path(f"{out_dir}/plots/{year}")
        plot_dir.mkdir(parents=True, exist_ok=True)

        template_dir = out_dir
        template_dir.mkdir(parents=True, exist_ok=True)

        templates = {}
        for jshift in jshift_keys:
            # Determine the pt and mass variations
            if jshift == "":
                pt_branch = "bbFatJetPt0"
                mass_branch = "bbFatJetParTmassVis0"
            elif jshift.startswith(("JES", "JER")):
                pt_branch = f"bbFatJetPt_{jshift}0"
                mass_branch = "bbFatJetParTmassVis0"
            elif jshift.startswith(("JMS", "JMR")):
                pt_branch = "bbFatJetPt0"
                mass_branch = f"bbFatJetParTmassVis_{jshift}0"
            else:
                raise ValueError(f"Unknown jshift: {jshift}")

            # Different different pass regions based on TXbb and pT bins
            selection_regions = {}
            for (txbb_low, txbb_high), (pt_low, pt_high) in itertools.product(txbb_bins, pt_bins):
                # Convert to strings
                txbb_low_str = str(txbb_low).replace(".", "p")
                txbb_high_str = str(txbb_high).replace(".", "p")
                pt_low_str = str(pt_low)
                pt_high_str = str(pt_high)
                region_key = (
                    f"pass_TXbb{txbb_low_str}to{txbb_high_str}_pT{pt_low_str}to{pt_high_str}"
                )

                # determine which trigger to use
                if pt_low < 550:
                    events = events_PNet[year]
                    weight_shifts_final = {
                        **weight_shifts,
                        **weight_shifts_trig_sf,
                    }
                    print(f"Using PNet trigger for {region_key} in {year}")
                else:
                    events = events_high_pt[year]
                    weight_shifts_final = weight_shifts
                    print(f"Using high pT trigger for {region_key} in {year}")

                cutflows = {}
                for sample in events:
                    cutflows[sample] = OrderedDict()
                    cutflows[sample]["Skimmer Preselection"] = events_combined[year][sample][
                        "finalWeight"
                    ].sum()
                    cutflows[sample]["HLT"] = events_combined[year][sample]["finalWeight"].sum()
                cutflows = pd.DataFrame.from_dict(cutflows).transpose()

                # Create a region
                selection_regions[region_key] = postprocessing.Region(
                    cuts={
                        pt_branch: [pt_low, pt_high],
                        mass_branch: [m_low, m_high],
                        "bbFatJetParTTXbb0": [txbb_low, txbb_high],
                    },
                    label=region_key,
                )

            selection_regions["fail"] = postprocessing.Region(
                cuts={
                    pt_branch: [pt_low, pt_high],
                    mass_branch: [m_low, m_high],
                    "bbFatJetParTTXbb0": [0.1, min(0.9, min_txbb)],
                },
                label="fail",
            )
            print(f"Selection regions for {year} with jshift {jshift}: {selection_regions.keys()}")

            fit_shape_var = postprocessing.ShapeVar(
                mass_branch,
                r"$m_\mathrm{reg}$ (GeV)",
                [n_mass_bins, m_low, m_high],
                reg=True,
            )

            ttemps = postprocessing.get_templates(
                events,
                year=year,
                sig_keys=sig_keys,
                plot_sig_keys=sig_keys,
                selection_regions=selection_regions,
                shape_vars=[fit_shape_var],
                systematics={},
                template_dir=out_dir,
                bg_keys=bkg_keys,
                bg_order=bg_order,
                bg_err_mcstat=False,
                prev_cutflow=cutflows,
                plot_dir=plot_dir,
                weight_key="finalWeight",
                weight_shifts=weight_shifts_final,
                plot_shifts=False,
                show=False,
                energy=13.6,
                jshift=jshift,
                blind=False,
            )
            templates = {**templates, **ttemps}
        # Save the templates to a file
        outfile = template_dir / f"templates_{year}.root"
        save_to_root(outfile, templates)
        # Save as a pickle file
        outfile_pickle = template_dir / f"templates_{year}.pkl"
        with outfile_pickle.open("wb") as f:
            pd.to_pickle(templates, f)

    print(f"Templates saved to {template_dir} as {outfile} and {outfile_pickle}")


if __name__ == "__main__":
    main()
