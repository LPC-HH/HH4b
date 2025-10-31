from __future__ import annotations

import argparse
import os

# temp
import warnings

import correctionlib.convert
import correctionlib.schemav2 as cs
import hist
import numpy as np
import pandas as pd
import uproot

from HH4b import postprocessing
from HH4b.postprocessing import Region
from HH4b.utils import ShapeVar

warnings.simplefilter(action="ignore", category=FutureWarning)


def trigger_scale_factor(year):
    sf_file = uproot.open(f"../corrections/data/JetHTTriggerEfficiency_{year}.root")

    hist_sf_Xbb0p0To0p9 = sf_file["efficiency_ptmass_Xbb0p0To0p9"].to_hist()
    hist_sf_Xbb0p9To0p95 = sf_file["efficiency_ptmass_Xbb0p9To0p95"].to_hist()
    hist_sf_Xbb0p95To0p98 = sf_file["efficiency_ptmass_Xbb0p95To0p98"].to_hist()
    hist_sf_Xbb0p98To1p0 = sf_file["efficiency_ptmass_Xbb0p98To1p0"].to_hist()

    mass_axis, pt_axis = hist_sf_Xbb0p0To0p9.axes
    h_combined = hist.Hist(
        hist.axis.Variable([0, 0.9, 0.95, 0.98, 1.0], name="xbb", label="AK8 Xbb"),
        mass_axis,
        pt_axis,
    )
    h_combined[0, :, :] = hist_sf_Xbb0p0To0p9.values()
    h_combined[1, :, :] = hist_sf_Xbb0p9To0p95.values()
    h_combined[2, :, :] = hist_sf_Xbb0p95To0p98.values()
    h_combined[3, :, :] = hist_sf_Xbb0p98To1p0.values()

    h_combined.name = "trigger_sf"
    h_combined.label = "trigger_sf"
    trigger_sf = correctionlib.convert.from_histogram(h_combined)
    trigger_sf.data.flow = "clamp"
    trigger_sf_evaluator = trigger_sf.to_evaluator()

    return trigger_sf_evaluator


def trigger_scale_factor_signal_correction(year):

    if year == "2016":
        signal_correction = cs.Correction(
            name="trigger_signal_correction",
            version=1,
            inputs=[
                cs.Variable(name="cat", type="int", description="Event category"),
                cs.Variable(name="pt", type="real", description="AK8 Pt"),
            ],
            output=cs.Variable(name="weight", type="real", description="Trigger signal correction"),
            data=cs.Category(
                nodetype="category",
                input="cat",
                content=[
                    cs.CategoryItem(
                        key=1,
                        value=cs.Binning(
                            nodetype="binning",
                            input="pt",
                            edges=[300, 350, 400, 450],
                            content=[1.70, 1.32, 1.10],
                            flow=1.0,
                        ),
                    ),
                    cs.CategoryItem(
                        key=2,
                        value=cs.Binning(
                            nodetype="binning",
                            input="pt",
                            edges=[300, 350, 400, 450],
                            content=[1.53, 1.21, 1.05],
                            flow=1.0,
                        ),
                    ),
                    cs.CategoryItem(
                        key=3,
                        value=cs.Binning(
                            nodetype="binning",
                            input="pt",
                            edges=[300, 350, 400, 450],
                            content=[1.50, 1.18, 1.06],
                            flow=1.0,
                        ),
                    ),
                    cs.CategoryItem(key=4, value=1.0),
                ],
            ),
        )

    return signal_correction


def load_run2_samples(args, year):
    trigger_sf_evaluator = trigger_scale_factor(year)

    # also here: /eos/uscms/store/user/lpcdihiggsboost/cmantill/Run2Analysis/HH/HHTo4BNtupler/20220217/
    path_to_dir_run2 = f"root://cmsxrootd.fnal.gov//store/user/cmantill/analyzer/{args.tag}/"

    lumi = {
        "2016": 36330.0,
        "2017": 41480.0,
        "2018": 59830.0,
    }

    samples_run2 = {
        "2016": {
            "hh4b": [
                "GluGluToHHTo4B_node_cHHH1_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8_1pb_weighted_Testing_BDTs.root"
            ],
            # "vbfhh4b": [
            #    "VBF_HH_CV_1_C2V_1_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8_1pb_weighted_BDTs.root",
            # ],
            "data": [
                "JetHT_2016B-ver2_BDTs.root",
                "JetHT_2016C_BDTs.root",
                "JetHT_2016D_BDTs.root",
                "JetHT_2016E_BDTs.root",
                "JetHT_2016F_BDTs.root",
                "JetHT_2016G_BDTs.root",
                "JetHT_2016H_BDTs.root",
            ],
            "ttbar": [
                "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8_1pb_weighted_Testing_BDTs.root",
                "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8_1pb_weighted_Testing_BDTs.root",
            ],
            "qcd": [
                "QCD_HT1000to1500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-combined_1pb_weighted_Testing_BDTs.root",
                "QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-combined_1pb_weighted_Testing_BDTs.root",
                "QCD_HT2000toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-combined_1pb_weighted_Testing_BDTs.root",
                "QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-combined_1pb_weighted_Testing_BDTs.root",
                "QCD_HT300to500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-combined_1pb_weighted_Testing_BDTs.root",
                "QCD_HT500to700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-combined_1pb_weighted_Testing_BDTs.root",
                "QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-combined_1pb_weighted_Testing_BDTs.root",
            ],
            "vhtobb": [
                "WminusH_HToBB_WToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "WplusH_HToBB_WToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "ZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "ggZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
            ],
            "novhhtobb": [
                "GluGluHToBB_M-125_13TeV_powheg_MINLO_NNLOPS_pythia8_1pb_weighted_BDTs.root",
                "VBFHToBB_M-125_13TeV_powheg_pythia8_weightfix_1pb_weighted_BDTs.root",
                "ttHTobb_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
            ],
            "dibosonvjets": [
                "WW_TuneCUETP8M1_13TeV-pythia8-combined_1pb_weighted_BDTs.root ",
                "WZ_TuneCUETP8M1_13TeV-pythia8-combined_1pb_weighted_BDTs.root",
                "ZZ_TuneCUETP8M1_13TeV-pythia8-combined_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT-800toInf_qc19_3j_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT400to600_qc19_3j_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT600to800_qc19_3j_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT-800toInf_qc19_4j_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT400to600_qc19_4j_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT600to800_qc19_4j_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
            ],
        },
        "2017": {
            "hh4b": [
                "GluGluToHHTo4B_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8_1pb_weighted_Testing_BDTs.root",
            ],
            "data": [
                "JetHT_2017B_BDTs.root",
                "JetHT_2017C_BDTs.root",
                "JetHT_2017D_BDTs.root",
                "JetHT_2017F_BDTs.root",
            ],
            "ttbar": [
                "TTToHadronic_TuneCP5_13TeV-powheg-pythia8-combined_1pb_weighted_Testing_BDTs.root",
                "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8-combined_1pb_weighted_Testing_BDTs.root",
            ],
            "qcd": [
                "QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT500to700_TuneCP5_13TeV-madgraph-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8_1pb_weighted_Testing_BDTs.root",
            ],
            "vhtobb": [
                "WminusH_HToBB_WToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "WplusH_HToBB_WToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "ZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "ggZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
            ],
            "novhhtobb": [
                "GluGluHToBB_M-125_13TeV_powheg_MINLO_NNLOPS_pythia8_1pb_weighted_BDTs.root",
                "VBFHToBB_M-125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_1pb_weighted_BDTs.root",
            ],
            "dibosonvjets": [
                "WW_TuneCP5_13TeV-pythia8_1pb_weighted_BDTs.root",
                "WZ_TuneCP5_13TeV-pythia8_1pb_weighted_BDTs.root",
                "ZZ_TuneCP5_13TeV-pythia8_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT-800toInf_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT400to600_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT600to800_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT-800toInf_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT400to600_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT600to800_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
            ],
        },
        "2018": {
            "hh4b": [
                "GluGluToHHTo4B_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8_1pb_weighted_Testing_BDTs.root",
            ],
            # "hh4b-kl0": [
            #     "GluGluToHHTo4B_node_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8_1pb_weighted_BDTs.root",
            # ],
            # "hh4b-kl2p45": [
            #     "GluGluToHHTo4B_node_cHHH2p45_TuneCP5_PSWeights_13TeV-powheg-pythia8_1pb_weighted_BDTs.root",
            # ],
            "data": [
                "JetHT_2018A_BDTs.root",
                "JetHT_2018B_BDTs.root",
                "JetHT_2018C_BDTs.root",
                "JetHT_2018D_BDTs.root",
            ],
            "ttbar": [
                "TTToHadronic_TuneCP5_13TeV-powheg-pythia8-combined_1pb_weighted_Testing_BDTs.root",
                "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8-combined_1pb_weighted_Testing_BDTs.root",
            ],
            "qcd": [
                "QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_Testing_BDTs.root",
                "QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_Testing_BDTs.root",
            ],
            "vhtobb": [
                "WminusH_HToBB_WToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "WplusH_HToBB_WToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "ZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
                "ggZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8_1pb_weighted_BDTs.root",
            ],
            "novhhtobb": [
                "GluGluHToBB_M-125_13TeV_powheg_MINLO_NNLOPS_pythia8_1pb_weighted_BDTs.root",
                "VBFHToBB_M-125_13TeV_powheg_pythia8_weightfix_1pb_weighted_BDTs.root",
                "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_1pb_weighted_BDTs.root",
            ],
            "dibosonvjets": [
                "WW_TuneCP5_13TeV-pythia8_1pb_weighted_BDTs.root",
                "WZ_TuneCP5_13TeV-pythia8_1pb_weighted_BDTs.root",
                "ZZ_TuneCP5_13TeV-pythia8_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT-800toInf_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT400to600_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "WJetsToQQ_HT600to800_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT-800toInf_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT400to600_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
                "ZJetsToQQ_HT600to800_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8_1pb_weighted_BDTs.root",
            ],
        },
    }

    columns = [
        "run",
        "luminosityBlock",
        "event",
        "fatJet1Pt",
        "fatJet1Eta",
        "fatJet1Phi",
        "fatJet1Mass",
        "fatJet1MassSD",
        "fatJet1MassRegressed",
        "fatJet1PNetXbb",
        "fatJet1PNetQCDb",
        "fatJet1PNetQCDbb",
        "fatJet1PNetQCDothers",
        "fatJet1Tau3OverTau2",
        "fatJet2Pt",
        "fatJet2Eta",
        "fatJet2Phi",
        "fatJet2Mass",
        "fatJet2MassSD",
        "fatJet2PNetXbb",
        "fatJet2PNetQCDb",
        "fatJet2PNetQCDbb",
        "fatJet2PNetQCDothers",
        "fatJet2Tau3OverTau2",
        "fatJet2MassRegressed",
        "fatJet1PtOverMHH",
        "fatJet2PtOverMHH",
        "ptj2_over_ptj1",
        "hh_pt",
        "hh_eta",
        "hh_mass",
        "met",
        "disc_qcd_and_ttbar_Run2_enhanced_v8p2",
    ]

    if year == "2016":
        columns_data = [
            "HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20",
            "HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV_p20",
            "HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20",
            "HLT_AK8PFJet360_TrimMass30",
            "HLT_AK8PFJet450",
            "HLT_PFJet450",
        ]
    elif year == "2017":
        columns_data = [
            "HLT_PFJet450",
            "HLT_PFJet500",
            "HLT_AK8PFJet500",
            "HLT_PFHT1050",
        ]
    else:
        columns_data = [
            "HLT_PFJet450",
            "HLT_PFJet500",
            "HLT_AK8PFJet500",
            "HLT_PFHT1050",
            "HLT_AK8PFJet360_TrimMass30",
            "HLT_AK8PFJet380_TrimMass30",
            "HLT_AK8PFJet400_TrimMass30",
            "HLT_AK8PFHT800_TrimMass50",
            "HLT_AK8PFHT750_TrimMass50",
        ]

    columns_mc = [
        "xsecWeight",
        "weight",
        "genWeight",
        "l1PreFiringWeight",
        "puWeight",
    ]

    # pre-selection
    cut_string_presel = (
        "(fatJet1Pt>300) & (fatJet2Pt>300) & (fatJet1MassSD>50) & (fatJet2MassRegressed>50)"
    )
    if year == "2016":
        cut_string_trig = "((HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20) | (HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV_p20) | (HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20) | (HLT_AK8PFJet360_TrimMass30) | (HLT_AK8PFJet450) | (HLT_PFJet450) ) "
    elif year == "2017":
        cut_string_trig = "((HLT_PFJet450) | (HLT_PFJet500) | (HLT_AK8PFJet500) | (HLT_PFHT1050) )"
    else:
        cut_string_trig = "((HLT_PFHT1050) | (HLT_PFJet500) | (HLT_AK8PFJet500) | (HLT_AK8PFJet400_TrimMass30) | (HLT_AK8PFHT800_TrimMass50) )"

    dfs_dict = {}
    for key, datasets in samples_run2[year].items():
        dfs_dict[key] = []
        columns_to_load = columns + columns_data if "data" in key else (columns + columns_mc)

        for dset in datasets:
            if key == "data":
                if ("2018A" not in dset) and year == "2018":
                    columns_to_load_dset = columns_to_load + [
                        "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4"
                    ]
                    cut_string_trig_dset = (
                        f"{cut_string_trig} | (HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4)"
                    )
                    cut_string_dset = f"{cut_string_presel} & ({cut_string_trig_dset})"
                elif "2017F" in dset and year == "2017":
                    columns_to_load_dset = columns_to_load + [
                        "HLT_AK8PFJet360_TrimMass30",
                        "HLT_AK8PFJet380_TrimMass30",
                        "HLT_AK8PFJet400_TrimMass30",
                        "HLT_AK8PFHT800_TrimMass50",
                        "HLT_AK8PFHT750_TrimMass50",
                        "HLT_AK8PFJet330_PFAK8BTagCSV_p17",
                    ]
                    cut_string_trig_dset = f"{cut_string_trig} | (HLT_AK8PFJet360_TrimMass30) | (HLT_AK8PFJet380_TrimMass30) | (HLT_AK8PFJet400_TrimMass30) | (HLT_AK8PFHT800_TrimMass50) | (HLT_AK8PFHT750_TrimMass50) | (HLT_AK8PFJet330_PFAK8BTagCSV_p17)"
                    cut_string_dset = f"{cut_string_presel} & ({cut_string_trig_dset})"
                elif ("2017C" in dset and "2017D" in dset) and year == "2017":
                    columns_to_load_dset = columns_to_load + [
                        "HLT_AK8PFJet360_TrimMass30",
                        "HLT_AK8PFJet380_TrimMass30",
                        "HLT_AK8PFJet400_TrimMass30",
                        "HLT_AK8PFHT800_TrimMass50",
                        "HLT_AK8PFHT750_TrimMass50",
                    ]
                    cut_string_trig_dset = f"{cut_string_trig} | (HLT_AK8PFJet360_TrimMass30) | (HLT_AK8PFJet380_TrimMass30) | (HLT_AK8PFJet400_TrimMass30) | (HLT_AK8PFHT800_TrimMass50) | (HLT_AK8PFHT750_TrimMass50)"
                    cut_string_dset = f"{cut_string_presel} & ({cut_string_trig_dset})"
                else:
                    columns_to_load_dset = columns_to_load
                    cut_string_dset = f"{cut_string_presel} & ({cut_string_trig})"
            else:
                columns_to_load_dset = columns_to_load
                cut_string_dset = cut_string_presel

            df_events = uproot.open(f"{path_to_dir_run2}/{year}/{dset}:Events").arrays(
                columns_to_load_dset, cut_string_dset, library="pd"
            )

            # define category as boolean
            df_events["Category"] = 5  # not used
            df_events.loc[
                (df_events["fatJet2PNetXbb"] > 0.98)
                & (df_events["disc_qcd_and_ttbar_Run2_enhanced_v8p2"] > 0.43),
                "Category",
            ] = 1
            df_events.loc[
                (df_events["Category"] != 1)
                & (
                    (
                        (df_events["fatJet2PNetXbb"] > 0.98)
                        & (df_events["disc_qcd_and_ttbar_Run2_enhanced_v8p2"] > 0.11)
                    )
                    | (
                        (df_events["fatJet2PNetXbb"] > 0.95)
                        & (df_events["disc_qcd_and_ttbar_Run2_enhanced_v8p2"] > 0.43)
                    )
                ),
                "Category",
            ] = 2
            df_events.loc[
                (df_events["Category"] != 1)
                & (df_events["Category"] != 2)
                & (
                    (df_events["fatJet2PNetXbb"] > 0.95)
                    & (df_events["disc_qcd_and_ttbar_Run2_enhanced_v8p2"] > 0.03)
                ),
                "Category",
            ] = 3
            df_events.loc[
                (df_events["fatJet2PNetXbb"] < 0.95)
                & (df_events["disc_qcd_and_ttbar_Run2_enhanced_v8p2"] > 0.03),
                "Category",
            ] = 4

            # add trigger weight for mc
            trigger_weight = 1.0 - (
                1.0
                - trigger_sf_evaluator.evaluate(
                    df_events["fatJet1PNetXbb"], df_events["fatJet1MassSD"], df_events["fatJet1Pt"]
                )
            ) * (
                1.0
                - trigger_sf_evaluator.evaluate(
                    df_events["fatJet2PNetXbb"], df_events["fatJet2MassSD"], df_events["fatJet2Pt"]
                )
            )

            # event weights
            if "hh4b" in key:
                df_events["EventWeight"] = (
                    df_events["xsecWeight"]
                    * df_events["weight"]
                    * lumi[year]
                    * trigger_weight
                    * df_events["l1PreFiringWeight"]
                    * df_events["puWeight"]
                )
            elif key == "data":
                df_events["EventWeight"] = 1
            else:
                df_events["EventWeight"] = (
                    df_events["xsecWeight"]
                    * df_events["genWeight"]
                    * lumi[year]
                    * trigger_weight
                    * df_events["l1PreFiringWeight"]
                    * df_events["puWeight"]
                )

            dfs_dict[key].append(df_events[["EventWeight", "fatJet2MassRegressed", "Category"]])

    events_dict = {key: pd.concat(dfs) for key, dfs in dfs_dict.items()}
    return events_dict


def postprocess_run2(args):
    selection_regions = {
        "pass_bin1": Region(
            cuts={
                "Category": [1, 2],
            },
            label="Bin1",
        ),
        "pass_bin2": Region(
            cuts={
                "Category": [2, 3],
            },
            label="Bin1",
        ),
        "pass_bin3": Region(
            cuts={
                "Category": [3, 4],
            },
            label="Bin1",
        ),
        "fail": Region(
            cuts={
                "Category": [4, 5],
            },
            label="Fail",
        ),
    }

    # variable to fit
    fit_shape_var = ShapeVar(
        "fatJet2MassRegressed",
        r"$m^{2}_\mathrm{Reg}$ (GeV)",
        [17, 50, 220],
        reg=True,
        # blind_window=[110, 140],
    )

    data_yield = {}
    sig_yield = {}

    # load samples
    years = args.years.split(",")
    for year in years:
        events_dict = load_run2_samples(args, year)

        templ_dir = f"./templates/{args.template_dir}"
        os.system(f"mkdir -p {templ_dir}/cutflows/{year}")
        os.system(f"mkdir -p {templ_dir}/{year}")

        bkg_keys = ["qcd", "ttbar", "vhtobb", "dibosonvjets", "novhhtobb"]

        # individual templates per year
        templates = postprocessing.get_templates(
            events_dict,
            bb_masks=None,
            year=year,
            sig_keys=["hh4b"],
            selection_regions=selection_regions,
            shape_vars=[fit_shape_var],
            systematics={},
            template_dir=templ_dir,
            bg_keys=bkg_keys,
            plot_dir=f"{templ_dir}/{year}",
            weight_key="EventWeight",
            show=True,
            energy=13,
        )

        data_yield[year] = {
            "pass_bin1": templates["pass_bin1"][{"Sample": "data"}][6:9].sum().value,
            "pass_bin2": templates["pass_bin2"][{"Sample": "data"}][6:9].sum().value,
            "pass_bin3": templates["pass_bin3"][{"Sample": "data"}][6:9].sum().value,
        }
        sig_yield[year] = {
            "pass_bin1": templates["pass_bin1"][{"Sample": "hh4b"}][6:9].sum().value,
            "pass_bin2": templates["pass_bin2"][{"Sample": "hh4b"}][6:9].sum().value,
            "pass_bin3": templates["pass_bin3"][{"Sample": "hh4b"}][6:9].sum().value,
        }

        # save templates per year
        postprocessing.save_templates(templates, f"{templ_dir}/{year}_templates.pkl", fit_shape_var)

    bins = ["pass_bin1", "pass_bin2", "pass_bin3"]
    print("Data ", data_yield)
    print("Data Run 2", [np.sum([data_yield[year][bin] for year in years]) for bin in bins])
    print("Signal ", sig_yield)
    print("Signal Run 2", [np.sum([sig_yield[year][bin] for year in years]) for bin in bins])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template-dir",
        type=str,
        required=True,
        help="output pickle directory of hist.Hist templates",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="tag for input ntuples",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016,2017,2018",
        help="years to postprocess",
    )
    args = parser.parse_args()

    postprocess_run2(args)
