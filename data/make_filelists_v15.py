"""
Create filelists for each sample using the new flat sample mapping structure.
Takes a sample config JSON and creates txt files containing lists of root files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MC_SAMPLES_DICT = {
    "2016": {
        "Diboson": {
            "WW_13TeV": "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WZ_13TeV": "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ZZ_13TeV": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
        },
        # TODO: Add HH (GluGlutoHHto4B*) samples
        "HH": {
            # "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
        },
        "Hbb": {
            "GluGluHto2B_M-125_13TeV": "/GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext1-v1/NANOAODSIM",
            "GluGluHto2B_PT-200_M-125_13TeV": "/GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v2/NANOAODSIM",
            "VBFHto2B_M-125_dipoleRecoilOn_13TeV": "/VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WminusH_Hto2B_Wto2Q_M-125_13TeV": "/WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WminusH_Hto2B_WtoLNu_M-125_13TeV": "/WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WplusH_Hto2B_Wto2Q_M-125_13TeV": "/WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WplusH_Hto2B_WtoLNu_M-125_13TeV": "/WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2L_M-125_13TeV": "/ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2Nu_M-125_13TeV": "/ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2Q_M-125_13TeV": "/ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            # TODO: ZH_Hto2C_Zto2Q_M-125
            # "ZH_Hto2C_Zto2Q_M-125": "",
            "ggZH_Hto2B_Zto2L_M-125_13TeV": "/ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ggZH_Hto2B_Zto2Nu_M-125_13TeV": "/ggZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ggZH_Hto2B_Zto2Q_M-125_13TeV": "/ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            # TODO: ggZH_Hto2C_Zto2Q_M-125"
            # "ggZH_Hto2C_Zto2Q_M-125": "",
            "ttHto2B_M-125_13TeV": "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
        },
        "QCD": {
            "QCD_HT-100to200_13TeV": "/QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "QCD_HT-200to300_13TeV": "/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "QCD_HT-300to500_13TeV": "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "QCD_HT-500to700_13TeV": "/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "QCD_HT-700to1000_13TeV": "/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "QCD_HT-1000to1500_13TeV": "/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "QCD_HT-1500to2000_13TeV": "/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "QCD_HT-2000_13TeV": "/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
        },
        # TODO: Add SingleTop samples
        "SingleTop": {
            # "TBbarQ_t-channel_4FS": "",
            # "TWminusto4Q": "",
            # "TWminustoLNu2Q": "",
            # "TbarBQ_t-channel_4FS": "",
            # "TbarWplusto4Q": "",
            # "TbarWplustoLNu2Q": "",
        },
        "TT": {
            "TTto2L2Nu_13TeV": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v2/NANOAODSIM",
            "TTto4Q_13TeV": "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v2/NANOAODSIM",
            "TTtoLNu2Q_13TeV": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v2/NANOAODSIM",
        },
        # TODO: Add VBFHH samples
        "VBFHH": {
            # "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "",
        },
        "VJets": {
            "DYJetsToLL_M-50_13TeV_NLO": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "DYJetsToLL_M-50_13TeV_LO": "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            # TODO: Vto2Q NLO samples
            # Vto2Q LO samples
            "WJetsToQQ_HT-200to400_13TeV": "/WJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-400to600_13TeV": "/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-600to800_13TeV": "/WJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-800_13TeV": "/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-200to400_13TeV": "/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-400to600_13TeV": "/ZJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-600to800_13TeV": "/ZJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-800_13TeV": "/ZJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            # WtoLNu LO
            "WJetsToLNu_HT-70To100_13TeV": "/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-100To200_13TeV": "/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-200To400_13TeV": "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-400To600_13TeV": "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-600To800_13TeV": "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-800To1200_13TeV": "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-1200To2500_13TeV": "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-2500_13TeV": "/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            # WtoLNu NLO
            "WJetsToLNu_0J_13TeV": "/WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_1J_13TeV": "/WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
            "WJetsToLNu_2J_13TeV": "/WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1/NANOAODSIM",
        },
    },
    "2017": {
        "Diboson": {
            "WW_13TeV": "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WZ_13TeV": "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ZZ_13TeV": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
        },
        # TODO: Add HH (GluGlutoHHto4B*) samples
        "HH": {
            # "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
        },
        "Hbb": {
            "GluGluHto2B_M-125_13TeV": "/GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext1-v1/NANOAODSIM",
            "GluGluHto2B_PT-200_M-125_13TeV": "/GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v2/NANOAODSIM",
            "VBFHto2B_M-125_dipoleRecoilOn_13TeV": "/VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WminusH_Hto2B_Wto2Q_M-125_13TeV": "/WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WminusH_Hto2B_WtoLNu_M-125_13TeV": "/WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WplusH_Hto2B_Wto2Q_M-125_13TeV": "/WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WplusH_Hto2B_WtoLNu_M-125_13TeV": "/WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2L_M-125_13TeV": "/ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2Nu_M-125_13TeV": "/ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2Q_M-125_13TeV": "/ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            # TODO: ZH_Hto2C_Zto2Q_M-125
            # "ZH_Hto2C_Zto2Q_M-125": "",
            "ggZH_Hto2B_Zto2L_M-125_13TeV": "/ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ggZH_Hto2B_Zto2Nu_M-125_13TeV": "/ggZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ggZH_Hto2B_Zto2Q_M-125_13TeV": "/ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            # TODO: ggZH_Hto2C_Zto2Q_M-125"
            # "ggZH_Hto2C_Zto2Q_M-125": "",
            "ttHto2B_M-125_13TeV": "125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
        },
        "QCD": {
            "QCD_HT-100to200_13TeV": "/QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-200to300_13TeV": "/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-300to500_13TeV": "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-500to700_13TeV": "/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-700to1000_13TeV": "/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-1000to1500_13TeV": "/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-1500to2000_13TeV": "/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-2000_13TeV": "/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
        },
        # TODO: Add SingleTop samples
        "SingleTop": {
            # "TBbarQ_t-channel_4FS": "",
            # "TWminusto4Q": "",
            # "TWminustoLNu2Q": "",
            # "TbarBQ_t-channel_4FS": "",
            # "TbarWplusto4Q": "",
            # "TbarWplustoLNu2Q": "",
        },
        "TT": {
            "TTto2L2Nu_13TeV": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v2/NANOAODSIM",
            "TTto4Q_13TeV": "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v2/NANOAODSIM",
            "TTtoLNu2Q_13TeV": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v2/NANOAODSIM",
        },
        # TODO: Add VBFHH samples
        "VBFHH": {
            # "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "",
        },
        "VJets": {
            "DYJetsToLL_M-50_13TeV_NLO": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "DYJetsToLL_M-50_13TeV_LO": "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v2/NANOAODSIM",
            # TODO: Vto2Q NLO samples
            # Vto2Q LO samples
            "WJetsToQQ_HT-200to400_13TeV": "/WJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-400to600_13TeV": "/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-600to800_13TeV": "/WJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-800_13TeV": "/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-200to400_13TeV": "/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-400to600_13TeV": "/ZJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-600to800_13TeV": "/ZJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-800_13TeV": "/ZJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            # WtoLNu LO
            "WJetsToLNu_HT-70To100_13TeV": "/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-100To200_13TeV": "/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-200To400_13TeV": "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-400To600_13TeV": "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-600To800_13TeV": "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-800To1200_13TeV": "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-1200To2500_13TeV": "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-2500_13TeV": "/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            # WtoLNu NLO
            "WJetsToLNu_0J_13TeV": "/WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_1J_13TeV": "/WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_2J_13TeV": "/WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1/NANOAODSIM",
        },
    },
    "2018": {
        "Diboson": {
            "WW_13TeV": "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WZ_13TeV": "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ZZ_13TeV": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
        },
        # TODO: Add HH (GluGlutoHHto4B*) samples
        "HH": {
            # "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
            # "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "",
        },
        "Hbb": {
            "GluGluHto2B_M-125_13TeV": "/GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext1-v1/NANOAODSIM",
            "GluGluHto2B_PT-200_M-125_13TeV": "/GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v2/NANOAODSIM",
            "VBFHto2B_M-125_dipoleRecoilOn_13TeV": "/VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WminusH_Hto2B_Wto2Q_M-125_13TeV": "/WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WminusH_Hto2B_WtoLNu_M-125_13TeV": "/WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WplusH_Hto2B_Wto2Q_M-125_13TeV": "/WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WplusH_Hto2B_WtoLNu_M-125_13TeV": "/WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2L_M-125_13TeV": "/ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2Nu_M-125_13TeV": "/ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ZH_Hto2B_Zto2Q_M-125_13TeV": "/ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            # TODO: ZH_Hto2C_Zto2Q_M-125
            # "ZH_Hto2C_Zto2Q_M-125": "",
            "ggZH_Hto2B_Zto2L_M-125_13TeV": "/ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ggZH_Hto2B_Zto2Nu_M-125_13TeV": "/ggZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ggZH_Hto2B_Zto2Q_M-125_13TeV": "/ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            # TODO: ggZH_Hto2C_Zto2Q_M-125"
            # "ggZH_Hto2C_Zto2Q_M-125": "",
            "ttHto2B_M-125": "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
        },
        "QCD": {
            "QCD_HT-100to200_13TeV": "/QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-200to300_13TeV": "/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-300to500_13TeV": "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-500to700_13TeV": "/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-700to1000_13TeV": "/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-1000to1500_13TeV": "/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-1500to2000_13TeV": "/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "QCD_HT-2000_13TeV": "/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
        },
        # TODO: Add SingleTop samples
        "SingleTop": {
            # "TBbarQ_t-channel_4FS": "",
            # "TWminusto4Q": "",
            # "TWminustoLNu2Q": "",
            # "TbarBQ_t-channel_4FS": "",
            # "TbarWplusto4Q": "",
            # "TbarWplustoLNu2Q": "",
        },
        "TT": {
            "TTto2L2Nu_13TeV": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v2/NANOAODSIM",
            "TTto4Q_13TeV": "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v2/NANOAODSIM",
            "TTtoLNu2Q_13TeV": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v2/NANOAODSIM",
        },
        # TODO: Add VBFHH samples
        "VBFHH": {
            # "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "",
            # "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "",
        },
        "VJets": {
            "DYJetsToLL_M-50_13TeV_NLO": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "DYJetsToLL_M-50_13TeV_LO": "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v2/NANOAODSIM",
            # TODO: Vto2Q NLO samples
            # Vto2Q LO samples
            "WJetsToQQ_HT-200to400_13TeV": "/WJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-400to600_13TeV": "/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-600to800_13TeV": "/WJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToQQ_HT-800_13TeV": "/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-200to400_13TeV": "/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-400to600_13TeV": "/ZJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-600to800_13TeV": "/ZJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "ZJetsToQQ_HT-800_13TeV": "/ZJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            # WtoLNu LO
            "WJetsToLNu_HT-70To100_13TeV": "/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-100To200_13TeV": "/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-200To400_13TeV": "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-400To600_13TeV": "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-600To800_13TeV": "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-800To1200_13TeV": "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_HT-1200To2500_13TeV": "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            # TODO: WJetsToLNu_HT-2500
            # "WJetsToLNu_HT-2500": "",
            # WtoLNu NLO
            "WJetsToLNu_0J_13TeV": "/WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_1J_13TeV": "/WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
            "WJetsToLNu_2J_13TeV": "/WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1/NANOAODSIM",
        },
    },
    "2024": {
        "Diboson": {
            "WW": "/WW_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
        "HH": {
            "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/GluGluHHto4B_Par-c2-0p00-kl-0p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-PowhegBugFix_150X_mcRun3_2024_realistic_v2-v1/NANOAODSIM",
            "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/GluGluHHto4B_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-PowhegBugFix_150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/GluGluHHto4B_Par-c2-0p00-kl-2p45-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-PowhegBugFix_150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/GluGluHHto4B_Par-c2-0p00-kl-5p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-PowhegBugFix_150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
        "Hbb": {
            "GluGluHto2B_M-125": "/GluGluH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "GluGluHto2B_PT-200_M-125": "/GluGluH-Hto2B_Bin-PT-200_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHto2B_M-125": "/VBFH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WminusH_Hto2B_Wto2Q_M-125": "/WminusH-Wto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WminusH_Hto2B_WtoLNu_M-125": "/WminusH-WtoLNu-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WplusH_Hto2B_Wto2Q_M-125": "/WplusH-Wto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WplusH_Hto2B_WtoLNu_M-125": "/WplusH-WtoLNu-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ZH_Hto2B_Zto2L_M-125": "/ZH-Zto2L-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ZH_Hto2B_Zto2Nu_M-125": "/ZH-Zto2Nu-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ZH_Hto2B_Zto2Q_M-125": "/ZH-Zto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ZH_Hto2C_Zto2Q_M-125": "/ZH-Zto2Q-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ggZH_Hto2B_Zto2L_M-125": "/GluGluZH-Zto2L-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ggZH_Hto2B_Zto2Nu_M-125": "/GluGluZH-Zto2Nu-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ggZH_Hto2B_Zto2Q_M-125": "/GluGluZH-Zto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ggZH_Hto2C_Zto2Q_M-125": "/GluGluZH-Zto2Q-Hto2C_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "ttHto2B_M-125": "/TTH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
        "QCD": {
            "QCD_HT-40to70": "/QCD-4Jets_Bin-HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-70to100": "/QCD-4Jets_Bin-HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-100to200": "/QCD-4Jets_Bin-HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-200to400": "/QCD-4Jets_Bin-HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-400to600": "/QCD-4Jets_Bin-HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-600to800": "/QCD-4Jets_Bin-HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-800to1000": "/QCD-4Jets_Bin-HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-1000to1200": "/QCD-4Jets_Bin-HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-1200to1500": "/QCD-4Jets_Bin-HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-1500to2000": "/QCD-4Jets_Bin-HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "QCD_HT-2000": "/QCD-4Jets_Bin-HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
        "SingleTop": {
            "TBbarQ_t-channel_4FS": "/TBbarQto2Q-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "TWminusto4Q": "/TWminusto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "TWminustoLNu2Q": "/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "TbarBQ_t-channel_4FS": "/TbarBQto2Q-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "TbarWplusto4Q": "/TbarWplusto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "TbarWplustoLNu2Q": "/TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
        "TT": {
            "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
        "VBFHH": {
            "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-1p74-C2V-1p37-C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-m0p012-C2V-0p030-C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-m0p758-C2V-1p44-C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-m0p962-C2V-0p959-C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-m1p21-C2V-1p94-C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-m1p60-C2V-2p72-C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-m1p83-C2V-3p57-C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-2p12-C2V-3p87-C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-1-C2V-0-C3-1_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-1-C2V-1-C3-1_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
        "VJets": {
            # DYto2L NLO samples
            "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            # Vto2Q NLO samples
            "Wto2Q-2Jets_Bin-PTQQ-100": "/Wto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "Wto2Q-2Jets_Bin-PTQQ-200": "/Wto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Wto2Q-2Jets_Bin-PTQQ-400": "/Wto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "Wto2Q-2Jets_Bin-PTQQ-600": "/Wto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-2Jets_Bin-PTQQ-100": "/Zto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-2Jets_Bin-PTQQ-200": "/Zto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-2Jets_Bin-PTQQ-400": "/Zto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-2Jets_Bin-PTQQ-600": "/Zto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            # Vto2Q LO samples
            "Wto2Q-3Jets_Bin-HT-100to400": "/Wto2Q-3Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Wto2Q-3Jets_Bin-HT-400to800": "/Wto2Q-3Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Wto2Q-3Jets_Bin-HT-800to1500": "/Wto2Q-3Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Wto2Q-3Jets_Bin-HT-1500to2500": "/Wto2Q-3Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "Wto2Q-3Jets_Bin-HT-2500": "/Wto2Q-3Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-4Jets_Bin-HT-100to400": "/Zto2Q-4Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-4Jets_Bin-HT-400to800": "/Zto2Q-4Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-4Jets_Bin-HT-800to1500": "/Zto2Q-4Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-4Jets_Bin-HT-1500to2500": "/Zto2Q-4Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-4Jets_Bin-HT-2500": "/Zto2Q-4Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            # WtoLNu LO
            "WtoLNu-4Jets_Bin-1J": "/WtoLNu-4Jets_Bin-1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WtoLNu-4Jets_Bin-2J": "/WtoLNu-4Jets_Bin-2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WtoLNu-4Jets_Bin-3J": "/WtoLNu-4Jets_Bin-3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WtoLNu-4Jets_Bin-4J": "/WtoLNu-4Jets_Bin-4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            # WtoLNu NLO
            "WtoLNu-2Jets_Bin-1J-PTLNu-40to100": "/WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "WtoLNu-2Jets_Bin-1J-PTLNu-100to200": "/WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "WtoLNu-2Jets_Bin-1J-PTLNu-200to400": "/WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WtoLNu-2Jets_Bin-1J-PTLNu-400to600": "/WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WtoLNu-2Jets_Bin-1J-PTLNu-600": "/WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WtoLNu-2Jets_Bin-2J-PTLNu-40to100": "/WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "WtoLNu-2Jets_Bin-2J-PTLNu-100to200": "/WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "WtoLNu-2Jets_Bin-2J-PTLNu-200to400": "/WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WtoLNu-2Jets_Bin-2J-PTLNu-400to600": "/WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "WtoLNu-2Jets_Bin-2J-PTLNu-600": "/WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
    },
    # TODO: add 2025 MC samples once available
}

DATA_SAMPLES_DICT = {
    "2016": {
        "JetHT": {
            "JetHT_Run2016B": [
                "/JetHT/Run2016B-HIPM_UL2016_NanoAODv15_v2-v1/NANOAOD",
            ],
            "JetHT_Run2016C": [
                "/JetHT/Run2016C-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2016D": [
                "/JetHT/Run2016D-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2016E": [
                "/JetHT/Run2016E-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2016F": [
                "/JetHT/Run2016F-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
                # "/JetHT/Run2016F-UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2016G": [
                "/JetHT/Run2016G-UL2016_NanoAODv15-v1/NANOAOD",
            ],
        },
        "MET": {
            "MET_Run2016B": [
                "/JetHT/Run2016G-UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2016C": [
                "/MET/Run2016C-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2016D": [
                "/MET/Run2016D-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2016E": [
                "/MET/Run2016E-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2016F": [
                "/MET/Run2016F-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2016G": [
                "/MET/Run2016G-UL2016_NanoAODv15-v1/NANOAOD",
            ],
        },
        "MuonEG": {
            "MuonEG_Run2016B": [
                "/MuonEG/Run2016B-HIPM_UL2016_NanoAODv15_v2-v1/NANOAOD",
            ],
            "MuonEG_Run2016C": [
                "/MuonEG/Run2016C-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2016D": [
                "/MuonEG/Run2016D-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2016E": [
                "/MuonEG/Run2016E-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2016F": [
                "/MuonEG/Run2016F-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
                # "/MuonEG/Run2016F-UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2016G": [
                "/MuonEG/Run2016G-UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2016H": [
                "/MuonEG/Run2016H-UL2016_NanoAODv15-v1/NANOAOD",
            ],
        },
        "SingleElectron": {
            "SingleElectron_Run2016B": [
                "/SingleElectron/Run2016B-HIPM_UL2016_NanoAODv15_v2-v1/NANOAOD",
            ],
            "SingleElectron_Run2016C": [
                "/SingleElectron/Run2016C-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2016D": [
                "/SingleElectron/Run2016D-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2016E": [
                "/SingleElectron/Run2016E-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2016F": [
                "/SingleElectron/Run2016F-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
                # "/SingleElectron/Run2016F-UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2016G": [
                "/SingleElectron/Run2016G-UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2016H": [
                "/SingleElectron/Run2016H-UL2016_NanoAODv15-v1/NANOAOD",
            ],
        },
        "SingleMuon": {
            "SingleMuon_Run2016B": [
                " /SingleMuon/Run2016B-HIPM_UL2016_NanoAODv15_v2-v1/NANOAOD",
            ],
            "SingleMuon_Run2016C": [
                "/SingleMuon/Run2016C-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2016D": [
                "/SingleMuon/Run2016D-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2016E": [
                "/SingleMuon/Run2016E-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2016F": [
                "/SingleMuon/Run2016F-HIPM_UL2016_NanoAODv15-v1/NANOAOD",
            ],
        },
    },
    "2017": {
        "JetHT": {
            "JetHT_Run2017B": [
                "/JetHT/Run2017B-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2017C": [
                "/JetHT/Run2017C-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2017D": [
                "/JetHT/Run2017D-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2017E": [
                "/JetHT/Run2017E-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2017F": [
                "/JetHT/Run2017F-UL2017_NanoAODv15-v1/NANOAOD",
            ],
        },
        "MET": {
            "MET_Run2017B": [
                "/MET/Run2017B-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2017C": [
                "/MET/Run2017C-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2017D": [
                "/MET/Run2017D-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2017E": [
                "/MET/Run2017E-UL2017_NanoAODv15-v1/NANOAOD",
            ],
        },
        "MuonEG": {
            "MuonEG_Run2017B": [
                "/MuonEG/Run2017B-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2017C": [
                "/MuonEG/Run2017C-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2017D": [
                "/MuonEG/Run2017D-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2017E": [
                "/MuonEG/Run2017E-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "MuonEG_Run2017F": [
                "/MuonEG/Run2017F-UL2017_NanoAODv15-v1/NANOAOD",
            ],
        },
        "SingleElectron": {
            "SingleElectron_Run2017B": [
                "/SingleElectron/Run2017B-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2017C": [
                "/SingleElectron/Run2017C-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2017D": [
                "/SingleElectron/Run2017D-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2017E": [
                "/SingleElectron/Run2017E-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleElectron_Run2017F": [
                "/SingleElectron/Run2017F-UL2017_NanoAODv15-v1/NANOAOD",
            ],
        },
        "SingleMuon": {
            "SingleMuon_Run2017B": [
                "/SingleMuon/Run2017B-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2017C": [
                "/SingleMuon/Run2017C-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2017D": [
                "/SingleMuon/Run2017D-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2017E": [
                "/SingleMuon/Run2017E-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2017F": [
                "/SingleMuon/Run2017F-UL2017_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2017G": [
                "/SingleMuon/Run2017G-UL2017_NanoAODv15-v1/NANOAOD",
            ],
        },
    },
    "2018": {
        "JetHT": {
            "JetHT_Run2018A": [
                "/JetHT/Run2018A-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2018B": [
                "/JetHT/Run2018B-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2018C": [
                "/JetHT/Run2018C-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "JetHT_Run2018D": [
                "/JetHT/Run2018D-UL2018_NanoAODv15-v1/NANOAOD",
            ],
        },
        "MET": {
            "MET_Run2018A": [
                "/MET/Run2018A-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2018B": [
                "/MET/Run2018B-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "MET_Run2018C": [
                "/MET/Run2018C-UL2018_NanoAODv15-v1/NANOAOD",
            ],
        },
        "EGamma": {
            "EGamma_Run2018A": [
                "/EGamma/Run2018A-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "EGamma_Run2018B": [
                "/EGamma/Run2018B-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "EGamma_Run2018C": [
                "/EGamma/Run2018C-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "EGamma_Run2018D": [
                "/EGamma/Run2018D-UL2018_NanoAODv15-v1/NANOAOD",
            ],
        },
        "MuonEG": {
            "Muon_Run2018A": [
                "/MuonEG/Run2018A-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "Muon_Run2018B": [
                "/MuonEG/Run2018B-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "Muon_Run2018C": [
                "/MuonEG/Run2018C-UL2018_NanoAODv15-v1/NANOAOD",
            ],
        },
        "SingleMuon": {
            "SingleMuon_Run2018A": [
                "/SingleMuon/Run2018A-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2018B": [
                "/SingleMuon/Run2018B-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2018C": [
                "/SingleMuon/Run2018C-UL2018_NanoAODv15-v1/NANOAOD",
            ],
            "SingleMuon_Run2018D": [
                "/SingleMuon/Run2018D-UL2018_NanoAODv15-v1/NANOAOD",
            ],
        },
    },
    "2024": {
        "JetMET": {
            "JetMET_Run2024C": [
                "/JetMET0/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                "/JetMET1/Run2024C-MINIv6NANOv15-v1/NANOAOD",
            ],
            "JetMET_Run2024D": [
                "/JetMET0/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                "/JetMET1/Run2024D-MINIv6NANOv15-v1/NANOAOD",
            ],
            "JetMET_Run2024E": [
                "/JetMET0/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                "/JetMET1/Run2024E-MINIv6NANOv15-v1/NANOAOD",
            ],
            "JetMET_Run2024F": [
                "/JetMET0/Run2024F-MINIv6NANOv15-v2/NANOAOD",
                "/JetMET1/Run2024F-MINIv6NANOv15-v2/NANOAOD",
            ],
            "JetMET_Run2024G": [
                "/JetMET0/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                "/JetMET1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
            ],
            "JetMET_Run2024H": [
                "/JetMET0/Run2024H-MINIv6NANOv15-v2/NANOAOD",
                "/JetMET1/Run2024H-MINIv6NANOv15-v2/NANOAOD",
            ],
            "JetMET_Run2024I": [
                "/JetMET0/Run2024I-MINIv6NANOv15-v2/NANOAOD",
                "/JetMET1/Run2024I-MINIv6NANOv15-v1/NANOAOD",
            ],
        },
        "EGamma": {
            "EGamma_Run2024C": [
                "/EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                "/EGamma1/Run2024C-MINIv6NANOv15-v1/NANOAOD",
            ],
            "EGamma_Run2024D": [
                "/EGamma0/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                "/EGamma1/Run2024D-MINIv6NANOv15-v1/NANOAOD",
            ],
            "EGamma_Run2024E": [
                "/EGamma0/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                "/EGamma1/Run2024E-MINIv6NANOv15-v1/NANOAOD",
            ],
            "EGamma_Run2024F": [
                "/EGamma0/Run2024F-MINIv6NANOv15-v1/NANOAOD",
                "/EGamma1/Run2024F-MINIv6NANOv15-v1/NANOAOD",
            ],
            "EGamma_Run2024G": [
                "/EGamma0/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                "/EGamma1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
            ],
            "EGamma_Run2024H": [
                "/EGamma0/Run2024H-MINIv6NANOv15-v2/NANOAOD",
                "/EGamma1/Run2024H-MINIv6NANOv15-v1/NANOAOD",
            ],
            "EGamma_Run2024I": [
                "/EGamma0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
                "/EGamma1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
            ],
        },
        "Muon": {
            "Muon_Run2024C": [
                "/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                "/Muon1/Run2024C-MINIv6NANOv15-v1/NANOAOD",
            ],
            "Muon_Run2024D": [
                "/Muon0/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                "/Muon1/Run2024D-MINIv6NANOv15-v1/NANOAOD",
            ],
            "Muon_Run2024E": [
                "/Muon0/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                "/Muon1/Run2024E-MINIv6NANOv15-v1/NANOAOD",
            ],
            "Muon_Run2024F": [
                "/Muon0/Run2024F-MINIv6NANOv15-v1/NANOAOD",
                "/Muon1/Run2024F-MINIv6NANOv15-v1/NANOAOD",
            ],
            "Muon_Run2024G": [
                "/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD",
                "/Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
            ],
            "Muon_Run2024H": [
                "/Muon0/Run2024H-MINIv6NANOv15-v1/NANOAOD",
                "/Muon1/Run2024H-MINIv6NANOv15-v2/NANOAOD",
            ],
            "Muon_Run2024I": [
                "/Muon0/Run2024I-MINIv6NANOv15-v1/NANOAOD",
                "/Muon1/Run2024I-MINIv6NANOv15-v1/NANOAOD",
            ],
        },
    },
    "2025": {
        "JetMET": {
            "JetMET_Run2025C": [
                "/JetMET0/Run2025C-PromptReco-v2/NANOAOD",
                "/JetMET1/Run2025C-PromptReco-v2/NANOAOD",
            ],
            "JetMET_Run2025D": [
                "/JetMET0/Run2025D-PromptReco-v1/NANOAOD",
                "/JetMET1/Run2025D-PromptReco-v1/NANOAOD",
            ],
            "JetMET_Run2025E": [
                "/JetMET0/Run2025E-PromptReco-v1/NANOAOD",
                "/JetMET1/Run2025E-PromptReco-v1/NANOAOD",
            ],
            "JetMET_Run2025F": [
                "/JetMET0/Run2025F-PromptReco-v2/NANOAOD",
                "/JetMET1/Run2025F-PromptReco-v2/NANOAOD",
            ],
            "JetMET_Run2025G": [
                "/JetMET0/Run2025G-PromptReco-v1/NANOAOD",
                "/JetMET1/Run2025G-PromptReco-v1/NANOAOD",
            ],
        },
        "EGamma": {
            "EGamma_Run2025C": [
                "/EGamma0/Run2025C-PromptReco-v2/NANOAOD",
                "/EGamma1/Run2025C-PromptReco-v2/NANOAOD",
                "/EGamma2/Run2025C-PromptReco-v2/NANOAOD",
                "/EGamma3/Run2025C-PromptReco-v2/NANOAOD",
            ],
            "EGamma_Run2025D": [
                "/EGamma0/Run2025D-PromptReco-v1/NANOAOD",
                "/EGamma1/Run2025D-PromptReco-v1/NANOAOD",
                "/EGamma2/Run2025D-PromptReco-v1/NANOAOD",
                "/EGamma3/Run2025D-PromptReco-v1/NANOAOD",
            ],
            "EGamma_Run2025E": [
                "/EGamma0/Run2025E-PromptReco-v1/NANOAOD",
                "/EGamma1/Run2025E-PromptReco-v1/NANOAOD",
                "/EGamma2/Run2025E-PromptReco-v1/NANOAOD",
                "/EGamma3/Run2025E-PromptReco-v1/NANOAOD",
            ],
            "EGamma_Run2025F": [
                "/EGamma0/Run2025F-PromptReco-v2/NANOAOD",
                "/EGamma1/Run2025F-PromptReco-v2/NANOAOD",
                "/EGamma2/Run2025F-PromptReco-v2/NANOAOD",
                "/EGamma3/Run2025F-PromptReco-v2/NANOAOD",
            ],
            "EGamma_Run2025G": [
                "/EGamma0/Run2025G-PromptReco-v1/NANOAOD",
                "/EGamma1/Run2025G-PromptReco-v1/NANOAOD",
                "/EGamma2/Run2025G-PromptReco-v1/NANOAOD",
                "/EGamma3/Run2025G-PromptReco-v1/NANOAOD",
            ],
        },
        "Muon": {
            "Muon_Run2025C": [
                "/Muon0/Run2025C-PromptReco-v2/NANOAOD",
                "/Muon1/Run2025C-PromptReco-v2/NANOAOD",
            ],
            "Muon_Run2025D": [
                "/Muon0/Run2025D-PromptReco-v1/NANOAOD",
                "/Muon1/Run2025D-PromptReco-v1/NANOAOD",
            ],
            "Muon_Run2025E": [
                "/Muon0/Run2025E-PromptReco-v1/NANOAOD",
                "/Muon1/Run2025E-PromptReco-v1/NANOAOD",
            ],
            "Muon_Run2025F": [
                "/Muon0/Run2025F-PromptReco-v2/NANOAOD",
                "/Muon1/Run2025F-PromptReco-v2/NANOAOD",
            ],
            "Muon_Run2025G": [
                "/Muon0/Run2025G-PromptReco-v1/NANOAOD",
                "/Muon1/Run2025G-PromptReco-v1/NANOAOD",
            ],
        },
    },
}


def get_sample_list(dataset: str, base_url: str) -> list[str]:
    command = f'dasgoclient --query="file dataset={dataset}"'
    print(f"Running command: {command}")
    result = subprocess.run(command, check=False, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
    # get the output
    output = result.stdout.strip()
    if output == "":
        warnings.warn(
            f"No files found for dataset {dataset}." f" Command executed: {command}", stacklevel=2
        )
    # split the output into lines
    samples = output.split("\n")
    samples = [f"{base_url}/{sample}" for sample in samples]
    # Sort files for consistent output
    sorted_files = sorted(samples)
    return sorted_files


def main():
    parser = argparse.ArgumentParser(description="Create filelists from sample config")

    parser.add_argument(
        "-b",
        "--base-config",
        type=str,
        # default=str(SCRIPT_DIR / "nanoindex_v12v2_private.json"),
        default=str(SCRIPT_DIR / "nanoindex_v14_25v2.json"),
        help="Path to the base config file",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        choices=["2016", "2017", "2018", "2024", "2025"],
        default=["2016", "2017", "2018", "2024", "2025"],
        help="Years to process",
    )
    parser.add_argument(
        "-o",
        "--output-config",
        type=str,
        default=str(SCRIPT_DIR / "nanoindex_v15.json"),
        help="Path to the output config file",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        # default="root://cms-xrd-global.cern.ch",
        default="root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/",
        choices=[
            "root://cms-xrd-global.cern.ch",
            "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/",
        ],
        help="Base URL for files",
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")

    base_url = args.base_url
    if args.base_config:
        print(f"Loading base config from {args.base_config}")
        with Path(args.base_config).open("r") as f:
            index_dict = json.load(f)
        if "v14" in args.base_config:
            print("Updating index_dict from v14 to v15 index format")
            # update keys
            # {VJetsNLO, VJetsLO} -> VJets
            for year in index_dict:
                vjets_nlo = index_dict[year].pop("VJetsNLO", {})
                vjets_lo = index_dict[year].pop("VJetsLO", {})
                dyjets_lo = index_dict[year].pop("DYJetsLO", {})
                dyjets_nlo = index_dict[year].pop("DYJetsNLO", {})
                vjets = {**vjets_nlo, **vjets_lo, **dyjets_lo, **dyjets_nlo}
                index_dict[year]["VJets"] = vjets

                # HH4b -> {HH, VBFHH}
                hh4b = index_dict[year].pop("HH4b", {})
                indexed = set()
                if "VBFHH" not in index_dict[year]:
                    vbfhh = {}
                    for sample_key in hh4b:
                        if sample_key.startswith("VBFHH"):
                            vbfhh[sample_key] = hh4b[sample_key]
                            indexed.add(sample_key)
                    index_dict[year]["VBFHH"] = vbfhh
                if "HH" not in index_dict[year]:
                    hh = {}
                    for sample_key in hh4b:
                        if sample_key.startswith("GluGlutoHH"):
                            hh[sample_key] = hh4b[sample_key]
                            indexed.add(sample_key)
                    index_dict[year]["HH"] = hh
                # not indexed
                for sample_key in hh4b:
                    if sample_key not in indexed:
                        warnings.warn(
                            f"Sample {sample_key} from HH4b not indexed in HH or VBFHH",
                            stacklevel=2,
                        )
            print("Updated index_dict to v15 index format")
    else:
        print("Creating new index_dict")
        index_dict = {}

    for year in args.years:
        index_dict[year] = {}

        # Data
        for sample_category in DATA_SAMPLES_DICT[year]:
            index_dict[year][sample_category] = {}
            for sample_key in DATA_SAMPLES_DICT[year][sample_category]:
                files = []
                for dataset in DATA_SAMPLES_DICT[year][sample_category][sample_key]:
                    samples = get_sample_list(dataset, base_url)
                    files.extend(samples)
                # Sort files for consistent output
                sorted_files = sorted(files)
                index_dict[year][sample_category][sample_key] = sorted_files

        # MC
        year_mc = year
        if year == "2025":
            # TODO: define 2025 MC samples when available
            warnings.warn(
                "2025 MC samples are not defined, skipping MC samples for 2025", stacklevel=2
            )
            continue
        for sample_category in MC_SAMPLES_DICT[year_mc]:
            index_dict[year][sample_category] = {}
            for sample_key in MC_SAMPLES_DICT[year_mc][sample_category]:
                dataset = MC_SAMPLES_DICT[year_mc][sample_category][sample_key]
                samples = get_sample_list(dataset, base_url)
                index_dict[year][sample_category][sample_key] = samples

    path_output = Path(args.output_config)
    with path_output.open("w") as f:
        json.dump(index_dict, f, indent=4)
    print(f"Wrote filelist to {path_output}")


if __name__ == "__main__":
    main()
