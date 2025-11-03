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
            # "QCD_HT-40to70": "/QCD-4Jets_Bin-HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            # "QCD_HT-70to100": /QCD-4Jets_Bin-HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM,
            # "QCD_HT-100to200": "/QCD-4Jets_Bin-HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
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
            "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
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
            "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-2p12-C2V-3p87-C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-1-C2V-0-C3-1_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_Par-CV-1-C2V-1-C3-1_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
        },
        "VJets": {
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
            # NLO samples
            "Wto2Q-2Jets_Bin-PTQQ-100": "/Wto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "Wto2Q-2Jets_Bin-PTQQ-200": "/Wto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Wto2Q-2Jets_Bin-PTQQ-400": "/Wto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            "Wto2Q-2Jets_Bin-PTQQ-600": "/Wto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-2Jets_Bin-PTQQ-100": "/Zto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-2Jets_Bin-PTQQ-200": "/Zto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-2Jets_Bin-PTQQ-400": "/Zto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            "Zto2Q-2Jets_Bin-PTQQ-600": "/Zto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            # LO samples
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
        },
    },
    # TODO: add 2025 MC samples once available
}

DATA_SAMPLES_DICT = {
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
                "/EGamma0/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                "/EGamma1/Run2024G-MINIv6NANOv15-v1/NANOAOD",
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
                "/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD",
                "/Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
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
                "/JetMET0/Run2025E-PromptReco-v2/NANOAOD",
                "/JetMET1/Run2025E-PromptReco-v2/NANOAOD",
            ],
            "JetMET_Run2025G": [
                "/JetMET0/Run2025E-PromptReco-v1/NANOAOD",
                "/JetMET1/Run2025E-PromptReco-v1/NANOAOD",
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
                "/Muon0/Run2025E-PromptReco-v2/NANOAOD",
                "/Muon1/Run2025E-PromptReco-v2/NANOAOD",
            ],
            "Muon_Run2025G": [
                "/Muon0/Run2025G-PromptReco-v1/NANOAOD",
                "/Muon1/Run2025G-PromptReco-v1/NANOAOD",
            ],
        },
    },
}


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
        choices=["2024", "2025"],
        default=["2024", "2025"],
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
                    command = f'dasgoclient --query="file dataset={dataset}"'
                    print(f"Running command: {command}")
                    result = subprocess.run(
                        command, check=False, shell=True, capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        print(f"Error executing command: {command}")
                    # get the output
                    output = result.stdout.strip()
                    # split the output into lines
                    samples = output.split("\n")
                    samples = [f"{base_url}/{sample}" for sample in samples]
                    files.extend(samples)
                # Sort files for consistent output
                sorted_files = sorted(files)
                index_dict[year][sample_category][sample_key] = sorted_files

        # MC
        year_mc = year
        if year == "2025":
            # TODO: define 2025 MC samples when available
            warnings.warn(
                "2025 MC samples are not defined, skipping MC samples for 2025", stacklevel=1
            )
            continue
        for sample_category in MC_SAMPLES_DICT[year_mc]:
            index_dict[year][sample_category] = {}
            for sample_key in MC_SAMPLES_DICT[year_mc][sample_category]:
                dataset = MC_SAMPLES_DICT[year_mc][sample_category][sample_key]
                command = f'dasgoclient --query="file dataset={dataset}"'
                print(f"Running command: {command}")
                result = subprocess.run(
                    command, check=False, shell=True, capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"Error executing command: {command}")
                # get the output
                output = result.stdout.strip()
                # split the output into lines
                samples = output.split("\n")
                samples = [f"{base_url}/{sample}" for sample in samples]
                # Sort files for consistent output
                sorted_files = sorted(samples)
                index_dict[year][sample_category][sample_key] = sorted_files

    path_output = Path(args.output_config)
    with path_output.open("w") as f:
        json.dump(index_dict, f, indent=4)
    print(f"Wrote filelist to {path_output}")


if __name__ == "__main__":
    main()
