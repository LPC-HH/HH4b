from __future__ import annotations

import json
import subprocess
from pathlib import Path

MC_MINIAOD_DICT = {
    # Run 2
    # Diboson
    "WW_13TeV": "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WZ_13TeV": "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "ZZ_13TeV": "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    # Hbb
    "GluGluHto2B_M-125_13TeV": "/GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17_ext1-v4/MINIAODSIM",
    "GluGluHto2B_PT-200_M-125_13TeV": "/GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "VBFHto2B_M-125_dipoleRecoilOn_13TeV": "/VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "WminusH_Hto2B_Wto2Q_M-125_13TeV": "/WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "WminusH_Hto2B_WtoLNu_M-125_13TeV": "/WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WplusH_Hto2B_Wto2Q_M-125_13TeV": "/WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "WplusH_Hto2B_WtoLNu_M-125_13TeV": "/WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "ZH_Hto2B_Zto2L_M-125_13TeV": "/ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "ZH_Hto2B_Zto2Nu_M-125_13TeV": "/ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "ZH_Hto2C_Zto2Q_M-125_13TeV": "/ZH_HToCC_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "ggZH_Hto2B_Zto2L_M-125_13TeV": "/ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "ggZH_Hto2B_Zto2Nu_M-125_13TeV": "/ggZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "ggZH_Hto2B_Zto2Q_M-125_13TeV": "/ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "ggZH_Hto2C_Zto2Q_M-125_13TeV": "/ggZH_HToCC_ZToQQ_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "ttHto2B_M-125_13TeV": "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    # QCD
    "QCD_HT-100to200_13TeV": "/QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "QCD_HT-200to300_13TeV": "/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "QCD_HT-300to500_13TeV": "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "QCD_HT-500to700_13TeV": "/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "QCD_HT-700to1000_13TeV": "/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "QCD_HT-1000to1500_13TeV": "/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "QCD_HT-1500to2000_13TeV": "/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "QCD_HT-2000_13TeV": "/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    # TODO: Add SingleTop samples
    # TT
    "TTto2L2Nu_13TeV": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "TTto4Q_13TeV": "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "TTtoLNu2Q_13TeV": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    # TODO: Add VBFHH samples
    # VJets
    "DYJetsToLL_M-50_13TeV_NLO": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "DYJetsToLL_M-50_13TeV_LO": "/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    # Vto2Q LO samples
    "WJetsToQQ_HT-200to400_13TeV": "/WJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "WJetsToQQ_HT-400to600_13TeV": "/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "WJetsToQQ_HT-600to800_13TeV": "/WJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "WJetsToQQ_HT-800_13TeV": "/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "ZJetsToQQ_HT-200to400_13TeV": "/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "ZJetsToQQ_HT-400to600_13TeV": "/ZJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "ZJetsToQQ_HT-600to800_13TeV": "/ZJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    "ZJetsToQQ_HT-800_13TeV": "/ZJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    # WtoLNu LO
    "WJetsToLNu_HT-70To100_13TeV": "/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_HT-100To200_13TeV": "/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_HT-200To400_13TeV": "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_HT-400To600_13TeV": "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_HT-600To800_13TeV": "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_HT-800To1200_13TeV": "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_HT-1200To2500_13TeV": "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_HT-2500_13TeV": "/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM",
    # WtoLNu NLO
    "WJetsToLNu_0J_13TeV": "/WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_1J_13TeV": "/WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    "WJetsToLNu_2J_13TeV": "/WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM",
    # Run 3
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200": " /DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    # NLO samples
    "Wto2Q-2Jets_Bin-PTQQ-100": "/Wto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "Wto2Q-2Jets_Bin-PTQQ-200": "/Wto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Wto2Q-2Jets_Bin-PTQQ-400": "/Wto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "Wto2Q-2Jets_Bin-PTQQ-600": "/Wto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-2Jets_Bin-PTQQ-100": "/Zto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-2Jets_Bin-PTQQ-200": "/Zto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-2Jets_Bin-PTQQ-400": "/Zto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-2Jets_Bin-PTQQ-600": "/Zto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    # LO samples
    "Wto2Q-3Jets_Bin-HT-100to400": "/Wto2Q-3Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Wto2Q-3Jets_Bin-HT-400to800": "/Wto2Q-3Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Wto2Q-3Jets_Bin-HT-800to1500": "/Wto2Q-3Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Wto2Q-3Jets_Bin-HT-1500to2500": "/Wto2Q-3Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "Wto2Q-3Jets_Bin-HT-2500": "/Wto2Q-3Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-100to400": "/Zto2Q-4Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-400to800": "/Zto2Q-4Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-800to1500": "/Zto2Q-4Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-1500to2500": "/Zto2Q-4Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-2500": "/Zto2Q-4Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
}

filepath_dict = {}
xsec_dict = {}

miniaod_json = Path("miniaod_filepaths.json")
if miniaod_json.exists():
    with miniaod_json.open() as f:
        filepath_dict = json.load(f)
else:
    for sample_key, dataset in MC_MINIAOD_DICT.items():
        command = f'dasgoclient --query="file dataset={dataset}"'
        # print(f"Running command: {command}")
        result = subprocess.run(command, check=False, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing command: {command}")
        # get the output
        output = result.stdout.strip()
        # split the output into lines
        samples = output.split("\n")
        example_sample = samples[0]
        print(f"Example sample for {sample_key}: {example_sample}")
        filepath_dict[sample_key] = example_sample

    # save to json
    with miniaod_json.open("w") as f:
        json.dump(filepath_dict, f, indent=4)

for sample_key in filepath_dict:
    filepath = filepath_dict[sample_key]
    command = f'cmsRun $CMSSW_BASE/src/ana.py inputFiles="{filepath}" maxEvents=-1'
    print(f"Running command to get xsec: {command}")
    result = subprocess.run(command, check=False, shell=True, capture_output=True, text=True)
    # output = result.stdout.strip()
    # output_lines = output.split("\n")
    # print(output)
    output = result.stderr.strip()
    output_lines = output.split("\n")
    # print(output)
    for line in output_lines:
        if "After filter: final cross section = " in line:
            xsec_str = line.split("After filter: final cross section = ")[1].split(" pb")[0]
            xsec_info = xsec_str.split(" +- ")
            xsec = float(xsec_info[0])
            xsec_err = float(xsec_info[1])
            xsec_dict[sample_key] = (xsec, xsec_err)
            print(f"Extracted cross-section for {sample_key}: ({xsec} +- {xsec_err}) pb")
            break
        else:
            continue

# save to json
with Path("miniaod_xsecs.json").open("w") as f:
    json.dump(xsec_dict, f, indent=4)
