from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/x86_64/rhel7/py3/current"

qcd_bins = [
    # "0to80",
    "15to30",  # unclear if these are needed
    "30to50",
    "50to80",
    "80to120",
    "120to170",
    "170to300",
    "300to470",
    "470to600",
    "600to800",
    "800to1000",
    "1000to1400",
    "1400to1800",
    "1800to2400",
    "2400to3200",
    "3200",
]

qcd_ht_bins = [
    "40to70",
    "70to100",
    "40to100",
    "100to200",
    "200to400",
    "400to600",
    "600to800",
    "800to1000",
    "1000to1200",
    "1200to1500",
    "1500to2000",
    "2000",
]

qcd_ht_bins_run2 = [
    "200to300",
    "300to500",
    "500to700",
    "700to1000",
    "1000to1500",
    "1500to2000",
    "2000toInf",
]


def get_v9():
    return {
        "2018": {
            "HH": {
                "GluGlutoHHto4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "GluGlutoHHto4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "GluGlutoHHto4B_cHHH2p45_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/GluGluToHHTo4B_cHHH2p45_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "GluGlutoHHto4B_cHHH5_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/GluGluToHHTo4B_cHHH5_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
            },
            "Hbb": {
                "GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8": "/GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8": "/VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8": "/WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8": "/WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8": "/WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8": "/WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8": "/ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8": "/ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8": "/ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8": "/ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8": "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
            },
            "Diboson": {
                "ZZTo4B01j_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8": "/ZZTo4B01j_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
            },
            "QCD": {
                **{
                    f"QCD_HT-{qbin}-13TeV": f"/QCD_HT{qbin}_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"
                    for qbin in qcd_ht_bins_run2
                },
            },
            "TTbar": {
                "TTToHadronic_13TeV": "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "TTTo2L2Nu_13TeV": "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "TTToSemiLeptonic_13TeV": "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
            },
            "VJets": {
                "WJetsToQQ_HT-200to400_13TeV": "/WJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "WJetsToQQ_HT-400to600_13TeV": "/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "WJetsToQQ_HT-600to800_13TeV": "/WJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "WJetsToQQ_HT-800toInf_13TeV": "/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "ZJetsToQQ_HT-200to400_13TeV": "/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "ZJetsToQQ_HT-400to600_13TeV": "/ZJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "ZJetsToQQ_HT-600to800_13TeV": "/ZJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
                "ZJetsToQQ_HT-800toInf_13TeV": "/ZJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
            },
        }
    }


path_hqu = "/store/group/lpcjme/NanoAOD/NanoAODv9-ParticleNetAK4/2018"


def get_v9_private():
    return {
        "2018": {
            "HH": {
                "GluGlutoHHto4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8": f"{path_hqu}/mc/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8",
                "GluGlutoHHto4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8": f"{path_hqu}/mc/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8",
                "GluGlutoHHto4B_cHHH5_TuneCP5_PSWeights_13TeV-powheg-pythia8": f"{path_hqu}/mc/GluGluToHHTo4B_cHHH5_TuneCP5_PSWeights_13TeV-powheg-pythia8",
            },
            "Hbb": {
                "GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8": f"{path_hqu}/mc/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8",
                "VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8": f"{path_hqu}/mc/VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8",
                "WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8": f"{path_hqu}/mc/WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8",
                "WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8": f"{path_hqu}/mc/WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8",
                "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8": f"{path_hqu}/mc/ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8",
                "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8": f"{path_hqu}/mc/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8",
            },
            "Diboson": {
                "ZZTo4B01j_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8": f"{path_hqu}/mc/ZZTo4B01j_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            },
            "QCD": {
                **{
                    f"QCD_HT-{qbin}-13TeV": f"{path_hqu}/mc/QCD_HT{qbin}_TuneCP5_13TeV-madgraphMLM-pythia8"
                    for qbin in qcd_ht_bins_run2
                },
            },
            "TTbar": {
                "TTToHadronic_13TeV": f"{path_hqu}/mc/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",
                "TTTo2L2Nu_13TeV": f"{path_hqu}/mc/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
                "TTToSemiLeptonic_13TeV": f"{path_hqu}/mc/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
            },
            "VJets": {
                "WJetsToQQ_HT-200to400_13TeV": f"{path_hqu}/mc/WJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToQQ_HT-400to600_13TeV": f"{path_hqu}/mc/WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToQQ_HT-600to800_13TeV": f"{path_hqu}/mc/WJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToQQ_HT-800toInf_13TeV": f"{path_hqu}/mc/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                "ZJetsToQQ_HT-200to400_13TeV": f"{path_hqu}/mc/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8",
                "ZJetsToQQ_HT-400to600_13TeV": f"{path_hqu}/mc/ZJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8",
                "ZJetsToQQ_HT-600to800_13TeV": f"{path_hqu}/mc/ZJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8",
                "ZJetsToQQ_HT-800toInf_13TeV": f"{path_hqu}/mc/ZJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
            },
        }
    }


def get_v10():
    return {
        "2022": {
            "JetMET": {
                "Run2022C_single": "/JetHT/Run2022C-PromptNanoAODv10_v1-v1/NANOAOD",
                "Run2022C": "/JetMET/Run2022C-PromptNanoAODv10_v1-v1/NANOAOD",
                "Run2022D_v1": "/JetMET/Run2022D-PromptNanoAODv10_v1-v1/NANOAOD",
                "Run2022D_v2": "/JetMET/Run2022D-PromptNanoAODv10_v2-v1/NANOAOD",
            },
            "Muon": {
                "Run2022C_single": "/SingleMuon/Run2022C-PromptNanoAODv10_v1-v1/NANOAOD",
                "Run2022C": "/Muon/Run2022C-PromptNanoAODv10_v1-v1/NANOAOD",
                "Run2022D_v1": "/Muon/Run2022D-PromptNanoAODv10_v1-v1/NANOAOD",
                "Run2022D_v2": "/Muon/Run2022D-PromptNanoAODv10_v2-v1/NANOAOD",
            },
        },
        "2022EE": {
            "JetMET": {
                "Run2022E": "/JetMET/Run2022E-PromptNanoAODv10_v1-v3/NANOAOD",
                "Run2022F": "/JetMET/Run2022F-PromptNanoAODv10_v1-v2/NANOAOD",
                "Run2022G": "/JetMET/Run2022G-PromptNanoAODv10_v1-v1/NANOAOD",
            },
            "Muon": {
                "Run2022E": "/Muon/Run2022E-PromptNanoAODv10_v1-v3/NANOAOD",
                "Run2022F": "/Muon/Run2022F-PromptNanoAODv10_v1-v2/NANOAOD",
                "Run2022G": "/Muon/Run2022G-PromptNanoAODv10_v1-v1/NANOAOD",
            },
        },
    }


def get_v11():
    """
    DEPRECATED
    """
    return {
        # MC campaign pre EE+ leak
        "2022": {
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "VBFHto2B_M-125_dipoleRecoilOn": "/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "WminusH_Hto2B_Wto2Q_M-125": "/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "WminusH_Hto2B_WtoLNu_M-125": "/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "WplusH_Hto2B_Wto2Q_M-125": "/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "WplusH_Hto2B_WtoLNu_M-125": "/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "ZH_Hto2B_Zto2L_M-125": "/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "ZH_Hto2B_Zto2Q_M-125": "/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "ggZH_Hto2B_Zto2L_M-125": "/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "ggZH_Hto2B_Zto2Nu_M-125": "/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "ggZH_Hto2B_Zto2Q_M-125": "/ggZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "ttHto2B_M-125": "/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
            },
            "HHH6b": {
                # nominal?
                "HHHTo6B_c3_0_d4_0": "/HHHTo6B_c3_0_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_0_d4_99": "/HHHTo6B_c3_0_d4_99_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_0_d4_minus1": "/HHHTo6B_c3_0_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_19_d4_19": "/HHHTo6B_c3_19_d4_19_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_1_d4_0": "/HHHTo6B_c3_1_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_1_d4_2": "/HHHTo6B_c3_1_d4_2_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_2_d4_minus1": "/HHHTo6B_c3_2_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_4_d4_9": "/HHHTo6B_c3_4_d4_9_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_minus1_d4_0": "/HHHTo6B_c3_minus1_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_minus1_d4_minus1": "/HHHTo6B_c3_minus1_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "HHHTo6B_c3_minus1p5_d4_minus0p5": "/HHHTo6B_c3_minus1p5_d4_minus0p5_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
            },
            "QCD": {
                **{
                    f"QCD_PT-{qbin}": [
                        f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2_ext1-v1/NANOAODSIM",
                    ]
                    for qbin in qcd_bins
                },
                **{
                    f"QCDB_HT-{qbin}": f"/QCDB-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
                **{
                    f"QCD_HT-{qbin}": f"/QCD-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
                "QCD_PT-15to7000": "/QCD_PT-15to7000_TuneCP5_Flat2018_13p6TeV_pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
            },
            "TT": {
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
            },
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                "WWto4Q": "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "WZto4Q-1Jets-4FS": "/WZto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                # missing ZZto4B
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
            },
            "HH": {
                # 1M
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM",
                # 100k
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
            },
            "JetMET": {
                "Run2022C_single": "/JetHT/Run2022C-22Sep2023-v1/NANOAOD",
                "Run2022C": "/JetMET/Run2022C-22Sep2023-v1/NANOAOD",
                "Run2022D": "/JetMET/Run2022D-22Sep2023-v1/NANOAOD",
            },
            "Muon": {
                "Run2022C_single": "/SingleMuon/Run2022C-22Sep2023-v1/NANOAOD",
                "Run2022C": "/Muon/Run2022C-22Sep2023-v1/NANOAOD",
                "Run2022D": "/Muon/Run2022D-22Sep2023-v1/NANOAOD",
            },
        },
        # MC campaign post EE+ leak
        "2022EE": {
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "VBFHto2B_M-125_dipoleRecoilOn": "/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "WminusH_Hto2B_Wto2Q_M-125": "/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "WminusH_Hto2B_WtoLNu_M-125": "/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "WplusH_Hto2B_Wto2Q_M-125": "/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "WplusH_Hto2B_WtoLNu_M-125": "/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "ZH_Hto2B_Zto2L_M-125": "/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "ZH_Hto2B_Zto2Q_M-125": "/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "ggZH_Hto2B_Zto2L_M-125": "/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "ggZH_Hto2B_Zto2Nu_M-125": "/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "ggZH_Hto2B_Zto2Q_M-125": "/ggZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "ttHto2B_M-125": "/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            },
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
            },
            "HHH6b": {
                "HHHTo6B_c3_0_d4_0": "/HHHTo6B_c3_0_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_0_d4_99": "/HHHTo6B_c3_0_d4_99_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_0_d4_minus1": "/HHHTo6B_c3_0_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_19_d4_19": "/HHHTo6B_c3_19_d4_19_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_1_d4_0": "/HHHTo6B_c3_1_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_1_d4_2": "/HHHTo6B_c3_1_d4_2_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_2_d4_minus1": "/HHHTo6B_c3_2_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_4_d4_9": "/HHHTo6B_c3_4_d4_9_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_minus1_d4_0": "/HHHTo6B_c3_minus1_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_minus1_d4_minus1": "/HHHTo6B_c3_minus1_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "HHHTo6B_c3_minus1p5_d4_minus0p5": "/HHHTo6B_c3_minus1p5_d4_minus0p5_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            },
            "QCD": {
                **{
                    f"QCD_PT-{qbin}": [
                        f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1_ext1-v1/NANOAODSIM",
                    ]
                    for qbin in qcd_bins
                },
                **{
                    f"QCDB_HT-{qbin}": f"/QCDB-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
                **{
                    f"QCD_HT-{qbin}": f"/QCD-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
                "QCD_PT-15to7000": "/QCD_PT-15to7000_TuneCP5_Flat2018_13p6TeV_pythia8/Run3Summer22EENanoAODv11-EpsilonPU_126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
            },
            "TT": {
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            },
            # "Single-Top": {
            # "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8" # submitted
            # "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8" # submitted
            # }
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "WWto4Q": "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "WZto4Q-1Jets-4FS": "/WZto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                # missing ZZto4B
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v3/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v3/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            },
            # Prompt-Reco
            "JetMET": {
                # "Run2022E": "/JetMET/Run2022E-22Sep2023-v1/NANOAOD", # in production
                # "Run2022F": "/JetMET/Run2022F-22Sep2023-v1/NANOAOD", # in production
                # "Run2022G": "/JetMET/Run2022G-22Sep2023-v1/NANOAOD", # in production
                "Run2022F": "/JetMET/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
                "Run2022G": "/JetMET/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
            },
            "Muon": {
                # "Run2022E": "/Muon/Run2022E-22Sep2023-v1/NANOAOD",
                # "Run2022F": "/Muon/Run2022F-22Sep2023-v1/NANOAOD", # in production
                # "Run2022G": "/Muon/Run2022G-22Sep2023-v1/NANOAOD",
                "Run2022F": "/Muon/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
                "Run2022G": "/Muon/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
            },
        },
    }


def get_v12():
    return {
        "2022": {
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "VBFHto2B_M-125_dipoleRecoilOn": "/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "WminusH_Hto2B_Wto2Q_M-125": "/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "WminusH_Hto2B_WtoLNu_M-125": "/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "WplusH_Hto2B_Wto2Q_M-125": "/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "WplusH_Hto2B_WtoLNu_M-125": "/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "ZH_Hto2B_Zto2L_M-125": "/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "ZH_Hto2B_Zto2Q_M-125": "/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2L_M-125": "/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2Nu_M-125": "/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2Q_M-125": "/ggZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "ttHto2B_M-125": "/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
            },
            "HHH6b": {
                "HHHTo6B_c3_0_d4_0": "/HHHTo6B_c3_0_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_0_d4_99": "/HHHTo6B_c3_0_d4_99_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_0_d4_minus1": "/HHHTo6B_c3_0_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_19_d4_19": "/HHHTo6B_c3_19_d4_19_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_1_d4_0": "/HHHTo6B_c3_1_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_1_d4_2": "/HHHTo6B_c3_1_d4_2_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_2_d4_minus1": "/HHHTo6B_c3_2_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_4_d4_9": "/HHHTo6B_c3_4_d4_9_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_minus1_d4_0": "/HHHTo6B_c3_minus1_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_minus1_d4_minus1": "/HHHTo6B_c3_minus1_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "HHHTo6B_c3_minus1p5_d4_minus0p5": "/HHHTo6B_c3_minus1p5_d4_minus0p5_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
            },
            "HH": {
                # 100k events
                "VBFHHto4B_CV-1_C2V-1_C3-1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV-1_C2V-1_C3-1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "VBFHHto4B_CV-1_C2V-1_C3-2_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV-1_C2V-1_C3-2_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "VBFHHto4B_CV-1_C2V-2_C3-1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV-1_C2V-2_C3-1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                # 1M events
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
            },
            "QCD": {
                **{
                    f"QCD_PT-{qbin}": [
                        f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                        f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM",
                    ]
                    for qbin in qcd_bins
                },
                **{
                    f"QCD_HT-{qbin}": f"/QCD-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
            },
            "TT": {
                "TTto4Q": [
                    "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                    "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM",
                ],
                "TTto2L2Nu": [
                    "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                    "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM",
                ],
                "TTtoLNu2Q": [
                    "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                    "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM",
                ],
            },
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "WWto4Q": [
                    "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                    "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM",
                ],
                "WWto4Q_1Jets-4FS": "/WWto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "WZto4Q-1Jets-4FS": "/WZto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                # missing ZZto4
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
            },
            "JetMET": {
                "Run2022C_single": "/JetHT/Run2022C-22Sep2023-v1/NANOAOD",
                "Run2022C": "/JetMET/Run2022C-22Sep2023-v1/NANOAOD",
                "Run2022D": "/JetMET/Run2022D-22Sep2023-v1/NANOAOD",
            },
            "Muon": {
                "Run2022C_single": "/SingleMuon/Run2022C-22Sep2023-v1/NANOAOD",
                "Run2022C": "/Muon/Run2022C-22Sep2023-v1/NANOAOD",
                "Run2022D": "/Muon/Run2022D-22Sep2023-v1/NANOAOD",
            },
        },
        "2022EE": {
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "VBFHto2B_M-125_dipoleRecoilOn": "/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "WminusH_Hto2B_Wto2Q_M-125": "/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "WminusH_Hto2B_WtoLNu_M-125": [
                    "/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                    "/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                ],
                "WplusH_Hto2B_Wto2Q_M-125": "/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "WplusH_Hto2B_WtoLNu_M-125": [
                    "/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                    "/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                ],
                "ZH_Hto2B_Zto2L_M-125": [
                    "/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                    "/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                ],
                "ZH_Hto2B_Zto2Q_M-125": "/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2L_M-125": "/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2Nu_M-125": "/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2Q_M-125": "/ggZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "ttHto2B_M-125": "/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
            },
            "HHH6b": {
                "HHHTo6B_c3_0_d4_0": "/HHHTo6B_c3_0_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_0_d4_99": "/HHHTo6B_c3_0_d4_99_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_0_d4_minus1": "/HHHTo6B_c3_0_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_19_d4_19": "/HHHTo6B_c3_19_d4_19_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_1_d4_0": "/HHHTo6B_c3_1_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_1_d4_2": "/HHHTo6B_c3_1_d4_2_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_2_d4_minus1": "/HHHTo6B_c3_2_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_4_d4_9": "/HHHTo6B_c3_4_d4_9_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_minus1_d4_0": "/HHHTo6B_c3_minus1_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_minus1_d4_minus1": "/HHHTo6B_c3_minus1_d4_minus1_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "HHHTo6B_c3_minus1p5_d4_minus0p5": "/HHHTo6B_c3_minus1p5_d4_minus0p5_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
            },
            "HH": {
                "VBFHHto4B_CV-1_C2V-1_C3-1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV-1_C2V-1_C3-1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "VBFHHto4B_CV-1_C2V-1_C3-2_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV-1_C2V-1_C3-2_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG_Pu60": "/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
            },
            "QCD": {
                **{
                    f"QCD_PT-{qbin}": [
                        f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                        f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                    ]
                    for qbin in qcd_bins
                },
                **{
                    f"QCD_HT-{qbin}": f"/QCD-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
            },
            "TT": {
                "TTto4Q": [
                    "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                    "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                ],
                "TTto2L2Nu": [
                    "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                    "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                ],
                "TTtoLNu2Q": [
                    "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                    "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                ],
            },
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "WWto4Q": [
                    "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                    "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                ],
                "WWto4Q_1Jets-4FS": "/WWto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "WZto4Q-1Jets-4FS": "/WZto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
            },
            "VJets": {
                # NLO
                "Wto2Q-2Jets_PTQQ-100to200_1J": "/Wto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-200to400_1J": "/Wto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-600_2J": "/Wto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                # LO
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                # NLO
                "Zto2Q-2Jets_PTQQ-100to200_1J": "/Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-200to400_2J": "/Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-400to600_2J": "/Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                # LO
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
            },
            "JetMET": {
                "Run2022E": "/JetMET/Run2022E-22Sep2023-v1/NANOAOD",
                # "Run2022F": "/JetMET/Run2022F-22Sep2023-v1/NANOAOD", # in production
                # "Run2022G": "/JetMET/Run2022G-22Sep2023-v1/NANOAOD", # in production
                "Run2022F": "/JetMET/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
                "Run2022G": "/JetMET/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
            },
            "Muon": {
                "Run2022E": "/Muon/Run2022E-22Sep2023-v1/NANOAOD",
                # "Run2022F": "/Muon/Run2022F-22Sep2023-v1/NANOAOD", # in production
                # "Run2022G": "/Muon/Run2022G-22Sep2023-v1/NANOAOD",
                "Run2022F": "/Muon/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
                "Run2022G": "/Muon/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
            },
        },
        "2023": {
            "JetMET": {
                "Run2023C-v1_0": "/JetMET0/Run2023C-22Sep2023_v1-v1/NANOAOD",
                "Run2023C-v2_0": "/JetMET0/Run2023C-22Sep2023_v2-v1/NANOAOD",
                "Run2023C-v3_0": "/JetMET0/Run2023C-22Sep2023_v3-v1/NANOAOD",
                "Run2023C-v4_0": "/JetMET0/Run2023C-22Sep2023_v4-v1/NANOAOD",
                "Run2023D-v1_0": "/JetMET0/Run2023D-22Sep2023_v1-v1/NANOAOD",
                "Run2023D-v2_0": "/JetMET0/Run2023D-22Sep2023_v2-v1/NANOAOD",
                "Run2023C-v1_1": "/JetMET1/Run2023C-22Sep2023_v1-v1/NANOAOD",
                "Run2023C-v2_1": "/JetMET1/Run2023C-22Sep2023_v2-v1/NANOAOD",
                "Run2023C-v3_1": "/JetMET1/Run2023C-22Sep2023_v3-v1/NANOAOD",
                "Run2023C-v4_1": "/JetMET1/Run2023C-22Sep2023_v4-v1/NANOAOD",
                "Run2023D-v1_1": "/JetMET1/Run2023D-22Sep2023_v1-v1/NANOAOD",
                "Run2023D-v2_1": "/JetMET1/Run2023D-22Sep2023_v2-v1/NANOAOD",
            },
            # should use Muon0+Muon1
            # https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset+status%3D*+dataset%3D%2FMuon*%2F*2023C*-22Sep2023_v*-*%2FNANOAOD
            # 22Sep2023 refers to 2022 Re-Mini+Re-Nano, that were built on top of particle rereco tag 27Jul2023 (for ERAs CDE) and on top of promptReco (for ERAs FG)
            "Muon": {
                "Run2023C-v1_0": "/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD",
                "Run2023C-v2_0": "/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD",
                "Run2023C-v3_0": "/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD",
                "Run2023C-v4_0": "/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD",
                "Run2023D-v1_0": "/Muon0/Run2023D-22Sep2023_v1-v1/NANOAOD",
                "Run2023D-v2_0": "/Muon0/Run2023D-22Sep2023_v2-v1/NANOAOD",
                "Run2023C-v1_1": "/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD",
                "Run2023C-v2_1": "/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD",
                "Run2023C-v3_1": "/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD",
                "Run2023C-v4_1": "/Muon1/Run2023C-22Sep2023_v4-v1/NANOAOD",
                "Run2023D-v1_1": "/Muon1/Run2023D-22Sep2023_v1-v1/NANOAOD",
                "Run2023D-v2_1": "/Muon1/Run2023D-22Sep2023_v2-v1/NANOAOD",
            },
            "MuonEG": {
                "Run2023C-v1_0": "/MuonEG/Run2023C-22Sep2023_v1-v1/NANOAOD",
                "Run2023C-v2_0": "/MuonEG/Run2023C-22Sep2023_v2-v1/NANOAOD",
                "Run2023C-v3_0": "/MuonEG/Run2023C-22Sep2023_v3-v1/NANOAOD",
                "Run2023C-v4_0": "/MuonEG/Run2023C-22Sep2023_v4-v1/NANOAOD",
                "Run2023D-v1_0": "/MuonEG/Run2023D-22Sep2023_v1-v1/NANOAOD",
                "Run2023D-v2_0": "/MuonEG/Run2023D-22Sep2023_v2-v1/NANOAOD",
            },
            "ParkingHH": {
                # missing v1 and v2 for RunC
                "Run2023C-v3": "/ParkingHH/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                "Run2023C-v4": "/ParkingHH/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                "Run2023D-v1": "/ParkingHH/Run2023D-PromptReco-v1/NANOAOD",
                "Run2023D-v2": "/ParkingHH/Run2023D-PromptReco-v2/NANOAOD",
            },
        },
    }


path_mkolosov = "/store/user/mkolosov/CRAB3_TransferData/PrivateNanoAOD/"
path_dihiggsboost = "/store/user/lpcdihiggsboost/nanov12/"
path_dihiggsboost_trigobj = "/store/user/lpcdihiggsboost/nanov11_trigobj/"


def get_v11_private():
    return {
        "2022": {
            "JetMET": {
                # path_to_dataset/folder/name_of_dataset
                "Run2022B_single": f"{path_mkolosov}/JetHT/JetHT_Run2022B_PromptReco_v1_11June2023",
                "Run2022C_single": f"{path_mkolosov}/JetHT/JetHT_Run2022C_PromptReco_v1_11June2023",
                "Run2022C": f"{path_mkolosov}/JetMET/JetMET_Run2022C_PromptReco_v1_11June2023",
                "Run2022D_v1": f"{path_mkolosov}/JetMET/JetMET_Run2022D_PromptReco_v1",
                "Run2022D_v2": f"{path_mkolosov}/JetMET/JetMET_Run2022D_PromptReco_v2",
                "Run2022D_v3": f"{path_mkolosov}/JetMET/JetMET_Run2022D_PromptReco_v3",
                # "Run2022C_single": f"{path_dihiggsboost_trigobj}/cmantill/2022/JetMET/JetHT/Run2022C_single",
                # "Run2022C": f"{path_dihiggsboost_trigobj}/cmantill/2022/JetMET/JetMET/Run2022C",
                # "Run2022D_v1": f"{path_dihiggsboost_trigobj}/cmantill/2022/JetMET/JetMET/Run2022D_v1",
                # "Run2022D_v2": f"{path_dihiggsboost_trigobj}/cmantill/2022/JetMET/JetMET/Run2022D_v2",
                # "Run2022D_v3": f"{path_dihiggsboost_trigobj}/cmantill/2022/JetMET/JetMET/Run2022D_v3",
            },
            "MuonEG": {
                "Run2022B": f"{path_mkolosov}/MuonEG/MuonEG_Run2022B_PromptReco_v1_11June2023",
                "Run2022C": f"{path_mkolosov}/MuonEG/MuonEG_Run2022C_PromptReco_v1_11June2023",
                "Run2022D_v1": f"{path_mkolosov}/MuonEG/MuonEG_Run2022D_PromptReco_v1",
                "Run2022D_v2": f"{path_mkolosov}/MuonEG/MuonEG_Run2022D_PromptReco_v2",
                "Run2022D_v3": f"{path_mkolosov}/MuonEG/MuonEG_Run2022D_PromptReco_v3",
            },
            "Muon": {
                # "Run2022B_single": f"{path_mkolosov}/SingleMuon/SingleMuon_Run2022B_PromptReco_v1_11June2023",
                # "Run2022C_single": f"{path_mkolosov}/SingleMuon/SingleMuon_Run2022C_PromptReco_v1_11June2023",
                # "Run2022C": f"{path_mkolosov}/Muon/Muon_Run2022C_PromptReco_v1_11June2023",
                # "Run2022D_v1": f"{path_mkolosov}/Muon/Muon_Run2022D_PromptReco_v1_11June2023",
                # "Run2022D_v2": f"{path_mkolosov}/Muon/Muon_Run2022D_PromptResco_v2_11June2023",
                "Run2022C_single": f"{path_dihiggsboost_trigobj}/cmantill/2022/Muon/SingleMuon/Muon_Run2022C_single/",
                "Run2022C": f"{path_dihiggsboost_trigobj}/cmantill/2022/Muon/Muon/Muon_Run2022C/",
                "Run2022D_v1": f"{path_dihiggsboost_trigobj}/cmantill/2022/Muon/Muon/Muon_Run2022D_v1/",
                "Run2022D_v2": f"{path_dihiggsboost_trigobj}/cmantill/2022/Muon/Muon/Muon_Run2022D_v2/",
                "Run2022D_v3": f"{path_dihiggsboost_trigobj}/cmantill/2022/Muon/Muon/Muon_Run2022D_v3/",
            },
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": f"{path_dihiggsboost_trigobj}/cmantill/2022/HH/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": f"{path_dihiggsboost_trigobj}/cmantill/2022/HH/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8",
                # "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": f"{path_dihiggsboost}/cmantill/2022/HH/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": f"{path_dihiggsboost}/cmantill/2022/Diboson/WW_TuneCP5_13p6TeV_pythia8",
                "WZ": f"{path_dihiggsboost}/cmantill/2022/Diboson/WZ_TuneCP5_13p6TeV_pythia8",
                "ZZ": f"{path_dihiggsboost}/cmantill/2022/Diboson/ZZ_TuneCP5_13p6TeV_pythia8",
            },
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8",
                "VBFHto2B_M-125_dipoleRecoilOn": f"{path_dihiggsboost}/cmantill/2022/Higgs/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8",
                "WminusH_Hto2B_Wto2Q_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "WminusH_Hto2B_WtoLNu_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "WplusH_Hto2B_Wto2Q_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "WplusH_Hto2B_WtoLNu_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ZH_Hto2B_Zto2L_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ZH_Hto2B_Zto2Q_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ggZH_Hto2B_Zto2L_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ggZH_Hto2B_Zto2Nu_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ggZH_Hto2B_Zto2Q_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/ggZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ttHto2B_M-125": f"{path_dihiggsboost}/cmantill/2022/Higgs/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "TT": {
                "TTto4Q": f"{path_dihiggsboost}/cmantill/2022/TTbar/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": f"{path_dihiggsboost}/cmantill/2022/TTbar/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": f"{path_dihiggsboost}/cmantill/2022/TTbar/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": f"{path_dihiggsboost}/cmantill/2022/VJets/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Wto2Q-3Jets_HT-400to600": f"{path_dihiggsboost}/cmantill/2022/VJets/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Wto2Q-3Jets_HT-600to800": f"{path_dihiggsboost}/cmantill/2022/VJets/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Wto2Q-3Jets_HT-800": f"{path_dihiggsboost}/cmantill/2022/VJets/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-200to400": f"{path_dihiggsboost}/cmantill/2022/VJets/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-400to600": f"{path_dihiggsboost}/cmantill/2022/VJets/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-600to800": f"{path_dihiggsboost}/cmantill/2022/VJets/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-800": f"{path_dihiggsboost}/cmantill/2022/VJets/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "QCD": {
                **{
                    f"QCD_PT-{qbin}": f"{path_dihiggsboost}/cmantill/2022/QCD/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8"
                    for qbin in qcd_bins
                },
                **{
                    f"QCDB_HT-{qbin}": f"{path_dihiggsboost}/cmantill/2022/QCD/QCDB-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8"
                    for qbin in qcd_ht_bins
                },
            },
        },
        "2022EE": {
            "JetMET": {
                # "Run2022E": f"{path_mkolosov}/JetMET/JetMET_Run2022E_PromptReco_v1",
                # "Run2022F": f"{path_mkolosov}/JetMET/JetMET_Run2022F_PromptReco_v1",
                # "Run2022G": f"{path_mkolosov}/JetMET/JetMET_Run2022G_PromptReco_v1",
                "Run2022E": f"{path_dihiggsboost_trigobj}/cmantill/2022EE/JetMET/JetMET/JetMET_Run2022E",
                "Run2022F": f"{path_dihiggsboost_trigobj}/cmantill/2022EE/JetMET/JetMET/JetMET_Run2022F",
                "Run2022G": f"{path_dihiggsboost_trigobj}/cmantill/2022EE/JetMET/JetMET/JetMET_Run2022G",
            },
            "MuonEG": {
                "Run2022E": f"{path_mkolosov}/MuonEG/MuonEG_Run2022E_PromptReco_v1",
                "Run2022F": f"{path_mkolosov}/MuonEG/MuonEG_Run2022F_PromptReco_v1",
                "Run2022G": f"{path_mkolosov}/MuonEG/MuonEG_Run2022G_PromptReco_v1",
            },
            "Muon": {
                # "Run2022E": f"{path_mkolosov}/Muon/Muon_Run2022E_PromptReco_v1_11June2023",
                # "Run2022F": f"{path_mkolosov}/Muon/Muon_Run2022F_PromptReco_v1_11June2023",
                # "Run2022G": f"{path_mkolosov}/Muon/Muon_Run2022G_PromptReco_v1_11June2023",
                "Run2022E": f"{path_dihiggsboost_trigobj}/cmantill/2022EE/Muon/Muon/Muon_Run2022E",
                "Run2022F": f"{path_dihiggsboost_trigobj}/cmantill/2022EE/Muon/Muon/Muon_Run2022F",
                "Run2022G": f"{path_dihiggsboost_trigobj}/cmantill/2022EE/Muon/Muon/Muon_Run2022G",
            },
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": f"{path_dihiggsboost}/cmantill/2022EE/HH/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": f"{path_dihiggsboost}/cmantill/2022EE/Diboson/WW_TuneCP5_13p6TeV_pythia8",
                "WZ": f"{path_dihiggsboost}/cmantill/2022EE/Diboson/WZ_TuneCP5_13p6TeV_pythia8",
                "ZZ": f"{path_dihiggsboost}/cmantill/2022EE/Diboson/ZZ_TuneCP5_13p6TeV_pythia8",
            },
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8",
                "VBFHto2B_M-125_dipoleRecoilOn": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8",
                "WminusH_Hto2B_Wto2Q_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "WminusH_Hto2B_WtoLNu_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "WplusH_Hto2B_Wto2Q_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "WplusH_Hto2B_WtoLNu_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ZH_Hto2B_Zto2L_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ZH_Hto2B_Zto2Q_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ggZH_Hto2B_Zto2L_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ggZH_Hto2B_Zto2Nu_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ggZH_Hto2B_Zto2Q_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/ggZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8",
                "ttHto2B_M-125": f"{path_dihiggsboost}/cmantill/2022EE/Higgs/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "TT": {
                "TTto4Q": f"{path_dihiggsboost}/cmantill/2022EE/TTbar/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": f"{path_dihiggsboost}/cmantill/2022EE/TTbar/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": f"{path_dihiggsboost}/cmantill/2022EE/TTbar/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": f"{path_dihiggsboost}/cmantill/2022EE/VJets/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Wto2Q-3Jets_HT-400to600": f"{path_dihiggsboost}/cmantill/2022EE/VJets/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Wto2Q-3Jets_HT-600to800": f"{path_dihiggsboost}/cmantill/2022EE/VJets/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Wto2Q-3Jets_HT-800": f"{path_dihiggsboost}/cmantill/2022EE/VJets/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-200to400": f"{path_dihiggsboost}/cmantill/2022EE/VJets/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-400to600": f"{path_dihiggsboost}/cmantill/2022EE/VJets/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-600to800": f"{path_dihiggsboost}/cmantill/2022EE/VJets/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-800": f"{path_dihiggsboost}/cmantill/2022EE/VJets/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "QCD": {
                **{
                    f"QCD_PT-{qbin}": f"{path_dihiggsboost}/cmantill/2022EE/QCD/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8"
                    for qbin in qcd_bins
                },
                **{
                    f"QCDB_HT-{qbin}": f"{path_dihiggsboost}/cmantill/2022EE/QCD/QCDB-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8"
                    for qbin in qcd_ht_bins
                },
            },
        },
    }


def eos_rec_search(startdir, suffix, dirs):
    eosbase = "root://cmseos.fnal.gov/"
    try:
        dirlook = (
            subprocess.check_output(f"eos {eosbase} ls {startdir}", shell=True)
            .decode("utf-8")
            .split("\n")[:-1]
        )
    except:
        print(f"No files found for {startdir}")
        return dirs

    donedirs = [[] for d in dirlook]
    for di, d in enumerate(dirlook):
        if d.endswith(suffix):
            donedirs[di].append(startdir + "/" + d)
        elif d == "log":
            continue
        else:
            donedirs[di] = donedirs[di] + eos_rec_search(
                startdir + "/" + d, suffix, dirs + donedirs[di]
            )
    donedir = [d for da in donedirs for d in da]
    return dirs + donedir


def get_files(dataset, version):
    if "private" in version:
        files = eos_rec_search(dataset, ".root", [])
        return [f"root://cmsxrootd-site.fnal.gov/{f}" for f in files]
    else:
        import requests
        from rucio_utils import get_dataset_files, get_proxy_path

        proxy = get_proxy_path()
        link = f"https://cmsweb.cern.ch:8443/dbs/prod/global/DBSReader/files?dataset={dataset}&detail=True"
        r = requests.get(
            link,
            cert=proxy,
            verify=False,
        )
        filesjson = r.json()
        files = []
        not_valid = []
        for fj in filesjson:
            if "is_file_valid" in fj:
                if fj["is_file_valid"] == 0:
                    print(f"ERROR: File not valid on DAS: {fj['logical_file_name']}")
                    not_valid.append(fj["logical_file_name"])
                else:
                    files.append(fj["logical_file_name"])
            else:
                continue

        if "WplusH_Hto2B_Wto2Q_M-125" in dataset and year == "2022EE":
            not_valid = [
                "/store/mc/Run3Summer22EENanoAODv11/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/126X_mcRun3_2022_realistic_postEE_v1-v1/30000/ff739627-b8b6-46be-8432-eef281ffe178.root"
            ]

        if len(files) == 0:
            print(f"Found 0 files for sample {dataset}!")
            return []

        # Now query rucio to get the concrete dataset passing the sites filtering options
        sites_cfg = {
            "whitelist_sites": None,
            "blacklist_sites": ["T2_FR_IPHC"],
            "regex_sites": None,
        }
        # append trouble datasets for now..
        sites_cfg["blacklist_sites"].append("T2_US_MIT")
        sites_cfg["blacklist_sites"].append("T2_US_Vanderbilt")
        sites_cfg["blacklist_sites"].append("T2_UK_London_Brunel")
        sites_cfg["blacklist_sites"].append("T2_UK_SGrid_RALPP")
        sites_cfg["blacklist_sites"].append("T1_UK_RAL_Disk")

        if "QCD-4Jets_HT-600to800" in dataset:
            sites_cfg["blacklist_sites"].append("T3_KR_KNU")
        files_rucio, sites = get_dataset_files(dataset, **sites_cfg, output="first")

        # Get rid of invalid files
        files_valid = []
        for f in files_rucio:
            invalid = False
            for nf in not_valid:
                if nf in f:
                    invalid = True
                    break
            if not invalid:
                files_valid.append(f)

        return files_valid


for version in ["v9", "v10", "v11", "v11_private", "v9_private", "v12"]:
    # for version in ["v11"]:
    datasets = globals()[f"get_{version}"]()
    index = datasets.copy()
    for year, ydict in datasets.items():
        for sample, sdict in ydict.items():
            for sname, dataset in sdict.items():
                if isinstance(dataset, list):
                    files = []
                    for d in dataset:
                        files.extend(get_files(d, version))
                    index[year][sample][sname] = files
                else:
                    index[year][sample][sname] = get_files(dataset, version)

    with Path(f"nanoindex_{version}.json").open("w") as f:
        json.dump(index, f, indent=4)
