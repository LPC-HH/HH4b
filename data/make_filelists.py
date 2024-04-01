from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/x86_64/rhel7/py3/current"

# request for non-SM signal samples
# https://gitlab.cern.ch/cms-hig-mc/mc-hig-requests/-/issues/55

# https://cms-pdmv-prod.web.cern.ch/pmp/present?r=HIG-Run3Summer22wmLHEGS-00351


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
    # "40to70",
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
            "TT": {
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
            "TT": {
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
            "JetHT": {
                "JetHT_Run2018A": f"{path_hqu}/data/JetHT/NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4_Run2018A-UL2018_MiniAODv2_GT36-v1",
                "JetHT_Run2018B": f"{path_hqu}/data/JetHT/NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4_Run2018B-UL2018_MiniAODv2_GT36-v1",
                "JetHT_Run2018C": f"{path_hqu}/data/JetHT/NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4_Run2018C-UL2018_MiniAODv2_GT36-v1",
                "JetHT_Run2018D": f"{path_hqu}/data/JetHT/NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4_Run2018D-UL2018_MiniAODv2_GT36-v1",
            },
        }
    }


# ntuples used in run-2 analysis
def get_v9_hh_private():
    return {
        "2018": {
            "HH": {
                "GluGlutoHHto4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/store/group/lpcdihiggsboost/NanoTuples/V2p0/MC_Autumn18/v1/GluGluToHHTo4B_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8"
            },
            # "Hbb": {
            #     "GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/GluGluHToBB_M-125_13TeV_powheg_MINLO_NNLOPS_pythia8",
            #     "VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/VBFHToBB_M-125_13TeV_powheg_pythia8_weightfix",
            #     "WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/WminusH_HToBB_WToLNu_M125_13TeV_powheg_pythia8",
            #     "WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/WplusH_HToBB_WToLNu_M125_13TeV_powheg_pythia8",
            # },
            # "Diboson": {
            #     "ZZ_TuneCP5_13TeV-pythia8": "/store/group/lpcdihiggsboost/NanoTuples/V2p0/MC_Autumn18/v1/ZZ_TuneCP5_13TeV-pythia8",
            # },
            # "QCD": {
            #     **{
            #         f"QCD_HT-{qbin}-13TeV": f"/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/QCD_HT{qbin}_TuneCP5_13TeV-madgraphMLM-pythia8"
            #         for qbin in qcd_ht_bins_run2
            #     },
            # },
            # "TT": {
            #     "TTToHadronic_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",
            #     "TTTo2L2Nu_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
            #     "TTToSemiLeptonic_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
            # },
            # "VJets": {
            #     "WJetsToQQ_HT-400to600_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/WJetsToQQ_HT400to600_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8",
            #     "WJetsToQQ_HT-600to800_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/WJetsToQQ_HT600to800_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8",
            #     "WJetsToQQ_HT-800toInf_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/WJetsToQQ_HT-800toInf_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8",
            #     "ZJetsToQQ_HT-400to600_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/ZJetsToQQ_HT400to600_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8",
            #     "ZJetsToQQ_HT-600to800_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/ZJetsToQQ_HT600to800_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8",
            #     "ZJetsToQQ_HT-800toInf_13TeV": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/ZJetsToQQ_HT-800toInf_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8",
            # },
            # "JetHT": {
            #     "JetHT_Run2018A": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/data/JetHT/NanoTuples-30Apr2020_Run2018A-17Sep2018-v1",
            #     "JetHT_Run2018B": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/data/JetHT/NanoTuples-30Apr2020_Run2018B-17Sep2018-v1",
            #     "JetHT_Run2018C": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/data/JetHT/NanoTuples-30Apr2020_Run2018C-17Sep2018-v1",
            #     "JetHT_Run2018D": "/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/data/JetHT/NanoTuples-30Apr2020_Run2018D-PromptReco-v2",
            # },
        }
    }


def get_v11():
    return {
        "2022EE": {
            # "Hbb": {
            #    "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            # },
            "QCD": {
                **{
                    f"QCD_HT-{qbin}": f"/QCD-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
            },
            "TT": {
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            },
        }
    }


def get_v11_private():
    return {
        "2022EE": {
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/user/lpcdihiggsboost/cmantill/NanoAOD_v11/2022EE/HH/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
            }
        }
    }


def get_v12_private():
    return {
        "2022": {
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "VBFHto2B_M-125_dipoleRecoilOn": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ttHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/Hbb/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "QCD": {
                "QCD_HT-1000to1200": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-100to200": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-200to400": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-400to600": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-2000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1500to2000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1200to1500": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-600to800": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-800to1000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2022/QCD/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "TT": {
                "TTto4Q": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/TT/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/TT/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/TT/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/diboson/WW_TuneCP5_13p6TeV_pythia8/",
                "WZ": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/diboson/WZ_TuneCP5_13p6TeV_pythia8/",
                "ZZ": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/diboson/ZZ_TuneCP5_13p6TeV_pythia8/",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/VJets/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-400to600": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/VJets/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-600to800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/VJets/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/VJets/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-200to400": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/VJets/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-400to600": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/VJets/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-600to800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/VJets/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2022/VJets/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
            },
            "JetMET": {
                "JetMET_Run2022C_single": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022/JetMET/JetHT/JetHT_Run2022C/",
                "JetMET_Run2022C": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022/JetMET/JetMET/JetMET_Run2022C/",
                "JetMET_Run2022D": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022/JetMET/JetMET/JetMET_Run2022D/",
            },
        },
        "2022EE": {
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/HH4b/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "VBFHto2B_M-125_dipoleRecoilOn": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ttHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/Hbb/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "QCD": {
                "QCD_HT-1000to1200": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-100to200": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-200to400": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-400to600": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-2000": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1500to2000": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1200to1500": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-600to800": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-800to1000": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/QCD/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "TT": {
                "TTto4Q": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/TT/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/TT/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": "/store/group/lpcdihiggsboost/cmantill/NanoAOD_v12/2022EE/TT/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "Diboson": {
                "WW": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/diboson/WW_TuneCP5_13p6TeV_pythia8/",
                "WZ": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/diboson/WZ_TuneCP5_13p6TeV_pythia8/",
                "ZZ": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/diboson/ZZ_TuneCP5_13p6TeV_pythia8/",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/VJets/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-400to600": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/VJets/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-600to800": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/VJets/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-800": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/VJets/",
                "Zto2Q-4Jets_HT-200to400": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/VJets/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-400to600": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/VJets/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-600to800": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/VJets/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-800": "/store/group/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/VJets/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
            },
            "JetMET": {
                "JetMET_Run2022E": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/JetMET/JetMET/JetMET_Run2022E",
                "JetMET_Run2022F": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/JetMET/JetMET/JetMET_Run2022F",
                "JetMET_Run2022G": "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2022EE/JetMET/JetMET/JetMET_Run2022G",
            },
        },
        "2023": {
            "JetMET": {
                "JetMET_Run2023Cv1": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-pre-Bpix/JetMET/JetMET0/JetMET_Run2023C_0v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-pre-Bpix/JetMET/JetMET1/JetMET_Run2023C_1v1/",
                ],
                "JetMET_Run2023Cv2": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-pre-Bpix/JetMET/JetMET0/JetMET_Run2023C_0v2/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-pre-Bpix/JetMET/JetMET1/JetMET_Run2023C_1v2/",
                ],
                "JetMET_Run2023Cv3": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-pre-Bpix/JetMET/JetMET0/JetMET_Run2023C_0v3/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-pre-Bpix/JetMET/JetMET1/JetMET_Run2023C_1v3/",
                ],
                "JetMET_Run2023Cv4": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-pre-Bpix/JetMET/JetMET0/JetMET_Run2023C_0v4/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-pre-Bpix/JetMET/JetMET1/JetMET_Run2023C_1v4/",
                ],
            },
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "VBFHto2B_M-125_dipoleRecoilOn": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/VBFHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ttHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/Hbb/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "QCD": {
                "QCD_HT-1000to1200": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-100to200": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-200to400": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-400to600": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-2000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1500to2000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1200to1500": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-600to800": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-800to1000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/QCD/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "TT": {
                "TTto4Q": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/TT/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/TT/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/TT/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/diboson/WW_TuneCP5_13p6TeV_pythia8/",
                "WZ": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/diboson/WZ_TuneCP5_13p6TeV_pythia8/",
                "ZZ": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/diboson/ZZ_TuneCP5_13p6TeV_pythia8/",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/VJets/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-400to600": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/VJets/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-600to800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/VJets/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/VJets/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-200to400": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/VJets/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-400to600": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/VJets/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-600to800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/VJets/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-pre-Bpix/VJets/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
            },
        },
        "2023BPix": {
            "JetMET": {
                "JetMET_Run2023D": [
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-Bpix/JetMET/JetMET0/JetMET_Run2023D_0v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-Bpix/JetMET/JetMET0/JetMET_Run2023D_0v2/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-Bpix/JetMET/JetMET1/JetMET_Run2023D_1v1/",
                    "/store/user/lpcdihiggsboost/NanoAOD_v12/cmantill/2023-Bpix/JetMET/JetMET1/JetMET_Run2023D_1v2/",
                ],
            },
            "Hbb": {
                "GluGluHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/GluGluHto2B_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "GluGluHto2B_PT-200_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/",
                "VBFHto2B_M-125_dipoleRecoilOn": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/VBFHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WminusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_Wto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "WplusH_Hto2B_WtoLNu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2L_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Nu_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ggZH_Hto2B_Zto2Q_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
                "ttHto2B_M-125": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/Hbb/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/",
            },
            "QCD": {
                "QCD_HT-1000to1200": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-100to200": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-200to400": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-400to600": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-2000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1500to2000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-1200to1500": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-600to800": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "QCD_HT-800to1000": "/store/user/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/QCD/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8",
            },
            "TT": {
                "TTto4Q": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/TT/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
                "TTto2L2Nu": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/TT/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                "TTtoLNu2Q": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/TT/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            },
            "Diboson": {
                "WW": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/diboson/WW_TuneCP5_13p6TeV_pythia8/",
                "WZ": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/diboson/WZ_TuneCP5_13p6TeV_pythia8/",
                "ZZ": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/diboson/ZZ_TuneCP5_13p6TeV_pythia8/",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/VJets/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-400to600": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/VJets/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-600to800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/VJets/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Wto2Q-3Jets_HT-800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/VJets/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
                "Zto2Q-4Jets_HT-200to400": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/VJets/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-400to600": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/VJets/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-600to800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/VJets/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
                "Zto2Q-4Jets_HT-800": "/store/group/lpcdihiggsboost/NanoAOD_v12/sixie/2023-Bpix/VJets/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/",
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
                "ZH_Hto2B_Zto2Q_M-125": [
                    "/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                    "/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v3/NANOAODSIM",
                ],
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
                # private 10M?
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_Private": "/ggHHTo4B_cHHH_1/lpcdihiggsboost-crab_PrivateProduction_Summer22_DR_step4_NANOAODSIM_ggHHTo4B_cHHH_1_batch1_v1-00000000000000000000000000000000/USER",
                # 1M events
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                # 100k events
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
            "ZHH": {
                "ZHH_HHto4B_CV_1_0_C2V_1_0_C3_1_0_13p6TeV_TuneCP5_madgraph-pythia8": "/ZHH_HHto4B_CV_1_0_C2V_1_0_C3_1_0_13p6TeV_TuneCP5_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
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
                # missing ZZto4Q
            },
            "VJets": {
                # NLO
                "Wto2Q-2Jets_PTQQ-100to200_1J": "/Wto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-100to200_2J": "/Wto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-200to400_1J": "/Wto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-200to400_2J": "/Wto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-400to600_1J": "/Wto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-400to600_2J": "/Wto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-600_1J": "/Wto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-600_2J": "/Wto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                # LO
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                # NLO
                "Zto2Q-2Jets_PTQQ-100to200_1J": "/Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-100to200_2J": "/Zto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-200to400_1J": "/Zto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-200to400_2J": "/Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-400to600_1J": "/Zto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-400to600_2J": "/Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-600_1J": "/Zto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-600_2J": "/Zto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM",
                # LO
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                # WJetsToLNu
                "WtoLNu-2Jets": "/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                "WtoLNu-4Jets": "/WtoLNu-4Jets_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
                # DYToLL
                "DYto2L-4Jets_MLL-50": "/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",
            },
            "JetMET": {
                "JetMET_Run2022C_single": "/JetHT/Run2022C-22Sep2023-v1/NANOAOD",
                "JetMET_Run2022C": "/JetMET/Run2022C-22Sep2023-v1/NANOAOD",
                "JetMET_Run2022D": "/JetMET/Run2022D-22Sep2023-v1/NANOAOD",
            },
            "Muon": {
                "Muon_Run2022C_single": "/SingleMuon/Run2022C-22Sep2023-v1/NANOAOD",
                "Muon_Run2022C": "/Muon/Run2022C-22Sep2023-v1/NANOAOD",
                "Muon_Run2022D": "/Muon/Run2022D-22Sep2023-v1/NANOAOD",
            },
            "EGamma": {
                "EGamma_Run2022C": "/EGamma/Run2022C-22Sep2023-v1/NANOAOD",
                "EGamma_Run2022D": "/EGamma/Run2022D-22Sep2023-v1/NANOAOD",
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
                # 3M events
                "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8": "/VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM",
                # 7.3M events
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                # 1.4M events (production):
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV": "/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                # TSG samples
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
            "ZHH": {
                "ZHH_HHto4B_CV_1_0_C2V_1_0_C3_1_0_13p6TeV_TuneCP5_madgraph-pythia8": "/ZHH_HHto4B_CV_1_0_C2V_1_0_C3_1_0_13p6TeV_TuneCP5_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
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
                # "TWminusto4Q": "/TWminusto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                # "TbarWplusto4Q": "/TbarWplusto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
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
                "Wto2Q-2Jets_PTQQ-100to200_2J": "/Wto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-200to400_1J": "/Wto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-200to400_2J": "/Wto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-400to600_1J": "/Wto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-400to600_2J": "/Wto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-600_1J": "/Wto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-600_2J": "/Wto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                # LO
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                # WJetsToLNu NLO
                "WtoLNu-2Jets": "/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                # WJetsToLNu LO
                "WtoLNu-4Jets": "/WtoLNu-4Jets_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                # NLO
                "Zto2Q-2Jets_PTQQ-100to200_1J": "/Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-100to200_2J": "/Zto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-200to400_1J": "/Zto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-200to400_2J": "/Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-400to600_1J": "/Zto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-400to600_2J": "/Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-600_1J": "/Zto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-600_2J": "/Zto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
                # LO
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                # DYJetsToLL NLO
                "DYto2L-2Jets_MLL-50": "/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                # DYJetsToLL LO
                # /DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v5-v2/NANOAODSIM
            },
            "JetMET": {
                "JetMET_Run2022E": "/JetMET/Run2022E-22Sep2023-v1/NANOAOD",
                "JetMET_Run2022F": "/JetMET/Run2022F-22Sep2023-v2/NANOAOD",
                "JetMET_Run2022G": "/JetMET/Run2022G-22Sep2023-v2/NANOAOD",
            },
            "Muon": {
                "Muon_Run2022E": "/Muon/Run2022E-22Sep2023-v1/NANOAOD",
                "Muon_Run2022F": "/Muon/Run2022F-22Sep2023-v2/NANOAOD",
                "Muon_Run2022G": "/Muon/Run2022G-22Sep2023-v1/NANOAOD",
            },
            "EGamma": {
                "EGamma_Run2022E": "/EGamma/Run2022E-22Sep2023-v1/NANOAOD",
                "EGamma_Run2022F": "/EGamma/Run2022F-22Sep2023-v1/NANOAOD",
                "EGamma_Run2022G": "/EGamma/Run2022G-22Sep2023-v2/NANOAOD",
            },
        },
        "2023": {
            "JetMET": {
                # B is commissioning ERA
                # "JetMET_Run2023B": [
                #     "/JetMET0/Run2023B-22Sep2023-v1/NANOAOD",
                #     "/JetMET1/Run2023B-22Sep2023-v2/NANOAOD",
                # ],
                "JetMET_Run2023Cv1": [
                    "/JetMET0/Run2023C-22Sep2023_v1-v1/NANOAOD",
                    "/JetMET1/Run2023C-22Sep2023_v1-v1/NANOAOD",
                ],
                "JetMET_Run2023Cv2": [
                    "/JetMET0/Run2023C-22Sep2023_v2-v1/NANOAOD",
                    "/JetMET1/Run2023C-22Sep2023_v2-v1/NANOAOD",
                ],
                "JetMET_Run2023Cv3": [
                    "/JetMET0/Run2023C-22Sep2023_v3-v1/NANOAOD",
                    "/JetMET1/Run2023C-22Sep2023_v3-v1/NANOAOD",
                ],
                "JetMET_Run2023Cv4": [
                    "/JetMET0/Run2023C-22Sep2023_v4-v1/NANOAOD",
                    "/JetMET1/Run2023C-22Sep2023_v4-v1/NANOAOD",
                ],
            },
            # should use Muon0+Muon1
            # 22Sep2023 refers to 2022 Re-Mini+Re-Nano, that were built on top of particle rereco tag 27Jul2023 (for ERAs CDE) and on top of promptReco (for ERAs FG)
            "Muon": {
                "Muon_Run2023Cv1": [
                    "/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD",
                    "/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD",
                ],
                "Muon_Run2023Cv2": [
                    "/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD",
                    "/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD",
                ],
                "Muon_Run2023Cv3": [
                    "/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD",
                    "/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD",
                ],
                "Muon_Run2023Cv4": [
                    "/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD",
                    "/Muon1/Run2023C-22Sep2023_v4-v1/NANOAOD",
                ],
            },
            "EGamma": {
                "EGamma_Run2023Cv1": [
                    "/EGamma0/Run2023C-22Sep2023_v1-v1/NANOAOD",
                    "/EGamma1/Run2023C-22Sep2023_v1-v1/NANOAOD",
                ],
                "EGamma_Run2023Cv2": [
                    "/EGamma0/Run2023C-22Sep2023_v2-v1/NANOAOD",
                    "/EGamma1/Run2023C-22Sep2023_v2-v1/NANOAOD",
                ],
                "EGamma_Run2023Cv3": [
                    "/EGamma0/Run2023C-22Sep2023_v3-v1/NANOAOD",
                    "/EGamma1/Run2023C-22Sep2023_v3-v1/NANOAOD",
                ],
                "EGamma_Run2023Cv4": [
                    "/EGamma0/Run2023C-22Sep2023_v4-v1/NANOAOD",
                    "/EGamma1/Run2023C-22Sep2023_v4-v1/NANOAOD",
                ],
            },
            "QCD": {
                "QCD_HT-1000to1200": "/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-100to200": "/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-1200to1500": "/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-1500to2000": "/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "QCD_HT-2000": "/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-200to400": "/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-400to600": "/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "QCD_HT-40to70": "/QCD-4Jets_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-600to800": "/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-70to100": "/QCD-4Jets_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-800to1000": "/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
            },
            "HH": {
                # 1M (TSG?)
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v4/NANOAODSIM",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v4/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-tsg_130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG": "/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v4/NANOAODSIM",
            },
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "VBFHto2B_M-125_dipoleRecoilOn": "/VBFHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WminusH_Hto2B_Wto2Q_M-125": "/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WminusH_Hto2B_WtoLNu_M-125": "/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "WplusH_Hto2B_Wto2Q_M-125": "/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "WplusH_Hto2B_WtoLNu_M-125": "/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "ZH_Hto2B_Zto2L_M-125": "/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "ZH_Hto2B_Zto2Q_M-125": "/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2L_M-125": "/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "ggZH_Hto2B_Zto2Nu_M-125": "/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "ggZH_Hto2B_Zto2Q_M-125": "/ggZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "ttHto2B_M-125": "/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
            },
            "HHH": {
                "HHHTo6B_c3_0_d4_0": "/HHHTo6B_c3_0_d4_0_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
            },
            "TT": {
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
            },
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "WWto4Q": "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                # /Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM
                # /Zto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM
                # /Zto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM
                # /Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM
                # /Zto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM
                # /Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM
                # /Zto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM
                # /Zto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM",
                # "WtoLNu-2Jets_0J": "/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                # "WtoLNu-2Jets_1J": "/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                # "WtoLNu-2Jets_2J": "/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
                "WtoLNu-4Jets": "/WtoLNu-4Jets_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
            },
        },
        "2023BPix": {
            "JetMET": {
                "JetMET_Run2023D": [
                    "/JetMET0/Run2023D-22Sep2023_v1-v1/NANOAOD",
                    "/JetMET0/Run2023D-22Sep2023_v2-v1/NANOAOD",
                    "/JetMET1/Run2023D-22Sep2023_v1-v1/NANOAOD",
                    "/JetMET1/Run2023D-22Sep2023_v2-v1/NANOAOD",
                ],
            },
            "Muon": {
                "Muon_Run2023D": [
                    "/Muon0/Run2023D-22Sep2023_v1-v1/NANOAOD",
                    "/Muon0/Run2023D-22Sep2023_v2-v1/NANOAOD",
                    "/Muon1/Run2023D-22Sep2023_v1-v1/NANOAOD",
                    "/Muon1/Run2023D-22Sep2023_v2-v1/NANOAOD",
                ],
            },
            "EGamma": {
                "EGamma_Run2023D": [
                    "/EGamma0/Run2023D-22Sep2023_v1-v1/NANOAOD",
                    "/EGamma0/Run2023D-22Sep2023_v2-v1/NANOAOD",
                    "/EGamma1/Run2023D-22Sep2023_v1-v1/NANOAOD",
                    "/EGamma1/Run2023D-22Sep2023_v2-v1/NANOAOD",
                ],
            },
            "HH": {
                # 500k in production (TSG)
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-tsg_130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM",
            },
            "QCD": {
                "QCD_HT-1000to1200": "/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM",
                "QCD_HT-100to200": "/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM",
                "QCD_HT-1200to1500": "/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM",
                "QCD_HT-1500to2000": "/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "QCD_HT-2000": "/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM",
                "QCD_HT-200to400": "/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "QCD_HT-400to600": "/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM",
                "QCD_HT-40to70": "/QCD-4Jets_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "QCD_HT-600to800": "/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "QCD_HT-70to100": "/QCD-4Jets_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "QCD_HT-800to1000": "/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM",
            },
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "VBFHto2B_M-125_dipoleRecoilOn": "/VBFHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "WminusH_Hto2B_Wto2Q_M-125": "/WminusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "WminusH_Hto2B_WtoLNu_M-125": "/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "WplusH_Hto2B_Wto2Q_M-125": "/WplusH_Hto2B_Wto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "WplusH_Hto2B_WtoLNu_M-125": "/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "ZH_Hto2B_Zto2L_M-125": "/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "ZH_Hto2B_Zto2Q_M-125": "/ZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2L_M-125": "/ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "ggZH_Hto2B_Zto2Nu_M-125": "/ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "ggZH_Hto2B_Zto2Q_M-125": "/ggZH_Hto2B_Zto2Q_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "ttHto2B_M-125": "/ttHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
            },
            "TT": {
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
            },
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "WWto4Q": "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
            },
            "VJets": {
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM",
                "WtoLNu-2Jets_0J": "/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_1J": "/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_2J": "/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM",
                "WtoLNu-4Jets": "/WtoLNu-4Jets_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM",
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
        if "USER" in dataset:
            link = f"https://cmsweb.cern.ch:8443/dbs/prod/phys03/DBSReader/files?dataset={dataset}&detail=True"
        else:
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
                    # print(f"ERROR: File not valid on DAS: {fj['logical_file_name']}")
                    not_valid.append(fj["logical_file_name"])
                else:
                    files.append(fj["logical_file_name"])
            else:
                continue

        if "USER" in dataset:
            files_valid = [f"root://cmseos.fnal.gov/{f}" for f in files]
            return files_valid

        if len(files) == 0:
            print(f"Found 0 files for sample {dataset}!")
            return []

        # Now query rucio to get the concrete dataset passing the sites filtering options
        sites_cfg = {
            "whitelist_sites": [],
            "blacklist_sites": [
                "T2_FR_IPHC" "T2_US_MIT",
                "T2_US_Vanderbilt",
                "T2_UK_London_Brunel",
                "T2_UK_SGrid_RALPP",
                "T1_UK_RAL_Disk",
                "T2_PT_NCG_Lisbon",
            ],
            "regex_sites": None,
        }
        if version == "v12" or version == "v11":
            sites_cfg["whitelist_sites"] = ["T1_US_FNAL_Disk"]

        files_rucio, sites = get_dataset_files(dataset, **sites_cfg, output="first")

        # print(dataset, sites)

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


# for version in ["v12"]:
# for version in ["v9", "v9_private", "v9_hh_private", "v11", "v11_private"]:
for version in ["v12"]:
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
