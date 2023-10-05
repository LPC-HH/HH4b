import json
import subprocess
import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

import os

os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/x86_64/rhel7/py3/current"

qcd_bins = [
    # "0to80",
    "80to120",
    "15to30",  # unclear if these are needed
    "30to50",
    "50to80",
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
    "40to100",
    "100to200",
    "200to400",
    "400to600",
    "600to800",
    "800to1000",
    "1000to1500",
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
                # ~ 100k events
                "GluGlutoHHto4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "GluGlutoHHto4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "GluGlutoHHto4B_cHHH2p45_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/GluGluToHHTo4B_cHHH2p45_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
                "GluGlutoHHto4B_cHHH5_TuneCP5_PSWeights_13TeV-powheg-pythia8": "/GluGluToHHTo4B_cHHH5_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
            },
            "QCD": {
                **{
                    f"QCD_HT-{qbin}": f"/QCD_HT{qbin}_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"
                    for qbin in qcd_ht_bins_run2
                },
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
                    f"QCD_PT-{qbin}": f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2_ext1-v1/NANOAODSIM"
                    for qbin in qcd_bins
                },
                **{
                    f"QCD_HT-{qbin}": f"/QCDB-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
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
            },
            # ReReco should be 27Jun2023...
            # "JetMET": {
            #     "Run2022C_single":
            #     "Run2022C":
            #     "Run2022D":
            # },
            # "Muon": {
            #     "Run2022C_single":
            #     "Run2022C":
            #     "Run2022D":
            # },
        },
        # MC campaign post EE+ leak
        "2022EE": {
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
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
                    f"QCD_PT-{qbin}": f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1_ext1-v1/NANOAODSIM"
                    for qbin in qcd_bins
                },
                **{
                    f"QCDB_HT-{qbin}": f"/QCDB-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
            },
            "TT": {
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            },
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "WWto4Q": "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
                "WZto4Q-1Jets-4FS": "/WZto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                # missing ZZto4B
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
            },
            "VJets": {
                # why is this marked as invalid?
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                # why is this marked as invalid?
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v3/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            },
            # Prompt-Reco
            "JetMET": {
                # "Run2022E": This needs to be re-reco
                "Run2022F": "/JetMET/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
                "Run2022G": "/JetMET/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
            },
            "Muon": {
                # "Run2022E": This needs to be re-reco
                "Run2022F": "/Muon/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
                "Run2022G": "/Muon/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
            },
        },
        "2023": {
            "JetMET": {
                "Run2023B-v1_0": "/JetMET0/Run2023B-PromptNanoAODv11p9_v1-v1/NANOAOD",
                "Run2023C-v1_0": "/JetMET0/Run2023C-PromptMiniAOD_v1_NanoAODv12-v1/NANOAOD",
                "Run2023C-v2_0": "/JetMET0/Run2023C-PromptMiniAOD_v2_NanoAODv12-v1/NANOAOD",
                "Run2023C-v3_0": "/JetMET0/Run2023C-PromptMiniAOD_v3_NanoAODv12-v1/NANOAOD",
                "Run2023C-v4_0": "/JetMET0/Run2023C-PromptMiniAOD_v4_NanoAODv12-v1/NANOAOD",
                "Run2023D-v1_0": "/JetMET0/Run2023D-PromptReco-v1/NANOAOD",
                "Run2023D-v2_0": "/JetMET0/Run2023D-PromptReco-v2/NANOAOD",
                "Run2023B-v1_1": "/JetMET1/Run2023B-PromptNanoAODv11p9_v1-v1/NANOAOD",
                "Run2023C-v1_1": "/JetMET1/Run2023C-PromptMiniAOD_v1_NanoAODv12-v1/NANOAOD",
                "Run2023C-v2_1": "/JetMET1/Run2023C-PromptMiniAOD_v2_NanoAODv12-v1/NANOAOD",
                "Run2023C-v3_1": "/JetMET1/Run2023C-PromptMiniAOD_v3_NanoAODv12-v1/NANOAOD",
                "Run2023C-v4_1": "/JetMET1/Run2023C-PromptMiniAOD_v4_NanoAODv12-v1/NANOAOD",
                "Run2023D-v1_1": "/JetMET1/Run2023D-PromptReco-v1/NANOAOD",
                "Run2023D-v2_1": "/JetMET1/Run2023D-PromptReco-v2/NANOAOD",
            },
            "Muon": {
                "Run2023B-v1_0": "/Muon0/Run2023B-PromptNanoAODv11p9_v1-v2/NANOAOD",
                "Run2023C-v1_0": "/Muon0/Run2023C-PromptMiniAOD_v1_NanoAODv12-v1/NANOAOD",
                "Run2023C-v2_0": "/Muon0/Run2023C-PromptMiniAOD_v2_NanoAODv12-v1/NANOAOD",
                "Run2023C-v3_0": "/Muon0/Run2023C-PromptMiniAOD_v2_NanoAODv12-v1/NANOAOD",
                "Run2023C-v4_0": "/Muon0/Run2023C-PromptMiniAOD_v3_NanoAODv12-v1/NANOAOD",
                "Run2023D-v1_0": "/Muon0/Run2023D-PromptReco-v1/NANOAOD",
                "Run2023D-v2_0": "/Muon0/Run2023D-PromptReco-v2/NANOAOD",
                "Run2023B-v1_1": "/Muon1/Run2023B-PromptNanoAODv11p9_v1-v2/NANOAOD",
                "Run2023C-v1_1": "/Muon1/Run2023C-PromptMiniAOD_v1_NanoAODv12-v1/NANOAOD",
                "Run2023C-v2_1": "/Muon1/Run2023C-PromptMiniAOD_v2_NanoAODv12-v1/NANOAOD",
                "Run2023C-v3_1": "/Muon1/Run2023C-PromptMiniAOD_v3_NanoAODv12-v1/NANOAOD",
                "Run2023C-v4_1": "/Muon1/Run2023C-PromptReco-v4/NANOAOD",
                "Run2023D-v1_1": "/Muon1/Run2023D-PromptReco-v1/NANOAOD",
                "Run2023D-v2_1": "/Muon1/Run2023D-PromptReco-v2/NANOAOD",
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


def get_v12():
    return {
        "2023": {
            "JetMET": {
                "Run2023C_0_v2": "/JetMET0/Run2023C-PromptNanoAODv12_v2-v2/NANOAOD",
                "Run2023C_0_v3": "/JetMET0/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                "Run2023C_0_v4": "/JetMET0/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                "Run2023C_1_v2": "/JetMET1/Run2023C-PromptNanoAODv12_v2-v4/NANOAOD",
                "Run2023C_1_v3": "/JetMET1/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                "Run2023C_1_v4": "/JetMET1/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
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
            donedirs[di].append(f"root://cmsxrootd-site.fnal.gov/{startdir}/{d}")
        elif d == "log":
            continue
        else:
            donedirs[di] = donedirs[di] + eos_rec_search(
                startdir + "/" + d, suffix, dirs + donedirs[di]
            )
    donedir = [d for da in donedirs for d in da]
    return dirs + donedir


for version in ["v9", "v10", "v11", "v11_private"]:
    datasets = globals()[f"get_{version}"]()
    index = datasets.copy()
    for year, ydict in datasets.items():
        for sample, sdict in ydict.items():
            for sname, dataset in sdict.items():
                if "private" in version:
                    files = eos_rec_search(dataset, ".root", [])
                else:
                    from rucio_utils import get_proxy_path
                    from rucio_utils import get_dataset_files
                    import requests

                    proxy = get_proxy_path()
                    link = f"https://cmsweb.cern.ch:8443/dbs/prod/global/DBSReader/files?dataset={dataset}&detail=True"
                    r = requests.get(
                        link,
                        cert=proxy,
                        verify=False,
                    )
                    filesjson = r.json()
                    files = []
                    for fj in filesjson:
                        if fj["is_file_valid"] == 0:
                            print(f"ERROR: File not valid on DAS: {fj['logical_file_name']}")
                        else:
                            files.append(fj['logical_file_name'])
                            #self.metadata["nevents"] += fj['event_count']
                            #self.metadata["size"] += fj['file_size']
                    # Now query rucio to get the concrete dataset passing the sites filtering options
                    sites_cfg = {
                        "whitelist_sites": None,
                        "blacklist_sites": None,
                        "regex_sites": None,
                    }
                    files_rucio, sites = get_dataset_files(
                        dataset, **sites_cfg, output="first"
                    )
                    index[year][sample][sname] = files_rucio

    with open(f"nanoindex_{version}.json", "w") as f:
        json.dump(index, f, indent=4)
