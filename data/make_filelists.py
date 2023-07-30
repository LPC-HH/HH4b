import json
import subprocess
from dbs.apis.dbsClient import DbsApi

dbs = DbsApi("https://cmsweb.cern.ch/dbs/prod/global/DBSReader")

qcd_bins = [
    "0to80",
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


def get_v10():
    return {
        "2022": {
            "QCD": {
                f"QCD_PT-{qbin}": f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv10-124X_mcRun3_2022_realistic_v12-v1/NANOAODSIM"
                for qbin in qcd_bins
            },
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
            "HHbb": {
                # TSG samples (100k events)
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00": "/GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00": "/GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00": "/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
            },
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
            "JetMET": {
                # replace ReReco with later?
                "Run2022C": "/JetMET/Run2022C-ReRecoNanoAODv11-v1/NANOAOD",
                "Run2022D": "/JetMET/Run2022D-ReRecoNanoAODv11-v1/NANOAOD",
            },
            "Muon": {},
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
                    f"QCD_HT-{qbin}": f"/QCDB-4Jets_HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM"
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
                # "Wto2Q-3Jets_HT-200to400": ,
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v3/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
            },
            "JetMET": {
                "Run2022E": "/JetMET/Run2022E-ReRecoNanoAODv11-v1/NANOAOD",
                "Run2022F": "/JetMET/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
                "Run2022G": "/JetMET/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
            },
            "Muon": {
                "Run2022F": "/Muon/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
                "Run2022G": "/Muon/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
            },
        },
    }


path_mkolosov = "/store/user/mkolosov/CRAB3_TransferData/PrivateNanoAOD/"


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
            },
            "MuonEG": {
                "Run2022B": f"{path_mkolosov}/MuonEG/MuonEG_Run2022B_PromptReco_v1_11June2023",
                "Run2022C": f"{path_mkolosov}/MuonEG/MuonEG_Run2022C_PromptReco_v1_11June2023",
                "Run2022D_v1": f"{path_mkolosov}/MuonEG/MuonEG_Run2022D_PromptReco_v1",
                "Run2022D_v2": f"{path_mkolosov}/MuonEG/MuonEG_Run2022D_PromptReco_v2",
                "Run2022D_v3": f"{path_mkolosov}/MuonEG/MuonEG_Run2022D_PromptReco_v3",
            },
            "Muon": {
                "Run2022B_single": f"{path_mkolosov}/SingleMuon/SingleMuon_Run2022B_PromptReco_v1_11June2023",
                "Run2022C_single": f"{path_mkolosov}/SingleMuon/SingleMuon_Run2022C_PromptReco_v1_11June2023",
                "Run2022C": f"{path_mkolosov}/Muon/Muon_Run2022C_PromptReco_v1_11June2023",
                "Run2022D_v1": f"{path_mkolosov}/Muon/Muon_Run2022D_PromptReco_v1_11June2023",
                "Run2022D_v2": f"{path_mkolosov}/Muon/Muon_Run2022D_PromptReco_v2_11June2023",
            },
            "TT": {
                "TTto2L2Nu": f"{path_mkolosov}/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
            },
        },
        "2022EE": {
            "JetMET": {
                "Run2022E": f"{path_mkolosov}/JetMET/JetMET_Run2022E_PromptReco_v1",
                "Run2022F": f"{path_mkolosov}/JetMET/JetMET_Run2022F_PromptReco_v1",
                "Run2022G": f"{path_mkolosov}/JetMET/JetMET_Run2022G_PromptReco_v1",
            },
            "MuonEG": {
                "Run2022E": f"{path_mkolosov}/MuonEG/MuonEG_Run2022E_PromptReco_v1",
                "Run2022F": f"{path_mkolosov}/MuonEG/MuonEG_Run2022F_PromptReco_v1",
                "Run2022G": f"{path_mkolosov}/MuonEG/MuonEG_Run2022G_PromptReco_v1",
            },
            "Muon": {
                "Run2022E": f"{path_mkolosov}/Muon/Muon_Run2022E_PromptReco_v1_11June2023",
                "Run2022F": f"{path_mkolosov}/Muon/Muon_Run2022F_PromptReco_v1_11June2023",
                "Run2022G": f"{path_mkolosov}/Muon/Muon_Run2022G_PromptReco_v1_11June2023",
            },
            "TT": {
                "TTto2L2Nu": f"{path_mkolosov}/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_Run3Summer22MiniAODv3_preEE",
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
    dirlook = (
        subprocess.check_output(f"eos {eosbase} ls {startdir}", shell=True)
        .decode("utf-8")
        .split("\n")[:-1]
    )
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


for version in ["v10", "v11", "v11_private", "v12"]:
    datasets = globals()[f"get_{version}"]()
    index = datasets.copy()
    for year, ydict in datasets.items():
        for sample, sdict in ydict.items():
            for sname, dataset in sdict.items():
                if "private" in version:
                    files = eos_rec_search(dataset, ".root", [])
                else:
                    files = [ifile["logical_file_name"] for ifile in dbs.listFiles(dataset=dataset)]
                index[year][sample][sname] = files

    with open(f"nanoindex_{version}.json", "w") as f:
        json.dump(index, f, indent=4)
