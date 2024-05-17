"""
Common variables

Authors: Raghav Kansal, Cristina Suarez
"""

from __future__ import annotations

years = ["2022", "2022EE", "2023", "2023BPix"]

# in pb^-1
LUMI = {
    "2022": 7971.4,
    "2022EE": 26337.0,
    "2022All": 34308.0,
    "2023": 17650.0,
    "2023BPix": 9451.0,
    "2023All": 27101.0,
    "2022-2023": 61409.0,
    "2018": 59830.0,
    "2017": 41480.0,
    "2016": 36330.0,
    "Run2": 137640.0,
}


DATA_SAMPLES = ["JetMET", "Muon", "EGamma"]

# sample key -> list of samples or selectors
common_samples_bg = {
    "qcd": ["QCD_HT"],
    "data": [f"{key}_Run" for key in DATA_SAMPLES],
    "ttbar": ["TTto4Q", "TTto2L2Nu", "TTtoLNu2Q"],
    "gghtobb": ["GluGluHto2B_PT-200_M-125"],
    "vbfhtobb": ["VBFHto2B_M-125_dipoleRecoilOn"],
    # TODO: Add single top!
    "vhtobb": [
        "WplusH_Hto2B_Wto2Q_M-125",
        # "WplusH_Hto2B_WtoLNu_M-125",
        "WminusH_Hto2B_Wto2Q_M-125",
        # "WminusH_Hto2B_WtoLNu_M-125",
        "ZH_Hto2B_Zto2Q_M-125",
        "ggZH_Hto2B_Zto2Q_M-125",
        # "ggZH_Hto2B_Zto2L_M-125",
        # "ggZH_Hto2B_Zto2Nu_M-125",
    ],
    "novhhtobb": ["GluGluHto2B_PT-200_M-125", "VBFHto2B_M-125_dipoleRecoilOn"],
    "tthtobb": ["ttHto2B_M-125"],
    "diboson": ["ZZ", "WW", "WZ"],
    # "vjetslnu": ["WtoLNu-4Jets"],  # TODO: didn't run on these?
    "vjets": ["Wto2Q-3Jets_HT", "Zto2Q-4Jets_HT"],
}

common_samples_sig = {}  # TODO: none yet

samples_run3_sig = {
    "2022": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
    },
    "2022EE": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
    },
    "2023": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
    },
    "2023BPix": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        # "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG"],
    },
}

samples_run3 = {
    "2022": {
        **common_samples_bg,
        **samples_run3_sig["2022"],
    },
    "2022EE": {
        **common_samples_bg,
        **samples_run3_sig["2022EE"],
    },
    "2023": {
        **common_samples_bg,
        **samples_run3_sig["2023"],
    },
    "2023BPix": {
        **common_samples_bg,
        **samples_run3_sig["2023BPix"],
    },
}

samples_2018 = {
    "hh4b": [
        "GluGlutoHHto4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8",
    ],
    "qcd": [
        "QCD_HT-1000to1500-13TeV",
        "QCD_HT-1500to2000-13TeV",
        "QCD_HT-2000toInf-13TeV",
        "QCD_HT-200to300-13TeV",
        "QCD_HT-300to500-13TeV",
        "QCD_HT-500to700-13TeV",
        "QCD_HT-700to1000-13TeV",
    ],
    "data": [
        "Run2018A",
        "Run2018B",
        "Run2018C",
        "Run2018D",
    ],
    "ttbar": [
        "TTTo2L2Nu_13TeV",
        "TTToHadronic_13TeV",
        "TTToSemiLeptonic_13TeV",
    ],
    "vjetslnu": [
        "WtoLNu-4Jets",
    ],
    "vjets": [
        "WJetsToQQ_HT-200to400_13TeV",
        "WJetsToQQ_HT-400to600_13TeV",
        "WJetsToQQ_HT-600to800_13TeV",
        "WJetsToQQ_HT-800toInf_13TeV",
        "ZJetsToQQ_HT-200to400_13TeV",
        "ZJetsToQQ_HT-400to600_13TeV",
        "ZJetsToQQ_HT-600to800_13TeV",
        "ZJetsToQQ_HT-800toInf_13TeV",
    ],
    "diboson": [
        "ZZTo4B01j_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    ],
    # "diboson": [
    #    "ZZ_TuneCP5_13TeV-pythia8"
    # ],
    "gghtobb": [
        "GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8",
    ],
    # "gghtobb": [
    #    "GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8",
    # ],
    "vbfhtobb": ["VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8"],
    "vhtobb": [
        "WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8",
        "WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8",
        "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8",
    ],
    "tthtobb": ["ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8"],
}

samples = {
    "2018": samples_2018,
    **samples_run3,
}

data_key = "data"
qcd_key = "qcd"

bg_keys = list(common_samples_bg.keys())
bg_keys.remove("data")
bg_keys = ["qcd"]  # for testing just use qcd
hbb_bg_keys = ["gghtobb", "vbfhtobb", "vhtobb", "tthtobb", "novhhtobb"]

sig_keys_ggf = ["hh4b", "hh4b-kl0", "hh4b-kl2p45", "hh4b-kl5"]
sig_keys_vbf = ["vbfhh4b", "vbfhh4b-k2v0"]  # TODO
sig_keys = sig_keys_ggf + sig_keys_vbf

# keys that require running up/down systematics
syst_keys = sig_keys

norm_preserving_weights = ["genweight", "pileup", "ISRPartonShower", "FSRPartonShower"]

jecs = {
    "JES": "JES",
    # "JER": "JER",
    #####
    # including reduced sources
    #####
    # "JES_Abs": "JES_AbsoluteMPFBias", # goes in Absolute
    # # "JES_AbsoluteScale": "JES_AbsoluteScale", # goes in Absolute
    # "JES_Abs_year": "JES_AbsoluteStat",  # goes in Abs_year
    # "JES_FlavorQCD": "JES_FlavorQCD",
    # # "JES_Fragmentation": "JES_Fragmentation", # goes in Absolute
    # # "JES_PileUpDataMC": "JES_PileUpDataMC", # goes in Absolute
    # "JES_BBEC1": "JES_PileUpPtBB", # goes in BBEC1
    # # "JES_PileUpPtEC1": "JES_PileUpPtEC1", # goes in BBEC1
    # "JES_EC2": "JES_PileUpPtEC2",
    # "JES_HF": "JES_PileUpPtHF",
    # # "JES_PileUpPtRef": "JES_PileUpPtRef", # goes in Absolute
    # # "JES_RelativeFSR": "JES_RelativeFSR", # goes in Absolute
    # "JES_BBEC1_year": "JES_RelativeJEREC1", # goes in BBEC1_year
    # "JES_EC2_year": "JES_RelativeJEREC2", # goes in EC2_year
    # # "JES_RelativeJERHF": "JES_RelativeJERHF", # goes in HF
    # # "JES_RelativePtBB": "JES_RelativePtBB", # goes in BBEC1
    # # "JES_RelativePtEC1": "RelativePtEC1", # goes in BBEC1_year
    # # "JES_RelativePtEC2": "JES_RelativePtEC2": # goes in EC2_year
    # # "JES_RelativePtHF": "JES_RelativePtHF" # goes in HF
    # "JES_RelativeBal": "JES_RelativeBal",
    # "JES_RelativeSample_year": "JES_RelativeSample",
    # # "JEs_RelativeStatEC": "JES_RelativeStatEC", # goes in BBEC1_year
    # # "JES_RelativeStatFSR": "JES_RelativeStatFSR", # goes in Abs_year
    # "JES_HF_year": "JES_RelativeStatHF",
    # # "JES_SinglePionHCAL": "JES_SinglePionHCAL", # goes in Absolute
    # # "JES_SinglePionECAL": "JES_SinglePionECAL", # goes in Absolute
    # # "JES_TimePtEta": "JES_TimePtEta", # goes in Abs_year
}

jec_shifts = []
for key in jecs:
    for shift in ["up", "down"]:
        jec_shifts.append(f"{key}_{shift}")

jmsr = {
    "JMS": "JMS",
    "JMR": "JMR",
}

jmsr_shifts = []
for key in jmsr:
    for shift in ["up", "down"]:
        jmsr_shifts.append(f"{key}_{shift}")

# variables affected by JECs
jec_vars = [
    "bbFatJetPt",
    "VBFJetPt",
    "bdt_score",
    "bdt_score_vbf",
    "HHPt",
    "HHeta",
    "HHmass",
    "H1Pt",
    "H2Pt",
    "H1Pt_HHmass",
    "H2Pt_HHmass",
    "H1Pt/H2Pt",
    "VBFjjMass",
    "VBFjjDeltaEta",
]

# variables affected by JMS/JMR
jmsr_vars = []  # TODO
