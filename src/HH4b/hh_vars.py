"""
Common variables

Authors: Raghav Kansal, Cristina Suarez
"""

from __future__ import annotations

from collections import OrderedDict

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
    # "singletop": [
    #     "TbarBQ_t-channel_4FS",
    #     "TBbarQ_t-channel_4FS",
    #     "TWminustoLNu2Q",
    #     "TWminusto4Q",
    #     "TbarWplustoLNu2Q",
    #     "TbarWplusto4Q",
    # ],
    "vhtobb": [
        "WplusH_Hto2B_Wto2Q_M-125",
        "WminusH_Hto2B_Wto2Q_M-125",
        "ZH_Hto2B_Zto2Q_M-125",
        "ggZH_Hto2B_Zto2Q_M-125",
        # "WplusH_Hto2B_WtoLNu_M-125",
        # "WminusH_Hto2B_WtoLNu_M-125",
        # "ggZH_Hto2B_Zto2L_M-125",
        # "ggZH_Hto2B_Zto2Nu_M-125",
    ],
    "novhhtobb": ["GluGluHto2B_PT-200_M-125", "VBFHto2B_M-125_dipoleRecoilOn"],
    "tthtobb": ["ttHto2B_M-125"],
    "diboson": ["ZZ", "WW", "WZ"],
    "vjets": ["Wto2Q-3Jets_HT", "Zto2Q-4Jets_HT"],
}

common_samples_sig = {}

samples_run3_sig = {
    "2022": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-k2v2": ["VBFHHto4B_CV-1_C2V-2_C3-1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-kl2": ["VBFHHto4B_CV-1_C2V-1_C3-2_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-kv1p74-k2v1p37-kl14p4": [
            "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p012-k2v0p03-kl10p2": [
            "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p758-k2v1p44-klm19p3": [
            "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p962-k2v0p959-klm1p43": [
            "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm1p21-k2v1p94-klm0p94": [
            "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm1p6-k2v2p72-klm1p36": [
            "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
    },
    "2022EE": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-k2v2": ["VBFHHto4B_CV-1_C2V-2_C3-1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-kl2": ["VBFHHto4B_CV-1_C2V-1_C3-2_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-kv1p74-k2v1p37-kl14p4": [
            "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p012-k2v0p03-kl10p2": [
            "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p758-k2v1p44-klm19p3": [
            "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p962-k2v0p959-klm1p43": [
            "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm1p21-k2v1p94-klm0p94": [
            "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm1p6-k2v2p72-klm1p36": [
            "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
    },
    "2023": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-kv1p74-k2v1p37-kl14p4": [
            "VBFHHto4B_CV_1p74_C2V_1p37_C3_14p4_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p962-k2v0p959-klm1p43": [
            "VBFHHto4B_CV_m0p962_C2V_0p959_C3_m1p43_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": [
            "VBFHHto4B_CV_m1p83_C2V_3p57_C3_m3p39_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV_m2p12_C2V_3p87_C3_m5p96_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
    },
    "2023BPix": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"],
        "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
        "vbfhh4b-kv1p74-k2v1p37-kl14p4": [
            "VBFHHto4B_CV_1p74_C2V_1p37_C3_14p4_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p012-k2v0p03-kl10p2": [
            "VBFHHto4B_CV_m0p012_C2V_0p030_C3_10p2_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p758-k2v1p44-klm19p3": [
            "VBFHHto4B_CV_m0p758_C2V_1p44_C3_m19p3_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm0p962-k2v0p959-klm1p43": [
            "VBFHHto4B_CV_m0p962_C2V_0p959_C3_m1p43_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm1p21-k2v1p94-klm0p94": [
            "VBFHHto4B_CV_m1p21_C2V_1p94_C3_m0p94_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm1p6-k2v2p72-klm1p36": [
            "VBFHHto4B_CV_m1p60_C2V_2p72_C3_m1p36_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": [
            "VBFHHto4B_CV_m1p83_C2V_3p57_C3_m3p39_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV_m2p12_C2V_3p87_C3_m5p96_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
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
hbb_bg_keys = ["gghtobb", "vbfhtobb", "vhtobb", "tthtobb", "novhhtobb"]

sig_keys_ggf = ["hh4b", "hh4b-kl0", "hh4b-kl2p45", "hh4b-kl5"]
sig_keys_vbf = [
    "vbfhh4b",
    "vbfhh4b-k2v0",
    "vbfhh4b-k2v2",
    "vbfhh4b-kl2",
    "vbfhh4b-kv1p74-k2v1p37-kl14p4",
    "vbfhh4b-kvm0p012-k2v0p03-kl10p2",
    "vbfhh4b-kvm0p758-k2v1p44-klm19p3",
    "vbfhh4b-kvm0p962-k2v0p959-klm1p43",
    "vbfhh4b-kvm1p21-k2v1p94-klm0p94",
    "vbfhh4b-kvm1p6-k2v2p72-klm1p36",
    "vbfhh4b-kvm1p83-k2v3p57-klm3p39",
    "vbfhh4b-kvm2p12-k2v3p87-klm5p96",
]
sig_keys = sig_keys_ggf + sig_keys_vbf

# keys that require running up/down systematics
syst_keys = sig_keys + bg_keys
syst_keys.remove("qcd")

norm_preserving_weights = ["genweight", "pileup", "ISRPartonShower", "FSRPartonShower"]

jecs = {
    # "JES": "JES",
    "JER": "JER",
    # #####
    # # including all sources
    # #####
    # "JES_AbsoluteMPFBias": "JES_AbsoluteMPFBias",
    "JES_AbsoluteScale": "JES_AbsoluteScale",
    # "JES_AbsoluteStat": "JES_AbsoluteStat",
    # "JES_FlavorQCD": "JES_FlavorQCD",
    # "JES_Fragmentation": "JES_Fragmentation",
    # "JES_PileUpDataMC": "JES_PileUpDataMC",
    # "JES_PileUpPtBB": "JES_PileUpPtBB",
    # "JES_PileUpPtEC1": "JES_PileUpPtEC1",
    # "JES_PileUpPtEC2": "JES_PileUpPtEC2",
    # "JES_PileUpPtHF": "JES_PileUpPtHF",
    # "JES_PileUpPtRef": "JES_PileUpPtRef",
    # "JES_RelativeFSR": "JES_RelativeFSR",
    # "JES_RelativeJEREC1": "JES_RelativeJEREC1",
    # "JES_RelativeJEREC2": "JES_RelativeJEREC2",
    # "JES_RelativeJERHF": "JES_RelativeJERHF",
    # "JES_RelativePtBB": "JES_RelativePtBB",
    # "JES_RelativePtEC1": "JES_RelativePtEC1",
    # "JES_RelativePtEC2": "JES_RelativePtEC2",
    # "JES_RelativePtHF": "JES_RelativePtHF",
    # "JES_RelativeBal": "JES_RelativeBal",
    # "JES_RelativeSample": "JES_RelativeSample",
    # "JES_RelativeStatEC": "JES_RelativeStatEC",
    # "JES_RelativeStatFSR": "JES_RelativeStatFSR",
    # "JES_RelativeStatHF": "JES_RelativeStatHF",
    # "JES_SinglePionHCAL": "JES_SinglePionHCAL",
    # "JES_SinglePionECAL": "JES_SinglePionECAL",
    # "JES_TimePtEta": "JES_TimePtEta",
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
    "Category",
]

# variables affected by JMS/JMR
jmsr_vars = [
    "bbFatJetPNetMassLegacy",
    "bdt_score",
    "bdt_score_vbf",
    "HHmass",
    "H1Pt_HHmass",
    "H2Pt_HHmass",
    "Category",
]

jmsr_values = {}
jmsr_values["JMR"] = {
    "2022": {"nom": 1.13, "down": 1.06, "up": 1.20},
    "2022EE": {"nom": 1.20, "down": 1.15, "up": 1.25},
    "2023": {"nom": 1.20, "down": 1.16, "up": 1.24},
    "2023BPix": {"nom": 1.16, "down": 1.09, "up": 1.23},
}
jmsr_values["JMS"] = {
    "2022": {"nom": 1.015, "down": 1.010, "up": 1.020},
    "2022EE": {"nom": 1.021, "down": 1.018, "up": 1.024},
    "2023": {"nom": 0.999, "down": 0.996, "up": 1.003},
    "2023BPix": {"nom": 0.974, "down": 0.970, "up": 0.980},
}
jmsr_keys = sig_keys + ["vhtobb", "diboson"]
jmsr_res = {sig_key: 14.4 for sig_key in sig_keys}
jmsr_res["vhtobb"] = 14.4 * 80. / 125.
jmsr_res["diboson"] = 14.4 * 80. / 125.


ttbarsfs_decorr_txbb_bins = [0, 0.8, 0.94, 0.99, 1]
ttbarsfs_decorr_bdt_bins = [0.03, 0.3, 0.5, 0.7, 0.93, 1.0]

txbbsfs_decorr_txbb_wps = OrderedDict(
    [("WP3", [0.92, 0.95]), ("WP2", [0.95, 0.975]), ("WP1", [0.975, 1])]
)
txbbsfs_decorr_pt_bins = [250, 300, 400, 500, 100000]
