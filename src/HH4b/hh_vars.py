"""
Common variables

Authors: Raghav Kansal, Cristina Suarez
"""

from __future__ import annotations

from collections import OrderedDict

years = ["2022", "2022EE", "2023", "2023BPix", "2024"]

# in pb^-1
LUMI = {
    "2022": 7980.5,
    "2022EE": 26671.6,
    "2022All": 34652.1,
    "2023": 18084.4,
    "2023BPix": 9692.1,
    "2023All": 27776.5,
    "2024": 1.0,
    "2022-2023": 62428.6,
    "2024": 108960.0,
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
    # "gghtobb": ["GluGluHto2B_PT-200_M-125"],
    # "vbfhtobb": ["VBFHto2B_M-125"],
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
    "zz": ["ZZ"],
    "nozzdiboson": ["WW", "WZ"],
    "vjets": ["Wto2Q-2Jets_PTQQ", "Zto2Q-2Jets_PTQQ"],
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
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": [
            "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8"
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
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": [
            "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8"
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
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": [
            "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8"
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
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": [
            "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
        "vbfhh4b-kvm2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
    },
    "2024": {},
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
    "2024": {
        
            "data": [
                "JetMET_Run2024B",
                # "JetMET_Run2024C",
                # "JetMET_Run2024D",
                # "JetMET_Run2024E",
                # "JetMET_Run2024F",
                # "JetMET_Run2024G",
                # "JetMET_Run2024H",
                # "JetMET_Run2024I",
            ]
        ,
        **samples_run3_sig["2024"],
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

sig_keys_ggf = [
    "hh4b",
    "hh4b-kl0",
    "hh4b-kl2p45",
    "hh4b-kl5",
]
sig_keys_vbf = [
    "vbfhh4b",
    "vbfhh4b-k2v0",
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

# bkg keys that require running up/down systematics
syst_keys = sig_keys + bg_keys
syst_keys.remove("qcd")

norm_preserving_weights = ["genweight", "pileup", "ISRPartonShower", "FSRPartonShower"]

jecs = {
    "JES": "JES",
    "JER": "JER",
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
    "bbFatJetParTmassVis",
    "bdt_score",
    "bdt_score_vbf",
    "HHmass",
    "H1Pt_HHmass",
    "H2Pt_HHmass",
    "H1Mass",
    "H2Mass",
    "H1PNetMass",
    "H2PNetMass",
    "Category",
]

jmsr_values = {}
jmsr_values["bbFatJetPNetMassLegacy"] = {}
jmsr_values["bbFatJetPNetMassLegacy"]["JMR"] = {
    "2022": {"nom": 1.13, "down": 1.06, "up": 1.20},
    "2022EE": {"nom": 1.20, "down": 1.15, "up": 1.25},
    "2023": {"nom": 1.20, "down": 1.16, "up": 1.24},
    "2023BPix": {"nom": 1.16, "down": 1.09, "up": 1.23},
    "2024": {"nom": 1.0, "down": 1.0, "up": 1.0},  # placeholder for future
}
jmsr_values["bbFatJetPNetMassLegacy"]["JMS"] = {
    "2022": {"nom": 1.015, "down": 1.010, "up": 1.020},
    "2022EE": {"nom": 1.021, "down": 1.018, "up": 1.024},
    "2023": {"nom": 0.999, "down": 0.996, "up": 1.003},
    "2023BPix": {"nom": 0.974, "down": 0.970, "up": 0.980},
    "2024": {"nom": 1.0, "down": 1.0, "up": 1.0},  # placeholder for future
}
jmsr_values["bbFatJetParTmassVis"] = {}
# numbers from template-morphing fit
# jmsr_values["bbFatJetParTmassVis"]["JMR"] = {
#     "2022": {"nom": 1.14, "down": 1.12, "up": 1.16},
#     "2022EE": {"nom": 1.14, "down": 1.12, "up": 1.16},
#     "2023": {"nom": 1.095, "down": 1.052, "up": 1.139},
#     "2023BPix": {"nom": 1.095, "down": 1.052, "up": 1.139},
# }
# jmsr_values["bbFatJetParTmassVis"]["JMS"] = {
#     "2022": {"nom": 1.015, "down": 1.0102, "up": 1.020},
#     "2022EE": {"nom": 1.015, "down": 1.0102, "up": 1.020},
#     "2023": {"nom": 0.974, "down": 0.967, "up": 0.981},
#     "2023BPix": {"nom": 0.974, "down": 0.967, "up": 0.981},
# }
# numbers from weighted template fit
jmsr_values["bbFatJetParTmassVis"]["JMR"] = {
    "2022": {"nom": 1.0354, "down": 1.028, "up": 1.042},
    "2022EE": {"nom": 1.0354, "down": 1.028, "up": 1.042},
    "2023": {"nom": 1.0335, "down": 1.025, "up": 1.042},
    "2023BPix": {"nom": 1.0335, "down": 1.025, "up": 1.042},
    "2024": {"nom": 1.0, "down": 1.0, "up": 1.0},  # placeholder for future
}
jmsr_values["bbFatJetParTmassVis"]["JMS"] = {
    "2022": {"nom": 1.011, "down": 1.007, "up": 1.014},
    "2022EE": {"nom": 1.011, "down": 1.007, "up": 1.014},
    "2023": {"nom": 0.9867, "down": 0.983, "up": 0.9903},
    "2023BPix": {"nom": 0.9867, "down": 0.983, "up": 0.9903},
    "2024": {"nom": 1.0, "down": 1.0, "up": 1.0},  # placeholder for future
}
jmsr_keys = sig_keys + ["vhtobb", "zz", "nozzdiboson"]
jmsr_res = {}
jmsr_res["bbFatJetPNetMassLegacy"] = dict.fromkeys(sig_keys, 14.4)
jmsr_res["bbFatJetPNetMassLegacy"]["vhtobb"] = 14.4 * 80.0 / 125.0
jmsr_res["bbFatJetPNetMassLegacy"]["zz"] = 14.4 * 80.0 / 125.0
jmsr_res["bbFatJetPNetMassLegacy"]["nozzdiboson"] = 14.4 * 80.0 / 125.0
jmsr_res["bbFatJetParTmassVis"] = dict.fromkeys(sig_keys, 10.7)
jmsr_res["bbFatJetParTmassVis"]["vhtobb"] = 10.7 * 80.0 / 125.0
jmsr_res["bbFatJetParTmassVis"]["zz"] = 10.7 * 80.0 / 125.0
jmsr_res["bbFatJetParTmassVis"]["nozzdiboson"] = 10.7 * 80.0 / 125.0

ttbarsfs_decorr_txbb_bins = {}
ttbarsfs_decorr_txbb_bins["pnet-legacy"] = [0, 0.8, 0.94, 0.99, 1]
ttbarsfs_decorr_txbb_bins["glopart-v2"] = [0, 0.31, 0.7, 0.8, 0.87, 0.92, 0.96, 1]
ttbarsfs_decorr_ggfbdt_bins = {}
ttbarsfs_decorr_ggfbdt_bins["24May31_lr_0p02_md_8_AK4Away"] = [0.03, 0.3, 0.5, 0.7, 0.93, 1.0]
ttbarsfs_decorr_ggfbdt_bins["24Nov7_v5_glopartv2_rawmass"] = [0.03, 0.6375, 0.9075, 1.0]
ttbarsfs_decorr_ggfbdt_bins["25Feb5_v13_glopartv2_rawmass"] = [0.03, 0.755, 0.94, 1.0]
ttbarsfs_decorr_vbfbdt_bins = {}
ttbarsfs_decorr_vbfbdt_bins["24Nov7_v5_glopartv2_rawmass"] = [0.975, 1]
ttbarsfs_decorr_vbfbdt_bins["25Feb5_v13_glopartv2_rawmass"] = [0.9667, 1.0]


txbbsfs_decorr_txbb_wps = {}
txbbsfs_decorr_txbb_wps["pnet-legacy"] = OrderedDict(
    [
        ("WP6", [0.92, 0.95]),
        ("WP5", [0.95, 0.975]),
        ("WP4", [0.975, 0.99]),
        ("WP3", [0.99, 0.995]),
        ("WP2", [0.995, 0.998]),
        ("WP1", [0.998, 1]),
    ]
)
txbbsfs_decorr_txbb_wps["glopart-v2"] = OrderedDict(
    [
        ("WP5", [0.8, 0.9]),
        ("WP4", [0.9, 0.94]),
        ("WP3", [0.94, 0.97]),
        ("WP2", [0.97, 0.99]),
        ("WP1", [0.99, 1]),
    ]
)

txbbsfs_decorr_pt_bins = {}
txbbsfs_decorr_pt_bins["pnet-legacy"] = OrderedDict(
    [
        ("WP6", [200, 250, 300, 400, 500, 100000]),
        ("WP5", [200, 250, 300, 400, 500, 100000]),
        ("WP4", [200, 250, 300, 400, 500, 100000]),
        ("WP3", [200, 250, 300, 400, 500, 100000]),
        ("WP2", [200, 400, 100000]),
        ("WP1", [200, 400, 100000]),
    ]
)
txbbsfs_decorr_pt_bins["glopart-v2"] = OrderedDict(
    [
        ("WP5", [200, 400, 100000]),
        ("WP4", [200, 400, 100000]),
        ("WP3", [200, 400, 100000]),
        ("WP2", [200, 400, 100000]),
        ("WP1", [200, 400, 100000]),
    ]
)

txbb_strings = {
    "pnet-legacy": "bbFatJetPNetTXbbLegacy",
    "pnet-v12": "bbFatJetPNetTXbb",
    "glopart-v2": "bbFatJetParTTXbb",
}

mreg_strings = {
    "pnet-legacy": "bbFatJetPNetMassLegacy",
    "pnet-v12": "bbFatJetPNetMass",
    "glopart-v2": "bbFatJetParTmassVis",
}
