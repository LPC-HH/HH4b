# in pb^-1
LUMI = {"2022EE": 20700}

# sample key -> list of samples
samples = {
    "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG"],
    "qcd": [
        "QCD_PT-120to170",
        "QCD_PT-170to300",
        "QCD_PT-470to600",
        "QCD_PT-600to800",
        "QCD_PT-800to1000",
        "QCD_PT-1000to1400",
        "QCD_PT-1400to1800",
        "QCD_PT-1800to2400",
        "QCD_PT-2400to3200",
        "QCD_PT-3200",
    ],
    "data": [
        "Run2022F",
        "Run2022G",
    ],
    "ttbar": [
        "TTto2L2Nu",
        "TTtoLNu2Q",
        "TTto4Q",
    ],
    "gghtobb": [
        "GluGluHto2B_PT-200_M-125",
    ],
    "vbfhtobb": [
        "VBFHto2B_M-125_dipoleRecoilOn",
    ],
    "vhtobb": [
        "WplusH_Hto2B_Wto2Q_M-125",
        "WplusH_Hto2B_WtoLNu_M-125",
        "WminusH_Hto2B_Wto2Q_M-125",
        "WminusH_Hto2B_WtoLNu_M-125",
        "ZH_Hto2B_Zto2Q_M-125",
        "ggZH_Hto2B_Zto2Q_M-125",
        "ggZH_Hto2B_Zto2L_M-125",
        "ggZH_Hto2B_Zto2Nu_M-125",
    ],
    "tthtobb": [
        "ttHto2B_M-125",
    ],
    "diboson": [
        "ZZ",
        "WW",
        "WZ",
    ],
    "vjets": [
        "Wto2Q-3Jets_HT-200to400",
        "Wto2Q-3Jets_HT-400to600",
        "Wto2Q-3Jets_HT-600to800",
        "Wto2Q-3Jets_HT-800",
        "Zto2Q-4Jets_HT-200to400",
        "Zto2Q-4Jets_HT-400to600",
        "Zto2Q-4Jets_HT-600to800",
        "Zto2Q-4Jets_HT-800",
    ],
}

data_key = "data"
qcd_key = "qcd"
hbb_bg_keys = ["gghtobb", "vbfhtobb", "vhtobb", "tthtobb"]
bg_keys = [qcd_key, "ttbar", "diboson", "vjets"] + hbb_bg_keys
sig_keys_ggf = ["hh4b", "hh4b-kl0", "hh4b-kl2p45", "hh4b-kl5"]
sig_keys_vbf = []  # TODO
sig_keys = sig_keys_ggf + sig_keys_vbf

# TODO
jecs = {
    # "JES": "JES_jes",
    # "JER": "JER",
}

jmsr = {
    # "JMS": "JMS",
    # "JMR": "JMR",
}

jec_shifts = []
for key in jecs:
    for shift in ["up", "down"]:
        jec_shifts.append(f"{key}_{shift}")

jmsr_shifts = []
for key in jmsr:
    for shift in ["up", "down"]:
        jmsr_shifts.append(f"{key}_{shift}")
