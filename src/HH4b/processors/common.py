# TODO: Check if this is correct for JetHT
LUMI = {  # in pb^-1
    "2022": 38480.0,
    "2023": 26120.0,
}

HLTs = {
    "2022": [],
    "2023": [],
}

jecs = {
    "JES": "JES_jes",
    "JER": "JER",
}

jmsr = {
    "JMS": "JMS",
    "JMR": "JMR",
}

jec_shifts = []
for key in jecs:
    for shift in ["up", "down"]:
        jec_shifts.append(f"{key}_{shift}")

jmsr_shifts = []
for key in jmsr:
    for shift in ["up", "down"]:
        jmsr_shifts.append(f"{key}_{shift}")

# variables affected by JECs
jec_vars = [
    "ak8FatJetPt",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    # "bbFatJetPtOverDijetPt",
    # "VVFatJetPtOverDijetPt",
    # "VVFatJetPtOverbbFatJetPt",
    # "BDTScore",
]


# variables affected by JMS/R
jmsr_vars = [
    "ak8FatJetMsd",
    "ak8FatJetParticleNetMass",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    # "bbFatJetPtOverDijetPt",
    # "VVFatJetPtOverDijetPt",
    # "VVFatJetPtOverbbFatJetPt",
    # "BDTScore",
]
