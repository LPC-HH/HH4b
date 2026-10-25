"""
Common variables

Authors: Raghav Kansal, Cristina Suarez
"""

from __future__ import annotations

from collections import OrderedDict

years = ["2022", "2022EE", "2023", "2023BPix", "2024", "2025"]

# Golden JSON of each year's data (corrections/data/golden_json/, corrections/fetch_golden_json.py);
# utils.load_samples drops the data events outside it.
GOLDEN_JSON = {
    "2022": "Cert_Collisions2022_355100_362760_Golden.json",
    "2022EE": "Cert_Collisions2022_355100_362760_Golden.json",
    "2023": "Cert_Collisions2023_366442_370790_Golden.json",
    "2023BPix": "Cert_Collisions2023_366442_370790_Golden.json",
    "2024": "Cert_Collisions2024_378981_386951_Golden.json",
    "2025": "Cert_Collisions2025_391658_398903_Golden.json",
}

# analyzed eras: first and last run of their golden lumisections, from the CAT per-era golden JSONs
# (2022-2024) and the runs of the JetMET0 PromptReco datasets on DAS (2025); the B eras are not used
DATA_ERAS = {
    "2022": {"C": (355862, 357482), "D": (357538, 357900)},
    "2022EE": {"E": (359022, 360331), "F": (360390, 362167), "G": (362433, 362760)},
    "2023": {"C": (367095, 368823)},
    "2023BPix": {"D": (369803, 370790)},
    "2024": {
        "C": (379416, 380236),
        "D": (380306, 380947),
        "E": (380963, 381544),
        "F": (382229, 383779),
        "G": (383811, 385801),
        "H": (385836, 386319),
        "I": (386478, 386951),
    },
    "2025": {
        "C": (392174, 393516),
        "D": (394393, 395948),
        "E": (395982, 396422),
        "F": (396629, 397853),
        "G": (397867, 398903),
    },
}

# in pb^-1; Run 3: brilcalc of the golden JSONs above in the analyzed eras
# (corrections/golden_lumi.py), normtag_PHYSICS; 2025 normtag_BRIL, preliminary (normtag_PHYSICS has
# no 2025 runs yet)
LUMI = {
    "2022": 7989.5,
    "2022EE": 26674.9,
    "2023": 17964.2,
    "2023BPix": 9676.7,
    "2024": 109804.9,
    "2024C": 7250.7,
    "2024D": 7976.1,
    "2024E": 11423.2,
    "2024F": 28037.8,
    "2024G": 38069.1,
    "2024H": 5489.7,
    "2024I": 11558.2,
    "2025": 110374.7,
    "2025C": 21563.6,
    "2025D": 25823.9,
    "2025E": 14048.2,
    "2025F": 26693.6,
    "2025G": 22245.3,
    "2018": 59830.0,
    "2017": 41480.0,
    "2016": 36330.0,
    "Run2": 137640.0,
}
# combined eras, summed from the per-era values above
LUMI["2022All"] = LUMI["2022"] + LUMI["2022EE"]
LUMI["2023All"] = LUMI["2023"] + LUMI["2023BPix"]
LUMI["2022-2023"] = LUMI["2022All"] + LUMI["2023All"]
LUMI["2022-2024"] = LUMI["2022-2023"] + LUMI["2024"]
LUMI["2022-2025"] = LUMI["2022-2024"] + LUMI["2025"]


DATA_SAMPLES = ["JetMET", "Muon", "EGamma"]

# sample key -> list of samples or selectors
common_samples_bg = {
    "qcd": ["QCD-4Jets_HT", "QCD_HT-"],
    "data": [f"{key}_Run" for key in DATA_SAMPLES],
    "ttbar": ["TTto4Q", "TTto2L2Nu", "TTtoLNu2Q"],
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
    "novhhtobb": ["GluGluHto2B_PT-200_M-125", "VBFHto2B_M-125_dipoleRecoilOn", "VBFHto2B_M-125"],
    "tthtobb": ["ttHto2B_M-125"],
    # Use exact-match selectors to avoid pulling in ZZto*/WWto*/WZto* sub-samples (ZZto4B-1Jets
    # is loaded only with PostProcess --zz4b-tag, zz4b_replace_selectors).
    "zz": ["ZZ?"],
    "nozzdiboson": ["WW?", "WZ?"],
    # 2022-2023: *_PTQQ-XtoY_{1J,2J} are exclusive bins -> sum them all.
    # 2024 (Summer24): *_Bin-PTQQ-X means PTQQ > X with no upper edge, so the four samples overlap
    # (summing them over-counts V pT > 200 GeV 2-4x). Default: only the inclusive PTQQ-100 sample
    # (exact match, trailing "?"). PostProcess --vjets-stitch uses vjets_stitch_selectors below.
    "vjets": [
        "Wto2Q-2Jets_PTQQ",
        "Wto2Q-2Jets_Bin-PTQQ-100?",
        "Zto2Q-2Jets_PTQQ",
        "Zto2Q-2Jets_Bin-PTQQ-100?",
    ],
}

# Opt-in (PostProcess --vjets-stitch): the default vjets selectors plus the three higher-threshold
# open-ended 2024 samples, loaded with utils.load_samples(ptqq_stitch=...) so that the four
# Bin-PTQQ samples are stitched in GenVPt (utils._stitch_open_ptqq), or with --vjets-stitch-mode
# range in LHE V pT ranges (utils.ptqq_range_norm). No 2022-2023 name matches them.
# Wto2Q-2Jets_Bin-PTQQ-600 is the v2-v3 production; skims of the INVALID v2-v2 one (LHE filter at
# 400 GeV instead of 600) are left out by utils.load_samples (PTQQ_OPEN_FILTER_CHECK), and W above
# 600 GeV then comes from W-100/200/400 (range mode: W-400 alone, whole sample).
vjets_stitch_selectors = common_samples_bg["vjets"] + [
    f"{v}to2Q-2Jets_Bin-PTQQ-{x}?" for v in ["W", "Z"] for x in [200, 400, 600]
]

# Opt-in (PostProcess --zz4b-tag): the dedicated ZZto4B-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8
# sample (2024 NanoAODv15; 2025 through --split-shared-mc) replaces part of the inclusive pythia8
# ZZ in utils.load_samples (utils.ZZ4bReplace): the inclusive ZZ events with Z->bb, Z->bb and both
# Z masses in the ZZto4B MadSpin window are dropped and taken from ZZto4B alone; the other ZZ
# events stay.
ZZ4B_SAMPLE = "ZZto4B-1Jets"
ZZ_INCLUSIVE_SAMPLE = "ZZ"
zz4b_replace_selectors = common_samples_bg["zz"] + [f"{ZZ4B_SAMPLE}?"]
# MadSpin window of the ZZto4B gridpack (param_card mZ = 91.15351 GeV, Gamma_Z = 2.493202 GeV;
# run_card bwcutoff = 15): mZ +- 15 Gamma_Z = [53.755, 128.552] GeV for both Z.
ZZ4B_MZ_WINDOW = (91.15351 - 15 * 2.493202, 91.15351 + 15 * 2.493202)

# ttH(bb): 2022 and 2022EE hold three skimmed productions of the same Powheg ttH(bb) sample
# (TTHto2B_M-125, TTH_Hto2B_M-125, ttHto2B_M-125: same Powheg input and integration grids,
# independent events). The case-insensitive prefix selector "ttHto2B_M-125" matched both
# TTHto2B_M-125 and ttHto2B_M-125, each normalised to sigma x L on its own, so ttH was counted
# twice. These eras load exactly one production, TTHto2B_M-125 (the 2023/2023BPix production, the
# largest), with a case-sensitive exact selector ("=" prefix, utils.check_selector). 2023/2023BPix
# (TTHto2B_M-125) and 2024 (ttHto2B_M-125, dataset TTH-Hto2B_Par) have one directory each and keep
# the common selector.
tthtobb_era_selectors = {year: ["=TTHto2B_M-125"] for year in ["2022", "2022EE"]}

# VBF H(bb): 2022 and 2022EE also hold two productions, VBFHto2B_M-125 and
# VBFHto2B_M-125_dipoleRecoilOn. The common novhhtobb selectors name both (and "VBFHto2B_M-125" is
# also a prefix of the dipole name), so both were loaded, each normalised to sigma x L. These eras
# load exactly VBFHto2B_M-125, the production of 2023BPix and 2024 (2023 has none skimmed).
novhhtobb_era_selectors = {
    year: ["GluGluHto2B_PT-200_M-125", "=VBFHto2B_M-125"] for year in ["2022", "2022EE"]
}

common_samples_sig = {}

samples_run3_sig = {
    "2022": {
        # Use prefix selectors so both short and fully qualified dataset names match.
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00"],
        "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1"],
        "vbfhh4b-kv1p74-k2v1p37-kl14p4": ["VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4"],
        "vbfhh4b-kvm0p012-k2v0p03-kl10p2": ["VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2"],
        "vbfhh4b-kvm0p758-k2v1p44-klm19p3": ["VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3"],
        "vbfhh4b-kvm0p962-k2v0p959-klm1p43": ["VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43"],
        "vbfhh4b-kvm1p21-k2v1p94-klm0p94": ["VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94"],
        "vbfhh4b-kvm1p6-k2v2p72-klm1p36": ["VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36"],
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": ["VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39"],
        # "vbfhh4b-kvm2p12-k2v3p87-klm5p96": ["VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96"],  # wrong sign, should be +2p12
        "vbfhh4b-kv2p12-k2v3p87-klm5p96": ["VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96"],
    },
    "2022EE": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00"],
        "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1"],
        "vbfhh4b-kv1p74-k2v1p37-kl14p4": ["VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4"],
        "vbfhh4b-kvm0p012-k2v0p03-kl10p2": ["VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2"],
        "vbfhh4b-kvm0p758-k2v1p44-klm19p3": ["VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3"],
        "vbfhh4b-kvm0p962-k2v0p959-klm1p43": ["VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43"],
        "vbfhh4b-kvm1p21-k2v1p94-klm0p94": ["VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94"],
        "vbfhh4b-kvm1p6-k2v2p72-klm1p36": ["VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36"],
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": ["VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39"],
        # "vbfhh4b-kvm2p12-k2v3p87-klm5p96": ["VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96"],  # wrong sign, should be +2p12
        "vbfhh4b-kv2p12-k2v3p87-klm5p96": ["VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96"],
    },
    "2023": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00"],
        "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1"],
        # nanov15 uses underscores, older data uses hyphens
        "vbfhh4b-kv1p74-k2v1p37-kl14p4": [
            "VBFHHto4B_CV_1p74_C2V_1p37_C3_14p4",
            "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4",
        ],
        "vbfhh4b-kvm0p012-k2v0p03-kl10p2": [
            "VBFHHto4B_CV_m0p012_C2V_0p030_C3_10p2",
            "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2",
        ],
        "vbfhh4b-kvm0p758-k2v1p44-klm19p3": [
            "VBFHHto4B_CV_m0p758_C2V_1p44_C3_m19p3",
            "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3",
        ],
        "vbfhh4b-kvm0p962-k2v0p959-klm1p43": [
            "VBFHHto4B_CV_m0p962_C2V_0p959_C3_m1p43",
            "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43",
        ],
        "vbfhh4b-kvm1p21-k2v1p94-klm0p94": [
            "VBFHHto4B_CV_m1p21_C2V_1p94_C3_m0p94",
            "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94",
        ],
        "vbfhh4b-kvm1p6-k2v2p72-klm1p36": [
            "VBFHHto4B_CV_m1p60_C2V_2p72_C3_m1p36",
            "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36",
        ],
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": [
            "VBFHHto4B_CV_m1p83_C2V_3p57_C3_m3p39",
            "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39",
        ],
        # "vbfhh4b-kvm2p12-k2v3p87-klm5p96": [  # wrong sign, should be +2p12
        #     "VBFHHto4B_CV_m2p12_C2V_3p87_C3_m5p96",
        #     "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96",
        # ],
        "vbfhh4b-kv2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV_2p12_C2V_3p87_C3_m5p96",
            "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96",
        ],
    },
    "2023BPix": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00"],
        "vbfhh4b": ["VBFHHto4B_CV_1_C2V_1_C3_1"],
        "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1"],
        # nanov15 uses underscores, older data uses hyphens
        "vbfhh4b-kv1p74-k2v1p37-kl14p4": [
            "VBFHHto4B_CV_1p74_C2V_1p37_C3_14p4",
            "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4",
        ],
        "vbfhh4b-kvm0p012-k2v0p03-kl10p2": [
            "VBFHHto4B_CV_m0p012_C2V_0p030_C3_10p2",
            "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2",
        ],
        "vbfhh4b-kvm0p758-k2v1p44-klm19p3": [
            "VBFHHto4B_CV_m0p758_C2V_1p44_C3_m19p3",
            "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3",
        ],
        "vbfhh4b-kvm0p962-k2v0p959-klm1p43": [
            "VBFHHto4B_CV_m0p962_C2V_0p959_C3_m1p43",
            "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43",
        ],
        "vbfhh4b-kvm1p21-k2v1p94-klm0p94": [
            "VBFHHto4B_CV_m1p21_C2V_1p94_C3_m0p94",
            "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94",
        ],
        "vbfhh4b-kvm1p6-k2v2p72-klm1p36": [
            "VBFHHto4B_CV_m1p60_C2V_2p72_C3_m1p36",
            "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36",
        ],
        "vbfhh4b-kvm1p83-k2v3p57-klm3p39": [
            "VBFHHto4B_CV_m1p83_C2V_3p57_C3_m3p39",
            "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39",
        ],
        # "vbfhh4b-kvm2p12-k2v3p87-klm5p96": [  # wrong sign, should be +2p12
        #     "VBFHHto4B_CV_m2p12_C2V_3p87_C3_m5p96",
        #     "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96",
        # ],
        "vbfhh4b-kv2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV_2p12_C2V_3p87_C3_m5p96",
            "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96",
        ],
    },
    "2024": {
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
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
        "vbfhh4b-kv2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
    },
    "2025": {
        # Using 2024 MC now since only 2024 MC samples are available
        # TODO: change to 2025 once available
        "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
        "hh4b-kl0": ["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
        "hh4b-kl2p45": ["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
        "hh4b-kl5": ["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV"],
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
        "vbfhh4b-kv2p12-k2v3p87-klm5p96": [
            "VBFHHto4B_CV-2p12_C2V-3p87_C3-m5p96_TuneCP5_13p6TeV_madgraph-pythia8"
        ],
    },
}

samples_run3 = {
    "2022": {
        **common_samples_bg,
        "tthtobb": tthtobb_era_selectors["2022"],
        "novhhtobb": novhhtobb_era_selectors["2022"],
        **samples_run3_sig["2022"],
    },
    "2022EE": {
        **common_samples_bg,
        "tthtobb": tthtobb_era_selectors["2022EE"],
        "novhhtobb": novhhtobb_era_selectors["2022EE"],
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
        **common_samples_bg,
        "data": [
            "JetMET_Run2024B",
            "JetMET_Run2024C",
            "JetMET_Run2024D",
            "JetMET_Run2024E",
            "JetMET_Run2024F",
            "JetMET_Run2024G",
            "JetMET_Run2024H",
            "JetMET_Run2024I",
        ],
        **samples_run3_sig["2024"],
    },
    "2025": {
        **common_samples_bg,
        "data": [
            "JetMET_Run2025C",
            "JetMET_Run2025D",
            "JetMET_Run2025E",
            "JetMET_Run2025F",
            "JetMET_Run2025G",
        ],
        **samples_run3_sig["2025"],
    },
}

samples_2018 = {
    "hh4b": [
        "GluGlutoHHto4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8",
    ],
    "qcd": [
        "QCD-4Jets_HT-1000to1500-13TeV",
        "QCD-4Jets_HT-1500to2000-13TeV",
        "QCD-4Jets_HT-2000toInf-13TeV",
        "QCD-4Jets_HT-200to300-13TeV",
        "QCD-4Jets_HT-300to500-13TeV",
        "QCD-4Jets_HT-500to700-13TeV",
        "QCD-4Jets_HT-700to1000-13TeV",
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
    # sample dict carries CV=+2.12 (VBFHHto4B_CV-2p12); the old kvm2p12 (CV=-2.12)
    # was the wrong-sign point and is commented out in samples_run3, so match kv2p12.
    "vbfhh4b-kv2p12-k2v3p87-klm5p96",
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
    "VBFJetMass",
    "AK4JetAwayPt",
    "AK4JetAwayMass",
    "MET_pt",
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
    "bbFatJetParTmassVis",  # glopart-v2 regressed mass
    "bbFatJetParT3massX2p",  # glopart-v3 regressed mass (v15 skimmer)
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
    "2024": {"nom": 1.0, "down": 1.0, "up": 1.0},  # TODO: placeholder for future
    "2025": {"nom": 1.0, "down": 1.0, "up": 1.0},  # TODO: placeholder for future
}
jmsr_values["bbFatJetPNetMassLegacy"]["JMS"] = {
    "2022": {"nom": 1.015, "down": 1.010, "up": 1.020},
    "2022EE": {"nom": 1.021, "down": 1.018, "up": 1.024},
    "2023": {"nom": 0.999, "down": 0.996, "up": 1.003},
    "2023BPix": {"nom": 0.974, "down": 0.970, "up": 0.980},
    "2024": {"nom": 1.0, "down": 1.0, "up": 1.0},  # TODO: placeholder for future
    "2025": {"nom": 1.0, "down": 1.0, "up": 1.0},  # TODO: placeholder for future
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
    "2024": {"nom": 1.0, "down": 1.0, "up": 1.0},  # TODO: placeholder for future
    "2025": {"nom": 1.0, "down": 1.0, "up": 1.0},  # TODO: placeholder for future
}
jmsr_values["bbFatJetParTmassVis"]["JMS"] = {
    "2022": {"nom": 1.011, "down": 1.007, "up": 1.014},
    "2022EE": {"nom": 1.011, "down": 1.007, "up": 1.014},
    "2023": {"nom": 0.9867, "down": 0.983, "up": 0.9903},
    "2023BPix": {"nom": 0.9867, "down": 0.983, "up": 0.9903},
    "2024": {"nom": 1.0, "down": 1.0, "up": 1.0},  # TODO: placeholder for future
    "2025": {"nom": 1.0, "down": 1.0, "up": 1.0},  # TODO: placeholder for future
}
# GloParT-v3 regressed mass (v15 skims). There is no GloParT-v3 jet-mass measurement, so these are
# PLACEHOLDERS with safe uncertainties chosen by the user (2026-09-26), the same in every era: JMS
# nominal 1 (no correction), down/up 0.95/1.05; JMR nominal 1 (no extra smearing), down 1 (one-sided),
# up 1.15. JMR is a RESOLUTION scaling (sigma_new = JMR * sigma_MC): postprocessing.jmsr_correct_mass
# adds a Gaussian smear of width sqrt(JMR^2 - 1) * jmsr_res (below), so the nominal and JMR-down masses
# are the uncorrected MC mass. The v15 skimmer applies JMS/JMR only to ParTmassVis, so PostProcess
# applies these at load time (--no-glopartv3-jmsr = off).
jmsr_values["bbFatJetParT3massX2p"] = {
    "JMR": {
        year: {"nom": 1.0, "down": 1.0, "up": 1.15}
        for year in ["2022", "2022EE", "2023", "2023BPix", "2024", "2025"]
    },
    "JMS": {
        year: {"nom": 1.0, "down": 0.95, "up": 1.05}
        for year in ["2022", "2022EE", "2023", "2023BPix", "2024", "2025"]
    },
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
# GloParT-v3: core resolution (GeV) of the uncorrected bbFatJetParT3massX2p of Higgs-matched, fully
# merged jets in HH signal MC (iterative Gaussian fit in +-1.5 sigma; 2026-09-26): 10.9-11.6 in ggF and
# VBF SM, 2022-2024, mean 11.4; 10.5-10.7 in VBF kappa_2V = 0 (the same fit gives 10.9-11.5 for the
# GloParT-v2 ParTmassVis above). It sets the
# JMR smear of every sample postprocessing.jmsr_correct_mass corrects (all MC except QCD): Higgs-jet
# samples take the signal value, the others the value x 80/125 as VH, ZZ and WW/WZ above. ttbar, ttH,
# single H and V+jets have no JMS/JMR nuisance (not in jmsr_keys); their entries only fill the unused
# shifted columns.
jmsr_res["bbFatJetParT3massX2p"] = dict.fromkeys(sig_keys + ["novhhtobb", "tthtobb"], 11.4)
jmsr_res["bbFatJetParT3massX2p"].update(
    dict.fromkeys(["vhtobb", "zz", "nozzdiboson", "vjets", "ttbar"], 11.4 * 80.0 / 125.0)
)

ttbarsfs_decorr_txbb_bins = {}
ttbarsfs_decorr_txbb_bins["pnet-legacy"] = [0, 0.8, 0.94, 0.99, 1]
ttbarsfs_decorr_txbb_bins["glopart-v2"] = [0, 0.31, 0.7, 0.8, 0.87, 0.92, 0.96, 1]
ttbarsfs_decorr_txbb_bins["glopart-v3"] = ttbarsfs_decorr_txbb_bins["glopart-v2"]
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
        ("WP4", [0.3, 0.95]),
        ("WP3", [0.95, 0.975]),
        ("WP2", [0.975, 0.99]),
        ("WP1", [0.99, 1]),
    ]
)
txbbsfs_decorr_txbb_wps["glopart-v3"] = txbbsfs_decorr_txbb_wps["glopart-v2"]

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
        ("WP4", [250, 450, 100000]),
        ("WP3", [250, 450, 100000]),
        ("WP2", [250, 450, 100000]),
        ("WP1", [250, 450, 100000]),
    ]
)
txbbsfs_decorr_pt_bins["glopart-v3"] = txbbsfs_decorr_pt_bins["glopart-v2"]

txbb_strings = {
    "pnet-legacy": "bbFatJetPNetTXbbLegacy",
    "pnet-v12": "bbFatJetPNetTXbb",
    "glopart-v2": "bbFatJetParTTXbb",
    # ParT v3 ntuples (new): use ParT3 TXbb branch
    "glopart-v3": "bbFatJetParT3TXbb",
}

mreg_strings = {
    "pnet-legacy": "bbFatJetPNetMassLegacy",
    "pnet-v12": "bbFatJetPNetMass",
    "glopart-v2": "bbFatJetParTmassVis",
    # ParT v3 ntuples: use X2p ParT3 mass
    "glopart-v3": "bbFatJetParT3massX2p",
}
