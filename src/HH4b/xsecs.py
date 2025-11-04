"""
Cross Sections for 13.6 TeV,
"""

from __future__ import annotations

import json

BR_WQQ = 0.676
BR_WLNU = 0.324
BR_ZQQ = 0.69911
BR_ZNUNU = 0.27107
BR_ZLL = 0.02982
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR at mH=125.40
BR_HBB = 5.760e-01
BR_HCC = 2.860e-02

xsecs = {}

# QCD
# also see https://indico.cern.ch/event/1324651/contributions/5576411/attachments/2713143/4711677/2023_09_14_PPD_DijetsAndPairedDijets_JECAndMCNeeds.pdf page 22
xsecs["QCD_PT-30to50"] = 112800000.0
xsecs["QCD_PT-50to80"] = 16660000.0
xsecs["QCD_PT-80to120"] = 2507000.0
xsecs["QCD_PT-120to170"] = 441100.0
xsecs["QCD_PT-170to300"] = 113400.0
xsecs["QCD_PT-300to470"] = 7589.0
xsecs["QCD_PT-470to600"] = 626.4
xsecs["QCD_PT-600to800"] = 178.6
xsecs["QCD_PT-800to1000"] = 30.57
xsecs["QCD_PT-1000to1400"] = 8.92
xsecs["QCD_PT-1400to1800"] = 0.8103
xsecs["QCD_PT-1800to2400"] = 0.1148
xsecs["QCD_PT-2400to3200"] = 0.007542
xsecs["QCD_PT-3200"] = 0.0002331

xsecs["QCD_PT-15to7000"] = 1440000000.0

# xsdb
xsecs["QCDB_HT-40to100"] = 12950000.0
xsecs["QCDB_HT-100to200"] = 1195000.0
xsecs["QCDB_HT-200to400"] = 100600.0
xsecs["QCDB_HT-400to600"] = 4938.0
xsecs["QCDB_HT-600to800"] = 654.0
xsecs["QCDB_HT-800to1000"] = 139.0
xsecs["QCDB_HT-1000to1500"] = 55.15
xsecs["QCDB_HT-1500to2000"] = 4729
xsecs["QCDB_HT-2000"] = 0.8673

# got using genXsecAnalyzer
xsecs["QCD_HT-40to70"] = 311600000.0
xsecs["QCD_HT-70to100"] = 58520000.0
xsecs["QCD_HT-100to200"] = 25220000.0
xsecs["QCD_HT-200to400"] = 1963000.0
xsecs["QCD_HT-400to600"] = 94870.0
xsecs["QCD_HT-600to800"] = 13420.0
xsecs["QCD_HT-800to1000"] = 2992.0
xsecs["QCD_HT-1000to1200"] = 879.1
xsecs["QCD_HT-1200to1500"] = 384.5
xsecs["QCD_HT-1500to2000"] = 125.5
xsecs["QCD_HT-2000"] = 25.78

# xsdb
xsecs["QCD_PT-120to170_MuEnrichedPt5"] = 22980.0
xsecs["QCD_PT-170to300_MuEnrichedPt5"] = 7763.0
xsecs["QCD_PT-300to470_MuEnrichedPt5"] = 699.1
xsecs["QCD_PT-470to600_MuEnrichedPt5"] = 68.24
xsecs["QCD_PT-600to800_MuEnrichedPt5"] = 21.37
xsecs["QCD_PT-800to1000_MuEnrichedPt5"] = 3.913
xsecs["QCD_PT-1000_MuEnrichedPt5"] = 1.323

# Top
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO: 923.6
# https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=TOP-22-012: 887
xsecs["TTto4Q"] = 923.6 * 0.667 * 0.667  # = 410.89  (762.1) - 431.5
xsecs["TTto2L2Nu"] = 923.6 * 0.333 * 0.333  # = 102.41 (96.9) - 91.29
xsecs["TTtoLNu2Q"] = 923.6 * 2 * (0.667 * 0.333)  # = 410.28 (404.0) - 405.1

# Diboson
xsecs["WW"] = 116.8  #  173.4 (116.8 at NNLO)
xsecs["WZ"] = 54.3
xsecs["ZZ"] = 16.7
xsecs["ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8"] = 2.36
xsecs["ZZto2Nu2Q_TuneCP5_13p6TeV_powheg-pythia8"] = 4.48
xsecs["ZZto4L_TuneCP5_13p6TeV_powheg-pythia8"] = 0.170
xsecs["ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8"] = 0.674
xsecs["WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8"] = 12.368
xsecs["WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8"] = 3.696
xsecs["WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8"] = 1.786
xsecs["WWto4Q_TuneCP5_13p6TeV_powheg-pythia8"] = 78.79
xsecs["WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8"] = 76.16
xsecs["WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8"] = 18.41

# SingleTop
xsecs["TbarBQ_t-channel_4FS"] = 87.2
xsecs["TBbarQ_t-channel_4FS"] = 145.0
xsecs["TbarWplusto4Q"] = 10.8
xsecs["TWminusto4Q"] = 10.8
xsecs["TbarWplustoLNu2Q"] = 10.7
xsecs["TWminustoLNu2Q"] = 10.7
xsecs["TbarWplusto2L2Nu"] = 2.62
xsecs["TWminusto2L2Nu"] = 2.62

# Higgs
# SX: took XSDB NLO number (0.5246) and multiplied it by the NNLO/NLO ratio for inclusive ggH from 13 TeV
xsecs["GluGluHto2B_PT-200_M-125"] = 0.5246 * (43.92 / 27.8) * BR_HBB
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap (updated 2024-03 for mH=125.0)
xsecs["GluGluHto2B_M-125"] = 52.23 * BR_HBB  # 30.34
xsecs["VBFHto2B_M-125_dipoleRecoilOn"] = 4.078 * BR_HBB  # 2.368
xsecs["VBFHto2B_M-125"] = xsecs["VBFHto2B_M-125_dipoleRecoilOn"]
xsecs["WminusH_Hto2B_Wto2Q_M-125"] = (
    0.5677 * BR_WQQ * BR_HBB
)  # 0.2229 (0.3916 from xsecdb, but missing Hdecay, 0.3916*BR_HBB=0.227)
xsecs["WminusH_Hto2B_WtoLNu_M-125"] = (
    0.5677 * BR_WLNU * BR_HBB
)  # 0.1068 (0.1887 from xsecdb, but missing Hdecay, 0.1887*BR_HBB=0.1096)
xsecs["WplusH_Hto2B_Wto2Q_M-125"] = (
    0.8889 * BR_WQQ * BR_HBB
)  # 0.349 (0.623 from xsecdb, but missing Hdecay, 0.623*BR_HBB=0.3619)
xsecs["WplusH_Hto2B_WtoLNu_M-125"] = (
    0.8889 * BR_WLNU * BR_HBB
)  # 0.1673 (0.3001 from xsecdb, but missing Hdecay, 0.3001*BR_HBB=0.1743)
xsecs["ZH_Hto2B_Zto2L_M-125"] = (
    0.9439 * BR_ZLL * BR_HBB
)  # 0.01635 (0.08545 from xsecdb, 0.08545*BR_HBB=0.049)
xsecs["ZH_Hto2B_Zto2Q_M-125"] = (
    0.9439 * BR_ZQQ * BR_HBB
)  # 0.3833 (0.5958 from xsecdb, 0.5958*BR_HBB=0.346)
xsecs["ZH_Hto2B_Zto2Nu_M-125"] = (
    0.9439 * BR_ZNUNU * BR_HBB
)  # 0.1486 (0.01351 from xsecdb, 0.01351*BR_HBB=0.00784)
xsecs["ZH_Hto2C_Zto2Q_M-125"] = (
    0.9439 * BR_ZQQ * BR_HCC
)  # 0.019 (0.5958 from xsecdb, 0.5958*BR_HCC=0.0172)
xsecs["ggZH_Hto2B_Zto2L_M-125"] = (
    0.1360 * BR_ZLL * BR_HBB
)  # 0.00235 (0.006838 from xsecdb, 0.006838*BR_HBB=0.00397)
xsecs["ggZH_Hto2B_Zto2Nu_M-125"] = (
    0.1360 * BR_ZNUNU * BR_HBB
)  # 0.0214 (0.01351 from xsecdb, 0.01351*BR_HBB=0.00784)
xsecs["ggZH_Hto2B_Zto2Q_M-125"] = (
    0.1360 * BR_ZQQ * BR_HBB
)  # 0.055 (0.04776 from xsecdb, 0.04776*BR_HBB=0.0277)
xsecs["ggZH_Hto2C_Zto2Q_M-125"] = (
    0.1360 * BR_ZQQ * BR_HCC
)  # 0.0027 (0.04776 from xsecdb, 0.04776*BR_HCC=0.00138)
xsecs["ttHto2B_M-125"] = 0.5700 * BR_HBB  # 0.3319 (0.5742 from xsecdb, 0.5742*BR_HBB=0.334)

# Triple-higgs
# SM sample
xsecs["HHHTo6B_c3_1_d4_0"] = 2.908e-05 * BR_HBB * BR_HBB * BR_HBB  # (from xsecdb)

# Di-Higgs
# ggF HH
# https://gitlab.cern.ch/hh/recommendations/-/blob/master/CrossSections.md?ref_type=heads (including mass k-factor)
# ggHH xsec 13.6: = 75.7617-53.2855*κλ+11.6126*κλ2 in fb
# κmass=(33.97/34.09)=0.99647990613083
hh = {
    # kl scan
    "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": 0.075495 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": 0.033969 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": 0.099298 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": 0.014864 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV": 0.017809 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35_TuneCP5_13p6TeV": 0.010448 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV": 2.900686 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00_TuneCP5_13p6TeV": 1.985733 * BR_HBB * BR_HBB,
}
for key, value in hh.items():
    xsecs[key] = value
    xsecs[f"{key}_TSG"] = value
    xsecs[f"{key}_Private"] = value
    xsecs[f"{key}_TSG_Pu60"] = value

# VBF HH
# https://gitlab.cern.ch/hh/recommendations/-/blob/master/CrossSections.md?ref_type=heads (including mass k-factor)
# κmass​=(1.861/1.870)=0.995187165775401
vbfhh = {
    "VBFHHto4B_CV-1_C2V-1_C3-1": 0.0019292 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-1_C2V-0_C3-1": 0.0296772 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-1_C2V-1_C3-0": 0.0049909 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-1_C2V-1_C3-2": 0.0016124 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-1_C2V-2_C3-1": 0.0159014 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-1p74_C2V-1p37_C3-14p4": 0.4002163 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-m0p012_C2V-0p030_C3-10p2": 0.0000127 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-m0p758_C2V-1p44_C3-m19p3": 0.3593242 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-m0p962_C2V-0p959_C3-m1p43": 0.0011275 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-m1p21_C2V-1p94_C3-m0p94": 0.0037987 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-m1p60_C2V-2p72_C3-m1p36": 0.0117008 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-m1p83_C2V-3p57_C3-m3p39": 0.0168528 * BR_HBB * BR_HBB,
    "VBFHHto4B_CV-m2p12_C2V-3p87_C3-m5p96": 0.6800842 * BR_HBB * BR_HBB,
}
for key, value in vbfhh.items():
    key_nounderscore = key.replace("-", "_")
    xsecs[f"{key}_TuneCP5_13p6TeV_madgraph-pythia8"] = value
    xsecs[f"{key_nounderscore}_TuneCP5_13p6TeV_madgraph-pythia8"] = value

# V+Jets (xsdb)
xsecs["Wto2Q-3Jets_HT-200to400"] = 2723.0
xsecs["Wto2Q-3Jets_HT-400to600"] = 299.8
xsecs["Wto2Q-3Jets_HT-600to800"] = 63.9
xsecs["Wto2Q-3Jets_HT-800"] = 31.9
xsecs["Zto2Q-4Jets_HT-200to400"] = 1082.0
xsecs["Zto2Q-4Jets_HT-400to600"] = 124.1
xsecs["Zto2Q-4Jets_HT-600to800"] = 27.28
xsecs["Zto2Q-4Jets_HT-800"] = 14.57

# LO samples in 2024
# from xsec analyzer: https://cms-generators.docs.cern.ch/useful-tools-and-links/HowToGenXSecAnalyzer
xsecs["Wto2Q-3Jets_Bin-HT-100to400"] = 16120
xsecs["Wto2Q-3Jets_Bin-HT-400to800"] = 354.2
xsecs["Wto2Q-3Jets_Bin-HT-800to1500"] = 29.6
xsecs["Wto2Q-3Jets_Bin-HT-1500to2500"] = 1.852
xsecs["Wto2Q-3Jets_Bin-HT-2500"] = 0.1177
xsecs["Zto2Q-4Jets_Bin-HT-100to400"] = 6328
xsecs["Zto2Q-4Jets_Bin-HT-400to800"] = 145.1
xsecs["Zto2Q-4Jets_Bin-HT-800to1500"] = 12.93
xsecs["Zto2Q-4Jets_Bin-HT-1500to2500"] = 0.8496
xsecs["Zto2Q-4Jets_Bin-HT-2500"] = 0.05672

xsecs["Wto2Q-2Jets_PTQQ-100to200_1J"] = 1517.0
xsecs["Wto2Q-2Jets_PTQQ-100to200_2J"] = 1757.0
xsecs["Wto2Q-2Jets_PTQQ-200to400_1J"] = 103.6
xsecs["Wto2Q-2Jets_PTQQ-200to400_2J"] = 227.1
xsecs["Wto2Q-2Jets_PTQQ-400to600_1J"] = 3.496
xsecs["Wto2Q-2Jets_PTQQ-400to600_2J"] = 12.75
xsecs["Wto2Q-2Jets_PTQQ-600_1J"] = 0.4221
xsecs["Wto2Q-2Jets_PTQQ-600_2J"] = 2.128
xsecs["Zto2Q-2Jets_PTQQ-100to200_1J"] = 302.0
xsecs["Zto2Q-2Jets_PTQQ-100to200_2J"] = 343.9
xsecs["Zto2Q-2Jets_PTQQ-200to400_1J"] = 21.64
xsecs["Zto2Q-2Jets_PTQQ-200to400_2J"] = 48.36
xsecs["Zto2Q-2Jets_PTQQ-400to600_1J"] = 0.7376
xsecs["Zto2Q-2Jets_PTQQ-400to600_2J"] = 2.683
xsecs["Zto2Q-2Jets_PTQQ-600_1J"] = 0.08717
xsecs["Zto2Q-2Jets_PTQQ-600_2J"] = 0.4459

# LO samples in 2024
# from xsec analyzer: https://cms-generators.docs.cern.ch/useful-tools-and-links/HowToGenXSecAnalyzer
xsecs["Wto2Q-2Jets_Bin-PTQQ-100"] = 1751.0
xsecs["Wto2Q-2Jets_Bin-PTQQ-200"] = 164.3
xsecs["Wto2Q-2Jets_Bin-PTQQ-400"] = 9.205
xsecs["Wto2Q-2Jets_Bin-PTQQ-600"] = 2.23
xsecs["Zto2Q-2Jets_Bin-PTQQ-100"] = 695.0
xsecs["Zto2Q-2Jets_Bin-PTQQ-200"] = 71.56
xsecs["Zto2Q-2Jets_Bin-PTQQ-400"] = 3.811
xsecs["Zto2Q-2Jets_Bin-PTQQ-600"] = 0.5086

xsecs["WtoLNu-4Jets"] = 55390.0
xsecs["WtoLNu-2Jets"] = 64481.58
xsecs["WtoLNu-2Jets_0J"] = 55760.0
xsecs["WtoLNu-2Jets_1J"] = 9529.0
xsecs["WtoLNu-2Jets_2J"] = 3532.0
xsecs["WtoLNu-4Jets_1J"] = 9625.0
xsecs["WtoLNu-4Jets_2J"] = 3161.0
xsecs["WtoLNu-4Jets_3J"] = 1468.0

# WtoLNu-4Jets 2024 samples
xsecs["WtoLNu-4Jets_Bin-1J"] = 9141
xsecs["WtoLNu-4Jets_Bin-2J"] = 2931
xsecs["WtoLNu-4Jets_Bin-3J"] = 864.6
xsecs["WtoLNu-4Jets_Bin-4J"] = 417.8
# WtoLNu-2Jets 2024 samples
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-40to100"] = 4211
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-100to200"] = 342.3
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-200to400"] = 21.84
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-400to600"] = 0.6845
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-600"] = 0.07753
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-40to100"] = 1581
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-100to200"] = 411.1
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-200to400"] = 53.59
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-400to600"] = 3.099
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-600"] = 0.5259


xsecs["DYto2L-4Jets_MLL-50"] = 5467.0
xsecs["DYto2L-2Jets_MLL-50"] = 6688.0
xsecs["DYto2L-2Jets_MLL-50_0J"] = 5378.0
xsecs["DYto2L-2Jets_MLL-50_1J"] = 1017.0
xsecs["DYto2L-2Jets_MLL-50_2J"] = 385.5

# LO samples in 2024 (xsdb)
xsecs["DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100"] = 475.3
xsecs["DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200"] = 45.42
xsecs["DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400"] = 3.382
xsecs["DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600"] = 0.1162
xsecs["DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600"] = 0.01392

xsecs["DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100"] = 179.3
xsecs["DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200"] = 51.68
xsecs["DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400"] = 7.159
xsecs["DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600"] = 0.4157
xsecs["DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600"] = 0.07019


########################################################
# Cross Sections for 13 TeV.
########################################################
xsecs["QCD_HT-50to100-13TeV"] = 2.486e08
xsecs["QCD_HT-100to200-13TeV"] = 27990000.0
xsecs["QCD_HT-200to300-13TeV"] = 1712000
xsecs["QCD_HT-300to500-13TeV"] = 347700
xsecs["QCD_HT-500to700-13TeV"] = 30330.0
xsecs["QCD_HT-700to1000-13TeV"] = 6412.0
xsecs["QCD_HT-1000to1500-13TeV"] = 1118.0
xsecs["QCD_HT-1500to2000-13TeV"] = 108.5
xsecs["QCD_HT-2000toInf-13TeV"] = 21.94

xsecs["GluGlutoHHto4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8"] = (
    31.05e-3 * 5.824e-01 * 5.824e-01
)
xsecs["GluGlutoHHto4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8"] = (
    0.069725 * 5.824e-01 * 5.824e-01
)
xsecs["GluGlutoHHto4B_cHHH2p45_TuneCP5_PSWeights_13TeV-powheg-pythia8"] = (
    0.013124 * 5.824e-01 * 5.824e-01
)
xsecs["GluGlutoHHto4B_cHHH5_TuneCP5_PSWeights_13TeV-powheg-pythia8"] = (
    0.091172 * 5.824e-01 * 5.824e-01
)

xsecs["TTToHadronic_13TeV"] = 670.3 * 1.24088 * 0.667 * 0.667
xsecs["TTTo2L2Nu_13TeV"] = 670.3 * 1.24088 * 0.333 * 0.333
xsecs["TTToSemiLeptonic_13TeV"] = 670.3 * 1.24088 * 2 * (0.667 * 0.333)

xsecs["WJetsToQQ_HT-200to400_13TeV"] = 2549.0
xsecs["WJetsToQQ_HT-400to600_13TeV"] = 277.0
xsecs["WJetsToQQ_HT-600to800_13TeV"] = 59.06
xsecs["WJetsToQQ_HT-800toInf_13TeV"] = 28.75

xsecs["ZJetsToQQ_HT-200to400_13TeV"] = 1012.0
xsecs["ZJetsToQQ_HT-400to600_13TeV"] = 114.5
xsecs["ZJetsToQQ_HT-600to800_13TeV"] = 25.41
xsecs["ZJetsToQQ_HT-800toInf_13TeV"] = 12.91

xsecs["GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8"] = 0.27395244
xsecs["GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8"] = 27.8
xsecs["VBFHToBB_M-125_dipoleRecoilOn_TuneCP5_13TeV-powheg-pythia8"] = 2.2498257
xsecs["WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.21348075
xsecs["WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.1028193
xsecs["WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.3421501
xsecs["WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.16451088
xsecs["ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.326
xsecs["ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.0463
xsecs["ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.0913
xsecs["ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8"] = 0.04319
xsecs["ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8"] = 0.2912

xsecs["ZZTo4B01j_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8"] = 0.3707
xsecs["ZZ_TuneCP5_13TeV-pythia8"] = 16.91


def main():
    with open("xsecs.json", "w") as outfile:  # noqa: PTH123
        json.dump(xsecs, outfile, indent=4)


if __name__ == "__main__":
    main()
