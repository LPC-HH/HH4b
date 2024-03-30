"""
Cross Sections for 13.6 TeV,
"""
from __future__ import annotations

BR_WQQ = 0.676
BR_WLNU = 0.324
BR_ZQQ = 0.69911
BR_ZLNU = 0.27107
BR_ZLL = 0.02982

BR_HBB = 0.5809
# BR_HBB = 0.5824

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

# QCD-HT (obtained by Cristina manually with genXsecAnalyzer
# QCDB-4Jets_HT*
# these numbers seem unusually high - xcheck
xsecs["QCDB_HT-40to100"] = 12950000.0
xsecs["QCDB_HT-100to200"] = 1195000.0
xsecs["QCDB_HT-200to400"] = 100600.0
xsecs["QCDB_HT-400to600"] = 4938.0
xsecs["QCDB_HT-600to800"] = 654.0
xsecs["QCDB_HT-800to1000"] = 139.0
xsecs["QCDB_HT-1000to1500"] = 55.15
xsecs["QCDB_HT-1500to2000"] = 4729
xsecs["QCDB_HT-2000"] = 0.8673

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

# Top
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
# cross check these?
# https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=TOP-22-012
xsecs["TTto4Q"] = 923.6 * 0.667 * 0.667  # = 410.89  (762.1)
xsecs["TTto2L2Nu"] = 923.6 * 0.333 * 0.333  # = 102.41 (96.9)
xsecs["TTtoLNu2Q"] = 923.6 * 2 * (0.667 * 0.333)  # = 410.28 (404.0)

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
# xsecs["WWto4Q_1Jets-4FS"]
# xsecs["WZto4Q-1Jets-4FS"]

# Higgs
# SX: took XSDB NLO number (0.5246) and multiplied it by the NNLO/NLO ratio for inclusive ggH from 13 TeV
xsecs["GluGluHto2B_PT-200_M-125"] = 0.5246 * (43.92 / 27.8) * BR_HBB
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap
xsecs["VBFHto2B_M-125_dipoleRecoilOn"] = 4.078 * BR_HBB
xsecs["WminusH_Hto2B_Wto2Q_M-125"] = 0.8889 * BR_WQQ * BR_HBB
xsecs["WminusH_Hto2B_WtoLNu_M-125"] = 0.8889 * BR_WLNU * BR_HBB
xsecs["WplusH_Hto2B_Wto2Q_M-125"] = 0.5677 * BR_WQQ * BR_HBB
xsecs["ZH_Hto2B_Zto2L_M-125"] = 0.8079 * BR_ZLL * BR_HBB
xsecs["ZH_Hto2B_Zto2Q_M-125"] = 0.8079 * BR_ZQQ * BR_HBB
xsecs["ggZH_Hto2B_Zto2L_M-125"] = 0.1360 * BR_ZLL * BR_HBB
xsecs["ggZH_Hto2B_Zto2Nu_M-125"] = 0.1360 * BR_ZLNU * BR_HBB
xsecs["ggZH_Hto2B_Zto2Q_M-125"] = 0.1360 * BR_ZQQ * BR_HBB
xsecs["ttHto2B_M-125"] = 0.5700 * BR_HBB

# Triple-higgs
# ggHHHto6B_13TeV         0.0894e-3 (not including BR_HBB)
xsecs["HHHTo6B_c3_0_d4_0"] = 3.707e-05 * BR_HBB * BR_HBB * BR_HBB
xsecs["HHHTo6B_c3_0_d4_99"] = 0.005855 * BR_HBB * BR_HBB * BR_HBB
xsecs["HHHTo6B_c3_0_d4_minus1"] = 4.117e-05 * BR_HBB * BR_HBB * BR_HBB
xsecs["HHHTo6B_c3_19_d4_19"] = 0.1476 * BR_HBB * BR_HBB * BR_HBB
# SM sample
xsecs["HHHTo6B_c3_1_d4_0"] = 2.908e-05 * BR_HBB * BR_HBB * BR_HBB
xsecs["HHHTo6B_c3_1_d4_2"] = 1.616e-05 * BR_HBB * BR_HBB * BR_HBB
# xsecs["HHHTo6B_c3_2_d4_minus1"] =
# xsecs["HHHTo6B_c3_4_d4_9"] =
xsecs["HHHTo6B_c3_minus1_d4_0"] = 0.0001127 * BR_HBB * BR_HBB * BR_HBB
xsecs["HHHTo6B_c3_minus1_d4_minus1"] = 0.0001087 * BR_HBB * BR_HBB * BR_HBB
xsecs["HHHTo6B_c3_minus1p5_d4_minus0p5"] = 0.0001941 * BR_HBB * BR_HBB * BR_HBB

# Di-Higgs
# TO- XCHECK (from xsecdb)
# ggF SM from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH#Latest_recommendations_for_gluon
# TODO: find official values for VBF non-SM samples
hh = {
    "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": 0.06648 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-0p00_kt-1p00_c2-1p00_TuneCP5_13p6TeV": 0.1492 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": 34.43e-3 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV": 0.01493 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p35_TuneCP5_13p6TeV": 0.01052 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-3p00_TuneCP5_13p6TeV": 2.802 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-m2p00_TuneCP5_13p6TeV": 1.875 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV": 0.01252 * BR_HBB * BR_HBB,
    "GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": 0.08664 * BR_HBB * BR_HBB,
}
for key, value in hh.items():
    xsecs[key] = value
    xsecs[f"{key}_TSG"] = value
    xsecs[f"{key}_Private"] = value
    xsecs[f"{key}_TSG_Pu60"] = value

# TSG Samples (?)
xsecs["VBFHHto4B_CV-1_C2V-1_C3-1_TuneCP5_13p6TeV_madgraph-pythia8"] = 0.001904 * BR_HBB * BR_HBB
xsecs["VBFHHto4B_CV-1_C2V-1_C3-2_TuneCP5_13p6TeV_madgraph-pythia8"] = 0.001588 * BR_HBB * BR_HBB
xsecs["VBFHHto4B_CV-1_C2V-2_C3-1_TuneCP5_13p6TeV_madgraph-pythia8"] = 0.0156 * BR_HBB * BR_HBB

# Copying the TSG samples ones from xsecdb for now
# xsecs["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"] =
xsecs["VBFHHto4B_CV_1_C2V_1_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"] = 0.001904 * BR_HBB * BR_HBB

# V+Jets
xsecs["Wto2Q-3Jets_HT-200to400"] = 2723.0
xsecs["Wto2Q-3Jets_HT-400to600"] = 299.8
xsecs["Wto2Q-3Jets_HT-600to800"] = 63.9
xsecs["Wto2Q-3Jets_HT-800"] = 31.9
xsecs["WtoLNu-4Jets"] = 55390.0
xsecs["Zto2Q-4Jets_HT-200to400"] = 1082.0
xsecs["Zto2Q-4Jets_HT-400to600"] = 124.1
xsecs["Zto2Q-4Jets_HT-600to800"] = 27.28
xsecs["Zto2Q-4Jets_HT-800"] = 14.57

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

xsecs["WtoLNu-2Jets"] = 64481.58
"""
Cross Sections for 13 TeV.
"""
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
