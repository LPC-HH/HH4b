"""
Cross Sections for 13.6 TeV,
"""

BR_WQQ = 0.676
BR_WLNU = 0.324
BR_ZQQ = 0.69911
BR_ZLNU = 0.27107
BR_ZLL = 0.02982

BR_HBB = 0.5809
# BR_HBB = 0.5824

xsecs = {}

# QCD
xsecs["QCD_PT-80to120"] = 2534000.0
xsecs["QCD_PT-120to170"] = 441100.0
xsecs["QCD_PT-170to300"] = 113400.0
xsecs["QCD_PT-300to470"] = 7707.0
xsecs["QCD_PT-470to600"] = 628.8
xsecs["QCD_PT-600to800"] = 178.6
xsecs["QCD_PT-800to1000"] = 30.57
xsecs["QCD_PT-1000to1400"] = 8.92
xsecs["QCD_PT-1400to1800"] = 0.8103
xsecs["QCD_PT-1800to2400"] = 0.1148
xsecs["QCD_PT-2400to3200"] = 0.007542
xsecs["QCD_PT-3200"] = 0.0002331

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

# Top
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
xsecs["TTto4Q"] = 923.6 * 0.667 * 0.667
xsecs["TTto2L2Nu"] = 923.6 * 0.333 * 0.333
xsecs["TTtoLNu2Q"] = 923.6 * 2 * (0.667 * 0.333)

# Diboson
xsecs["WW"] = 80.23
xsecs["WZ"] = 29.1
xsecs["ZZ"] = 12.75

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

# Di-Higgs
# TO- XCHECK
xsecs["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"] = 34.43e-3 * BR_HBB * BR_HBB  # 0.02964 in xsecdb
xsecs["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG"] = 34.43e-3 * BR_HBB * BR_HBB
xsecs["GluGlutoHHto4B_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG"] = (
    0.06648 * BR_HBB * BR_HBB
)  # from xsecdb
xsecs["GluGlutoHHto4B_kl-2p45_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG"] = 0.01252 * BR_HBB * BR_HBB
xsecs["GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG"] = 0.08664 * BR_HBB * BR_HBB

# V+Jets
xsecs["Wto2Q-3Jets_HT-200to400"] = 2723.0
xsecs["Wto2Q-3Jets_HT-400to600"] = 299.8
xsecs["Wto2Q-3Jets_HT-600to800"] = 63.9
xsecs["Wto2Q-3Jets_HT-800"] = 31.9
xsecs["Zto2Q-4Jets_HT-200to400"] = 1082.0
xsecs["Zto2Q-4Jets_HT-400to600"] = 124.1
xsecs["Zto2Q-4Jets_HT-600to800"] = 27.28
xsecs["Zto2Q-4Jets_HT-800"] = 14.57

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
