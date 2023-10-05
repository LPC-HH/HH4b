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
xsecs["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"] = 34.43e-3 * BR_HBB * BR_HBB

# V+Jets
xsecs["Wto2Q-3Jets_HT-200to400"] = 2723.0
xsecs["Wto2Q-3Jets_HT-400to600"] = 299.8
xsecs["Wto2Q-3Jets_HT-600to800"] = 63.9
xsecs["Wto2Q-3Jets_HT-800"] = 31.9
xsecs["Zto2Q-4Jets_HT-200to400"] = 1082.0
xsecs["Zto2Q-4Jets_HT-400to600"] = 124.1
xsecs["Zto2Q-4Jets_HT-600to800"] = 27.28
xsecs["Zto2Q-4Jets_HT-800"] = 14.57
