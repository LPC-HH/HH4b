""" 
Cross Sections for 13.6 TeV in pb
"""
import json

BR_HBB = 0.5824

xs = {}

# QCD
xs["QCD_PT-80to120"] = 2534000.0
xs["QCD_PT-120to170"] = 445800.0
xs["QCD_PT-170to300"] = 113700.0
xs["QCD_PT-300to470"] = 7589.0
xs["QCD_PT-470to600"] = 626.4
xs["QCD_PT-600to800"] = 178.6
xs["QCD_PT-800to1000"] = 30.57
xs["QCD_PT-1000to1400"] = 8.92
xs["QCD_PT-1400to1800"] = 0.8103
xs["QCD_PT-1800to2400"] = 0.1148
xs["QCD_PT-2400to3200"] = 0.007542
xs["QCD_PT-3200"] = 0.0002331

# Top
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
xs["TTto4Q"] = 923.6 * 0.667 * 0.667
# xs["TTto2L2Nu"] =
xs["TTtoLNu2Q"] = 923.6 * 2 * (0.667 * 0.333)

# Diboson
xs["WW"] = 80.23
xs["WZ"] = 29.1
xs["ZZ"] = 12.75

# Higgs
# SX: took XSDB NLO number (0.5246) and multiplied it by the NNLO/NLO ratio for inclusive ggH from 13 TeV
xs["GluGluHto2B_PT-200_M-125"] = 0.5246 * (43.92 / 27.8) * BR_HBB
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap
xs["VBFHto2B_M-125_dipoleRecoilOn"] = 4.078 * BR_HBB
xs["WminusH_Hto2B_Wto2Q_M-125"] = 0.8889 * 0.676 * BR_HBB
# xs["WminusH_Hto2B_WtoLNu_M-125"] =
xs["WplusH_Hto2B_Wto2Q_M-125"] = 0.5677 * 0.676 * BR_HBB
# xs["ZH_Hto2B_Zto2L_M-125"] =
xs["ZH_Hto2B_Zto2Q_M-125"] = 0.8079 * 0.69911 * BR_HBB
# xs["ggZH_Hto2B_Zto2L_M-125"] =
# xs["ggZH_Hto2B_Zto2Nu_M-125"] =
xs["ggZH_Hto2B_Zto2Q_M-125"] = 0.1360 * 0.69911 * BR_HBB
xs["ttHto2B_M-125"] = 0.5700 * BR_HBB

# Di-Higgs
xs["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"] = 34.43e-3 * 5.824e-01 * 5.824e-01

with open("xsecs.json", "w") as outfile:
    json.dump(xs, outfile, indent=4)
