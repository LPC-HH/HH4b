"""
Cross Sections for 13.6 TeV,
"""

BR_HBB = 0.5809


xsecs = {
    # From XSDB
    # "QCD_Pt_5to10_TuneCUETP8M1_13TeV_pythia8":
    # "QCD_Pt_10to15_TuneCUETP8M1_13TeV_pythia8":
    # "QCD_Pt_15to30_TuneCUETP8M1_13TeV_pythia8":
    # "QCD_Pt_30to50_TuneCUETP8M1_13TeV_pythia8":
    # "QCD_Pt_50to80_TuneCUETP8M1_13TeV_pythia8":
    "QCD_Pt_80to120_TuneCUETP8M1_13TeV_pythia8": 2534000.0,
    "QCD_Pt_120to170_TuneCUETP8M1_13TeV_pythia8": 445800.0,
    "QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8": 113700.0,
    "QCD_Pt_300to470_TuneCUETP8M1_13TeV_pythia8": 7589.0,
    "QCD_Pt_470to600_TuneCUETP8M1_13TeV_pythia8": 626.4,
    "QCD_Pt_600to800_TuneCUETP8M1_13TeV_pythia8": 178.6,
    "QCD_Pt_800to1000_TuneCUETP8M1_13TeV_pythia8": 30.57,
    "QCD_Pt_1000to1400_TuneCUETP8M1_13TeV_pythia8": 8.92,
    "QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8": 0.8103,
    "QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8": 0.1148,
    "QCD_Pt_2400to3200_TuneCUETP8M1_13TeV_pythia8": 0.007542,
    "QCD_Pt_3200toInf_TuneCUETP8M1_13TeV_pythia8": 0.0002331,
    # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
    "TTto4Q": 923.6 * 0.667 * 0.667,
    "TTtoLNu2Q": 923.6 * 2 * (0.667 * 0.333),
    "WW": 80.23,
    "WZ": 29.1,
    "ZZ": 12.75,
    # SX: took XSDB NLO number (0.5246) and multiplied it by the NNLO/NLO ratio for inclusive ggH from 13 TeV
    "ggHto2B_Pt200ToInf": 0.5246 * (43.92 / 27.8) * 5.824e-01,
    # from here: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap,
    "VBFHto2B": 4.078 * 5.824e-01,
    "WminusH_Hto2B_Wto2Q": 0.8889 * 5.824e-01 * 0.676,
    "WplusH_Hto2B_Wto2Q": 0.5677 * 5.824e-01 * 0.676,
    "ZH_Hto2B_Zto2Q": 0.8079 * 5.824e-01 * 0.69911,
    "ggZH_Hto2B_Zto2Q": 0.1360 * 5.824e-01 * 0.69911,
    "ttH_Hto2B": 0.5700 * 5.824e-01,
    "ggHH_cHHH_1_TSG": 34.43e-3 * 5.824e-01 * 5.824e-01,
}
