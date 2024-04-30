# =================================
# Plotting binning, labels, etc
# =================================
from __future__ import annotations

bins = {}
bins["trgSF_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_central"] = [
    i / 100 for i in range(0, 205, 5)
]
bins["btagSF_central"] = [i / 100 for i in range(0, 205, 5)]
bins["passjetvetomap"] = [0, 1, 2]

# bins["Jet1Pt"] = [i for i in range(0, 605, 5)]
# bins["Jet2Pt"] = [i for i in range(0, 605, 5)]
# bins["Jet3Pt"] = [i for i in range(0, 405, 5)]
# bins["Jet4Pt"] = [i for i in range(0, 405, 5)]
# bins["Jet4InBDisc"] = [i/100 for i in range(0, 102, 2)]
bins["avgbdisc_twoldgbdiscjets"] = [i / 100 for i in range(0, 102, 2)]

bins["dHiggsDeltaRegMass"] = [i for i in range(0, 405, 5)]

bins["alljets_ht"] = [i for i in range(200, 2550, 50)]

bins["dHH_H1_pt"] = [i for i in range(40, 820, 20)]
bins["dHH_H1_regpt"] = [i for i in range(40, 820, 20)]
bins["dHH_H2_pt"] = [i for i in range(40, 820, 20)]
bins["dHH_H2_regpt"] = [i for i in range(40, 820, 20)]

bins["dHH_H1_regmass"] = [i for i in range(0, 405, 5)]
bins["dHH_H2_regmass"] = [i for i in range(0, 405, 5)]
bins["dHH_HH_mass"] = [i for i in range(200, 2050, 50)]
bins["dHH_HH_regmass"] = [i for i in range(200, 2050, 50)]

labels = {}

labels["Jet4InBDisc"] = ";4th ldg-in-bdisc jet b-disc;Events"
labels["reco_ttbar_chi2"] = ";t#bar{t} #chi^{2};Events"
labels["reco_top_0_mass"] = ";m_{top}^{1} [GeV];Events"
labels["reco_top_1_mass"] = ";m_{top}^{2} [GeV];Events"
labels["reco_top_0_chi2"] = ";#chi^{2} top_{1};Events"
labels["reco_top_1_chi2"] = ";#chi^{2} top_{2};Events"

labels["nTopCandidates"] = ";Reconstructed tops;Events"
labels["dHiggsDeltaRegMass"] = ";#Delta m_{H};Events"
labels["alljets_ht"] = ";PF H_{T} [GeV];Events"
labels["Jet1Pt"] = ";Leading-in-p_{T} jet p_{T};Events"
labels["Jet2Pt"] = ";Subldg-in-p_{T} jet p_{T};Events"
labels["Jet3Pt"] = ";Third ldg-in-p_{T} jet p_{T};Events"
labels["Jet4Pt"] = ";Forth ldg-in-p_{T} jet p_{T};Events"
labels["avgbdisc_twoldgbdiscjets"] = ";Mean PNet;Events"

labels["passjetvetomap"] = ";Passes jet-veto-map;Events"
labels["btagSF_central"] = ";b-tagging SF;Events"
labels["trgSF_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_central"] = (
    ";trigger SF;Events"
)

labels["genp_H1_pt"] = ";True H_{1} p_{T} [GeV];Events"
labels["genp_H2_pt"] = ";True H_{2} p_{T} [GeV];Events"
labels["genp_H1b1_pt"] = ";True H_{1} b_{1} p_{T} [GeV];Events"
labels["genp_H2b1_pt"] = ";True H_{2} b_{1} p_{T} [GeV];Events"
labels["genp_H1b2_pt"] = ";True H_{1} b_{2} p_{T} [GeV];Events"
labels["genp_H2b2_pt"] = ";True H_{2} b_{2} p_{T} [GeV];Events"

labels["dHH_H1_pt"] = ";Leading Higgs p_{T} [GeV];Events"
labels["dHH_H1_regpt"] = ";Leading Higgs p_{T}^{reg.} [GeV];Events"
labels["dHH_H2_pt"] = ";Subleading Higgs p_{T} [GeV];Events"
labels["dHH_H2_regpt"] = ";Subleading Higgs p_{T}^{reg.} [GeV];Events"
labels["dHH_H1_mass"] = ";Leading Higgs mass [GeV];Events"
labels["dHH_H1_regmass"] = ";Leading Higgs mass^{reg.} [GeV];Events"
labels["dHH_H2_mass"] = ";Subleading Higgs mass [GeV];Events"
labels["dHH_H2_regmass"] = ";Subleading Higgs mass^{reg.} [GeV];Events"
labels["dHH_HH_mass"] = ";m_{HH} [GeV];Events"
labels["dHH_HH_regmass"] = ";m_{HH}^{reg.} [GeV];Events"
