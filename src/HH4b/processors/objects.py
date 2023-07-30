from typing import Dict

import numpy as np
from coffea.nanoevents.methods.nanoaod import JetArray, FatJetArray, MuonArray, ElectronArray
from coffea.nanoevents.methods.base import NanoEventsArray
from .corrections import get_jetveto
import awkward as ak

muon_selection = {
    "pt": 30,
    "eta": 2.4,
    "miniPFRelIso_all": 0.2,
    "id": "tight",
}

electron_selection = {
    "pt": 35,
    "eta": 2.5,
    "miniPFRelIso_all": 0.2,
    # "id": "Tight",
}


ak4_selection = {
    "eta": 4.7,
    # "id": "Tight",
    # "pt": 50
}

ak8_selection = {
    "eta": 2.5,
    # "id": "Tight",
}


def good_muons(muons: MuonArray, selection: Dict = muon_selection):
    sel = (
        (muons.pt >= selection["pt"])
        & (muons.eta <= selection["eta"])
        & (muons.miniPFRelIso_all <= selection["miniPFRelIso_all"])
        & (muons[f"{selection['id']}Id"])
    )
    return muons[sel]


def good_electrons(electrons: ElectronArray, selection: Dict = electron_selection):
    sel = (
        (electrons.pt >= selection["pt"])
        & (electrons.eta <= selection["eta"])
        & (electrons.miniPFRelIso_all <= selection["miniPFRelIso_all"])
        & (electrons.isTight)
    )
    return electrons[sel]


# ak4 jet definition
def good_ak4jets(
    jets: JetArray, year: str, run: np.ndarray, isData: bool, selection: Dict = ak4_selection
):
    # FIXME: check PuID WP and JetIDWP for Run3
    # PuID might only be needed for forward region (WIP)
    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    # 2 working points: tight and tightLepVeto
    goodjets = (
        (jets.isTight)
        # & ((jets.pt < 50) & (jets.puId >=6)) | (jets.pt >=50)
        & (abs(jets.eta) < selection["eta"])
    )
    if year == "2022" and isData:
        jet_veto = get_jetveto(jets, year, run)
        goodjets = goodjets & ~(jet_veto)

    return jets[goodjets]


# add extra variables to FatJet collection
def get_ak8jets(fatjets: FatJetArray):
    fatjets["Txbb"] = ak.nan_to_num(fatjets.particleNetMD_Xbb / (
        fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb
    ), nan=-1.0)
    fatjets["Txjj"] = ak.nan_to_num(
        (fatjets.particleNetMD_Xbb + fatjets.particleNetMD_Xcc + fatjets.particleNetMD_Xqq
     ) / (
         fatjets.particleNetMD_Xbb
         + fatjets.particleNetMD_Xcc
         + fatjets.particleNetMD_Xqq
         + fatjets.particleNetMD_QCD
     )
    , nan=-1.0)
    return fatjets

# ak8 jet definition
def good_ak8jets(fatjets: FatJetArray):
    fatjets = get_ak8jets(fatjets)
    sel = (abs(fatjets.eta) < selection["eta"]) & (fatjets.isTight)
    return fatjets[sel]
