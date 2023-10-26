from typing import Dict

import numpy as np
from coffea.nanoevents.methods.nanoaod import JetArray, FatJetArray, MuonArray, ElectronArray
from coffea.nanoevents.methods.base import NanoEventsArray
from .corrections import get_jetveto
import awkward as ak

# https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022

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
    "id": "cutBased:4",
}

veto_muon_selection = {
    "pt": 15,
    "eta": 2.4,
    "miniPFRelIso_all": 0.4,
    "id": "loose",
}

veto_electron_selection = {
    "pt": 20,
    "eta": 2.5,
    "miniPFRelIso_all": 0.4,
    "id": "mvaFall17V2noIso_WP90",
}

veto_muon_selection_run2_bbbb = {
    "pt": 10,
    "eta": 2.4,
    "miniPFRelIso_all": 0.15,
    "id": "loose",
    "barrel_dxy": 0.05,
    "barrel_dz": 0.10,
    "endcap_dxy": 0.10,
    "endcap_dz": 0.20,
}

veto_electron_selection_run2_bbbb = {
    "pt": 15,
    "eta": 2.4,
    "miniPFRelIso_all": 0.15,
    "id": "cutBased:4",
    "barrel_dxy": 0.05,
    "barrel_dz": 0.10,
    "endcap_dxy": 0.10,
    "endcap_dz": 0.20,
}

ak4_selection = {
    "eta": 4.7,
    # "pt": 50
}

ak8_selection = {
    "eta": 2.5,
}


def base_muons(muons: MuonArray):
    # base selection of muons
    sel = (muons.pt >= 5) & (abs(muons.eta) <= 2.4)
    return sel


def base_electrons(electrons: ElectronArray):
    # base selection of electrons
    sel = (electrons.pt >= 7) & (abs(electrons.eta) <= 2.5)
    return sel


def good_muons(muons: MuonArray, selection: Dict = muon_selection):
    sel = (
        (muons.pt >= selection["pt"])
        & (abs(muons.eta) <= selection["eta"])
        & (muons.miniPFRelIso_all <= selection["miniPFRelIso_all"])
        & (muons[f"{selection['id']}Id"])
    )
    return sel


def good_electrons(electrons: ElectronArray, selection: Dict = electron_selection):
    if "cutBased" in selection["id"]:
        wp = selection["id"].split(":")[1]
        id_selection = electrons["cutBased"] >= electrons[wp]
    else:
        id_selection = electrons[selection["id"]]
    sel = (
        (electrons.pt >= selection["pt"])
        & (abs(electrons.eta) <= selection["eta"])
        & (electrons.miniPFRelIso_all <= selection["miniPFRelIso_all"])
        & id_selection
    )
    if "barrel_dxy" in selection.keys() and "endcap_dxy" in selection.keys():
        sel = sel & (
            ((abs(electrons.dxy) < selection["barrel_dxy"]) & (abs(electrons.eta) < 1.2))
            | ((abs(electrons.dxy) < selection["endcap_dxy"]) & (abs(electrons.eta) >= 1.2))
        )
    if "barrel_dz" in selection.keys() and "endcap_dz" in selection.keys():
        sel = sel & (
            ((abs(electrons.dz) < selection["barrel_dz"]) & (abs(electrons.eta) < 1.2))
            | ((abs(electrons.dz) < selection["endcap_dz"]) & (abs(electrons.eta) >= 1.2))
        )
    return sel


# ak4 jet definition
def good_ak4jets(
    jets: JetArray, year: str, run: np.ndarray, isData: bool, selection: Dict = ak4_selection
):
    # Since the main AK4 collection for Run3 is the AK4 Puppi collection, jets originating from pileup are already suppressed at the jet clustering level
    # PuID might only be needed for forward region (WIP)
    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    # 2 working points: tight and tightLepVeto
    sel = (jets.isTight) & (abs(jets.eta) < selection["eta"])

    if year == "2022" or year == "2022EE":
        jet_veto = get_jetveto(jets, year, run, isData)
        jet_veto = jet_veto & (jets.pt > 15)
        sel = sel & ~jet_veto
    return sel


# apply ak4 b-jet regression
def bregcorr(jets: JetArray):
    # pt correction for b-jet energy regression
    return ak.zip(
        {
            "pt": jets.pt * jets.bRegCorr,
            "eta": jets.eta,
            "phi": jets.phi,
            "energy": jets.energy * jets.bRegCorr,
        },
        with_name="PtEtaPhiELorentzVector",
    )


# add extra variables to FatJet collection
def get_ak8jets(fatjets: FatJetArray):
    if "particleNetMD_Xbb" in fatjets.fields:
        fatjets["Txbb"] = ak.nan_to_num(
            fatjets.particleNetMD_Xbb / (fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb),
            nan=-1.0,
        )
        fatjets["Txjj"] = ak.nan_to_num(
            (fatjets.particleNetMD_Xbb + fatjets.particleNetMD_Xcc + fatjets.particleNetMD_Xqq)
            / (
                fatjets.particleNetMD_Xbb
                + fatjets.particleNetMD_Xcc
                + fatjets.particleNetMD_Xqq
                + fatjets.particleNetMD_QCD
            ),
            nan=-1.0,
        )
    else:
        fatjets["Txbb"] = fatjets.particleNet_XbbVsQCD
        fatjets["Txjj"] = fatjets.particleNet_XqqVsQCD
        fatjets["particleNet_mass"] = fatjets.particleNet_massCorr
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)

    return fatjets


# ak8 jet definition
def good_ak8jets(fatjets: FatJetArray, selection: Dict = ak8_selection):
    sel = (abs(fatjets.eta) < selection["eta"]) & (fatjets.isTight)
    return sel
