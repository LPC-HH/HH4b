from __future__ import annotations

import awkward as ak
import numpy as np
from coffea.nanoevents.methods.nanoaod import (
    ElectronArray,
    FatJetArray,
    JetArray,
    MuonArray,
)

from .corrections import get_jetveto

# https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022


def base_muons(muons: MuonArray):
    # base selection of muons
    return (muons.pt >= 5) & (abs(muons.eta) <= 2.4)


def base_electrons(electrons: ElectronArray):
    # base selection of electrons
    return (electrons.pt >= 7) & (abs(electrons.eta) <= 2.5)


def veto_muons_run2(muons: MuonArray):
    return (
        (muons.pt >= 30)
        & (abs(muons.eta) <= 2.4)
        & (muons.tightId)
        & (muons.miniPFRelIso_all <= 0.2)
    )


def veto_electrons_run2(electrons: ElectronArray):
    return (
        (electrons.pt >= 35)
        & (abs(electrons.eta) <= 2.5)
        & (electrons.cutBased > 3)
        & (electrons.miniPFRelIso_all <= 0.2)
    )


def veto_muons(muons: MuonArray):
    sel = (
        (muons.pt >= 10) & (abs(muons.eta) <= 2.4) & (muons.looseId) & (muons.pfRelIso04_all < 0.15)
    )
    sel = sel & (
        ((abs(muons.dxy) < 0.05) & (abs(muons.dz) < 0.10) & (abs(muons.eta) < 1.2))
        | ((abs(muons.dxy) < 0.10) & (abs(muons.dz) < 0.20) & (abs(muons.eta) >= 1.2))
    )
    return sel


def veto_electrons(electrons: ElectronArray):
    sel = (
        (electrons.pt >= 15)
        & (abs(electrons.eta) <= 2.4)
        & (electrons.mvaIso_WP90)
        & (electrons.pfRelIso03_all < 0.15)
    )
    sel = sel & (
        ((abs(electrons.dxy) < 0.05) & (abs(electrons.dz) < 0.10) & (abs(electrons.eta) < 1.2))
        | ((abs(electrons.dxy) < 0.10) & (abs(electrons.dz) < 0.20) & (abs(electrons.eta) >= 1.2))
    )
    return sel


# ak4 jet definition
def good_ak4jets(jets: JetArray, year: str, run: np.ndarray, isData: bool):
    # Since the main AK4 collection for Run3 is the AK4 Puppi collection, jets originating from pileup are already suppressed at the jet clustering level
    # PuID might only be needed for forward region (WIP)
    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    # 2 working points: tight and tightLepVeto
    sel = (jets.isTight) & (abs(jets.eta) < 4.7)

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
        # fatjets["particleNet_mass"] = fatjets.mass * fatjets.particleNet_massCorr
        fatjets["particleNet_mass"] = (
            (1 - fatjets.rawFactor) * fatjets.mass * fatjets.particleNet_massCorr
        )

    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)

    return fatjets


# ak8 jet definition
def good_ak8jets(fatjets: FatJetArray):
    return (abs(fatjets.eta) < 2.5) & (fatjets.isTight)
