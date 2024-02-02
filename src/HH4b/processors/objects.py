from __future__ import annotations

import awkward as ak
import numpy as np
from coffea.nanoevents.methods.nanoaod import (
    ElectronArray,
    FatJetArray,
    JetArray,
    MuonArray,
    TauArray,
)

# https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022


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
    return (
        (muons.pt >= 10)
        & (abs(muons.eta) <= 2.4)
        & (muons.looseId)
        & (muons.pfRelIso04_all < 0.15)
        & (
            ((abs(muons.dxy) < 0.05) & (abs(muons.dz) < 0.10) & (abs(muons.eta) < 1.2))
            | ((abs(muons.dxy) < 0.10) & (abs(muons.dz) < 0.20) & (abs(muons.eta) >= 1.2))
        )
    )


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


def good_muons(muons: MuonArray):
    sel = (
        (muons.pt >= 30)
        & (np.abs(muons.eta) < 2.4)
        & (np.abs(muons.dz) < 0.1)
        & (np.abs(muons.dxy) < 0.05)
        & (muons.sip3d <= 4.0)
        & muons.mediumId
        & (muons.pfRelIso04_all < 0.15)
    )
    return sel


def good_electrons(electrons: ElectronArray):
    sel = (
        (electrons.pt >= 30)
        & (abs(electrons.eta) <= 2.4)
        & (np.abs(electrons.dz) < 0.1)
        & (np.abs(electrons.dxy) < 0.05)
        & (electrons.mvaIso_WP90)
        & (electrons.pfRelIso03_all < 0.15)
    )
    return sel


def loose_taus(taus: TauArray):
    sel = (
        (taus.pt > 20)
        & (abs(taus.eta) <= 2.5)
        & (taus.idDeepTau2017v2p1VSe >= 2)
        & (taus.idDeepTau2017v2p1VSmu >= 2)
        & (taus.idDeepTau2017v2p1VSjet >= 8)
    )
    return sel


# ak4 jet definition
def good_ak4jets(jets: JetArray, year: str, run: np.ndarray, isData: bool):
    # Since the main AK4 collection for Run3 is the AK4 Puppi collection, jets originating from pileup are already suppressed at the jet clustering level
    # PuID might only be needed for forward region (WIP)

    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    # 2 working points: tight and tightLepVeto
    sel = (jets.isTight) & (abs(jets.eta) < 4.7)

    if year == "2018":
        pu_id = sel & ((jets.pt >= 50) | (jets.puId >= 6))
        sel = sel & pu_id

    if year == "2022" or year == "2022EE":
        from .corrections import get_jetveto

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
    fatjets_fields = fatjets.fields
    if "particleNetMD_Xbb" in fatjets_fields:
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
        fatjets["Tqcd"] = fatjets.particleNetMD_QCD
    else:
        fatjets["Txbb"] = fatjets.particleNet_XbbVsQCD
        fatjets["Txjj"] = fatjets.particleNet_XqqVsQCD
        # save both until we confirm which is correct
        fatjets["particleNet_mass"] = fatjets.mass * fatjets.particleNet_massCorr
        fatjets["particleNet_massraw"] = (
            (1 - fatjets.rawFactor) * fatjets.mass * fatjets.particleNet_massCorr
        )
        # dummy
        fatjets["Tqcd"] = fatjets.particleNet_XbbVsQCD

    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)

    if "ParticleNetMD_probQCDb" in fatjets_fields:
        fatjets["TQCDb"] = fatjets.ParticleNetMD_probQCDb
        fatjets["TQCDbb"] = fatjets.ParticleNetMD_probQCDbb
        fatjets["TQCDothers"] = fatjets.ParticleNetMD_probQCDothers
    else:
        # dummy
        fatjets["TQCDb"] = fatjets["Txbb"]
        fatjets["TQCDbb"] = fatjets["Txbb"]
        fatjets["TQCDothers"] = fatjets["Txbb"]

    return fatjets


# ak8 jet definition
def good_ak8jets(fatjets: FatJetArray):
    return (abs(fatjets.eta) < 2.5) & (fatjets.isTight)
