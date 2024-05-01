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
    """
    Definition used in Run 2 boosted VBF Hbb
    https://github.com/rkansal47/HHbbVV/blob/bdcc8c672a0fa0aa2182a9d4d677e07fc3c9a281/src/HHbbVV/processors/bbVVSkimmer.py#L551-L553
    """
    return (
        (muons.pt >= 10) & (abs(muons.eta) <= 2.4) & (muons.looseId) & (muons.pfRelIso04_all < 0.25)
    )


def veto_electrons(electrons: ElectronArray):
    """
    Definition used in Run 2 boosted VBF Hbb
    https://github.com/rkansal47/HHbbVV/blob/bdcc8c672a0fa0aa2182a9d4d677e07fc3c9a281/src/HHbbVV/processors/bbVVSkimmer.py#L542-L547
    """
    return (
        (electrons.pt >= 20)
        & (abs(electrons.eta) <= 2.5)
        & (electrons.miniPFRelIso_all < 0.4)
        & (electrons.cutBased >= electrons.LOOSE)
    )


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
        & (electrons.pfRelIso03_all < 0.15)
    )
    if "mvaIso_WP90" in electrons.fields:
        sel = sel & (electrons.mvaIso_WP90)
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
def good_ak4jets(jets: JetArray, year: str):
    # Since the main AK4 collection for Run3 is the AK4 Puppi collection, jets originating from pileup are already suppressed at the jet clustering level
    # PuID might only be needed for forward region (WIP)

    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    # 2 working points: tight and tightLepVeto
    sel = (jets.pt > 15) & (jets.isTight) & (abs(jets.eta) < 4.7)

    if year == "2018":
        pu_id = sel & ((jets.pt >= 50) | (jets.puId >= 6))
        sel = sel & pu_id

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
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)

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
    elif "ParticleNetMD_probXbb" in fatjets_fields:
        fatjets["Tqcd"] = (
            fatjets.ParticleNetMD_probQCDb
            + fatjets.ParticleNetMD_probQCDbb
            + fatjets.ParticleNetMD_probQCDc
            + fatjets.ParticleNetMD_probQCDcc
            + fatjets.ParticleNetMD_probQCDothers
        )
        fatjets["Txbb"] = fatjets.ParticleNetMD_probXbb / (
            fatjets.ParticleNetMD_probXbb + fatjets["Tqcd"]
        )
        fatjets["Pxjj"] = (
            fatjets.ParticleNetMD_probXbb
            + fatjets.ParticleNetMD_probXcc
            + fatjets.ParticleNetMD_probXqq
        )
        fatjets["Txjj"] = fatjets["Pxjj"] / (fatjets["Pxjj"] + fatjets["Tqcd"])
    else:
        fatjets["Txbb"] = fatjets.particleNet_XbbVsQCD
        fatjets["Txjj"] = fatjets.particleNet_XqqVsQCD
        fatjets["Tqcd"] = fatjets.particleNet_QCD

    if "particleNet_mass" not in fatjets_fields:
        fatjets["particleNet_mass"] = fatjets.mass

    if "particleNet_massCorr" in fatjets_fields:
        fatjets["particleNet_mass"] = fatjets.mass * fatjets.particleNet_massCorr
        fatjets["particleNet_massraw"] = (
            (1 - fatjets.rawFactor) * fatjets.mass * fatjets.particleNet_massCorr
        )
    else:
        if "particleNet_mass" not in fatjets_fields:
            fatjets["particleNet_mass"] = fatjets.mass
            fatjets["particleNet_massraw"] = fatjets.mass
        else:
            fatjets["particleNet_massraw"] = fatjets.particleNet_mass

    if "ParticleNetMD_probQCDb" in fatjets_fields:
        fatjets["PQCDb"] = fatjets.ParticleNetMD_probQCDb
        fatjets["PQCDbb"] = fatjets.ParticleNetMD_probQCDbb
        fatjets["PQCDothers"] = fatjets.ParticleNetMD_probQCDothers
    elif "particleNet_QCD1HF" in fatjets_fields:
        fatjets["PQCDb"] = fatjets.particleNet_QCD1HF
        fatjets["PQCDbb"] = fatjets.particleNet_QCD2HF
        fatjets["PQCDothers"] = fatjets.particleNet_QCD0HF
    else:
        # dummy
        fatjets["PQCDb"] = fatjets.particleNetMD_QCD
        fatjets["PQCDbb"] = fatjets.particleNetMD_QCD
        fatjets["PQCDothers"] = fatjets.particleNetMD_QCD

    if "particleNetLegacy_Xbb" in fatjets_fields:
        fatjets["TXbb_legacy"] = fatjets.particleNetLegacy_Xbb / (
            fatjets.particleNetLegacy_Xbb + fatjets.particleNetLegacy_QCD
        )
        fatjets["PXbb_legacy"] = fatjets.particleNetLegacy_Xbb
        fatjets["PQCD_legacy"] = fatjets.particleNetLegacy_QCD
        fatjets["PQCDb_legacy"] = fatjets.particleNetLegacy_QCDb
        fatjets["PQCDbb_legacy"] = fatjets.particleNetLegacy_QCDbb
        fatjets["PQCDothers_legacy"] = fatjets.particleNetLegacy_QCDothers
    else:
        fatjets["TXbb_legacy"] = fatjets["Txbb"]

    if "particleNetLegacy_mass" in fatjets_fields:
        fatjets["particleNet_mass_legacy"] = fatjets.particleNetLegacy_mass
    else:
        fatjets["particleNet_mass_legacy"] = fatjets["particleNet_mass"]

    if "particleNetWithMass_TvsQCD" in fatjets_fields:
        fatjets["particleNetWithMass_TvsQCD"] = fatjets.particleNetWithMass_TvsQCD

    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt

    return fatjets


# ak8 jet definition
def good_ak8jets(fatjets: FatJetArray, pt: float, eta: float, msd: float, mreg: float):
    fatjets_fields = fatjets.fields
    legacy = "particleNetLegacy_mass" in fatjets_fields
    mreg_val = fatjets["particleNet_mass_legacy"] if legacy else fatjets["particleNet_mass"]

    fatjet_sel = (
        fatjets.isTight
        & (fatjets.pt > pt)
        & (abs(fatjets.eta) < eta)
        & ((fatjets.msoftdrop > msd) | (mreg_val > mreg))
    )
    return fatjets[fatjet_sel]


def vbf_jets(
    jets: JetArray,
    fatjets: FatJetArray,
    events,
    pt: float,
    id: str,  # noqa: ARG001
    eta_max: float,
    dr_fatjets: float,
    dr_leptons: float,
    electron_pt: float,
    muon_pt: float,
):
    """Top 2 jets in pT passing the VBF selections"""
    electrons = events.Electron
    electrons = electrons[electrons.pt > electron_pt]

    muons = events.Muon
    muons = muons[muons.pt > muon_pt]

    ak4_sel = (
        jets.isTight
        & (jets.pt >= pt)
        & (np.abs(jets.eta) <= eta_max)
        & (ak.all(jets.metric_table(fatjets) > dr_fatjets, axis=2))
        & ak.all(jets.metric_table(electrons) > dr_leptons, axis=2)
        & ak.all(jets.metric_table(muons) > dr_leptons, axis=2)
    )

    return jets[ak4_sel][:, :2]


def ak4_jets_awayfromak8(
    jets: JetArray,
    fatjets: FatJetArray,
    events,
    pt: float,
    id: str,  # noqa: ARG001
    eta_max: float,
    dr_fatjets: float,
    dr_leptons: float,
    electron_pt: float,
    muon_pt: float,
):
    """Top 2 jets in b-tag away from AK8 fatjets"""
    electrons = events.Electron
    electrons = electrons[electrons.pt > electron_pt]

    muons = events.Muon
    muons = muons[muons.pt > muon_pt]

    ak4_sel = (
        jets.isTight
        & (jets.pt >= pt)
        & (np.abs(jets.eta) <= eta_max)
        & (ak.all(jets.metric_table(fatjets) > dr_fatjets, axis=2))
        & ak.all(jets.metric_table(electrons) > dr_leptons, axis=2)
        & ak.all(jets.metric_table(muons) > dr_leptons, axis=2)
    )

    # sort by btagPNetB
    jets_pnetb = jets[ak.argsort(jets.btagPNetB, ascending=False)]

    return jets_pnetb[ak4_sel][:, :2]
