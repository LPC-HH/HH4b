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


def jetid_v12(jets: ak.Array) -> tuple[ak.Array, ak.Array]:
    """
    Jet ID fix for NanoAOD v12 copying
    # https://gitlab.cern.ch/cms-jetmet/coordination/coordination/-/issues/117#note_8880716
    """

    jetidtightbit = jets.isTight
    jetidtight = (
        ((np.abs(jets.eta) <= 2.7) & jetidtightbit)
        | (
            ((np.abs(jets.eta) > 2.7) & (np.abs(jets.eta) <= 3.0))
            & jetidtightbit
            & (jets.neHEF < 0.99)
        )
        | ((np.abs(jets.eta) > 3.0) & jetidtightbit & (jets.neEmEF < 0.4))
    )

    jetidtightlepveto = (
        (np.abs(jets.eta) <= 2.7) & jetidtight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8)
    ) | ((np.abs(jets.eta) > 2.7) & jetidtight)

    return jetidtight, jetidtightlepveto


def jetid_v14(jets: ak.Array) -> tuple[ak.Array, ak.Array]:
    """
    Jet ID fix for NanoAOD v14 copying
    # https://gitlab.cern.ch/cms-jetmet/coordination/coordination/-/issues/117#note_8880788
    """

    jetidtight = (
        (
            (np.abs(jets.eta) <= 2.6)
            & (jets.neHEF < 0.99)
            & (jets.neEmEF < 0.9)
            & ((jets.chMultiplicity + jets.neMultiplicity) > 1)
            & (jets.chHEF > 0.01)
            & (jets.chMultiplicity > 0)
        )
        | (
            ((np.abs(jets.eta) > 2.6) & (np.abs(jets.eta) <= 2.7))
            & (jets.neHEF < 0.90)
            & (jets.neEmEF < 0.99)
        )
        | (((np.abs(jets.eta) > 2.7) & (np.abs(jets.eta) <= 3.0)) & (jets.neHEF < 0.99))
        | ((np.abs(jets.eta) > 3.0) & (jets.neMultiplicity >= 2) & (jets.neEmEF < 0.4))
    )

    jetidtightlepveto = (
        (np.abs(jets.eta) <= 2.7) & jetidtight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8)
    ) | ((np.abs(jets.eta) > 2.7) & jetidtight)

    return jetidtight, jetidtightlepveto


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


def veto_taus(taus: TauArray):
    # https://github.com/jeffkrupa/zprime-bamboo/blob/main/zprlegacy.py#L371
    return (
        (taus.pt > 20)
        & (taus.decayMode >= 0)
        & (taus.decayMode != 5)
        & (taus.decayMode != 6)
        & (taus.decayMode != 7)
        & (abs(taus.eta) < 2.3)
        & (abs(taus.dz) < 0.2)
        & (taus.idDeepTau2017v2p1VSe >= 2)
        & (taus.idDeepTau2017v2p1VSmu >= 8)
        & (taus.idDeepTau2017v2p1VSjet >= 16)
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


def ZbbSF_global_highPt_muons(muons: MuonArray, pt_min: float = 30.0):
    sel = (
        (muons.pt >= pt_min)
        & (np.abs(muons.eta) < 2.4)
        & (muons.isGlobal != 0)
        & (muons.highPtId != 0)
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
def good_ak4jets(jets: JetArray, year: str, nano_version: str):
    # Since the main AK4 collection for Run3 is the AK4 Puppi collection, jets originating from pileup are already suppressed at the jet clustering level
    # PuID might only be needed for forward region (WIP)

    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    # 2 working points: tight and tightLepVeto
    sel = (jets.pt > 15) & (abs(jets.eta) < 4.7)

    if year == "2018":
        pu_id = sel & ((jets.pt >= 50) | (jets.puId >= 6))
        sel = sel & pu_id
    else:
        if nano_version.startswith("v12") or "25v2" in nano_version:
            jetidtight, jetidtightlepveto = jetid_v12(jets)  # v12 jetid fix
        elif nano_version.startswith("v14"):
            jetidtight = jets.isTight
            jetidtightlepveto = jets.isTightLeptonVeto
        elif nano_version.startswith(("v13", "v15")):
            raise NotImplementedError("Jet ID fix for NanoAOD v13, v14, v15 not implemented yet!")
        else:
            jetidtight, jetidtightlepveto = jets.isTight, jets.isTightLepVeto

        sel = sel & jetidtight & jetidtightlepveto

    return jets[sel]


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


def get_ak8jets(fatjets: FatJetArray):
    """
    Add extra variables to FatJet collection (mostly renaming)
    """
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)
    fatjets["t21"] = ak.nan_to_num(fatjets.tau2 / fatjets.tau1, nan=-1.0)
    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt

    fatjets_fields = fatjets.fields

    # ParticleNet and ParT bb tagging

    # discriminators
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
        fatjets["Txjj"] = fatjets["Pxjj"] / (fatjets["Pxjj"] + fatjets["Tqcd"])
    else:
        fatjets["Txbb"] = fatjets.particleNet_XbbVsQCD
        fatjets["Txjj"] = fatjets.particleNet_XqqVsQCD
        fatjets["Tqcd"] = fatjets.particleNet_QCD

    if "particleNetWithMass_TvsQCD" in fatjets_fields:
        fatjets["particleNetWithMass_TvsQCD"] = fatjets.particleNetWithMass_TvsQCD

    # "legacy version"
    if "particleNetLegacy_Xbb" in fatjets_fields:
        fatjets["TXbb_legacy"] = fatjets.particleNetLegacy_Xbb / (
            fatjets.particleNetLegacy_Xbb + fatjets.particleNetLegacy_QCD
        )
        fatjets["TXqq_legacy"] = fatjets.particleNetLegacy_Xqq / (
            fatjets.particleNetLegacy_Xqq + fatjets.particleNetLegacy_QCD
        )
        fatjets["PXbb_legacy"] = fatjets.particleNetLegacy_Xbb
        fatjets["PQCD_legacy"] = fatjets.particleNetLegacy_QCD
        fatjets["PQCDb_legacy"] = fatjets.particleNetLegacy_QCDb
        fatjets["PQCDbb_legacy"] = fatjets.particleNetLegacy_QCDbb
        fatjets["PQCD0HF_legacy"] = fatjets.particleNetLegacy_QCDothers
        if "particleNetLegacy_QCDc" in fatjets_fields:
            fatjets["PQCD1HF_legacy"] = (
                fatjets.particleNetLegacy_QCDb + fatjets.particleNetLegacy_QCDc
            )
            fatjets["PQCD2HF_legacy"] = (
                fatjets.particleNetLegacy_QCDbb + fatjets.particleNetLegacy_QCDcc
            )
        else:
            fatjets["PQCD1HF_legacy"] = fatjets.particleNetLegacy_QCDb
            fatjets["PQCD2HF_legacy"] = fatjets.particleNetLegacy_QCDbb

    # mass regression
    fatjets["particleNet_mass"] = fatjets.mass
    fatjets["particleNet_massraw"] = fatjets.mass
    if "particleNet_massCorr" in fatjets_fields:
        fatjets["particleNet_mass"] = fatjets.mass * fatjets.particleNet_massCorr
        fatjets["particleNet_massraw"] = (
            (1 - fatjets.rawFactor) * fatjets.mass * fatjets.particleNet_massCorr
        )
    if "particleNet_mass" in fatjets_fields:
        fatjets["particleNet_massraw"] = fatjets.particleNet_mass

    fatjets["particleNet_mass_legacy"] = fatjets["particleNet_mass"]
    if "particleNetLegacy_mass" in fatjets_fields:
        fatjets["particleNet_mass_legacy"] = fatjets.particleNetLegacy_mass

    # individual probabilities
    if "ParticleNetMD_probQCDb" in fatjets_fields:
        fatjets["PQCD1HF"] = fatjets.ParticleNetMD_probQCDb
        fatjets["PQCD2HF"] = fatjets.ParticleNetMD_probQCDbb
        fatjets["PQCD0HF"] = fatjets.ParticleNetMD_probQCDothers
    elif "particleNet_QCD1HF" in fatjets_fields:
        fatjets["PQCD1HF"] = fatjets.particleNet_QCD1HF
        fatjets["PQCD2HF"] = fatjets.particleNet_QCD2HF
        fatjets["PQCD0HF"] = fatjets.particleNet_QCD0HF
    else:
        # dummy
        fatjets["PQCD1HF"] = fatjets.particleNetMD_QCD
        fatjets["PQCD2HF"] = fatjets.particleNetMD_QCD
        fatjets["PQCD0HF"] = fatjets.particleNetMD_QCD

    if "globalParT_Xbb" in fatjets_fields:
        # P for individual probabilities
        fatjets["ParTPQCD1HF"] = fatjets.globalParT_QCD1HF
        fatjets["ParTPQCD2HF"] = fatjets.globalParT_QCD2HF
        fatjets["ParTPQCD0HF"] = fatjets.globalParT_QCD0HF
        fatjets["ParTPTopW"] = fatjets.globalParT_TopW
        fatjets["ParTPTopbW"] = fatjets.globalParT_TopbW
        fatjets["ParTPTopbWev"] = fatjets.globalParT_TopbWev
        fatjets["ParTPTopbWmv"] = fatjets.globalParT_TopbWmv
        fatjets["ParTPTopbWtauhv"] = fatjets.globalParT_TopbWtauhv
        fatjets["ParTPTopbbWq"] = fatjets.globalParT_TopbWq
        fatjets["ParTPTopbbWqq"] = fatjets.globalParT_TopbWqq
        fatjets["ParTPXbb"] = fatjets.globalParT_Xbb
        fatjets["ParTPXcc"] = fatjets.globalParT_Xcc
        fatjets["ParTPXcs"] = fatjets.globalParT_Xcs
        fatjets["ParTPXgg"] = fatjets.globalParT_Xgg
        fatjets["ParTPXqq"] = fatjets.globalParT_Xqq
        fatjets["ParTPXtauhtaue"] = fatjets.globalParT_Xtauhtaue
        fatjets["ParTPXtauhtauh"] = fatjets.globalParT_Xtauhtauh
        fatjets["ParTPXtauhtaum"] = fatjets.globalParT_Xtauhtaum
        # T for discriminator
        fatjets["ParTTXbb"] = fatjets.globalParT_XbbVsQCD
        # Mass Regression (Raw)
        fatjets["ParTmassRes"] = fatjets.globalParT_massRes * (1 - fatjets.rawFactor) * fatjets.mass
        fatjets["ParTmassVis"] = fatjets.globalParT_massVis * (1 - fatjets.rawFactor) * fatjets.mass
    elif "globalParT2_Xbb" in fatjets_fields:
        # renamed in Nano v14
        fatjets["ParT2PQCD1HF"] = fatjets.globalParT2_QCD1HF
        fatjets["ParT2PQCD2HF"] = fatjets.globalParT2_QCD2HF
        fatjets["ParT2PQCD0HF"] = fatjets.globalParT2_QCD0HF

        fatjets["ParT2PTopW"] = fatjets.globalParT2_TopW
        fatjets["ParT2PTopbW"] = fatjets.globalParT2_TopbW
        fatjets["ParT2PTopbWev"] = fatjets.globalParT2_TopbWev
        fatjets["ParT2PTopbWmv"] = fatjets.globalParT2_TopbWmv
        fatjets["ParT2PTopbWtauhv"] = fatjets.globalParT2_TopbWtauhv
        fatjets["ParT2PTopbWq"] = fatjets.globalParT2_TopbWq
        fatjets["ParT2PTopbWqq"] = fatjets.globalParT2_TopbWqq

        fatjets["ParT2PXbb"] = fatjets.globalParT2_Xbb
        fatjets["ParT2PXcc"] = fatjets.globalParT2_Xcc
        fatjets["ParT2PXcs"] = fatjets.globalParT2_Xcs
        fatjets["ParT2PXgg"] = fatjets.globalParT2_Xgg
        fatjets["ParT2PXqq"] = fatjets.globalParT2_Xqq

        fatjets["ParT2PXtauhtaue"] = fatjets.globalParT2_Xtauhtaue
        fatjets["ParT2PXtauhtauh"] = fatjets.globalParT2_Xtauhtauh
        fatjets["ParT2PXtauhtaum"] = fatjets.globalParT2_Xtauhtaum

        # T for discriminator
        fatjets["ParT2TXbb"] = fatjets.globalParT2_XbbVsQCD
        # Mass Regression (Raw)
        fatjets["ParT2massRes"] = (
            fatjets.globalParT2_massRes * (1 - fatjets.rawFactor) * fatjets.mass
        )
        fatjets["ParT2massVis"] = (
            fatjets.globalParT2_massVis * (1 - fatjets.rawFactor) * fatjets.mass
        )

    if "globalParT3_Xbb" in fatjets_fields:
        fatjets["ParT3PQCD"] = fatjets.globalParT3_QCD

        # no globalParT3_TopbW
        fatjets["ParT3PTopbWev"] = fatjets.globalParT3_TopbWev
        fatjets["ParT3PTopbWmv"] = fatjets.globalParT3_TopbWmv
        fatjets["ParT3PTopbWq"] = fatjets.globalParT3_TopbWq
        fatjets["ParT3PTopbWqq"] = fatjets.globalParT3_TopbWqq
        fatjets["ParT3PTopbWtauhv"] = fatjets.globalParT3_TopbWtauhv

        fatjets["ParT3PXbb"] = fatjets.globalParT3_Xbb
        fatjets["ParT3PXcc"] = fatjets.globalParT3_Xcc
        fatjets["ParT3PXcs"] = fatjets.globalParT3_Xcs
        fatjets["ParT3PXqq"] = fatjets.globalParT3_Xqq

        fatjets["ParT3PXtauhtaue"] = fatjets.globalParT3_Xtauhtaue
        fatjets["ParT3PXtauhtauh"] = fatjets.globalParT3_Xtauhtauh
        fatjets["ParT3PXtauhtaum"] = fatjets.globalParT3_Xtauhtaum

        # T for discriminator
        fatjets["ParT3TXbb"] = fatjets.globalParT3_Xbb / (
            fatjets.globalParT3_Xbb + fatjets.globalParT3_QCD
        )

        # Mass Regression
        fatjets["ParT3massGeneric"] = (
            fatjets.globalParT3_massCorrGeneric * (1 - fatjets.rawFactor) * fatjets.mass
        )
        fatjets["ParT3massCorrX2p"] = (
            fatjets.globalParT3_massCorrX2p * (1 - fatjets.rawFactor) * fatjets.mass
        )

    return fatjets


# ak8 jet definition
def good_ak8jets(
    fatjets: FatJetArray,
    pt: float,
    eta: float,
    msd: float,
    mreg: float,
    nano_version: str,
    mreg_str="particleNet_mass_legacy",
):
    # if nano_version.startswith("v12"):
    #     jetidtight, jetidtightlepveto = jetid_v12(fatjets)  # v12 jetid fix
    # elif nano_version.startswith(("v13", "v14", "v15")):
    #     raise NotImplementedError("Jet ID fix for NanoAOD v13, v14, v15 not implemented yet!")
    # else:
    #     jetidtight, _jetidtightlepveto = fatjets.isTight, fatjets.isTightLepVeto

    # Data does not have .neHEF etc. fields for fatjets, so above recipe doesn't work
    # Either way, doesn't matter since we only use tightID, and it is correct for eta < 2.7
    if nano_version.startswith("v14"):
        jetidtight, _ = jetid_v14(fatjets)
    else:
        jetidtight = fatjets.isTight

    fatjet_sel = (
        jetidtight
        & (fatjets.pt > pt)
        & (abs(fatjets.eta) < eta)
        & ((fatjets.msoftdrop > msd) | (fatjets[mreg_str] > mreg))
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
        (jets.pt >= pt)
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
    sort_by: str = "btag",
):
    """AK4 jets nonoverlapping with AK8 fatjets"""
    electrons = events.Electron
    electrons = electrons[electrons.pt > electron_pt]

    muons = events.Muon
    muons = muons[muons.pt > muon_pt]

    ak4_sel = (
        (jets.pt >= pt)
        & (np.abs(jets.eta) <= eta_max)
        & (ak.all(jets.metric_table(fatjets) > dr_fatjets, axis=2))
        & ak.all(jets.metric_table(electrons) > dr_leptons, axis=2)
        & ak.all(jets.metric_table(muons) > dr_leptons, axis=2)
    )

    # return top 2 jets sorted by btagPNetB
    if sort_by == "btag":
        jets_pnetb = jets[ak.argsort(jets.btagPNetB, ascending=False)]
        return jets_pnetb[ak4_sel][:, :2]
    # return 2 jets closet to fatjet0 and fatjet1, respectively
    elif sort_by == "nearest":
        jets_away = jets[ak4_sel]
        FirstFatjet = ak.firsts(fatjets[:, 0:1])
        SecondFatjet = ak.firsts(fatjets[:, 1:2])
        jet_near_fatjet0 = jets_away[ak.argsort(jets_away.delta_r(FirstFatjet), ascending=True)][
            :, 0:1
        ]
        jet_near_fatjet1 = jets_away[ak.argsort(jets_away.delta_r(SecondFatjet), ascending=True)][
            :, 0:1
        ]
        return [jet_near_fatjet0, jet_near_fatjet1]
    # return all nonoverlapping jets, no sorting
    else:
        return jets[ak4_sel]
