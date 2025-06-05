"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal, Cristina Mantilla Suarez
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, JetArray

from .utils import add_selection, pad_val

d_PDGID = 1
u_PDGID = 2
s_PDGID = 3
c_PDGID = 4
b_PDGID = 5
g_PDGID = 21
TOP_PDGID = 6

ELE_PDGID = 11
vELE_PDGID = 12
MU_PDGID = 13
vMU_PDGID = 14
TAU_PDGID = 15
vTAU_PDGID = 16

G_PDGID = 22
Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25

b_PDGIDS = [511, 521, 523]

GEN_FLAGS = ["fromHardProcess", "isLastCopy"]


def gen_selection_HHbbbb_simplified(
    events: NanoEventsArray,
    jets: JetArray,  # noqa: ARG001
    fatjets: FatJetArray,  # noqa: ARG001
    selection_args: list,  # noqa: ARG001
    skim_vars: dict,
    fatjet_str: str,  # noqa: ARG001
):
    """
    Save GenVars for HH(4b) events
    Does not make use of fatjet or jet matching
    """
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}
    higgs_children = higgs.children
    is_bb = abs(higgs_children.pdgId) == b_PDGID
    bs = ak.flatten(higgs_children[is_bb], axis=2)
    GenbVars = {f"Genb{key}": pad_val(bs[var], 4, axis=1) for (var, key) in skim_vars.items()}

    return {**GenHiggsVars, **GenbVars}


def gen_selection_HHbbbb(
    events: NanoEventsArray,
    jets: JetArray,
    fatjets: FatJetArray,
    selection_args: list,
    skim_vars: dict,
    fatjet_str: str,
):
    """
    Gets HH, bb 4-vectors, and matches to AK4 jets and AK8 jets
    """
    assert fatjet_str in [
        "bbFatJet",
        "ak8FatJet",
    ], "fatjet_str parameter must be bbFatJet or ak8FatJet"

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]

    # saving 4-vector info
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}

    higgs_children = higgs.children
    is_bb = abs(higgs_children.pdgId) == b_PDGID

    # checking that is a 4b decay
    has_4b = ak.sum(ak.flatten(is_bb, axis=2), axis=1) == 4

    # only select events with 4 b's
    add_selection("has_bbbb", has_4b, *selection_args)

    # children 4-vector
    bs = ak.flatten(higgs_children[is_bb], axis=2)
    GenbVars = {f"Genb{key}": pad_val(bs[var], 4, axis=1) for (var, key) in skim_vars.items()}

    bs_unflat = higgs_children[is_bb]
    b_h1 = higgs_children[is_bb][:, 0]
    b_h2 = higgs_children[is_bb][:, 1]

    # match jets to each b-quark
    matched_to_bs = jets.metric_table(bs_unflat) < 0.4
    num_b_matched = ak.sum(matched_to_bs, axis=2)
    matched_to_higgs = jets.metric_table(higgs) < 0.5

    # require 1 b matched to the jet (but not necessarily matched to the Higgs)
    is_matched = num_b_matched == 1
    is_jet_matched = ak.any(is_matched, axis=2)
    jets["HiggsMatch"] = is_jet_matched

    # index of the higgs to which the jet is closest to
    # This line in particular is taking ak.argmin (index of the b quark that is closest to the jet)
    # we take np.floor of the number divided by 2 to get the index of the higgs
    #  e.g. if it is matched to b quarks 0 or 1 => HiggsMatchIndex = 0
    #  e.g. if it is matched to b quarks 2 or 3 => HiggsMatchIndex = 1
    jets["HiggsMatchIndex"] = ak.mask(
        np.floor(ak.argmin(jets.metric_table(bs), axis=2) / 2), jets["HiggsMatch"] == 1
    )

    num_jets = 6
    ak4JetVars = {
        f"ak4Jet{var}": pad_val(jets[var], num_jets, axis=1)
        for var in ["HiggsMatch", "HiggsMatchIndex", "hadronFlavour"]
    }

    # match fatjets to bb
    num_b_matched = ak.sum(fatjets.metric_table(bs_unflat) < 0.8, axis=2)
    matched_to_higgs = fatjets.metric_table(higgs) < 0.8

    # require 2 bs matched to the jet
    is_matched = matched_to_higgs
    is_fatjet_matched = ak.any(is_matched, axis=2)

    fatjets["HiggsMatch"] = is_fatjet_matched
    fatjets["HiggsMatchIndex"] = ak.mask(
        ak.argmin(fatjets.metric_table(higgs), axis=2), fatjets["HiggsMatch"] == 1
    )
    fatjets["NumBMatchedH1"] = ak.sum(fatjets.metric_table(b_h1) < 0.8, axis=2)
    fatjets["NumBMatchedH2"] = ak.sum(fatjets.metric_table(b_h2) < 0.8, axis=2)
    fatjets["MaxdRH1"] = ak.max(fatjets.metric_table(b_h1), axis=2)
    fatjets["MaxdRH2"] = ak.max(fatjets.metric_table(b_h2), axis=2)

    num_fatjets = 2
    FatJetVars = {
        f"{fatjet_str}{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in [
            "HiggsMatch",
            "HiggsMatchIndex",
            "NumBMatchedH1",
            "NumBMatchedH2",
            "MaxdRH1",
            "MaxdRH2",
        ]
    }

    return {**GenHiggsVars, **GenbVars, **ak4JetVars, **FatJetVars}


def gen_selection_Hbb(
    events: NanoEventsArray,
    jets: JetArray,  # noqa: ARG001
    fatjets: FatJetArray,
    selection_args: list,  # noqa: ARG001
    skim_vars: dict,
    fatjet_str: str,
):
    """Gets H, bb, 4-vectors + Higgs children information"""
    assert fatjet_str in [
        "bbFatJet",
        "ak8FatJet",
    ], "fatjet_str parameter must be bbFatJet or ak8FatJet"

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    higgs_children = higgs.children

    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}
    GenHiggsVars["GenHiggsChildren"] = abs(higgs_children.pdgId[:, :, 0]).to_numpy()

    is_bb = abs(higgs_children.pdgId) == b_PDGID
    bs = ak.flatten(higgs_children[is_bb], axis=2)
    GenbVars = {f"Genb{key}": pad_val(bs[var], 4, axis=1) for (var, key) in skim_vars.items()}

    # match fatjets to bb
    # bs_unflat = higgs_children[is_bb]
    # num_b_matched = ak.sum(fatjets.metric_table(bs_unflat) < 0.8, axis=2)
    b_h1 = ak.firsts(higgs_children[is_bb][:, 0:1])
    b_h2 = ak.firsts(higgs_children[is_bb][:, 1:2])
    matched_to_higgs = fatjets.metric_table(higgs) < 0.8
    is_fatjet_matched = ak.any(matched_to_higgs, axis=2)

    fatjets["HiggsMatch"] = is_fatjet_matched
    fatjets["HiggsMatchIndex"] = ak.mask(
        ak.argmin(fatjets.metric_table(higgs), axis=2), fatjets["HiggsMatch"] == 1
    )
    fatjets["NumBMatchedH1"] = ak.sum(fatjets.metric_table(b_h1) < 0.8, axis=2)
    fatjets["NumBMatchedH2"] = ak.sum(fatjets.metric_table(b_h2) < 0.8, axis=2)

    num_fatjets = 2
    FatJetVars = {
        f"{fatjet_str}{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in [
            "HiggsMatch",
            "HiggsMatchIndex",
            "NumBMatchedH1",
            "NumBMatchedH2",
        ]
    }

    return {**GenHiggsVars, **GenbVars, **FatJetVars}


def gen_selection_Top(
    events: NanoEventsArray,
    jets: JetArray,  # noqa: ARG001
    fatjets: FatJetArray,
    selection_args: list,  # noqa: ARG001
    skim_vars: dict,
    fatjet_str: str,
):
    """Get Hadronic Top and children information"""
    assert fatjet_str in [
        "bbFatJet",
        "ak8FatJet",
    ], "fatjet_str parameter must be bbFatJet or ak8FatJet"

    # finding tops
    tops = events.GenPart[
        (abs(events.GenPart.pdgId) == TOP_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    GenTopVars = {f"GenTop{key}": tops[var].to_numpy() for (var, key) in skim_vars.items()}

    daughters = ak.flatten(tops.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    daughters_pdgId = abs(daughters.pdgId)

    wboson_0 = ak.firsts(daughters[(daughters_pdgId == W_PDGID)][:, 0:1])
    wboson_1 = ak.firsts(daughters[(daughters_pdgId == W_PDGID)][:, 1:2])
    GenTopVars = {
        **GenTopVars,
        **{f"GenTopW0{key}": wboson_0[var].to_numpy() for (var, key) in skim_vars.items()},
        **{f"GenTopW1{key}": wboson_1[var].to_numpy() for (var, key) in skim_vars.items()},
    }

    wboson_daughters = ak.flatten(daughters[(daughters_pdgId == W_PDGID)].distinctChildren, axis=2)
    wboson_daughters = wboson_daughters[
        wboson_daughters.hasFlags(["fromHardProcess", "isLastCopy"])
    ]

    bquark = daughters[(daughters_pdgId == 5)]
    matched_to_top = fatjets.metric_table(tops) < 0.8
    is_fatjet_matched = ak.any(matched_to_top, axis=2)

    qs_0 = ak.firsts(wboson_daughters[:, 0:1])
    qs_1 = ak.firsts(wboson_daughters[:, 1:2])
    qs_2 = ak.firsts(wboson_daughters[:, 2:3])
    qs_3 = ak.firsts(wboson_daughters[:, 3:4])
    bs_0 = ak.firsts(bquark[:, 0:1])
    bs_1 = ak.firsts(bquark[:, 1:2])

    numtop1 = ak.values_astype(fatjets.delta_r(qs_0) < 0.8, np.int32) + ak.values_astype(
        fatjets.delta_r(qs_1) < 0.8, np.int32
    )
    numtop2 = ak.values_astype(fatjets.delta_r(qs_2) < 0.8, np.int32) + ak.values_astype(
        fatjets.delta_r(qs_3) < 0.8, np.int32
    )

    fatjets["TopMatch"] = is_fatjet_matched
    fatjets["TopMatchIndex"] = ak.mask(
        ak.argmin(fatjets.metric_table(tops), axis=2), fatjets["TopMatch"] == 1
    )
    fatjets["NumBMatchedTop1"] = ak.values_astype(fatjets.delta_r(bs_0) < 0.8, np.int32)
    fatjets["NumBMatchedTop2"] = ak.values_astype(fatjets.delta_r(bs_1) < 0.8, np.int32)
    fatjets["NumQMatchedTop1"] = numtop1
    fatjets["NumQMatchedTop2"] = numtop2

    num_fatjets = 2
    FatJetVars = {
        f"{fatjet_str}{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in [
            "TopMatch",
            "TopMatchIndex",
            "NumBMatchedTop1",
            "NumBMatchedTop2",
            "NumQMatchedTop1",
            "NumQMatchedTop2",
        ]
    }

    return {**GenTopVars, **FatJetVars}


def gen_selection_V(
    events: NanoEventsArray,
    jets: JetArray,  # noqa: ARG001
    fatjets: FatJetArray,
    selection_args: list,  # noqa: ARG001
    skim_vars: dict,
    fatjet_str: str,
):
    """Get W/Z and children information"""
    assert fatjet_str in [
        "bbFatJet",
        "ak8FatJet",
    ], "fatjet_str parameter must be bbFatJet or ak8FatJet"

    # get V boson
    vs = events.GenPart[
        ((abs(events.GenPart.pdgId) == W_PDGID) | (abs(events.GenPart.pdgId) == Z_PDGID))
        * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    GenVVars = {f"GenV{key}": pad_val(vs[var], 1, axis=1) for (var, key) in skim_vars.items()}

    # get V daughters
    daughters = vs.children
    daughter0_pdgId = ak.firsts(abs(daughters.pdgId[:, :, 0]))
    daughter1_pdgId = ak.firsts(abs(daughters.pdgId[:, :, 1]))
    GenVVars["GenVBB"] = ((daughter0_pdgId == b_PDGID) & (daughter1_pdgId == b_PDGID)).to_numpy()
    GenVVars["GenVCC"] = ((daughter0_pdgId == c_PDGID) & (daughter1_pdgId == c_PDGID)).to_numpy()
    GenVVars["GenVCS"] = (
        ((daughter0_pdgId == c_PDGID) & (daughter1_pdgId == s_PDGID))
        | ((daughter1_pdgId == c_PDGID) & (daughter0_pdgId == s_PDGID))
    ).to_numpy()

    # match V to fatjet
    matched_to_v = fatjets.metric_table(vs) < 0.8
    is_fatjet_matched = ak.any(matched_to_v, axis=2)
    fatjets["VMatch"] = is_fatjet_matched
    fatjets["VMatchIndex"] = ak.mask(
        ak.argmin(fatjets.metric_table(vs), axis=2), fatjets["VMatch"] == 1
    )

    # match V daughters to fatjet
    d_v1 = daughters[:, 0]
    matched_mask = fatjets.metric_table(d_v1) < 0.8
    fatjets["NumQMatched"] = ak.sum(matched_mask, axis=2)

    # save fatjet matching information
    num_fatjets = 2
    FatJetVars = {
        f"{fatjet_str}{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in [
            "VMatch",
            "VMatchIndex",
            "NumQMatched",
        ]
    }

    return {**GenVVars, **FatJetVars}


def gen_selection_VV(
    events: NanoEventsArray,
    jets: JetArray,  # noqa: ARG001
    fatjets: FatJetArray,
    selection_args: list,  # noqa: ARG001
    skim_vars: dict,
    fatjet_str: str,
):
    """Get W/Z and children information"""
    assert fatjet_str in [
        "bbFatJet",
        "ak8FatJet",
    ], "fatjet_str parameter must be bbFatJet or ak8FatJet"

    # get V boson or Higgs boson
    vs = events.GenPart[
        (
            (abs(events.GenPart.pdgId) == W_PDGID)
            | (abs(events.GenPart.pdgId) == Z_PDGID)
            | (abs(events.GenPart.pdgId) == HIGGS_PDGID)
        )
        * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    GenVVars = {f"GenV{key}": pad_val(vs[var], 2, axis=1) for (var, key) in skim_vars.items()}

    # get V daughters
    daughters = vs.children

    v0_daughter0_pdgId = abs(daughters.pdgId[:, 0, 0])
    v0_daughter1_pdgId = abs(daughters.pdgId[:, 0, 1])
    v1_daughter0_pdgId = abs(daughters.pdgId[:, 1, 0])
    v1_daughter1_pdgId = abs(daughters.pdgId[:, 1, 1])
    GenVVars["GenV1BB"] = (
        (v0_daughter0_pdgId == b_PDGID) & (v0_daughter1_pdgId == b_PDGID)
    ).to_numpy()
    GenVVars["GenV1CC"] = (
        (v0_daughter0_pdgId == c_PDGID) & (v0_daughter1_pdgId == c_PDGID)
    ).to_numpy()
    GenVVars["GenV1CS"] = (
        ((v0_daughter0_pdgId == c_PDGID) & (v0_daughter1_pdgId == s_PDGID))
        | ((v0_daughter0_pdgId == c_PDGID) & (v0_daughter1_pdgId == s_PDGID))
    ).to_numpy()
    GenVVars["GenV2BB"] = (
        (v1_daughter0_pdgId == b_PDGID) & (v1_daughter1_pdgId == b_PDGID)
    ).to_numpy()
    GenVVars["GenV2CC"] = (
        (v1_daughter0_pdgId == c_PDGID) & (v1_daughter1_pdgId == c_PDGID)
    ).to_numpy()
    GenVVars["GenV2CS"] = (
        ((v1_daughter0_pdgId == c_PDGID) & (v1_daughter1_pdgId == s_PDGID))
        | ((v1_daughter0_pdgId == c_PDGID) & (v1_daughter1_pdgId == s_PDGID))
    ).to_numpy()

    # match V to fatjet
    matched_to_v = fatjets.metric_table(vs) < 0.8
    is_fatjet_matched = ak.any(matched_to_v, axis=2)
    fatjets["VMatch"] = is_fatjet_matched
    fatjets["VMatchIndex"] = ak.mask(
        ak.argmin(fatjets.metric_table(vs), axis=2), fatjets["VMatch"] == 1
    )
    num_fatjets = 2
    FatJetVars = {
        f"{fatjet_str}{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in [
            "VMatch",
            "VMatchIndex",
        ]
    }
    return {**GenVVars, **FatJetVars}


def gen_selection_ZbbSF_ZQQ(
    events: NanoEventsArray,
    jets: JetArray,  # noqa: ARG001
    fatjets: FatJetArray,
    selection_args: list,  # noqa: ARG001
    skim_vars: dict,
    fatjet_str: str,
):
    assert fatjet_str in [
        "bbFatJet",
        "ak8FatJet",
    ], "fatjet_str parameter must be bbFatJet or ak8FatJet"

    # get Z boson
    zs = events.GenPart[(abs(events.GenPart.pdgId) == Z_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)]

    # Get Z boson properties
    GenZVars = {f"GenZ{key}": pad_val(zs[var], 1, axis=1) for (var, key) in skim_vars.items()}

    daughters = zs.children
    daughter0 = daughters[:, :, 0]
    daughter1 = daughters[:, :, 1]
    daughter0_pdgId = ak.firsts(abs(daughter0.pdgId))
    daughter1_pdgId = ak.firsts(abs(daughter1.pdgId))

    # Get Q1 and Q2 properties
    GenQ1Vars = {
        f"GenQ1{key}": pad_val(daughter0[var], 1, axis=1) for (var, key) in skim_vars.items()
    }
    GenQ2Vars = {
        f"GenQ2{key}": pad_val(daughter1[var], 1, axis=1) for (var, key) in skim_vars.items()
    }

    is_Z_bb = (daughter0_pdgId == b_PDGID) & (daughter1_pdgId == b_PDGID)
    is_Z_cc = (daughter0_pdgId == c_PDGID) & (daughter1_pdgId == c_PDGID)
    is_Z_ss = (daughter0_pdgId == s_PDGID) & (daughter1_pdgId == s_PDGID)
    is_Z_uu = (daughter0_pdgId == u_PDGID) & (daughter1_pdgId == u_PDGID)
    is_Z_dd = (daughter0_pdgId == d_PDGID) & (daughter1_pdgId == d_PDGID)

    GenZVars["GenZBB"] = is_Z_bb.to_numpy()
    GenZVars["GenZCC"] = is_Z_cc.to_numpy()
    GenZVars["GenZSS"] = is_Z_ss.to_numpy()
    GenZVars["GenZUU"] = is_Z_uu.to_numpy()
    GenZVars["GenZDD"] = is_Z_dd.to_numpy()
    GenZVars["GenZLight"] = (is_Z_dd | is_Z_uu | is_Z_ss).to_numpy()

    # Whether fatjets are matched to Z and daughters
    matched_to_z = ak.any(fatjets.metric_table(zs) < 0.8, axis=2)
    matched_to_d1 = ak.any(fatjets.metric_table(daughter0) < 0.8, axis=2)
    matched_to_d2 = ak.any(fatjets.metric_table(daughter1) < 0.8, axis=2)
    matched = matched_to_z & matched_to_d1 & matched_to_d2
    fatjets["VMatch"] = matched_to_z
    fatjets["Q1Match"] = matched_to_d1
    fatjets["Q2Match"] = matched_to_d2
    fatjets["VQQMatch"] = matched

    num_fatjets = 2
    FatJetVars = {
        f"{fatjet_str}{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in [
            "VMatch",
            "Q1Match",
            "Q2Match",
            "VQQMatch",
        ]
    }
    return {**GenZVars, **GenQ1Vars, **GenQ2Vars, **FatJetVars}


def gen_selection_ZbbSF_DYto2L(
    events: NanoEventsArray,
    jets: JetArray,  # noqa: ARG001
    fatjets: FatJetArray,  # noqa: ARG001
    selection_args: list,  # noqa: ARG001
    skim_vars: dict,
    fatjet_str: str,
):
    assert fatjet_str in [
        "bbFatJet",
        "ak8FatJet",
    ], "fatjet_str parameter must be bbFatJet or ak8FatJet"

    # get Z boson
    zs = events.GenPart[(abs(events.GenPart.pdgId) == Z_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)]

    # Get Z boson properties
    GenZVars = {f"GenZ{key}": pad_val(zs[var], 1, axis=1) for (var, key) in skim_vars.items()}

    # off-shell Z might not be present
    # find the first leptons and antileptons
    daughters = events.GenPart[
        (abs(events.GenPart.pdgId) == ELE_PDGID)
        | (abs(events.GenPart.pdgId) == MU_PDGID)
        | (abs(events.GenPart.pdgId) == TAU_PDGID)
    ]
    daughter0 = daughters[daughters.pdgId < 0][:, 0:1]  # first lepton
    daughter1 = daughters[daughters.pdgId > 0][:, 0:1]  # first antilepton
    daughter0_pdgId = ak.firsts(abs(daughter0.pdgId))
    daughter1_pdgId = ak.firsts(abs(daughter1.pdgId))

    # Get Lep1 and Lep2 properties
    GenLep1Vars = {
        f"GenLep1{key}": pad_val(daughter0[var], 1, axis=1) for (var, key) in skim_vars.items()
    }
    GenLep2Vars = {
        f"GenLep2{key}": pad_val(daughter1[var], 1, axis=1) for (var, key) in skim_vars.items()
    }

    is_Z_ee = (daughter0_pdgId == ELE_PDGID) & (daughter1_pdgId == ELE_PDGID)
    is_Z_mumu = (daughter0_pdgId == MU_PDGID) & (daughter1_pdgId == MU_PDGID)
    is_Z_tautau = (daughter0_pdgId == TAU_PDGID) & (daughter1_pdgId == TAU_PDGID)
    GenZVars["GenZEleEle"] = is_Z_ee.to_numpy()
    GenZVars["GenZMuMu"] = is_Z_mumu.to_numpy()
    GenZVars["GenZTauTau"] = is_Z_tautau.to_numpy()

    return {**GenZVars, **GenLep1Vars, **GenLep2Vars}


def gen_selection_ZbbSF_WQQ(
    events: NanoEventsArray,
    jets: JetArray,  # noqa: ARG001
    fatjets: FatJetArray,
    selection_args: list,  # noqa: ARG001
    skim_vars: dict,
    fatjet_str: str,
):
    assert fatjet_str in [
        "bbFatJet",
        "ak8FatJet",
    ], "fatjet_str parameter must be bbFatJet or ak8FatJet"

    # get W boson
    ws = events.GenPart[(abs(events.GenPart.pdgId) == W_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)]

    # Get W boson properties
    GenWVars = {f"GenW{key}": pad_val(ws[var], 1, axis=1) for (var, key) in skim_vars.items()}

    daughters = ws.children
    daughter0 = daughters[:, :, 0]
    daughter1 = daughters[:, :, 1]
    daughter0_pdgId = ak.firsts(abs(daughter0.pdgId))
    daughter1_pdgId = ak.firsts(abs(daughter1.pdgId))

    # Get Q1 and Q2 properties
    GenQ1Vars = {
        f"GenQ1{key}": pad_val(daughter0[var], 1, axis=1) for (var, key) in skim_vars.items()
    }
    GenQ2Vars = {
        f"GenQ2{key}": pad_val(daughter1[var], 1, axis=1) for (var, key) in skim_vars.items()
    }

    is_W_cs = ((daughter0_pdgId == c_PDGID) & (daughter1_pdgId == s_PDGID)) | (
        (daughter0_pdgId == s_PDGID) & (daughter1_pdgId == c_PDGID)
    )
    is_W_ud = ((daughter0_pdgId == u_PDGID) & (daughter1_pdgId == d_PDGID)) | (
        (daughter0_pdgId == d_PDGID) & (daughter1_pdgId == u_PDGID)
    )

    GenWVars["GenWCS"] = is_W_cs.to_numpy()
    GenWVars["GenWUD"] = is_W_ud.to_numpy()

    # Whether fatjets are matched to W and daughters
    matched_to_w = ak.any(fatjets.metric_table(ws) < 0.8, axis=2)
    matched_to_d1 = ak.any(fatjets.metric_table(daughter0) < 0.8, axis=2)
    matched_to_d2 = ak.any(fatjets.metric_table(daughter1) < 0.8, axis=2)
    matched = matched_to_w & matched_to_d1 & matched_to_d2
    fatjets["VMatch"] = matched_to_w
    fatjets["Q1Match"] = matched_to_d1
    fatjets["Q2Match"] = matched_to_d2
    fatjets["VQQMatch"] = matched

    num_fatjets = 2
    FatJetVars = {
        f"{fatjet_str}{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in [
            "VMatch",
            "Q1Match",
            "Q2Match",
            "VQQMatch",
        ]
    }
    return {**GenWVars, **GenQ1Vars, **GenQ2Vars, **FatJetVars}
