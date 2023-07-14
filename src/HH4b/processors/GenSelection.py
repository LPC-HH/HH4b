"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan
"""

import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray

from typing import List, Dict, Tuple, Union

from .utils import pad_val, add_selection, PAD_VAL


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


def gen_selection_HHbbVV(
    events: NanoEventsArray,
    fatjets: FatJetArray,
    selection: PackedSelection,
    cutflow: dict,
    signGenWeights: ak.Array,
    skim_vars: dict,
):
    """Gets HH, bb, VV, and 4q 4-vectors + Higgs children information"""

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]

    # saving 4-vector info
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}

    higgs_children = higgs.children

    # saving whether H->bb or H->VV
    GenHiggsVars["GenHiggsChildren"] = abs(higgs_children.pdgId[:, :, 0]).to_numpy()

    # finding bb and VV children
    is_bb = abs(higgs_children.pdgId) == b_PDGID
    is_VV = (abs(higgs_children.pdgId) == W_PDGID) + (abs(higgs_children.pdgId) == Z_PDGID)

    # checking that there are 2 b's and 2 V's
    has_bb = ak.sum(ak.flatten(is_bb, axis=2), axis=1) == 2
    has_VV = ak.sum(ak.flatten(is_VV, axis=2), axis=1) == 2

    # only select events with 2 b's and 2 V's
    add_selection("has_bbVV", has_bb * has_VV, selection, cutflow, False, signGenWeights)

    # saving bb and VV 4-vector info
    bb = ak.flatten(higgs_children[is_bb], axis=2)
    VV = ak.flatten(higgs_children[is_VV], axis=2)

    # have to pad to 2 because of some 4V events
    GenbbVars = {f"Genbb{key}": pad_val(bb[var], 2, axis=1) for (var, key) in skim_vars.items()}

    # selecting only up to the 2nd index because of some 4V events
    # (doesn't matter which two are selected since these events will be excluded anyway)
    GenVVVars = {f"GenVV{key}": VV[var][:, :2].to_numpy() for (var, key) in skim_vars.items()}

    # checking that each V has 2 q children
    VV_children = VV.children

    # iterate through the children in photon scattering events to get final daughter quarks
    for i in range(5):
        photon_mask = ak.any(ak.flatten(abs(VV_children.pdgId), axis=2) == G_PDGID, axis=1)
        if not np.any(photon_mask):
            break

        # use a where condition to get next layer of children for photon scattering events
        VV_children = ak.where(photon_mask, ak.flatten(VV_children.children, axis=3), VV_children)

    quarks = abs(VV_children.pdgId) <= b_PDGID
    all_q = ak.all(ak.all(quarks, axis=2), axis=1)
    add_selection("all_q", all_q, selection, cutflow, False, signGenWeights)

    V_has_2q = ak.count(VV_children.pdgId, axis=2) == 2
    has_4q = ak.values_astype(ak.prod(V_has_2q, axis=1), np.bool)
    add_selection("has_4q", has_4q, selection, cutflow, False, signGenWeights)

    # saving 4q 4-vector info
    Gen4qVars = {
        f"Gen4q{key}": ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak.pad_none(VV_children[var], 2, axis=1, clip=True), 2, axis=2, clip=True
                ),
                PAD_VAL,
            )
        )
        for (var, key) in skim_vars.items()
    }

    # fatjet gen matching
    Hbb = higgs[ak.sum(is_bb, axis=2) == 2]
    Hbb = ak.pad_none(Hbb, 1, axis=1, clip=True)[:, 0]

    HVV = higgs[ak.sum(is_VV, axis=2) == 2]
    HVV = ak.pad_none(HVV, 1, axis=1, clip=True)[:, 0]

    bbdr = fatjets[:, :2].delta_r(Hbb)
    vvdr = fatjets[:, :2].delta_r(HVV)

    match_dR = 0.8
    Hbb_match = bbdr <= match_dR
    HVV_match = vvdr <= match_dR

    # overlap removal - in the case where fatjet is matched to both, match it only to the closest Higgs
    Hbb_match = (Hbb_match * ~HVV_match) + (bbdr <= vvdr) * (Hbb_match * HVV_match)
    HVV_match = (HVV_match * ~Hbb_match) + (bbdr > vvdr) * (Hbb_match * HVV_match)

    VVJets = ak.pad_none(fatjets[HVV_match], 1, axis=1)[:, 0]
    quarkdrs = ak.flatten(VVJets.delta_r(VV_children), axis=2)
    num_prongs = ak.sum(quarkdrs < match_dR, axis=1)

    GenMatchingVars = {
        "ak8FatJetHbb": pad_val(Hbb_match, 2, axis=1),
        "ak8FatJetHVV": pad_val(HVV_match, 2, axis=1),
        "ak8FatJetHVVNumProngs": ak.fill_none(num_prongs, PAD_VAL).to_numpy(),
    }

    return {**GenHiggsVars, **GenbbVars, **GenVVVars, **Gen4qVars, **GenMatchingVars}, (
        bb,
        ak.flatten(VV_children, axis=2),
    )
    
    
# TODO!!
def gen_selection_HHbbbb():
    return