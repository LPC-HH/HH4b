"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal, Cristina Mantilla Suarez
"""

import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray, JetArray

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


def gen_selection_HHbbbb(
    events: NanoEventsArray,
    jets: JetArray,
    fatjets: FatJetArray,
    selection: PackedSelection,
    cutflow: dict,
    signGenWeights: ak.Array,
    skim_vars: dict,
):
    """Gets HH, bb 4-vectors"""

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]

    # saving 4-vector info
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}

    higgs_children = higgs.children
    is_bb = abs(higgs_children.pdgId) == b_PDGID
    bb = ak.flatten(higgs.children[is_bb], axis=2)

    # checking that is a 4b decay
    has_4b = ak.sum(ak.flatten(is_bb, axis=2), axis=1) == 4

    # only select events with 4 b's
    add_selection("has_bbbb", has_4b, selection, cutflow, False, signGenWeights)

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
    is_matched = (num_b_matched == 2) & matched_to_higgs
    is_fatjet_matched = ak.any(is_matched, axis=2)

    fatjets["HiggsMatch"] = is_fatjet_matched
    fatjets["HiggsMatchIndex"] = ak.mask(
        ak.argmin(fatjets.metric_table(higgs), axis=2), fatjets["HiggsMatch"] == 1
    )
    fatjets["NumBMatchedH1"] = ak.sum(fatjets.metric_table(b_h1) < 0.8, axis=2)
    fatjets["NumBMatchedH2"] = ak.sum(fatjets.metric_table(b_h2) < 0.8, axis=2)

    num_fatjets = 3
    ak8FatJetVars = {
        f"ak8FatJet{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in ["HiggsMatch", "HiggsMatchIndex", "NumBMatchedH1", "NumBMatchedH2"]
    }

    return {**GenHiggsVars, **GenbVars, **ak4JetVars, **ak8FatJetVars}
