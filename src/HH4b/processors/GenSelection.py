"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan
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

    # match jets to bb
    num_jets = 6
    # is the jet matched to the Higgs?
    jets["HiggsMatch"] = ak.any(jets.metric_table(higgs) < 0.4, axis=2)
    # index of higgs to which the jet is closest to
    jets["HiggsMatchIndex"] = ak.mask(
        ak.argmin(jets.metric_table(higgs), axis=2), jets["HiggsMatch"] == 1
    )
    ak4JetVars = {
        f"ak4Jet{var}": pad_val(jets[var], num_jets, axis=1)
        for var in ["HiggsMatch", "HiggsMatchIndex"]
    }

    # match fatjets to bb
    num_fatjets = 3
    fatjets["HiggsMatch"] = ak.any(fatjets.metric_table(higgs) < 0.8, axis=2)
    fatjets["HiggsMatchIndex"] = ak.mask(
        ak.argmin(fatjets.metric_table(higgs), axis=2), fatjets["HiggsMatch"] == 1
    )
    fatjets["NumBMatched"] = ak.sum(fatjets.metric_table(bs) < 0.8, axis=2)
    ak8FatJetVars = {
        f"ak8FatJet{var}": pad_val(fatjets[var], num_fatjets, axis=1)
        for var in ["HiggsMatch", "HiggsMatchIndex", "NumBMatched"]
    }

    return {**GenHiggsVars, **ak4JetVars}
