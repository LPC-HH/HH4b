"""
Common functions for processors.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection

P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}


PAD_VAL = -99999

jecs = {
    "JES": "JES_jes",
    "JER": "JER",
    # split JECs (unused for now)
    # "JES_AbsoluteMPFBias": "JES_AbsoluteMPFBias",  # goes in Absolute
    # "JES_AbsoluteScale": "JES_AbsoluteScale",  # goes in Absolute
    # "JES_AbsoluteStat": "JES_AbsoluteStat",  # goes in Abs_year
    # "JES_FlavorQCD": "JES_FlavorQCD",
    # "JES_Fragmentation": "JES_Fragmentation",  # goes in Absolute
    # "JES_PileUpDataMC": "JES_PileUpDataMC",  # goes in Absolute
    # "JES_PileUpPtBB": "JES_PileUpPtBB",  # goes in BBEC1
    # "JES_PileUpPtEC1": "JES_PileUpPtEC1",  # goes in BBEC1
    # "JES_PileUpPtEC2": "JES_PileUpPtEC2",
    # "JES_PileUpPtHF": "JES_PileUpPtHF",
    # "JES_PileUpPtRef": "JES_PileUpPtRef",  # goes in Absolute
    # "JES_RelativeFSR": "JES_RelativeFSR",  # goes in Absolute
    # "JES_RelativeJEREC1": "JES_RelativeJEREC1",  # goes in BBEC1_year
    # "JES_RelativeJEREC2": "JES_RelativeJEREC2",  # goes in EC2_year
    # "JES_RelativeJERHF": "JES_RelativeJERHF",  # goes in HF
    # "JES_RelativePtBB": "JES_RelativePtBB",  # goes in BBEC1
    # "JES_RelativePtEC1": "JES_RelativePtEC1",  # goes in BBEC1_year
    # "JES_RelativePtEC2": "JES_RelativePtEC2",  # goes in EC2_year
    # "JES_RelativePtHF": "JES_RelativePtHF",  # goes in HF
    # "JES_RelativeBal": "JES_RelativeBal",
    # "JES_RelativeSample": "JES_RelativeSample",
    # "JES_RelativeStatEC": "JES_RelativeStatEC",  # goes in BBEC1_year
    # "JES_RelativeStatFSR": "JES_RelativeStatFSR",  # goes in Abs_year
    # "JES_RelativeStatHF": "JES_RelativeStatHF",
    # "JES_SinglePionHCAL": "JES_SinglePionHCAL",  # goes in Absolute
    # "JES_SinglePionECAL": "JES_SinglePionECAL",  # goes in Absolute
    # "JES_TimePtEta": "JES_TimePtEta",  # goes in Abs_year
}
jec_shifts = []
for key in jecs:
    for shift in ["up", "down"]:
        jec_shifts.append(f"{key}_{shift}")

jmsr = {
    "JMS": "JMS",
    "JMR": "JMR",
}
jmsr_shifts = []
for key in jmsr:
    for shift in ["up", "down"]:
        jmsr_shifts.append(f"{key}_{shift}")

# variables affected by JECs
jec_vars = [
    "bbFatJetPt",
    "VBFJetPt",
    "HHPt",
    "HHeta",
    "HHmass",
    "H1Pt",
    "H2Pt",
    "H1Pt_HHmass",
    "H2Pt_HHmass",
    "H1Pt/H2Pt",
    "VBFjjMass",
    "VBFjjDeltaEta",
]

# variables affected by JMS/JMR
jmsr_vars = [
    "bbFatJetPNetMassLegacy",
    "HHmass",
    "H1Pt_HHmass",
    "H2Pt_HHmass",
    "H1Mass",
    "H2Mass",
    "H1PNetMass",
    "H2PNetMass",
]


def pad_val(
    arr: ak.Array,
    target: int,
    value: float = PAD_VAL,
    axis: int = 0,
    to_numpy: bool = True,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=axis)
    return ret.to_numpy() if to_numpy else ret


def add_selection(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
    cutflow: dict,
    isData: bool,
    genWeights: ak.Array = None,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    if isinstance(sel, ak.Array):
        sel = sel.to_numpy()

    selection.add(name, sel.astype(bool))
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up genWeights for MC
        else np.sum(genWeights[selection.all(*selection.names)])
    )


def add_selection_no_cutflow(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
):
    """adds selection to PackedSelection object"""
    selection.add(name, ak.fill_none(sel, False))


def concatenate_dicts(dicts_list: list[dict[str, np.ndarray]]):
    """given a list of dicts of numpy arrays, concatenates the numpy arrays across the lists"""
    if len(dicts_list) > 1:
        return {
            key: np.concatenate(
                [
                    dicts_list[i][key].reshape(dicts_list[i][key].shape[0], -1)
                    for i in range(len(dicts_list))
                ],
                axis=1,
            )
            for key in dicts_list[0]
        }

    return dicts_list[0]


def select_dicts(dicts_list: list[dict[str, np.ndarray]], sel: np.ndarray):
    """given a list of dicts of numpy arrays, select the entries per array across the lists according to ``sel``"""
    return {
        key: np.stack(
            [
                dicts_list[i][key].reshape(dicts_list[i][key].shape[0], -1)
                for i in range(len(dicts_list))
            ],
            axis=1,
        )[sel]
        for key in dicts_list[0]
    }


def remove_variation_suffix(var: str):
    """removes the variation suffix from the variable name"""
    if var.endswith("Down"):
        return var.split("Down")[0]
    elif var.endswith("Up"):
        return var.split("Up")[0]
    return var


def check_get_jec_var(var, jshift):
    """Checks if var is affected by the JEC / JMSR and if so, returns the shifted var name"""

    if jshift in jec_shifts and var in jec_vars:
        return var + "_" + jshift

    if jshift in jmsr_shifts and var in jmsr_vars:
        return var + "_" + jshift

    return var


def get_var_mapping(jshift):
    """Returns function that maps var to shifted var for a given systematic shift [JES|JER|JMS|JMR]_[up|down]"""

    def var_mapping(var):
        return check_get_jec_var(var, jshift)

    return var_mapping
