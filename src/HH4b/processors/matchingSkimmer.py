"""
Skimmer for matching studies
Author(s): Raghav Kansal, Cristina Suarez
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
import vector

import pathlib
import pickle
import gzip
import os

from typing import Dict
from collections import OrderedDict

from .GenSelection import gen_selection_HHbbbb
from .utils import pad_val, add_selection, concatenate_dicts, select_dicts, P4, PAD_VAL
from .common import LUMI, jec_shifts, jmsr_shifts
from .objects import good_ak4jets, good_ak8jets, good_muons, good_electrons
from . import common


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "GluGlutoHHto4B": gen_selection_HHbbbb,
}


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class matchingSkimmer(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {
        "Jet": {
            **P4,
            "btagDeepB": "btagDeepB",
            "btagDeepFlavB": "btagDeepFlavB",
            # TODO: add hadron flavour
            # TODO: add matched_fj_idx
        },
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "Txbb": "PNetXbb",
            "Txjj": "PNetXjj",
            "particleNet_mass": "PNetMass",
        },
        "GenJet": P4,
    }

    preselection = {
        "fatjet_pt": 200,
    }

    jecs = common.jecs

    def __init__(self):
        super(matchingSkimmer, self).__init__()

        self._accumulator = processor.dict_accumulator({})

    def to_pandas(self, events: Dict[str, np.array]):
        """
        Convert our dictionary of numpy arrays into a pandas data frame
        Uses multi-index columns for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        return pd.concat(
            [pd.DataFrame(v) for k, v in events.items()],
            axis=1,
            keys=list(events.keys()),
        )

    def dump_table(self, pddf: pd.DataFrame, fname: str, odir_str: str = None) -> None:
        """
        Saves pandas dataframe events to './outparquet'
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        local_dir = os.path.abspath(os.path.join(".", "outparquet"))
        if odir_str:
            local_dir += odir_str
        os.system(f"mkdir -p {local_dir}")

        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, f"{local_dir}/{fname}")

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])

        btag_vars = {} 
        if year != "2018":
            # for now, only in v11_private
            btag_vars = {
                "btagPNetProb": "btagPNetProb",
                "btagPNetProbbb": "btagPNetProbbb",
                "btagPNetProbc": "btagPNetProbc",
                "btagPNetProbuds": "btagPNetProbuds",
                "btagPNetProbg": "btagPNetProbg",
                "btagPNetBvsAll": "btagPNetBvsAll",
            }

        isData = not hasattr(events, "genWeight")
        isSignal = "HHTobbbb" in dataset

        if isSignal:
            # take only signs for HH samples
            gen_weights = np.sign(events["genWeight"])
        elif not isData:
            gen_weights = events["genWeight"].to_numpy()
        else:
            gen_weights = None

        n_events = len(events) if isData else np.sum(gen_weights)
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        selection_args = (selection, cutflow, isData, gen_weights)

        #########################
        # Object definitions
        #########################
        num_jets = 6
        # In Run3 nanoAOD events.Jet = AK4 Puppi Jets
        jets = good_ak4jets(events.Jet, year, events.run.to_numpy(), isData)

        num_fatjets = 3
        fatjets = good_ak8jets(events.FatJet)

        #########################
        # Save / derive variables
        #########################
        skimmed_events = {}

        # Jet variables
        jet_vars = {**self.skim_vars["Jet"], **btag_vars}
        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_jets, axis=1)
            for (var, key) in jet_vars.items()
        }

        # FatJet variables
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        # gen variables
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](
                    events, jets, fatjets, selection, cutflow, gen_weights, P4
                )
                skimmed_events = {**skimmed_events, **vars_dict}

        ak4GenJetVars = {
            f"ak4GenJet{key}": pad_val(events.GenJet[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["GenJet"].items()
        }

        ak8GenJetVars = {
            f"ak8GenJet{key}": pad_val(events.GenJetAK8[var], num_fatjets, axis=1)
            for (var, key) in self.skim_vars["GenJet"].items()
        }

        skimmed_events = {
            **skimmed_events,
            **ak4JetVars,
            **ak8FatJetVars,
            **ak4GenJetVars,
            **ak8GenJetVars,
        }

        ######################
        # Selection
        ######################

        # # jet veto map for 2022
        # if year == "2022" and isData:
        #     jetveto = get_jetveto_event(jets, year, events.run.to_numpy())
        #     add_selection("ak4_jetveto", jetveto, *selection_args)

        # require at least one ak8 jet
        # add_selection("ak8_pt", np.any(fatjets.pt > 200, axis=1), *selection_args)

        ######################
        # Weights
        ######################

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights.add("genweight", gen_weights)
            skimmed_events["weight"] = weights.weight()

        # reshape and apply selections
        sel_all = selection.all(*selection.names)

        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        df = self.to_pandas(skimmed_events)

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator
