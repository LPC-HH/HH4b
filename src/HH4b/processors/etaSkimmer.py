"""
Skimmer for simple analysis with Eta meson into electrons.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import awkward as ak
import numpy as np
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

import HH4b

from . import utils
from .GenSelection import gen_selection_Eta
from .SkimmerABC import SkimmerABC
from .utils import P4, add_selection, pad_val


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "Eta": gen_selection_Eta,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class etaSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {  # noqa: RUF012
        "LowPtElectron": {
            "bdtID": "ID",
        },
        "Electron": {
            
        },
    }

    def __init__(
        self,
        xsecs=None,
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        start = time.time()
        print("Starting")
        print("# events", len(events))

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])

        isData = not hasattr(events, "genWeight")
        gen_weights = events["genWeight"].to_numpy() if not isData else None

        n_events = len(events) if isData else np.sum(gen_weights)

        cutflow = OrderedDict()
        cutflow["all"] = n_events
        selection = PackedSelection()
        selection_args = (selection, cutflow, isData, gen_weights)

        #########################
        # Object definitions
        #########################

        #########################
        # Derive variables
        #########################

        # Gen variables
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, P4)
                genVars = {**genVars, **vars_dict}

        # used for normalization to cross section below
        gen_selected = (
            selection.all(*selection.names)
            if len(selection.names)
            else np.ones(len(events)).astype(bool)
        )

        HLTs = [
            "DoubleEle4_eta1p22_mMax6",
            "DoubleEle4p5_eta1p22_mMax6",
            "DoubleEle5_eta1p22_mMax6",
            "DoubleEle5p5_eta1p22_mMax6",
            "DoubleEle6_eta1p22_mMax6",
            "DoubleEle6p5_eta1p22_mMax6",
            "DoubleEle7_eta1p22_mMax6",
            "DoubleEle7p5_eta1p22_mMax6",
            "DoubleEle8_eta1p22_mMax6",
            "DoubleEle8p5_eta1p22_mMax6",
            "DoubleEle9_eta1p22_mMax6",
            "DoubleEle9p5_eta1p22_mMax6",
            "DoubleEle10_eta1p22_mMax6",
        ]
        zeros = np.zeros(len(events), dtype="bool")
        HLTVars = {
            trigger: (
                events.HLT[trigger].to_numpy().astype(int)
                if trigger in events.HLT.fields
                else zeros
            )
            for trigger in HLTs
        }
        
        skimmed_events = {
            **genVars,
            **HLTVars,
        }

        print("Vars", f"{time.time() - start:.2f}")


        ######################
        # Weights
        ######################

        # used for normalization to cross section below
        gen_selected = (
            selection.all(*selection.names)
            if len(selection.names)
            else np.ones(len(events)).astype(bool)
        )

        totals_dict = {"nevents": n_events}

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights_dict, totals_temp = self.add_weights(
                events,
                year,
                dataset,
                gen_weights,
                gen_selected,
            )
            skimmed_events = {**skimmed_events, **weights_dict}
            totals_dict = {**totals_dict, **totals_temp}

        ##############################
        # Reshape and apply selections
        ##############################

        sel_all = selection.all(*selection.names) if len(selection.names) else np.ones(len(events)).astype(bool)

        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        dataframe = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(dataframe, fname)

        print("Return ", f"{time.time() - start:.2f}")
        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def add_weights(
        self,
        events,
        year,
        dataset,
        gen_weights,
        gen_selected,
    ) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights = Weights(len(events), storeIndividual=True)
        weights.add("genweight", gen_weights)

        logger.debug("weights", extra=weights._weights.keys())

        ###################### Save all the weights and variations ######################

        # these weights should not change the overall normalization, so are saved separately
        norm_preserving_weights = HH4b.hh_vars.norm_preserving_weights

        # dictionary of all weights and variations
        weights_dict = {}
        # dictionary of total # events for norm preserving variations for normalization in postprocessing
        totals_dict = {}

        # nominal
        weights_dict["weight"] = weights.weight()

        # norm preserving weights, used to do normalization in post-processing
        weight_np = weights.partial_weight(include=norm_preserving_weights)
        totals_dict["np_nominal"] = np.sum(weight_np[gen_selected])

        ###################### Normalization (Step 1) ######################

        weight_norm = self.get_dataset_norm(year, dataset)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.weight()

        return weights_dict, totals_dict
