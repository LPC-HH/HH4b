"""
Skimmer for simple analysis with FatJets.
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
from .GenSelection import gen_selection_V
from .objects import (
    get_ak8jets,
    veto_electrons,
    veto_taus,
)
from .SkimmerABC import SkimmerABC
from .utils import P4, add_selection, pad_val


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "Zto2Q": gen_selection_V,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class simpleSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {  # noqa: RUF012
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "Txbb": "PNetTXbb",
            "Txjj": "PNetTXjj",
            "Tqcd": "PNetQCD",
            "PQCDb": "PNetQCD1HF",
            "PQCDbb": "PNetQCD2HF",
            "PQCDothers": "PNetQCD0HF",
            "particleNet_mass": "PNetMass",
            "particleNet_massraw": "PNetMassRaw",
            "t21": "Tau2OverTau1",
            "t32": "Tau3OverTau2",
            "rawFactor": "rawFactor",
            "msoftdrop": "msoftdrop",
            "particleNet_mass": "particleNet_mass",
        },
    }

    def __init__(
        self,
        xsecs=None,
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_3_recommendations
        self.met_filters = [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
            "hfNoisyHitsFilt",
        ]

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
        print("Starting object definition", f"{time.time() - start:.2f}")

        num_fatjets = 2  # number to save
        fatjets = get_ak8jets(events.FatJet)

        ak4_jets = events.Jet
        
        print("Object definition", f"{time.time() - start:.2f}")

        #########################
        # Derive variables
        #########################

        # Gen variables - saving HH and bbbb 4-vector info
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, ak4_jets, fatjets, selection_args, P4)
                genVars = {**genVars, **vars_dict}

        # used for normalization to cross section below
        gen_selected = (
            selection.all(*selection.names)
            if len(selection.names)
            else np.ones(len(events)).astype(bool)
        )

        # FatJet variables
        fatjet_skimvars = self.skim_vars["FatJet"]
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, axis=1)
            for (var, key) in fatjet_skimvars.items()
        }

        print("FatJet vars", f"{time.time() - start:.2f}")


        HLTs = [
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
            "PFHT1050",
            "AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet275_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet230_SoftDropMass40",
            "AK8PFJet425_SoftDropMass40",
            "AK8PFJet400_SoftDropMass40",
            "AK8DiPFJet250_250_MassSD50",
            "AK8DiPFJet260_260_MassSD30",
            "AK8PFJet420_MassSD30",
            "AK8PFJet230_SoftDropMass40_PNetBB0p06",
            "AK8PFJet230_SoftDropMass40_PNetBB0p10",
            "AK8PFJet250_SoftDropMass40_PNetBB0p06",
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
            **ak8FatJetVars,
            **HLTVars,
        }

        print("Vars", f"{time.time() - start:.2f}")

        #########################
        # Selection Starts
        #########################
        # at least one good reconstructed primary vertex
        add_selection("npvsGood", events.PV.npvsGood >= 1, *selection_args)

        # AK8 jet
        fatjet_selector = (
            (fatjets.pt > 250)
            * (np.abs(fatjets.eta) < 2.4)
            * fatjets.isTight
        )
        fatjet_selector = ak.any(fatjet_selector, axis=1)

        add_selection("ak8_jet", fatjet_selector, *selection_args)

        # metfilters
        cut_metfilters = np.ones(len(events), dtype="bool")
        for mf in self.met_filters:
            if mf in events.Flag.fields:
                cut_metfilters = cut_metfilters & events.Flag[mf]
        add_selection("met_filters", cut_metfilters, *selection_args)

        print("Selection", f"{time.time() - start:.2f}")

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

        sel_all = selection.all(*selection.names)

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
