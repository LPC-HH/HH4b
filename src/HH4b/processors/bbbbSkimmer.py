"""
Skimmer for bbbb analysis with FatJets.
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
from .corrections import (
    add_pileup_weight,
    add_VJets_kFactors,
    add_top_pt_weight,
    add_ps_weight,
    add_pdf_weight,
    add_scalevar_7pt,
    get_jec_jets,
    get_jmsr,
    get_jetveto_event,
    add_trig_weights,
)
from .common import LUMI, jec_shifts, jmsr_shifts
from .objects import *
from . import common

import time


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "GluGlutoHHto4B": gen_selection_HHbbbb,
}


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class bbbbSkimmer(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {
        "Jet": {
            **P4,
        },
        "Muon": {**P4},
        "Electron": {**P4},
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "Txbb": "PNetXbb",
            "Txjj": "PNetXjj",
            "particleNet_mass": "PNetMass",
        },
        "GenHiggs": P4,
        "Event": {
            "run",
            "event",
            "lumi",
        },
        "Pileup": {
            "nPU",
        },
        "Other": {
            # "MET_pt": "MET_pt",
        },
    }

    preselection = {
        "fatjet_pt": 200,
        "fatjet_msd": 60,
        "Txbb0": 0.8,
    }

    jecs = common.jecs

    def __init__(self, xsecs={}, save_systematics=False, region="signal", save_array=False):
        super(bbbbSkimmer, self).__init__()

        self.XSECS = xsecs  # in pb

        # HLT selection
        HLTs = {
            "signal": {
                "2022": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet425_SoftDropMass40",
                ],
                "2022EE": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet425_SoftDropMass40",
                ],
            },
        }

        self.HLTs = HLTs[region]

        self._systematics = save_systematics

        """
        signal: 
        """
        self._region = region

        self._accumulator = processor.dict_accumulator({})

        # save arrays (for dask accumulator)
        self._save_array = save_array

        logger.info(f"Running skimmer with systematics {self._systematics}")

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

        start = time.time()
        # print("Starting")

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])

        isData = not hasattr(events, "genWeight")
        isSignal = "HHTobbbb" in dataset

        if isSignal:
            # take only signs of gen-weights for HH samples
            # TODO: corss check
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
        # TODO: this is tricky, should we apply JEC first and then selection (including vetoes)
        jets, jec_shifted_jetvars = get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            self.jecs,
            fatjets=False,
            applyData=True,
            dataset=dataset,
        )
        jets_sel = good_ak4jets(jets, year, events.run.to_numpy(), isData)
        jets = jets[jets_sel]
        ht = ak.sum(jets.pt, axis=1)

        num_fatjets = 2  # number to save
        num_fatjets_cut = 2  # number to consider for selection
        fatjets = get_ak8jets(events.FatJet)
        fatjets, jec_shifted_fatjetvars = get_jec_jets(
            events, fatjets, year, isData, self.jecs, fatjets=True, applyData=True, dataset=dataset
        )
        fatjets_sel = good_ak8jets(fatjets)
        fatjets = fatjets[fatjets_sel]

        jmsr_shifted_vars = get_jmsr(fatjets, num_fatjets, year, isData)

        # Sort fatjets by PNXbb
        # fatjets = fatjets[ak.argsort(fatjets.Txbb, ascending=False)]

        num_leptons = 2
        veto_muon_sel = good_muons(events.Muon, selection=veto_muon_selection_run2_bbbb)
        veto_electron_sel = good_electrons(
            events.Electron, selection=veto_electron_selection_run2_bbbb
        )
        # muons = events.Muon[good_muons(events.Muon)]
        # electrons = events.Electron[good_electrons(events.Electron)]

        # print("Objects", f"{time.time() - start:.2f}")

        #########################
        # Save / derive variables
        #########################

        # gen variables - saving HH and bbbb 4-vector info
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](
                    events, jets, fatjets, selection, cutflow, gen_weights, P4
                )
                genVars = {**genVars, **vars_dict}

        # Jet variables
        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        # Jet JEC variables
        for var in ["pt"]:
            key = self.skim_vars["Jet"][var]
            for shift, vals in jec_shifted_jetvars[var].items():
                if shift != "":
                    ak4JetVars[f"ak4Jet{key}_{shift}"] = pad_val(vals, num_jets, axis=1)

        # FatJet variables
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        # FatJet JEC variables
        for var in ["pt"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jec_shifted_fatjetvars[var].items():
                if shift != "":
                    ak8FatJetVars[f"ak8FatJet{key}_{shift}"] = pad_val(vals, num_fatjets, axis=1)

        # JMSR variables
        """
        for var in ["msoftdrop", "particleNet_mass"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jmsr_shifted_vars[var].items():
                # overwrite saved mass vars with corrected ones
                label = "" if shift == "" else "_" + shift
                ak8FatJetVars[f"ak8FatJet{key}{label}"] = vals
        """

        # dijet variables
        """
        fatDijetVars = {}
        for shift in jec_shifted_fatjetvars["pt"]:
            label = "" if shift == "" else "_" + shift
            fatDijetVars = {**fatDijetVars, **self.getFatDijetVars(ak8FatJetVars, pt_shift=label)}

        for shift in jmsr_shifted_vars["msoftdrop"]:
            if shift != "":
                label = "_" + shift
                fatDijetVars = {
                    **fatDijetVars,
                    **self.getFatDijetVars(ak8FatJetVars, mass_shift=label),
                }
        """

        # lepton variables
        """
        lepton_skim = {"Muon": muons, "Electron": electrons}
        leptonVars = {
            f"{label}{key}": pad_val(obj[var], num_leptons, axis=1)
            for (label, obj) in lepton_skim.items()
            for (var, key) in self.skim_vars[label].items()
        }
        """

        eventVars = {
            key: events[key].to_numpy() for key in self.skim_vars["Event"] if key in events.fields
        }

        eventVars["ht"] = ht.to_numpy()
        eventVars["nJets"] = ak.sum(jets_sel, axis=1).to_numpy()
        eventVars["nFatJets"] = ak.sum(fatjets_sel, axis=1).to_numpy()

        if isData:
            pileupVars = {key: np.ones(len(events)) * PAD_VAL for key in self.skim_vars["Pileup"]}
        else:
            eventVars["lumi"] = np.ones(len(events)) * PAD_VAL
            pileupVars = {key: events.Pileup[key].to_numpy() for key in self.skim_vars["Pileup"]}

        pileupVars = {**pileupVars, **{"nPV": events.PV["npvs"].to_numpy()}}

        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["Other"].items()
        }

        HLTs = self.HLTs[year]
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
            # **ak4JetVars,
            **ak8FatJetVars,
            # **fatDijetVars,
            # **leptonVars,
            **eventVars,
            **pileupVars,
            **otherVars,
            **HLTVars,
        }

        # print("Vars", f"{time.time() - start:.2f}")

        ######################
        # Selection
        ######################

        # OR-ing HLT triggers
        for trigger in self.HLTs[year]:
            if trigger not in events.HLT.fields:
                logger.warning(f"Missing HLT {trigger}!")

        HLT_triggered = np.any(
            np.array(
                [events.HLT[trigger] for trigger in self.HLTs[year] if trigger in events.HLT.fields]
            ),
            axis=0,
        )

        # apply trigger to both data and simulation
        # add_selection("trigger", HLT_triggered, *selection_args)

        # temporary metfilters https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_3_recommendations
        met_filters = [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ]
        metfilters = np.ones(len(events), dtype="bool")
        metfilterkey = "data" if isData else "mc"
        for mf in met_filters:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        # add_selection("met_filters", metfilters, *selection_args)

        # jet veto maps
        if year == "2022" or year == "2022EE":
            jetveto_selection = get_jetveto_event(jets, year, events.run.to_numpy(), isData)
            add_selection("ak4_jetveto", jetveto_selection, *selection_args)

        # pt cuts: check if fatjet passes pt cut in any of the JEC variations
        cuts = []
        for pts in jec_shifted_fatjetvars["pt"].values():
            cut = np.prod(
                pad_val(
                    (pts > self.preselection["fatjet_pt"]),
                    num_fatjets_cut,
                    False,
                    axis=1,
                ),
                axis=1,
            )
            cuts.append(cut)
        add_selection("ak8_pt", np.any(cuts, axis=0), *selection_args)

        # mass cuts: check if fatjet passes mass cut in any of the JMS/R variations
        cuts = []
        for shift in jmsr_shifted_vars["msoftdrop"]:
            msds = jmsr_shifted_vars["msoftdrop"][shift]
            cut = np.prod(
                pad_val(msds > self.preselection["fatjet_msd"], num_fatjets_cut, False, axis=1),
                axis=1,
            )
            cuts.append(cut)
        # add_selection("ak8_msd", np.any(cuts, axis=0), *selection_args)

        # veto leptons
        add_selection(
            "0lep",
            (ak.sum(veto_muon_sel, axis=1) == 0) & (ak.sum(veto_electron_sel, axis=1) == 0),
            *selection_args,
        )

        # Txbb pre-selection cut on leading jet
        txbb_cut = ak8FatJetVars["ak8FatJetPNetXbb"][:, 0] >= self.preselection["Txbb0"]
        # add_selection("ak8bb_txbb0", txbb_cut, *selection_args)

        # print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights.add("genweight", gen_weights)

            add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy())

            # add_VJets_kFactors(weights, events.GenPart, dataset)

            add_trig_weights(weights, fatjets, year, num_fatjets_cut)

            # if dataset.startswith("TTTo"):
            #     # TODO: need to add uncertainties and rescale yields (?)
            #     add_top_pt_weight(weights, events)

            # TODO: figure out which of these apply to VBF, single Higgs, ttbar etc.

            """
            if "GluGlutoHHto4B" in dataset or "WJets" in dataset or "ZJets" in dataset:
                add_ps_weight(weights, events.PSWeight)

            if "GluGlutoHHto4B" in dataset:
                if "LHEPdfWeight" in events.fields:
                    add_pdf_weight(weights, events.LHEPdfWeight)
                else:
                    add_pdf_weight(weights, [])
                if "LHEScaleWeight" in events.fields:
                    add_scalevar_7pt(weights, events.LHEScaleWeight)
                else:
                    add_scalevar_7pt(weights, [])
            """

            # xsec and luminosity and normalization
            # this still needs to be normalized with the acceptance of the pre-selection (done in post processing)
            if dataset in self.XSECS:
                xsec = self.XSECS[dataset]
                weight_norm = xsec * LUMI[year]
            else:
                logger.warning("Weight not normalized to cross section")
                weight_norm = 1

            systematics = [""]
            if self._systematics:
                systematics += list(weights.variations)

            # TODO: need to be careful about the sum of gen weights used for the LHE/QCDScale uncertainties
            logger.debug("weights ", weights._weights.keys())

            # TEMP: save each individual weight
            for key in weights._weights.keys():
                skimmed_events[f"single_weight_{key}"] = weights.partial_weight([key])

            for systematic in systematics:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)
                    weight_name = f"weight_{systematic}"
                elif systematic == "":
                    weight = weights.weight()
                    weight_name = "weight"

                # includes genWeight (or signed genWeight)
                skimmed_events[weight_name] = weight * weight_norm

                if systematic == "":
                    # to check in postprocessing for xsec & lumi normalisation
                    skimmed_events["weight_noxsec"] = weight

        # reshape and apply selections
        sel_all = selection.all(*selection.names)

        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        df = self.to_pandas(skimmed_events)

        # print("To Pandas", f"{time.time() - start:.2f}")

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        # print("Dump table", f"{time.time() - start:.2f}")

        if self._save_array:
            output = {}
            for key in df.columns:
                if isinstance(key, tuple):
                    column = "".join(str(k) for k in key)
                output[column] = processor.column_accumulator(df[key].values)

            # print("Save Array", f"{time.time() - start:.2f}")
            return {
                "array": output,
                "pkl": {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}},
            }
        else:
            # print("Return ", f"{time.time() - start:.2f}")
            return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def getFatDijetVars(self, ak8FatJetVars: Dict, pt_shift: str = None, mass_shift: str = None):
        """Calculates Dijet variables for given pt / mass JEC / JMS/R variation"""
        dijetVars = {}

        ptlabel = pt_shift if pt_shift is not None else ""
        mlabel = mass_shift if mass_shift is not None else ""

        jets = vector.array(
            {
                "pt": ak8FatJetVars[f"ak8FatJetPt{ptlabel}"],
                "phi": ak8FatJetVars["ak8FatJetPhi"],
                "eta": ak8FatJetVars["ak8FatJetEta"],
                "M": ak8FatJetVars[f"ak8FatJetPNetMass{mlabel}"],
            }
        )

        # get dijet with two first fatjets
        # TODO: probably want to use first two ordered by Txbb in the future?
        jet0 = jets[:, 0]
        jet1 = jets[:, 1]
        Dijet = jet0 + jet1

        shift = ptlabel + mlabel

        dijetVars[f"DijetPt{shift}"] = Dijet.pt
        dijetVars[f"DijetMass{shift}"] = Dijet.M
        dijetVars[f"DijetEta{shift}"] = Dijet.eta
        dijetVars[f"DijetPtJ2overPtJ1{shift}"] = jet1.pt / jet0.pt
        dijetVars[f"DijetMJ2overMJ1{shift}"] = jet1.M / jet0.M

        if shift == "":
            dijetVars["DijetDeltaEta"] = abs(jet0.eta - jet1.eta)
            dijetVars["DijetDeltaPhi"] = jet0.deltaRapidityPhi(jet1)
            dijetVars["DijetDeltaR"] = jet0.deltaR(jet1)

        return dijetVars
