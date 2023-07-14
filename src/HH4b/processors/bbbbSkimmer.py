"""
Skimmer for bbbb analysis.
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
from .utils import pad_val, add_selection, concatenate_dicts, select_dicts, P4
from .corrections import (
    add_pileup_weight,
    add_VJets_kFactors,
    add_top_pt_weight,
    add_ps_weight,
    add_pdf_weight,
    add_scalevar_7pt,
    add_trig_effs,
    get_jec_key,
    get_jec_jets,
    get_jmsr,
)
from .common import LUMI, HLTs, jec_shifts, jmsr_shifts
from . import common


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "GluGluToHHTobbbb_node_cHHH": gen_selection_HHbbbb,
}


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class bbbbSkimmer(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data), for preliminary cut-based analysis and BDT studies.

    Args:
        xsecs (dict, optional): sample cross sections,
          if sample not included no lumi and xsec will not be applied to weights
        save_ak15 (bool, optional): save ak15 jets as well, for HVV candidate
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "particleNetMD_QCD": "ParticleNetMD_QCD",
            "particleNetMD_Xbb": "ParticleNetMD_Xbb",
            "particleNet_mass": "ParticleNetMass",
        },
        "GenHiggs": P4,
        "other": {"MET_pt": "MET_pt"},
    }

    preselection = {
        "pt": 200.0,
        "eta": 2.4,
        "msd": 50,
        # "mreg": 50,
        # "bbFatJetParticleNetMD_Txbb": 0.8,
    }

    jecs = common.jecs
    
    def __init__(
        self, xsecs={}, save_systematics=True, inference=True
    ):
        super(bbbbSkimmer, self).__init__()

        self.XSECS = xsecs  # in pb

        # save systematic variations
        self._systematics = save_systematics

        # run inference
        self._inference = inference

        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        self._accumulator = processor.dict_accumulator({})

        logger.info(
            f"Running skimmer with inference {self._inference} and systematics {self._systematics}"
        )

    def to_pandas(self, events: Dict[str, np.array]):
        """
        Convert our dictionary of numpy arrays into a pandas data frame
        Uses multi-index columns for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        return pd.concat(
            # [pd.DataFrame(v.reshape(v.shape[0], -1)) for k, v in events.items()],
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

        isData = "JetHT" in dataset
        isQCD = "QCD" in dataset
        isSignal = "HHTobbbb" in dataset

        if isSignal:
            # only signs for HH
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

        num_jets = 2
        jemc_year = "2018"
        fatjets, jec_shifted_vars = get_jec_jets(events, jemc_year, isData, self.jecs)

        # change to year with suffix after updated JMS/R values
        jmsr_shifted_vars = get_jmsr(fatjets, num_jets, jemc_year, isData)

        skimmed_events = {}

        #########################
        # Save / derive variables
        #########################

        # gen vars - saving HH and bbbb 4-vector info
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](
                    events, fatjets, selection, cutflow, gen_weights, P4
                )
                skimmed_events = {**skimmed_events, **vars_dict}

        # FatJet vars

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        # JEC vars

        for var in ["pt"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jec_shifted_vars[var].items():
                if shift != "":
                    ak8FatJetVars[f"ak8FatJet{key}_{shift}"] = pad_val(vals, num_jets, axis=1)

        # JMSR vars

        for var in ["msoftdrop", "particleNet_mass"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jmsr_shifted_vars[var].items():
                # overwrite saved mass vars with corrected ones
                label = "" if shift == "" else "_" + shift
                ak8FatJetVars[f"ak8FatJet{key}{label}"] = vals

        # particlenet xbb vs qcd

        fatjets["Txbb"] = fatjets.particleNetMD_Xbb / (
            fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb
        )

        ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"] = pad_val(
            fatjets["Txbb"],
            num_jets,
            axis=1,
        )

        # dijet variables

        dijetVars = {}

        for shift in jec_shifted_vars["pt"]:
            label = "" if shift == "" else "_" + shift
            dijetVars = {**dijetVars, **self.getDijetVars(ak8FatJetVars, bb_mask, pt_shift=label)}

        for shift in jmsr_shifted_vars["msoftdrop"]:
            if shift != "":
                label = "_" + shift
                dijetVars = {
                    **dijetVars,
                    **self.getDijetVars(ak8FatJetVars, bb_mask, mass_shift=label),
                }

        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["other"].items()
        }

        skimmed_events = {**skimmed_events, **ak8FatJetVars, **dijetVars, **otherVars}

        ######################
        # Selection
        ######################

        # OR-ing HLT triggers
        if isData:
            for trigger in HLTs[year]:
                if trigger not in events.HLT.fields:
                    logger.warning(f"Missing HLT {trigger}!")

            HLT_triggered = np.any(
                np.array(
                    [
                        events.HLT[trigger]
                        for trigger in HLTs[year]
                        if trigger in events.HLT.fields
                    ]
                ),
                axis=0,
            )
            add_selection("trigger", HLT_triggered, *selection_args)

        # pt, eta cuts: check if jet passes pT cut in any of the JEC variations
        cuts = []

        for pts in jec_shifted_vars["pt"].values():
            cut = np.prod(
                pad_val(
                    (pts > self.preselection["pt"])
                    * (np.abs(fatjets.eta) < self.preselection["eta"]),
                    num_jets,
                    False,
                    axis=1,
                ),
                axis=1,
            )
            cuts.append(cut)

        add_selection("ak8_pt_eta", np.any(cuts, axis=0), *selection_args)

        # mass cuts: check if jet passes mass cut in any of the JMS/R variations
        cuts = []

        for shift in jmsr_shifted_vars["msoftdrop"]:
            msds = jmsr_shifted_vars["msoftdrop"][shift]
            cut = np.prod(
                pad_val(
                    msds > self.preselection["msd"],
                    num_jets,
                    False,
                    axis=1
                ),
                axis=1,
            )
            cuts.append(cut)

        add_selection("ak8_mass", np.any(cuts, axis=0), *selection_args)

        # TODO: dijet mass: check if dijet mass cut passes in any of the JEC or JMC variations

        # Txbb pre-selection cut

        # txbb_cut = (
        #     ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"][bb_mask]
        #     >= self.preselection["bbFatJetParticleNetMD_Txbb"]
        # )
        # add_selection("ak8bb_txbb", txbb_cut, *selection_args)
        

        ######################
        # Weights
        ######################

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights.add("genweight", gen_weights)

            add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy())
            add_VJets_kFactors(weights, events.GenPart, dataset)

            # if dataset.startswith("TTTo"):
            #     # TODO: need to add uncertainties and rescale yields (?)
            #     add_top_pt_weight(weights, events)

            # TODO: figure out which of these apply to VBF, single Higgs, ttbar etc.

            if "GluGluToHHTobbbb" in dataset or "WJets" in dataset or "ZJets" in dataset:
                add_ps_weight(weights, events.PSWeight)

            if "GluGluToHHTobbbb" in dataset:
                if "LHEPdfWeight" in events.fields:
                    add_pdf_weight(weights, events.LHEPdfWeight)
                else:
                    add_pdf_weight(weights, [])
                if "LHEScaleWeight" in events.fields:
                    add_scalevar_7pt(weights, events.LHEScaleWeight)
                else:
                    add_scalevar_7pt(weights, [])

            # add_trig_effs(weights, fatjets, year, num_jets)

            # xsec and luminosity and normalization
            # this still needs to be normalized with the acceptance of the pre-selection (done in post processing)
            if dataset in self.XSECS:
                xsec = self.XSECS[dataset]
                weight_norm = xsec * LUMI[year]
            else:
                logger.warning("Weight not normalized to cross section")
                weight_norm = 1

            systematics = [""]
            # systematics = ["", "notrigeffs"]

            if self._systematics:
                systematics += list(weights.variations)

            # TODO: need to be careful about the sum of gen weights used for the LHE/QCDScale uncertainties
            logger.debug("weights ", weights._weights.keys())
            for systematic in systematics:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)
                    weight_name = f"weight_{systematic}"
                elif systematic == "":
                    weight = weights.weight()
                    weight_name = "weight"
                elif systematic == "notrigeffs":
                    weight = weights.partial_weight(exclude=["trig_effs"])
                    weight_name = "weight_noTrigEffs"

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

        bb_mask = bb_mask[sel_all]

        df = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def getDijetVars(
        self, ak8FatJetVars: Dict, bb_mask: np.ndarray, pt_shift: str = None, mass_shift: str = None
    ):
        """Calculates Dijet variables for given pt / mass JEC / JMS/R variation"""
        dijetVars = {}

        ptlabel = pt_shift if pt_shift is not None else ""
        mlabel = mass_shift if mass_shift is not None else ""
        bbJet = vector.array(
            {
                "pt": ak8FatJetVars[f"ak8FatJetPt{ptlabel}"][bb_mask],
                "phi": ak8FatJetVars["ak8FatJetPhi"][bb_mask],
                "eta": ak8FatJetVars["ak8FatJetEta"][bb_mask],
                "M": ak8FatJetVars[f"ak8FatJetParticleNetMass{mlabel}"][bb_mask],
            }
        )

        VVJet = vector.array(
            {
                "pt": ak8FatJetVars[f"ak8FatJetPt{ptlabel}"][~bb_mask],
                "phi": ak8FatJetVars["ak8FatJetPhi"][~bb_mask],
                "eta": ak8FatJetVars["ak8FatJetEta"][~bb_mask],
                "M": ak8FatJetVars[f"ak8FatJetMsd{mlabel}"][~bb_mask],
                # TODO: change this to ParticleNetMass for next run
                # "M": ak8FatJetVars[f"ak8FatJetParticleNetMass{mlabel}"][~bb_mask],
            }
        )

        Dijet = bbJet + VVJet

        shift = ptlabel + mlabel

        dijetVars[f"DijetPt{shift}"] = Dijet.pt
        dijetVars[f"DijetMass{shift}"] = Dijet.M
        dijetVars[f"DijetEta{shift}"] = Dijet.eta

        dijetVars[f"bbFatJetPtOverDijetPt{shift}"] = bbJet.pt / Dijet.pt
        dijetVars[f"VVFatJetPtOverDijetPt{shift}"] = VVJet.pt / Dijet.pt
        dijetVars[f"VVFatJetPtOverbbFatJetPt{shift}"] = VVJet.pt / bbJet.pt

        return dijetVars
