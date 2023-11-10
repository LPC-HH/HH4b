"""
Skimmer for bbbb analysis with FatJets.
Author(s): Raghav Kansal, Cristina Suarez
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from copy import deepcopy

import awkward as ak
import numpy as np
import vector
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

from . import common, objects
from .common import LUMI
from .corrections import (
    add_pileup_weight,
    add_trig_weights,
    get_jec_jets,
    # get_jmsr,
    get_jetveto_event,
)
from .GenSelection import gen_selection_Hbb, gen_selection_HHbbbb
from .utils import P4, PAD_VAL, add_selection, dump_table, pad_val, to_pandas

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {"HHto4B": gen_selection_HHbbbb, "HToBB": gen_selection_Hbb}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class bbbbSkimmer(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {  # noqa: RUF012
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

    preselection = {  # noqa: RUF012
        "fatjet_pt": 300,
        "fatjet_msd": 40,
        "fatjet_mreg": 40,
        "Txbb0": 0.8,
    }

    jecs = common.jecs

    def __init__(
        self,
        xsecs=None,
        save_systematics=False,
        region="signal",
        save_array=False,
        nano_version="v12",
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb

        # HLT selection
        HLTs = {
            "signal": {
                "2018": [
                    "PFJet500",
                    #
                    "AK8PFJet500",
                    #
                    "AK8PFJet360_TrimMass30",
                    "AK8PFJet380_TrimMass30",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFHT750_TrimMass50",
                    "AK8PFHT800_TrimMass50",
                    #
                    "PFHT1050",
                    #
                    "AK8PFJet330_TrimMass30_PFAK8BTagCSV_p17_v",
                ],
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

        self._nano_version = nano_version

        """
        signal:
        """
        self._region = region

        self._accumulator = processor.dict_accumulator({})

        # save arrays (for dask accumulator)
        self._save_array = save_array

        logger.info(f"Running skimmer with systematics {self._systematics}")

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
        isSignal = "HHTobbbb" in dataset or "HHto4B" in dataset

        if isSignal:
            # take only signs of gen-weights for HH samples
            # TODO: cross check when new samples arrive
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
        print("Starting Objects", f"{time.time() - start:.2f}")
        # TODO: this is tricky, should we apply JEC first and then selection (including vetoes)
        jets, jec_shifted_jetvars = get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            # jecs=self.jecs,
            jecs=None,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
        print("ak4 JECs", f"{time.time() - start:.2f}")
        jets_sel = objects.good_ak4jets(jets, year, events.run.to_numpy(), isData)
        print("ak4 Jets", f"{time.time() - start:.2f}")
        jets = jets[jets_sel]
        ht = ak.sum(jets.pt, axis=1)

        num_fatjets = 2  # number to save
        num_fatjets_cut = 2  # number to consider for selection
        fatjets = objects.get_ak8jets(events.FatJet)
        print("ak8 jets", f"{time.time() - start:.2f}")
        fatjets, jec_shifted_fatjetvars = get_jec_jets(
            events,
            fatjets,
            year,
            isData,
            # jecs=self.jecs,
            jecs=None,
            fatjets=True,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
        print("ak8 JECs", f"{time.time() - start:.2f}")
        fatjets_sel = objects.good_ak8jets(fatjets)
        print("good ak8 jets", f"{time.time() - start:.2f}")
        fatjets = fatjets[fatjets_sel]

        # jmsr_shifted_vars = get_jmsr(fatjets, num_fatjets, year, isData)

        #########################
        # Save / derive variables
        #########################

        # gen variables - saving HH and bbbb 4-vector info
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, jets, fatjets, selection_args, P4)
                genVars = {**genVars, **vars_dict}

        # Jet variables
        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        # FatJet variables
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        print("Jet vars", f"{time.time() - start:.2f}")

        """
        # Jet JEC variables
        for var in ["pt"]:
            key = self.skim_vars["Jet"][var]
            for shift, vals in jec_shifted_jetvars[var].items():
                if shift != "":
                    ak4JetVars[f"ak4Jet{key}_{shift}"] = pad_val(vals, num_jets, axis=1)

        # FatJet JEC variables
        for var in ["pt"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jec_shifted_fatjetvars[var].items():
                if shift != "":
                    ak8FatJetVars[f"ak8FatJet{key}_{shift}"] = pad_val(vals, num_fatjets, axis=1)

        # JMSR variables
        for var in ["msoftdrop", "particleNet_mass"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jmsr_shifted_vars[var].items():
                # overwrite saved mass vars with corrected ones
                label = "" if shift == "" else "_" + shift
                ak8FatJetVars[f"ak8FatJet{key}{label}"] = vals

        # dijet variables
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

        eventVars = {
            key: events[key].to_numpy() for key in self.skim_vars["Event"] if key in events.fields
        }

        eventVars["ht"] = ht.to_numpy()
        eventVars["nJets"] = ak.sum(jets_sel, axis=1).to_numpy()
        eventVars["nFatJets"] = ak.sum(fatjets_sel, axis=1).to_numpy()

        print("Event vars", f"{time.time() - start:.2f}")

        if isData:
            pileupVars = {key: np.ones(len(events)) * PAD_VAL for key in self.skim_vars["Pileup"]}
        else:
            eventVars["lumi"] = np.ones(len(events)) * PAD_VAL
            pileupVars = {key: events.Pileup[key].to_numpy() for key in self.skim_vars["Pileup"]}

        pileupVars = {**pileupVars, "nPV": events.PV["npvs"].to_numpy()}

        print("pileup vars", f"{time.time() - start:.2f}")

        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["Other"].items()
        }

        HLTs = deepcopy(self.HLTs[year])
        if year != "2018":
            # add extra hlts as variables
            HLTs.extend(
                [
                    "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                    "PFHT1050",
                    "AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet275_SoftDropMass40_PFAK8ParticleNetBB0p35",
                ]
            )

        zeros = np.zeros(len(events), dtype="bool")
        HLTVars = {
            trigger: (
                events.HLT[trigger].to_numpy().astype(int)
                if trigger in events.HLT.fields
                else zeros
            )
            for trigger in HLTs
        }

        print("HLT vars", f"{time.time() - start:.2f}")

        skimmed_events = {
            **genVars,
            **ak4JetVars,
            **ak8FatJetVars,
            **eventVars,
            **pileupVars,
            **otherVars,
            **HLTVars,
        }

        print("Vars", f"{time.time() - start:.2f}")

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
        apply_trigger = True
        if year == "2018" and not isData:
            apply_trigger = False
        if apply_trigger:
            add_selection("trigger", HLT_triggered, *selection_args)

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
        # metfilterkey = "data" if isData else "mc"
        for mf in met_filters:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        # add_selection("met_filters", metfilters, *selection_args)

        # jet veto maps
        if year == "2022" or year == "2022EE":
            jetveto_selection = get_jetveto_event(jets, year, events.run.to_numpy(), isData)
            add_selection("ak4_jetveto", jetveto_selection, *selection_args)

        # TODO: check if fatjet passes pt cut in any of the JEC variations
        cut = np.sum(ak8FatJetVars["ak8FatJetPt"] >= self.preselection["fatjet_pt"], axis=1)
        add_selection("ak8_pt", cut, *selection_args)

        # TODO: check if fatjet passes mass cut in any of the JMS/R variations
        # cut_mpnet = np.all(
        #     ak8FatJetVars["ak8FatJetPNetMass"] >= self.preselection["fatjet_mreg"], axis=1
        # )
        cut_msd = np.all(ak8FatJetVars["ak8FatJetMsd"] >= self.preselection["fatjet_msd"], axis=1)
        add_selection("ak8_msd", cut_msd, *selection_args)

        # num_leptons = 2
        if year == "2018":
            veto_muon_sel = objects.veto_muons_run2(events.Muon)
            veto_electron_sel = objects.veto_electrons_run2(events.Electron)
        else:
            veto_muon_sel = objects.veto_muons(events.Muon)
            veto_electron_sel = objects.veto_electrons(events.Electron)

        print("Lepton vetoes", f"{time.time() - start:.2f}")

        # veto leptons
        add_selection(
            "0lep",
            (ak.sum(veto_muon_sel, axis=1) == 0) & (ak.sum(veto_electron_sel, axis=1) == 0),
            *selection_args,
        )

        # Txbb pre-selection cut
        # txbb_cut = np.sum(ak8FatJetVars["ak8FatJetPNetXbb"] >= self.preselection["Txbb0"], axis=1)
        # add_selection("ak8bb_txbb0", txbb_cut, *selection_args)

        print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights.add("genweight", gen_weights)

            add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy(), dataset)

            add_trig_weights(weights, fatjets, year, num_fatjets_cut)

            # add_VJets_kFactors(weights, events.GenPart, dataset)

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
            logger.debug("weights", extra=weights._weights.keys())

            # TEMP: save each individual weight
            for key in weights._weights:
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

        dataframe = to_pandas(skimmed_events)

        print("To Pandas", f"{time.time() - start:.2f}")

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        dump_table(dataframe, fname)

        print("Dump table", f"{time.time() - start:.2f}")

        if self._save_array:
            output = {}
            for key in dataframe.columns:
                if isinstance(key, tuple):
                    column = "".join(str(k) for k in key)
                output[column] = processor.column_accumulator(dataframe[key].values)

            # print("Save Array", f"{time.time() - start:.2f}")
            return {
                "array": output,
                "pkl": {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}},
            }

        print("Return ", f"{time.time() - start:.2f}")
        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def getFatDijetVars(
        self, ak8FatJetVars: dict, pt_shift: str | None = None, mass_shift: str | None = None
    ):
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
