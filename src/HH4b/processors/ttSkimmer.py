"""
Skimmer for tt analysis with FatJets.
Author(s): Billy Li, Raghav Kansal, Cristina Suarez
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
from .corrections import (
    JECs,
    add_pileup_weight,
    get_jmsr,
)
from .GenSelection import gen_selection_Hbb, gen_selection_HHbbbb, gen_selection_Top
from .objects import (
    get_ak8jets,
    veto_electrons,
    veto_taus,
)
from .SkimmerABC import SkimmerABC
from .utils import P4, add_selection, pad_val

MU_PDGID = 13

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {"HHto4B": gen_selection_HHbbbb, "HToBB": gen_selection_Hbb}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "TTto4Q": gen_selection_Top,
    "TTto2L2Nu": gen_selection_Top,
    "TTtoLNu2Q": gen_selection_Top,
}


class ttSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {  # noqa: RUF012
        "Lepton": {
            **P4,
            "id": "Id",
        },
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
        },
        "Event": {
            "run": "run",
            "event": "event",
            "luminosityBlock": "luminosityBlock",
        },
    }

    muon_selection = {  # noqa: RUF012
        "Id": "tight",
        "pt": 55,
        "eta": 2.4,
        "miniPFRelIso_all": 0.1,
        "dxy": 0.2,
        "count": 1,
        "delta_trigObj": 0.15,
    }

    electron_selection = {  # noqa: RUF012
        "count": 0,
    }

    ak8_jet_selection = {  # noqa: RUF012
        "jetId": "tight",
        "pt": 200.0,
        "eta": 2.5,
        "delta_phi_muon": 2,
    }

    ak4_jet_selection = {  # noqa: RUF012
        "jetId": "tight",
        "pt": 25,  # from JME-18-002
        "eta": 2.4,
        "delta_phi_muon": 2,
        "num": 1,
        # "closest_muon_dr": 0.4,
        # "closest_muon_ptrel": 25,
    }

    btag_medium_deepJet = {  # noqa: RUF012
        "2022": 0.3086,
        "2022EE": 0.3196,
        "2023": 0.2431,
        "2023BPix": 0.2435,
    }
    btag_medium_pNet = {  # noqa: RUF012
        "2022": 0.245,
        "2022EE": 0.2605,
        "2023": 0.1917,
        "2023BPix": 0.1919,
    }

    met_selection = {"pt": 50}  # noqa: RUF012

    lepW_selection = {"pt": 100}  # noqa: RUF012

    num_jets = 1

    jecs = {  # noqa: RUF012
        "JES": "JES",
        "JER": "JER",
    }

    def __init__(
        self,
        xsecs=None,
        nano_version="v12",
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb
        self._nano_version = nano_version

        # HLT selection
        self.HLTs = {
            "2022EE": [
                "Mu50",
            ],
            "2022": [
                "Mu50",
            ],
            "2023": [
                "Mu50",
            ],
            "2023BPix": [
                "Mu50",
            ],
        }

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

        # JMSR
        self.jmsr_vars = ["msoftdrop", "particleNet_mass"]
        if self._nano_version == "v12v2_private":
            self.jmsr_vars += ["particleNet_mass_legacy", "ParTmassVis"]
        if self._nano_version == "v12_private":
            self.jmsr_vars += ["particleNet_mass_legacy"]

        # FatJet Vars
        if self._nano_version == "v12_private" or self._nano_version == "v12v2_private":
            extra_vars = [
                "TXbb",
                "PXbb",
                "PQCD",
                "PQCDb",
                "PQCDbb",
                "PQCD0HF",
                "PQCD1HF",
                "PQCD2HF",
            ]
            self.skim_vars["FatJet"] = {
                **self.skim_vars["FatJet"],
                "particleNet_mass_legacy": "PNetMassLegacy",
                **{f"{var}_legacy": f"PNet{var}Legacy" for var in extra_vars},
            }
        if self._nano_version == "v12v2_private":
            extra_vars = [
                "ParTPQCD1HF",
                "ParTPQCD0HF",
                "ParTPQCD2HF",
                "ParTPTopW",
                "ParTPTopbW",
                "ParTPXbb",
                "ParTPXqq",
                "ParTTXbb",
                "ParTmassRes",
                "ParTmassVis",
            ]
            self.skim_vars["FatJet"] = {
                **self.skim_vars["FatJet"],
                **{var: var for var in extra_vars},
            }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        start = time.time()
        print("Starting")
        print("# events", len(events))

        year = events.metadata["dataset"].split("_")[0]
        is_run3 = year in ["2022", "2022EE", "2023", "2023BPix"]
        assert is_run3  # run2 not implemented yet

        dataset = "_".join(events.metadata["dataset"].split("_")[1:])

        isData = not hasattr(events, "genWeight")

        gen_weights = events["genWeight"].to_numpy() if not isData else None

        n_events = len(events) if isData else np.sum(gen_weights)

        cutflow = OrderedDict()
        cutflow["all"] = n_events
        selection = PackedSelection()
        selection_args = (selection, cutflow, isData, gen_weights)

        # JEC factory loader
        JEC_loader = JECs(year)

        #########################
        # Object definitions
        #########################
        print("Starting object definition", f"{time.time() - start:.2f}")

        muon = events.Muon
        muon["id"] = muon.charge * (13)

        ak4_jets, jec_shifted_jetvars = JEC_loader.get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            jecs=self.jecs,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )

        met = JEC_loader.met_factory.build(events.MET, ak4_jets, {}) if isData else events.MET

        num_fatjets = 2  # number to save
        fatjets = get_ak8jets(events.FatJet)
        fatjets, jec_shifted_fatjetvars = JEC_loader.get_jec_jets(
            events,
            fatjets,
            year,
            isData,
            jecs=self.jecs,
            fatjets=True,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )

        # save variations with 10\%
        jmsr_shifted_vars = get_jmsr(
            fatjets,
            num_fatjets,
            jmsr_vars=self.jmsr_vars,
            jms_values={key: [1.0, 0.9, 1.1] for key in self.jmsr_vars},
            jmr_values={key: [1.0, 0.9, 1.1] for key in self.jmsr_vars},
            isData=isData,
        )

        print("Object definition", f"{time.time() - start:.2f}")

        #########################
        # Derive variables
        #########################

        # Gen variables
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
        if not isData:
            fatjet_skimvars = {
                **fatjet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, axis=1)
            for (var, key) in fatjet_skimvars.items()
        }

        for var in self.jmsr_vars:
            key = fatjet_skimvars[var]
            ak8FatJetVars[f"ak8FatJet{key}_raw"] = ak8FatJetVars[f"ak8FatJet{key}"]
            for shift, vals in jmsr_shifted_vars[var].items():
                label = "" if shift == "" else "_" + shift
                ak8FatJetVars[f"ak8FatJet{key}{label}"] = vals

        print("FatJet vars", f"{time.time() - start:.2f}")

        # lepton variables
        num_leptons = 2
        lepton_skimvars = self.skim_vars["Lepton"]
        leptonVars = {
            f"lepton{key}": pad_val(muon[var], num_leptons, axis=1)
            for (var, key) in lepton_skimvars.items()
        }
        print("Lepton vars", f"{time.time() - start:.2f}")

        # event variable
        met_pt = met.pt

        eventVars = {
            key: events[val].to_numpy()
            for key, val in self.skim_vars["Event"].items()
            if key in events.fields
        }
        eventVars["MET_pt"] = met_pt.to_numpy()

        skimmed_events = {
            **genVars,
            **eventVars,
            **ak8FatJetVars,
            **leptonVars,
        }

        print("Vars", f"{time.time() - start:.2f}")

        #########################
        # Selection Starts
        #########################
        add_selection("npvsGood", events.PV.npvsGood >= 1, *selection_args)

        # muon
        muon_selector = (
            (muon[f"{self.muon_selection['Id']}Id"])
            * (muon.pt > self.muon_selection["pt"])
            * (np.abs(muon.eta) < self.muon_selection["eta"])
            * (muon.miniPFRelIso_all < self.muon_selection["miniPFRelIso_all"])
            * (np.abs(muon.dxy) < self.muon_selection["dxy"])
        )
        muon_selector = muon_selector * (
            ak.count(events.Muon.pt[muon_selector], axis=1) == self.muon_selection["count"]
        )
        muon = ak.pad_none(muon[muon_selector], 1, axis=1)[:, 0]
        muon_selector = ak.any(muon_selector, axis=1)

        # match muon to trigger object
        trigObj_muon = events.TrigObj[
            (events.TrigObj.id == MU_PDGID) * (events.TrigObj.filterBits >= 1024)
        ]

        muon_selector = muon_selector * ak.any(
            np.abs(muon.delta_r(trigObj_muon)) <= self.muon_selection["delta_trigObj"],
            axis=1,
        )
        add_selection("muon", muon_selector, *selection_args)

        # veto loose taus/electrons
        veto_electron_sel = veto_electrons(events.Electron)
        veto_tau_sel = veto_taus(events.Tau)
        add_selection(
            "no_other_leptons",
            (ak.sum(veto_electron_sel, axis=1) == 0) & (ak.sum(veto_tau_sel, axis=1) == 0),
            *selection_args,
        )

        # MET
        met_selection = met.pt >= self.met_selection["pt"]

        cut_metfilters = np.ones(len(events), dtype="bool")
        for mf in self.met_filters:
            if mf in events.Flag.fields:
                cut_metfilters = cut_metfilters & events.Flag[mf]

        add_selection("met", met_selection * cut_metfilters, *selection_args)

        # AK4 jet
        ak4_jet_selector = (
            ak4_jets.isTight
            * (ak4_jets.pt > self.ak4_jet_selection["pt"])
            * (np.abs(ak4_jets.eta) < self.ak4_jet_selection["eta"])
        )

        # b-tagged and dPhi from muon < 2
        # consider btagDeepFlavB for 2022 only (problem with v12 private production)
        if year == "2022" and self._nano_version == "v12_private":
            ak4_jet_selector_btag_muon = ak4_jet_selector * (
                (ak4_jets.btagDeepFlavB > self.btag_medium_deepJet[year])
                * (np.abs(ak4_jets.delta_phi(muon)) < self.ak4_jet_selection["delta_phi_muon"])
            )
        else:
            ak4_jet_selector_btag_muon = ak4_jet_selector * (
                (ak4_jets.btagPNetB > self.btag_medium_pNet[year])
                * (np.abs(ak4_jets.delta_phi(muon)) < self.ak4_jet_selection["delta_phi_muon"])
            )
        ak4_selection = (
            # at least 1 b-tagged jet close to the muon
            (ak.any(ak4_jet_selector_btag_muon, axis=1))
            # at least 2 ak4 jets overall
            * (ak.sum(ak4_jet_selector, axis=1) >= self.ak4_jet_selection["num"])
        )

        add_selection("ak4_jet", ak4_selection, *selection_args)

        # AK8 jet
        fatjet_selector = (
            (fatjets.pt > self.ak8_jet_selection["pt"])
            * (np.abs(fatjets.eta) < self.ak8_jet_selection["eta"])
            * (np.abs(fatjets.delta_phi(muon)) > self.ak8_jet_selection["delta_phi_muon"])
            * fatjets.isTight
        )
        fatjet_selector = ak.any(fatjet_selector, axis=1)
        add_selection("ak8_jet", fatjet_selector, *selection_args)

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
        add_selection("trigger", HLT_triggered, *selection_args)

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

        add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy(), dataset)

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

        if self._systematics:
            for systematic in list(weights.variations):
                weights_dict[f"weight_{systematic}"] = weights.weight(modifier=systematic)

                if utils.remove_variation_suffix(systematic) in norm_preserving_weights:
                    var_weight = weights.partial_weight(include=norm_preserving_weights)
                    # modify manually
                    if "Down" in systematic and systematic not in weights._modifiers:
                        var_weight = (
                            var_weight / weights._modifiers[systematic.replace("Down", "Up")]
                        )
                    else:
                        var_weight = var_weight * weights._modifiers[systematic]

                    # var_weight = weights.partial_weight(
                    #    include=norm_preserving_weights, modifier=systematic
                    # )

                    # need to save total # events for each variation for normalization in post-processing
                    totals_dict[f"np_{systematic}"] = np.sum(var_weight[gen_selected])

            # TEMP: save each individual weight TODO: remove
            for key in weights._weights:
                weights_dict[f"single_weight_{key}"] = weights.partial_weight([key])

        ###################### alpha_S and PDF variations ######################

        # TODO

        ###################### Normalization (Step 1) ######################

        weight_norm = self.get_dataset_norm(year, dataset)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.weight()

        return weights_dict, totals_dict
