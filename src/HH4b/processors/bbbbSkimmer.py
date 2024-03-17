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

from .common import LUMI
from .corrections import (
    JECs,
    add_pileup_weight,
    add_trig_weights,
    get_jetveto,
    get_jetveto_event,
)
from .GenSelection import gen_selection_Hbb, gen_selection_HHbbbb
from .objects import (
    get_ak8jets,
    good_ak8jets,
    good_electrons,
    good_muons,
    veto_electrons,
    veto_electrons_run2,
    veto_muons,
    veto_muons_run2,
)
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
            "rawFactor": "rawFactor",
        },
        "Lepton": {
            **P4,
            "id": "Id",
        },
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "Txbb": "PNetXbb",
            "Txjj": "PNetXjj",
            "Tqcd": "PNetQCD",
            "TQCDb": "PNetQCD1HF",
            "TQCDbb": "PNetQCD2HF",
            "TQCDothers": "PNetQCD0HF",
            "particleNet_mass": "PNetMass",
            "particleNet_massraw": "PNetMassRaw",
            "t32": "Tau3OverTau2",
            "rawFactor": "rawFactor",
        },
        "GenHiggs": P4,
        "Event": {
            "run": "run",
            "event": "event",
            "luminosityBlock": "luminosityBlock",
        },
        "Pileup": {
            "nPU",
        },
        "TriggerObject": {
            "pt": "Pt",
            "eta": "Eta",
            "phi": "Phi",
            "filterBits": "Bit",
        },
    }

    preselection = {  # noqa: RUF012
        "fatjet_pt": 270,
        "fatjet_msd": 60,
        "fatjet_mreg": 60,
        "Txbb0": 0.8,
    }

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
                    "AK8PFJet500",
                    "AK8PFJet360_TrimMass30",
                    "AK8PFJet380_TrimMass30",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFHT750_TrimMass50",
                    "AK8PFHT800_TrimMass50",
                    "PFHT1050",
                ],
                "2022": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet425_SoftDropMass40",
                ],
                "2022EE": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet425_SoftDropMass40",
                ],
                "2023-pre-BPix": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet425_SoftDropMass40",
                    "AK8PFJet420_MassSD30",
                ],
                "2023-BPix": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet425_SoftDropMass40",
                    "AK8PFJet420_MassSD30",
                ],
            },
            "semilep-tt": {
                "2022": [
                    "Ele32_WPTight_Gsf",
                    "IsoMu27",
                ],
                "2022EE": [
                    "Ele32_WPTight_Gsf",
                    "IsoMu27",
                ],
            },
            "had-tt": {
                "2022": [
                    "AK8PFJet425_SoftDropMass40",
                ],
                "2022EE": [
                    "AK8PFJet425_SoftDropMass40",
                ],
            },
            "pre-sel": {
                "2018": [
                    "PFJet500",
                    "AK8PFJet500",
                    "AK8PFJet360_TrimMass30",
                    "AK8PFJet380_TrimMass30",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFHT750_TrimMass50",
                    "AK8PFHT800_TrimMass50",
                    "PFHT1050",
                ],
                "2022": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet425_SoftDropMass40",
                    "Ele32_WPTight_Gsf",
                    "IsoMu27",
                ],
                "2022EE": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet425_SoftDropMass40",
                    "Ele32_WPTight_Gsf",
                    "IsoMu27",
                ],
            },
        }

        self.HLTs = HLTs[region]

        self._systematics = save_systematics

        self._nano_version = nano_version

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_3_recommendations
        self.met_filters = [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ]

        """
        pre-sel region:
        - >=1 AK8 jets with pT>250
        - >=1 AK8 jets (ordered by pT) mSD >= 40
        signal region:
        - HLT OR for both data and MC
          - in Run-2 only applied for data
        - >=2 AK8 jets
        - >=2 AK8 jets with pT>250
        - >=2 AK8 jets with mSD>60 or mReg>60
        - >=1 bb AK8 jets (ordered by TXbb) with TXbb > 0.8
        - 0 veto leptons
        semilep-tt region:
        - HLT OR for both data and MC
        - >=1 "good" isolated lepton with pT>50
        - >=1 AK8 jets with pT>250, mSD>50
        - MET > 50
        - >=1 AK4 jet with medium DeepJet
        had tt region:
        - HLT OR for both data and MC
        - == 2 AK8 jets with pT>450 and mSD>50
        - == 2 AK8 jets with Xbb>0.1
        - == 2 AK8 jets with Tau3OverTau2<0.46
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
        is_run3 = year in ["2022", "2022EE", "2023-pre-BPix", "2023-BPix"]
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

        cutflow = OrderedDict()
        cutflow["all"] = n_events
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)
        selection_args = (selection, cutflow, isData, gen_weights)

        # JEC factory loader
        JEC_loader = JECs(year)

        #########################
        # Object definitions
        #########################
        print("starting object selection", f"{time.time() - start:.2f}")

        run = events.run.to_numpy()

        if is_run3:
            veto_muon_sel = veto_muons(events.Muon)
            veto_electron_sel = veto_electrons(events.Electron)
        else:
            veto_muon_sel = veto_muons_run2(events.Muon)
            veto_electron_sel = veto_electrons_run2(events.Electron)

        if self._region != "signal":
            good_muon_sel = good_muons(events.Muon)
            muons = events.Muon[good_muon_sel]
            muons["id"] = muons.charge * (13)

            good_electron_sel = good_electrons(events.Electron)
            electrons = events.Electron[good_electron_sel]
            electrons["id"] = electrons.charge * (11)

        num_jets = 4
        jets, jec_shifted_jetvars = JEC_loader.get_jec_jets(
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

        if JEC_loader.met_factory is not None:
            met = JEC_loader.met_factory.build(events.MET, jets, {}) if isData else events.MET
        else:
            met = events.MET

        print("ak4 JECs", f"{time.time() - start:.2f}")
        jets_sel = (jets.isTight) & (abs(jets.eta) < 4.7)

        if year == "2022" or year == "2022EE":
            jet_veto = get_jetveto(jets, year, run, isData)
            jet_veto = jet_veto & (jets.pt > 15)
            jets_sel = jets_sel & ~jet_veto

        if not is_run3:
            jets_sel = jets_sel & ((jets.pt >= 50) | (jets.puId >= 6))

        jets = jets[jets_sel]
        ht = ak.sum(jets.pt, axis=1)
        print("ak4", f"{time.time() - start:.2f}")

        num_fatjets = 3  # number to save
        num_fatjets_cut = 2  # number to consider for selection
        fatjets = get_ak8jets(events.FatJet)
        fatjets, jec_shifted_fatjetvars = JEC_loader.get_jec_jets(
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
        fatjets_sel = good_ak8jets(fatjets)
        fatjets = fatjets[fatjets_sel]
        # fatjets ordered by xbb
        fatjets_xbb = fatjets[ak.argsort(fatjets.Txbb, ascending=False)]

        # VBF objects
        vbf_jets = jets[(jets.pt > 25) & (jets.delta_r(ak.firsts(fatjets_xbb)) > 1.2)]
        vbf_jet_0 = vbf_jets[:, 0:1]
        vbf_jet_1 = vbf_jets[:, 1:2]
        vbf_mass = (ak.firsts(vbf_jet_0) + ak.firsts(vbf_jet_1)).mass
        vbf_deta = abs(ak.firsts(vbf_jet_0).eta - ak.firsts(vbf_jet_1).eta)

        # JMSR
        # jmsr_shifted_vars = get_jmsr(fatjets_xbb, 2, year, isData)

        #########################
        # Save / derive variables
        #########################

        # Gen variables - saving HH and bbbb 4-vector info
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, jets, fatjets_xbb, selection_args, P4)
                genVars = {**genVars, **vars_dict}

        # Jet variables
        jet_skimvars = self.skim_vars["Jet"]
        if self._nano_version == "v12":
            jet_skimvars = {
                **jet_skimvars,
                "btagDeepFlavB": "btagDeepFlavB",
                "btagPNetB": "btagPNetB",
                "btagPNetCvB": "btagPNetCvB",
                "btagPNetCvL": "btagPNetCvL",
                "btagPNetQvG": "btagPNetQvG",
                "btagRobustParTAK4B": "btagRobustParTAK4B",
            }
        if not isData:
            jet_skimvars = {
                **jet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }

        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        # FatJet variables
        fatjet_skimvars = self.skim_vars["FatJet"]
        if not isData:
            fatjet_skimvars = {
                **fatjet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }
        if self._nano_version == "v12_private":
            fatjet_skimvars = {
                **fatjet_skimvars,
                "Txbb_legacy": "PNetXbbLegacy",
                "particleNet_mass_legacy": "PNetMassLegacy",
            }

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, axis=1)
            for (var, key) in fatjet_skimvars.items()
        }
        # FatJet ordered by bb
        bbFatJetVars = {
            f"bbFatJet{key}": pad_val(fatjets_xbb[var], 2, axis=1)
            for (var, key) in fatjet_skimvars.items()
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
                    # TODO: add also for bb jets
                    ak8FatJetVars[f"ak8FatJet{key}_{shift}"] = pad_val(vals, num_fatjets, axis=1)
        """

        """
        # JMSR variables
        # TODO: add pnetmass
        for var in ["msoftdrop"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jmsr_shifted_vars[var].items():
                # overwrite saved mass vars with corrected ones
                label = "" if shift == "" else "_" + shift
                # do not save other variations for now
                if shift != "": continue
                bbFatJetVars[f"bbFatJet{key}{label}"] = vals
        """

        met_pt = met.pt

        eventVars = {
            key: events[val].to_numpy()
            for key, val in self.skim_vars["Event"].items()
            if key in events.fields
        }
        eventVars["MET_pt"] = met_pt.to_numpy()
        eventVars["ht"] = ht.to_numpy()
        eventVars["nJets"] = ak.sum(jets_sel, axis=1).to_numpy()
        eventVars["nFatJets"] = ak.sum(fatjets_sel, axis=1).to_numpy()

        cut_vbf = (vbf_mass > 500) & (vbf_deta > 4.0)
        eventVars["vbfVeto"] = (~cut_vbf).to_numpy().astype(int)

        if isData:
            pileupVars = {key: np.ones(len(events)) * PAD_VAL for key in self.skim_vars["Pileup"]}
        else:
            pileupVars = {key: events.Pileup[key].to_numpy() for key in self.skim_vars["Pileup"]}
        pileupVars = {**pileupVars, "nPV": events.PV["npvs"].to_numpy()}

        # Trigger variables
        HLTs = deepcopy(self.HLTs[year])
        if is_run3:
            # add extra paths as variables
            HLTs.extend(
                [
                    "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                    "PFHT1050",
                    "AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet275_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet230_SoftDropMass40",
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

        # add trigger objects (for fatjets, id==6)
        # fields: 'pt', 'eta', 'phi', 'l1pt', 'l1pt_2', 'l2pt', 'id', 'l1iso', 'l1charge', 'filterBits'
        num_trigobjs = 4
        fatjet_objs = events.TrigObj[(events.TrigObj.id == 6) & (events.TrigObj.pt >= 100)]
        # sort trigger objects by distance to fatjet_xbb_0 and only save first 3
        dr_bb0 = ak.flatten(fatjet_objs.metric_table(fatjets_xbb[:, 0:1]), axis=-1)
        fatjet_objs = fatjet_objs[ak.argsort(dr_bb0)]

        fatjet_objs["matched_bbFatJet0"] = fatjet_objs.metric_table(fatjets_xbb[:, 0:1]) < 1.0
        fatjet_objs["matched_bbFatJet1"] = fatjet_objs.metric_table(fatjets_xbb[:, 1:2]) < 1.0

        trigObjFatJetVars = {
            f"TriggerObject{key}": pad_val(fatjet_objs[var], num_trigobjs, axis=1)
            for (var, key) in self.skim_vars["TriggerObject"].items()
        }
        # save booleans after padding (flatten will make a different shape than usual arrays)
        trigObjFatJetVars["TriggerObjectMatched_bbFatJet0"] = pad_val(
            ak.flatten(fatjet_objs["matched_bbFatJet0"], axis=-1), num_trigobjs, axis=1
        ).astype(int)
        trigObjFatJetVars["TriggerObjectMatched_bbFatJet1"] = pad_val(
            ak.flatten(fatjet_objs["matched_bbFatJet1"], axis=-1), num_trigobjs, axis=1
        ).astype(int)

        print("TrigObj vars", f"{time.time() - start:.2f}")

        skimmed_events = {
            **genVars,
            **ak4JetVars,
            **ak8FatJetVars,
            **bbFatJetVars,
            **eventVars,
            **pileupVars,
            **HLTVars,
            **trigObjFatJetVars,
        }

        if self._region == "signal" or self._region == "pre-sel":
            # TODO: add shifts from JECs and JSMR
            bbFatDijetVars = self.getFatDijetVars(bbFatJetVars, pt_shift="")

            # VBF Jets
            vbfJetVars = {
                f"vbfJet{key}": pad_val(vbf_jets[var], 2, axis=1)
                for (var, key) in self.skim_vars["Jet"].items()
            }

            skimmed_events = {
                **skimmed_events,
                **vbfJetVars,
                **bbFatDijetVars,
            }

        if self._region == "semilep-tt" or self._region == "pre-sel":
            # concatenate leptons
            leptons = ak.concatenate([muons, electrons], axis=1)
            # sort by pt
            leptons = leptons[ak.argsort(leptons.pt, ascending=False)]

            lepVars = {
                f"lep{key}": pad_val(leptons[var], 2, axis=1)
                for (var, key) in self.skim_vars["Lepton"].items()
            }

            skimmed_events = {
                **skimmed_events,
                **lepVars,
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

        # apply trigger
        apply_trigger = True
        if not is_run3 and (~isData) and self._region == "signal":
            # in run2 we do not apply the trigger to MC
            apply_trigger = False
        if apply_trigger:
            add_selection("trigger", HLT_triggered, *selection_args)

        # metfilters
        cut_metfilters = np.ones(len(events), dtype="bool")
        for mf in self.met_filters:
            if mf in events.Flag.fields:
                cut_metfilters = cut_metfilters & events.Flag[mf]
        add_selection("met_filters", cut_metfilters, *selection_args)

        # jet veto maps
        if year == "2022" or year == "2022EE":
            cut_jetveto = get_jetveto_event(jets, year, run, isData)
            add_selection("ak4_jetveto", cut_jetveto, *selection_args)

        if self._region == "signal":
            # >=2 AK8 jets
            add_selection("ak8_numjets", (ak.num(fatjets) >= 2), *selection_args)

            # >=2 AK8 jets with pT>250
            # TODO: check if fatjet passes pt cut in any of the JEC variations
            cut_pt = (
                np.sum(ak8FatJetVars["ak8FatJetPt"] >= self.preselection["fatjet_pt"], axis=1) >= 2
            )
            add_selection("ak8_pt", cut_pt, *selection_args)

            # >=2 AK8 jets with mSD>60 or mReg>60
            # TODO: check if fatjet passes mass cut in any of the JMS/R variations
            cut_mass = (
                np.sum(
                    (ak8FatJetVars["ak8FatJetMsd"] >= self.preselection["fatjet_msd"])
                    | (ak8FatJetVars["ak8FatJetPNetMass"] >= self.preselection["fatjet_mreg"]),
                    axis=1,
                )
                >= 2
            )
            add_selection("ak8_mass", cut_mass, *selection_args)

            # >=1 bb AK8 jets (ordered by TXbb) with TXbb > 0.8
            cut_txbb = (
                np.sum(bbFatJetVars["bbFatJetPNetXbb"] >= self.preselection["Txbb0"], axis=1) >= 1
            )
            add_selection("ak8bb_txbb0", cut_txbb, *selection_args)

            # 0 veto leptons
            add_selection(
                "0lep",
                (ak.sum(veto_muon_sel, axis=1) == 0) & (ak.sum(veto_electron_sel, axis=1) == 0),
                *selection_args,
            )

            # VBF veto cut
            # add_selection("vbf_veto", ~(cut_vbf), *selection_args)

        elif self._region == "pre-sel":
            # >=1 AK8 jets with pT>250
            cut_pt = np.sum(ak8FatJetVars["ak8FatJetPt"] >= 250, axis=1) >= 1
            add_selection("ak8_pt", cut_pt, *selection_args)

            # >=1 AK8 jets (ordered by pT) mSD >= 40
            cut_mass = np.sum(ak8FatJetVars["ak8FatJetMsd"] >= 40, axis=1) >= 1
            add_selection("ak8_mass", cut_mass, *selection_args)

        elif self._region == "semilep-tt":
            # >=1 "good" isolated lepton with pT>50
            add_selection("lepton_pt", np.sum((leptons.pt > 50), axis=1) >= 1, *selection_args)

            # >=1 AK8 jets with pT>250, mSD>50
            cut_pt_msd = (
                np.sum(
                    (ak8FatJetVars["ak8FatJetPt"] >= 250) & (ak8FatJetVars["ak8FatJetMsd"] >= 50),
                    axis=1,
                )
                >= 1
            )
            add_selection("ak8_pt_msd", cut_pt_msd, *selection_args)

            # MET > 50
            add_selection("met_50", met_pt > 50, *selection_args)

            # >=1 AK4 jet with medium b-tagging ( DeepJet)
            add_selection(
                "ak4jet_btag", ak.sum((jets.btagDeepFlavB >= 0.3091), axis=1) >= 1, *selection_args
            )

        elif self._region == "had-tt":
            # == 2 AK8 jets with pT>450 and mSD>50
            cut_pt_msd = (
                np.sum(
                    (ak8FatJetVars["ak8FatJetPt"] >= 450) & (ak8FatJetVars["ak8FatJetMsd"] >= 50),
                    axis=1,
                )
                == 2
            )
            add_selection("ak8_pt_msd", cut_pt_msd, *selection_args)

            # == 2 AK8 jets with Xbb>0.1
            cut_txbb = np.sum(ak8FatJetVars["ak8FatJetPNetXbb"] >= 0.1, axis=1) == 2
            add_selection("ak8bb_txbb", cut_txbb, *selection_args)

            # == 2 AK8 jets with Tau3OverTau2 < 0.46
            cut_t32 = np.sum(ak8FatJetVars["ak8FatJetTau3OverTau2"] < 0.46, axis=1) == 2
            add_selection("ak8bb_t32", cut_t32, *selection_args)

        print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################

        if isData:
            skimmed_events["weight"] = np.ones(len(events))
        else:
            weights.add("genweight", gen_weights)

            add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy(), dataset)

            # TODO: update trigger weights with those derived by Armen
            add_trig_weights(weights, fatjets, year, num_fatjets_cut)

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

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        dump_table(dataframe, fname)

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
        self, bbFatJetVars: dict, pt_shift: str | None = None, mass_shift: str | None = None
    ):
        """Calculates Dijet variables for given pt / mass JEC / JMS/R variation"""
        dijetVars = {}

        ptlabel = pt_shift if pt_shift is not None else ""
        mlabel = mass_shift if mass_shift is not None else ""

        jets = vector.array(
            {
                "pt": bbFatJetVars[f"bbFatJetPt{ptlabel}"],
                "phi": bbFatJetVars["bbFatJetPhi"],
                "eta": bbFatJetVars["bbFatJetEta"],
                "M": bbFatJetVars[f"bbFatJetMsd{mlabel}"],
                # "M": bbFatJetVars[f"ak8FatJetPNetMass{mlabel}"],
            }
        )

        # get dijet with two first fatjets
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
