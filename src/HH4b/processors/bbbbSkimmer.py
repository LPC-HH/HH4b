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

import HH4b

from . import objects, utils
from .corrections import (
    JECs,
    add_pileup_weight,
    add_ps_weight,
    add_trig_weights,
    get_jetveto_event,
    get_jmsr,
    get_pdf_weights,
    get_scale_weights,
)
from .GenSelection import gen_selection_Hbb, gen_selection_HHbbbb, gen_selection_Top
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
from .SkimmerABC import SkimmerABC
from .utils import P4, PAD_VAL, add_selection, pad_val

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "HHto4B": gen_selection_HHbbbb,
    "HToBB": gen_selection_Hbb,
    "TTto4Q": gen_selection_Top,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class bbbbSkimmer(SkimmerABC):
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
            "Txbb": "PNetTXbb",  # these are the PXbb / (PXbb + PQCD) discriminants
            "Txjj": "PNetTXjj",
            "Tqcd": "PNetTQCD",
            "PQCDb": "PNetQCD1HF",  # these are raw probabilities
            "PQCDbb": "PNetQCD2HF",
            "PQCDothers": "PNetQCD0HF",
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
        "Txbb0": 0.8,
    }

    fatjet_selection = {  # noqa: RUF012
        "pt": 300,
        "eta": 2.5,
        "msd": 60,
        "mreg": 60,
    }

    vbf_jet_selection = {  # noqa: RUF012
        "pt": 25,
        "eta_max": 4.7,
        "id": "tight",
        "dr_fatjets": 1.2,
        "dr_leptons": 0.4,
    }

    vbf_veto_lepton_selection = {  # noqa: RUF012
        "electron_pt": 5,
        "muon_pt": 7,
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
                "2023": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
                    "AK8PFJet230_SoftDropMass40",
                    "AK8PFJet425_SoftDropMass40",
                    "AK8PFJet420_MassSD30",
                ],
                "2023BPix": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
                    "AK8PFJet230_SoftDropMass40",
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
                "2023": [
                    "Ele32_WPTight_Gsf",
                    "IsoMu27",
                ],
                "2023BPix": [
                    "Ele32_WPTight_Gsf",
                    "IsoMu27",
                ],
            },
            "had-tt": {
                "2022": [
                    "AK8PFJet425_SoftDropMass40",
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                ],
                "2022EE": [
                    "AK8PFJet425_SoftDropMass40",
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                ],
                "2023": [
                    "AK8PFJet425_SoftDropMass40",
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet230_SoftDropMass40",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
                ],
                "2023BPix": [
                    "AK8PFJet230_SoftDropMass40",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
                ],
            },
        }

        self.HLTs = HLTs[region]

        self._systematics = save_systematics

        self.jecs = HH4b.hh_vars.jecs

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
        is_run3 = year in ["2022", "2022EE", "2023", "2023BPix"]
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

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        selection_args = (selection, cutflow, isData, gen_weights)

        # JEC factory loader
        JEC_loader = JECs(year)

        #########################
        # Object definitions
        #########################
        print("starting object selection", f"{time.time() - start:.2f}")

        # run = events.run.to_numpy()

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
            jecs=self.jecs,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
        print(" ", jec_shifted_jetvars)

        if JEC_loader.met_factory is not None:
            met = JEC_loader.met_factory.build(events.MET, jets, {}) if isData else events.MET
        else:
            met = events.MET

        print("ak4 JECs", f"{time.time() - start:.2f}")
        jets_sel = (jets.pt > 15) & (jets.isTight) & (abs(jets.eta) < 4.7)
        if not is_run3:
            jets_sel = jets_sel & ((jets.pt >= 50) | (jets.puId >= 6))

        jets = jets[jets_sel]
        ht = ak.sum(jets.pt, axis=1)
        print("ak4", f"{time.time() - start:.2f}")

        num_fatjets = 3  # number to save
        num_fatjets_cut = 2  # number to consider for selection
        fatjets = get_ak8jets(events.FatJet)  # this adds all our extra variables e.g. TXbb
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
        print("ak8 JECs", f"{time.time() - start:.2f}")

        fatjets = good_ak8jets(fatjets, **self.fatjet_selection)
        legacy = "particleNetLegacy_mass" in fatjets.fields

        if not legacy:
            # fatjets ordered by xbb
            fatjets_xbb = fatjets[ak.argsort(fatjets.Txbb, ascending=False)]
        else:
            fatjets_xbb = fatjets[ak.argsort(fatjets.Txbb_legacy, ascending=False)]

        # variations for bb fatjets (TODO: not only for signal)
        jec_shifted_bbfatjetvars = {}
        if self._region == "signal" and isSignal:
            for jec_var in ["pt"]:
                tdict = {"": fatjets_xbb[jec_var]}
                for key, shift in self.jecs.items():
                    for var in ["up", "down"]:
                        if shift in ak.fields(fatjets_xbb):
                            tdict[f"{key}_{var}"] = fatjets_xbb[shift][var][jec_var]
                jec_shifted_bbfatjetvars[jec_var] = tdict

        # VBF objects
        vbf_jets = objects.vbf_jets(
            jets,
            fatjets_xbb[:, :2],
            events,
            **self.vbf_jet_selection,
            **self.vbf_veto_lepton_selection,
        )

        # JMSR
        jmsr_vars = (
            ["msoftdrop", "particleNet_mass_legacy"]
            if self._nano_version == "v12_private"
            else ["msoftdrop", "particleNet_mass"]
        )
        jmsr_shifted_vars = get_jmsr(fatjets_xbb, 2, year, isData, jmsr_vars=jmsr_vars)

        #########################
        # Save / derive variables
        #########################

        # Gen variables - saving HH and bbbb 4-vector info
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, jets, fatjets_xbb, selection_args, P4)
                genVars = {**genVars, **vars_dict}

        # remove unnecessary ak4 gen variables for signal region
        if self._region == "signal":
            genVars = {key: val for (key, val) in genVars.items() if not key.startswith("ak4Jet")}

        # used for normalization to cross section below
        gen_selected = (
            selection.all(*selection.names)
            if len(selection.names)
            else np.ones(len(events)).astype(bool)
        )
        logging.info(f"Passing gen selection: {np.sum(gen_selected)} / {len(events)}")

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
            extra_vars = ["TXbb", "PXbb", "PQCD", "PQCDb", "PQCDbb", "PQCDothers"]
            fatjet_skimvars = {
                **fatjet_skimvars,
                "particleNet_mass_legacy": "PNetMassLegacy",
                **{f"{var}_legacy": f"PNet{var}Legacy" for var in extra_vars},
            }
        if self._nano_version == "v12_private" or self._nano_version == "v12":
            fatjet_skimvars = {
                **fatjet_skimvars,
                "particleNetWithMass_TvsQCD": "particleNetWithMass_TvsQCD",
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

        # JEC and JMSR  (TODO: for signal only for now, add others)
        if self._region == "signal" and isSignal:
            # Jet JEC variables
            for var in ["pt"]:
                key = self.skim_vars["Jet"][var]
                for shift, vals in jec_shifted_jetvars[var].items():
                    if shift != "":
                        ak4JetVars[f"ak4Jet{key}_{shift}"] = pad_val(vals, num_jets, axis=1)

            # FatJet JEC variables
            for var in ["pt"]:
                key = self.skim_vars["FatJet"][var]
                for shift, vals in jec_shifted_bbfatjetvars[var].items():
                    if shift != "":
                        bbFatJetVars[f"bbFatJet{key}_{shift}"] = pad_val(vals, num_fatjets, axis=1)

            # FatJet JMSR
            for var in jmsr_vars:
                key = fatjet_skimvars[var]
                for shift, vals in jmsr_shifted_vars[var].items():
                    # overwrite saved mass vars with corrected ones
                    label = "" if shift == "" else "_" + shift
                    bbFatJetVars[f"bbFatJet{key}{label}"] = vals

        met_pt = met.pt

        eventVars = {
            key: events[val].to_numpy()
            for key, val in self.skim_vars["Event"].items()
            if key in events.fields
        }
        eventVars["MET_pt"] = met_pt.to_numpy()
        eventVars["ht"] = ht.to_numpy()
        eventVars["nJets"] = ak.sum(jets_sel, axis=1).to_numpy()
        eventVars["nFatJets"] = ak.num(fatjets).to_numpy()

        if isData:
            pileupVars = {key: np.ones(len(events)) * PAD_VAL for key in self.skim_vars["Pileup"]}
        else:
            pileupVars = {key: events.Pileup[key].to_numpy() for key in self.skim_vars["Pileup"]}

        pileupVars = {**pileupVars, "nPV": events.PV["npvs"].to_numpy()}

        # Trigger variables
        HLTs = deepcopy(self.HLTs[year])
        if is_run3 and self._region != "signal":
            # add extra paths as variables
            HLTs.extend(
                [
                    "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                    "PFHT1050",
                    "AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet275_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet230_SoftDropMass40",
                    "AK8PFJet425_SoftDropMass40",
                    "AK8DiPFJet250_250_MassSD50",
                    "AK8DiPFJet260_260_MassSD30",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p10",
                    "AK8PFJet250_SoftDropMass40_PNetBB0p06",
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
            **bbFatJetVars,
            **eventVars,
            **pileupVars,
            **HLTVars,
        }

        if self._region == "signal":
            # TODO: add shifts from JECs and JSMR
            bbFatDijetVars = self.getFatDijetVars(bbFatJetVars, pt_shift="")

            # VBF Jets
            vbfJetVars = {
                f"VBFJet{key}": pad_val(vbf_jets[var], 2, axis=1)
                for (var, key) in self.skim_vars["Jet"].items()
            }

            # JEC variations for VBF Jets (for signal only for now)
            if self._region == "signal" and isSignal:
                for var in ["pt"]:
                    key = self.skim_vars["Jet"][var]
                    for label, shift in self.jecs.items():
                        if shift in ak.fields(vbf_jets):
                            for vari in ["up", "down"]:
                                vbfJetVars[f"VBFJet{key}_{label}_{vari}"] = pad_val(
                                    vbf_jets[shift][vari][var], 2, axis=1
                                )

            skimmed_events = {
                **skimmed_events,
                **vbfJetVars,
                **bbFatDijetVars,
            }
        else:
            # these variables aren't needed for signal region
            skimmed_events = {
                **skimmed_events,
                **ak8FatJetVars,
                **ak4JetVars,
                **trigObjFatJetVars,
            }

        if self._region == "semilep-tt":
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
        if is_run3:
            cut_jetveto = get_jetveto_event(jets, year)
            add_selection("ak4_jetveto", cut_jetveto, *selection_args)

        if self._region == "signal":
            # >=2 AK8 jets passing selections
            add_selection("ak8_numjets", (ak.num(fatjets) >= 2), *selection_args)

            # >=1 bb AK8 jets (ordered by TXbb) with TXbb > 0.8
            if not legacy:
                cut_txbb = (
                    np.sum(bbFatJetVars["bbFatJetPNetTXbb"] >= self.preselection["Txbb0"], axis=1)
                    >= 1
                )
            else:
                # using an OR of legacy and v12 TXbb
                cut_txbb = (
                    np.sum(bbFatJetVars["bbFatJetPNetTXbb"] >= self.preselection["Txbb0"], axis=1)
                    >= 1
                ) | (
                    np.sum(
                        bbFatJetVars["bbFatJetPNetTXbbLegacy"] >= self.preselection["Txbb0"], axis=1
                    )
                    >= 1
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
            # == 2 AK8 jets with pT>300 and mSD>40
            cut_pt_msd = (
                np.sum(
                    (ak8FatJetVars["ak8FatJetPt"] >= 300) & (ak8FatJetVars["ak8FatJetMsd"] >= 40),
                    axis=1,
                )
                == 2
            )
            add_selection("ak8_pt_msd", cut_pt_msd, *selection_args)

            # == 2 AK8 jets with Xbb>0.1
            cut_txbb = np.sum(ak8FatJetVars["ak8FatJetPNetXbb"] >= 0.1, axis=1) == 2
            add_selection("ak8bb_txbb", cut_txbb, *selection_args)

            # == 2 AK8 jets with Tau3OverTau2 < 0.46
            # cut_t32 = np.sum(ak8FatJetVars["ak8FatJetTau3OverTau2"] < 0.46, axis=1) == 2
            # add_selection("ak8bb_t32", cut_t32, *selection_args)

        print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################

        totals_dict = {"nevents": n_events}

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights_dict, totals_temp = self.add_weights(
                events, year, dataset, gen_weights, gen_selected, fatjets, num_fatjets_cut
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
        return {year: {dataset: {"totals": totals_dict, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def add_weights(
        self, events, year, dataset, gen_weights, gen_selected, fatjets, num_fatjets_cut
    ) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights = Weights(len(events), storeIndividual=True)
        weights.add("genweight", gen_weights)

        add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy(), dataset)
        add_ps_weight(weights, events.PSWeight)

        # TODO: update trigger weights with those derived by Armen
        add_trig_weights(weights, fatjets, year, num_fatjets_cut)

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
        if ("HHTobbbb" in dataset or "HHto4B" in dataset) or dataset.startswith("TTTo"):
            scale_weights = get_scale_weights(events)
            if scale_weights is not None:
                weights_dict["scale_weights"] = (
                    scale_weights * weights_dict["weight"][:, np.newaxis]
                )
                totals_dict["np_scale_weights"] = np.sum(
                    (scale_weights * weight_np[:, np.newaxis])[gen_selected], axis=0
                )

        if "HHTobbbb" in dataset or "HHto4B" in dataset:
            pdf_weights = get_pdf_weights(events)
            weights_dict["pdf_weights"] = pdf_weights * weights_dict["weight"][:, np.newaxis]
            totals_dict["np_pdf_weights"] = np.sum(
                (pdf_weights * weight_np[:, np.newaxis])[gen_selected], axis=0
            )

        ###################### Normalization (Step 1) ######################

        weight_norm = self.get_dataset_norm(year, dataset)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.weight()

        return weights_dict, totals_dict

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
