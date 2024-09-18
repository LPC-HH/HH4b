"""
Skimmer for bbbb analysis with FatJets.
Author(s): Raghav Kansal, Cristina Suarez
"""

from __future__ import annotations

import logging
import pathlib
import time
from collections import OrderedDict
from copy import deepcopy

import awkward as ak
import numpy as np
import pandas as pd
import vector
import xgboost as xgb
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

import HH4b

from . import objects, utils
from .corrections import (
    JECs,
    add_pileup_weight,
    add_ps_weight,
    get_jetveto_event,
    get_jmsr,
    get_pdf_weights,
    get_scale_weights,
)
from .GenSelection import (
    gen_selection_Hbb,
    gen_selection_HHbbbb,
    gen_selection_Top,
    gen_selection_V,
)
from .objects import (
    get_ak8jets,
    good_ak8jets,
    good_electrons,
    good_muons,
    veto_electrons,
    veto_muons,
)
from .SkimmerABC import SkimmerABC
from .utils import P4, PAD_VAL, add_selection, get_var_mapping, pad_val

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "HHto4B": gen_selection_HHbbbb,
    "HToBB": gen_selection_Hbb,
    "Hto2B": gen_selection_Hbb,
    "TTto4Q": gen_selection_Top,
    "Wto2Q-": gen_selection_V,
    "Zto2Q-": gen_selection_V,
    "WtoLNu-": gen_selection_V,
    "DYto2L-": gen_selection_V,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

package_path = str(pathlib.Path(__file__).parent.parent.resolve())


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
            "Txbb": "PNetTXbb",  # these are discriminants
            "Txjj": "PNetTXjj",
            "Tqcd": "PNetTQCD",
            "PQCD1HF": "PNetQCD1HF",  # these are raw probabilities
            "PQCD2HF": "PNetQCD2HF",
            "PQCD0HF": "PNetQCD0HF",
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
        "pt": 250,
        "eta": 2.5,
        "msd": 50,
        "mreg": 0,
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

    ak4_bjet_selection = {  # noqa: RUF012
        "pt": 25,
        "eta_max": 2.5,
        "id": "tight",
        "dr_fatjets": 0.9,
        "dr_leptons": 0.4,
    }

    ak4_bjet_lepton_selection = {  # noqa: RUF012
        "electron_pt": 5,
        "muon_pt": 7,
    }

    def __init__(
        self,
        xsecs=None,
        save_systematics=False,
        region="signal",
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
                    "AK8PFJet400_SoftDropMass40",
                    "AK8PFJet425_SoftDropMass40",
                    "AK8PFJet400_SoftDropMass40",
                    "AK8PFJet420_MassSD30",
                ],
                "2023BPix": [
                    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
                    "AK8PFJet230_SoftDropMass40",
                    "AK8PFJet400_SoftDropMass40",
                    "AK8PFJet425_SoftDropMass40",
                    "AK8PFJet400_SoftDropMass40",
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
                    "AK8PFJet400_SoftDropMass40",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
                ],
                "2023BPix": [
                    "AK8PFJet230_SoftDropMass40",
                    "AK8PFJet400_SoftDropMass40",
                    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
                ],
            },
        }
        HLTs["pre-sel"] = HLTs["signal"]

        self.HLTs = HLTs[region]

        self._systematics = save_systematics

        self.jecs = utils.jecs

        self._nano_version = nano_version

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_3_recommendations
        self.met_filters = [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
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

        # BDT model
        bdt_model_name = "24May31_lr_0p02_md_8_AK4Away"
        self.bdt_model = xgb.XGBClassifier()
        self.bdt_model.load_model(
            fname=f"{package_path}/boosted/bdt_trainings_run3/{bdt_model_name}/trained_bdt.model"
        )

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

        # datasets for saving jec variations
        isJECs = (
            "HHto4B" in dataset
            or "TT" in dataset
            or "Wto2Q" in dataset
            or "Zto2Q" in dataset
            or "Hto2B" in dataset
            or "WW" in dataset
            or "ZZ" in dataset
            or "WZ" in dataset
        )

        # gen-weights
        gen_weights = events["genWeight"].to_numpy() if not isData else None
        n_events = len(events) if isData else np.sum(gen_weights)

        # selection and cutflow
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

        # Leptons
        veto_muon_sel = veto_muons(events.Muon)
        veto_electron_sel = veto_electrons(events.Electron)
        if self._region != "signal":
            good_muon_sel = good_muons(events.Muon)
            muons = events.Muon[good_muon_sel]
            muons["id"] = muons.charge * (13)

            good_electron_sel = good_electrons(events.Electron)
            electrons = events.Electron[good_electron_sel]
            electrons["id"] = electrons.charge * (11)

        # AK4 Jets
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

        # AK8 Jets
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

        # fatjets ordered by xbb
        if not legacy:
            fatjets_xbb = fatjets[ak.argsort(fatjets.Txbb, ascending=False)]
        else:
            fatjets_xbb = fatjets[ak.argsort(fatjets.TXbb_legacy, ascending=False)]

        # variations for bb fatjets
        jec_shifted_bbfatjetvars = {}
        if self._region == "signal" and isJECs:
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

        # AK4 objects away from first two fatjets
        ak4_jets_awayfromak8 = objects.ak4_jets_awayfromak8(
            jets,
            fatjets_xbb[:, :2],
            events,
            **self.ak4_bjet_selection,
            **self.ak4_bjet_lepton_selection,
            sort_by="nearest",
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

        # AK4 Jet variables
        jet_skimvars = self.skim_vars["Jet"]
        if "v12" in self._nano_version:
            jet_skimvars = {
                **jet_skimvars,
                "btagDeepFlavB": "btagDeepFlavB",
                "btagPNetB": "btagPNetB",
                "btagPNetCvB": "btagPNetCvB",
                "btagPNetCvL": "btagPNetCvL",
                "btagPNetQvG": "btagPNetQvG",
            }
        if not isData:
            jet_skimvars = {
                **jet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }

        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_jets, axis=1)
            for (var, key) in jet_skimvars.items()
        }

        if len(ak4_jets_awayfromak8) == 2:
            ak4JetAwayVars = {
                f"AK4JetAway{key}": pad_val(
                    ak.concatenate(
                        [ak4_jets_awayfromak8[0][var], ak4_jets_awayfromak8[1][var]], axis=1
                    ),
                    2,
                    axis=1,
                )
                for (var, key) in jet_skimvars.items()
            }
        else:
            ak4JetAwayVars = {
                f"AK4JetAway{key}": pad_val(ak4_jets_awayfromak8[var], 2, axis=1)
                for (var, key) in jet_skimvars.items()
            }

        # AK8 Jet variables
        fatjet_skimvars = self.skim_vars["FatJet"]
        if not isData:
            fatjet_skimvars = {
                **fatjet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }
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
            fatjet_skimvars = {
                **fatjet_skimvars,
                "particleNet_mass_legacy": "PNetMassLegacy",
                **{f"{var}_legacy": f"PNet{var}Legacy" for var in extra_vars},
            }
        if self._nano_version == "v12_private" or self._nano_version == "v12":
            fatjet_skimvars = {
                **fatjet_skimvars,
                "particleNetTvsQCD": "particleNetWithMass_TvsQCD",
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
            fatjet_skimvars = {**fatjet_skimvars, **{var: var for var in extra_vars}}

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], 3, axis=1)
            for (var, key) in fatjet_skimvars.items()
        }
        bbFatJetVars = {
            f"bbFatJet{key}": pad_val(fatjets_xbb[var], 2, axis=1)
            for (var, key) in fatjet_skimvars.items()
        }
        print("Jet vars", f"{time.time() - start:.2f}")

        # JEC and JMSR
        if self._region == "signal" and isJECs:
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
                        bbFatJetVars[f"bbFatJet{key}_{shift}"] = pad_val(vals, 2, axis=1)

            # FatJet JMSR
            for var in jmsr_vars:
                key = fatjet_skimvars[var]
                for shift, vals in jmsr_shifted_vars[var].items():
                    # overwrite saved mass vars with corrected ones
                    label = "" if shift == "" else "_" + shift
                    bbFatJetVars[f"bbFatJet{key}{label}"] = vals

        # Event variables
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
                    "AK8PFJet400_SoftDropMass40",
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
        fatjet_objs["matched_ak8FatJet0"] = fatjet_objs.metric_table(fatjets[:, 0:1]) < 1.0
        fatjet_objs["matched_ak8FatJet1"] = fatjet_objs.metric_table(fatjets[:, 1:2]) < 1.0

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
        trigObjFatJetVars["TriggerObjectMatched_ak8FatJet0"] = pad_val(
            ak.flatten(fatjet_objs["matched_ak8FatJet0"], axis=-1), num_trigobjs, axis=1
        ).astype(int)
        trigObjFatJetVars["TriggerObjectMatched_ak8FatJet1"] = pad_val(
            ak.flatten(fatjet_objs["matched_ak8FatJet1"], axis=-1), num_trigobjs, axis=1
        ).astype(int)
        print("TrigObj vars", f"{time.time() - start:.2f}")

        # vbfJets
        vbfJetVars = {
            f"VBFJet{key}": pad_val(vbf_jets[var], 2, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        # JEC variations for VBF Jets
        if self._region == "signal" and isJECs:
            for var in ["pt"]:
                key = self.skim_vars["Jet"][var]
                for label, shift in self.jecs.items():
                    if shift in ak.fields(vbf_jets):
                        for vari in ["up", "down"]:
                            vbfJetVars[f"VBFJet{key}_{label}_{vari}"] = pad_val(
                                vbf_jets[shift][vari][var], 2, axis=1
                            )

        skimmed_events = {
            **genVars,
            **eventVars,
            **pileupVars,
            **HLTVars,
            **ak4JetAwayVars,
            **ak8FatJetVars,
            **bbFatJetVars,
            **trigObjFatJetVars,
            **vbfJetVars,
        }

        if self._region == "signal":
            bdtVars = self.getBDT(bbFatJetVars, vbfJetVars, ak4JetAwayVars, met_pt, "")
            print(bdtVars)
            skimmed_events = {
                **skimmed_events,
                **bdtVars,
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
        if (~is_run3) and (~isData) and self._region == "signal":
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

        if self._region == "pre-sel" or self._region == "signal":
            # >=2 AK8 jets passing selections
            add_selection("ak8_numjets", (ak.num(fatjets) >= 2), *selection_args)

            # >=1 AK8 jets with pT>250 GeV
            cut_pt = np.sum(ak8FatJetVars["ak8FatJetPt"] >= 250, axis=1) >= 1
            add_selection("ak8_pt", cut_pt, *selection_args)

            # >=1 AK8 jets with mSD >= 40 GeV
            cut_mass = np.sum(ak8FatJetVars["ak8FatJetMsd"] >= 40, axis=1) >= 1
            add_selection("ak8_mass", cut_mass, *selection_args)

            # Veto leptons
            add_selection(
                "0lep",
                (ak.sum(veto_muon_sel, axis=1) == 0) & (ak.sum(veto_electron_sel, axis=1) == 0),
                *selection_args,
            )

            if self._region == "signal":
                # >=1 bb AK8 jets (ordered by TXbb) with TXbb > 0.8
                if not legacy:
                    cut_txbb = (
                        np.sum(
                            bbFatJetVars["bbFatJetPNetTXbb"] >= self.preselection["Txbb0"], axis=1
                        )
                        >= 1
                    )
                else:
                    # using an OR of legacy and v12 TXbb
                    cut_txbb = (np.sum(bbFatJetVars["bbFatJetPNetTXbb"] >= 0.5, axis=1) >= 1) | (
                        np.sum(
                            bbFatJetVars["bbFatJetPNetTXbbLegacy"] >= self.preselection["Txbb0"],
                            axis=1,
                        )
                        >= 1
                    )
                add_selection("ak8bb_txbb0", cut_txbb, *selection_args)

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
            cut_txbb = np.sum(ak8FatJetVars["ak8FatJetPNetTXbb"] >= 0.1, axis=1) == 2
            add_selection("ak8bb_txbb", cut_txbb, *selection_args)

        print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################

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
        return {year: {dataset: {"totals": totals_dict, "cutflow": cutflow}}}

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
        add_ps_weight(weights, events.PSWeight)

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

    def getBDT(
        self, bbFatJetVars: dict, vbfJetVars: dict, ak4JetAwayVars: dict, met_pt, jshift: str = ""
    ):
        """Calculates BDT"""
        key_map = get_var_mapping(jshift)

        # makedataframe from 24May31_lr_0p02_md_8_AK4Away
        jets = vector.array(
            {
                "pt": bbFatJetVars["bbFatJetPt"],
                "phi": bbFatJetVars["bbFatJetPhi"],
                "eta": bbFatJetVars["bbFatJetEta"],
                "M": bbFatJetVars["bbFatJetPNetMassLegacy"],
            }
        )
        h1 = jets[:, 0]
        h2 = jets[:, 1]
        hh = jets[:, 0] + jets[:, 1]
        vbfjets = vector.array(
            {
                "pt": vbfJetVars["VBFJetPt"],
                "phi": vbfJetVars["VBFJetPhi"],
                "eta": vbfJetVars["VBFJetEta"],
                "M": vbfJetVars["VBFJetMass"],
            }
        )
        vbf1 = vbfjets[:, 0]
        vbf2 = vbfjets[:, 1]
        jj = vbfjets[:, 0] + vbfjets[:, 1]
        ak4away = vector.array(
            {
                "pt": ak4JetAwayVars["AK4JetAwayPt"],
                "phi": ak4JetAwayVars["AK4JetAwayPhi"],
                "eta": ak4JetAwayVars["AK4JetAwayEta"],
                "M": ak4JetAwayVars["AK4JetAwayMass"],
            }
        )
        ak4away1 = ak4away[:, 0]
        ak4away2 = ak4away[:, 1]
        h1ak4away1 = h1 + ak4away1
        h2ak4away2 = h2 + ak4away2
        bdt_events = pd.DataFrame(
            {
                # dihiggs system
                key_map("HHPt"): hh.pt,
                key_map("HHeta"): hh.eta,
                key_map("HHmass"): hh.mass,
                # met in the event
                key_map("MET"): met_pt,
                # fatjet tau32
                key_map("H1T32"): bbFatJetVars[key_map("bbFatJetTau3OverTau2")][:, 0],
                key_map("H2T32"): bbFatJetVars[key_map("bbFatJetTau3OverTau2")][:, 1],
                # fatjet mass
                key_map("H1Mass"): bbFatJetVars[key_map("bbFatJetPNetMassLegacy")][:, 0],
                # fatjet kinematics
                key_map("H1Pt"): h1.pt,
                key_map("H2Pt"): h2.pt,
                key_map("H1eta"): h1.eta,
                # "H2eta": h2.eta,
                # xbb
                key_map("H1Xbb"): bbFatJetVars[key_map("bbFatJetPNetPXbbLegacy")][:, 0],
                key_map("H1QCDb"): bbFatJetVars[key_map("bbFatJetPNetPQCDbLegacy")][:, 0],
                key_map("H1QCDbb"): bbFatJetVars[key_map("bbFatJetPNetPQCDbbLegacy")][:, 0],
                key_map("H1QCDothers"): bbFatJetVars[key_map("bbFatJetPNetPQCD0HFLegacy")][:, 0],
                # ratios
                key_map("H1Pt_HHmass"): h1.pt / hh.mass,
                key_map("H2Pt_HHmass"): h2.pt / hh.mass,
                key_map("H1Pt/H2Pt"): h1.pt / h2.pt,
                # vbf mjj and eta_jj
                key_map("VBFjjMass"): jj.mass,
                key_map("VBFjjDeltaEta"): np.abs(vbf1.eta - vbf2.eta),
                # AK4JetAway
                key_map("H1AK4JetAway1dR"): h1.deltaR(ak4away1),
                key_map("H2AK4JetAway2dR"): h2.deltaR(ak4away2),
                key_map("H1AK4JetAway1mass"): h1ak4away1.mass,
                key_map("H2AK4JetAway2mass"): h2ak4away2.mass,
            }
        )
        # perform BDT inference
        preds = self.bdt_model.predict_proba(bdt_events)

        # store BDT output
        bdtVars = {}
        jlabel = "" if jshift == "" else "_" + jshift
        # weight for ttbar probability
        weight_ttbar = 1
        if preds.shape[1] == 2:  # binary BDT only
            bdtVars[f"bdt_score{jlabel}"] = preds[:, 1]
        elif preds.shape[1] == 3:  # multi-class BDT with ggF HH, QCD, ttbar classes
            bdtVars[f"bdt_score{jlabel}"] = preds[:, 0]  # ggF HH
        elif preds.shape[1] == 4:  # multi-class BDT with ggF HH, VBF HH, QCD, ttbar classes
            bg_tot = np.sum(preds[:, 2:], axis=1)
            bdtVars[f"bdt_score{jlabel}"] = preds[:, 0] / (preds[:, 0] + bg_tot)
            bdtVars[f"bdt_score_vbf{jlabel}"] = preds[:, 1] / (
                preds[:, 1] + preds[:, 2] + weight_ttbar * preds[:, 3]
            )

        return bdtVars
