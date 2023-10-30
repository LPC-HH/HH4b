"""
Skimmer for matching studies
Author(s): Raghav Kansal, Cristina Suarez
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from coffea.nanoevents.methods.nanoaod import JetArray
import vector

import itertools
import pathlib
import pickle
import gzip
import os

from typing import Dict
from collections import OrderedDict

from .GenSelection import gen_selection_HHbbbb
from .utils import pad_val, add_selection, concatenate_dicts, select_dicts, P4, PAD_VAL
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

MIN_JETS = 4
MAX_JETS = 4
HIGGS_MASS = 125.0


class matchingSkimmer(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {
        "Jet": {
            "btagDeepFlavB": "btagDeepFlavB",
        },
        # TODO: add hadron flavour
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "Txbb": "PNetXbb",
            "Txjj": "PNetXjj",
            "particleNet_mass": "PNetMass",
        },
        "GenJet": P4,
    }

    # compute possible jet assignments lookup table
    JET_ASSIGNMENTS = {}
    for nj in range(MIN_JETS, MAX_JETS + 1):
        a = list(itertools.combinations(range(nj), 2))
        b = np.array(
            [(i, j) for i, j in itertools.combinations(a, 2) if len(set(i + j)) == MIN_JETS]
        )
        JET_ASSIGNMENTS[nj] = b

    def __init__(self, xsecs={}, apply_selection=True):
        super(matchingSkimmer, self).__init__()

        self.XSECS = xsecs  # in pb
        self._accumulator = processor.dict_accumulator({})
        self.apply_selection = apply_selection

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
        print("Starting")

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])

        # if year != "2018":
        #     # for now, only in v11_private
        #     btag_vars = {
        #         "btagPNetProb": "btagPNetProb",
        #         "btagPNetProbbb": "btagPNetProbbb",
        #         "btagPNetProbc": "btagPNetProbc",
        #         "btagPNetProbuds": "btagPNetProbuds",
        #         "btagPNetProbg": "btagPNetProbg",
        #         "btagPNetBvsAll": "btagPNetBvsAll",
        #     }

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

        print("Before objects", f"{time.time() - start:.2f}")

        #########################
        # Object definitions
        #########################
        veto_muon_sel = veto_muons(events.Muon)
        print("Before electrons ", f"{time.time() - start:.2f}")

        veto_electron_sel = veto_electrons(events.Electron)

        print("Before jets ", f"{time.time() - start:.2f}")

        jets = events.Jet
        jets = jets[good_ak4jets(jets, year, events.run.to_numpy(), isData)]
        ht = ak.sum(jets.pt, axis=1)
        vbf_jets = jets[(jets.pt > 25) & (((jets.pt < 50) & (jets.puId >= 6)) | (jets.pt >= 50))]

        print("Before fatjets ", f"{time.time() - start:.2f}")

        fatjets = get_ak8jets(events.FatJet)
        fatjets = fatjets[good_ak8jets(fatjets)]
        fatjets = fatjets[ak.argsort(fatjets.Txbb, ascending=False)]

        print("Before outside ", f"{time.time() - start:.2f}")

        # get jets outside the leading fj (in bb)
        leading_fj = ak.firsts(fatjets)
        jets_outside_fj = jets[jets.delta_r(leading_fj) > 0.8]
        jets_outside_fj = jets_outside_fj[
            ak.argsort(jets_outside_fj["btagDeepFlavB"], ascending=False)
        ]

        print("Before b-jet energy regression", f"{time.time() - start:.2f}")

        # b-jet energy regression
        jets_p4 = bregcorr(jets)
        jets_p4_outside_fj = bregcorr(jets_outside_fj)

        print("Objects", f"{time.time() - start:.2f}")

        #########################
        # Save / derive variables
        #########################
        skimmed_events = {}

        # HLT variables
        HLTs = {
            "2018": [
                "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5",
                "HLT_PFHT1050",
                "HLT_PFJet500",
                "HLT_AK8PFJet500",
                "HLT_AK8PFJet400_TrimMass30",
                "HLT_AK8PFHT800_TrimMass50",
                "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4",
                "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
                "HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59",
                "HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17",
                "HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "HLT_QuadPFJet98_83_71_15_PFBTagDeepCSV_1p3_VBF2",
                "HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1",
            ]
        }[year]
        zeros = np.zeros(len(events), dtype="bool")
        HLTVars = {
            trigger: (
                events.HLT[trigger].to_numpy().astype(int)
                if trigger in events.HLT.fields
                else zeros
            )
            for trigger in HLTs
        }
        
        # Jet variables
        num_jets = 6
        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }
        ak4JetVarsOutsideFJ = {
            f"ak4JetOutside{key}": pad_val(jets_outside_fj[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }
        for (var, key) in P4.items():
            ak4JetVars[f"ak4Jet{key}"] = pad_val(getattr(jets_p4, var), num_jets, axis=1)
            ak4JetVarsOutsideFJ[f"ak4JetOutside{key}"] = pad_val(getattr(jets_p4_outside_fj, var), num_jets, axis=1)
        skimmed_events["ht"] = ht.to_numpy()
            
        print("AK4Vars", f"{time.time() - start:.2f}")

        # FatJet variables
        num_fatjets = 2
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        ak4JetVars = {
            **ak4JetVars,
            **ak4JetVarsOutsideFJ,
            **self.getJetAssignmentVars(ak4JetVars, ak8Vars=ak8FatJetVars),
            **self.getJetAssignmentVars(ak4JetVarsOutsideFJ, name="ak4JetOutside", method="chi2"),
        }

        # gen variables
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](
                    events, jets, fatjets, selection, cutflow, gen_weights, P4
                )
                skimmed_events = {**skimmed_events, **vars_dict}

        ak4GenJetVars = {}
        if not isData:
            ak4GenJetVars = {
                f"ak4GenJet{key}": pad_val(events.GenJet[var], num_jets, axis=1)
                for (var, key) in self.skim_vars["GenJet"].items()
            }
            ak4JetVars = {
                **ak4JetVars,
                **{"ak4JetGenJetIdx": pad_val(getattr(jets, "genJetIdx"), num_jets, axis=1)},
            }

        skimmed_events = {
            **skimmed_events,
            **HLTVars,
            **ak4JetVars,
            **ak8FatJetVars,
            **ak4GenJetVars,
        }

        print("Vars", f"{time.time() - start:.2f}")

        ######################
        # Selection
        ######################

        # met filter selection
        met_filters = [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
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
        add_selection("met_filters", metfilters, *selection_args)
        # require at least one ak8 jet with PNscore > 0.8
        add_selection("1ak8_pt", np.any(fatjets.pt > 200, axis=1), *selection_args)
        add_selection("1ak8_xbb", np.any(fatjets.Txbb > 0.8, axis=1), *selection_args)
        # require at least two ak4 jets with Medium DeepJetM score (0.2783 for 2018)
        add_selection("2ak4_b", ak.sum(jets.btagDeepFlavB > 0.2783, axis=1) >= 2, *selection_args)
        # veto leptons
        add_selection(
            "0lep",
            (ak.sum(veto_muon_sel, axis=1) == 0) & (ak.sum(veto_electron_sel, axis=1) == 0),
            *selection_args,
        )

        """
        HLT_triggered = np.zeros(len(events), dtype="bool")
        for trigger in HLTs:
            if trigger in events.HLT.fields:
                HLT_triggered = HLT_triggered | events.HLT[trigger] 
        print(HLT_triggered)
        add_selection("trigger", HLT_triggered, *selection_args)
        add_selection("nak4", ak.sum(jets.pt > 15, axis=1) >= 4, *selection_args)
        """

        print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################
        if dataset in self.XSECS:
            xsec = self.XSECS[dataset]
            weight_norm = xsec * LUMI[year]
        else:
            logger.warning("Weight not normalized to cross section")
            weight_norm = 1

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights.add("genweight", gen_weights)
            skimmed_events["weight"] = weights.weight() * weight_norm

        if not self.apply_selection:
            skimmed_events = {
                key: value.reshape(len(skimmed_events["weight"]), -1)
                for (key, value) in skimmed_events.items()
            }
        else:
            # reshape and apply selections
            sel_all = selection.all(*selection.names)
            skimmed_events = {
                key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
                for (key, value) in skimmed_events.items()
            }

        df = self.to_pandas(skimmed_events)

        print("To Pandas", f"{time.time() - start:.2f}")

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def getJetAssignmentVars(self, ak4JetVars, name="ak4Jet", method="dhh", ak8Vars=None):
        """
        Calculates Jet assignment variables
        based on: https://github.com/cjmikkels/spanet_hh_test/blob/main/src/models/test_baseline.py
        """

        # just consider top 4 jets (already sorted by b-jet score)
        nj = 4
        jets = vector.array(
            {
                "pt": ak4JetVars[f"{name}Pt"],
                "eta": ak4JetVars[f"{name}Eta"],
                "phi": ak4JetVars[f"{name}Phi"],
                "M": ak4JetVars[f"{name}Mass"],
            },
        )

        if ak8Vars:
            fatjet_0 = vector.array(
                {
                    "pt": ak8Vars["ak8FatJetPt"][:, 0],
                    "eta": ak8Vars["ak8FatJetEta"][:, 0],
                    "phi": ak8Vars["ak8FatJetPhi"][:, 0],
                    "M": ak8Vars["ak8FatJetMass"][:, 0],
                },
            )

        # get array of dijets for each possible higgs combination
        jj = jets[:, self.JET_ASSIGNMENTS[nj][:, :, 0]] + jets[:, self.JET_ASSIGNMENTS[nj][:, :, 1]]
        mjj = jj.M

        if method == "chi2":
            chi2 = ak.sum(np.square(mjj - HIGGS_MASS), axis=-1)
            index = ak.argmin(chi2, axis=-1)

            first_bb_pair = self.JET_ASSIGNMENTS[nj][index][:, 0, :]
            second_bb_pair = self.JET_ASSIGNMENTS[nj][index][:, 1, :]
            return {
                "ak4Pair0chi2": first_bb_pair,
                "ak4Pair1chi2": second_bb_pair,
            }

        # TODO: add dR
        # elif method == "dr":

        elif method == "dhh":
            # https://github.com/UF-HH/bbbbAnalysis/blob/master/src/OfflineProducerHelper.cc#L4109
            mjj_sorted = ak.sort(mjj, ascending=False)

            # compute \delta d
            k = 125 / 120
            delta_d = np.absolute(mjj_sorted[:, :, 0] - k * mjj_sorted[:, :, 1]) / np.sqrt(
                1 + k**2
            )

            # take combination with smallest distance to the diagonal
            index_mindhh = ak.argmin(delta_d, axis=-1)

            # except, if |dhh^1 - dhh^2| < 30 GeV
            # this is when the pairing method starts to make mistakes
            d_sorted = ak.sort(delta_d, ascending=False)
            is_dhh_tooclose = (d_sorted[:, 0] - d_sorted[:, 1]) < 30

            # order dijets with the highest sum pt in their own event CoM frame
            # CoM frame of dijets
            cm = jj[:, :, 0] + jj[:, :, 1]
            com_pt = jj[:, :, 0].boostCM_of(cm).pt + jj[:, :, 1].boostCM_of(cm).pt
            index_max_com_pt = ak.argmax(com_pt, axis=-1)

            index = ak.where(is_dhh_tooclose, index_max_com_pt, index_mindhh)

            # TODO: is there an exception if the index chosen is the same?
            # is_same_index = (index == index_max_com_pt)

        # now get the resulting bb pairs
        first_bb_pair = self.JET_ASSIGNMENTS[nj][index][:, 0, :]
        first_bb_j1 = jets[np.arange(len(jets.pt)), first_bb_pair[:, 0]]
        first_bb_j2 = jets[np.arange(len(jets.pt)), first_bb_pair[:, 1]]
        first_bb_dijet = first_bb_j1 + first_bb_j2

        second_bb_pair = self.JET_ASSIGNMENTS[nj][index][:, 1, :]
        second_bb_j1 = jets[np.arange(len(jets.pt)), second_bb_pair[:, 0]]
        second_bb_j2 = jets[np.arange(len(jets.pt)), second_bb_pair[:, 1]]
        second_bb_dijet = second_bb_j1 + second_bb_j2

        # stack pairs
        bb_pairs = np.stack([first_bb_pair, second_bb_pair], axis=1)

        # sort by deltaR with leading fatjet
        bbs_dRfj = np.concatenate(
            [
                first_bb_dijet.deltaR(fatjet_0).reshape(-1, 1),
                second_bb_dijet.deltaR(fatjet_0).reshape(-1, 1),
            ],
            axis=1,
        )
        # sort from larger dR to smaller
        sort_by_dR = np.argsort(-bbs_dRfj, axis=-1)

        bb_pairs_sorted = np.array(
            [
                [bb_pair_e[sort_e[0]], bb_pair_e[sort_e[1]]]
                for bb_pair_e, sort_e in zip(bb_pairs, sort_by_dR)
            ]
        )

        first_bb_pair_sort = bb_pairs_sorted[:, 0]
        second_bb_pair_sort = bb_pairs_sorted[:, 1]

        first_bb_j1 = jets[np.arange(len(jets.pt)), first_bb_pair_sort[:, 0]]
        first_bb_j2 = jets[np.arange(len(jets.pt)), first_bb_pair_sort[:, 1]]
        first_bb_dijet = first_bb_j1 + first_bb_j2

        second_bb_j1 = jets[np.arange(len(jets.pt)), second_bb_pair_sort[:, 0]]
        second_bb_j2 = jets[np.arange(len(jets.pt)), second_bb_pair_sort[:, 1]]
        second_bb_dijet = second_bb_j1 + second_bb_j2

        jetAssignmentDict = {
            f"{name}Pair0": first_bb_pair_sort,
            f"{name}Pair1": second_bb_pair_sort,
            f"{name}DijetPt0": first_bb_dijet.pt,
            f"{name}DijetEta0": first_bb_dijet.eta,
            f"{name}DijetPhi0": first_bb_dijet.phi,
            f"{name}DijetMass0": first_bb_dijet.mass,
            f"{name}DijetPt1": second_bb_dijet.pt,
            f"{name}DijetEta1": second_bb_dijet.eta,
            f"{name}DijetPhi1": second_bb_dijet.phi,
            f"{name}DijetMass1": second_bb_dijet.mass,
            f"{name}DijetDeltaR": first_bb_dijet.deltaR(second_bb_dijet),
        }
        return jetAssignmentDict
