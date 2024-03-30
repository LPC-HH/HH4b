"""
Skimmer for matching studies
Author(s): Raghav Kansal, Cristina Suarez
"""
from __future__ import annotations

import itertools
import logging
import time
from collections import OrderedDict

import awkward as ak
import numpy as np
import vector
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

from . import objects
from .common import LUMI
from .GenSelection import gen_selection_HHbbbb
from .utils import P4, PAD_VAL, add_selection, pad_val

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "GluGlutoHHto4B": gen_selection_HHbbbb,
}


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
    skim_vars = {  # noqa: RUF012
        "Jet": {
            "btagDeepFlavB": "btagDeepFlavB",
        },
        # TODO: add hadron flavour
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "Txbb": "PNetXbb",
            "Txjj": "PNetXjj",
            "Tqcd": "PNetQCD",
            "particleNet_mass": "PNetMass",
        },
        "GenJet": P4,
        "Lepton": P4,
        "Tau": P4,
        "Other": {
            "MET_pt": "MET_pt",
        },
    }

    # compute possible jet assignments lookup table
    JET_ASSIGNMENTS = {}  # noqa: RUF012
    for nj in range(MIN_JETS, MAX_JETS + 1):
        a = list(itertools.combinations(range(nj), 2))
        b = np.array(
            [(i, j) for i, j in itertools.combinations(a, 2) if len(set(i + j)) == MIN_JETS]
        )
        JET_ASSIGNMENTS[nj] = b

    def __init__(self, xsecs=None, apply_selection=True, nano_version="v9"):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb
        self._accumulator = processor.dict_accumulator({})
        self.apply_selection = apply_selection
        self._nano_version = nano_version
        print(self.apply_selection)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        start = time.time()
        print("Starting")

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])

        btag_vars = {}
        if self._nano_version == "v9_private":
            btag_vars = {
                "ParticleNetAK4_probb": "btagPNetProbb",
                "ParticleNetAK4_probbb": "btagPNetProbbb",
                "ParticleNetAK4_probc": "btagPNetProbc",
                "ParticleNetAK4_probcc": "btagPNetProbcc",
                "ParticleNetAK4_probpu": "btagPNetProbpu",
                "ParticleNetAK4_probuds": "btagPNetProbuds",
                "ParticleNetAK4_probg": "btagPNetProbg",
                "ParticleNetAK4_probundef": "btagPNetProbundef",
                # "btagPNetBvsAll": "btagPNetBvsAll",
            }

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
        if year == "2018":
            veto_muon_sel = objects.veto_muons_run2(events.Muon)
            veto_electron_sel = objects.veto_electrons_run2(events.Electron)
            veto_muons = events.Muon[veto_muon_sel]
            veto_electrons = events.Electron[veto_electron_sel]
        else:
            veto_muon_sel = objects.veto_muons(events.Muon)
            veto_electron_sel = objects.veto_electrons(events.Electron)
            veto_muons = events.Muon[veto_muon_sel]
            veto_electrons = events.Electron[veto_electron_sel]

        loose_taus = events.Tau[objects.loose_taus(events.Tau)]

        print("Before jets ", f"{time.time() - start:.2f}")

        jets = events.Jet
        jets = jets[objects.good_ak4jets(jets, year, events.run.to_numpy(), isData)]
        central_jets_sel = np.abs(jets.eta) < 2.5
        central_jets = jets[central_jets_sel]
        ht = ak.sum(jets.pt, axis=1)

        print("Before fatjets ", f"{time.time() - start:.2f}")

        fatjets = objects.get_ak8jets(events.FatJet)
        fatjets_sel = objects.good_ak8jets(fatjets)
        fatjets = fatjets[fatjets_sel]
        fatjets = fatjets[ak.argsort(fatjets.Txbb, ascending=False)]

        print("Before outside ", f"{time.time() - start:.2f}")

        # get jets outside the leading fj (in bb)
        leading_fj = ak.firsts(fatjets)
        outside_fj_sel = central_jets.delta_r(leading_fj) > 0.8
        jets_outside_fj = central_jets[outside_fj_sel]
        jets_outside_fj = jets_outside_fj[
            ak.argsort(jets_outside_fj["btagDeepFlavB"], ascending=False)
        ]

        print("Before b-jet energy regression", f"{time.time() - start:.2f}")

        # b-jet energy regression
        jets_p4 = objects.bregcorr(jets)
        jets_p4_outside_fj = objects.bregcorr(jets_outside_fj)

        # vbf jets (preliminary)
        vbf_jets = jets[(jets.pt > 25) & (jets.delta_r(leading_fj) > 1.2)]
        vbf_jet_0 = vbf_jets[:, 0:1]
        vbf_jet_1 = vbf_jets[:, 1:2]
        vbf_mass = (ak.firsts(vbf_jet_0) + ak.firsts(vbf_jet_1)).mass
        vbf_deta = abs(ak.firsts(vbf_jet_0).eta - ak.firsts(vbf_jet_1).eta)
        vbf_selection = (vbf_mass > 500) & (vbf_deta > 4.0)

        print("Objects", f"{time.time() - start:.2f}")

        #########################
        # Save / derive variables
        #########################
        skimmed_events = {}
        skimmed_events["ht"] = ht.to_numpy()
        skimmed_events["nCentralJets"] = ak.sum(central_jets_sel, axis=1).to_numpy()
        skimmed_events["nOutsideJets"] = ak.sum(outside_fj_sel, axis=1).to_numpy()
        skimmed_events["nFatJets"] = ak.sum(fatjets_sel, axis=1).to_numpy()
        skimmed_events["vbf_selection"] = vbf_selection.to_numpy()

        # HLT variables
        HLTs = {
            "2018": [
                "PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5",
                "PFHT1050",
                "PFJet500",
                "AK8PFJet500",
                "AK8PFJet400_TrimMass30",
                "AK8PFHT800_TrimMass50",
                "AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4",
                "QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",
                "PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
                "PFHT450_SixPFJet36_PFBTagDeepCSV_1p59",
                "AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17",
                "QuadPFJet98_83_71_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",
                "QuadPFJet98_83_71_15_PFBTagDeepCSV_1p3_VBF2",
                "PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1",
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
        # num_jets = 6
        num_jets = 10
        jet_vars = {**self.skim_vars["Jet"], **btag_vars}
        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_jets, value=PAD_VAL, axis=1)
            for (var, key) in jet_vars.items()
        }
        ak4JetVarsOutsideFJ = {
            f"ak4JetOutside{key}": pad_val(jets_outside_fj[var], num_jets, value=PAD_VAL, axis=1)
            for (var, key) in jet_vars.items()
        }
        for var, key in P4.items():
            ak4JetVars[f"ak4Jet{key}"] = pad_val(
                getattr(jets_p4, var), num_jets, value=PAD_VAL, axis=1
            )
            ak4JetVarsOutsideFJ[f"ak4JetOutside{key}"] = pad_val(
                getattr(jets_p4_outside_fj, var), num_jets, value=PAD_VAL, axis=1
            )

        print("AK4Vars", f"{time.time() - start:.2f}")

        # FatJet variables
        # num_fatjets = 2
        num_fatjets = 3
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, value=PAD_VAL, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        ak4JetVars = {
            **ak4JetVars,
            **ak4JetVarsOutsideFJ,
            # **self.getJetAssignmentVars(ak4JetVars, ak8Vars=ak8FatJetVars),
            # **self.getJetAssignmentVars(ak4JetVarsOutsideFJ, name="ak4JetOutside", method="chi2"),
        }

        # GenMatching variables
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, jets, fatjets, selection_args, P4)
                skimmed_events = {**skimmed_events, **vars_dict}

        ak4GenJetVars = {}
        """
        if not isData:
            ak4GenJetVars = {
                f"ak4GenJet{key}": pad_val(events.GenJet[var], num_jets, axis=1)
                for (var, key) in self.skim_vars["GenJet"].items()
            }
            ak4JetVars = {
                **ak4JetVars,
                "ak4JetGenJetIdx": pad_val(jets.genJetIdx, num_jets, axis=1),
            }
        """

        # Other variables
        num_leptons = 2
        leptonVars = {}
        veto_leptons_pt = ak.concatenate([veto_electrons.pt, veto_muons.pt], axis=1)
        for var, key in self.skim_vars["Lepton"].items():
            veto_leptons_var = ak.concatenate([veto_electrons[var], veto_muons[var]], axis=1)
            veto_leptons_var = veto_leptons_var[ak.argsort(veto_leptons_pt, ascending=False)]
            leptonVars[f"Lepton{key}"] = pad_val(veto_leptons_var, num_leptons, value=0, axis=1)

        num_taus = 2
        tauVars = {
            f"tau{key}": pad_val(loose_taus[var], num_taus, value=0, axis=1)
            for (var, key) in self.skim_vars["Tau"].items()
        }

        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["Other"].items()
        }

        skimmed_events = {
            **skimmed_events,
            **HLTVars,
            **ak4JetVars,
            **ak8FatJetVars,
            **ak4GenJetVars,
            **leptonVars,
            **tauVars,
            **otherVars,
        }

        print("Vars", f"{time.time() - start:.2f}")

        ######################
        # Selection
        ######################

        # trigger selection
        HLTs_selection = {
            "2018": [
                "PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5",
            ]
        }[year]
        HLT_triggered = np.any(
            np.array(
                [events.HLT[trigger] for trigger in HLTs_selection if trigger in events.HLT.fields]
            ),
            axis=0,
        )
        # apply trigger to both data and mc
        add_selection("trigger", HLT_triggered, *selection_args)

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
        for mf in met_filters:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        add_selection("met_filters", metfilters, *selection_args)

        # require one good ak8 jet
        add_selection("1ak8_pt", leading_fj.pt > 300, *selection_args)
        add_selection("1ak8_mass", leading_fj.particleNet_mass > 60, *selection_args)
        add_selection("1ak8_xbb", leading_fj.Txbb > 0.8, *selection_args)

        # require at least two ak4 jets with Medium DeepJetM score
        # DeepJetM (0.2783 for 2018)
        # PNetAK4M (0.2605 for 2022EE)
        nbjets_deepjet = ak.sum(central_jets.btagDeepFlavB > 0.2783, axis=1)
        nbjets_sel = nbjets_deepjet >= 2
        # if self._nano_version == "v9_private":
        #    nbjets_pnet = ak.sum(jets.
        add_selection("2ak4_b", nbjets_sel, *selection_args)

        # veto leptons
        # add_selection(
        #     "0lep",
        #     (ak.sum(veto_muon_sel, axis=1) == 0) & (ak.sum(veto_electron_sel, axis=1) == 0),
        #     *selection_args,
        # )

        # veto events with VBF jets
        # add_selection("vbf_veto", ~(vbf_selection), *selection_args)

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

        dataframe = self.to_pandas(skimmed_events)
        print("To Pandas", f"{time.time() - start:.2f}")
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(dataframe, fname)

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

        if method == "dhh":
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

        return {
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
