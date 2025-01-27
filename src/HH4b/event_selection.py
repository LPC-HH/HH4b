"""
Event selection module for the boosted HH->4b analysis.

Author: Daniel Primosch
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class EventSelection:
    def __init__(self, year, jet_coll_tagger, jet_coll_mass):
        # self.path_to_dir = path_to_dir
        self.year = year
        self.jet_collection = "bbFatJet"
        self.jet_coll_tagger = jet_coll_tagger
        self.jet_coll_mass = jet_coll_mass

        self.txbb_preselection = {
            "bbFatJetPNetTXbb": 0.3,
            "bbFatJetTXbbLegacy": 0.8,
            "bbFatJetParTTXbb": 0.3,
        }
        self.msd1_preselection = {
            "bbFatJetPNetTXbb": 40,
            "bbFatJetPNetTXbbLegacy": 40,
            "bbFatJetParTTXbb": 40,
        }
        self.msd2_preselection = {
            "bbFatJetPNetTXbb": 30,
            "bbFatJetPNetTXbbLegacy": 0,
            "bbFatJetParTTXbb": 30,
        }
        self.txbb_str = self.jet_collection + jet_coll_tagger
        self.mass_str = self.jet_collection + jet_coll_mass
        self.reorder_txbb = True

        self.sample_dirs = {
            year: {
                "qcd": [
                    "QCD_HT-1000to1200",
                    "QCD_HT-1200to1500",
                    "QCD_HT-1500to2000",
                    "QCD_HT-2000",
                    "QCD_HT-400to600",
                    "QCD_HT-600to800",
                    "QCD_HT-800to1000",
                ],
                "ttbar": [
                    "TTto4Q",
                ],
                "diboson": [
                    "WW",
                    "WZ",
                    "ZZ",
                ],
                "VBFHH": [
                    # ...
                ],
                "VBFH": [
                    "VBFHto2B_M-125_dipoleRecoilOn",
                ],
            },
        }

        self.sample_dirs_sig = {
            year: {
                "hh4b": [
                    "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"
                ],  # the ? enforces exact matching
            }
        }

        self.num_jets = 2
        self.columns = [
            ("weight", 1),
            ("event", 1),
            ("MET_pt", 1),
            ("bbFatJetTau3OverTau2", 2),
            ("VBFJetPt", 2),
            ("VBFJetEta", 2),
            ("VBFJetPhi", 2),
            ("VBFJetMass", 2),
            ("AK4JetAwayPt", 2),
            ("AK4JetAwayEta", 2),
            ("AK4JetAwayPhi", 2),
            ("AK4JetAwayMass", 2),
            (f"{self.jet_collection}Pt", self.num_jets),
            (f"{self.jet_collection}Msd", self.num_jets),
            (f"{self.jet_collection}Eta", self.num_jets),
            (f"{self.jet_collection}Phi", self.num_jets),
            (f"{self.jet_collection}rawFactor", self.num_jets),
            (f"{self.jet_collection}PNetPXbbLegacy", self.num_jets),
            (f"{self.jet_collection}PNetPQCDbLegacy", self.num_jets),
            (f"{self.jet_collection}PNetPQCDbbLegacy", self.num_jets),
            (f"{self.jet_collection}PNetPQCD0HFLegacy", self.num_jets),
            (f"{self.jet_collection}PNetMassLegacy", self.num_jets),
            (f"{self.jet_collection}PNetTXbbLegacy", self.num_jets),
            (f"{self.jet_collection}PNetTXbb", self.num_jets),
            (f"{self.jet_collection}PNetMass", self.num_jets),
            (f"{self.jet_collection}PNetQCD0HF", self.num_jets),
            (f"{self.jet_collection}PNetQCD1HF", self.num_jets),
            (f"{self.jet_collection}PNetQCD2HF", self.num_jets),
            (f"{self.jet_collection}ParTmassVis", self.num_jets),
            (f"{self.jet_collection}ParTTXbb", self.num_jets),
            (f"{self.jet_collection}ParTPXbb", self.num_jets),
            (f"{self.jet_collection}ParTPQCD0HF", self.num_jets),
            (f"{self.jet_collection}ParTPQCD1HF", self.num_jets),
            (f"{self.jet_collection}ParTPQCD2HF", self.num_jets),
        ]

        # Additional columns for signal only:
        self.signal_exclusive_columns = []

        # selection to apply
        # TODO: is this redundant with boosted selection?
        self.filters = [
            [
                (f"('{self.jet_collection}Pt', '0')", ">=", 300),
                (f"('{self.jet_collection}Pt', '1')", ">=", 250),
            ],
        ]

    def set_jet_collection(self, jet_collection: str):
        self.jet_collection = jet_collection
        self.txbb_str = self.jet_collection + self.jet_coll_tagger
        self.mass_str = self.jet_collection + self.jet_coll_mass

    def get_samples(self):
        # Combine the background samples + signal samples
        return self.sample_dirs, self.sample_dirs_sig

    def get_columns(self):
        return self.columns, self.signal_exclusive_columns

    def apply_boosted(self, events_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        txbb_str = self.txbb_str
        mass_str = self.mass_str
        jet_collection = self.jet_collection

        for key, df in events_dict.items():
            msd1 = df[f"{jet_collection}Msd"][0]
            msd2 = df[f"{jet_collection}Msd"][1]
            pt1 = df[f"{jet_collection}Pt"][0]
            pt2 = df[f"{jet_collection}Pt"][1]
            txbb1 = df[txbb_str][0]
            mass1 = df[mass_str][0]
            mass2 = df[mass_str][1]

            selection_mask = (
                (pt1 > 300)
                & (pt2 > 250)
                & (txbb1 > self.txbb_preselection[txbb_str])
                & (msd1 > self.msd1_preselection[txbb_str])
                & (msd2 > self.msd2_preselection[txbb_str])
                & (mass1 > 50)
                & (mass2 > 50)
            )
            events_dict[key] = df[selection_mask].copy()
        return events_dict
