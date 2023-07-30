from collections import OrderedDict
from typing import Dict
import pandas as pd
import awkward as ak
import numpy as np
import os

from coffea import processor
from coffea.analysis_tools import PackedSelection
from hist import Hist

from .utils import add_selection, P4, pad_val, flatten_dict
from .objects import get_ak8jets

import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in divide")


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TriggerProcessor(processor.ProcessorABC):
    HLTs = {
        "2022": [
            # "QuadPFJet70_50_40_30",
            # "QuadPFJet70_50_40_30_PFBTagParticleNet_2BTagSum0p65",
            # "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
            # "QuadPFJet70_50_45_35_PFBTagParticleNet_2BTagSum0p65",
            "PFHT1050",
            "AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet275_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet400_SoftDropMass40",
            "AK8PFJet425_SoftDropMass40",
            "AK8PFJet450_SoftDropMass40",
            "AK8DiPFJet250_250_MassSD30",
            "AK8DiPFJet250_250_MassSD50",
            "AK8DiPFJet260_260_MassSD30",
            "AK8DiPFJet270_270_MassSD30",
        ],
        "2022EE": [
            # "QuadPFJet70_50_40_30",
            # "QuadPFJet70_50_40_30_PFBTagParticleNet_2BTagSum0p65",
            # "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
            # "QuadPFJet70_50_45_35_PFBTagParticleNet_2BTagSum0p65",
            "PFHT1050",
            "AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet275_SoftDropMass40_PFAK8ParticleNetBB0p35",
            "AK8PFJet400_SoftDropMass40",
            "AK8PFJet425_SoftDropMass40",
            "AK8PFJet450_SoftDropMass40",
            "AK8DiPFJet250_250_MassSD30",
            "AK8DiPFJet250_250_MassSD50",
            "AK8DiPFJet260_260_MassSD30",
            "AK8DiPFJet270_270_MassSD30",
        ],
    }

    muon_HLTs = {
        "2022": [
            "IsoMu24",
            "IsoMu27",
            "Mu27",
            "Mu50",
            "Mu55",
        ],
        "2022EE": [
            "IsoMu24",
            "IsoMu27",
            "Mu27",
            "Mu50",
            "Mu55",
        ],
    }

    egamma_HLTs = {
        "2022": [
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "2022EE": [
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
    }

    L1s = {
        "2022": [
            "Mu6_HTT240er",  # un-prescaled
            "QuadJet60er2p5",
            "HTT280er",
            "HTT320er",
            "HTT360er",  # un-prescaled
            "HTT400er",  # un-prescaled
            "HTT450er",  # un-prescaled
            "HTT280er_QuadJet_70_55_40_35_er2p5",
            "HTT320er_QuadJet_70_55_40_40_er2p5",  # un-prescaled
            "HTT320er_QuadJet_80_60_er2p1_45_40_er2p3",  # un-prescaled
            "HTT320er_QuadJet_80_60_er2p1_50_45_er2p3",  # un-prescaled
        ],
    }

    def __init__(self):
        pass

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


class BoostedTriggerSkimmer(TriggerProcessor):
    def __init__(self):
        super(TriggerProcessor, self).__init__()

        self.skim_vars = {
            "FatJet": {
                **P4,
                "msoftdrop": "Msd",
                "Txbb": "PNetXbb",
                "Txjj": "PNetXjj",
                "particleNet_mass": "PNetMass",
            },
        }

        self.preselection = {
            "fatjet_pt": 200,
            "fatjet_eta": 2.5,
            "fatjet_dr_muon": 1.5,
            "muon_id": "tight",
            "muon_pt": 30,
            "muon_eta": 2.4,
            "muon_pfIsoId": 4,  # tight PF isolation
            "nmuons": 1,
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        """Returns processed information for trigger studies"""

        year = events.metadata["dataset"][:4]

        selection = PackedSelection()

        cutflow = OrderedDict()
        cutflow["all"] = len(events)

        selection_args = (selection, cutflow, True)

        """
        Save variables
        """
        skimmed_events = {}

        HLTs = self.HLTs[year]
        zeros = np.zeros(len(events), dtype="bool")
        HLT_vars = {
            trigger: (
                events.HLT[trigger].to_numpy().astype(int)
                if trigger in events.HLT.fields
                else zeros
            )
            for trigger in HLTs
        }

        skimmed_events = {**HLT_vars, **{"run": events.run.to_numpy()}}

        muons = events.Muon
        muon_selector = (
            (muons[f"{self.preselection['muon_id']}Id"])
            & (muons.pt > self.preselection["muon_pt"])
            & (np.abs(muons.eta) < self.preselection["muon_eta"])
            & (muons.pfIsoId >= self.preselection["muon_pfIsoId"])
        )
        leading_muon = ak.pad_none(muons[muon_selector], 1, axis=1)[:, 0]
        nmuons = ak.sum(muon_selector, axis=1)

        num_fatjets = 2
        fatjets = get_ak8jets(events.FatJet)
        fatjet_selector = (
            (fatjets.pt > self.preselection["fatjet_pt"])
            & (np.abs(fatjets.eta) < self.preselection["fatjet_eta"])
            & (np.abs(fatjets.delta_r(leading_muon)) > self.preselection["fatjet_dr_muon"])
        )
        fatjets = fatjets[fatjet_selector]
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_fatjets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }
        ak8FatJetVars = flatten_dict(ak8FatJetVars, "ak8FatJet")
        skimmed_events = {**skimmed_events, **ak8FatJetVars}

        # add muon reference trigger
        muon_triggered = np.any(
            np.array([events.HLT[trigger] for trigger in self.muon_HLTs[year]]),
            axis=0,
        )
        add_selection("muon_trigger", muon_triggered, *selection_args)

        # add one muon
        add_selection("one_muon", (nmuons == 1), *selection_args)

        # add at least one fat jet selection
        add_selection("ak8jet_pt_and_dR", ak.any(fatjet_selector, axis=1), *selection_args)

        # reshape and apply selections
        sel_all = selection.all(*selection.names)

        skimmed_events = {
            key: value.reshape(len(events), -1)[sel_all] for (key, value) in skimmed_events.items()
        }

        df = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {}

    def postprocess(self, accumulator):
        return accumulator


class BoostedTriggerProcessor(TriggerProcessor):
    def __init__(self):
        super(TriggerProcessor, self).__init__()

        self.skim_vars = {
            "FatJet": {
                **P4,
                "msoftdrop": "Msd",
                "Txbb": "PNetXbb",
                "Txjj": "PNetXjj",
                "particleNet_mass": "PNetMass",
            },
        }

        self.preselection = {
            "fatjet_pt": 200,
            "fatjet_eta": 2.5,
            "fatjet_dr_muon": 1.5,
            "muon_id": "tight",
            "muon_pt": 30,
            "muon_eta": 2.4,
            "muon_pfIsoId": 4,
            "nmuons": 1,
        }


class ResolvedTriggerSkimmer(TriggerProcessor):
    """
    Note: to be used with NanoAOD that has PNet AK4 scores
    """

    def __init__(self):
        super(TriggerProcessor, self).__init__()

        self.skim_vars = {
            "Jet": {
                **P4,
            },
            "FatJet": {
                **P4,
                "msoftdrop": "Msd",
                "Txbb": "PNetXbb",
                "Txjj": "PNetXjj",
                "particleNet_mass": "PNetMass",
            },
        }

        self.preselection = {
            "muon_pt": 10,
            "muon_eta": 2.4,
            "muon_id": "medium",
            "muon_dz": 0.5,
            "muon_dxy": 0.2,
            "muon_iso": 0.20,
            "electron_pt": 25,
            "electron_eta": 2.5,
            "electron_mvaIso": 0.80,
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        """Returns processed information for trigger studies"""

        year = events.metadata["dataset"][:4]

        selection = PackedSelection()

        cutflow = OrderedDict()
        cutflow["all"] = len(events)

        selection_args = (selection, cutflow, True)

        # save variables
        skimmed_events = {}

        HLTs = self.HLTs[year]
        zeros = np.zeros(len(events), dtype="bool")
        HLT_vars = {
            trigger: (
                events.HLT[trigger].to_numpy().astype(int)
                if trigger in events.HLT.fields
                else zeros
            )
            for trigger in HLTs
        }

        skimmed_events = {**HLT_vars, **{"run": events.run.to_numpy()}}

        # muon and electron
        muons = events.Muon
        muon_selector = (
            (muons[f"{self.preselection['muon_id']}Id"])
            & (muons.pt > self.preselection["muon_pt"])
            & (np.abs(muons.eta) < self.preselection["muon_eta"])
            & (muons.pfIsoId >= self.preselection["muon_pfIsoId"])
            & (np.abs(muons.dz) < self.preselection["muon_dz"])
            & (np.abs(muons.dxy) < self.preselection["muon_dxy"])
            & (muons.pfRelIso04_all < self.preselection["muon_iso"])
        )
        nmuons = ak.sum(muon_selector, axis=1)

        veto_muon_selector = (muons.pt > 10) & (muons.looseId) & (muons.pfRelIso04_all < 0.25)

        electrons = events.Electron
        electron_selector = (
            (electrons.pt > self.preselection["electron_pt"])
            & (np.abs(electrons.eta) < self.preselection["electron_eta"])
            & (electrons.mvaFall17V2Iso_WP80)
        )
        nelectrons = ak.sum(electron_selector, axis=1)

        veto_electron_selector = (electrons.pt > 15) & (electrons.mvaFall17V2Iso_WP90)
