from __future__ import annotations

import logging
import warnings
from collections import OrderedDict

import awkward as ak
import numpy as np
from coffea import processor
from coffea.analysis_tools import PackedSelection
from hist import Hist

from .corrections import (
    get_jec_jets,
    get_jetveto_event,
)
from .GenSelection import gen_selection_HHbbbb, gen_selection_HHbbbb_simplified
from .objects import get_ak8jets, good_ak4jets
from .utils import P4, add_selection, dump_table, pad_val, to_pandas

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "GluGlutoHHto4B": gen_selection_HHbbbb,
    "VBFHHto4B": gen_selection_HHbbbb_simplified,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# some instructions to get filters from an HLT path
# https://github.com/IzaakWN/TriggerChecks/blob/master/plugins/CheckTriggers.cc
# https://github.com/IzaakWN/TriggerChecks/blob/master/python/TrigObjMatcher.py

# hh4b resolved measurement:
#  https://indico.cern.ch/event/1233765/contributions/5436483/attachments/2665045/4617964/CMSWeek_HiggsMeeting_13June2023.pdf
# hh4b boosted measurement:
#  https://indico.cern.ch/event/1301652/contributions/5513762/attachments/2689584/4666987/20230717%20_%20HH-_4b%20Boosted%20Run3%20Triggers.pdf

# semi-resolved trigger:
#  https://indico.cern.ch/event/1233757/contributions/5281432/attachments/2598122/4485548/HLTforHHto4b_MKolosova.pdf
#  https://indico.cern.ch/event/1251452/contributions/5288361/attachments/2602604/4494289/HLTforGluGluHHTo4b_Semiresolved_MKolosova_01March2023.pdf

# how to check for prescales
# e.g. https://cmsoms.cern.ch/cms/triggers/prescale?cms_run=368566&cms_run_sequence=GLOBAL-RUN
#  here look at Index 2 2p3E34 column
#  1=active, 0=disabled, N=1 in N events kept


class TriggerProcessor(processor.ProcessorABC):
    HLTs = {  # noqa: RUF012
        "2022": [
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
            "PFHT1050",
            "AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35",  # prescaled for fraction of 2022
            "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",  # un-prescaled
            "AK8PFJet275_SoftDropMass40_PFAK8ParticleNetBB0p35",  # un-prescaled
            "AK8PFJet400_SoftDropMass40",  # prescaled for fraction of 2022
            "AK8PFJet425_SoftDropMass40",  # un-prescaled
            "AK8PFJet450_SoftDropMass40",  # un-prescaled
            "AK8DiPFJet250_250_MassSD30",
            "AK8DiPFJet250_250_MassSD50",
            "AK8DiPFJet260_260_MassSD30",
            "AK8DiPFJet270_270_MassSD30",
        ],
        "2022EE": [
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
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
        "2023": [
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
            "PFHT1050",
            "AK8PFJet230_SoftDropMass40_PNetBB0p06",  # un-prescaled
            "AK8PFJet230_SoftDropMass40_PNetBB0p10",  # un-prescaled
            # "AK8PFJet250_SoftDropMass40_PNetBB0p06", # un-prescaled
            # "AK8PFJet250_SoftDropMass40_PNetBB0p10", # un-prescaled
            # "AK8PFJet275_SoftDropMass40_PNetBB0p06",
            # "AK8PFJet275_SoftDropMass40_PNetBB0p10",
            "AK8PFJet425_SoftDropMass40",
            # "AK8DiPFJet250_250_MassSD30", # inactive
            "AK8DiPFJet250_250_MassSD50",
            "AK8DiPFJet260_260_MassSD30",
            "AK8DiPFJet270_270_MassSD30",
            # https://hlt-config-editor-confdbv3.app.cern.ch/open?cfg=%2Ffrozen%2F2023%2F2e34%2Fv1.2%2FHLT%2FV1&db=offline-run3
            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/B2GTrigger#2023_online_menus
            "AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p50",  # main? un-prescaled
            "AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p53",
            "AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p55",  # backup?
            "AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p60",
        ],
    }

    # https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2022
    muon_HLTs = {  # noqa: RUF012
        "2022": [
            "IsoMu24",
            "Mu50",
            # CascadeMu100
            # HighPtTkMu100
        ],
        "2022EE": [
            "IsoMu24",
            "Mu50",
        ],
        "2023": [
            "IsoMu24",
            "Mu50",
        ],
    }

    L1s = {  # noqa: RUF012
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

    # check trigger object filters
    # https://hlt-config-editor-confdbv3.app.cern.ch/open?cfg=%2Fcdaq%2Fphysics%2FRun2022%2F2e34%2Fv1.5.0%2FHLT%2FV6&db=online

    # sequence for QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65
    #  L1sQuadJetOrHTTOrMuonHTT
    #  hlt4PixelOnlyPFCentralJetTightIDPt20
    #  hlt3PixelOnlyPFCentralJetTightIDPt30
    #  hlt2PixelOnlyPFCentralJetTightIDPt40
    #  hlt1PixelOnlyPFCentralJetTightIDPt60
    #  hlt4PFCentralJetTightIDPt35
    #  hlt3PFCentralJetTightIDPt40
    #  hlt2PFCentralJetTightIDPt50
    #  hlt1PFCentralJetTightIDPt70
    #  hltPFCentralJetTightIDPt35
    #  hltBTagCentralJetPt35PFParticleNet2BTagSum0p65

    # sequence for AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35
    #  L1sSingleJetOrHTTOrMuHTT
    #  hltAK8SingleCaloJet200
    ### hltSingleAK8PFJet250
    ### hltAK8PFJets250Constituents
    ### hltAK8PFSoftDropJets250
    ### hltAK8SinglePFJets250SoftDropMass40
    ### hltAK8PFJets250SoftDropMass40
    #  hltAK8SinglePFJets250SoftDropMass40BTagParticleNetBB0p35

    triggerObj_bits = {  # noqa: RUF012
        "hlt4PixelOnlyPFCentralJetTightIDPt20": 0,  # 4 jets, jetForthHighestPt_pt
        "hlt3PixelOnlyPFCentralJetTightIDPt30": 1,  # 3 jets, jetThirdHighestPt_pt
        "hlt2PixelOnlyPFCentralJetTightIDPt40": 6,  # 2 jets, jetSecondHighestPt_pt
        "hlt1PixelOnlyPFCentralJetTightIDPt60": 23,  # 1 jet, jetFirstHighestPt_pt
        "hlt4PFCentralJetTightIDPt35": 4,  # 4 jets, jetForthHighestPt_pt
        "hlt2PFCentralJetTightIDPt50": 22,  # 2 jets, jetSecondHighestPt_pt
        "hlt1PFCentralJetTightIDPt70": 24,  # 1 jet, jetFirstHighestPt_pt
        # hltBTagCentralJetPt35PFParticleNet2BTagSum0p65 OR hltBTagCentralJetPt30PFParticleNet2BTagSum0p65 OR hltPFJetTwoC30PFBTagParticleNet2BTagSum0p65
        "hltBTagCentralJetPt35PFParticleNet2BTagSum0p65": 27,  # jetFirstHighestPN_PNBtag
        "hltAK8PFJetsCorrected": 1,  # 1 fatjet
        "hltAK8SingleCaloJet200": 2,  # 1 fatjet
        "hltAK8PFSoftDropJets230": 4,  # 1 fatjet,
        # hltAK8SinglePFJets230SoftDropMass40BTagParticleNetBB0p35 OR hltAK8SinglePFJets250SoftDropMass40BTagParticleNetBB0p35 OR hltAK8SinglePFJets275SoftDropMass40BTagParticleNetBB0p35
        "hltAK8SinglePFJets230SoftDropMass40BTagParticleNetBB0p35": 12,  # 1 fatjet, fatjetFirstHighestPN_PNBB
        "hltAK8DoublePFJetSDModMass30": 16,  # 2 fatjet
        "hltAK8DoublePFJetSDModMass50": 48,  # 2 fatjet
    }

    def __init__(self):
        pass


class BoostedTriggerSkimmer(TriggerProcessor):
    def __init__(self, save_hist=False, nano_version="v12"):
        super(TriggerProcessor, self).__init__()
        self._nano_version = nano_version

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

        self.save_hist = save_hist

        # OR of HLTS for which to fill histograms
        self.HLTs_hist = {
            "2022": [
                "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                "AK8PFJet425_SoftDropMass40",
            ],
            "2022EE": [
                "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                "AK8PFJet425_SoftDropMass40",
            ],
        }

        # # bins, min, max
        msd_bins = (15, 0, 300)

        # edges
        pt_bins = [250, 275, 300, 325, 350, 375, 400, 450, 500, 600, 800, 1000]
        xbb_bins = [0.0, 0.8, 0.9, 0.95, 0.98, 1.0]

        # histogram
        self.h = (
            Hist.new.Var(xbb_bins, name="jet0txbb", label="$T_{Xbb}$ Score")
            .Var(pt_bins, name="jet0pt", label="$p_T$ (GeV)")
            .Reg(*msd_bins, name="jet0msd", label="$m_{SD}$ (GeV)")
            .Double()
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        """Returns processed information for trigger studies"""

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])
        isData = not hasattr(events, "genWeight")

        selection = PackedSelection()

        cutflow = OrderedDict()
        cutflow["all"] = len(events)

        selection_args = (selection, cutflow, True)

        # dictionary to save variables
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

        skimmed_events = {**HLT_vars, "run": events.run.to_numpy()}

        muons = events.Muon
        muon_selector = (
            (muons[f"{self.preselection['muon_id']}Id"])
            & (muons.pt > self.preselection["muon_pt"])
            & (np.abs(muons.eta) < self.preselection["muon_eta"])
            & (muons.pfIsoId >= self.preselection["muon_pfIsoId"])
        )
        leading_muon = ak.pad_none(muons[muon_selector], 1, axis=1)[:, 0]
        nmuons = ak.sum(muon_selector, axis=1)

        # Apply JECs from JME recommendations instead of MINI/NANOAOD
        num_fatjets = 2
        fatjets = get_ak8jets(events.FatJet)
        fatjets, _ = get_jec_jets(
            events,
            fatjets,
            year,
            isData,
            jecs=None,
            fatjets=True,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
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

        # add ht variable
        jets, _ = get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            jecs=None,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
        jets_sel = good_ak4jets(jets, year, events.run.to_numpy(), isData)
        jets = jets[jets_sel]
        ht = ak.sum(jets.pt, axis=1)
        skimmed_events["ht"] = ht.to_numpy()

        if year == "2022" or year == "2022EE":
            jetveto_selection = get_jetveto_event(jets, year, events.run.to_numpy(), isData)
            add_selection("ak4_jetveto", jetveto_selection, *selection_args)

        # trigger objects
        """
        # fields: 'pt', 'eta', 'phi', 'l1pt', 'l1pt_2', 'l2pt', 'id', 'l1iso', 'l1charge', 'filterBits'
        fatjet_obj = events.TrigObj[(events.TrigObj.id == 6)]
        jet_obj = events.TrigObj[(events.TrigObj.id == 1)]

        # trigger objects matched to leading jet and
        matched_obj_fatjet0 = fatjet_obj[
            ak.any(fatjet_obj.metric_table(fatjets[:, 0:1]) < 1.0, axis=1)
        ]
        matched_obj_fatjet1 = fatjet_obj[
            ak.any(fatjet_obj.metric_table(fatjets[:, 1:2]) < 1.0, axis=1)
        ]

        num_trigobjs = 1
        trigObjFatJetVars = {
            # not matched to fatjets
            f"trigObj_filterBits": pad_val(fatjet_obj["filterBits"], num_trigobjs, axis=1),
            # matched to leading fatjet
            f"trigObjFatJet0_filterBits": pad_val(
                matched_obj_fatjet0["filterBits"], num_trigobjs, axis=1
            ),
            # matched to sub-leading fatjet
            f"trigObjFatJet1_filterBits": pad_val(
                matched_obj_fatjet1["filterBits"], num_trigobjs, axis=1
            ),
        }
        skimmed_events = {**skimmed_events, **trigObjFatJetVars}
        """

        # gen variables
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, jets, fatjets, selection_args, P4)
                skimmed_events = {**skimmed_events, **vars_dict}

        # reshape and apply selections
        sel_all = selection.all(*selection.names)

        if self.save_hist:
            bbbb_triggered = np.any(
                np.array(
                    [
                        events.HLT[trigger]
                        for trigger in self.HLTs_hist[year]
                        if trigger in events.HLT.fields
                    ]
                ),
                axis=0,
            )

            selections = {
                # select events which pass the muon triggers and selection
                "den": sel_all,
                # add our triggers
                "num": sel_all * bbbb_triggered,
            }

            hists = {}
            for key, sel in selections.items():
                hists[key] = self.h.copy().fill(
                    jet0txbb=fatjets.txbb[sel][:, 0].to_numpy(),
                    jet0pt=fatjets.pt[sel][:, 0].to_numpy(),
                    jet0msd=fatjets.msoftdrop[sel][:, 0].to_numpy(),
                )
            return hists

        skimmed_events = {
            key: value.reshape(len(events), -1)[sel_all] for (key, value) in skimmed_events.items()
        }
        """
        # no selection  (for signal study)
        skimmed_events = {
            key: value.reshape(len(events), -1) for (key, value) in skimmed_events.items()
        }
        """

        dataframe = to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        dump_table(dataframe, fname)

        return {}

    def postprocess(self, accumulator):
        return accumulator
