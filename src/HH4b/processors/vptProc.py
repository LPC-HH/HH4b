"""
VJets cross check
"""

import logging
import time
from collections import OrderedDict
from copy import deepcopy

import awkward as ak
import numpy as np
import vector
import hist
from coffea import processor

class vptProc(processor.ProcessorABC):
    edges = np.array([  30.,   40.,   50.,   60.,   70.,   80.,   90.,  100.,  110.,
        120.,  130.,  140.,  150.,  200.,  250.,  300.,  350.,  400.,
        450.,  500.,  550.,  600.,  650.,  700.,  750.,  800.,  850.,
        900.,  950., 1000., 1100., 1200., 1300., 1400., 1600., 1800.,
       2000., 2200., 2400., 2600., 2800., 3000., 6500.])

    def process(self, events):
        boson = ak.firsts(events.GenPart[
            ((events.GenPart.pdgId == 23)|(abs(events.GenPart.pdgId) == 24))
            & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
        ])
        vpt = ak.fill_none(boson.pt, 0.)
        offshell = events.GenPart[
            events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(events.GenPart.pdgId) >= 11) & (abs(events.GenPart.pdgId) <= 16)
        ].sum()
        scalevar = (
            hist.Hist.new
            .IntCat(range(8))
            .Var(self.edges).Weight()
        )
        for i in range(8):
            scalevar.fill(i, vpt + offshell.pt, weight=events.genWeight * events.LHEScaleWeight[:, i])

        return {
            events.metadata["dataset"]: {
                "Vpt_resonant": (
                    hist.Hist.new
                    .Var(self.edges).Weight()
                    .fill(vpt, weight=events.genWeight)
                ),
                "Vpt_nano": (
                    hist.Hist.new
                    .Var(self.edges).Weight()
                    .fill(events.LHE.Vpt, weight=events.genWeight)
                ),
                "Vpt": (
                    hist.Hist.new
                    .Var(self.edges).Weight()
                    .fill(vpt + offshell.pt, weight=events.genWeight)
                ),
                "Vptscale": scalevar,
                "Vmass": (
                    hist.Hist.new
                    .StrCat(["V", "offshell"])
                    .Reg(105, 30, 240, name="mass")
                    .Weight()
                    .fill("V", ak.fill_none(boson.mass, 0.), weight=events.genWeight)
                    .fill("offshell", offshell.mass, weight=events.genWeight)
                ),
                "sumw": ak.sum(events.genWeight),
            }
        }

    def postprocess(self, x):
        return x
