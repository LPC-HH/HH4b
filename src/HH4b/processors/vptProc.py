"""
VJets cross check
"""
from __future__ import annotations

import awkward as ak
import hist
import numpy as np
from coffea import processor


class vptProc(processor.ProcessorABC):
    edges = np.array(
        [
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
            150.0,
            200.0,
            250.0,
            300.0,
            350.0,
            400.0,
            450.0,
            500.0,
            550.0,
            600.0,
            650.0,
            700.0,
            750.0,
            800.0,
            850.0,
            900.0,
            950.0,
            1000.0,
            1100.0,
            1200.0,
            1300.0,
            1400.0,
            1600.0,
            1800.0,
            2000.0,
            2200.0,
            2400.0,
            2600.0,
            2800.0,
            3000.0,
            6500.0,
        ]
    )

    def process(self, events):
        boson = ak.firsts(
            events.GenPart[
                ((events.GenPart.pdgId == 23) | (abs(events.GenPart.pdgId) == 24))
                & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            ]
        )
        vpt = ak.fill_none(boson.pt, 0.0)
        offshell = events.GenPart[
            events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(events.GenPart.pdgId) >= 11)
            & (abs(events.GenPart.pdgId) <= 16)
        ].sum()
        scalevar = hist.Hist.new.IntCat(range(8)).Var(self.edges).Weight()
        for i in range(8):
            scalevar.fill(
                i, vpt + offshell.pt, weight=events.genWeight * events.LHEScaleWeight[:, i]
            )

        return {
            events.metadata["dataset"]: {
                "Vpt_resonant": (
                    hist.Hist.new.Var(self.edges).Weight().fill(vpt, weight=events.genWeight)
                ),
                "Vpt_nano": (
                    hist.Hist.new.Var(self.edges)
                    .Weight()
                    .fill(events.LHE.Vpt, weight=events.genWeight)
                ),
                "Vpt": (
                    hist.Hist.new.Var(self.edges)
                    .Weight()
                    .fill(vpt + offshell.pt, weight=events.genWeight)
                ),
                "Vptscale": scalevar,
                "Vmass": (
                    hist.Hist.new.StrCat(["V", "offshell"])
                    .Reg(105, 30, 240, name="mass")
                    .Weight()
                    .fill("V", ak.fill_none(boson.mass, 0.0), weight=events.genWeight)
                    .fill("offshell", offshell.mass, weight=events.genWeight)
                ),
                "sumw": ak.sum(events.genWeight),
            }
        }

    def postprocess(self, x):
        return x
