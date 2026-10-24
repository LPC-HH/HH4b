"""
JECs and JER of every Run 3 year from the official correctionlib files: correctionlib-backed
stand-ins for the four parts of a coffea 0.7 ``JECStack``. The unchanged ``CorrectedJetsFactory`` then applies the nominal JEC, the hybrid JER
smearing and the JES/JER variations, and the corrected jets keep the fields of the text-file path
(``pt``, ``mass``, ``JES_jes``, ``JER``).

Author(s): Zichun Hao
"""

from __future__ import annotations

from functools import lru_cache

import awkward as ak
import correctionlib
import numpy as np
from coffea.jetmet_tools import CorrectedJetsFactory

# correctionlib input names -> jet fields; as build_jec.jec_name_map, plus the run number for the
# run-dependent data residuals
jec_name_map = {
    "JetPt": "pt",
    "JetMass": "mass",
    "JetEta": "eta",
    "JetPhi": "phi",
    "JetA": "area",
    "ptGenJet": "pt_gen",
    "ptRaw": "pt_raw",
    "massRaw": "mass_raw",
    "Rho": "event_rho",
    "run": "run",
}


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, ak.Array):
        arr = ak.to_numpy(ak.materialized(arr))
    return np.asarray(arr, dtype=np.float64)


class ClibJetCorrector:
    """Compound L1L2L3Res correction, in place of ``FactorizedJetCorrector``."""

    def __init__(self, correction: correctionlib.highlevel.CompoundCorrection):
        self._corr = correction
        self.signature = [inp.name for inp in correction.inputs]

    def getCorrection(self, **kwargs):
        args = [_to_numpy(kwargs[name]) for name in self.signature]
        return ak.Array(self._corr.evaluate(*args).astype(np.float32))


class ClibJetCorrectionUncertainty:
    """Total JES uncertainty, in place of ``JetCorrectionUncertainty``. It is exposed as the level
    ``jes`` so that the shifted jets keep the ``JES_jes`` field of the text-file path."""

    def __init__(self, correction: correctionlib.highlevel.Correction):
        self._corr = correction
        self.signature = [inp.name for inp in correction.inputs]
        self.levels = ["jes"]

    def getUncertainty(self, **kwargs):
        unc = self._corr.evaluate(*[_to_numpy(kwargs[name]) for name in self.signature])
        return zip(self.levels, [ak.Array(np.stack([1 + unc, 1 - unc], axis=1).astype(np.float32))])


class ClibJetResolution:
    """Relative pT resolution, in place of ``JetResolution``."""

    def __init__(self, correction: correctionlib.highlevel.Correction):
        self._corr = correction
        self.signature = [inp.name for inp in correction.inputs]

    def getResolution(self, **kwargs):
        args = [_to_numpy(kwargs[name]) for name in self.signature]
        return ak.Array(self._corr.evaluate(*args).astype(np.float32))


class ClibJetResolutionScaleFactor:
    """JER scale factor, in place of ``JetResolutionScaleFactor``. The JRV2 files split the nominal
    scale factor and its uncertainty: [nominal, up, down] = [SF, SF + unc, SF - unc]."""

    def __init__(
        self, sf: correctionlib.highlevel.Correction, unc: correctionlib.highlevel.Correction
    ):
        self._sf = sf
        self._unc = unc
        self.signature = [inp.name for inp in sf.inputs]
        if [inp.name for inp in unc.inputs] != self.signature:
            raise ValueError(f"{sf.name} and {unc.name} have different inputs")

    def getScaleFactor(self, **kwargs):
        args = [_to_numpy(kwargs[name]) for name in self.signature]
        sf = self._sf.evaluate(*args)
        unc = self._unc.evaluate(*args)
        return ak.Array(np.stack([sf, sf + unc, sf - unc], axis=1).astype(np.float32))


class ClibJECStack:
    """The attributes of ``coffea.jetmet_tools.JECStack`` that ``CorrectedJetsFactory`` reads."""

    def __init__(self, jec, junc=None, jer=None, jersf=None):
        self.jec = jec
        self.junc = junc
        self.jer = jer
        self.jersf = jersf
        self.blank_name_map = {}


@lru_cache(maxsize=None)
def _correction_set(path: str) -> correctionlib.CorrectionSet:
    return correctionlib.CorrectionSet.from_file(path)


@lru_cache(maxsize=None)
def clib_jet_factory(
    path: str, jec_tag: str, jer_tag: str, algo: str, isData: bool
) -> CorrectedJetsFactory:
    """``CorrectedJetsFactory`` for one jet algorithm (e.g. ``AK8PFPuppi``) from the JERC file
    ``path``: data get the nominal JEC only (run-dependent residuals, so the jets need a ``run``
    field), MC the nominal JEC, the JER smearing and the total JES uncertainty."""
    cset = _correction_set(path)
    if isData:
        stack = ClibJECStack(ClibJetCorrector(cset.compound[f"{jec_tag}_DATA_L1L2L3Res_{algo}"]))
    else:
        stack = ClibJECStack(
            ClibJetCorrector(cset.compound[f"{jec_tag}_MC_L1L2L3Res_{algo}"]),
            ClibJetCorrectionUncertainty(cset[f"{jec_tag}_MC_Total_{algo}"]),
            ClibJetResolution(cset[f"{jer_tag}_MC_PtResolution_{algo}"]),
            ClibJetResolutionScaleFactor(
                cset[f"{jer_tag}_MC_ScaleFactor_{algo}"], cset[f"{jer_tag}_MC_SFUncertainty_{algo}"]
            ),
        )
    return CorrectedJetsFactory(dict(jec_name_map), stack)
