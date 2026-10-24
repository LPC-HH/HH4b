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

# correctionlib input names -> jet fields; as jec_name_map of corrections/archive/build_jec.py,
# plus the run number for the run-dependent data residuals
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


# TODO: use coffea's CorrectedMETFactory (Type-1 mode) once on coffea >= 2026.9.0 or a 0.7.x
# release after 0.7.31. It gives the same nominal and, with the total JES uncertainty passed as
# jes_uncertainties, the same JES variations; its JER variation of the unsmeared MET differs (it
# scales each jet by the JER-varied over the nominal smeared pT, which blows up for jets smeared
# to ~0 GeV).
def type1_puppi_met(
    events, path: str, jec_tag: str, isData: bool, jer_smear_factor: ak.Array | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{"": (pt, phi)} of the Type-1 PUPPI MET from RawPuppiMET with the AK4 PUPPI JEC ``jec_tag``
    of ``path``, as in CMSSW: every Jet and CorrT1METJet with muon-subtracted corrected pT > 15 GeV
    and EM fraction < 0.9 shifts it by -(f_L1L2L3Res - f_L1) x muon-subtracted raw pT; no smearing.
    Adapted from coffea's Type-1 CorrectedMETFactory (scikit-hep/coffea#1537); same output as
    DAZSLE/hbb-run3#74 (correct_met).

    With ``jer_smear_factor`` (MC: the nominal JER smearing factor of each events.Jet), also the
    variations of the unsmeared Type-1 MET as in CMSJMECalculators
    (gitlab.cern.ch/cms-analysis/general/CMSJMECalculators, Type1METVariationsCalculator with
    isT1SmearedMET false; the MET_T1 variations of nanoAOD-tools' jetmetUncertainties are the
    same). They move the same jets, at pT_L1L2L3, the corrected pT of their muon-subtracted part
    plus the muon:
    - JES_up/down: each jet at pT_L1L2L3 x (1 +- delta), delta the total JES uncertainty at
      (eta, pT_L1L2L3);
    - JER_up = JER_down: the Jets at pT_L1L2L3 x their smearing factor, i.e. the smeared Type-1 MET
      (the CorrT1METJets unsmeared, as in coffea's Type-1 MET without their smearing factors);
    - UES_up/down: plus/minus the unclustered-energy shift of NanoAOD PuppiMET.
    """
    cset = _correction_set(path)
    level = "DATA" if isData else "MC"
    corr_full = cset.compound[f"{jec_tag}_{level}_L1L2L3Res_AK4PFPuppi"]
    corr_l1 = cset[f"{jec_tag}_{level}_L1FastJet_AK4PFPuppi"]
    nev = len(events)
    rho = _to_numpy(events.Rho.fixedGridRhoFastjetAll)
    run = _to_numpy(events.run)
    raw_met_pt = _to_numpy(events.RawPuppiMET.pt)
    raw_met_phi = _to_numpy(events.RawPuppiMET.phi)
    met_x = raw_met_pt * np.cos(raw_met_phi)
    met_y = raw_met_pt * np.sin(raw_met_phi)
    jets, low_pt_jets = events.Jet, events.CorrT1METJet
    type1_jets = []  # per collection: what the variations need
    for coll, raw_pt, em_fraction, smear_factor in [
        (jets, (1 - jets.rawFactor) * jets.pt, jets.chEmEF + jets.neEmEF, jer_smear_factor),
        (low_pt_jets, low_pt_jets.rawPt, low_pt_jets.EmEF, None),
    ]:
        idx = np.repeat(np.arange(nev), ak.to_numpy(ak.num(raw_pt)))
        raw = _to_numpy(ak.flatten(raw_pt))
        inputs = {
            "JetA": _to_numpy(ak.flatten(coll.area)),
            "JetEta": _to_numpy(ak.flatten(coll.eta)),
            "JetPhi": _to_numpy(ak.flatten(coll.phi)),
            "JetPt": raw,
            "Rho": rho[idx],
            "run": run[idx],
        }
        f = corr_full.evaluate(*[inputs[inp.name] for inp in corr_full.inputs])
        f_l1 = corr_l1.evaluate(*[inputs[inp.name] for inp in corr_l1.inputs])
        raw_nomu = raw * (1 - _to_numpy(ak.flatten(coll.muonSubtrFactor)))
        phi = inputs["JetPhi"]
        if "muonSubtrDeltaPhi" in coll.fields:
            phi = phi + _to_numpy(ak.flatten(coll.muonSubtrDeltaPhi))
        use = (f * raw_nomu > 15) & (_to_numpy(ak.flatten(em_fraction)) < 0.9)
        shift = np.where(use, (f - f_l1) * raw_nomu, 0.0)
        met_x -= np.bincount(idx, weights=shift * np.cos(phi), minlength=nev)
        met_y -= np.bincount(idx, weights=shift * np.sin(phi), minlength=nev)
        pt_l1l2l3 = f * raw_nomu + (raw - raw_nomu)
        type1_jets.append((idx, use, pt_l1l2l3, inputs["JetEta"], phi, smear_factor))

    def polar(x, y):
        return np.hypot(x, y), np.arctan2(y, x)

    out = {"": polar(met_x, met_y)}
    if jer_smear_factor is None:
        return out

    def vector_sum(idx, values, phi):
        return (
            np.bincount(idx, weights=values * np.cos(phi), minlength=nev),
            np.bincount(idx, weights=values * np.sin(phi), minlength=nev),
        )

    # sums over the jets of their pT changes: pT_L1L2L3 x delta (JES), pT_L1L2L3 x (s - 1) (JER)
    jes_unc = cset[f"{jec_tag}_MC_Total_AK4PFPuppi"]
    jes_x = jes_y = jer_x = jer_y = 0.0
    for idx, use, pt_l1l2l3, eta, phi, smear_factor in type1_jets:
        jes_inputs = {"JetEta": eta, "JetPt": pt_l1l2l3}
        delta = jes_unc.evaluate(*[jes_inputs[inp.name] for inp in jes_unc.inputs])
        dx, dy = vector_sum(idx, np.where(use, pt_l1l2l3 * delta, 0.0), phi)
        jes_x, jes_y = jes_x + dx, jes_y + dy
        if smear_factor is not None:
            smear = _to_numpy(ak.flatten(smear_factor))
            dx, dy = vector_sum(idx, np.where(use, pt_l1l2l3 * (smear - 1), 0.0), phi)
            jer_x, jer_y = jer_x + dx, jer_y + dy
    out["JES_up"] = polar(met_x - jes_x, met_y - jes_y)
    out["JES_down"] = polar(met_x + jes_x, met_y + jes_y)
    out["JER_up"] = out["JER_down"] = polar(met_x - jer_x, met_y - jer_y)
    puppi = events.PuppiMET
    ues_x = _to_numpy(puppi.ptUnclusteredUp) * np.cos(_to_numpy(puppi.phiUnclusteredUp))
    ues_y = _to_numpy(puppi.ptUnclusteredUp) * np.sin(_to_numpy(puppi.phiUnclusteredUp))
    ues_x -= _to_numpy(puppi.pt) * np.cos(_to_numpy(puppi.phi))
    ues_y -= _to_numpy(puppi.pt) * np.sin(_to_numpy(puppi.phi))
    out["UES_up"] = polar(met_x + ues_x, met_y + ues_y)
    out["UES_down"] = polar(met_x - ues_x, met_y - ues_y)
    return out
