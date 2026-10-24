"""
Collection of utilities for corrections and systematics in processors.

Loosely based on https://github.com/jennetd/hbb-coffea/blob/master/boostedhiggs/corrections.py

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/

Authors: Raghav Kansal, Cristina Suarez, Zichun Hao
"""

from __future__ import annotations

import gzip
import pathlib
import pickle

import awkward as ak
import cachetools
import correctionlib
import numpy as np
import uproot
from coffea.analysis_tools import Weights
from coffea.jetmet_tools import CorrectedMETFactory
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, JetArray

from .jec_clib import clib_jet_factory
from .utils import pad_val

ak.behavior.update(vector.behavior)
package_path = str(pathlib.Path(__file__).parent.parent.resolve())

# Important Run3 start of Run
FirstRun_2022C = 355794
FirstRun_2022D = 357487
LastRun_2022D = 359021
FirstRun_2022E = 359022
LastRun_2022F = 362180

"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "fatjet_jec": ["JME", "fatJet_jerc.json.gz"],
    "jet_jec": ["JME", "jet_jerc.json.gz"],
    "jetveto": ["JME", "jetvetomaps.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
}

"""
2024 and later: the jsonpog-integration rsync above is frozen since 2025-09; the official files are
in the CMS analysis-corrections (CAT) area. Its dated folders never change ("latest" does), so they
are pinned here. 2025 data are corrected for use with the Summer24 MC (the 2025 MC is the 2024 MC):
same MC truth and JES uncertainty as 2024, residuals and JER scale factors derived on 2025 data.
"""
cat_correction_path = "/cvmfs/cms-griddata.cern.ch/cat/metadata"
cat_corrections = {
    "2024": {
        "jme": f"{cat_correction_path}/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-07-16",
        "jec": "Summer24Prompt24_V5",
        "jer": "Summer24Prompt24_JRV2",
        "jetveto": "Summer24Prompt24_RunBCDEFGHI_V1",
        "pileup_json": f"{cat_correction_path}/LUM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-04-15/puWeights_CDEFGHI.json.gz",
        "pileup": "Collisions24_CDEFGHI_goldenJSON",
    },
    "2025": {
        "jme": f"{cat_correction_path}/JME/Run3-25Prompt-Summer24-NanoAODv15/2026-07-16",
        "jec": "Summer24Prompt25_V3",
        "jer": "Summer24Prompt25_JRV2",
        "jetveto": "Summer24Prompt25_RunCDEFG_V1",
        "pileup_json": f"{cat_correction_path}/LUM/Run3-25Prompt-Summer24-NanoAODv15/2026-06-05/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz",
        "pileup": "Collisions25_goldenJSON",
    },
}

"""
JECs and JER of every Run 3 year from the correctionlib files of the CAT area (processors/jec_clib.py),
no text files: pinned dated folders with the latest versions (identical to "latest" on 2026-10-20).
The 2022-2023 data residuals of these versions are run dependent, as the 2024 and 2025 ones.
"""
cat_jerc = {
    "2022": {
        "jme": f"{cat_correction_path}/JME/Run3-22CDSep23-Summer22-NanoAODv12/2026-06-05",
        "jec": "Summer22_22Sep2023_V4",
        "jer": "Summer22_22Sep2023_JRV2",
    },
    "2022EE": {
        "jme": f"{cat_correction_path}/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2026-06-05",
        "jec": "Summer22EE_22Sep2023_V4",
        "jer": "Summer22EE_22Sep2023_JRV2",
    },
    "2023": {
        "jme": f"{cat_correction_path}/JME/Run3-23CSep23-Summer23-NanoAODv12/2026-07-15",
        "jec": "Summer23Prompt23_V4",
        "jer": "Summer23Prompt23_RunCv1234_JRV3",
    },
    "2023BPix": {
        "jme": f"{cat_correction_path}/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2026-07-15",
        "jec": "Summer23BPixPrompt23_V4",
        "jer": "Summer23BPixPrompt23_RunD_JRV3",
    },
    **{y: {k: cat[k] for k in ("jme", "jec", "jer")} for y, cat in cat_corrections.items()},
}

# input names of CorrectedMETFactory -> MET and jet fields (as corrections/build_jec.py)
met_name_map = {
    "JetPt": "pt",
    "JetPhi": "phi",
    "ptRaw": "pt_raw",
    "METpt": "pt",
    "METphi": "phi",
    "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
    "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
}


def get_UL_year(year: str) -> str:
    return f"{year}_UL"


def get_pog_json(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = get_UL_year(year) if year == "2018" else year
    if "2022" in year or "2023" in year or "2024" in year or "2025" in year:
        year = {
            "2022": "2022_Summer22",
            "2022EE": "2022_Summer22EE",
            "2023": "2023_Summer23",
            "2023BPix": "2023_Summer23BPix",
            # "2024": "2024_Winter24",
            "2024": "2024_Summer24",
            "2025": "2025_Summer25",
        }[year]
    return f"{pog_correction_path}/POG/{pog_json[0]}/{year}/{pog_json[1]}"


def get_pileup_weights(year: str, n_true_int: np.ndarray) -> dict[str, np.ndarray]:
    """Official pileup weights (nominal, up, down) of ``year`` (2024 and later) at Pileup_nTrueInt,
    clipped to [0, 10] to avoid large weights."""
    corr = correctionlib.CorrectionSet.from_file(cat_corrections[year]["pileup_json"])[
        cat_corrections[year]["pileup"]
    ]
    return {
        var: np.clip(corr.evaluate(n_true_int, var), 0, 10) for var in ["nominal", "up", "down"]
    }


def add_pileup_weight(
    weights: Weights,
    year: str,
    nPU: np.ndarray,
    dataset: str | None = None,
    n_true_int: np.ndarray | None = None,
):
    # clip nPU from 0 to 100
    nPU = np.clip(nPU, 0, 99)
    # print(list(nPU))

    if "Pu60" in dataset or "Pu70" in dataset:
        # pileup profile from data
        path_pileup = package_path + "/corrections/data/MyDataPileupHistogram2022FG.root"
        pileup_profile = uproot.open(path_pileup)["pileup"]
        pileup_profile = pileup_profile.to_numpy()[0]
        # normalise
        pileup_profile /= pileup_profile.sum()

        # https://indico.cern.ch/event/695872/contributions/2877123/attachments/1593469/2522749/pileup_ppd_feb_2018.pdf
        # pileup profile from MC
        pu_name = "Pu60" if "Pu60" in dataset else "Pu70"
        path_pileup_dataset = package_path + f"/corrections/data/pileup/{pu_name}.npy"
        pileup_MC = np.load(path_pileup_dataset)

        # avoid division by 0 (?)
        pileup_MC[pileup_MC == 0.0] = 1
        pileup_correction = pileup_profile / pileup_MC
        # remove large MC reweighting factors to prevent artifacts
        pileup_correction[pileup_correction > 10] = 10
        sf = pileup_correction[nPU]
        # no uncertainties
        weights.add("pileup", sf)

    elif year in cat_corrections:
        # official weights, a function of the true number of interactions (Pileup_nTrueInt)
        if n_true_int is None:
            raise ValueError(f"The {year} pileup weights need Pileup_nTrueInt")
        values = get_pileup_weights(year, n_true_int)
        weights.add("pileup", values["nominal"], values["up"], values["down"])

    else:
        # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3
        # NOTE: these weights are defined in the true number of interactions but are evaluated at
        # nPU; kept as is so that new 2022-2023 skims match the existing ones
        values = {}

        cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))
        corr = {
            "2018": "Collisions18_UltraLegacy_goldenJSON",
            "2022": "Collisions2022_355100_357900_eraBCD_GoldenJson",
            "2022EE": "Collisions2022_359022_362760_eraEFG_GoldenJson",
            "2023": "Collisions2023_366403_369802_eraBC_GoldenJson",
            "2023BPix": "Collisions2023_369803_370790_eraD_GoldenJson",
        }[year]
        # evaluate and clip up to 10 to avoid large weights
        values["nominal"] = np.clip(cset[corr].evaluate(nPU, "nominal"), 0, 10)
        values["up"] = np.clip(cset[corr].evaluate(nPU, "up"), 0, 10)
        values["down"] = np.clip(cset[corr].evaluate(nPU, "down"), 0, 10)

        weights.add("pileup", values["nominal"], values["up"], values["down"])


def get_vpt(genpart, check_offshell=False):
    """Only the leptonic samples have no resonance in the decay tree, and only
    when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
    """
    boson = ak.firsts(
        genpart[
            ((genpart.pdgId == 23) | (abs(genpart.pdgId) == 24))
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
    )
    if check_offshell:
        offshell = genpart[
            genpart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(genpart.pdgId) >= 11)
            & (abs(genpart.pdgId) <= 16)
        ].sum()
        return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
    return np.array(ak.fill_none(boson.pt, 0.0))


def add_ps_weight(weights, ps_weights):
    """
    Parton Shower Weights (FSR and ISR)
    """

    nweights = len(weights.weight())
    nom = np.ones(nweights)

    up_isr = np.ones(nweights)
    down_isr = np.ones(nweights)
    up_fsr = np.ones(nweights)
    down_fsr = np.ones(nweights)

    if len(ps_weights[0]) == 4:
        up_isr = ps_weights[:, 0]  # ISR=2, FSR=1
        down_isr = ps_weights[:, 2]  # ISR=0.5, FSR=1

        up_fsr = ps_weights[:, 1]  # ISR=1, FSR=2
        down_fsr = ps_weights[:, 3]  # ISR=1, FSR=0.5

    elif len(ps_weights[0]) > 1:
        print("PS weight vector has length ", len(ps_weights[0]))

    weights.add("ISRPartonShower", nom, up_isr, down_isr)
    weights.add("FSRPartonShower", nom, up_fsr, down_fsr)


def get_pdf_weights(events):
    """
    For the PDF acceptance uncertainty:
        - store 103 variations. 0-100 PDF values
        - The last two values: alpha_s variations.
        - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
        e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
        and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488

    Some references:
    Scale/PDF weights in MC https://twiki.cern.ch/twiki/bin/view/CMS/HowToPDF
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#PDF
    """
    return events.LHEPdfWeight.to_numpy()


def get_scale_weights(events):
    """
    QCD Scale variations, best explanation I found is here:
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio

    TLDR: we want to vary the renormalization and factorization scales by a factor of 0.5 and 2,
    and then take the envelope of the variations on our final observation as the up/down uncertainties.

    Importantly, we need to keep track of the normalization for each variation,
    so that this uncertainty takes into account the acceptance effects of our selections.

    LHE scale variation weights (w_var / w_nominal) (from https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv9/2018UL/doc_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1.html#LHEScaleWeight)
    [0] is renscfact=0.5d0 facscfact=0.5d0 ; <=
    [1] is renscfact=0.5d0 facscfact=1d0 ; <=
    [2] is renscfact=0.5d0 facscfact=2d0 ;
    [3] is renscfact=1d0 facscfact=0.5d0 ; <=
    [4] is renscfact=1d0 facscfact=1d0 ;
    [5] is renscfact=1d0 facscfact=2d0 ; <=
    [6] is renscfact=2d0 facscfact=0.5d0 ;
    [7] is renscfact=2d0 facscfact=1d0 ; <=
    [8] is renscfact=2d0 facscfact=2d0 ; <=

    See also https://git.rwth-aachen.de/3pia/cms_analyses/common/-/blob/11e0c5225416a580d27718997a11dc3f1ec1e8d1/processor/generator.py#L93 for an example.
    """
    # Some files (e.g. certain PFNano skims) don't store the LHEScaleWeight branch at all;
    # treat that the same as "no scale weights" (return None, handled by the caller) so the
    # job doesn't crash with AttributeError.
    if "LHEScaleWeight" not in events.fields:
        return None
    if len(events[0].LHEScaleWeight) > 0:
        if len(events[0].LHEScaleWeight) == 9:
            variations = events.LHEScaleWeight[:, [0, 1, 3, 5, 7, 8]].to_numpy()
            nominal = events.LHEScaleWeight[:, 4].to_numpy()[:, np.newaxis]
            variations /= nominal
        else:
            variations = events.LHEScaleWeight[:, [0, 1, 3, 4, 6, 7]].to_numpy()
        return np.clip(variations, 0.0, 4.0)
    else:
        return None


class JECs:
    def __init__(self, year):
        self.jet_factory = {}
        self.met_factory = None

        if year in cat_jerc:
            # official correctionlib JECs and JER (run-dependent data residuals)
            cat = cat_jerc[year]
            for jet_type, (jerc_file, algo) in {
                "ak4": ("jet_jerc", "AK4PFPuppi"),
                "ak8": ("fatJet_jerc", "AK8PFPuppi"),
            }.items():
                path = f"{cat['jme']}/{jerc_file}.json.gz"
                self.jet_factory[jet_type] = {
                    f"{year}mc": clib_jet_factory(path, cat["jec"], cat["jer"], algo, isData=False),
                    year: clib_jet_factory(path, cat["jec"], cat["jer"], algo, isData=True),
                }
            self.met_factory = CorrectedMETFactory(met_name_map)
        elif year in ["2016", "2016APV", "2017", "2018"]:
            jec_compiled = package_path + "/corrections/jec_compiled_run2.pkl.gz"
            with gzip.open(jec_compiled, "rb") as filehandler:
                jmestuff = pickle.load(filehandler)

            self.jet_factory["ak4"] = jmestuff["jet_factory"]
            self.jet_factory["ak8"] = jmestuff["fatjet_factory"]
            self.met_factory = jmestuff["met_factory"]

    def _add_jec_variables(
        self, jets: JetArray, event_rho: ak.Array, isData: bool, run: ak.Array | None = None
    ) -> JetArray:
        """add variables needed for JECs"""
        jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
        jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
        jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
        if not isData:
            # gen pT needed for smearing
            jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        if run is not None:
            # run number for the run-dependent data residuals
            jets["run"] = ak.broadcast_arrays(run, jets.pt)[0]
        return jets

    def get_jec_jets(
        self,
        events: NanoEventsArray,
        jets: FatJetArray,
        year: str,
        isData: bool = False,
        jecs: dict[str, str] | None = None,
        fatjets: bool = True,
        applyData: bool = False,
        dataset: str | None = None,
        nano_version: str = "v12",
    ) -> FatJetArray:
        """
        If ``jecs`` is not None, returns the shifted values of variables are affected by JECs.
        """

        rho = (
            events.Rho.fixedGridRhoFastjetAll
            if "Rho" in events.fields
            else events.fixedGridRhoFastjetAll
        )
        run = events.run if (isData and year in cat_jerc) else None
        jets = self._add_jec_variables(jets, rho, isData, run)

        apply_jecs = ak.any(jets.pt) if (applyData or not isData) else False
        if (
            ("v12" not in nano_version)
            and ("v14" not in nano_version)
            and ("v15" not in nano_version)
        ):
            apply_jecs = False
        if not apply_jecs:
            return jets, None

        jec_vars = ["pt"]  # variables we are saving that are affected by JECs
        jet_factory_str = "ak4"
        if fatjets:
            jet_factory_str = "ak8"

        if self.jet_factory[jet_factory_str] is None:
            print("No factory available")
            return jets, None

        jec_cache = cachetools.Cache(np.inf)

        if isData:
            if year in cat_jerc:
                # run-dependent residuals: one correction for all eras of the year
                corr_key = year
            else:
                print(dataset, year)
                print("warning, no valid dataset, JECs won't be applied to data")
                applyData = False
        else:
            corr_key = f"{year}mc"

        # fatjet_factory.build gives an error if there are no jets in event
        if apply_jecs:
            jets = self.jet_factory[jet_factory_str][corr_key].build(jets, jec_cache)

        # return only jets if no variations are given
        if jecs is None or isData:
            return jets, None

        jec_shifted_vars = {}
        for jec_var in jec_vars:
            tdict = {"": jets[jec_var]}
            if apply_jecs:
                for key, shift in jecs.items():
                    for var in ["up", "down"]:
                        if shift in ak.fields(jets):
                            tdict[f"{key}_{var}"] = jets[shift][var][jec_var]
            jec_shifted_vars[jec_var] = tdict

        return jets, jec_shifted_vars


def get_jmsr(
    fatjets: FatJetArray,
    num_jets: int,
    jmsr_vars: list[str],
    jms_values: dict,
    jmr_values: dict,
    isData: bool = False,
    seed: int = 42,
) -> dict:
    """Calculates post JMS/R masses and shifts"""

    jmsr_shifted_vars = {}

    for mkey in jmsr_vars:
        tdict = {}

        mass = pad_val(fatjets[mkey], num_jets, axis=1)
        jms = jms_values[mkey]
        jmr = jmr_values[mkey]

        if isData:
            tdict[""] = mass
        else:
            rng = np.random.default_rng(seed)
            smearing = rng.normal(size=mass.shape)
            # scale to JMR nom, down, up (minimum at 0)
            jmr_nom, jmr_down, jmr_up = ((smearing * max(jmr[i] - 1, 0) + 1) for i in range(3))
            jms_nom, jms_down, jms_up = jms

            mass_jms = mass * jms_nom
            mass_jmr = mass * jmr_nom

            tdict[""] = mass * jms_nom * jmr_nom
            tdict["JMS_down"] = mass_jmr * jms_down
            tdict["JMS_up"] = mass_jmr * jms_up
            tdict["JMR_down"] = mass_jms * jmr_down
            tdict["JMR_up"] = mass_jms * jmr_up

        jmsr_shifted_vars[mkey] = tdict

    return jmsr_shifted_vars


# Jet Veto Maps
# the JERC group recommends ALL analyses use these maps, as the JECs are derived excluding these zones.
# apply to both Data and MC
# https://cms-talk.web.cern.ch/t/jet-veto-maps-for-run3-data/18444?u=anmalara
# https://cms-talk.web.cern.ch/t/jes-for-2022-re-reco-cde-and-prompt-fg/32873
def get_jetveto_event(jets: JetArray, year: str):
    """
    Get event selection that rejects events with jets in the veto map
    """
    # correction: Non-zero value for (eta, phi) indicates that the region is vetoed
    if year in cat_corrections:
        cset = correctionlib.CorrectionSet.from_file(
            f"{cat_corrections[year]['jme']}/jetvetomaps.json.gz"
        )
    else:
        cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetveto", year))
    j, nj = ak.flatten(jets), ak.num(jets)

    def get_veto(j, nj, csetstr):
        j_phi = np.clip(np.array(j.phi), -3.1415, 3.1415)
        j_eta = np.clip(np.array(j.eta), -4.7, 4.7)
        veto = cset[csetstr].evaluate("jetvetomap", j_eta, j_phi)
        return ak.unflatten(veto, nj)

    corr_str = {
        "2022": "Summer22_23Sep2023_RunCD_V1",
        "2022EE": "Summer22EE_23Sep2023_RunEFG_V1",
        "2023": "Summer23Prompt23_RunC_V1",
        "2023BPix": "Summer23BPixPrompt23_RunD_V1",
        **{y: cat["jetveto"] for y, cat in cat_corrections.items()},
    }[year]

    jet_veto = get_veto(j, nj, corr_str) > 0

    event_sel = ~(ak.any((jets.pt > 15) & jet_veto, axis=1))
    return event_sel
