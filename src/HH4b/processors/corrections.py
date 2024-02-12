"""
Collection of utilities for corrections and systematics in processors.

Loosely based on https://github.com/jennetd/hbb-coffea/blob/master/boostedhiggs/corrections.py

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/

Authors: Raghav Kansal, Cristina Suarez
"""
from __future__ import annotations

import gzip
import pathlib
import pickle
import random
from pathlib import Path

import awkward as ak
import correctionlib
import numpy as np
import uproot
from coffea.analysis_tools import Weights
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, JetArray

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

# Jet mass scale and Jet mass resolution
# FIXME: Using placeholder Run 2 values !!
# nominal, down, up
jmsr_vars = ["msoftdrop", "particleNet_mass"]

jmsValues = {}
jmrValues = {}

jmrValues["msoftdrop"] = {
    "2016": [1.00, 1.0, 1.09],
    "2017": [1.03, 1.00, 1.07],
    "2018": [1.065, 1.031, 1.099],
    "2022": [1.0, 1.0, 1.0],
    "2022EE": [1.0, 1.0, 1.0],
}

jmsValues["msoftdrop"] = {
    "2016": [1.00, 0.9906, 1.0094],
    "2017": [1.0016, 0.978, 0.986],
    "2018": [0.997, 0.993, 1.001],
    "2022": [1.0, 1.0, 1.0],
    "2022EE": [1.0, 1.0, 1.0],
}

jmrValues["particleNet_mass"] = {
    "2016": [1.028, 1.007, 1.063],
    "2017": [1.026, 1.009, 1.059],
    "2018": [1.031, 1.006, 1.075],
    "2022": [1.0, 1.0, 1.0],
    "2022EE": [1.0, 1.0, 1.0],
}

jmsValues["particleNet_mass"] = {
    "2016": [1.00, 0.998, 1.002],
    "2017": [1.002, 0.996, 1.008],
    "2018": [0.994, 0.993, 1.001],
    "2022": [1.0, 1.0, 1.0],
    "2022EE": [1.0, 1.0, 1.0],
}


def get_Prompt_year(year: str) -> str:
    return f"{year}_Prompt"


def get_UL_year(year: str) -> str:
    return f"{year}_UL"


def get_pog_json(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    # TODO: fix prompt when switching to re-reco 23Sep datasets
    year = get_Prompt_year(year) if year == "2022" else year
    year = get_UL_year(year) if year == "2018" else year
    return f"{pog_correction_path}/POG/{pog_json[0]}/{year}/{pog_json[1]}"


def add_pileup_weight(weights: Weights, year: str, nPU: np.ndarray, dataset: str | None = None):
    # clip nPU from 0 to 100
    nPU = np.clip(nPU, 0, 99)
    # print(list(nPU))

    if year == "2018":
        values = {}

        cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))
        y = year
        year_to_corr = {
            "2018": "Collisions18_UltraLegacy_goldenJSON",
        }
        # evaluate and clip up to 10 to avoid large weights
        values["nominal"] = np.clip(cset[year_to_corr[y]].evaluate(nPU, "nominal"), 0, 10)
        values["up"] = np.clip(cset[year_to_corr[y]].evaluate(nPU, "up"), 0, 10)
        values["down"] = np.clip(cset[year_to_corr[y]].evaluate(nPU, "down"), 0, 10)

        weights.add("pileup", values["nominal"], values["up"], values["down"])

    # TODO: Switch to official recommendation when and if any
    else:
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

            print("Data profile  ", np.round(pileup_profile, 3))
            print("MC profile ", np.round(pileup_MC, 3))

            pileup_correction = pileup_profile / pileup_MC
            print("correction ", np.round(pileup_correction, 2))

            # remove large MC reweighting factors to prevent artifacts
            pileup_correction[pileup_correction > 10] = 10

            print("correction ", np.round(pileup_correction, 2))

            sf = pileup_correction[nPU]
            print("sf ", sf)

            # no uncertainties
            weights.add("pileup", sf)

        else:
            y = year.replace("EE", "")
            cset = correctionlib.CorrectionSet.from_file(
                package_path + f"/corrections/{y}_puWeights.json.gz"
            )
            name = f"Collisions_{y}_PromptReco_goldenJSON"
            nominal = np.clip(cset[name].evaluate(nPU, "nominal"), 0, 10)
            weights.add("pileup", nominal)


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

    # TODO: do we need to update sumgenweights?
    # e.g. as in https://git.rwth-aachen.de/3pia/cms_analyses/common/-/blob/11e0c5225416a580d27718997a11dc3f1ec1e8d1/processor/generator.py#L74


def add_pdf_weight(weights, pdf_weights):
    """
    LHEPDF Weights
    """
    nweights = len(weights.weight())
    nom = np.ones(nweights)
    # up = np.ones(nweights)
    # down = np.ones(nweights)

    # NNPDF31_nnlo_hessian_pdfas
    # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info

    # Hessian PDF weights
    # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
    arg = pdf_weights[:, 1:-2] - np.ones((nweights, 100))
    summed = ak.sum(np.square(arg), axis=1)
    pdf_unc = np.sqrt((1.0 / 99.0) * summed)
    # weights.add("PDF", nom, pdf_unc + nom)

    # alpha_S weights
    # Eq. 27 of same ref
    as_unc = 0.5 * (pdf_weights[:, 102] - pdf_weights[:, 101])
    # weights.add('alphaS', nom, as_unc + nom)

    # PDF + alpha_S weights
    # Eq. 28 of same ref
    pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
    weights.add("PDFalphaS", nom, pdfas_unc + nom)


def add_scalevar_7pt(weights, var_weights):
    """
    QCD Scale variations:
    7 point is where the renorm. and factorization scale are varied separately
    docstring:
    LHE scale variation weights (w_var / w_nominal);
    [0] is renscfact=0.5d0 facscfact=0.5d0 ;
    [1] is renscfact=0.5d0 facscfact=1d0 ; <=
    [2] is renscfact=0.5d0 facscfact=2d0 ;
    [3] is renscfact=1d0 facscfact=0.5d0 ; <=
    [4] is renscfact=1d0 facscfact=1d0 ;
    [5] is renscfact=1d0 facscfact=2d0 ; <=
    [6] is renscfact=2d0 facscfact=0.5d0 ;
    [7] is renscfact=2d0 facscfact=1d0 ; <=
    [8] is renscfact=2d0 facscfact=2d0 ; <=
    """
    nweights = len(weights.weight())

    nom = np.ones(nweights)
    up = np.ones(nweights)
    down = np.ones(nweights)

    if len(var_weights) > 0:
        if len(var_weights[0]) == 9:
            up = np.maximum.reduce(
                [
                    var_weights[:, 0],
                    var_weights[:, 1],
                    var_weights[:, 3],
                    var_weights[:, 5],
                    var_weights[:, 7],
                    var_weights[:, 8],
                ]
            )
            down = np.minimum.reduce(
                [
                    var_weights[:, 0],
                    var_weights[:, 1],
                    var_weights[:, 3],
                    var_weights[:, 5],
                    var_weights[:, 7],
                    var_weights[:, 8],
                ]
            )
        elif len(var_weights[0]) > 1:
            print("Scale variation vector has length ", len(var_weights[0]))
    # NOTE: I think we should take the envelope of these weights w.r.t to [4]
    weights.add("QCDscale7pt", nom, up, down)
    weights.add("QCDscale4", var_weights[:, 4])


def add_scalevar_3pt(weights, var_weights):
    nweights = len(weights.weight())

    nom = np.ones(nweights)
    up = np.ones(nweights)
    down = np.ones(nweights)

    if len(var_weights) > 0:
        if len(var_weights[0]) == 9:
            up = np.maximum(var_weights[:, 0], var_weights[:, 8])
            down = np.minimum(var_weights[:, 0], var_weights[:, 8])
        elif len(var_weights[0]) > 1:
            print("Scale variation vector has length ", len(var_weights[0]))

    weights.add("QCDscale3pt", nom, up, down)


class JECs:
    def __init__(self, year):
        if year in ["2022", "2022EE"]:
            jec_compiled = package_path + "/corrections/jec_compiled.pkl.gz"
        elif year in ["2016", "2016APV", "2017", "2018"]:
            jec_compiled = package_path + "/corrections/jec_compiled_run2.pkl.gz"
        else:
            jec_compiled = None

        self.jet_factory = {}
        self.met_factory = None

        if jec_compiled is not None:
            with gzip.open(jec_compiled, "rb") as filehandler:
                jmestuff = pickle.load(filehandler)

            self.jet_factory["ak4"] = jmestuff["jet_factory"]
            self.jet_factory["ak8"] = jmestuff["fatjet_factory"]
            self.met_factory = jmestuff["met_factory"]

    def _add_jec_variables(self, jets: JetArray, event_rho: ak.Array, isData: bool) -> JetArray:
        """add variables needed for JECs"""
        jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
        jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
        jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
        if not isData:
            # gen pT needed for smearing
            jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
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
        # RunC,D,E are re-reco, should wait for new jecs
        datasets_no_jecs = [
            "Run2022C_single",
            "Run2022C",
            "Run2022D",
            "Run2022E",
        ]
        for dset in datasets_no_jecs:
            if dataset in dset and nano_version == "v12":
                return jets, None

        if year == "2023":
            # no corrections available yet
            return jets, None

        jec_vars = ["pt"]  # variables we are saving that are affected by JECs
        jet_factory_str = "ak4"
        if fatjets:
            jet_factory_str = "ak8"

        if self.jet_factory[jet_factory_str] is None:
            return jets, None

        import cachetools

        jec_cache = cachetools.Cache(np.inf)

        if isData:
            if year == "2022EE" and (dataset == "Run2022F" or dataset == "Run2022E"):
                corr_key = f"{year}NOJER_runF"
            elif year == "2022EE" and dataset == "Run2022G":
                corr_key = f"{year}NOJER_runG"
            elif year == "2022" and dataset == "Run2022C":
                corr_key = f"{year}NOJER_runC"
            elif year == "2022" and dataset == "Run2022D":
                corr_key = f"{year}NOJER_runD"
            else:
                print(dataset, year)
                print(
                    "warning, no valid dataset for 2022 corrections, JECs won't be applied to data"
                )
                applyData = False
        else:
            corr_key = f"{year}mc"

        apply_jecs = ak.any(jets.pt) if (applyData or not isData) else False

        # fatjet_factory.build gives an error if there are no jets in event
        if apply_jecs:
            rho = (
                events.Rho.fixedGridRhoFastjetAll
                if "Rho" in events.fields
                else events.fixedGridRhoFastjetAll
            )
            jets = self.jet_factory[jet_factory_str][corr_key].build(
                self._add_jec_variables(jets, rho, isData), jec_cache
            )

        # return only jets if no jecs given
        if jecs is None or isData:
            return jets, None

        jec_shifted_vars = {}

        for jec_var in jec_vars:
            tdict = {"": jets[jec_var]}
            if apply_jecs:
                for key, shift in jecs.items():
                    for var in ["up", "down"]:
                        tdict[f"{key}_{var}"] = jets[shift][var][jec_var]

            jec_shifted_vars[jec_var] = tdict

        return jets, jec_shifted_vars


def get_jmsr(
    fatjets: FatJetArray, num_jets: int, year: str, isData: bool = False, seed: int = 42
) -> dict:
    """Calculates post JMS/R masses and shifts"""
    jmsr_shifted_vars = {}

    for mkey in jmsr_vars:
        tdict = {}

        mass = pad_val(fatjets[mkey], num_jets, axis=1)

        if isData:
            tdict[""] = mass
        else:
            rng = np.random.default_rng(seed)
            smearing = rng.normal(size=mass.shape)
            # scale to JMR nom, down, up (minimum at 0)
            jmr_nom, jmr_down, jmr_up = (
                (smearing * max(jmrValues[mkey][year][i] - 1, 0) + 1) for i in range(3)
            )
            jms_nom, jms_down, jms_up = jmsValues[mkey][year]

            corr_mass_JMRUp = random.gauss(0.0, jmrValues[mkey][year][2] - 1.0)
            corr_mass = (
                max(jmrValues[mkey][year][0] - 1.0, 0.0)
                / (jmrValues[mkey][year][2] - 1.0)
                * corr_mass_JMRUp
            )

            mass_jms = mass * jms_nom
            mass_jmr = mass * jmr_nom

            tdict[""] = mass * jms_nom * (1.0 + corr_mass)
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
def get_jetveto(jets: JetArray, year: str, run: np.ndarray, isData: bool):
    # https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/JME_2022_Prompt_jetvetomaps.html
    # correction: Non-zero value for (eta, phi) indicates that the region is vetoed
    # for samples related to RunEFG, it is recommended to utilize the vetomap that has been derived for RunEFG
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetveto", year.replace("EE", "")))

    j, nj = ak.flatten(jets), ak.num(jets)

    def get_veto(j, nj, csetstr):
        j_phi = np.clip(np.array(j.phi), -3.1415, 3.1415)
        j_eta = np.clip(np.array(j.eta), -4.7, 4.7)
        veto = cset[csetstr].evaluate("jetvetomap", j_eta, j_phi)
        return ak.unflatten(veto, nj)

    if isData:
        jet_veto = (
            # RunCD
            (
                (run >= FirstRun_2022C)
                & (run <= LastRun_2022D)
                & (get_veto(j, nj, "Winter22Run3_RunCD_V1") > 0)
            )
            |
            # RunE (and later?)
            ((run >= FirstRun_2022E) & (get_veto(j, nj, "Winter22Run3_RunE_V1") > 0))
        )
    else:
        if year == "2022":
            jet_veto = get_veto(j, nj, "Winter22Run3_RunCD_V1") > 0
        else:
            jet_veto = get_veto(j, nj, "Winter22Run3_RunE_V1") > 0

    return jet_veto


def get_jetveto_event(jets: JetArray, year: str, run: np.ndarray, isData: bool):
    """
    Get event selection that rejects events with jets in the veto map
    """
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetveto", year.replace("EE", "")))

    j, nj = ak.flatten(jets), ak.num(jets)

    def get_veto(j, nj, csetstr):
        j_phi = np.clip(np.array(j.phi), -3.1415, 3.1415)
        j_eta = np.clip(np.array(j.eta), -4.7, 4.7)
        veto = cset[csetstr].evaluate("jetvetomap_eep", j_eta, j_phi)
        return ak.unflatten(veto, nj)

    if isData:
        event_sel = (
            # Run CD
            (run >= FirstRun_2022C)
            & (run <= LastRun_2022D)
            & ~(ak.any((jets.pt > 30) & (get_veto(j, nj, "Winter22Run3_RunCD_V1") > 0), axis=1))
        ) | (
            # RunE
            (run >= FirstRun_2022E)
            & ~(ak.any((jets.pt > 30) & (get_veto(j, nj, "Winter22Run3_RunE_V1") > 0), axis=1))
        )
    else:
        if year == "2022":
            event_sel = ~(
                ak.any((jets.pt > 30) & (get_veto(j, nj, "Winter22Run3_RunCD_V1") > 0), axis=1)
            )
        else:
            event_sel = ~(
                ak.any((jets.pt > 30) & (get_veto(j, nj, "Winter22Run3_RunE_V1") > 0), axis=1)
            )

    return event_sel


def add_trig_weights(weights: Weights, fatjets: FatJetArray, year: str, num_jets: int = 2):
    """
    Add the trigger scale factor weights and uncertainties

    Give number of jets in pre-selection to obtain event weight
    """
    if year == "2018":
        with Path(f"{package_path}/corrections/data/fatjet_triggereff_{year}_combined.pkl").open(
            "rb"
        ) as filehandler:
            combined = pickle.load(filehandler)

        # sum over TH4q bins
        effs_txbb = combined["num"][:, sum, :, :] / combined["den"][:, sum, :, :]

        ak8TrigEffsLookup = dense_lookup(
            np.nan_to_num(effs_txbb.view(flow=False), 0), np.squeeze(effs_txbb.axes.edges)
        )

        fj_trigeffs = ak8TrigEffsLookup(
            pad_val(fatjets.Txbb, num_jets, axis=1),
            pad_val(fatjets.pt, num_jets, axis=1),
            pad_val(fatjets.msoftdrop, num_jets, axis=1),
        )

        combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1)

        weights.add("trig_effs", combined_trigEffs)
        return

    # TODO: replace year
    year = "2022EE"
    jet_triggerSF = correctionlib.CorrectionSet.from_file(
        package_path + f"/corrections/data/fatjet_triggereff_{year}_combined_nodijet.json"
    )

    fatjets_xbb = pad_val(fatjets.Txbb, num_jets, axis=1)
    fatjets_pt = pad_val(fatjets.pt, num_jets, axis=1)

    nom_data = jet_triggerSF[f"fatjet_triggereff_{year}_data"].evaluate(
        "nominal", fatjets_pt, fatjets_xbb
    )
    nom_mc = jet_triggerSF[f"fatjet_triggereff_{year}_MC"].evaluate(
        "nominal", fatjets_pt, fatjets_xbb
    )

    nom_data_up = jet_triggerSF[f"fatjet_triggereff_{year}_data"].evaluate(
        "stat_up", fatjets_pt, fatjets_xbb
    )
    nom_mc_up = jet_triggerSF[f"fatjet_triggereff_{year}_MC"].evaluate(
        "stat_up", fatjets_pt, fatjets_xbb
    )

    nom_data_dn = jet_triggerSF[f"fatjet_triggereff_{year}_data"].evaluate(
        "stat_dn", fatjets_pt, fatjets_xbb
    )
    nom_mc_dn = jet_triggerSF[f"fatjet_triggereff_{year}_MC"].evaluate(
        "stat_dn", fatjets_pt, fatjets_xbb
    )

    # calculate trigger weight per event and take ratio from data/MC
    combined_eff_data = 1 - np.prod(1 - nom_data, axis=1)
    combined_eff_mc = 1 - np.prod(1 - nom_mc, axis=1)
    sf = combined_eff_data / combined_eff_mc

    sf_up = (1 - np.prod(1 - nom_data_up, axis=1)) / (1 - np.prod(1 - nom_mc_up, axis=1))
    sf_dn = (1 - np.prod(1 - nom_data_dn, axis=1)) / (1 - np.prod(1 - nom_mc_dn, axis=1))

    weights.add(f"trigsf_{num_jets}jet", sf, sf_up, sf_dn)
