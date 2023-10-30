"""
Collection of utilities for corrections and systematics in processors.

Loosely based on https://github.com/jennetd/hbb-coffea/blob/master/boostedhiggs/corrections.py

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/

Authors: Raghav Kansal, Cristina Suarez
"""

import os
from typing import Dict, List, Tuple, Union
import numpy as np
import gzip
import pickle
import correctionlib
import awkward as ak

from coffea import util as cutil
from coffea.analysis_tools import Weights
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents.methods.nanoaod import MuonArray, JetArray, FatJetArray, GenParticleArray
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)

import pathlib

from . import utils
from .utils import P4, pad_val


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


def add_pileup_weight(weights: Weights, year: str, nPU: np.ndarray):
    # TODO: Switch to official recommendation when and if any
    if year=="2018":
        cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))
        y = year
    else:
        cset = correctionlib.CorrectionSet.from_file(
            package_path + "/corrections/2022_puWeights.json.gz"
        )
        y = get_Prompt_year(year)

    year_to_corr = {
        "2018": "Collisions18_UltraLegacy_goldenJSON",
        "2022_Prompt": "Collisions_2022_PromptReco_goldenJSON",
        "2022EE_Prompt": "Collisions_2022_PromptReco_goldenJSON",
    }
    
    values = {}

    # evaluate and clip up to 10 to avoid large weights
    values["nominal"] = np.clip(cset[year_to_corr[y]].evaluate(nPU, "nominal"), 0, 10)
    values["up"] = np.clip(cset[year_to_corr[y]].evaluate(nPU, "up"), 0, 10)
    values["down"] = np.clip(cset[year_to_corr[y]].evaluate(nPU, "down"), 0, 10)

    # add weights (for now only the nominal weight)
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


def add_VJets_kFactors(weights, genpart, dataset):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""

    vjets_kfactors = correctionlib.CorrectionSet.from_file(
        package_path + "/corrections/ULvjets_corrections.json"
    )

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    znlosysts = [
        "d1kappa_EW",
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(
                syst,
                ones,
                ewkcorr.evaluate(syst + "_up", vpt) / ewknom,
                ewkcorr.evaluate(syst + "_down", vpt) / ewknom,
            )

    if "ZJetsToQQ_HT" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = vjets_kfactors["ULZ_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)
    elif "DYJetsToLL" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = 1
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(znlosysts, qcdcorr, ewkcorr, vpt)
    elif "WJetsToQQ_HT" in dataset or "WJetsToLNu" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = vjets_kfactors["ULW_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)


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
    up = np.ones(nweights)
    down = np.ones(nweights)

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
    docstring = var_weights.__doc__

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
    docstring = var_weights.__doc__

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


TOP_PDGID = 6
GEN_FLAGS = ["fromHardProcess", "isLastCopy"]


def add_top_pt_weight(weights: Weights, events: NanoEventsArray):
    """https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting"""
    # finding the two gen tops
    tops = events.GenPart[
        (abs(events.GenPart.pdgId) == TOP_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]

    # reweighting formula from https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_dat
    # for POWHEG+Pythia8
    tops_sf = np.exp(0.0615 - 0.0005 * tops.pt)
    # SF is geometric mean of both tops' weight
    tops_sf = np.sqrt(tops_sf[:, 0] * tops_sf[:, 1]).to_numpy()
    weights.add("top_pt", tops_sf)


try:
    with gzip.open(package_path + "/corrections/jec_compiled.pkl.gz", "rb") as filehandler:
        jmestuff = pickle.load(filehandler)

    ak4jet_factory = jmestuff["jet_factory"]
    fatjet_factory = jmestuff["fatjet_factory"]
except:
    print("Failed loading compiled JECs")


def _add_jec_variables(jets: JetArray, event_rho: ak.Array, isData: bool) -> JetArray:
    """add variables needed for JECs"""
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    if not isData:
        # gen pT needed for smearing
        jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    return jets


def get_jec_jets(
    events: NanoEventsArray,
    jets: FatJetArray,
    year: str,
    isData: bool = False,
    jecs: Dict[str, str] = None,
    fatjets: bool = True,
    applyData: bool = False,
    dataset: str = None,
) -> FatJetArray:
    """
    If ``jecs`` is not None, returns the shifted values of variables are affected by JECs.
    """
    
    if year=="2018":
        return jets, None

    jec_vars = ["pt"]  # variables we are saving that are affected by JECs
    if fatjets:
        jet_factory = fatjet_factory
    else:
        jet_factory = ak4jet_factory

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
            print("warning, no valid dataset for 2022 corrections, JECs won't be applied to data")
            applyData = False
    else:
        corr_key = f"{year}mc"

    if applyData:
        apply_jecs = ak.any(jets.pt)
    else:
        # do not apply JECs to data
        apply_jecs = not (not ak.any(jets.pt) or isData)

    # fatjet_factory.build gives an error if there are no jets in event
    if apply_jecs:
        jets = jet_factory[corr_key].build(
            _add_jec_variables(jets, events.Rho.fixedGridRhoFastjetAll, isData), jec_cache
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


# Jet mass scale and Jet mass resolution
# FIXME: Using placeholder Run 2 values !!
# nominal, down, up
jmsr_vars = ["msoftdrop", "particleNet_mass"]

jmsValues = {}
jmrValues = {}

jmrValues["msoftdrop"] = {
    "2018": [1.09, 1.04, 1.14],
    "2022": [1.09, 1.04, 1.14],
    "2022EE": [1.09, 1.04, 1.14],
}

jmsValues["msoftdrop"] = {
    "2018": [0.982, 0.978, 0.986],
    "2022": [0.982, 0.978, 0.986],
    "2022EE": [0.982, 0.978, 0.986],
}

jmrValues["particleNet_mass"] = {
    "2018": [1.031, 1.006, 1.075],
    "2022": [1.031, 1.006, 1.075],
    "2022EE": [1.031, 1.006, 1.075],
}

jmsValues["particleNet_mass"] = {
    "2018": [0.994, 0.993, 1.001],
    "2022": [0.994, 0.993, 1.001],
    "2022EE": [0.994, 0.993, 1.001],
}


def get_jmsr(
    fatjets: FatJetArray, num_jets: int, year: str, isData: bool = False, seed: int = 42
) -> Dict:
    """Calculates post JMS/R masses and shifts"""
    jmsr_shifted_vars = {}

    for mkey in jmsr_vars:
        tdict = {}

        mass = utils.pad_val(fatjets[mkey], num_jets, axis=1)

        if isData:
            tdict[""] = mass
        else:
            np.random.seed(seed)
            smearing = np.random.normal(size=mass.shape)
            # scale to JMR nom, down, up (minimum at 0)
            jmr_nom, jmr_down, jmr_up = [
                (smearing * max(jmrValues[mkey][year][i] - 1, 0) + 1) for i in range(3)
            ]
            jms_nom, jms_down, jms_up = jmsValues[mkey][year]

            mass_jms = mass * jms_nom
            mass_jmr = mass * jmr_nom

            tdict[""] = mass_jms * jmr_nom
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
        with open(f"{package_path}/corrections/data/fatjet_triggereff_{year}_combined.pkl", "rb") as filehandler:
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
