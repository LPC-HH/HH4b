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


"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "jec": ["JME", "fatJet_jerc.json.gz"],
    "jetveto": ["JME", "jetvetomaps.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
}

# TODO: get rid of Run-2 years
def get_jec_key(year: str):
    thekey = f"{year}mc"
    if year == "2016":
        thekey = "2016postVFPmc"
    elif year == "2016APV":
        thekey = "2016preVFPmc"
    return thekey

def get_vfp_year(year: str) -> str:
    if year == "2016":
        year = "2016postVFP"
    elif year == "2016APV":
        year = "2016preVFP"

    return year

def get_UL_year(year: str) -> str:
    return f"{get_vfp_year(year)}_UL"

def get_Prompt_year(year: str) -> str:
    return f"{year}_Prompt"

def get_pog_json(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = get_UL_year(year) if year!="2022" else get_Prompt_year(year)
    return f"{pog_correction_path}/POG/{pog_json[0]}/{year}/{pog_json[1]}"


def add_pileup_weight(weights: Weights, year: str, nPU: np.ndarray):
    """
    Should be able to do something similar to lepton weight but w pileup
    e.g. see here: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/LUMI_puWeights_Run2_UL/
    """
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))

    year_to_corr = {
        "2016": "Collisions16_UltraLegacy_goldenJSON",
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
    }

    values = {}

    values["nominal"] = cset[year_to_corr[year]].evaluate(nPU, "nominal")
    values["up"] = cset[year_to_corr[year]].evaluate(nPU, "up")
    values["down"] = cset[year_to_corr[year]].evaluate(nPU, "down")

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


def _btagSF(cset, jets, flavour, wp="M", algo="deepJet", syst="central"):
    j, nj = ak.flatten(jets), ak.num(jets)
    corrs = cset[f"{algo}_comb"] if flavour == "bc" else cset[f"{algo}_incl"]
    sf = corrs.evaluate(
        syst,
        wp,
        np.array(j.hadronFlavour),
        np.array(abs(j.eta)),
        np.array(j.pt),
    )
    return ak.unflatten(sf, nj)


def _btag_prod(eff, sf):
    num = ak.fill_none(ak.prod(1 - sf * eff, axis=-1), 1)
    den = ak.fill_none(ak.prod(1 - eff, axis=-1), 1)
    return num, den


def add_btag_weights(
    weights: Weights,
    year: str,
    jets: JetArray,
    jet_selector: ak.Array,
    wp: str = "M",
    algo: str = "deepJet",
):
    ul_year = get_UL_year(year)
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("btagging", year))
    efflookup = cutil.load(package_path + f"/corrections/btag_effs/btageff_deepJet_M_{year}.coffea")

    lightJets = jets[jet_selector & (jets.hadronFlavour == 0)]
    bcJets = jets[jet_selector & (jets.hadronFlavour > 0)]

    lightEff = efflookup(lightJets.pt, abs(lightJets.eta), lightJets.hadronFlavour)
    bcEff = efflookup(bcJets.pt, abs(bcJets.eta), bcJets.hadronFlavour)

    lightSF = _btagSF(cset, lightJets, "light", wp, algo)
    bcSF = _btagSF(cset, bcJets, "bc", wp, algo)

    lightnum, lightden = _btag_prod(lightEff, lightSF)
    bcnum, bcden = _btag_prod(bcEff, bcSF)

    weight = np.nan_to_num((1 - lightnum * bcnum) / (1 - lightden * bcden), nan=1)
    weights.add("btagSF", weight)


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

# FIXME: update using Json Paths
# find corrections path using this file's path
try:
    with open(package_path + "/corrections/jec_compiled.pkl", "rb") as filehandler:
        jmestuff = pickle.load(filehandler)

    ak4jet_factory = jmestuff["jet_factory"]
    fatjet_factory = jmestuff["fatjet_factory"]
except:
    print("Failed loading compiled JECs")

def _add_jec_variables(jets: JetArray, event_rho: ak.Array) -> JetArray:
    """add variables needed for JECs"""
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    # gen pT needed for smearing
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


def get_jec_jets(
    events: NanoEventsArray,
    jets: FatJetArray,
    year: str,
    isData: bool = False,
    jecs: Dict[str, str] = None,
    fatjets: bool = True,
) -> FatJetArray:
    """
    Based on https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    Eventually update to V5 JECs once I figure out what's going on with the 2017 UL V5 JER scale factors

    See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/

    If ``jecs`` is not None, returns the shifted values of variables are affected by JECs.
    """

    jec_vars = ["pt"]  # variables we are saving that are affected by JECs
    if fatjets:
        jet_factory = fatjet_factory
    else:
        jet_factory = ak4jet_factory

    apply_jecs = not (not ak.any(jets.pt) or isData)

    import cachetools

    jec_cache = cachetools.Cache(np.inf)

    corr_key = f"{get_vfp_year(year)}mc"

    # fatjet_factory.build gives an error if there are no fatjets in event
    if apply_jecs:
        jets = jet_factory[corr_key].build(
            _add_jec_variables(jets, events.Rho.fixedGridRhoFastjetAll), jec_cache
        )

    # return only fatjets if no jecs given
    if jecs is None:
        return jets

    jec_shifted_vars = {}

    for jec_var in jec_vars:
        tdict = {"": jets[jec_var]}
        if apply_jecs:
            for key, shift in jecs.items():
                for var in ["up", "down"]:
                    tdict[f"{key}_{var}"] = jets[shift][var][jec_var]

        jec_shifted_vars[jec_var] = tdict

    return jets, jec_shifted_vars


jmsr_vars = ["msoftdrop", "particleNet_mass"]

jmsValues = {}
jmrValues = {}

# https://github.com/cms-nanoAOD/nanoAOD-tools/blob/959c9ffb084bc974fb26ba2db41e3369cee04ae7/python/postprocessing/modules/jme/jetmetHelperRun2.py#L85-L110

# jet mass resolution: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
# nominal, down, up (these are switched in the github!!!)
jmrValues["msoftdrop"] = {
    "2016": [1.0, 0.8, 1.2],
    "2017": [1.09, 1.04, 1.14],
    # Use 2017 values for 2018 until 2018 are released
    "2018": [1.09, 1.04, 1.14],
}

# jet mass scale
# W-tagging PUPPI softdrop JMS values: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
# 2016 values
jmsValues["msoftdrop"] = {
    "2016": [1.00, 0.9906, 1.0094],  # nominal, down, up
    "2017": [0.982, 0.978, 0.986],
    # Use 2017 values for 2018 until 2018 are released
    "2018": [0.982, 0.978, 0.986],
}

# https://github.com/cmantill/NanoNN/blob/6bd117357e2d7ec66866b5f74790e747411efcad/python/producers/hh4bProducer.py#L154-L159

# nominal, down, up
jmrValues["particleNet_mass"] = {
    "2016": [1.028, 1.007, 1.063],
    "2017": [1.026, 1.009, 1.059],
    "2018": [1.031, 1.006, 1.075],
}
jmsValues["particleNet_mass"] = {
    "2016": [1.00, 0.998, 1.002],
    "2017": [1.002, 0.996, 1.008],
    "2018": [0.994, 0.993, 1.001],
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


def add_trig_effs(weights: Weights, fatjets: FatJetArray, year: str, num_jets: int = 2):
    """Add the trigger efficiencies we measured in SingleMuon data"""
    with open(f"{package_path}/corrections/trigEffs/{year}_combined.pkl", "rb") as filehandler:
        combined = pickle.load(filehandler)

    # sum over TH4q bins
    effs_txbb = combined["num"][:, sum, :, :] / combined["den"][:, sum, :, :]

    ak8TrigEffsLookup = dense_lookup(
        np.nan_to_num(effs_txbb.view(flow=False), 0), np.squeeze(effs_txbb.axes.edges)
    )

    # TODO: confirm that these should be corrected pt, msd values
    fj_trigeffs = ak8TrigEffsLookup(
        pad_val(fatjets.Txbb, num_jets, axis=1),
        pad_val(fatjets.pt, num_jets, axis=1),
        pad_val(fatjets.msoftdrop, num_jets, axis=1),
    )

    # combined eff = 1 - (1 - fj1_eff) * (1 - fj2_eff)
    combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1)

    weights.add("trig_effs", combined_trigEffs)

FirstRun_2022C = 355794
LastRun_2022D = 359021
FirstRun_2022E = 359022

# Jet Veto Maps
# the JERC group recommends ALL analyses use these maps, as the JECs are derived excluding these zones.
# apply to both Data and MC
# https://cms-talk.web.cern.ch/t/jet-veto-maps-for-run3-data/18444?u=anmalara
def get_jetveto(jets: JetArray, year: str, run: np.ndarray):
    # https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/JME_2022_Prompt_jetvetomaps.html
    # correction: Non-zero value for (eta, phi) indicates that the region is vetoed
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetveto", year))
    
    j, nj = ak.flatten(jets), ak.num(jets)
    
    def get_veto(j, nj, csetstr):
        j_phi = np.clip(np.array(j.eta), -3.1415, 3.1415)
        veto = cset[csetstr].evaluate("jetvetomap", np.array(j.eta), j_phi)
        return ak.unflatten(veto, nj)

    jet_veto = (
        # RunCD
        (
            (run >= FirstRun_2022C) & (run <= LastRun_2022D)
            & ( get_veto(j, nj, "Winter22Run3_RunCD_V1") > 0 )
        ) |
        # RunE
        (
            (run >= FirstRun_2022E) 
            & ( get_veto(j, nj,"Winter22Run3_RunE_V1") > 0 )
        )
    )

    return jet_veto

def get_jetveto_event(jets: JetArray, year: str, run: np.ndarray):
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetveto", year))

    j, nj = ak.flatten(jets), ak.num(jets)
    
    def get_veto(j, nj, csetstr):
        j_phi = np.clip(np.array(j.eta), -3.1415, 3.1415)
        veto = cset[csetstr].evaluate("jetvetomap_eep", np.array(j.eta), j_phi)
        return ak.unflatten(veto, nj)

    event_veto = (
        (
            (run >= FirstRun_2022C) & (run <= LastRun_2022D)
            & ( ak.any((jets.pt > 30) & (get_veto(j, nj, "Winter22Run3_RunCD_V1") > 0), axis=1) )
        ) |
        (
            (run >= FirstRun_2022E)
            & ( ak.any((jets.pt > 30) & (get_veto(j, nj, "Winter22Run3_RunE_V1") > 0), axis=1) )
        )
    )

    return event_veto
