"""
Creates datacards for Higgs Combine using hist.Hist templates output from PostProcess.py
(1) Adds systematics for all samples,
(2) Sets up data-driven QCD background estimate ('rhalphabet' method)

Based on https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py

Authors: Raghav Kansal
"""

from __future__ import annotations

# from utils import add_bool_arg
import argparse
import logging
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rhalphalib as rl
from datacardHelpers import (
    ShapeVar,
    Syst,
    add_bool_arg,
    combine_templates,
    get_effect_updown,
    rem_neg,
    sum_templates,
)
from hist import Hist

from HH4b.hh_vars import LUMI, data_key, jecs, jmsr, qcd_key, sig_keys_ggf, sig_keys_vbf

try:
    rl.util.install_roofit_helpers()
    rl.ParametericSample.PreferRooParametricHist = False
except:
    print("rootfit install failed - not an issue for VBF")

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
adjust_posdef_yields = False


parser = argparse.ArgumentParser()
parser.add_argument(
    "--templates-dir",
    default="",
    type=str,
    help="input pickle file of dict of hist.Hist templates",
)

add_bool_arg(parser, "sig-separate", "separate templates for signals and bgs", default=False)
add_bool_arg(parser, "do-jshifts", "Do JEC/JMC corrections.", default=True)

parser.add_argument("--cards-dir", default="cards", type=str, help="output card directory")

parser.add_argument("--mcstats-threshold", default=100, type=float, help="mcstats threshold n_eff")
parser.add_argument(
    "--epsilon",
    default=1e-2,
    type=float,
    help="epsilon to avoid numerical errs - also used to decide whether to add mc stats error",
)
parser.add_argument(
    "--scale-templates", default=None, type=float, help="scale all templates for bias tests"
)
parser.add_argument(
    "--min-qcd-val", default=1e-3, type=float, help="clip the pass QCD to above a minimum value"
)

add_bool_arg(parser, "only-sm", "Only add SM HH samples", default=False)
parser.add_argument("--sig-samples", default="hh4b", nargs="*", type=str, help="specify signals")

parser.add_argument(
    "--nTF",
    default=None,
    nargs="*",
    type=int,
    help="order of polynomial for TF in [cat 1, cat 2, cat 3]. Default is [0, 1, 2]",
)
parser.add_argument(
    "--regions",
    default="all",
    type=str,
    help="regions for which to make cards",
    choices=["pass_vbf", "pass_bin1", "pass_bin2", "pass_bin3", "all"],
)
parser.add_argument("--model-name", default=None, type=str, help="output model name")
parser.add_argument(
    "--year",
    help="year",
    type=str,
    default="2022EE",
    choices=["2022EE", "2022-2023"],
)
add_bool_arg(parser, "mcstats", "add mc stats nuisances", default=True)
add_bool_arg(parser, "bblite", "use barlow-beeston-lite method", default=True)
add_bool_arg(parser, "temp-uncs", "Add temporary lumi, pileup, tagger uncs.", default=False)
add_bool_arg(parser, "vbf-region", "Add VBF region", default=False)
add_bool_arg(parser, "unblinded", "unblinded so skip blinded parts", default=False)
add_bool_arg(parser, "ttbar-rate-param", "Add freely floating ttbar rate param", default=False)
add_bool_arg(parser, "mc-closure", "Perform MC closure test (fill data_obs with sum of MC bkg.", default=False)
args = parser.parse_args()


CMS_PARAMS_LABEL = "CMS_bbbb_hadronic"
MCB_LABEL = "MCBlinded"
qcd_data_key = "qcd_datadriven"

if args.nTF is None:
    if args.regions == "all":
        args.nTF = [0, 1, 2]
        if args.vbf_region:
            args.nTF = [0] + args.nTF
    else:
        args.nTF = [0]

print("Transfer factors:", args.nTF)

if args.regions == "all":
    signal_regions = ["pass_bin1", "pass_bin2", "pass_bin3"]
    if args.vbf_region:
        signal_regions = ["pass_vbf"] + signal_regions
else:
    signal_regions = [args.regions]

# (name in templates, name in cards)
mc_samples = OrderedDict(
    [
        ("ttbar", "ttbar"),
        ("vhtobb", "VH_hbb"),
        ("diboson", "diboson"),
        ("vjets", "vjets"),
        ("tthtobb", "ttH_hbb"),
    ]
)

bg_keys = list(mc_samples.keys())

if args.only_sm:
    sig_keys_ggf, sig_keys_vbf = ["hh4b"], []

all_sig_keys = sig_keys_ggf + sig_keys_vbf
sig_keys = []
hist_names = {}  # names of hist files for the samples

for key in all_sig_keys:
    # check in case single sig sample is specified
    if args.sig_samples is None or key in args.sig_samples:
        # TODO: change names to match HH combination convention
        mc_samples[key] = key
        sig_keys.append(key)


print(sig_keys)
all_mc = list(mc_samples.keys())


years = [args.year]
full_lumi = LUMI[args.year]


# dictionary of nuisance params -> (modifier, samples affected by it, value)
nuisance_params = {
    # https://gitlab.cern.ch/hh/naming-conventions#experimental-uncertainties
    # https://gitlab.cern.ch/hh/naming-conventions#theory-uncertainties
    "BR_hbb": Syst(prior="lnN", samples=sig_keys, value=1.0124**2, value_down=0.9874**2),
    "pdf_gg": Syst(prior="lnN", samples=["ttbar"], value=1.042),
    # "pdf_qqbar": Syst(prior="lnN", samples=["ST"], value=1.027),
    "pdf_Higgs_ggHH": Syst(prior="lnN", samples=sig_keys_ggf, value=1.030),
    "pdf_Higgs_qqHH": Syst(prior="lnN", samples=sig_keys_vbf, value=1.021),
    # TODO: add these for single Higgs backgrounds
    # "pdf_Higgs_gg": Syst(prior="lnN", samples=ggfh_keys, value=1.019),
    "QCDscale_ttbar": Syst(
        prior="lnN",
        samples=["ST", "ttbar"],
        value={"ST": 1.03, "ttbar": 1.024},
        value_down={"ST": 0.978, "ttbar": 0.965},
        diff_samples=True,
    ),
    "QCDscale_qqHH": Syst(prior="lnN", samples=sig_keys_vbf, value=1.0003, value_down=0.9996),
    # "QCDscale_ggH": Syst(
    #     prior="lnN",
    #     samples=ggfh_keys,
    #     value=1.039,
    # ),
    # "alpha_s": for single Higgs backgrounds
    # f"{CMS_PARAMS_LABEL}_triggerEffSF_uncorrelated": Syst(
    #     prior="lnN", samples=all_mc, diff_regions=True
    # ),
    # THU_HH: combined Scale+mtop uncertainty from
    # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH#Latest_recommendations_for_gluon
    # remove for use with inference (assuming correct kl-dependent implementation there)
    "THU_HH": Syst(
        prior="lnN",
        samples=sig_keys_ggf,
        value={"hh4b": 1.06, "hh4b-kl0": 1.08, "hh4b-kl2p45": 1.06, "hh4b-kl5": 1.18},
        value_down={"hh4b": 0.77, "hh4b-kl0": 0.82, "hh4b-kl2p45": 0.75, "hh4b-kl5": 0.87},
        diff_samples=True,
    ),
    # apply 2022 uncertainty to all MC (until 2023 rec.)
    "lumi_2022": Syst(prior="lnN", samples=all_mc, value=1.014),
}

rate_params = {}
if args.ttbar_rate_param:
    rate_params = {"ttbar": rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_rp_ttbar", 1.0, 0, 10)}

# add temporary uncertainties
if args.temp_uncs:
    temp_nps = {
        "lumi_pileup": Syst(prior="lnN", samples=all_mc, value=1.04),
        "signal_eff": Syst(prior="lnN", samples=sig_keys, value=1.1, pass_only=True),
        "top_mistag": Syst(prior="lnN", samples=["ttbar"], value=1.1, pass_only=True),
    }
    nuisance_params = {**nuisance_params, **temp_nps}

nuisance_params_dict = {
    param: rl.NuisanceParameter(param, syst.prior) for param, syst in nuisance_params.items()
}

# dictionary of correlated shape systematics: name in templates -> name in cards, etc.
corr_year_shape_systs = {
    # "FSRPartonShower": Syst(name="ps_fsr", prior="shape", samples=nonres_sig_keys_ggf + ["V+Jets"]),
    # "ISRPartonShower": Syst(name="ps_isr", prior="shape", samples=nonres_sig_keys_ggf + ["V+Jets"]),
    # TODO: should we be applying QCDscale for "others" process?
    # https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L290
    # "QCDscale": Syst(
    #     name=f"{CMS_PARAMS_LABEL}_ggHHQCDacc", prior="shape", samples=nonres_sig_keys_ggf
    # ),
    # "PDFalphaS": Syst(
    #     name=f"{CMS_PARAMS_LABEL}_ggHHPDFacc", prior="shape", samples=nonres_sig_keys_ggf
    # ),
    # TODO: separate into individual
    "JES": Syst(name="CMS_scale_j", prior="shape", samples=sig_keys),  # TODO: update to all_mc
    "ttbarSF": Syst(
        name=f"{CMS_PARAMS_LABEL}_ttbar_sf",
        prior="shape",
        samples=["ttbar"],
        convert_shape_to_lnN=True,
    ),
    # "trigger": Syst(name=f"{CMS_PARAMS_LABEL}_trigger", prior="shape", samples=all_mc),  # TODO: fix
    # "txbb": Syst(
    #     name=f"{CMS_PARAMS_LABEL}_PNetHbbScaleFactors_correlated",
    #     prior="shape",
    #     samples=sig_keys,
    #     pass_only=True,
    # ),
    # "top_pt": Syst(name="CMS_top_pT_reweighting", prior="shape", samples=["ttbar"])  # TODO
}

uncorr_year_shape_systs = {
    # "pileup": Syst(name="CMS_pileup", prior="shape", samples=all_mc),
    "JER": Syst(name="CMS_res_j", prior="shape", samples=all_mc),
    "JMS": Syst(name=f"{CMS_PARAMS_LABEL}_jms", prior="shape", samples=all_mc),
    "JMR": Syst(name=f"{CMS_PARAMS_LABEL}_jmr", prior="shape", samples=all_mc),
}

if not args.do_jshifts:
    del corr_year_shape_systs["JES"]
    del uncorr_year_shape_systs["JER"]
    del uncorr_year_shape_systs["JMS"]
    del uncorr_year_shape_systs["JMR"]
else:
    # TODO: implement others; currently only JES
    del uncorr_year_shape_systs["JER"]
    del uncorr_year_shape_systs["JMS"]
    del uncorr_year_shape_systs["JMR"]


shape_systs_dict = {}
for skey, syst in corr_year_shape_systs.items():
    if syst.convert_shape_to_lnN:
        shape_systs_dict[skey] = rl.NuisanceParameter(syst.name, "lnN")
    else:
        shape_systs_dict[skey] = rl.NuisanceParameter(syst.name, "shape")
for skey, syst in uncorr_year_shape_systs.items():
    for year in years:
        if year in syst.uncorr_years:
            shape_systs_dict[f"{skey}_{year}"] = rl.NuisanceParameter(
                f"{syst.name}_{year}", "shape"
            )


def get_templates(
    templates_dir: str,
    years: list[str],
    sig_separate: bool,
    scale: float | None = None,
):
    """Loads templates, combines bg and sig templates if separate, sums across all years"""
    templates_dict: dict[str, dict[str, Hist]] = {}

    if not sig_separate:
        # signal and background templates in same hist, just need to load and sum across years
        for year in years:
            with Path(f"{templates_dir}/{year}_templates.pkl").open("rb") as f:
                templates_dict[year] = rem_neg(pickle.load(f))
    else:
        # signal and background in different hists - need to combine them into one hist
        for year in years:
            with Path(f"{templates_dir}/backgrounds/{year}_templates.pkl").open("rb") as f:
                bg_templates = rem_neg(pickle.load(f))

            sig_templates = []

            for sig_key in sig_keys:
                with Path(f"{templates_dir}/{hist_names[sig_key]}/{year}_templates.pkl").open(
                    "rb"
                ) as f:
                    sig_templates.append(rem_neg(pickle.load(f)))

            templates_dict[year] = combine_templates(bg_templates, sig_templates)

    if scale is not None and scale != 1:
        for year in templates_dict:
            for key in templates_dict[year]:
                templates_dict[year][key] = templates_dict[year][key] * scale

    templates_summed: dict[str, Hist] = sum_templates(templates_dict, years)  # sum across years
    return templates_dict, templates_summed


def get_year_updown(templates_dict, sample, region, region_noblinded, blind_str, year, skey):
    """
    Return templates with only the given year's shapes shifted up and down by the ``skey`` systematic.
    Returns as [up templates, down templates]
    """
    updown = []

    for shift in ["up", "down"]:
        sshift = f"{skey}_{shift}"
        # get nominal templates for each year
        templates = {y: templates_dict[y][region][sample, ...] for y in years}

        # replace template for this year with the shifted template
        if skey in jecs or skey in jmsr:
            # JEC/JMCs saved as different "region" in dict
            reg_name = f"{region_noblinded}_{sshift}{blind_str}"
            templates[year] = templates_dict[year][reg_name][sample, ...]
        else:
            # weight uncertainties saved as different "sample" in dict
            templates[year] = templates_dict[year][region][f"{sample}_{sshift}", ...]

        # sum templates with year's template replaced with shifted
        updown.append(sum(list(templates.values())).values())

    return updown


def fill_regions(
    model: rl.Model,
    regions: list[str],
    templates_dict: dict,
    templates_summed: dict,
    mc_samples: dict[str, str],
    nuisance_params: dict[str, Syst],
    nuisance_params_dict: dict[str, rl.NuisanceParameter],
    corr_year_shape_systs: dict[str, Syst],
    uncorr_year_shape_systs: dict[str, Syst],
    shape_systs_dict: dict[str, rl.NuisanceParameter],
    bblite: bool = True,
):
    """Fill samples per region including given rate, shape and mcstats systematics.
    Ties "MCBlinded" and "nonblinded" mc stats parameters together.

    Args:
        model (rl.Model): rhalphalib model
        regions (List[str]): list of regions to fill
        templates_dict (Dict): dictionary of all templates
        templates_summed (Dict): dictionary of templates summed across years
        mc_samples (Dict[str, str]): dict of mc samples and their names in the given templates -> card names
        nuisance_params (Dict[str, Tuple]): dict of nuisance parameter names and tuple of their
          (modifier, samples affected by it, value)
        nuisance_params_dict (Dict[str, rl.NuisanceParameter]): dict of nuisance parameter names
          and NuisanceParameter object
        corr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (correlated across years)
          and tuple of their (name in cards, samples affected by it)
        uncorr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (unccorrelated across years)
          and tuple of their (name in cards, samples affected by it)
        shape_systs_dict (Dict[str, rl.NuisanceParameter]): dict of shape syst names and
          NuisanceParameter object
        pass_only (List[str]): list of systematics which are only applied in the pass region(s)
        bblite (bool): use Barlow-Beeston-lite method or not (single mcstats param across MC samples)
    """

    for region in regions:
        region_templates = templates_summed[region]

        pass_region = region.startswith("pass")
        region_noblinded = region.split(MCB_LABEL)[0]
        blind_str = MCB_LABEL if region.endswith(MCB_LABEL) else ""

        logging.info("starting region: %s" % region)
        ch = rl.Channel(region.replace("_", ""))  # can't have '_'s in name
        model.addChannel(ch)

        for sample_name, card_name in mc_samples.items():
            # don't add signals in fail regions
            if sample_name in sig_keys and not pass_region:
                logging.info(f"\nSkipping {sample_name} in {region} region\n")
                continue

            logging.info("get templates for: %s" % sample_name)

            sample_template = region_templates[sample_name, :]

            stype = rl.Sample.SIGNAL if sample_name in sig_keys else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + card_name, stype, sample_template)

            # ttbar rate_param
            if sample_name in rate_params:
                rate_param = rate_params[sample_name]
                sample.setParamEffect(rate_param, 1 * rate_param)

            # # rate params per signal to freeze them for individual limits
            # if stype == rl.Sample.SIGNAL and len(sig_keys) > 1:
            #     srate = rate_params[sample_name]
            #     sample.setParamEffect(srate, 1 * srate)

            # nominal values, errors
            values_nominal = np.maximum(sample_template.values(), 0.0)

            mask = values_nominal > 0
            errors_nominal = np.ones_like(values_nominal)
            errors_nominal[mask] = (
                1.0 + np.sqrt(sample_template.variances()[mask]) / values_nominal[mask]
            )

            logging.debug(f"nominal   : {values_nominal}")
            logging.debug(f"error     : {errors_nominal}")

            if not bblite and args.mcstats:
                # set mc stat uncs
                logging.info(f"setting autoMCStats for {sample_name} in {region}")

                # tie MC stats parameters together in blinded and "unblinded" region
                stats_sample_name = f"{CMS_PARAMS_LABEL}_{region}_{card_name}"
                sample.autoMCStats(
                    sample_name=stats_sample_name,
                    # this function uses a different threshold convention from combine
                    threshold=np.sqrt(1 / args.mcstats_threshold),
                    epsilon=args.epsilon,
                )

            # rate systematics
            for skey, syst in nuisance_params.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"Getting {skey} rate")

                param = nuisance_params_dict[skey]

                val, val_down = syst.value, syst.value_down
                if syst.diff_regions:
                    val = val[region]
                    val_down = val_down[region] if val_down is not None else val_down
                if syst.diff_samples:
                    val = val[sample_name]
                    val_down = val_down[sample_name] if val_down is not None else val_down

                sample.setParamEffect(param, val, effect_down=val_down)

            # correlated shape systematics
            for skey, syst in corr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"Getting {skey} shapes")

                if skey in jecs or skey in jmsr:
                    # JEC/JMCs saved as different "region" in dict
                    up_hist = templates_summed[f"{region_noblinded}_{skey}_up{blind_str}"][
                        sample_name, :
                    ]
                    down_hist = templates_summed[f"{region_noblinded}_{skey}_down{blind_str}"][
                        sample_name, :
                    ]

                    values_up = up_hist.values()
                    values_down = down_hist.values()
                else:
                    # weight uncertainties saved as different "sample" in dict
                    values_up = region_templates[f"{sample_name}_{skey}_up", :].values()
                    values_down = region_templates[f"{sample_name}_{skey}_down", :].values()

                logger = logging.getLogger(f"validate_shapes_{region}_{sample_name}_{skey}")

                effect_up, effect_down = get_effect_updown(
                    values_nominal,
                    values_up,
                    values_down,
                    mask,
                    logger,
                    args.epsilon,
                    syst.convert_shape_to_lnN,
                )
                sample.setParamEffect(shape_systs_dict[skey], effect_up, effect_down)

            # uncorrelated shape systematics
            for skey, syst in uncorr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"Getting {skey} shapes")

                for year in years:
                    if year not in syst.uncorr_years:
                        continue

                    values_up, values_down = get_year_updown(
                        templates_dict,
                        sample_name,
                        region,
                        region_noblinded,
                        blind_str,
                        year,
                        skey,
                    )
                    logger = logging.getLogger(f"validate_shapes_{region}_{sample_name}_{skey}")

                    effect_up, effect_down = get_effect_updown(
                        values_nominal, values_up, values_down, mask, logger, args.epsilon
                    )
                    sample.setParamEffect(
                        shape_systs_dict[f"{skey}_{year}"], effect_up, effect_down
                    )

            ch.addSample(sample)

        if bblite and args.mcstats:
            # tie MC stats parameters together in blinded and "unblinded" region
            channel_name = region_noblinded
            ch.autoMCStats(
                channel_name=f"{CMS_PARAMS_LABEL}_{channel_name}",
                threshold=args.mcstats_threshold,
                epsilon=args.epsilon,
            )

        # data observed
        if args.mc_closure:
            all_bg = sum([region_templates[bg_key, :] for bg_key in bg_keys + ["qcd"]])
            ch.setObservation(all_bg)
        else:
           ch.setObservation(region_templates[data_key, :])


def alphabet_fit(
    model: rl.Model,
    shape_vars: list[ShapeVar],
    templates_summed: dict,
    scale: float | None = None,
    min_qcd_val: float | None = None,
    unblinded: bool = False,
):
    shape_var = shape_vars[0]
    m_obs = rl.Observable(shape_var.name, shape_var.bins)

    ##########################
    # Setup fail region first
    ##########################

    # Independent nuisances to float QCD in each fail bin
    qcd_params = np.array(
        [
            rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_tf_dataResidual_Bin{i}", 0)
            for i in range(m_obs.nbins)
        ]
    )

    fail_qcd_samples = {}

    blind_strs = [""] if unblinded else ["", MCB_LABEL]
    for blind_str in blind_strs:
        failChName = f"fail{blind_str}".replace("_", "")
        logging.info(f"Setting up fail region {failChName}")
        failCh = model[failChName]

        # sideband fail
        # was integer, and numpy complained about subtracting float from it
        initial_qcd = failCh.getObservation().astype(float)
        for sample in failCh:
            # don't subtract signals (#TODO: do we want to subtract SM signal?)
            if sample.sampletype == rl.Sample.SIGNAL:
                continue
            logging.debug(
                f"subtracting {sample._name}={sample.getExpectation(nominal=True)} from qcd"
            )
            initial_qcd -= sample.getExpectation(nominal=True)

        if np.any(initial_qcd < 0.0):
            raise ValueError("initial_qcd negative for some bins..", initial_qcd)

        # idea here is that the error should be 1/sqrt(N), so parametrizing it as (1 + 1/sqrt(N))^qcdparams
        # will result in qcdparams errors ~±1
        # but because qcd is poorly modelled we're scaling sigma scale

        sigmascale = 10  # to scale the deviation from initial
        if scale is not None:
            sigmascale *= scale

        scaled_params = (
            initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcd_params
        )

        # add samples
        fail_qcd = rl.ParametericSample(
            f"{failChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
            rl.Sample.BACKGROUND,
            m_obs,
            scaled_params,
        )
        failCh.addSample(fail_qcd)

        fail_qcd_samples[blind_str] = fail_qcd

    ##########################
    # Now do signal regions
    ##########################

    for sr in signal_regions:
        # QCD overall pass / fail efficiency
        qcd_eff = (
            templates_summed[sr][qcd_key, :].sum().value
            / templates_summed["fail"][qcd_key, :].sum().value
        )
        logging.info(f"qcd eff {qcd_eff:.5f}")

        # transfer factor
        tf_dataResidual = rl.BasisPoly(
            f"{CMS_PARAMS_LABEL}_tf_dataResidual_{sr}",
            (shape_var.orders[sr],),
            [shape_var.name],
            basis="Bernstein",
            limits=(-20, 20),
            square_params=True,
        )
        tf_dataResidual_params = tf_dataResidual(shape_var.scaled)
        tf_params_pass = qcd_eff * tf_dataResidual_params  # scale params initially by qcd eff

        for blind_str in blind_strs:
            passChName = f"{sr}{blind_str}".replace("_", "")
            passCh = model[passChName]

            pass_qcd = rl.TransferFactorSample(
                f"{passChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
                rl.Sample.BACKGROUND,
                tf_params_pass,
                fail_qcd_samples[blind_str],
                min_val=min_qcd_val,
            )
            passCh.addSample(pass_qcd)


def createDatacardAlphabet(args, templates_dict, templates_summed, shape_vars):
    # (pass, fail) x (unblinded, blinded)
    blind_strs = [""] if args.unblinded else ["", MCB_LABEL]

    regions: list[str] = [
        f"{pf}{blind_str}" for pf in [*signal_regions, "fail"] for blind_str in blind_strs
    ]

    # build actual fit model now
    model = rl.Model("HHModel")

    # Fill templates per sample, incl. systematics
    fill_args = [
        model,
        regions,
        templates_dict,
        templates_summed,
        mc_samples,
        nuisance_params,
        nuisance_params_dict,
        corr_year_shape_systs,
        uncorr_year_shape_systs,
        shape_systs_dict,
        args.bblite,
    ]

    fit_args = [
        model,
        shape_vars,
        templates_summed,
        args.scale_templates,
        args.min_qcd_val,
        args.unblinded,
    ]

    fill_regions(*fill_args)
    alphabet_fit(*fit_args)

    ##############################################
    # Save model
    ##############################################

    logging.info("rendering combine model")

    out_dir = (
        Path(args.cards_dir) / args.model_name if args.model_name is not None else args.cards_dir
    )
    model.renderCombine(out_dir)

    with Path(f"{out_dir}/model.pkl").open("wb") as fout:
        pickle.dump(model, fout, 2)  # use python 2 compatible protocol


def main(args):
    # templates per region per year, templates per region summed across years
    templates_dict, templates_summed = get_templates(
        args.templates_dir, years, args.sig_separate, args.scale_templates
    )

    # # TODO: check if / how to include signal trig eff uncs. (rn only using bg uncs.)
    # process_systematics(args.templates_dir, args.sig_separate)

    # random template from which to extract shape vars
    sample_templates: Hist = templates_summed[next(iter(templates_summed.keys()))]

    # [mH(bb)]
    shape_vars = [
        ShapeVar(
            name=axis.name,
            bins=axis.edges,
            orders={sr: args.nTF[i] for i, sr in enumerate(signal_regions)},
        )
        for _, axis in enumerate(sample_templates.axes[1:])
    ]

    Path(args.cards_dir).mkdir(parents=True, exist_ok=True)
    createDatacardAlphabet(args, templates_dict, templates_summed, shape_vars)


main(args)
