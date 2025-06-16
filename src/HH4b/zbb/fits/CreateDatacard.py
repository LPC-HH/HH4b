"""
Creates datacards for Higgs Combine using hist.Hist templates output from PostProcess.py
(1) Adds systematics for all samples,
(2) Sets up data-driven QCD background estimate ('rhalphabet' method)

Modified for Zbb analysis to extract TXbb scale factors

Based on https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
and https://github.com/LPC-HH/HH4b/blob/main/src/HH4b/postprocessing/CreateDatacard.py

Authors: Raghav Kansal, Zichun Hao
"""

# ruff: noqa: PLC0206

from __future__ import annotations

# from utils import add_bool_arg
import argparse
import itertools
import logging
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rhalphalib as rl
from hist import Hist

from HH4b.hh_vars import (
    LUMI,
    data_key,
    jecs,
    jmsr,
    qcd_key,
)
from HH4b.postprocessing.datacardHelpers import (
    ShapeVar,
    Syst,
    add_bool_arg,
    get_effect_updown,
    rem_neg,
    sum_templates,
)

# TXbb scale factor measurement binning
WPS = [0.95, 0.975, 0.99, 1.0]
PTS = [350, 450, 550, 10000]
# PTS = [350, 450, 500, 550, 10000]

pt_strs = [str(pt) for pt in PTS]
pt_bins = list(zip(pt_strs[:-1], pt_strs[1:]))

wp_strs = [str(wp).replace(".", "p") for wp in WPS]
wp_bins = list(zip(wp_strs[:-1], wp_strs[1:]))

ALL_PASS_REGIONS = [
    f"pass_TXbb{wp_low}to{wp_high}_pT{pt_low}to{pt_high}"
    for (pt_low, pt_high), (wp_low, wp_high) in itertools.product(pt_bins, wp_bins)
]
print(f"ALL_PASS_REGIONS: {ALL_PASS_REGIONS}")

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
    required=True,
    type=str,
    help="input pickle file of dict of hist.Hist templates",
)

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

parser.add_argument(
    "--sig-samples",
    default=["Zto2Q_BB"],
    nargs="*",
    type=str,
    help="specify signals",
)

parser.add_argument(
    "--nTF",
    default=None,
    type=int,
    help="order of polynomial for TF. Default is 0",
)
parser.add_argument(
    "--regions",
    default="all",
    type=str,
    help="regions for which to make cards",
    # choices=["pass", "all"],
)
parser.add_argument("--model-name", default=None, type=str, help="output model name")
parser.add_argument(
    "--year",
    type=str,
    required=True,
    choices=["2022All", "2023All"],
    help="years to make datacards for",
)
add_bool_arg(parser, "mcstats", "add mc stats nuisances", default=True)
add_bool_arg(parser, "bblite", "use barlow-beeston-lite method", default=True)
add_bool_arg(parser, "ttbar-rate-param", "Add freely floating ttbar rate param", default=False)
add_bool_arg(
    parser,
    "mc-closure",
    "Perform MC closure test (fill data_obs with sum of MC bkg.",
    default=False,
)
add_bool_arg(parser, "jmsr", "Do JMS/JMR uncertainties", default=True)
add_bool_arg(parser, "jesr", "Do JES/JER uncertainties", default=True)
args = parser.parse_args()


CMS_PARAMS_LABEL = "CMS_bbbb_hadronic"
MCB_LABEL = "MCBlinded"
qcd_data_key = "qcd_datadriven"

print("Transfer factors:", args.nTF)

signal_regions = ALL_PASS_REGIONS if args.regions == "all" else [args.regions]

if args.regions == "all":
    args.nTF = [args.nTF for _ in ALL_PASS_REGIONS]
else:
    args.nTF = [args.nTF]

print(f"nTF orders: {args.nTF}")

# (name in templates, name in cards)
mc_samples = OrderedDict(
    [
        ("ttbar", "ttbar"),
        ("hbb", "hbb"),
        ("Wto2Q", "Wto2Q"),
        ("Zto2Q_CC", "Zto2Q_CC"),
        ("Zto2Q_QQ", "Zto2Q_QQ"),
        ("Zto2Q_unmatched", "Zto2Q_unmatched"),
    ]
)

mc_samples_sig = OrderedDict(
    [
        ("Zto2Q_BB", "Zto2Q_BB"),
    ]
)

bg_keys = list(mc_samples.keys())
all_sig_keys = ["Zto2Q_BB"]
sig_keys = []
hist_names = {}  # names of hist files for the samples

for key in all_sig_keys:
    # check in case single sig sample is specified
    if args.sig_samples is None or key in args.sig_samples:
        # change names to match combination convention
        mc_samples[key] = mc_samples_sig[key]
        sig_keys.append(key)

all_mc = list(mc_samples.keys())

years = [args.year]
full_lumi = LUMI[args.year]

# Include all samples for JMS/JMR uncertainties
jmsr_keys = list(mc_samples.keys()) + list(mc_samples_sig.keys())
# Remove duplicates
jmsr_keys = list(dict.fromkeys(jmsr_keys))

# dictionary of nuisance params -> (modifier, samples affected by it, value)
nuisance_params = {
    "pdf_gg": Syst(prior="lnN", samples=["ttbar"], value=1.042),
    "QCDscale_ttbar": Syst(
        prior="lnN",
        samples=["ttbar"],
        value=1.024,
        value_down=0.965,
    ),
    # weight lumi uncertainties by corresponding integrated lumi
    "lumi_2022": Syst(
        prior="lnN", samples=all_mc, value=1 + 0.014 * LUMI["2022All"] / LUMI["2022-2023"]
    ),
    "lumi_2023": Syst(
        prior="lnN", samples=all_mc, value=1 + 0.013 * LUMI["2023All"] / LUMI["2022-2023"]
    ),
}

if args.year == "2022All":
    del nuisance_params["lumi_2023"]
    uncorr_years = {
        "2022All": ["2022All"],
    }
elif args.year == "2023All":
    del nuisance_params["lumi_2022"]
    uncorr_years = {
        "2023All": ["2023All"],
    }
else:
    raise ValueError(f"Invalid year {args.year}, must be one of ['2022All', '2023All']")

rate_params = {}
if args.ttbar_rate_param:
    rate_params = {
        "ttbar": {
            region: rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_rp_ttbar_{region}", 1.0, 0, 10)
            for region in signal_regions
        }
    }

nuisance_params_dict = {
    param: rl.NuisanceParameter(param, syst.prior) for param, syst in nuisance_params.items()
}

# dictionary of correlated shape systematics: name in templates -> name in cards, etc.
corr_year_shape_systs = {
    "JES": Syst(name="CMS_scale_j", prior="shape", samples=all_mc),
    "FSRPartonShower": Syst(name="ps_fsr", prior="shape", samples=sig_keys, samples_corr=True),
    "ISRPartonShower": Syst(name="ps_isr", prior="shape", samples=sig_keys, samples_corr=True),
}

uncorr_year_shape_systs = {
    "pileup": Syst(
        name="CMS_pileup",
        prior="shape",
        samples=all_mc,
        uncorr_years=uncorr_years,
    ),
    "JER": Syst(
        name="CMS_res_j",
        prior="shape",
        samples=all_mc,
        convert_shape_to_lnN=True,
        uncorr_years=uncorr_years,
    ),
    "JMS": Syst(
        name=f"{CMS_PARAMS_LABEL}_jms",
        prior="shape",
        samples=jmsr_keys,
        uncorr_years=uncorr_years,
    ),
    "JMR": Syst(
        name=f"{CMS_PARAMS_LABEL}_jmr",
        prior="shape",
        samples=jmsr_keys,
        uncorr_years=uncorr_years,
    ),
}

if not args.jmsr:
    del uncorr_year_shape_systs["JMR"]
    del uncorr_year_shape_systs["JMS"]

if not args.jesr:
    del corr_year_shape_systs["JES"]
    del uncorr_year_shape_systs["JER"]

if args.ttbar_rate_param:
    # remove all ttbarSF systematics
    for key in list(corr_year_shape_systs.keys()):
        if "ttbarSF" in key:
            del corr_year_shape_systs[key]

shape_systs_dict = {}
for skey, syst in corr_year_shape_systs.items():
    if not syst.samples_corr:
        # separate nuisance param for each affected sample
        for sample in syst.samples:
            if sample not in mc_samples:
                continue
            shape_systs_dict[f"{skey}_{sample}"] = rl.NuisanceParameter(
                f"{syst.name}_{mc_samples[sample]}", "lnN" if syst.convert_shape_to_lnN else "shape"
            )
    elif syst.decorrelate_regions:
        # separate nuisance param for each region
        for region in signal_regions + ["fail"]:
            shape_systs_dict[f"{skey}_{region}"] = rl.NuisanceParameter(
                f"{syst.name}_{region}", "lnN" if syst.convert_shape_to_lnN else "shape"
            )
    else:
        shape_systs_dict[skey] = rl.NuisanceParameter(
            syst.name, "lnN" if syst.convert_shape_to_lnN else "shape"
        )
for skey, syst in uncorr_year_shape_systs.items():
    for uncorr_label in syst.uncorr_years:
        shape_systs_dict[f"{skey}_{uncorr_label}"] = rl.NuisanceParameter(
            f"{syst.name}_{uncorr_label}", "lnN" if syst.convert_shape_to_lnN else "shape"
        )


def get_templates(
    templates_dir: str,
    years: list[str],
    scale: float | None = None,
):
    """Loads templates, combines bg and sig templates if separate, sums across all years"""
    templates_dict: dict[str, dict[str, Hist]] = {}

    for year in years:
        with Path(f"{templates_dir}/templates_{year}.pkl").open("rb") as f:
            templates_dict[year] = rem_neg(pickle.load(f))

    if scale is not None and scale != 1:
        for year in templates_dict:
            for key in templates_dict[year]:
                templates_dict[year][key] = templates_dict[year][key] * scale

    templates_summed: dict[str, Hist] = sum_templates(templates_dict, years)  # sum across years
    return templates_dict, templates_summed


def get_year_updown(templates_dict, sample, region, region_noblinded, blind_str, skey):
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
        for year in years:
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
        if region not in templates_summed:
            raise ValueError(
                f"Region {region} not found in templates_summed! Valid regions: {list(templates_summed.keys())}"
            )
        region_templates = templates_summed[region]

        pass_region = region.startswith("pass")
        region_noblinded = region
        blind_str = ""

        logging.info(f"starting region: {region}")
        ch = rl.Channel(region.replace("_", ""))  # can't have '_'s in name
        model.addChannel(ch)

        for sample_name, card_name in mc_samples.items():
            # don't add signals in fail regions
            if sample_name in sig_keys and not pass_region:
                logging.info(f"\nSkipping {sample_name} in {region} region\n")
                continue

            logging.info(f"get templates for: {sample_name}")

            sample_template = region_templates[sample_name, :]

            stype = rl.Sample.SIGNAL if sample_name in sig_keys else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + card_name, stype, sample_template)

            # ttbar rate_param
            if sample_name in rate_params and region_noblinded in rate_params[sample_name]:
                rate_param = rate_params[sample_name][region_noblinded]
                sample.setParamEffect(rate_param, 1 * rate_param)

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

                # tie MC stats parameters together in "unblinded" region
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
                    up_hist = templates_summed[f"{region_noblinded}_{skey}_up"][sample_name, :]
                    down_hist = templates_summed[f"{region_noblinded}_{skey}_down"][sample_name, :]

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
                if not syst.samples_corr:
                    # separate syst if not correlated across samples
                    sdkey = f"{skey}_{sample_name}"
                elif syst.decorrelate_regions:
                    # separate syst if not correlated across regions
                    sdkey = f"{skey}_{region_noblinded}"
                else:
                    sdkey = skey
                sample.setParamEffect(shape_systs_dict[sdkey], effect_up, effect_down)

            # uncorrelated shape systematics
            for skey, syst in uncorr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"Getting {skey} shapes")

                for uncorr_label in syst.uncorr_years:

                    values_up, values_down = get_year_updown(
                        templates_dict,
                        sample_name,
                        region,
                        region_noblinded,
                        blind_str,
                        skey,
                    )
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
                    sample.setParamEffect(
                        shape_systs_dict[f"{skey}_{uncorr_label}"], effect_up, effect_down
                    )

            ch.addSample(sample)

        if bblite and args.mcstats:
            # tie MC stats parameters together in "unblinded" region
            channel_name = region_noblinded
            ch.autoMCStats(
                channel_name=f"{CMS_PARAMS_LABEL}_{channel_name}",
                threshold=args.mcstats_threshold,
                epsilon=args.epsilon,
            )

        # data observed
        if args.mc_closure:
            all_bg = sum([region_templates[bg_key, :] for bg_key in bg_keys + [qcd_key]])
            ch.setObservation(all_bg)
        else:
            ch.setObservation(region_templates[data_key, :])


def alphabet_fit(
    model: rl.Model,
    shape_vars: list[ShapeVar],
    templates_summed: dict,
    scale: float | None = None,
    min_qcd_val: float | None = None,
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

    blind_strs = [""]
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
            templates_summed[sr][data_key, :].sum().value
            - np.sum([templates_summed[sr][bg_key, :].sum().value for bg_key in bg_keys])
        ) / (
            templates_summed["fail"][data_key, :].sum().value
            - np.sum([templates_summed["fail"][bg_key, :].sum().value for bg_key in bg_keys])
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
    blind_strs = [""]

    regions: list[str] = [
        f"{pf}{blind_str}" for pf in [*signal_regions, "fail"] for blind_str in blind_strs
    ]

    # build actual fit model now
    model = rl.Model("ZbbModel")

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
        args.templates_dir, years, args.scale_templates
    )

    # random template from which to extract shape vars
    sample_templates: Hist = templates_summed[next(iter(templates_summed.keys()))]

    # [mH(bb)] or whatever the observable is
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
