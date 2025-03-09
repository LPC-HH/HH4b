from __future__ import annotations

import logging
from dataclasses import dataclass, field

import hist
import numpy as np
from hist import Hist
from rhalphalib import MorphHistW2

from HH4b.hh_vars import sig_keys_ggf, sig_keys_vbf
from HH4b.hh_vars import years as all_years

#################################################
# Common
#################################################


@dataclass
class Syst:
    """For storing info about systematics"""

    name: str = None
    prior: str = None  # e.g. "lnN", "shape", etc.

    # float if same value in all regions/samples, dictionary of values per region/sample if not
    # if both, region should be the higher level of the dictionary
    value: float | dict[str, float] = None
    value_down: float | dict[str, float] = None  # if None assumes symmetric effect

    # if the value is different for different regions or samples
    diff_regions: bool = False
    diff_samples: bool = False

    samples: list[str] = None  # samples affected by it
    samples_corr: bool = True  # if it's correlated between samples
    separate_prod_modes: bool = False  # if it's uncorrelated between ggF and VBF
    # in case of uncorrelated unc., which years to split into
    # dictionary of label -> list of years to keep correlated
    uncorr_years: dict[str, list[str]] = field(
        default_factory=lambda: {year: [year] for year in all_years}
    )
    pass_only: bool = False  # is it applied only in the pass regions
    convert_shape_to_lnN: bool = False  # take shape uncertainty and convert to lnN
    decorrelate_regions: bool = False  # decorrelate in each region

    def __post_init__(self):
        if isinstance(self.value, dict) and not (self.diff_regions or self.diff_samples):
            raise RuntimeError(
                "Value for systematic is a dictionary but neither ``diff_regions`` nor ``diff_samples`` is set."
            )

        if self.decorrelate_regions and self.separate_prod_modes:
            raise NotImplementedError(
                "Decorrelated regions and separate production modes not implemented yet!"
            )


@dataclass
class ShapeVar:
    """For storing and calculating info about variables used in fit"""

    name: str = None
    bins: np.ndarray = None  # bin edges
    orders: dict = None  # TF order: dict of categories -> order

    def __post_init__(self):
        # use bin centers for polynomial fit
        self.pts = self.bins[:-1] + 0.5 * np.diff(self.bins)
        # scale to be between [0, 1]
        self.scaled = (self.pts - self.bins[0]) / (self.bins[-1] - self.bins[0])


def add_bool_arg(parser, name, help, default=False, no_name=None):
    """Add a boolean command line argument for argparse"""
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


#################################################
# Template manipulation
#################################################


def rem_neg(template_dict: dict):
    for _, template in template_dict.items():
        template.values()[template.values() < 0] = 0

    return template_dict


def sum_templates(template_dict: dict, years: list[str]):
    """Sum templates across years"""

    ttemplate = next(iter(template_dict.values()))  # sample templates from which to extract values
    combined = {}

    for region in ttemplate:
        thists = []

        for year in years:
            thists.append(template_dict[year][region])

        combined[region] = sum(thists)

    return combined


def combine_templates(
    bg_templates: dict[str, Hist], sig_templates: list[dict[str, Hist]]
) -> dict[str, Hist]:
    """
    Combines BG and signal templates into a single Hist (per region).

    Args:
        bg_templates (Dict[str, Hist]): dictionary of region -> Hist
        sig_templates (List[Dict[str, Hist]]): list of dictionaries of region -> Hist for each
          signal samples
    """
    ctemplates = {}

    for region, bg_template in bg_templates.items():
        # combined sig + bg samples
        csamples = list(bg_template.axes[0]) + [
            s for sig_template in sig_templates for s in list(sig_template[region].axes[0])
        ]

        # new hist with all samples
        ctemplate = Hist(
            hist.axis.StrCategory(csamples, name="Sample"),
            *bg_template.axes[1:],
            storage="weight",
        )

        # add background hists
        for sample in bg_template.axes[0]:
            sample_key_index = np.where(np.array(list(ctemplate.axes[0])) == sample)[0][0]
            ctemplate.view(flow=True)[sample_key_index, ...] = bg_template[sample, ...].view(
                flow=True
            )

        # add signal hists
        for st in sig_templates:
            sig_template = st[region]
            for sample in sig_template.axes[0]:
                sample_key_index = np.where(np.array(list(ctemplate.axes[0])) == sample)[0][0]
                ctemplate.view(flow=True)[sample_key_index, ...] = sig_template[sample, ...].view(
                    flow=True
                )

        ctemplates[region] = ctemplate

    return ctemplates


#################################################
# Shape Fills
#################################################


def _shape_checks(values_up, values_down, values_nominal, effect_up, effect_down, logger):
    norm_up = np.sum(values_up)
    norm_down = np.sum(values_down)
    norm_nominal = np.sum(values_nominal)
    prob_up = values_up / norm_up
    prob_down = values_down / norm_down
    prob_nominal = values_nominal / norm_nominal
    shapeEffect_up = np.sum(
        np.abs(prob_up - prob_nominal) / (np.abs(prob_up) + np.abs(prob_nominal))
    )
    shapeEffect_down = np.sum(
        np.abs(prob_down - prob_nominal) / (np.abs(prob_down) + np.abs(prob_nominal))
    )

    if np.allclose(effect_up, 1.0) and np.allclose(effect_down, 1.0):
        logger.warning("No shape effect")
    elif np.allclose(effect_up, effect_down):
        logger.warning("Up is the same as Down, but different from nominal")
    elif np.allclose(effect_up, 1.0) or np.allclose(effect_down, 1.0):
        logger.warning("Up or Down is the same as nominal (one-sided)")
    elif shapeEffect_up < 0.001 and shapeEffect_down < 0.001:
        logger.warning("No genuine shape effect (just norm)")
    elif (norm_up > norm_nominal and norm_down > norm_nominal) or (
        norm_up < norm_nominal and norm_down < norm_nominal
    ):
        logger.warning("Up and Down vary norm in the same direction")


def get_effect_updown(
    values_nominal, values_up, values_down, mask, logger, epsilon, convert_shape_to_lnN=False
):
    if convert_shape_to_lnN:
        effect_up = np.sum(values_up) / np.sum(values_nominal)
        effect_down = np.sum(values_down) / np.sum(values_nominal)
        logging.debug(f"effect_up  : {effect_up}")
        logging.debug(f"effect_down: {effect_down}")
        return effect_up, effect_down

    effect_up = np.ones_like(values_nominal)
    effect_down = np.ones_like(values_nominal)

    mask_up = mask & (values_up >= 0)
    mask_down = mask & (values_down >= 0)

    effect_up[mask_up] = values_up[mask_up] / values_nominal[mask_up]
    effect_down[mask_down] = values_down[mask_down] / values_nominal[mask_down]

    zero_up = values_up == 0
    zero_down = values_down == 0

    effect_up[mask_up & zero_up] = values_nominal[mask_up & zero_up] * epsilon
    effect_down[mask_down & zero_down] = values_nominal[mask_down & zero_down] * epsilon

    _shape_checks(values_up, values_down, values_nominal, effect_up, effect_down, logger)

    logging.debug(f"nominal   : {values_nominal}")
    logging.debug(f"effect_up  : {effect_up}")
    logging.debug(f"effect_down: {effect_down}")

    return effect_up, effect_down


def smass(sName):
    if sName in sig_keys_ggf + sig_keys_vbf:
        _mass = 125.0
    elif sName in ["vhtobb", "diboson"]:
        _mass = 80.379  # use W mass instead of Z mass = 91.
        # TODO: split W/Z processes?
    else:
        raise ValueError(f"What is {sName}")
    return _mass


def smorph(templ, sample_name, jms_value, jmr_value):
    if templ is None:
        return None

    for sample_check in sig_keys_ggf + sig_keys_vbf + ["vhtobb", "diboson"]:
        if sample_check in sample_name:
            return MorphHistW2(templ).get(
                shift=(jms_value - 1.0) * smass(sample_check), smear=jmr_value
            )
    return templ
