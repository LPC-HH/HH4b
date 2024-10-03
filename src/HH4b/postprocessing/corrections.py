from __future__ import annotations

import json
from pathlib import Path

import correctionlib
import hist
import numpy as np
import pandas as pd
from coffea.lookup_tools.dense_lookup import dense_lookup
from numpy.typing import ArrayLike

package_path = Path(__file__).parent.parent.resolve()


def _load_txbb_sfs(year: str, fname: str, txbb_wps: dict[str:list], pt_bins: dict[str:list]):
    """Create 2D lookup tables in [Txbb, pT] for Txbb SFs from given year"""

    with (package_path / f"corrections/data/txbb_sfs/{year}/{fname}.json").open() as f:
        txbb_sf = json.load(f)

    txbb_bins = np.array([txbb_wps[wp][0] for wp in txbb_wps] + [1])
    pt_fine_bins = np.unique(np.concatenate([pt_bins[wp] for wp in pt_bins]))
    edges = (txbb_bins, pt_fine_bins)
    keys = [
        ("final", "central"),
        ("final", "high"),
        ("final", "low"),
        ("stats", "high"),
        ("stats", "low"),
    ]
    vals = {key: [] for key in keys}

    for key1, key2 in keys:
        for wp in txbb_wps:
            wval = []
            for low_fine, high_fine in zip(pt_fine_bins[:-1], pt_fine_bins[1:]):
                for low, high in zip(pt_bins[wp][:-1], pt_bins[wp][1:]):
                    if low_fine >= low and high_fine <= high:
                        wval.append(txbb_sf[f"{wp}_pt{low}to{high}"][key1][key2])
                        break
            vals[key1, key2].append(wval)
    vals = {key: np.array(val) for key, val in list(vals.items())}

    corr_err_high = np.sqrt(np.maximum(vals["final", "high"] ** 2 - vals["stats", "high"] ** 2, 0))
    corr_err_low = np.sqrt(np.maximum(vals["final", "low"] ** 2 - vals["stats", "low"] ** 2, 0))

    txbb_sf = {
        "nominal": dense_lookup(vals["final", "central"], edges),
        "stat_up": dense_lookup(vals["final", "central"] + vals["stats", "high"], edges),
        "stat_dn": dense_lookup(vals["final", "central"] - vals["stats", "low"], edges),
        "stat3x_up": dense_lookup(vals["final", "central"] + 3 * vals["stats", "high"], edges),
        "stat3x_dn": dense_lookup(vals["final", "central"] - 3 * vals["stats", "low"], edges),
        "corr_up": dense_lookup(vals["final", "central"] + corr_err_high, edges),
        "corr_dn": dense_lookup(vals["final", "central"] - corr_err_low, edges),
        "corr3x_up": dense_lookup(vals["final", "central"] + 3 * corr_err_high, edges),
        "corr3x_dn": dense_lookup(vals["final", "central"] - 3 * corr_err_low, edges),
    }

    return txbb_sf


def restrict_SF(
    lookup: dense_lookup,
    txbb: ArrayLike,
    pt: ArrayLike,
    txbb_input_range: ArrayLike | None = None,
    pt_input_range: ArrayLike | None = None,
    lookup_right: dense_lookup | None = None,
    txbb_interp_range: ArrayLike | None = None,
):
    """Apply txbb scale factors"""
    sf = lookup(txbb, pt)
    if lookup_right is not None and txbb_interp_range is not None:
        sf_left = lookup(txbb, pt)
        sf_right = lookup_right(txbb, pt)
        mask = (txbb > txbb_interp_range[0]) & (txbb < txbb_interp_range[1])
        sf[mask] = sf_left[mask] + (sf_right[mask] - sf_left[mask]) * (
            txbb[mask] - txbb_interp_range[0]
        ) / (txbb_interp_range[1] - txbb_interp_range[0])
    if txbb_input_range is not None:
        sf[txbb < txbb_input_range[0]] = 1.0
        sf[txbb > txbb_input_range[1]] = 1.0
    if pt_input_range is not None:
        sf[pt < pt_input_range[0]] = 1.0
        sf[pt > pt_input_range[1]] = 1.0
    return sf


def _load_ttbar_sfs(year: str, corr: str):
    year_ = None
    if "2022" in year:
        year_ = "2022"
    elif "2023" in year:
        year_ = "2023"
    return correctionlib.CorrectionSet.from_file(
        f"{package_path}/corrections/data/ttbarcorr_{year_}.json"
    )[f"ttbar_corr_{corr}_{year_}"]


def _load_ttbar_bdtshape_sfs(cat: str, bdt_model: str):
    return correctionlib.CorrectionSet.from_file(
        f"{package_path}/corrections/data/ttbar_sfs/{bdt_model}/ttbar_bdtshape{cat}_2022-2023.json"
    )["ttbar_corr_bdtshape_2022-2023"]


def ttbar_SF(
    tt_sf: correctionlib.CorrectionSet,
    events_dict: dict[str, pd.DataFrame],
    branch: str,
    input_range: ArrayLike | None = None,
):
    # if input is outside of input_range, set correction to 1
    """Apply ttbar scale factors"""
    input_var = events_dict[branch]

    sfs = {}
    for syst in ["nominal", "stat_up", "stat_dn"]:
        sfs[syst] = tt_sf.evaluate(input_var, "nominal")
        if syst == "stat_up":
            sfs[syst] += tt_sf.evaluate(input_var, syst)
        elif syst == "stat_dn":
            sfs[syst] -= tt_sf.evaluate(input_var, syst)
        # replace zeros or negatives with 1
        sfs[syst][sfs[syst] <= 0] = 1.0
        # if input is outside of (defined) input_range, set to 1
        if input_range is not None:
            sfs[syst][input_var < input_range[0]] = 1.0
            sfs[syst][input_var > input_range[1]] = 1.0

    return sfs["nominal"], sfs["stat_up"], sfs["stat_dn"]


def _load_trig_effs(year: str, label: str, region: str):
    return correctionlib.CorrectionSet.from_file(
        f"{package_path}/corrections/data/fatjet_triggereff_{year}_{label}_{region}.json"
    )


def trigger_SF(
    year: str, events_dict: dict[str, pd.DataFrame], pnet_str: str, region: str, legacy: bool = True
):
    """
    Evaluate trigger Scale Factors
    """

    # hard code axes
    if "2023" in year:
        pt_range = [
            0.0,
            10.0,
            20.0,
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
            160.0,
            170.0,
            180.0,
            190.0,
            200.0,
            210.0,
            220.0,
            230.0,
            240.0,
            250.0,
            260.0,
            270.0,
            280.0,
            290.0,
            300.0,
            320.0,
            340.0,
            360.0,
            380.0,
            400.0,
            420.0,
            440.0,
            460.0,
            480.0,
            500.0,
            550.0,
            600.0,
            700.0,
            800.0,
            1000.0,
        ]
        xbb_range = [
            0.0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            1.0,
        ]
    else:
        pt_range = [
            0.0,
            30.0,
            60.0,
            90.0,
            120.0,
            150.0,
            180.0,
            210.0,
            240.0,
            270.0,
            300.0,
            360.0,
            420.0,
            480.0,
            600.0,
            1000.0,
        ]
        xbb_range = [
            0.0,
            0.04,
            0.08,
            0.12,
            0.16,
            0.2,
            0.24,
            0.28,
            0.32,
            0.36,
            0.4,
            0.44,
            0.48,
            0.52,
            0.56,
            0.6,
            0.64,
            0.68,
            0.72,
            0.76,
            0.8,
            0.84,
            0.88,
            0.92,
            0.96,
            1.0,
        ]
    xbbv11_range = [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]
    msd_range = [
        0.0,
        5.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        80.0,
        100.0,
        120.0,
        150.0,
        200.0,
        250.0,
        300.0,
        350.0,
    ]

    xbb_axis = hist.axis.Variable(xbbv11_range if legacy else xbb_range, name="xbb")
    pt_axis = hist.axis.Variable(pt_range, name="pt")
    msd_axis = hist.axis.Variable(msd_range, name="msd")
    # load trigger efficiencies
    triggereff_ptmsd = _load_trig_effs(year, "ptmsd", region)
    txbb = "txbbv11" if legacy else "txbb"
    triggereff_btag = _load_trig_effs(year, txbb, region)
    eff_data = triggereff_ptmsd[f"fatjet_triggereffdata_{year}_ptmsd"]
    eff_mc = triggereff_ptmsd[f"fatjet_triggereffmc_{year}_ptmsd"]
    eff_data_btag = triggereff_btag[f"fatjet_triggereffdata_{year}_{txbb}"]
    eff_mc_btag = triggereff_btag[f"fatjet_triggereffmc_{year}_{txbb}"]

    # efficiencies per jet
    eff_data_per_jet = {}
    eff_mc_per_jet = {}

    # weight (no trigger SF)
    weight = events_dict["finalWeight"]

    # yield histogram
    totals = []
    total_errs = []

    # iterate over jets
    for jet in range(2):
        pt = events_dict["bbFatJetPt"][jet]
        msd = events_dict["bbFatJetMsd"][jet]
        xbb = events_dict[f"bbFatJet{pnet_str}"][jet]

        num_ev = pt.shape[0]

        # TODO: add matching to trigger objects
        # for now, assuming both are matched
        matched = np.ones(num_ev)

        eff_data_per_jet[jet] = {}
        eff_mc_per_jet[jet] = {}

        for var in ["nominal", "stat_up"]:
            eff_data_val = np.zeros(num_ev)
            eff_data_btag_val = np.zeros(num_ev)
            eff_mc_val = np.zeros(num_ev)
            eff_mc_btag_val = np.zeros(num_ev)

            eff_data_all = eff_data.evaluate(pt, msd, var)
            eff_data_btag_all = eff_data_btag.evaluate(xbb, var)
            eff_mc_all = eff_mc.evaluate(pt, msd, var)
            eff_mc_btag_all = eff_mc_btag.evaluate(xbb, var)

            # replace zeros (!) should belong to unmatched...
            if var == "nominal":
                eff_data_all[eff_data_all == 0] = 1.0
                eff_data_btag_all[eff_data_btag_all == 0] = 1.0
                eff_mc_all[eff_mc_all == 0] = 1.0
                eff_mc_btag_all[eff_mc_btag_all == 0] = 1.0

            eff_data_val[matched == 1] = eff_data_all[matched == 1]
            eff_data_btag_val[matched == 1] = eff_data_btag_all[matched == 1]
            eff_mc_val[matched == 1] = eff_mc_all[matched == 1]
            eff_mc_btag_val[matched == 1] = eff_mc_btag_all[matched == 1]

            eff_data_per_jet[jet][var] = eff_data_val * eff_data_btag_val
            eff_mc_per_jet[jet][var] = eff_mc_val * eff_mc_btag_val

        sf_per_jet = eff_data_per_jet[jet]["nominal"] / eff_mc_per_jet[jet]["nominal"]
        sf_err_per_jet = sf_per_jet * np.sqrt(
            (eff_data_per_jet[jet]["stat_up"] / eff_data_per_jet[jet]["nominal"]) ** 2
            + (eff_mc_per_jet[jet]["stat_up"] / eff_mc_per_jet[jet]["nominal"]) ** 2
        )
        h_yield = hist.Hist(pt_axis, msd_axis, xbb_axis)
        h_yield_err = hist.Hist(pt_axis, msd_axis, xbb_axis)
        h_yield.fill(pt, msd, xbb, weight=weight * sf_per_jet)
        h_yield_err.fill(pt, msd, xbb, weight=weight * sf_err_per_jet)

        total = np.sum(h_yield.values(flow=True))
        totals.append(total)
        total_err = np.linalg.norm(np.nan_to_num(h_yield_err.values(flow=True)))
        total_errs.append(total_err)

    """
    fill histogram with the yields, with the same binning as the efficiencies,
    then take the product of that histogram * the efficiencies and * the errors
    """
    total = np.sum(totals)
    total_err = np.linalg.norm(total_errs)

    tot_eff_data = 1 - (1 - eff_data_per_jet[0]["nominal"]) * (1 - eff_data_per_jet[1]["nominal"])
    tot_eff_mc = 1 - (1 - eff_mc_per_jet[0]["nominal"]) * (1 - eff_mc_per_jet[1]["nominal"])

    if np.any(tot_eff_data == 0):
        print("Warning: eff data has 0 values")
    if np.any(tot_eff_mc == 0):
        print("Warning: eff mc has 0 values")

    sf = tot_eff_data / tot_eff_mc

    # unc on eff: (1 - z): dz
    # z = x * y = (1-eff_1)(1-eff_2)
    # dz = z * sqrt( (dx/x)**2 + (dy/y)**2 )
    for var in ["up"]:
        dx_data = eff_data_per_jet[0][f"stat_{var}"]
        dy_data = eff_data_per_jet[1][f"stat_{var}"]
        x_data = 1 - eff_data_per_jet[0]["nominal"]
        y_data = 1 - eff_data_per_jet[1]["nominal"]
        z = x_data * y_data
        dz = z * np.sqrt((dx_data / x_data) ** 2 + (dy_data / y_data) ** 2)
        unc_eff_data = dz

        dx_mc = eff_mc_per_jet[0][f"stat_{var}"]
        dy_mc = eff_mc_per_jet[1][f"stat_{var}"]
        x_mc = 1 - eff_mc_per_jet[0]["nominal"]
        y_mc = 1 - eff_mc_per_jet[1]["nominal"]
        z = x_mc * y_mc
        dz = z * np.sqrt((dx_mc / x_mc) ** 2 + (dy_mc / y_mc) ** 2)
        unc_eff_mc = dz

        unc_sf = sf * np.sqrt((unc_eff_data / tot_eff_data) ** 2 + (unc_eff_mc / tot_eff_mc) ** 2)

    return sf, unc_sf, total, total_err
