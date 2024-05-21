from __future__ import annotations

import correctionlib
import numpy as np
import pandas as pd


def ttbar_SF(year: str, events_dict: dict[str, pd.DataFrame], corr: str, branch: str):
    # corr: PTJJ, Tau3OverTau2, Xbb
    # branch: HHPt, H1T32, H2T32, H1TXbb, H2TXbb
    """Apply ttbar scale factors"""
    year_ = None
    if "2022" in year:
        year_ = "2022"
    elif "2023" in year:
        year_ = "2023"
    tt_sf = correctionlib.CorrectionSet.from_file(f"../corrections/data/ttbarcorr_{year_}.json")[
        corr
    ]
    input_var = events_dict[branch]
    sfs = tt_sf.evaluate(input_var)
    # replace zeros with 1
    sfs[sfs == 0] = 1.0

    return sfs


def _load_trig_effs(year: str, label: str, region: str):
    return correctionlib.CorrectionSet.from_file(
        f"../corrections/data/fatjet_triggereff_{year}_{label}_{region}.json"
    )


def trigger_SF(year: str, events_dict: dict[str, pd.DataFrame], pnet_str: str, region: str):
    """
    Evaluate trigger Scale Factors
    """

    triggereff_ptmsd = _load_trig_effs(year, "ptmsd", region)
    # triggereff_btag = _load_trig_effs(year, "txbb", region)
    triggereff_btag = _load_trig_effs(year, "txbbv11", region)

    eff_data = triggereff_ptmsd[f"fatjet_triggereffdata_{year}_ptmsd"]
    eff_mc = triggereff_ptmsd[f"fatjet_triggereffmc_{year}_ptmsd"]
    eff_data_btag = triggereff_btag[f"fatjet_triggereffdata_{year}_txbbv11"]
    eff_mc_btag = triggereff_btag[f"fatjet_triggereffmc_{year}_txbbv11"]

    eff_data_per_jet = {}
    eff_mc_per_jet = {}

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

        for var in ["nominal", "stat_up", "stat_dn"]:
            eff_data_val = np.zeros(num_ev)
            eff_data_btag_val = np.zeros(num_ev)
            eff_mc_val = np.zeros(num_ev)
            eff_mc_btag_val = np.zeros(num_ev)

            eff_data_all = eff_data.evaluate(pt, msd, var)
            eff_data_btag_all = eff_data_btag.evaluate(xbb, var)
            eff_mc_all = eff_mc.evaluate(pt, msd, var)
            eff_mc_btag_all = eff_mc_btag.evaluate(xbb, var)

            # replace zeros (!) should belong to unmatched...
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

    return sf, sf + unc_sf, sf - unc_sf
