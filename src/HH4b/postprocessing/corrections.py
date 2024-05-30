from __future__ import annotations

import correctionlib
import numpy as np
import pandas as pd
import hist

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


def trigger_SF(year: str, events_dict: dict[str, pd.DataFrame], pnet_str: str, region: str, legacy: bool = True):
    """
    Evaluate trigger Scale Factors
    """

    # hard code axes
    if "2023" in year:
        pt_range = [   0.,   10.,   20.,   30.,   40.,   50.,   60.,   70.,   80.,
                       90.,  100.,  110.,  120.,  130.,  140.,  150.,  160.,  170.,
                       180.,  190.,  200.,  210.,  220.,  230.,  240.,  250.,  260.,
                       270.,  280.,  290.,  300.,  320.,  340.,  360.,  380.,  400.,
                       420.,  440.,  460.,  480.,  500.,  550.,  600.,  700.,  800.,
                       1000.]
        xbb_range = [0., 0.05, 0.1, 0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,  0.55, 0.6,  0.65,
                     0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1.  ]
    else:
        pt_range = [   0.,   30.,   60.,   90.,  120.,  150.,  180.,  210.,  240., 270.,  300.,  360.,  420.,  480.,  600., 1000.]
        xbb_range = [0.,   0.04, 0.08, 0.12, 0.16, 0.2,  0.24, 0.28, 0.32, 0.36, 0.4,  0.44, 0.48, 0.52,
                     0.56, 0.6,  0.64, 0.68, 0.72, 0.76, 0.8,  0.84, 0.88, 0.92, 0.96, 1.  ]
    xbbv11_range = [0., 0.05, 0.1, 0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,  0.55, 0.6,  0.65,
                     0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1.  ]
    msd_range = [  0.,   5.,  10.,  20.,  30.,  40.,  50.,  60.,  80., 100., 120., 150., 200., 250., 300., 350.]

    xbb_axis = hist.axis.Variable(xbbv11_range, name="xbb") if legacy else hist.axis.Variable(xbb_range, name="xbb")
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

        sf_per_jet = eff_data_per_jet[jet]["nominal"]/eff_mc_per_jet[jet]["nominal"]
        sf_err_per_jet = sf_per_jet * np.sqrt(
            (eff_data_per_jet[jet]["stat_up"]/eff_data_per_jet[jet]["nominal"])**2
            + (eff_mc_per_jet[jet]["stat_up"]/eff_mc_per_jet[jet]["nominal"])**2
        )
        h_yield = hist.Hist(pt_axis, msd_axis, xbb_axis)
        h_yield_err = hist.Hist(pt_axis, msd_axis, xbb_axis)
        h_yield.fill(pt, msd, xbb, weight=weight*sf_per_jet)
        h_yield_err.fill(pt, msd, xbb, weight=weight*sf_err_per_jet)
        
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
    print("sf ", sf)

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
