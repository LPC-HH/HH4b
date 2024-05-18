from __future__ import annotations

import correctionlib
import numpy as np
import pandas as pd
from coffea.lookup_tools.dense_lookup import dense_lookup

from HH4b.utils import makeHH

TT_SF_2022 = np.array(
    [
        # lower bin edge, nom, up, down
        (-1000000, 1),
        (0, 0.886178),
        (35, 1.02858),
        (75, 1.04224),
        (130, 1.05555),
        (200, 1.0296),
        (315, 0.845703),
        (450, 0.699666),
        (700, 0.439261),
        (1000, 1),
        (1000000, 1),
    ]
)

TT_SF_2023 = np.array(
    [
        (-1000000, 1),
        (0, 0.876845),
        (50, 0.984064),
        (100, 0.99184),
        (150, 1.17205),
        (250, 1.36115),
        (450, 1.13521),
        (750, 1),
        (1000000, 1),
    ]
)

# TODO: update to include up/down variations like so
# tt_sfs_lookups = {
#     "nom": {
#         "2022": dense_lookup(TT_SF_2022[:, 1][:-1], TT_SF_2022[:, 0]),
#         "2023": dense_lookup(TT_SF_2023[:, 1][:-1], TT_SF_2023[:, 0]),
#     },
#     "up": {
#         "2022": dense_lookup(TT_SF_2022[:, 2][:-1], TT_SF_2022[:, 0]),
#         "2023": dense_lookup(TT_SF_2023[:, 2][:-1], TT_SF_2023[:, 0]),
#     }
# }

tt_sfs_lookups = {
    "2022": dense_lookup(TT_SF_2022[:, 1][:-1], TT_SF_2022[:, 0]),
    "2023": dense_lookup(TT_SF_2023[:, 1][:-1], TT_SF_2023[:, 0]),
}


def ttbar_pTjjSF(year: str, events_dict: dict[str, pd.DataFrame], mass: str = "bbFatJetPNetMass"):
    """Apply ttbar recoil scale factors"""
    year_ = None
    if "2022" in year:
        year_ = "2022"
    elif "2023" in year:
        year_ = "2023"
    hh = makeHH(events_dict, "ttbar", mass)
    lookup = tt_sfs_lookups[year_]
    sfs = lookup(hh.pt)
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
    triggereff_btag = _load_trig_effs(year, "txbb", region)

    eff_data = triggereff_ptmsd[f"fatjet_triggereffdata_{year}_ptmsd"]
    eff_mc = triggereff_ptmsd[f"fatjet_triggereffmc_{year}_ptmsd"]
    eff_data_btag = triggereff_btag[f"fatjet_triggereffdata_{year}_txbb"]
    eff_mc_btag = triggereff_btag[f"fatjet_triggereffmc_{year}_txbb"]

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
