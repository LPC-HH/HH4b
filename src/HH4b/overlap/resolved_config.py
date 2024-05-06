from __future__ import annotations

import numpy as np
import pandas as pd
import vector


def bdt_dataframe_resolved(events):
    """
    Make dataframe with BDT inputs for resolved HH4b events
    """

    # Assuming the first two b-tagged jets are H1 and the next two are H2
    h1 = vector.array(
        {
            "pt": events["ak8_pt"][:, 0:2].sum(axis=1),
            "phi": events["ak8_phi"][:, 0:2].mean(
                axis=1
            ),  # Approximate phi of the composite particle, check if valid
            "eta": events["ak8_eta"][:, 0:2].mean(axis=1),
            "M": events["dHH_H1_regmass"],
        }
    )
    h2 = vector.array(
        {
            "pt": events["ak8_pt"][:, 2:4].sum(axis=1),
            "phi": events["ak8_phi"][:, 2:4].mean(axis=1),
            "eta": events["ak8_eta"][:, 2:4].mean(axis=1),
            "M": events["dHH_H2_regmass"],
        }
    )
    hh = h1 + h2

    df_events = pd.DataFrame(
        {
            # dihiggs system
            "HHlogPt": np.log(hh.pt),
            "HHeta": hh.eta,
            "HHmass": hh.mass,
            # met in the event
            "MET": events["MET_pt"],  # Check how MET is stored and adjust accordingly
            # pseudo-fatjet tau32
            "H1T32": events["ak8_tau3"][:, 0] / events["ak8_tau2"][:, 0],
            "H2T32": events["ak8_tau3"][:, 1] / events["ak8_tau2"][:, 1],
            # pseudo-fatjet mass
            "H1Mass": h1.M,
            # pseudo-fatjet kinematics
            "H1logPt": np.log(h1.pt),
            "H2logPt": np.log(h2.pt),
            "H1eta": h1.eta,
            "H2eta": h2.eta,
            # xbb
            "H1Xbb": events["ak8_Txbb"].to_numpy()[:, 0],
            "H1QCD1HF": events["ak8_PQCDb"].to_numpy()[:, 0],
            "H1QCD2HF": events["ak8_PQCDbb"].to_numpy()[:, 0],
            "H1QCD0HF": events["ak8_PQCDothers"].to_numpy()[:, 0],
            # ratios
            "H1Pt_HHmass": h1.pt / hh.mass,
            "H2Pt_HHmass": h2.pt / hh.mass,
            "H1Pt/H2Pt": h1.pt / h2.pt,
        }
    )

    return df_events
