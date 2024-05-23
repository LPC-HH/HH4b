from __future__ import annotations

import numpy as np
import pandas as pd
import vector


def bdt_dataframe_resolved(events):
    """
    Make dataframe with BDT inputs for resolved HH4b events
    """

    # Assuming the first two b-tagged jets are H1 and the next two are H2
    jet0 = events.query("subentry == 0")
    jet1 = events.query("subentry == 0")

    h1 = vector.array(
        {
            "pt": jet0["ak8_pt"],
            "phi": jet0["ak8_phi"],
            "eta": jet0["ak8_eta"],
            "M": jet0["ak8_particleNet_mass"],  # use reg mass (PNET)'ak8_particleNet_mass'
        }
    )
    h2 = vector.array(
        {
            "pt": jet1["ak8_pt"],
            "phi": jet1["ak8_phi"],
            "eta": jet1["ak8_eta"],
            "M": jet1["ak8_particleNet_mass"],
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
            "MET": events["met"][:, 0],
            # pseudo-fatjet tau32
            "H1T32": jet0["ak8_tau3"].to_numpy() / jet0["ak8_tau2"].to_numpy(),
            "H2T32": jet1["ak8_tau3"].to_numpy() / jet1["ak8_tau2"].to_numpy(),
            # pseudo-fatjet mass
            "H1Mass": h1.M,
            # pseudo-fatjet kinematics
            "H1logPt": np.log(h1.pt),
            "H2logPt": np.log(h2.pt),
            "H1eta": h1.eta,
            "H2eta": h2.eta,
            # xbb
            "H1Xbb": jet0["ak8_Txbb"].to_numpy(),
            "H1QCD1HF": jet0["ak8_PQCDb"].to_numpy(),
            "H1QCD2HF": jet0["ak8_PQCDbb"].to_numpy(),
            "H1QCD0HF": jet0["ak8_PQCDothers"].to_numpy(),
            # ratios
            "H1Pt_HHmass": h1.pt / hh.mass,
            "H2Pt_HHmass": h2.pt / hh.mass,
            "H1Pt/H2Pt": h1.pt / h2.pt,
        }
    )

    return df_events
