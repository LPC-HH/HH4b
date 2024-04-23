from __future__ import annotations

import numpy as np
import pandas as pd
import vector


def bdt_dataframe(events):
    """
    Make dataframe with BDT inputs
    NOTE: this function should be saved along with the model for inference usage
    """

    h1 = vector.array(
        {
            "pt": events["bbFatJetPt"].to_numpy()[:, 0],
            "phi": events["bbFatJetPhi"].to_numpy()[:, 0],
            "eta": events["bbFatJetEta"].to_numpy()[:, 0],
            "M": events["bbFatJetPNetMassLegacy"].to_numpy()[:, 0],
        }
    )
    h2 = vector.array(
        {
            "pt": events["bbFatJetPt"].to_numpy()[:, 1],
            "phi": events["bbFatJetPhi"].to_numpy()[:, 1],
            "eta": events["bbFatJetEta"].to_numpy()[:, 1],
            "M": events["bbFatJetPNetMassLegacy"].to_numpy()[:, 1],
        }
    )
    hh = h1 + h2

    vbf1 = vector.array(
        {
            "pt": events[("VBFJetPt", 0)],
            "phi": events[("VBFJetPhi", 0)],
            "eta": events[("VBFJetEta", 0)],
            "M": events[("VBFJetMass", 0)],
        }
    )

    vbf2 = vector.array(
        {
            "pt": events[("VBFJetPt", 1)],
            "phi": events[("VBFJetPhi", 1)],
            "eta": events[("VBFJetEta", 1)],
            "M": events[("VBFJetMass", 1)],
        }
    )

    jj = vbf1 + vbf2

    df_events = pd.DataFrame(
        {
            # dihiggs system
            "HHPt": hh.pt,
            "HHeta": hh.eta,
            "HHmass": hh.mass,
            # met in the event
            "MET": events.MET_pt[0],
            # fatjet tau32
            "H1T32": events["bbFatJetTau3OverTau2"].to_numpy()[:, 0],
            "H2T32": events["bbFatJetTau3OverTau2"].to_numpy()[:, 1],
            # fatjet mass
            "H1Mass": events["bbFatJetPNetMassLegacy"].to_numpy()[:, 0],
            # fatjet kinematics
            "H1Pt": h1.pt,
            "H2Pt": h2.pt,
            "H1eta": h1.eta,
            # "H2eta": h2.eta,
            # xbb
            "H1Xbb": events["bbFatJetPNetPXbbLegacy"].to_numpy()[:, 0],
            "H1QCDb": events["bbFatJetPNetPQCDbLegacy"].to_numpy()[:, 0],
            "H1QCDbb": events["bbFatJetPNetPQCDbbLegacy"].to_numpy()[:, 0],
            "H1QCDothers": events["bbFatJetPNetPQCDothersLegacy"].to_numpy()[:, 0],
            # ratios
            "H1Pt_HHmass": h1.pt / hh.mass,
            "H2Pt_HHmass": h2.pt / hh.mass,
            "H1Pt/H2Pt": h1.pt / h2.pt,
            # vbf mjj and eta_jj
            "VBFjjMass": jj.mass,
            "VBFjjDeltaEta": np.abs(vbf1.eta - vbf2.eta),
        }
    )

    return df_events
