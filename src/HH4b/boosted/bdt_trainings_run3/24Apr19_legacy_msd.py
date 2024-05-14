from __future__ import annotations

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
            "M": events["bbFatJetMsd"].to_numpy()[:, 0],
        }
    )
    h2 = vector.array(
        {
            "pt": events["bbFatJetPt"].to_numpy()[:, 1],
            "phi": events["bbFatJetPhi"].to_numpy()[:, 1],
            "eta": events["bbFatJetEta"].to_numpy()[:, 1],
            "M": events["bbFatJetMsd"].to_numpy()[:, 1],
        }
    )
    hh = h1 + h2

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
            "H1Mass": events["bbFatJetMsd"].to_numpy()[:, 0],
            # fatjet kinematics
            "H1Pt": h1.pt,
            "H2Pt": h2.pt,
            "H1eta": h1.eta,
            # xbb
            "H1Xbb": events["bbFatJetPNetPXbbLegacy"].to_numpy()[:, 0],
            "H1QCDb": events["bbFatJetPNetPQCDbLegacy"].to_numpy()[:, 0],
            "H1QCDbb": events["bbFatJetPNetPQCDbbLegacy"].to_numpy()[:, 0],
            "H1QCDothers": events["bbFatJetPNetPQCDothersLegacy"].to_numpy()[:, 0],
            # ratios
            "H1Pt_HHmass": h1.pt / hh.mass,
            "H2Pt_HHmass": h2.pt / hh.mass,
            "H1Pt/H2Pt": h1.pt / h2.pt,
        }
    )

    return df_events
