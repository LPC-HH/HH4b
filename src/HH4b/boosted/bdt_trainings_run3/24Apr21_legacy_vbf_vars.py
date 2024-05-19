from __future__ import annotations

import numpy as np
import pandas as pd
import vector


def bdt_dataframe(events, key_map=lambda x: x):
    """
    Make dataframe with BDT inputs
    NOTE: this function should be saved along with the model for inference usage
    """

    h1 = vector.array(
        {
            "pt": events[key_map("bbFatJetPt")].to_numpy()[:, 0],
            "phi": events[key_map("bbFatJetPhi")].to_numpy()[:, 0],
            "eta": events[key_map("bbFatJetEta")].to_numpy()[:, 0],
            "M": events[key_map("bbFatJetPNetMassLegacy")].to_numpy()[:, 0],
        }
    )
    h2 = vector.array(
        {
            "pt": events[key_map("bbFatJetPt")].to_numpy()[:, 1],
            "phi": events[key_map("bbFatJetPhi")].to_numpy()[:, 1],
            "eta": events[key_map("bbFatJetEta")].to_numpy()[:, 1],
            "M": events[key_map("bbFatJetPNetMassLegacy")].to_numpy()[:, 1],
        }
    )
    hh = h1 + h2

    vbf1 = vector.array(
        {
            "pt": events[(key_map("VBFJetPt"), 0)],
            "phi": events[(key_map("VBFJetPhi"), 0)],
            "eta": events[(key_map("VBFJetEta"), 0)],
            "M": events[(key_map("VBFJetMass"), 0)],
        }
    )

    vbf2 = vector.array(
        {
            "pt": events[(key_map("VBFJetPt"), 1)],
            "phi": events[(key_map("VBFJetPhi"), 1)],
            "eta": events[(key_map("VBFJetEta"), 1)],
            "M": events[(key_map("VBFJetMass"), 1)],
        }
    )

    jj = vbf1 + vbf2

    df_events = pd.DataFrame(
        {
            # dihiggs system
            key_map("HHPt"): hh.pt,
            key_map("HHeta"): hh.eta,
            key_map("HHmass"): hh.mass,
            # met in the event
            key_map("MET"): events.MET_pt[0],
            # fatjet tau32
            key_map("H1T32"): events[key_map("bbFatJetTau3OverTau2")].to_numpy()[:, 0],
            key_map("H2T32"): events[key_map("bbFatJetTau3OverTau2")].to_numpy()[:, 1],
            # fatjet mass
            key_map("H1Mass"): events[key_map("bbFatJetPNetMassLegacy")].to_numpy()[:, 0],
            # fatjet kinematics
            key_map("H1Pt"): h1.pt,
            key_map("H2Pt"): h2.pt,
            key_map("H1eta"): h1.eta,
            # "H2eta": h2.eta,
            # xbb
            key_map("H1Xbb"): events[key_map("bbFatJetPNetPXbbLegacy")].to_numpy()[:, 0],
            key_map("H1QCDb"): events[key_map("bbFatJetPNetPQCDbLegacy")].to_numpy()[:, 0],
            key_map("H1QCDbb"): events[key_map("bbFatJetPNetPQCDbbLegacy")].to_numpy()[:, 0],
            key_map("H1QCDothers"): events[key_map("bbFatJetPNetPQCDothersLegacy")].to_numpy()[
                :, 0
            ],
            # ratios
            key_map("H1Pt_HHmass"): h1.pt / hh.mass,
            key_map("H2Pt_HHmass"): h2.pt / hh.mass,
            key_map("H1Pt/H2Pt"): h1.pt / h2.pt,
            # vbf mjj and eta_jj
            key_map("VBFjjMass"): jj.mass,
            key_map("VBFjjDeltaEta"): np.abs(vbf1.eta - vbf2.eta),
        }
    )

    return df_events
