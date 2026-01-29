from __future__ import annotations

import numpy as np
import pandas as pd
import vector

from HH4b.utils import discretize_var

"""
This config is based on v10_glopartv2.py, but with the following changes:
Discretized the TXbb variable into 5 integer categories

"""


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
            "M": events[key_map("bbFatJetParT3massX2p")].to_numpy()[:, 0],
        }
    )
    h2 = vector.array(
        {
            "pt": events[key_map("bbFatJetPt")].to_numpy()[:, 1],
            "phi": events[key_map("bbFatJetPhi")].to_numpy()[:, 1],
            "eta": events[key_map("bbFatJetEta")].to_numpy()[:, 1],
            "M": events[key_map("bbFatJetParT3massX2p")].to_numpy()[:, 1],
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

    # AK4JetAway
    ak4away1 = vector.array(
        {
            "pt": events[(key_map("AK4JetAwayPt"), 0)],
            "phi": events[(key_map("AK4JetAwayPhi"), 0)],
            "eta": events[(key_map("AK4JetAwayEta"), 0)],
            "M": events[(key_map("AK4JetAwayMass"), 0)],
        }
    )

    ak4away2 = vector.array(
        {
            "pt": events[(key_map("AK4JetAwayPt"), 1)],
            "phi": events[(key_map("AK4JetAwayPhi"), 1)],
            "eta": events[(key_map("AK4JetAwayEta"), 1)],
            "M": events[(key_map("AK4JetAwayMass"), 1)],
        }
    )

    h1ak4away1 = h1 + ak4away1
    h2ak4away2 = h2 + ak4away2

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
            key_map("H1Mass"): events[key_map("bbFatJetParT3massX2p")].to_numpy()[:, 0],
            # fatjet kinematics
            key_map("H1Pt"): h1.pt,
            key_map("H2Pt"): h2.pt,
            key_map("H1eta"): h1.eta,
            # xbb
            key_map("H1Xbb"): discretize_var(
                events[key_map("bbFatJetParT3TXbb")].to_numpy()[:, 0],
                bins=[0, 0.8, 0.9, 0.94, 0.97, 0.99, 1],
            ),
            # ratios
            key_map("H1Pt_HHmass"): h1.pt / hh.mass,
            key_map("H2Pt_HHmass"): h2.pt / hh.mass,
            key_map("H1Pt/H2Pt"): h1.pt / h2.pt,
            # vbf mjj and eta_jj
            key_map("VBFjjMass"): jj.mass,
            key_map("VBFjjDeltaEta"): np.abs(vbf1.eta - vbf2.eta),
            # AK4JetAway
            key_map("H1AK4JetAway1dR"): h1.deltaR(ak4away1),
            key_map("H2AK4JetAway2dR"): h2.deltaR(ak4away2),
            key_map("H1AK4JetAway1mass"): h1ak4away1.mass,
            key_map("H2AK4JetAway2mass"): h2ak4away2.mass,
        }
    )

    return df_events
