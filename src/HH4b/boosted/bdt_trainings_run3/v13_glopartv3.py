from __future__ import annotations

import numpy as np
import pandas as pd
import vector

from HH4b.utils import PAD_VAL, discretize_var

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

    # A jet is padded if pT < 0
    h1_pad = events[key_map("bbFatJetPt")].to_numpy()[:, 0] < 0
    h2_pad = events[key_map("bbFatJetPt")].to_numpy()[:, 1] < 0
    vbf1_pad = np.asarray(events[(key_map("VBFJetPt"), 0)]) < 0
    vbf2_pad = np.asarray(events[(key_map("VBFJetPt"), 1)]) < 0
    ak4away1_pad = np.asarray(events[(key_map("AK4JetAwayPt"), 0)]) < 0
    ak4away2_pad = np.asarray(events[(key_map("AK4JetAwayPt"), 1)]) < 0

    # Derived masks: pad if any constituent is padded
    hh_pad = h1_pad | h2_pad
    jj_pad = vbf1_pad | vbf2_pad
    h1ak4away1_pad = h1_pad | ak4away1_pad
    h2ak4away2_pad = h2_pad | ak4away2_pad

    def mask(arr, pad_mask):
        """Return arr with PAD_VAL where pad_mask is True."""
        result = np.where(pad_mask, PAD_VAL, arr)
        return result

    df_events = pd.DataFrame(
        {
            # dihiggs system
            key_map("HHPt"): mask(hh.pt, hh_pad),
            key_map("HHeta"): mask(hh.eta, hh_pad),
            key_map("HHmass"): mask(hh.mass, hh_pad),
            # met in the event
            key_map("MET"): events.MET_pt[0],
            # fatjet tau32
            key_map("H1T32"): mask(
                events[key_map("bbFatJetTau3OverTau2")].to_numpy()[:, 0], h1_pad
            ),
            key_map("H2T32"): mask(
                events[key_map("bbFatJetTau3OverTau2")].to_numpy()[:, 1], h2_pad
            ),
            # fatjet mass
            key_map("H1Mass"): mask(
                events[key_map("bbFatJetParT3massX2p")].to_numpy()[:, 0], h1_pad
            ),
            # fatjet kinematics
            key_map("H1Pt"): mask(h1.pt, h1_pad),
            key_map("H2Pt"): mask(h2.pt, h2_pad),
            key_map("H1eta"): mask(h1.eta, h1_pad),
            # xbb
            key_map("H1Xbb"): discretize_var(
                events[key_map("bbFatJetParT3TXbb")].to_numpy()[:, 0],
                bins=[0, 0.8, 0.9, 0.94, 0.97, 0.99, 1],
            ),
            # ratios
            key_map("H1Pt_HHmass"): mask(h1.pt / hh.mass, hh_pad),
            key_map("H2Pt_HHmass"): mask(h2.pt / hh.mass, hh_pad),
            key_map("H1Pt_H2Pt"): mask(h1.pt / h2.pt, h1_pad | h2_pad),
            # vbf mjj and eta_jj
            key_map("VBFjjMass"): mask(jj.mass, jj_pad),
            key_map("VBFjjDeltaEta"): mask(np.abs(vbf1.eta - vbf2.eta), jj_pad),
            # AK4JetAway
            key_map("H1AK4JetAway1dR"): mask(h1.deltaR(ak4away1), h1ak4away1_pad),
            key_map("H2AK4JetAway2dR"): mask(h2.deltaR(ak4away2), h2ak4away2_pad),
            key_map("H1AK4JetAway1mass"): mask(h1ak4away1.mass, h1ak4away1_pad),
            key_map("H2AK4JetAway2mass"): mask(h2ak4away2.mass, h2ak4away2_pad),
        }
    )

    return df_events
