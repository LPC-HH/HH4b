from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class EventSelection:
    """
    Applies physics-based selection criteria to the events.
    This includes applying pT, msd, txbb, and mass cuts.
    """

    def __init__(
        self,
        jet_collection: str,
        txbb_preselection: dict,
        msd1_preselection: dict,
        msd2_preselection: dict,
    ):
        self.jet_collection = jet_collection
        self.txbb_preselection = txbb_preselection
        self.msd1_preselection = msd1_preselection
        self.msd2_preselection = msd2_preselection

    def apply_boosted(
        self, events_dict: dict[str, pd.DataFrame], txbb_str: str, mass_str: str
    ) -> dict[str, pd.DataFrame]:
        for key, df in events_dict.items():
            msd1 = df[f"{self.jet_collection}Msd"][0]
            msd2 = df[f"{self.jet_collection}Msd"][1]
            pt1 = df[f"{self.jet_collection}Pt"][0]
            pt2 = df[f"{self.jet_collection}Pt"][1]
            txbb1 = df[txbb_str][0]
            mass1 = df[mass_str][0]
            mass2 = df[mass_str][1]

            selection_mask = (
                (pt1 > 300)
                & (pt2 > 250)
                & (txbb1 > self.txbb_preselection[txbb_str])
                & (msd1 > self.msd1_preselection[txbb_str])
                & (msd2 > self.msd2_preselection[txbb_str])
                & (mass1 > 50)
                & (mass2 > 50)
            )
            events_dict[key] = df[selection_mask].copy()
        return events_dict
