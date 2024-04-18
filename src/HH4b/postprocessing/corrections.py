from __future__ import annotations

import numpy as np
import pandas as pd
from coffea.lookup_tools.dense_lookup import dense_lookup

from HH4b.utils import makeHH

TT_SF_2022 = np.array(
    [
        # lower bin edge, nom, up, down
        (0, 0.886178),
        (35, 1.02858),
        (75, 1.04224),
        (130, 1.05555),
        (200, 1.0296),
        (315, 0.845703),
        (450, 0.699666),
        (700, 0.439261),
        (1000, 1),
    ]
)

TT_SF_2023 = np.array(
    [
        (0, 0.876845),
        (50, 0.984064),
        (100, 0.99184),
        (150, 1.17205),
        (250, 1.36115),
        (450, 1.13521),
        (750, 1),
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
    hh = makeHH(events_dict, "ttbar", mass)
    lookup = tt_sfs_lookups[year]
    sfs = lookup(hh.pt)
    return sfs
