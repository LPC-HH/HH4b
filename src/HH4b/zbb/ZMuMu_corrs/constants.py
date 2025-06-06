from __future__ import annotations

import numpy as np

MASS_RANGE = (80, 100)
PT_RANGE = (200, np.inf)

PT_BINS = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 1000, 2000])
MASS_BINS = np.arange(*MASS_RANGE, 1)

# https://pdg.lbl.gov/2025/tables/rpp2025-sum-gauge-higgs-bosons.pdf
BR_Z_QQ = 0.69911  # hadrons
BR_Z_EE = 0.033632
BR_Z_MUMU = 0.033662
BR_Z_TAUTAU = 0.033696
