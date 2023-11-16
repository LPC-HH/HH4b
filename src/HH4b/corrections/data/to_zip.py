from __future__ import annotations

import glob
import os

for ifile in glob.glob("*.txt.gz"):
    # os.system(f"gzip {ifile}")
    new_name = None
    if (
        "L1FastJet" in ifile
        or "L2Relative" in ifile
        or "L2Residual" in ifile
        or "L3Absolute" in ifile
        or "L2L3Residual" in ifile
    ):
        new_name = ifile.replace(".txt", ".jec.txt")
    elif "Uncertainty" in ifile:
        new_name = ifile.replace(".txt", ".junc.txt")
    elif "PtResolution" in ifile:
        new_name = ifile.replace(".txt", ".jr.txt")
    elif "SF_" in ifile:
        new_name = ifile.replace(".txt", ".jersf.txt")
    else:
        print(f"not changing name {ifile}")
    if new_name:
        # print(f"mv {ifile} {new_name}")
        os.system(f"mv {ifile} {new_name}")
