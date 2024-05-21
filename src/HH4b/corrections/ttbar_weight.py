"""
Convert ROOT files with trigger SFs derived by Armen to correctionlib
"""

from __future__ import annotations

from pathlib import Path

import correctionlib.convert
import uproot
from correctionlib import schemav2


def main():

    for year in ["2022", "2023"]:
        corrs = {}
        for corr_key in ["Tau3OverTau2", "Xbb", "PTJJ"]:
            f = uproot.open(f"data/tt_corr/TTbar_Offline_SFs_{year}/{corr_key}_SF.root")
            sfhist = f["SF"].to_hist()

            sfhist.name = corr_key
            sfhist.label = "weight"
            corr = correctionlib.convert.from_histogram(sfhist)
            corr.data.flow = "clamp"
            print(corr)

            corrs[corr_key] = corr
        cset = schemav2.CorrectionSet(
            schema_version=2, corrections=[corrs["Tau3OverTau2"], corrs["Xbb"], corrs["PTJJ"]]
        )
        path = Path(f"data/ttbarcorr_{year}.json")
        with path.open("w") as fout:
            fout.write(cset.json(exclude_unset=True))


if __name__ == "__main__":
    main()
