"""
Convert ROOT files with trigger SFs derived by Armen to correctionlib
"""

from __future__ import annotations

from pathlib import Path

import correctionlib.convert
import uproot
from correctionlib import schemav2
import numpy as np

def get_corr(corr_key, eff, eff_unc_up, eff_unc_dn, year, edges):
    def singlebinning(eff):
        return schemav2.Binning(
            nodetype="binning",
            input=corr_key,
            edges=edges,
            content=list(eff.flatten()),
            flow=1.0,  # SET FLOW TO 1.0
        )

    corr = schemav2.Correction(
        name=f"ttbar_corr_{corr_key}_{year}",
        description=f"ttbar correction {corr_key} for {year}",
        version=1,
        inputs=[
            schemav2.Variable(
                name=corr_key,
                type="real",
                description=corr_key,
            ),
            schemav2.Variable(
                name="systematic",
                type="string",
                description="Systematic variation",
            ),
        ],
        output=schemav2.Variable(
            name="weight", type="real", description=f"ttbar efficiency"
        ),
        data=schemav2.Category(
            nodetype="category",
            input="systematic",
            content=[
                {"key": "nominal", "value": singlebinning(eff)},
                {"key": "stat_up", "value": singlebinning(eff_unc_up)},
                {"key": "stat_dn", "value": singlebinning(eff_unc_dn)},
            ],
        ),
    )
    return corr

def main():

    for year in ["2022", "2023"]:
        corrs = {}
        for corr_key in ["Tau3OverTau2", "Xbb", "PTJJ"]:
            f = uproot.open(f"data/tt_corr/TTbar_Offline_SFs_{year}/{corr_key}_SF.root")
            sfhist = f["SF"].to_hist()

            sfhist.name = corr_key
            sfhist.label = "weight"
            edges = sfhist.axes.edges[0]
            # corr = correctionlib.convert.from_histogram(sfhist)
            corr = get_corr(
                corr_key, 
                sfhist.values(),
                np.sqrt(sfhist.variances()),
                np.sqrt(sfhist.variances()),
                year,
                edges,
            )

            corrs[corr_key] = corr
        cset = schemav2.CorrectionSet(
            schema_version=2, corrections=[corrs["Tau3OverTau2"], corrs["Xbb"], corrs["PTJJ"]]
        )
        path = Path(f"data/ttbarcorr_{year}.json")
        with path.open("w") as fout:
            fout.write(cset.json(exclude_unset=True))


if __name__ == "__main__":
    main()
