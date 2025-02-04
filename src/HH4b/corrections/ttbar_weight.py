"""
Convert ROOT files with trigger SFs derived by Armen to correctionlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import uproot
from correctionlib import schemav2


def get_corr(corr_key, eff, eff_unc_up, eff_unc_dn, year, edges):
    def singlebinning(eff):
        return schemav2.Binning(
            nodetype="binning",
            input=corr_key,
            edges=edges,
            content=list(eff.flatten()),
            flow=1.0,
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
        output=schemav2.Variable(name="weight", type="real", description="ttbar efficiency"),
        data=schemav2.Category(
            nodetype="category",
            input="systematic",
            content=[
                {"key": "nominal", "value": singlebinning(eff)},
                {"key": "stat_up", "value": singlebinning(eff_unc_up)},
                {"key": "stat_dn", "value": singlebinning(eff_unc_dn)},
            ],
            default=singlebinning(eff),
        ),
        generic_formulas=[],
    )
    return corr


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ROOT files with trigger SFs derived by Armen to correctionlib"
    )

    parser.add_argument(
        "--years", nargs="+", default=["2022", "2023"], help="Years to make corrections for"
    )
    parser.add_argument(
        "--corr-keys", nargs="+", default=["Tau3OverTau2", "Xbb"], help="Corrections keys available"
    )

    return parser.parse_args()


def main():

    args = get_args()

    for year in args.years:
        corrs = {}
        # for corr_key in ["Tau3OverTau2", "Xbb", "PTJJ"]:
        for corr_key in args.corr_keys:
            f = uproot.open(f"data/tt_corr/TTbar_Offline_SFs_{year}/{corr_key}_SF.root")
            sfhist = f["SF"].to_hist()

            sfhist.name = corr_key
            sfhist.label = "weight"
            edges = sfhist.axes.edges[0]
            corr = get_corr(
                corr_key=corr_key,
                eff=sfhist.values(),
                eff_unc_up=np.sqrt(sfhist.variances()),
                eff_unc_dn=np.sqrt(sfhist.variances()),
                year=year,
                edges=edges,
            )

            corrs[corr_key] = corr
        corrections = [corrs[var] for var in args.corr_keys]
        cset = schemav2.CorrectionSet(
            schema_version=2,
            # corrections=[corrs["Tau3OverTau2"], corrs["Xbb"], corrs["PTJJ"]],
            corrections=corrections,
            description=f"ttbar corrections for {year}",
            compound_corrections=[],
        )
        path = Path(f"data/ttbarcorr_{year}.json")
        with path.open("w") as fout:
            fout.write(cset.json(exclude_unset=True))


if __name__ == "__main__":
    main()
