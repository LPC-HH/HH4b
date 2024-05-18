"""
Convert ROOT files with trigger SFs derived by Armen to correctionlib
"""

from __future__ import annotations

import argparse

import numpy as np
import Path
import uproot
from correctionlib import schemav2


def get_corr_pt_msd(eff, eff_unc_up, eff_unc_dn, year, label, trigger_label, edges):
    def multibinning(eff):
        return schemav2.MultiBinning(
            nodetype="multibinning",
            inputs=["pt", "msd"],
            edges=edges,
            content=list(eff.flatten()),
            flow=1.0,  # SET FLOW TO 1.0
        )

    corr = schemav2.Correction(
        name=f"fatjet_triggereff{label}_{year}_{trigger_label}",
        description=f"{label} Trigger efficiency for trigger soup: {trigger_label}",
        version=1,
        inputs=[
            schemav2.Variable(
                name="pt",
                type="real",
                description="Jet transverse momentum",
            ),
            schemav2.Variable(
                name="msd",
                type="real",
                description="Jet softdrop mass",
            ),
            schemav2.Variable(
                name="systematic",
                type="string",
                description="Systematic variation",
            ),
        ],
        output=schemav2.Variable(
            name="weight", type="real", description=f"Jet {label} trigger efficiency"
        ),
        data=schemav2.Category(
            nodetype="category",
            input="systematic",
            content=[
                {"key": "nominal", "value": multibinning(eff)},
                {"key": "stat_up", "value": multibinning(eff_unc_up)},
                {"key": "stat_dn", "value": multibinning(eff_unc_dn)},
            ],
        ),
    )
    return corr


def get_corr_txbb(eff, eff_unc_up, eff_unc_dn, year, label, trigger_label, edges):
    def singlebinning(eff):
        return schemav2.Binning(
            nodetype="binning",
            input="txbb",
            edges=edges,
            content=list(eff.flatten()),
            flow=1.0,  # SET FLOW TO 1.0
        )

    corr = schemav2.Correction(
        name=f"fatjet_triggereff{label}_{year}_{trigger_label}",
        description=f"{label} Trigger efficiency for trigger soup: {trigger_label}",
        version=1,
        inputs=[
            schemav2.Variable(
                name="txbb",
                type="real",
                description="Jet TXbb",
            ),
            schemav2.Variable(
                name="systematic",
                type="string",
                description="Systematic variation",
            ),
        ],
        output=schemav2.Variable(
            name="weight", type="real", description=f"Jet {label} trigger efficiency"
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


def main(args):

    region = args.region

    corr_pt_msd = {
        "2022": {
            "fname": f"data/trigger/2022/PT_Mass_2dSF_{region}.root",
            "data": "Eff_Data_PreEE",
            "mc": "Eff_MC_PreEE",
        },
        "2022EE": {
            "fname": f"data/trigger/2022/PT_Mass_2dSF_{region}.root",
            "data": "Eff_Data_PostEE",
            "mc": "Eff_MC_PostEE",
        },
        "2023": {
            "fname": f"data/trigger/2023/PT_Mass_SF_{region}/PT_Mass_2dSF_PreBPix.root",
            "data": "Eff_Data_ETA0",
            "mc": "Eff_MC_ETA0",
        },
        "2023BPix": {
            "fname": f"data/trigger/2023/PT_Mass_SF_{region}/PT_Mass_2dSF_PostBPix.root",
            "data": "Eff_Data_ETA0",
            "mc": "Eff_MC_ETA0",
        },
    }

    corr_txbb = {
        "2022": {
            "fname": f"data/trigger/2022/BTG_SF_{region}_PreEE.root",
            "data": "BTG_Eff_Data",
            "mc": "BTG_Eff_MC",
        },
        "2022EE": {
            "fname": f"data/trigger/2022/BTG_SF_{region}_PostEE.root",
            "data": "BTG_Eff_Data",
            "mc": "BTG_Eff_MC",
        },
        "2023": {
            "fname": f"data/trigger/2023/Online_BTG_{region}/BTG_SF.root",
            "data": "BTG_Eff_Data",
            "mc": "BTG_Eff_MC",
        },
        "2023BPix": {
            "fname": f"data/trigger/2023/Online_BTG_{region}/BTG_SF.root",
            "data": "BTG_Eff_Data",
            "mc": "BTG_Eff_MC",
        },
    }

    corr_txbb_v11 = {
        "2022": {
            "fname": f"data/trigger/BTG_SFs_TXbb/{region}/2022/BTG_SF_TXbb_PreEE.root",
            "data": "BTG_Eff_Data",
            "mc": "BTG_Eff_MC",
        },
        "2022EE": {
            "fname": f"data/trigger/BTG_SFs_TXbb/{region}/2022/BTG_SF_TXbb_PostEE.root",
            "data": "BTG_Eff_Data",
            "mc": "BTG_Eff_MC",
        },
        "2023": {
            "fname": f"data/trigger/BTG_SFs_TXbb/{region}/2023/BTG_SF_TXbb_PreBPix.root",
            "data": "BTG_Eff_Data",
            "mc": "BTG_Eff_MC",
        },
        "2023BPix": {
            "fname": f"data/trigger/BTG_SFs_TXbb/{region}/2023/BTG_SF_TXbb_PostBPix.root",
            "data": "BTG_Eff_Data",
            "mc": "BTG_Eff_MC",
        },
    }

    for year, corr_items in corr_pt_msd.items():
        f = uproot.open(corr_items["fname"])
        corrs = {}
        for key in ["data", "mc"]:
            h = f[corr_items[key]].to_hist()
            edges = [
                h.axes[0].edges,
                h.axes[1].edges,
            ]

            corr = get_corr_pt_msd(
                h.values(),
                np.sqrt(h.variances()),
                np.sqrt(h.variances()),
                year,
                key,
                "ptmsd",
                edges,
            )

            # rich.print(corr)
            corrs[key] = corr

        cset = schemav2.CorrectionSet(schema_version=2, corrections=[corrs["mc"], corrs["data"]])
        with Path.open(f"data/fatjet_triggereff_{year}_ptmsd_{region}.json", "w") as fout:
            fout.write(cset.json(exclude_unset=True))

    for year, corr_items in corr_txbb.items():
        f = uproot.open(corr_items["fname"])
        corrs = {}
        for key in ["data", "mc"]:
            h = f[corr_items[key]].to_hist()
            edges = h.axes.edges[0]
            corr = get_corr_txbb(
                h.values(),
                np.sqrt(h.variances()),
                np.sqrt(h.variances()),
                year,
                key,
                "txbb",
                edges,
            )
            # rich.print(corr)
            corrs[key] = corr

        cset = schemav2.CorrectionSet(schema_version=2, corrections=[corrs["mc"], corrs["data"]])
        with Path.open(f"data/fatjet_triggereff_{year}_txbb_{region}.json", "w") as fout:
            fout.write(cset.json(exclude_unset=True))

    for year, corr_items in corr_txbb_v11.items():
        f = uproot.open(corr_items["fname"])
        corrs = {}
        for key in ["data", "mc"]:
            h = f[corr_items[key]].to_hist()
            edges = h.axes.edges[0]
            corr = get_corr_txbb(
                h.values(),
                np.sqrt(h.variances()),
                np.sqrt(h.variances()),
                year,
                key,
                "txbbv11",
                edges,
            )
            # rich.print(corr)
            corrs[key] = corr

        cset = schemav2.CorrectionSet(schema_version=2, corrections=[corrs["mc"], corrs["data"]])
        with Path.open(f"data/fatjet_triggereff_{year}_txbbv11_{region}.json", "w") as fout:
            fout.write(cset.json(exclude_unset=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--region", required=True, choices=["QCD", "TTbar"], type=str)
    args = parser.parse_args()

    main(args)
