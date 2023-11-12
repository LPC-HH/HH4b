from __future__ import annotations

import sys
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

hep.style.use(["CMS", "firamath"])
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 12})
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.color"] = "#CCCCCC"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["figure.edgecolor"] = "none"

# PU profile is gaussian around 70
# https://cms-pdmv-prod.web.cern.ch/mcm/public/restapi/requests/get_setup/TSG-Run3Summer22EEDR-00168
# https://github.com/cms-sw/cmssw/blob/master/SimGeneral/MixingModule/python/mix_POISSON_average_cfi.py
files = {
    "Pu70": [
        # 2022EE, v10, kl1
        "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv10/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson70KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v1/30000/d00363f4-0cac-410d-8fc7-bb6f60ccb6cd.root",
        "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv10/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson70KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v1/30000/ee9ea1b0-ee93-46b1-b2b9-164cf499ef22.root",
    ],
    #"Pu70_1": [
    #    # 2022EE, v10, kl5
    #    "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv10/GluGlutoHHto4B_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson70KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v1/30000/ff50a677-99ec-4148-9c88-648dad33766b.root"
    #],
    "Pu60": [
        # 2022EE, v12, kl1, Pu60
        "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv12/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/2540000/00d98799-ada3-4a26-8558-5052891a8d23.root",
        "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv12/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/30000/70a2f3a2-1cac-4633-bb22-9bb48ea4114c.root",
        "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv12/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/50000/fc603037-ef65-4bbf-9cef-934ecec40bbe.root",
    ],
    #"Pu60_1": [
    #    "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv12/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/2540000/a8f839de-fe30-4f04-be63-3284b91f23ce.root",
    #    "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv12/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/40000/7bd67f96-8b9e-465a-920d-3ee9e18131f7.root",
    #    "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/mc/Run3Summer22EENanoAODv12/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p10_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/Poisson60KeepRAW_130X_mcRun3_2022_realistic_postEE_v6-v2/50000/b3ba2704-adc1-436e-b111-1dece13a5de7.root",
    #],
}


def main():
    npu_axis = hist.axis.Regular(100, 0, 100, name="npu", label=r"nPUInt")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    h_npu = hist.Hist(npu_axis)
    bins = h_npu.to_numpy()[1]

    for sample in files:
        h_npu = hist.Hist(npu_axis)
        for fname in files[sample]:
            events = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema).events()
            # h_npu.fill(events.Pileup.nTrueInt.to_numpy())
            # print(events.Pileup.nTrueInt.to_numpy())
            h_npu.fill(events.Pileup.nPU.to_numpy())
        pileup_MC = h_npu.to_numpy()[0].astype("float64")
        # avoid division by zero
        pileup_MC[pileup_MC == 0.0] = 1
        # normalize
        pileup_MC /= pileup_MC.sum()
        print(np.round(pileup_MC, 3))
        with Path(f"data/pileup/{sample}.npy").open("wb") as f:
            np.save(f, pileup_MC)

        # plot
        hep.histplot(
            h_npu / pileup_MC.sum(), histtype="step", ax=ax, label=f"2022 MC {sample}", density=True, flow='none',
        )

    # plot profile for data
    path_pileup = "data/MyDataPileupHistogram2022FG.root"
    pileup_profile = uproot.open(path_pileup)["pileup"]
    pileup_profile = pileup_profile.to_numpy()[0]
    # normalise
    pileup_profile /= pileup_profile.sum()
    print("data ", np.round(pileup_profile, 3))
    print("bins ", bins)
    nomBins = bins
    nBinCenters = 0.5 * (nomBins[:-1] + nomBins[1:])
    ax.hist(
        nBinCenters,
        weights=pileup_profile,
        bins=nomBins,
        histtype="stepfilled",
        label="Data",
        hatch=r"\\\\",
        color='black',
        alpha=0.2,
    )
    ax.grid()
    ax.legend()
    ax.set_ylabel("Density")
    ax.set_xlabel("PU profile")
    fig.savefig("pileup_comparison_perfile.png")


if __name__ == "__main__":
    sys.exit(main())
