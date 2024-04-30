#!/usr/bin/env python3
"""
DESCRIPTION:
Script to compare two NTuples.
"""
from __future__ import annotations

import array

import ROOT

ROOT.gInterpreter.Declare(
    """
// A thread-safe stateful filter that lets only one event pass for each value of
// "category" (where "category" is a random character).
// It is using gCoreMutex, which is a read-write lock, to have a bit less contention between threads.
class FilterOnePerKind {
  std::unordered_set<char> _seenCategories;
public:
  bool operator()(char category) {
    {
      R__READ_LOCKGUARD(ROOT::gCoreMutex); // many threads can take a read lock concurrently
      if (_seenCategories.count(category) == 1)
        return false;
    }
    // if we are here, `category` was not already in _seenCategories
    R__WRITE_LOCKGUARD(ROOT::gCoreMutex); // only one thread at a time can take the write lock
    _seenCategories.insert(category);
    return true;
  }
};
"""
)

ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT()
ROOT.ROOT.EnableThreadSafety()

ROOT.PyConfig.IgnoreCommandLineOptions = True

# ===================================
# Common binning and labels
# ===================================
from aux import bins, labels


def plot(h0, savename, sample, region, isWeighted):
    # Canvas and general style options
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTextFont(42)
    d = ROOT.TCanvas("", "", 800, 700)

    # Add legend
    legend = ROOT.TLegend(0.55, 0.70, 0.82, 0.88)
    legend.SetHeader(sample + " " + region)
    legend.SetFillColor(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)

    # Make sure the canvas stays in the list of canvases after the macro execution
    ROOT.SetOwnership(d, False)
    d.SetLeftMargin(0.15)

    h0.SetLineColor(ROOT.kAzure + 1)
    h0.SetMarkerColor(ROOT.kAzure + 1)
    h0.SetFillColorAlpha(ROOT.kAzure - 4, 0.35)
    h0.SetMarkerStyle(23)

    h0.SetMinimum(0.0)
    h0.GetYaxis().SetTitle("Arbitrary Units")
    h0.Draw("HIST")
    d.Modified()
    d.Update()

    legend.AddEntry(h0.GetPtr(), "for overlap (" + str(round(h0.GetPtr().Integral(), 3)) + ")")
    legend.Draw("same")

    cms_label = ROOT.TLatex()
    cms_label.SetTextSize(0.04)
    cms_label.DrawLatexNDC(0.16, 0.92, "#bf{CMS}")

    prel_label = ROOT.TLatex()
    prel_label.SetTextSize(0.035)
    prel_label.DrawLatexNDC(0.24, 0.92, "#it{Preliminary Simulation}")

    header = ROOT.TLatex()
    header.SetTextSize(0.035)
    header.DrawLatexNDC(0.60, 0.92, "2022 EE, #sqrt{s} = 13.6 TeV")
    d.SaveAs(savename)
    return


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    samples = ["signal"]

    path_0 = "/eos/home-m/mkolosov/Run3_HHTo4B_NTuples/2022/Main_PNet_MinDiag_w4j35_w2bj30_dHHjw30_withoutSyst_25April2024_2022_0L/"
    files = {
        "0-signal": path_0
        + "mc/parts/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8_tree.root",
    }

    df = {}
    for s in list(files.keys()):
        df[s] = ROOT.RDataFrame("Events", files[s])

        df[s] = df[s].Define(
            "dHiggsDeltaRegMass",
            "sqrt((dHH_H1_regmass-125.0)*(dHH_H1_regmass-125.0)+(dHH_H2_regmass-120.0)*(dHH_H2_regmass-120.0))",
        )
        if "data" in s:
            df[s] = df[s].Define("weight", "1")
        else:
            df[s] = df[s].Define(
                "weight",
                "genWeight*xsecWeight*lumiwgt*btagSF_central*trgSF_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_central*puWeight",
            )

        # Apply filters here:
        df[s] = df[s].Filter("pass_resolved_skim", "Resolved skim")
        df[s] = df[s].Filter(
            "passTrig_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65", "Trigger"
        )
        df[s] = df[s].Filter(
            "passL1unprescaled_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
            "L1 unprescaled",
        )
        df[s] = df[s].Filter(
            "passTrigObjMatching_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
            "Trigger matching",
        )
        df[s] = df[s].Filter("passmetfilters", "MET filters")
        df[s] = df[s].Filter("passjetvetomap", "Passed jet veto map")

        print(
            "Event counts for resolved, after applying resolved skim : ", df[s].Count().GetValue()
        )

        # Split into categories:
        df[f"{s}-all"] = df[s]
        df[f"{s}-4b"] = df[s].Filter("dHH_NbtagM == 4", "4b sample")
        df[f"{s}-3b"] = df[s].Filter("dHH_NbtagM == 3", "3b sample")
        df[f"{s}-4b-SR"] = df[f"{s}-4b"].Filter("dHiggsDeltaRegMass < 30.0", "4b SR")
        df[f"{s}-3b-SR"] = df[f"{s}-3b"].Filter("dHiggsDeltaRegMass < 30.0", "3b SR")
        df[f"{s}-4b-CR"] = df[f"{s}-4b"].Filter(
            "dHiggsDeltaRegMass >= 30.0 && dHiggsDeltaRegMass < 55", "4b CR"
        )
        df[f"{s}-3b-CR"] = df[f"{s}-3b"].Filter(
            "dHiggsDeltaRegMass >= 30.0 && dHiggsDeltaRegMass < 55", "3b CR"
        )

        df[f"{s}-4b-SR-wboosted"] = df[f"{s}-4b"].Filter("pass_boosted_skim", "Boosted skim")
        df[f"{s}-4b-SR-wboosted"] = df[f"{s}-4b-SR-wboosted"].Filter(
            "n_ak8 >= 2", "At least two fatjets"
        )
        df[f"{s}-4b-SR-wboosted"] = df[f"{s}-4b-SR-wboosted"].Filter(
            "ak8_pt[0] > 300", "Leading AK8 with pT > 300"
        )
        print(
            "Counter for events passing both resolved and boosted skims : ",
            df[f"{s}-4b-SR-wboosted"].Count().GetValue(),
        )

    histos = {}
    regions = ["all", "4b", "4b-SR", "4b-SR-wboosted"]
    variables = list(bins.keys())
    for sample in list(files.keys()):
        for region in regions:
            for var in variables:
                if "data" in sample and "SF" in var:
                    continue
                h = ROOT.RDF.TH1DModel(
                    "%s_%s_%s" % (sample, region, var),
                    labels[var],
                    len(bins[var]) - 1,
                    array.array("d", bins[var]),
                )
                h_weighted = ROOT.RDF.TH1DModel(
                    "%s_%s_%s_weighted" % (sample, region, var),
                    labels[var],
                    len(bins[var]) - 1,
                    array.array("d", bins[var]),
                )
                histos["%s-%s-%s" % (sample, region, var)] = df["%s-%s" % (sample, region)].Histo1D(
                    h, var
                )
                histos["%s-%s-%s_weighted" % (sample, region, var)] = df[
                    "%s-%s" % (sample, region)
                ].Histo1D(h_weighted, var, "weight")

    for histo in histos:
        ROOT.RDF.RunGraphs([histos[histo]])

    for sample in samples:
        for region in regions:
            for var in variables:
                if "data" in sample and "SF" in var:
                    continue
                if "data" in sample and "SR" in region and "4b" in region:
                    pass
                else:
                    plot(
                        histos["0-%s-%s-%s" % (sample, region, var)],
                        "Debug_%s_%s_%s.pdf" % (sample, region, var),
                        sample,
                        region,
                        False,
                    )

                    plot(
                        histos["0-%s-%s-%s_weighted" % (sample, region, var)],
                        "Debug_%s_%s_%s_weighted.pdf" % (sample, region, var),
                        sample,
                        region,
                        True,
                    )

    return


if __name__ == "__main__":
    main()
