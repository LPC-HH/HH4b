#!/usr/bin/env python
"""
A script to generate a BinnedValues-JSON file for pileup reweighting of MC
https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/misc/LUM/makePUReWeightJSON.py
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)
import numpy as np

mcPUProfiles = {
    # ========#
    #  2022  #
    # ========#
    # https://github.com/cms-sw/cmssw/pull/34460
    # https://github.com/cms-sw/cmssw/blob/master/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py#L10-L54
    "2022Prompt_25ns": (
        np.linspace(0.0, 100.0, 101),
        [
            7.075550618391933e-8,
            1.8432226484975646e-7,
            4.6156514471969593e-7,
            0.0000011111611991838491,
            0.0000025719752161798103,
            0.000005724865812608344,
            0.000012255841383374045,
            0.000025239403069596116,
            0.00005001054998201597,
            0.00009536530158990567,
            0.00017505633393457624,
            0.00030942214916825035,
            0.0005268123536229287,
            0.0008642843968521786,
            0.0013669182280399903,
            0.0020851167548246985,
            0.0030695148409245446,
            0.004363635945105083,
            0.005995143197404548,
            0.007967247822222358,
            0.010252302872826594,
            0.01278957659177177,
            0.015488544412469806,
            0.01823784978331645,
            0.020918669702105028,
            0.023420019399650906,
            0.025652949149203495,
            0.027560835627835043,
            0.02912397347687914,
            0.030358091266301533,
            0.03130778480604892,
            0.03203676872496023,
            0.0326170853351521,
            0.03311902652393314,
            0.033602777248239,
            0.0341120235754556,
            0.03466927947785801,
            0.03527261707506484,
            0.035893786618889145,
            0.03647817900850185,
            0.036947435730750315,
            0.03720550450678737,
            0.037148460727673235,
            0.03667753703450604,
            0.03571377296329832,
            0.034211859754226276,
            0.032170439241889726,
            0.029636506070368274,
            0.02670262519076345,
            0.023497154911314072,
            0.020169158697337236,
            0.016870783471647905,
            0.013740289679427057,
            0.010888563843704815,
            0.008390977574442656,
            0.006285186751143873,
            0.004574246293656772,
            0.003233538335807419,
            0.002219622271900557,
            0.0014792038980537092,
            0.0009568560481315006,
            0.0006007171037926386,
            0.00036596934105178995,
            0.0002163349104153549,
            0.00012407362512604619,
            0.0000690356949524181,
            0.000037263645547231494,
            0.00001951170588910065,
            0.000009910336118978026,
            0.0000048826244075428666,
            0.0000023333596885075797,
            0.0000010816029570543702,
            4.863048449289416e-7,
            2.1208148308081624e-7,
            8.97121135679932e-8,
            3.6809172420519874e-8,
            1.4649459937201982e-8,
            5.655267024863598e-9,
            2.117664468591336e-9,
            7.692038404370259e-10,
            2.7102837405697987e-10,
            9.263749466613295e-11,
            3.071624552355945e-11,
            9.880298997379985e-12,
            3.0832214331312204e-12,
            9.33436314183754e-13,
            2.7417209623761203e-13,
            7.813293248960901e-14,
            2.1603865264197903e-14,
            5.796018523167997e-15,
            1.5088422256459697e-15,
            3.811436255838504e-16,
            9.342850737730402e-17,
            2.2224464483477953e-17,
            5.130498608124184e-18,
            1.1494216669980747e-18,
            2.499227229379666e-19,
            5.2741621866055994e-20,
            1.080281961755894e-20,
            2.1476863811171814e-21,
        ],
    ),
}


def getHist(fName, hName="pileup"):
    from cppyy import gbl

    tf = gbl.TFile.Open(fName)
    if not tf:
        raise RuntimeError(f"Could not open file '{fName}'")
    hist = tf.Get(hName)
    if not hist:
        raise RuntimeError(f"No histogram with name '{hName}' found in file '{fName}'")
    return tf, hist


def normAndExtract(hist, norm=1.0):
    nB = hist.GetNbinsX()
    xAx = hist.GetXaxis()
    if norm:
        hist.Scale(norm / (hist.Integral() * (xAx.GetXmax() - xAx.GetXmin()) / nB))
    bEdges = np.array([xAx.GetBinLowEdge(i) for i in range(1, nB + 1)] + [xAx.GetBinUpEdge(nB)])
    contents = np.array([hist.GetBinContent(i) for i in range(1, nB + 1)])
    return bEdges, contents


def getRatio(numBins, numCont, denBins, denCont):
    ## use numerator for output format
    if not all(db in numBins for db in denBins):
        raise RuntimeError(
            "Numerator (data) needs to have at least the bin edges that are the denominator (MC)"
        )
    ## ratios for the common range
    xMinC, xMaxC = denBins[0], denBins[-1]
    inMn = np.where(numBins == xMinC)[0][0]
    inMx = np.where(numBins == xMaxC)[0][0]
    ratio = np.zeros((inMx - inMn,))
    di = 0
    for ni in range(inMn, inMx):
        if numBins[ni + 1] > denBins[di + 1]:
            di += 1
        assert (denBins[di] <= numBins[ni]) and (numBins[ni + 1] <= denBins[di + 1])
        if denCont[di] == 0.0:
            ratio[ni - inMn] = 1.0  ## not in denominator -> will not be used, so any value works
        else:
            ratio[ni - inMn] = numCont[ni] / denCont[di]
    bR = np.array(numBins[inMn : inMx + 1])
    ## extend range of outside ratio bins until end of numerator ranges
    bR[0] = numBins[0]
    bR[-1] = numBins[-1]
    return bR, ratio


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Produce a BinnedValues-JSON file for pileup reweighting, using data pileup distributions obtained with `pileupCalc.py -i analysis-lumi-json.txt --inputLumiJSON pileup-json.txt --calcMode true --minBiasXsec MBXSECINNB --maxPileupBin NMAX --numPileupBins N outname.root` (see also https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData#Pileup_JSON_Files_For_Run_II)"
    )
    parser.add_argument(
        "-o", "--output", default="puweights.json", type=str, help="Output file name"
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["correctionlib", "cp3-llbb"],
        default="cp3-llbb",
        help="Output JSON format",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="puweights",
        help="Name of the correction inside the CorrectionSet (only used for the correctionlib format)",
    )
    parser.add_argument(
        "--mcprofile",
        help="Pileup profile used to generate the MC sample (use --listmcprofiles to see the list of defined profiles)",
    )
    parser.add_argument(
        "--listmcprofiles", action="store_true", help="list the available MC pileup profiles"
    )
    parser.add_argument(
        "--nominal",
        type=str,
        help="File with the data (true) pileup distribution histogram assuming the nominal minimum bias cross-section value",
    )
    parser.add_argument(
        "--up",
        type=str,
        help="File with the data (true) pileup distribution histogram assuming the nominal+1sigma minimum bias cross-section value",
    )
    parser.add_argument(
        "--down",
        type=str,
        help="File with the data (true) pileup distribution histogram assuming the nominal-1sigma minimum bias cross-section value",
    )
    parser.add_argument("--rebin", type=int, help="Factor to rebin the data histograms by")
    parser.add_argument(
        "--makePlot",
        action="store_true",
        help="Make a plot of the PU profiles and weights (requires matplotlib)",
    )
    parser.add_argument(
        "mcfiles", nargs="*", help="MC ROOT files to extract a pileup profile from (if used)"
    )
    parser.add_argument(
        "--mctreename", type=str, default="Events", help="Name of the tree to use in mcfiles"
    )
    parser.add_argument(
        "--mcreweightvar",
        type=str,
        default="Pileup_nTrueInt",
        help="Name of the branch in the tree of the mcfiles to use for getting a histogram",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--gzip", action="store_true", help="Save the output as gzip file")
    args = parser.parse_args()
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO))
    if args.makePlot:
        try:
            import matplotlib

            matplotlib.use("agg")
            from matplotlib import pyplot as plt
        except Exception:
            logger.warning("matplotlib could not be imported, so no plot will be produced")
            args.makePlot = False
    if args.gzip:
        try:
            import gzip
            import io
        except Exception:
            logger.warning(
                "gzip or io could not be imported, output will be stored as regular file"
            )
            args.gzip = False
    if args.listmcprofiles:
        logger.info(
            "The known PU profiles are: {0}".format(", ".join(repr(k) for k in mcPUProfiles))
        )
        return
    elif args.mcfiles:
        if args.mcprofile:
            logger.warning("MC PU profile and MC files are passed - extracting from the files")
        logger.info(
            "Extracting the MC profile from {0} in the {1} tree of: {2}".format(
                args.mcreweightvar, args.mctreename, ", ".join(args.mcfiles)
            )
        )
        from cppyy import gbl

        tup = gbl.TChain(args.mctreename)
        for mcfn in args.mcfiles:
            tup.Add(mcfn)
        hMCPU = gbl.TH1F("hMCPU", "MC PU profile", 100, 0.0, 100.0)
        tup.Draw(f"{args.mcreweightvar}>>hMCPU", "", "GOFF")
        mcPUBins, mcPUVals = normAndExtract(hMCPU)
    elif args.mcprofile:
        if args.mcprofile not in mcPUProfiles:
            raise ValueError(f"No MC PU profile with tag '{args.mcprofile}' is known")

        mcPUBins, mcPUVals = mcPUProfiles[args.mcprofile]
        if len(mcPUBins) != len(mcPUVals) + 1:
            logger.verbose(len(mcPUBins), len(mcPUVals))
    else:
        raise RuntimeError(
            "Either one of --listmcprofiles or --mcprofile, or a list of MC files to extract a MC profile from, must be passed"
        )

    if not args.nominal:
        raise RuntimeError("No --nominal argument")

    fNom, hNom = getHist(args.nominal)
    if args.rebin:
        hNom.Rebin(args.rebin)
    nomBins, nomCont = normAndExtract(hNom)
    ratioBins, nomRatio = getRatio(nomBins, nomCont, mcPUBins, mcPUVals)

    upCont, upRatio, downCont, downRatio = None, None, None, None
    if bool(args.up) != bool(args.down):
        raise ValueError("If either one of --up and --down is specified, both should be")
    if args.up and args.down:
        fUp, hUp = getHist(args.up)
        if args.rebin:
            hUp.Rebin(args.rebin)
        upBins, upCont = normAndExtract(hUp)
        # if not all(ub == nb for ub,nb in zip(upBins, nomBins)):
        #    raise RuntimeError("Up-variation and nominal binning is different: {0} vs {1}".format(upBins, nomBins))
        _, upRatio = getRatio(upBins, upCont, mcPUBins, mcPUVals)
        fDown, hDown = getHist(args.down)
        if args.rebin:
            hDown.Rebin(args.rebin)
        downBins, downCont = normAndExtract(hDown)
        # if not all(db == nb for db,nb in zip(downBins, nomBins)):
        #    raise RuntimeError("Up-variation and nominal binning is different: {0} vs {1}".format(upBins, nomBins))
        _, downRatio = getRatio(downBins, downCont, mcPUBins, mcPUVals)

    if args.format == "correctionlib":
        out = {
            "schema_version": 2,
            "corrections": [
                {
                    "name": args.name,
                    "version": 0,
                    "inputs": [
                        {
                            "name": "NumTrueInteractions",
                            "type": "real",
                            "description": "Number of true interactions",
                        },
                        {
                            "name": "weights",
                            "type": "string",
                            "description": "nominal, up, or down",
                        },
                    ],
                    "output": {
                        "name": "weight",
                        "type": "real",
                        "description": "Event weight for pileup reweighting",
                    },
                    "data": {
                        "nodetype": "category",
                        "input": "weights",
                        "content": (
                            [
                                {
                                    "key": "nominal",
                                    "value": {
                                        "nodetype": "binning",
                                        "input": "NumTrueInteractions",
                                        "flow": "clamp",
                                        "edges": list(ratioBins),
                                        "content": list(nomRatio),
                                    },
                                }
                            ]
                            + (
                                [
                                    {
                                        "key": "up",
                                        "value": {
                                            "nodetype": "binning",
                                            "input": "NumTrueInteractions",
                                            "flow": "clamp",
                                            "edges": list(ratioBins),
                                            "content": list(upRatio),
                                        },
                                    }
                                ]
                                if upRatio is not None
                                else []
                            )
                            + (
                                [
                                    {
                                        "key": "down",
                                        "value": {
                                            "nodetype": "binning",
                                            "input": "NumTrueInteractions",
                                            "flow": "clamp",
                                            "edges": list(ratioBins),
                                            "content": list(downRatio),
                                        },
                                    }
                                ]
                                if downRatio is not None
                                else []
                            )
                        ),
                    },
                }
            ],
        }
    elif args.format == "cp3-llbb":
        out = {
            "dimension": 1,
            "variables": ["NumTrueInteractions"],
            "binning": {"x": list(ratioBins)},
            "error_type": "absolute",
            "data": [
                {
                    "bin": [ratioBins[i], ratioBins[i + 1]],
                    "value": nomRatio[i],
                    "error_low": (nomRatio[i] - downRatio[i] if downRatio is not None else 0.0),
                    "error_high": (upRatio[i] - nomRatio[i] if upRatio is not None else 0.0),
                }
                for i in range(nomRatio.shape[0])
            ],
        }
    else:
        raise ValueError(f"Unsupported output format: {args.format}")
    if args.gzip:
        outN = args.output
        if not outN.endswith(".gz"):
            outN = outN + ".gz"
        with gzip.open(outN, "wb") as outF, io.TextIOWrapper(outF, encoding="utf-8") as outE:
            json.dump(out, outE)
    else:
        with open(args.output, "w") as outF:
            json.dump(out, outF)

    if args.makePlot:
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        rax.set_yscale("log")
        # rax = ax.twinx()
        dBinCenters = 0.5 * (mcPUBins[:-1] + mcPUBins[1:])
        nBinCenters = 0.5 * (nomBins[:-1] + nomBins[1:])
        rBinCenters = 0.5 * (ratioBins[:-1] + ratioBins[1:])
        ax.hist(dBinCenters, bins=mcPUBins, weights=mcPUVals, histtype="step", label="MC")
        ax.hist(
            nBinCenters, bins=nomBins, weights=nomCont, histtype="step", label="Nominal", color="k"
        )
        rax.hist(rBinCenters, bins=ratioBins, weights=nomRatio, histtype="step", color="k")
        if upCont is not None:
            ax.hist(
                nBinCenters, bins=nomBins, weights=upCont, histtype="step", label="Up", color="r"
            )
            ax.hist(
                nBinCenters,
                bins=nomBins,
                weights=downCont,
                histtype="step",
                label="Down",
                color="b",
            )
            rax.hist(rBinCenters, bins=ratioBins, weights=upRatio, histtype="step", color="r")
            rax.hist(rBinCenters, bins=ratioBins, weights=downRatio, histtype="step", color="b")
        rax.axhline(1.0)
        ax.legend()
        rax.set_ylim(0.02, 2.0)
        rax.set_xlim(ratioBins[0], ratioBins[-1])
        if args.mcfiles:
            rax.set_xlabel(args.mcreweightvar)
        elif args.mcprofile:
            ax.set_title(args.mcprofile)
        if args.output.endswith(".json"):
            plt.savefig(args.output.replace(".json", ".png"))


if __name__ == "__main__":
    main()
