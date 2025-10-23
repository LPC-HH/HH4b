from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import hist
import numpy as np
import uproot

from HH4b import plotting, run_utils
from HH4b.utils import ShapeVar


def plot_fits(args):
    signal_scale = args.signal_scale

    os.system(f"mkdir -p {args.plots_dir}")
    plot_dir = args.plots_dir

    # (name in templates -> name in cards)
    hist_label_map_inverse = OrderedDict(
        [
            ("qcd", "CMS_bbbb_hadronic_qcd_datadriven"),
            ("dibosonvjets", ["ZZ", "other_diboson", "vjets"]),
            ("ttbar", "ttbar"),
            ("vhtthtobb", ["VH_hbb", "ttH_hbb"]),
            ("data", "data_obs"),
        ]
    )

    hist_label_map_inverse_sig = OrderedDict(
        [
            ("hh4b", "ggHH_kl_1_kt_1_13p6TeV_hbbhbb"),
            ("hh4b-kl0", "ggHH_kl_0_kt_1_13p6TeV_hbbhbb"),
            ("hh4b-kl2p45", "ggHH_kl_2p45_kt_1_13p6TeV_hbbhbb"),
            ("hh4b-kl5", "ggHH_kl_5_kt_1_13p6TeV_hbbhbb"),
            ("vbfhh4b", "qqHH_CV_1_C2V_1_kl_1_13p6TeV_hbbhbb"),
            ("vbfhh4b-k2v0", "qqHH_CV_1_C2V_0_kl_1_13p6TeV_hbbhbb"),
            ("vbfhh4b-kv1p74-k2v1p37-kl14p4", "qqHH_CV_1p74_C2V_1p37_kl_14p4_13p6TeV_hbbhbb"),
            ("vbfhh4b-kvm0p012-k2v0p03-kl10p2", "qqHH_CV_m0p012_C2V_0p03_kl_10p2_13p6TeV_hbbhbb"),
            ("vbfhh4b-kvm0p758-k2v1p44-klm19p3", "qqHH_CV_m0p758_C2V_1p44_kl_m19p3_13p6TeV_hbbhbb"),
            (
                "vbfhh4b-kvm0p962-k2v0p959-klm1p43",
                "qqHH_CV_m0p962_C2V_0p959_kl_m1p43_13p6TeV_hbbhbb",
            ),
            ("vbfhh4b-kvm1p21-k2v1p94-klm0p94", "qqHH_CV_m1p21_C2V_1p94_kl_m0p94_13p6TeV_hbbhbb"),
            ("vbfhh4b-kvm1p6-k2v2p72-klm1p36", "qqHH_CV_m1p6_C2V_2p72_kl_m1p36_13p6TeV_hbbhbb"),
            ("vbfhh4b-kvm1p83-k2v3p57-klm3p39", "qqHH_CV_m1p83_C2V_3p57_kl_m3p39_13p6TeV_hbbhbb"),
            ("vbfhh4b-kvm2p12-k2v3p87-klm5p96", "qqHH_CV_m2p12_C2V_3p87_kl_m5p96_13p6TeV_hbbhbb"),
        ]
    )
    sig_keys = (
        ["hh4b", "vbfhh4b"] if not args.fit_file_k2v0 else ["hh4b", "vbfhh4b", "vbfhh4b-k2v0"]
    )
    for key in sig_keys:
        hist_label_map_inverse[key] = hist_label_map_inverse_sig[key]

    bkg_keys = ["qcd", "ttbar", "vhtthtobb", "dibosonvjets"]
    bkg_order = ["dibosonvjets", "vhtthtobb", "ttbar", "qcd"]

    samples = list(hist_label_map_inverse.keys())

    if args.mass == "H2Msd":
        fit_shape_var = ShapeVar(
            "H2Msd",
            r"$m_\mathrm{SD}(H_{2})$ (GeV)",
            [16, 60, 220],
            reg=False,
            blind_window=[110, 140],
        )
    else:
        fit_shape_var = ShapeVar(
            "H2PNetMass",
            r"$m_\mathrm{reg}(H_{2})$ (GeV)",
            [16, 60, 220],
            reg=True,
            blind_window=[110, 140],
        )

    shape_vars = [fit_shape_var]

    shapes = {
        "prefit": "Pre-Fit",
        "postfit": "B-only Post-Fit",
    }
    if args.unblinded:
        shapes["postfit"] = "S+B Post-Fit"

    selection_regions_labels = {
        "passvbf": "qqHH SR",
        "passbin1": "ggHH SR 1",
        "passbin2": "ggHH SR 2",
        "passbin3": "ggHH SR 3",
        "fail": "QCD CR",
    }
    ylims = {
        "passvbf": 10,
        "passbin1": 10,
        "passbin2": 40,
        "passbin3": 300,
        "fail": 300000,
    }

    if args.regions == "all":
        signal_regions = ["passbin1", "passbin2", "passbin3", "fail"]
        if args.vbf_region:
            signal_regions = ["passvbf"] + signal_regions
    else:
        signal_regions = [args.regions]
    bins = signal_regions
    selection_regions = {key: selection_regions_labels[key] for key in bins}

    data_key = "data"

    file = uproot.open(args.fit_file)
    if args.fit_file_k2v0:
        file_k2v0 = uproot.open(args.fit_file_k2v0)

    print(file.keys())
    # build histograms
    hists = {}
    bgerrs = {}
    data_errs = {}
    for shape in shapes:
        hists[shape] = {
            region: hist.Hist(
                hist.axis.StrCategory(samples, name="Sample"),
                *[shape_var.axis for shape_var in shape_vars],
                storage="double",
            )
            for region in selection_regions
        }

        bgerrs[shape] = {}
        data_errs[shape] = {}

        for region in selection_regions:
            h = hists[shape][region]
            templates = file[f"{region}_{shape}"]
            if args.fit_file_k2v0:
                templates_k2v0 = file_k2v0[f"{region}_{shape}"]
            for key, file_key in hist_label_map_inverse.items():
                if key != data_key:
                    if isinstance(file_key, list):
                        # sum components
                        values = None
                        for fk in file_key:
                            if fk not in templates:
                                print(f"No {fk} in {region}")
                                continue
                            if values is None:
                                values = templates[fk].values()
                            else:
                                values += templates[fk].values()
                        if values is None:
                            continue
                    else:
                        if file_key not in templates:
                            print(f"No {key} in {region}")
                            continue
                        if key == "vbfhh4b-k2v0" and args.fit_file_k2v0:
                            values = templates_k2v0[file_key].values()
                        else:
                            values = templates[file_key].values()
                    data_key_index = np.where(np.array(list(h.axes[0])) == key)[0][0]
                    h.view(flow=False)[data_key_index, :] = values

            data_key_index = np.where(np.array(list(h.axes[0])) == data_key)[0][0]
            h.view(flow=False)[data_key_index, :] = np.nan_to_num(
                templates[hist_label_map_inverse[data_key]].values()
            )
            bgerrs[shape][region] = templates["TotalBkg"].errors()

    year = "2022-2023"
    pass_ratio_ylims = [0, 2]
    fail_ratio_ylims = [0, 2]

    bg_err_mcstat = {"prefit": True, "postfit": False}
    add_pull = {
        "prefit": False,
        "postfit": True,
    }

    for shape, shape_label in shapes.items():
        for region, region_label in selection_regions.items():
            pass_region = region.startswith("pass")
            for shape_var in shape_vars:
                plot_params = {
                    "hists": hists[shape][region],
                    "sig_keys": sig_keys,
                    "sig_scale_dict": dict(zip(sig_keys, signal_scale)),
                    "bg_keys": bkg_keys,
                    "bg_err": bgerrs[shape][region],
                    "bg_err_mcstat": bg_err_mcstat[shape],
                    "year": year,
                    "ylim": ylims[region],
                    "xlim": 220,
                    "xlim_low": 60,
                    "ratio_ylims": pass_ratio_ylims if pass_region else fail_ratio_ylims,
                    "title": region_label,
                    "name": f"{plot_dir}/{shape}_{region}_{shape_var.var}",
                    "bg_order": bkg_order,
                    "energy": 13.6,
                    "add_pull": add_pull[shape],
                    "prefit_hists": (
                        hists["prefit"][region] if shape == "postfit" and pass_region else None
                    ),
                    "show": False,
                    "unblinded": args.unblinded,
                }

                plotting.ratioHistPlot(**plot_params, data_err=True)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-file",
        help="fitdiagnostics output root file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--fit-file-k2v0",
        help="fitdiagnostics output root file for k2v0 signal",
        required=False,
        type=str,
    )
    parser.add_argument("--plots-dir", help="plots directory", type=str)
    parser.add_argument(
        "--signal-scale", help="scale signal by", default=1.0, nargs="+", type=float
    )
    parser.add_argument(
        "--regions",
        default="all",
        type=str,
        help="regions to plot",
        choices=["passbin1", "passbin2", "passbin3", "passvbf", "fail", "all"],
    )
    parser.add_argument(
        "--mass",
        type=str,
        default="H2PNetMass",
        choices=["H2Msd", "H2PNetMass"],
        help="mass variable to make template",
    )

    run_utils.add_bool_arg(parser, "vbf-region", default=True, help="Include VBF region")
    run_utils.add_bool_arg(parser, "unblinded", "unblinded so skip blinded parts", default=False)

    args = parser.parse_args()

    plot_fits(args)
