from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import hist
import numpy as np
import uproot

from HH4b import plotting
from HH4b.utils import ShapeVar

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fit-file",
    help="fitdiagnostics output root file",
    required=True,
    type=str,
)
parser.add_argument("--plots-dir", help="plots directory", type=str)
parser.add_argument("--signal-scale", help="scale signal by", default=1.0, type=float)
parser.add_argument("--bin-name", default="passbin1", help="pass category", type=str)
args = parser.parse_args()

signal_scale = args.signal_scale

os.system(f"mkdir -p {args.plots_dir}")
plot_dir = args.plots_dir

# (name in templates -> name in cards)
hist_label_map_inverse = OrderedDict(
    [
        ("qcd", "CMS_bbbb_hadronic_qcd_datadriven"),
        ("vjets", "vjets"),
        ("diboson", "diboson"),
        ("ttbar", "ttbar"),
        ("vhtobb", "vhtobb"),
        ("hh4b", "hh4b"),
        ("data", "data_obs"),
    ]
)
hist_label_map = {val: key for key, val in hist_label_map_inverse.items()}
samples = list(hist_label_map.values())

fit_shape_var = ShapeVar(
    # "H2Msd",
    "H2PNetMass",
    # r"$m^{2}_\mathrm{SD}$ (GeV)",
    r"$m^{2}_\mathrm{reg}$ (GeV)",
    [16, 60, 220],
    # [17, 50, 220],
    reg=True,
    blind_window=[110, 140],
)
shape_vars = [fit_shape_var]

shapes = {
    "prefit": "Pre-Fit",
    # "postfit": "S+B Post-Fit",
    "postfit": "B-only Post-Fit",
}

selection_regions_labels = {
    "passbin1": "Pass Bin1",
    "fail": "Fail",
}
ylims = {
    "passbin1": 20,
    "fail": 45000,
}

bins = [args.bin_name, "fail"]
selection_regions = {key: selection_regions_labels[key] for key in bins}

data_key = "data"

file = uproot.open(args.fit_file)

print(file.keys())
# build histograms
hists = {}
for shape in shapes:
    hists[shape] = {
        region: hist.Hist(
            hist.axis.StrCategory(samples, name="Sample"),
            *[shape_var.axis for shape_var in shape_vars],
            storage="double",
        )
        for region in selection_regions
    }

    for region in selection_regions:
        h = hists[shape][region]
        templates = file[f"{region}_{shape}"]
        # print(templates)
        for key, file_key in hist_label_map_inverse.items():
            if key != data_key:
                if file_key not in templates:
                    print(f"No {key} in {region}")
                    continue

                data_key_index = np.where(np.array(list(h.axes[0])) == key)[0][0]
                h.view(flow=False)[data_key_index, :] = templates[file_key].values()

        data_key_index = np.where(np.array(list(h.axes[0])) == data_key)[0][0]
        h.view(flow=False)[data_key_index, :] = np.nan_to_num(
            templates[hist_label_map_inverse[data_key]].values()
        )

year = "2022-2023"
pass_ratio_ylims = [0, 2]
fail_ratio_ylims = [0, 2]

for shape, shape_label in shapes.items():
    for region, region_label in selection_regions.items():
        pass_region = region.startswith("pass")
        for shape_var in shape_vars:
            # print(hists[shape][region])
            plot_params = {
                "hists": hists[shape][region],
                "sig_keys": ["hh4b"],
                "sig_scale_dict": {"hh4b": signal_scale},
                "bg_keys": ["qcd", "ttbar", "vhtobb", "vjets", "diboson"],
                "show": True,
                "year": year,
                "ylim": ylims[region],
                "xlim": 220,
                # "xlim_low": 50,
                "xlim_low": 60,
                "ratio_ylims": pass_ratio_ylims if pass_region else fail_ratio_ylims,
                "title": f"{shape_label} {region_label} Region",
                "name": f"{plot_dir}/{shape}_{region}_{shape_var.var}.pdf",
                "bg_order": ["diboson", "vjets", "vhtobb", "ttbar", "qcd"],
                "energy": 13.6,
            }

            plotting.ratioHistPlot(**plot_params)
