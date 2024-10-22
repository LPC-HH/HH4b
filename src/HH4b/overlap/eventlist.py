from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import uproot

from HH4b.postprocessing.PostProcess import load_process_run3_samples


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process event lists for HH4b analysis.")
    parser.add_argument("--templates-tag", type=str, default="24June27", help="Templates tag")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/ceph/cms/store/user/cmantill/bbbb/skimmer/",
        help="Data directory",
    )
    parser.add_argument("--tag", type=str, default="24May24_v12_private_signal", help="Tag")
    parser.add_argument("--years", nargs="+", default=["2022"], help="Years to process")
    parser.add_argument("--mass", type=str, default="H2PNetMass", help="Mass")
    parser.add_argument(
        "--bdt-model", type=str, default="24May31_lr_0p02_md_8_AK4Away", help="BDT model"
    )
    parser.add_argument(
        "--bdt-config", type=str, default="24May31_lr_0p02_md_8_AK4Away", help="BDT config"
    )
    parser.add_argument("--txbb-wps", nargs="+", type=float, default=[0.975, 0.92], help="TXbb WPs")
    parser.add_argument(
        "--bdt-wps", nargs="+", type=float, default=[0.98, 0.88, 0.03], help="BDT WPs"
    )
    parser.add_argument("--vbf-txbb-wp", type=float, default=0.95, help="VBF TXbb WP")
    parser.add_argument("--vbf-bdt-wp", type=float, default=0.98, help="VBF BDT WP")
    parser.add_argument("--sig-keys", nargs="+", default=["hh4b", "vbfhh4b"], help="Signal keys")
    parser.add_argument("--pt-first", type=int, default=300, help="PT first")
    parser.add_argument("--pt-second", type=int, default=250, help="PT second")
    parser.add_argument("--bdt-roc", action="store_true", help="BDT ROC")
    parser.add_argument("--control-plots", action="store_true", help="Control plots")
    parser.add_argument("--fom-scan", action="store_true", help="FOM scan")
    parser.add_argument("--fom-scan-bin1", action="store_true", help="FOM scan bin1")
    parser.add_argument("--fom-scan-bin2", action="store_true", help="FOM scan bin2")
    parser.add_argument("--fom-scan-vbf", action="store_true", help="FOM scan VBF")
    parser.add_argument("--templates", action="store_true", help="Templates")
    parser.add_argument("--legacy", action="store_true", help="Legacy")
    parser.add_argument("--vbf", action="store_true", help="VBF")
    parser.add_argument("--vbf-priority", action="store_true", help="VBF priority")
    parser.add_argument("--weight-ttbar-bdt", type=float, default=1, help="Weight TTbar BDT")
    parser.add_argument("--blind", action="store_true", help="Blind")
    parser.add_argument("--out-dir", type=str, help="Output directory")

    return parser.parse_args()


def process_event_list(args):
    bdt_training_keys = ["qcd", "vbfhh4b-k2v0", "hh4b", "ttbar"]
    mass_window = np.array([105, 150])
    ev_dicts = []

    for year in args.years:
        ev_dict, _ = load_process_run3_samples(
            args,
            year=year,
            bdt_training_keys=bdt_training_keys,
            control_plots=False,
            plot_dir="plot_dir",
            mass_window=mass_window,
        )
        ev_dicts.append((year, ev_dict))

    eventlist_dict = [
        "event",
        "bdt_score",
        "bdt_score_vbf",
        "H2TXbb",
        "H2Msd",
        "run",
        "H2PNetMass",
        "luminosityBlock",
    ]

    eventlist_folder = "eventlist_files/" + args.out_dir
    for year, ev_dict in ev_dicts:
        hh4b = ev_dict["hh4b"]
        event_list = hh4b[eventlist_dict]
        array_to_save = {col: event_list[col].to_numpy() for col in event_list.columns}

        with uproot.recreate(f"{eventlist_folder}/eventlist_boostedHH4b_{year}.root") as file:
            file["tree"] = array_to_save

    eventlist_dfs_from_root = {}
    for year in args.years:
        filename = f"{eventlist_folder}/eventlist_boostedHH4b_{year}.root"
        with uproot.open(filename) as file:
            tree = file["tree"]
            arrays = tree.arrays(library="np")
            eventlist = pd.DataFrame(arrays)
            eventlist_dfs_from_root[year] = eventlist

    test_df = eventlist_dfs_from_root["2022"] - eventlist_dfs_from_root["2022EE"]
    print(test_df)
    for year, df in eventlist_dfs_from_root.items():
        print(f"DataFrame for year {year}:\n{df}\n")


if __name__ == "__main__":
    args = parse_arguments()
    process_event_list(args)
