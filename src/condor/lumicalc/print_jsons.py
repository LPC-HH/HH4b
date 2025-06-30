"""
Run over NanoAOD ntuples to extract run and luminosity block information,

Adapted from https://github.com/cms-lpc-llp/run3_llp_analyzer/blob/62492e6ad3a0eb76eea527ffc8acc41fb9f7c5e3/python/printJson.py
"""

from __future__ import annotations

import argparse
import json
import os
from array import *  # noqa: F403
from itertools import groupby
from operator import *  # noqa: F403
from pathlib import Path

import ROOT as rt
from tqdm import tqdm


def walk(top: rt.TDirectory, topdown: bool = True) -> tuple:
    """
    Recursively walk through ROOT TDirectory structure.

    Args:
        top: ROOT TDirectory to start from
        topdown: If True, yield directories before subdirectories

    Yields:
        tuple: (dirpath, dirnames, filenames, top_directory)
    """
    assert isinstance(top, rt.TDirectory)
    names = [k.GetName() for k in top.GetListOfKeys()]
    dirpath = top.GetPath()
    dirnames = []
    filenames = []

    # Filter names for directories
    for k in names:
        d = top.Get(k)
        if isinstance(d, rt.TDirectory):
            dirnames.append(k)
        else:
            filenames.append(k)

    # Sort
    dirnames.sort()
    filenames.sort()

    # Yield
    if topdown:
        yield dirpath, dirnames, filenames, top
    for dn in dirnames:
        d = top.Get(dn)
        yield from walk(d, topdown)
    if not topdown:
        yield dirpath, dirnames, filenames, top


def convert_tree_to_dict(
    run_lumi_dict: dict, tree: rt.TTree, lumi_branch: str, run_branch: str
) -> dict:
    """
    Extract run and luminosity block information from ROOT TTree.

    Args:
        run_lumi_dict: Existing run-lumi dictionary to update
        tree: ROOT TTree to process
        lumi_branch: Name of luminosity block branch
        run_branch: Name of run number branch

    Returns:
        dict: Updated run-lumi dictionary
    """
    if not (hasattr(tree, lumi_branch) and hasattr(tree, run_branch)):
        print("tree does not contain run and lumi branches, returning empty json")
        return run_lumi_dict

    # Loop over tree to get run, lumi "flat" dictionary
    tree.Draw(">>elist", "", "entrylist")
    elist = rt.gDirectory.Get("elist")
    if tree.GetEntries() == 0:
        return run_lumi_dict

    entry = -1
    while True:
        entry = elist.Next()
        if entry == -1:
            break
        tree.GetEntry(entry)
        run_str = str(getattr(tree, run_branch))
        lumi_val = int(getattr(tree, lumi_branch))

        if run_str in run_lumi_dict:
            current_lumi = run_lumi_dict[run_str]
            if lumi_val not in current_lumi:
                current_lumi.append(lumi_val)
                run_lumi_dict[run_str] = current_lumi
        else:
            run_lumi_dict[run_str] = [lumi_val]

    return run_lumi_dict


def fix_dict(run_lumi_dict: dict) -> dict:
    """
    Group consecutive luminosity blocks into ranges.

    Args:
        run_lumi_dict: Dictionary with run numbers as keys and lumi lists as values

    Returns:
        dict: Dictionary with consecutive lumis grouped into [start, end] ranges
    """
    for run, lumis in run_lumi_dict.items():
        lumi_groups = []
        sorted_lumis = sorted(lumis)
        for _k, g in groupby(enumerate(sorted_lumis), lambda x: x[0] - x[1]):
            consecutive_lumis = [x[1] for x in g]
            lumi_groups.append([consecutive_lumis[0], consecutive_lumis[-1]])
        run_lumi_dict[run] = lumi_groups

    return run_lumi_dict


def main():
    """Main function to process ROOT files and extract run-lumi information."""
    parser = argparse.ArgumentParser(
        description="Extract run and luminosity block information from ROOT files"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="test.json", help="Name of the json file to write to"
    )
    parser.add_argument(
        "-t",
        "--tree-name",
        type=str,
        default="LuminosityBlocks",
        help="Name of tree in ROOT file that contains run and lumi information",
    )
    parser.add_argument(
        "-l",
        "--lumi-branch",
        type=str,
        default="luminosityBlock",
        help="Name of lumi branch in tree",
    )
    parser.add_argument(
        "-r", "--run-branch", type=str, default="run", help="Name of run branch in tree"
    )
    parser.add_argument(
        "-i",
        "--input-list",
        type=str,
        default="test.list",
        help="Name of text file containing list of ROOT files",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")

    args = parser.parse_args()

    run_lumi_dict = {}

    with Path(args.input_list).open("r") as file1:
        lines = file1.readlines()

    for line in tqdm(lines):
        file_path = line.strip()
        root_file = rt.TFile.Open(file_path)
        if not root_file or root_file.IsZombie():
            print(f"Warning: Could not open file {line.strip()}")
            continue

        tree = root_file.Get(args.tree_name)
        run_lumi_dict = convert_tree_to_dict(run_lumi_dict, tree, args.lumi_branch, args.run_branch)

        root_file.Close()

    run_lumi_dict = fix_dict(run_lumi_dict)

    (Path(args.output)).parent.mkdir(parents=True, exist_ok=True)

    with Path(args.output).open("w") as output:
        json.dump(run_lumi_dict, output, sort_keys=True, indent=2)

    if args.verbose:
        print(f"\njson dumped to file {args.output}:")
        os.system(f"cat {args.output}")
        print("\n")
    else:
        print(f"\njson dumped to file {args.output}:")


if __name__ == "__main__":
    main()
