# from distributed.diagnostics.plugin import WorkerPlugin
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from colorama import Fore, Style

from .xsecs import xsecs


def add_bool_arg(parser, name, help, default=False, no_name=None):
    """Add a boolean command line argument for argparse"""
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


def add_mixins(nanoevents):
    # for running on condor
    nanoevents.PFNanoAODSchema.mixins["SubJet"] = "FatJet"
    nanoevents.PFNanoAODSchema.mixins["PFCands"] = "PFCand"
    nanoevents.PFNanoAODSchema.mixins["SV"] = "PFCand"


def print_red(s):
    return print(f"{Fore.RED}{s}{Style.RESET_ALL}")


def check_branch(git_branch: str, git_user: str = "LPC-HH", allow_diff_local_repo: bool = False):
    """Check that specified git branch exists in the repo, and local repo is up-to-date"""
    assert not bool(
        os.system(
            f'git ls-remote --exit-code --heads "https://github.com/{git_user}/HH4b" "{git_branch}"'
        )
    ), f"Branch {git_branch} does not exist"

    print(f"Using branch {git_branch}")

    # check if there are uncommitted changes
    uncommited_files = int(subprocess.getoutput("git status -s | wc -l"))

    if uncommited_files:
        print_red("There are local changes that have not been committed!")
        os.system("git status -s")
        if allow_diff_local_repo:
            print_red("Proceeding anyway...")
        else:
            print_red("Exiting! Use the --allow-diff-local-repo option to override this.")
            sys.exit(1)

    # check that the local repo's latest commit matches that on github
    remote_hash = subprocess.getoutput(f"git show origin/{git_branch} | head -n 1").split(" ")[1]
    local_hash = subprocess.getoutput("git rev-parse HEAD")

    if remote_hash != local_hash:
        print_red("Latest local and github commits do not match!")
        print(f"Local commit hash: {local_hash}")
        print(f"Remote commit hash: {remote_hash}")
        if allow_diff_local_repo:
            print_red("Proceeding anyway...")
        else:
            print_red("Exiting! Use the --allow-diff-local-repo option to override this.")
            sys.exit(1)


def get_fileset(
    processor: str,  # noqa: ARG001
    year: int,
    version: str,
    samples: list,
    subsamples: list,
    starti: int = 0,
    endi: int = -1,
    get_num_files: bool = False,
    # coffea_casa: str = False,
):
    with Path(f"data/nanoindex_{version}.json").open() as f:
        full_fileset_nano = json.load(f)

    fileset = {}

    for sample in samples:
        sample_set = full_fileset_nano[year][sample]

        set_subsamples = list(sample_set.keys())

        # check if any subsamples for this sample have been specified
        get_subsamples = set(set_subsamples).intersection(subsamples)

        if len(subsamples):
            for subs in subsamples:
                if subs not in get_subsamples:
                    raise ValueError(f"Subsample {subs} not found for sample {sample}!")

        # if so keep only that subset
        if len(get_subsamples):
            sample_set = {subsample: sample_set[subsample] for subsample in get_subsamples}

        if get_num_files:
            # return only the number of files per subsample (for splitting up jobs)
            fileset[sample] = {}
            for subsample, fnames in sample_set.items():
                fileset[sample][subsample] = len(fnames)

        else:
            # return all files per subsample
            sample_fileset = {}

            for subsample, fnames in sample_set.items():
                run_fnames = fnames[starti:] if endi < 0 else fnames[starti:endi]
                sample_fileset[f"{year}_{subsample}"] = run_fnames

            fileset = {**fileset, **sample_fileset}

    return fileset


def get_processor(
    processor: str,
    save_systematics: bool | None = None,
    region: str | None = None,
    apply_selection: bool | None = None,
    nano_version: str | None = None,
    pnet_txbb: str | None = None,
):
    # define processor
    if processor == "matching":
        from HH4b.processors import matchingSkimmer

        print(apply_selection)
        return matchingSkimmer(
            xsecs=xsecs, apply_selection=apply_selection, nano_version=nano_version
        )

    if processor == "skimmer":
        from HH4b.processors import bbbbSkimmer

        return bbbbSkimmer(
            xsecs=xsecs,
            save_systematics=save_systematics,
            region=region,
            nano_version=nano_version,
            pnet_txbb=pnet_txbb,
        )

    if processor == "ttSkimmer":
        from HH4b.processors import ttSkimmer

        return ttSkimmer(
            xsecs=xsecs,
            save_systematics=save_systematics,
            region=region,
            nano_version=nano_version,
        )

    if processor == "vpt":
        from HH4b.processors import vptProc

        return vptProc()


def parse_common_args(parser):
    parser.add_argument(
        "--processor",
        required=True,
        help="processor",
        type=str,
        choices=["skimmer", "matching", "ttSkimmer", "vpt"],
    )

    parser.add_argument(
        "--year",
        help="year",
        type=str,
        default="2022",
        choices=["2018", "2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument(
	"--pnet-txbb",
	type=str,
        required=True,
        choices=[
            "legacy", "v12", "part"
	],
        help="PNetTXbb version to be used to order FatJets",
    )
    parser.add_argument(
        "--nano-version",
        type=str,
        required=True,
        choices=[
            "v9",
            "v9_private",
            "v9_hh_private",
            "v10",
            "v11",
            "v11_private",
            "v12",
            "v12_private",
            "v12v2_private",
        ],
        help="NanoAOD version",
    )
    parser.add_argument(
        "--samples",
        default=[],
        help="which samples to run",  # , default will be all samples",
        nargs="*",
    )
    parser.add_argument(
        "--subsamples",
        default=[],
        help="which subsamples, by default will be all in the specified sample(s)",
        nargs="*",
    )

    parser.add_argument("--maxchunks", default=0, help="max chunks", type=int)
    parser.add_argument("--chunksize", default=10000, help="chunk size", type=int)
    parser.add_argument(
        "--region",
        help="region",
        default="signal",
        choices=["pre-sel", "signal", "semilep-tt", "had-tt"],
        type=str,
    )
    add_bool_arg(parser, "save-systematics", default=False, help="save systematic variations")
    parser.add_argument("--apply-selection", dest="apply_selection", action="store_true", help=help)
    parser.add_argument(
        "--no-apply-selection", dest="apply_selection", action="store_false", help=help
    )
    add_bool_arg(parser, "save-array", default=False, help="save array (for dask)")
    add_bool_arg(parser, "save-root", default=False, help="save root ntuples too")


def flatten_dict(var_dict: dict):
    """
    Flattens dictionary of variables so that each key has a 1d-array
    """
    new_dict = {}
    for key, var in var_dict.items():
        num_objects = var.shape[-1]
        if len(var.shape) >= 2 and num_objects > 1:
            temp_dict = {f"{key}{obj}": var[:, obj] for obj in range(num_objects)}
            new_dict = {**new_dict, **temp_dict}
        else:
            new_dict[key] = np.squeeze(var)

    return new_dict
