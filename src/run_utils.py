import warnings

# from distributed.diagnostics.plugin import WorkerPlugin
import json
import numpy as np

from xsecs import xsecs


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


def get_fileset(
    processor: str,
    year: int,
    version: str,
    samples: list,
    subsamples: list,
    starti: int = 0,
    endi: int = -1,
    get_num_files: bool = False,
    coffea_casa: str = False,
):

    with open(f"data/nanoindex_{version}.json", "r") as f:
        full_fileset_nano = json.load(f)

    fileset = {}

    for sample in samples:
        sample_set = full_fileset_nano[year][sample]

        set_subsamples = list(sample_set.keys())

        # check if any subsamples for this sample have been specified
        get_subsamples = set(set_subsamples).intersection(subsamples)

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
                fnames = fnames[starti:] if endi < 0 else fnames[starti:endi]
                sample_fileset[f"{year}_{subsample}"] = fnames

            fileset = {**fileset, **sample_fileset}

    return fileset


def get_processor(
    processor: str,
    save_systematics: bool = None,
    save_hist: bool = False,
    save_array: bool = False,
    region: str = None,
):
    # define processor
    if processor == "trigger_boosted":
        from HH4b.processors import BoostedTriggerSkimmer

        return BoostedTriggerSkimmer(save_hist=save_hist)

    elif processor == "matching":
        from HH4b.processors import matchingSkimmer

        return matchingSkimmer(xsecs=xsecs)

    elif processor == "skimmer":
        from HH4b.processors import bbbbSkimmer

        return bbbbSkimmer(
            xsecs=xsecs, save_systematics=save_systematics, region=region, save_array=save_array
        )


def parse_common_args(parser):
    parser.add_argument(
        "--processor",
        required=True,
        help="Trigger processor",
        type=str,
        choices=["trigger_boosted", "skimmer", "matching"],
    )

    parser.add_argument(
        "--year", help="year", type=str, default="2022", choices=["2018", "2022", "2022EE", "2023"]
    )
    parser.add_argument(
        "--nano-version",
        type=str,
        required=True,
        choices=["v9", "v10", "v11", "v11_private", "v12"],
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
    parser.add_argument("--region", help="region", default="signal", choices=["signal"], type=str)
    add_bool_arg(parser, "save-systematics", default=False, help="save systematic variations")
    add_bool_arg(parser, "save-array", default=False, help="save array (for dask)")
    add_bool_arg(
        parser,
        "save-hist",
        default=False,
        help="save histogram as output of the processor (for trigger processor)",
    )


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
