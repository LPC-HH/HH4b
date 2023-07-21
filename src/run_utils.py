import warnings

# from distributed.diagnostics.plugin import WorkerPlugin
import json


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
    if processor == "trigger":
        samples = [f"SingleMu{year[:4]}"]

    redirector = "root://cmsxrootd.fnal.gov//"

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
                sample_fileset[f"{year}_{subsample}"] = [redirector + fname for fname in fnames]

            fileset = {**fileset, **sample_fileset}

    return fileset


def get_xsecs():
    with open("data/xsecs.json") as f:
        xsecs = json.load(f)

    for key, value in xsecs.items():
        if type(value) == str:
            xsecs[key] = eval(value)

    return xsecs


def get_processor(
    processor: str,
    save_systematics: bool = None,
):
    # define processor
    if processor == "trigger":
        from HH4b.processors import JetHTTriggerEfficienciesProcessor

        return JetHTTriggerEfficienciesProcessor()
    elif processor == "skimmer":
        from HH4b.processors import bbbbSkimmer

        return bbbbSkimmer(
            xsecs=get_xsecs(),
            save_systematics=save_systematics,
        )


def parse_common_args(parser):
    parser.add_argument(
        "--processor",
        default="trigger",
        help="Trigger processor",
        type=str,
        choices=["trigger", "skimmer"],
    )

    parser.add_argument("--year", help="year", type=str, required=True, choices=["2022", "2023"])
    parser.add_argument(
        "--nano_version",
        type=str,
        required=True,
        choices=["v10", "v11", "v11_private", "v12"],
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

    # Skimmer args
    # REMEMBER TO PROPAGATE THIS TO SUBMIT TEMPLATE!!
    add_bool_arg(parser, "save-systematics", default=False, help="save systematic variations")
