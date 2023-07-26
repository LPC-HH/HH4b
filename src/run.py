#!/usr/bin/python

"""
Runs coffea processors on the LPC via either condor or dask.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

import pickle
import os
import argparse

import numpy as np
import uproot

from coffea import nanoevents
from coffea import processor

import run_utils


def run(p: processor, fileset: dict, args):
    """Run processor without fancy dask (outputs then need to be accumulated manually)"""
    run_utils.add_mixins(nanoevents)  # update nanoevents schema

    # outputs are saved here as pickles
    outdir = "./outfiles"
    os.system(f"mkdir -p {outdir}")

    if args.processor in ["skimmer", "input", "ttsfs"]:
        # these processors store intermediate files in the "./outparquet" local directory
        local_dir = os.path.abspath(".")
        local_parquet_dir = os.path.abspath(os.path.join(".", "outparquet"))

        if os.path.isdir(local_parquet_dir):
            os.system(f"rm -rf {local_parquet_dir}")

        os.system(f"mkdir {local_parquet_dir}")

    uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

    if args.executor == "futures":
        executor = processor.FuturesExecutor(status=True)
    else:
        executor = processor.IterativeExecutor(status=True)

    run = processor.Runner(
        executor=executor,
        savemetrics=True,
        schema=nanoevents.PFNanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=None if args.maxchunks == 0 else args.maxchunks,
    )

    out, metrics = run(fileset, "Events", processor_instance=p)

    filehandler = open(f"{outdir}/{args.starti}-{args.endi}.pkl", "wb")
    pickle.dump(out, filehandler)
    filehandler.close()

    # need to combine all the files from these processors before transferring to EOS
    # otherwise it will complain about too many small files
    if args.processor in ["skimmer"]:
        import pandas as pd
        import pyarrow.parquet as pq
        import pyarrow as pa

        pddf = pd.read_parquet(local_parquet_dir)

        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, f"{local_dir}/{args.starti}-{args.endi}.parquet")

        # save as root files as well
        import awkward as ak

        with uproot.recreate(
            f"{local_dir}/nano_skim_{args.starti}-{args.endi}.root", compression=uproot.LZ4(4)
        ) as rfile:
            rfile["Events"] = ak.Array(
                # take only top-level column names in multiindex df
                {key: np.squeeze(pddf[key].values) for key in pddf.columns.levels[0]}
            )


def main(args):
    p = run_utils.get_processor(args.processor, args.save_systematics)

    if len(args.files):
        fileset = {f"{args.year}_{args.files_name}": args.files}
    else:
        fileset = run_utils.get_fileset(
            args.processor,
            args.year,
            args.nano_version,
            args.samples,
            args.subsamples,
            args.starti,
            args.endi,
        )

    print(f"Running on fileset {fileset}")
    run(p, fileset, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_utils.parse_common_args(parser)
    parser.add_argument("--starti", default=0, help="start index of files", type=int)
    parser.add_argument("--endi", default=-1, help="end index of files", type=int)
    parser.add_argument(
        "--executor",
        type=str,
        default="iterative",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    parser.add_argument(
        "--files", default=[], help="set of files to run on instead of samples", nargs="*"
    )
    parser.add_argument(
        "--files-name",
        type=str,
        default="files",
        help="sample name of files being run on, if --files option used",
    )

    args = parser.parse_args()

    main(args)
