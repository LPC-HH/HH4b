"""
Runs the submit script but with samples specified in a yaml file.

Author(s): Raghav Kansal, Cristina Mantilla Suarez
"""
from __future__ import annotations

import argparse
from pathlib import Path

import submit
import yaml

from HH4b import run_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_utils.parse_common_args(parser)
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument("--jet", default="AK8", help="jet", type=str)
    parser.add_argument(
        "--submit", dest="submit", action="store_true", help="submit jobs when created"
    )
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    parser.add_argument("--yaml", default="", help="yaml file", type=str)

    args = parser.parse_args()

    with Path(args.yaml).open() as file:
        samples_to_submit = yaml.safe_load(file)

    args.script = "run.py"
    args.outdir = "outfiles"
    args.test = False
    tag = args.tag
    for key, tdict in samples_to_submit.items():
        print(f"Submitting for year {key}")
        args.year = key
        for sample, sdict in tdict.items():
            args.samples = [sample]
            args.subsamples = sdict.get("subsamples", [])
            args.files_per_job = sdict["files_per_job"]
            args.maxchunks = sdict.get("maxchunks", 0)
            args.chunksize = sdict.get("chunksize", 100000)
            args.tag = tag

            print(args)
            submit.main(args)
