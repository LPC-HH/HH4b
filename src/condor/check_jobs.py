"""
Checks that there is an output for each job submitted.

Author: Raghav Kansal
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from HH4b import run_utils
from HH4b.run_utils import print_red

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--processor",
    default="trigger",
    help="which processor",
    type=str,
    choices=["trigger_boosted", "skimmer", "matching"],
)

parser.add_argument("--tag", default="", help="tag for jobs", type=str)
parser.add_argument("--year", default="2017", help="year", type=str)
parser.add_argument("--user", default="rkansal", help="user", type=str)
parser.add_argument("--location", default="fermilab", help="fermilab or ucsd", type=str)
run_utils.add_bool_arg(parser, "submit-missing", default=False, help="submit missing files")
run_utils.add_bool_arg(
    parser,
    "check-running",
    default=False,
    help="check against running jobs as well (running_jobs.txt will be updated automatically)",
)

args = parser.parse_args()

if args.location == "fermilab":
    eosdir = f"/eos/uscms/store/user/{args.user}/bbbb/{args.processor}/{args.tag}/{args.year}/"
elif args.location == "ucsd":
    eosdir = f"/ceph/cms/store/user/{args.user}/bbbb/{args.processor}/{args.tag}/{args.year}/"

samples = [str(dir_path.name) for dir_path in Path(eosdir).iterdir()]
jdls = [str(jdl.name) for jdl in Path(f"condor/{args.processor}/{args.tag}/").iterdir() if str(jdl).endswith(".jdl")]

jdl_dict = {}
for sample in samples:
    x = [
        int(jdl[:-4].split("_")[-1])
        for jdl in jdls
        if jdl.split("_")[0] == args.year and "_".join(jdl.split("_")[1:-1]) == sample
    ]
    if len(x) > 0:
        jdl_dict[sample] = np.sort(x)[-1] + 1

"""
jdl_dict = {
    sample: np.sort(
        [
            int(jdl[:-4].split("_")[-1])
            for jdl in jdls
            if jdl.split("_")[0] == args.year and "_".join(jdl.split("_")[1:-1]) == sample
        ]
    )[-1]
    + 1
    for sample in samples
}
"""


running_jobs = []
if args.check_running:
    os.system("condor_q | awk '{print $9}' > running_jobs.txt")
    with Path("running_jobs.txt").open() as f:
        lines = f.readlines()

    running_jobs = [s[:-4] for s in lines if s.endswith(".sh\n")]


missing_files = []
err_files = []


for sample in samples:
    print(f"Checking {sample}")

    if args.processor != "trigger":
        if not Path(f"{eosdir}/{sample}/parquet").exists():
            print_red(f"No parquet directory for {sample}!")
            if sample not in jdl_dict:
                continue

            for i in range(jdl_dict[sample]):
                if f"{args.year}_{sample}_{i}" in running_jobs:
                    print(f"Job #{i} for sample {sample} is running.")
                    continue

                jdl_file = f"condor/{args.processor}/{args.tag}/{args.year}_{sample}_{i}.jdl"
                err_file = f"condor/{args.processor}/{args.tag}/logs/{args.year}_{sample}_{i}.err"
                print(jdl_file)
                missing_files.append(jdl_file)
                err_files.append(err_file)
                if args.submit_missing:
                    os.system(f"condor_submit {jdl_file}")

            continue

        outs_parquet = [
            int(str(out).split(".")[0].split("_")[-1])
            for out in Path(f"{eosdir}/{sample}/parquet").iterdir()
        ]
        print(f"Out parquets: {outs_parquet}")

    if not Path(f"{eosdir}/{sample}/pickles").exists():
        print_red(f"No pickles directory for {sample}!")
        continue

    outs_pickles = [
        int(str(out).split(".")[0].split("_")[-1]) for out in Path(f"{eosdir}/{sample}/pickles").iterdir()
    ]

    if args.processor == "trigger":
        print(f"Out pickles: {outs_pickles}")

    for i in range(jdl_dict[sample]):
        if i not in outs_pickles:
            if f"{args.year}_{sample}_{i}" in running_jobs:
                print(f"Job #{i} for sample {sample} is running.")
                continue

            print_red(f"Missing output pickle #{i} for sample {sample}")
            jdl_file = f"condor/{args.processor}/{args.tag}/{args.year}_{sample}_{i}.jdl"
            err_file = f"condor/{args.processor}/{args.tag}/logs/{args.year}_{sample}_{i}.err"
            missing_files.append(jdl_file)
            err_files.append(err_file)
            if args.submit_missing:
                os.system(f"condor_submit {jdl_file}")

        if args.processor != "trigger" and i not in outs_parquet:
            print_red(f"Missing output parquet #{i} for sample {sample}")


print(f"{len(missing_files)} files to re-run:")
for f in missing_files:
    print(f)

print("\nError files:")
for f in err_files:
    print(f)
