"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

from __future__ import annotations

import argparse
import os
import warnings
from math import ceil
from pathlib import Path
from string import Template

from HH4b import run_utils

t2_redirectors = {
    "lpc": "root://cmseos.fnal.gov//",
    "ucsd": "root://redirector.t2.ucsd.edu:1095//",
}


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with Path(templ_file).open() as f:
        templ = Template(f.read())

    with Path(out_file).open("w") as f:
        f.write(templ.substitute(templ_args))


def main(args):
    # check that branch exists
    run_utils.check_branch(args.git_branch, args.git_user, args.allow_diff_local_repo)
    username = os.environ["USER"]

    if args.site == "lpc":
        try:
            proxy = os.environ["X509_USER_PROXY"]
        except:
            print("No valid proxy. Exiting.")
            exit(1)
    elif args.site == "ucsd":
        if username == "rkansal":
            proxy = "/home/users/rkansal/x509up_u31735"
        elif username == "dprimosc":
            proxy = "/tmp/x509up_u150012"  # "/home/users/dprimosc/x509up_u150012"
    else:
        raise ValueError(f"Invalid site {args.site}")

    if args.site not in args.save_sites:
        warnings.warn(
            f"Your local sit e {args.site} is not in save sites {args.sites}!", stacklevel=1
        )

    t2_prefixes = [t2_redirectors[site] for site in args.save_sites]

    tag = f"{args.tag}_{args.nano_version}_{args.region}"

    # make eos dir
    pdir = Path(f"store/user/{username}/bbbb/{args.processor}/")
    outdir = pdir / tag

    # make local directory
    local_dir = Path(f"condor/{args.processor}/{tag}")
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print("Condor work dir: ", local_dir)

    print(args.subsamples)
    fileset = run_utils.get_fileset(
        args.processor,
        args.year,
        args.nano_version,
        args.samples,
        args.subsamples,
        get_num_files=True,
    )

    print(f"fileset: {fileset}")

    jdl_templ = "src/condor/submit.templ.jdl"
    sh_templ = "src/condor/submit.templ.sh"

    # submit jobs
    nsubmit = 0
    for sample in fileset:
        for subsample, tot_files in fileset[sample].items():
            if args.submit:
                print("Submitting " + subsample)

            sample_dir = outdir / args.year / subsample
            njobs = ceil(tot_files / args.files_per_job)

            for j in range(njobs):
                if args.test and j == 2:
                    break

                prefix = f"{args.year}_{subsample}"
                localcondor = f"{local_dir}/{prefix}_{j}.jdl"
                jdl_args = {"dir": local_dir, "prefix": prefix, "jobid": j, "proxy": proxy}
                write_template(jdl_templ, localcondor, jdl_args)

                localsh = f"{local_dir}/{prefix}_{j}.sh"
                sh_args = {
                    "branch": args.git_branch,
                    "gituser": args.git_user,
                    "script": args.script,
                    "year": args.year,
                    "starti": j * args.files_per_job,
                    "endi": (j + 1) * args.files_per_job,
                    "sample": sample,
                    "subsample": subsample,
                    "processor": args.processor,
                    "maxchunks": args.maxchunks,
                    "chunksize": args.chunksize,
                    "t2_prefixes": " ".join(t2_prefixes),
                    "outdir": sample_dir,
                    "jobnum": j,
                    "nano_version": args.nano_version,
                    "save_root": ("--save-root" if args.save_root else "--no-save-root"),
                    "txbb": args.txbb,
                    "save_systematics": (
                        "--save-systematics" if args.save_systematics else "--no-save-systematics"
                    ),
                    "apply_selection": (
                        "--apply-selection" if args.apply_selection else "--no-apply-selection"
                    ),
                    "region": f"--region {args.region}" if "skimmer" in args.processor else "",
                }
                write_template(sh_templ, localsh, sh_args)
                os.system(f"chmod u+x {localsh}")

                if Path(f"{localcondor}.log").exists():
                    Path(f"{localcondor}.log").unlink()

                if args.submit:
                    os.system("condor_submit %s" % localcondor)
                else:
                    print("To submit ", localcondor)
                nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


def parse_args(parser):
    parser.add_argument("--script", default="src/run.py", help="script to run", type=str)
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument(
        "--outdir", dest="outdir", default="outfiles", help="directory for output files", type=str
    )
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    parser.add_argument(
        "--save-sites",
        default=["lpc"],
        help="tier 2s in which we want to save the files",
        type=str,
        nargs="+",
        choices=["lpc", "ucsd"],
    )
    run_utils.add_bool_arg(
        parser,
        "test",
        default=False,
        help="test run or not - test run means only 2 jobs per sample will be created",
    )
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)
    run_utils.add_bool_arg(
        parser, "submit", default=False, help="submit files as well as create them"
    )
    parser.add_argument("--git-branch", required=True, help="git branch to use", type=str)
    parser.add_argument("--git-user", default="LPC-HH", help="which user's repo to use", type=str)
    run_utils.add_bool_arg(
        parser,
        "allow-diff-local-repo",
        default=False,
        help="Allow the local repo to be different from the specified remote repo (not recommended!)."
        "If false, submit script will exit if the latest commits locally and on Github are different.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_utils.parse_common_args(parser)
    parse_args(parser)
    args = parser.parse_args()
    main(args)
