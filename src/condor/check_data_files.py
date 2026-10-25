"""
Checks that the data skims of a tag read every entry of their input files (``HH4b.data_files``):
lists per data sample the files that were lost (not opened, read incompletely, in no job's output)
or read twice, with their jobs, and resubmits those jobs with ``--submit-missing``.

Run it from the directory the jobs were submitted from (job scripts in condor/skimmer/<tag>/):
    python src/condor/check_data_files.py --data-dir <skimmer output dir> --tag <tag> --year <year>
        [--submit-missing]

Author(s): Zichun Hao
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from HH4b.data_files import file_problems, index_files, lfn, read_records
from HH4b.hh_vars import DATA_SAMPLES


def job_of_files(condor_dir: Path, year: str, sample: str, files: list[str]) -> dict[str, int]:
    """{file: job}, from --starti/--endi of the job scripts."""
    jobs = {}
    for script in condor_dir.glob(f"{year}_{sample}_*.sh"):
        job = int(script.stem.rsplit("_", 1)[1])
        match = re.search(r"--starti (\d+) --endi (-?\d+)", script.read_text())
        starti, endi = int(match[1]), int(match[2])
        jobs |= dict.fromkeys(files[starti:] if endi < 0 else files[starti:endi], job)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", required=True, help="skimmer output directory")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--nano-version", default="v15")
    parser.add_argument("--submit-missing", action="store_true", help="resubmit the jobs")
    args = parser.parse_args()

    year_dir = Path(args.data_dir) / args.tag / args.year
    condor_dir = Path("condor") / "skimmer" / args.tag
    index = json.loads(Path(f"data/nanoindex_{args.nano_version}.json").read_text())
    samples = sorted(p.name for p in year_dir.iterdir() if p.name.startswith(tuple(DATA_SAMPLES)))
    jobs_to_rerun = set()
    for sample in samples:
        files = index_files(index, args.year, sample)
        records = read_records(year_dir / sample, args.year, sample)
        if records is None:
            print(f"{sample}: no file records (skimmed before they were saved)")
            continue
        job_of = job_of_files(condor_dir, args.year, sample, files)
        problems = file_problems(records, files)
        print(f"{sample}: {len(files)} files, {len(problems)} with problems")
        for fname, (problem, read_by) in sorted(problems.items()):
            jobs = read_by or ([job_of[fname]] if fname in job_of else [])
            print(f"    jobs {jobs}: {lfn(fname)}: {problem}")
            jobs_to_rerun.update((sample, job) for job in jobs)

    if args.submit_missing:
        for sample, job in sorted(jobs_to_rerun):
            jdl = condor_dir / f"{args.year}_{sample}_{job}.jdl"
            subprocess.run(["condor_submit", str(jdl)], check=True)


if __name__ == "__main__":
    main()
