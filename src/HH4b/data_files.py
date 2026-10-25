"""
Bookkeeping of the input files of the data skims, saved in the job pickles by run.py and the
skimmer: per file, the entries run.py found when it opened it, the entries the skimmer read, and,
for the files that did not open after retries, the error.

Author(s): Zichun Hao
"""

from __future__ import annotations

import pickle
from pathlib import Path


def lfn(fname: str) -> str:
    return "/store/" + fname.split("/store/", 1)[1]


def index_files(index: dict, year: str, sample: str) -> list[str]:
    """Files of a data sample in the nano index, in the order the jobs were split."""
    for category in index[year].values():
        if sample in category:
            return category[sample]
    raise KeyError(f"{sample} ({year}) is not in the nano index")


def read_records(sample_dir: Path, year: str, sample: str) -> dict[int, dict] | None:
    """{job: {"file_entries": .., "processed_entries": .., "bad_files": ..}} of the job pickles
    (``pickles/out_<job>.pkl``) of a data sample; None if they hold no file records (skims made
    before they were saved)."""
    records = {}
    for pkl in sorted((sample_dir / "pickles").glob("out_*.pkl")):
        with pkl.open("rb") as f:
            out = pickle.load(f).get(year, {}).get(sample, {})
        if "file_entries" not in out and "bad_files" not in out:
            return None
        records[int(pkl.stem.rsplit("_", 1)[1])] = out
    return records


def file_problems(records: dict[int, dict], files: list[str]) -> dict[str, tuple[str, list[int]]]:
    """{file: (problem, jobs)} of the ``files`` of a sample (its nano index files) that were lost
    (not opened, read incompletely, in no job's records) or read twice."""
    problems, read_by = {}, {}
    for job, out in sorted(records.items()):
        entries, bad = out.get("file_entries", {}), out.get("bad_files", {})
        processed = out.get("processed_entries", {})
        for fname, error in bad.items():
            problems[fname] = (f"not opened: {error}", [job])
            read_by.setdefault(fname, []).append(job)
        for fname, n in entries.items():
            read_by.setdefault(fname, []).append(job)
            if processed.get(fname, 0) != n:
                problems[fname] = (f"read {processed.get(fname, 0)} of {n} entries", [job])
    for fname, jobs in read_by.items():
        if len(jobs) > 1:
            problems[fname] = ("read twice", jobs)
    for fname in files:
        if fname not in read_by:
            problems[fname] = ("in no job's output", [])
    return problems
