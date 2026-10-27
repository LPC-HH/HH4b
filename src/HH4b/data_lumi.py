"""
Luminosity the data skims of a tag hold: ``<tag>/lumi_<year>.json``, one per analysis year, written
by ``lumi_corrections`` the first time the year is loaded. It exists only once the year's data
skims have been checked: ``{}`` if they hold every golden lumisection counted in ``LUMI[year]``;
else the golden lumisections they lost (``lost_lumis``) and the luminosity they hold (``lumi_pb``).
``utils.load_samples`` drops the lost lumisections from the data and scales the MC used for the year
to ``lumi_pb``. Delete the file when the data skims change; the next load writes it again.

Lost lumisections: those of the files the job records (``HH4b.data_files``) show as lost, from DAS
for /store/data files, else from the file itself; for skims made before those records, the golden
lumisections without any data event in the skims. A lumisection is split between JetMET0 and
JetMET1 and over several files, so it is dropped as a whole. ``lumi_pb`` = ``LUMI[year]`` x (1 - f),
f the fraction of the golden luminosity in the lost lumisections, from the per-lumisection recorded
luminosity of the DC pileup JSONs.

Author(s): Zichun Hao
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import uproot

from HH4b.data_files import file_problems, index_files, lfn, read_records
from HH4b.hh_vars import DATA_ERAS, DATA_SAMPLES, GOLDEN_JSON, LUMI

logger = logging.getLogger(__name__)

GOLDEN_JSON_DIR = Path(__file__).resolve().parent / "corrections" / "data" / "golden_json"
NANO_INDEX = Path(__file__).resolve().parents[2] / "data" / "nanoindex_v15.json"
_DC = Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/DC")
PILEUP_JSON = {
    "2022": _DC / "Collisions22/latest/PileUp/BCDEFG/pileup_JSON.txt",
    "2022EE": _DC / "Collisions22/latest/PileUp/BCDEFG/pileup_JSON.txt",
    "2023": _DC / "Collisions23/latest/PileUp/BCD/pileup_JSON.txt",
    "2023BPix": _DC / "Collisions23/latest/PileUp/BCD/pileup_JSON.txt",
    "2024": _DC / "Collisions24/latest/PileUp/pileup_JSON-2024BCDEFGHI_Golden.txt",
    "2025": _DC / "Collisions25/latest/PileUp/pileup_JSON-2025_Golden.txt",
}
DASGOCLIENT = "/cvmfs/cms.cern.ch/common/dasgoclient"
LS_SCALE = 1_000_000  # key of a lumisection: run x LS_SCALE + LS


def keys_of(lumis: dict) -> np.ndarray:
    """Sorted keys of the lumisections of a CMS JSON ({run: [[first, last], ...]})."""
    keys = [
        int(run) * LS_SCALE + ls
        for run, ranges in lumis.items()
        for first, last in ranges
        for ls in range(first, last + 1)
    ]
    return np.unique(np.array(keys, dtype=np.int64))


def cms_json(keys: np.ndarray) -> dict[str, list[list[int]]]:
    out = {}
    for key in np.unique(keys):
        run, ls = str(key // LS_SCALE), int(key % LS_SCALE)
        ranges = out.setdefault(run, [])
        if ranges and ranges[-1][1] == ls - 1:
            ranges[-1][1] = ls
        else:
            ranges.append([ls, ls])
    return out


def golden_keys(year: str) -> np.ndarray:
    """The golden lumisections of the analyzed eras of ``year`` (those ``LUMI[year]`` counts)."""
    keys = keys_of(json.loads((GOLDEN_JSON_DIR / GOLDEN_JSON[year]).read_text()))
    runs = keys // LS_SCALE
    in_eras = np.zeros(len(keys), dtype=bool)
    for first, last in DATA_ERAS[year].values():
        in_eras |= (runs >= first) & (runs <= last)
    return keys[in_eras]


def file_lumisections(fname: str) -> np.ndarray:
    """Lumisection keys of a file: from DAS for /store/data files, else from the file."""
    if lfn(fname).startswith("/store/data/"):
        das = Path(DASGOCLIENT) if Path(DASGOCLIENT).is_file() else Path("dasgoclient")
        query = [str(das), f"-query=lumi file={lfn(fname)}", "-json"]
        out = subprocess.run(query, capture_output=True, text=True, check=True).stdout
        lumis = [ls for record in json.loads(out) for ls in record["lumi"]]
        keys = [int(ls["run_number"]) * LS_SCALE + int(ls["lumi_section_num"]) for ls in lumis]
    else:
        with uproot.open(fname) as f:
            arrays = f["Events"].arrays(["run", "luminosityBlock"], library="np")
        keys = arrays["run"].astype(np.int64) * LS_SCALE + arrays["luminosityBlock"]
    if not len(keys):
        raise RuntimeError(f"no lumisections found for {lfn(fname)}")
    return np.unique(np.asarray(keys, dtype=np.int64))


def parquet_keys(parquet: Path) -> np.ndarray:
    """Lumisection keys with events in one skim file."""
    if not pq.read_metadata(parquet).num_rows:  # empty skim files have no columns
        return np.empty(0, np.int64)
    table = pq.read_table(parquet, columns=["('run', '0')", "('luminosityBlock', '0')"])
    run, ls = (table.column(i).to_numpy().astype(np.int64) for i in range(2))
    return np.unique(run * LS_SCALE + ls)


def skim_keys(sample_dirs: list[Path]) -> np.ndarray:
    """Lumisection keys with events in the parquet files of data samples."""
    files = [f for d in sample_dirs for f in sorted((d / "parquet").glob("*.parquet"))]
    with ThreadPoolExecutor(16) as pool:
        per_file = list(pool.map(parquet_keys, files))
    return np.unique(np.concatenate([np.empty(0, np.int64), *per_file]))


def lumi_fraction(lost: np.ndarray, golden: np.ndarray, year: str) -> float:
    """Fraction of the recorded luminosity of ``golden`` in ``lost``, from the DC pileup JSON."""
    pileup = json.loads(PILEUP_JSON[year].read_text())
    pu_keys = np.array([int(r) * LS_SCALE + int(x[0]) for r, rows in pileup.items() for x in rows])
    pu_lumi = np.array([float(x[1]) for rows in pileup.values() for x in rows])
    return float(pu_lumi[np.isin(pu_keys, lost)].sum() / pu_lumi[np.isin(pu_keys, golden)].sum())


def check_year(tag_dir: Path, year: str) -> dict | None:
    """Content of ``<tag>/lumi_<year>.json``; None if the tag has no data skims of ``year``."""
    year_dir = tag_dir / year
    samples = [p for p in sorted(year_dir.glob("*")) if p.name.startswith(tuple(DATA_SAMPLES))]
    if not samples:
        return None
    golden = golden_keys(year)
    records = {p.name: read_records(p, year, p.name) for p in samples}
    if all(r is not None for r in records.values()):
        method = "file records"
        index = json.loads(NANO_INDEX.read_text())
        lost = [np.empty(0, np.int64)]
        for sample, sample_records in records.items():
            problems = file_problems(sample_records, index_files(index, year, sample))
            for fname, (problem, jobs) in sorted(problems.items()):
                logger.warning(f"{year} {sample}: {lfn(fname)} (jobs {jobs}): {problem}")
                if problem != "read twice":
                    lost.append(file_lumisections(fname))
        lost = np.intersect1d(golden, np.concatenate(lost))
    else:
        method = "coverage of the skims (no file records)"
        lost = np.setdiff1d(golden, skim_keys(samples))
    if not len(lost):
        return {}
    fraction = lumi_fraction(lost, golden, year)
    logger.warning(f"{year}: {100 * fraction:.3g}% of the golden luminosity lost ({method})")
    return {
        "lumi_pb": LUMI[year] * (1 - fraction),
        "lost_fraction": fraction,
        "method": method,
        "lost_lumis": cms_json(lost),
    }


@lru_cache(maxsize=None)
def lumi_corrections(tag_dir: Path, year: str) -> dict:
    """``<tag>/lumi_<year>.json``, written first if it is missing ({} without data skims)."""
    path = Path(tag_dir) / f"lumi_{year}.json"
    if path.is_file():
        return json.loads(path.read_text())
    result = check_year(Path(tag_dir), year)
    if result is None:
        logger.warning(f"No data skims of {year} in {tag_dir}: luminosity not checked")
        return {}
    try:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(result) + "\n")
        tmp.replace(path)
    except OSError as e:
        logger.warning(f"Could not write {path} ({e}); using the result of this check only")
    return result
