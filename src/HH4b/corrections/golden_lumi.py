"""
Luminosity of the analyzed Run 3 data: per era, the recorded luminosity (brilcalc) of the
lumisections of the year's golden JSON (``hh_vars.GOLDEN_JSON``, in ``data/golden_json/``) in the
era's runs. These are the ``hh_vars.LUMI`` values. With ``--tag``, the lumisections the data skims
of that skimmer tag lost (``<tag>/lumi_<year>.json``, ``HH4b.data_lumi``) are left out, and the
golden lumisections that hold events in the skims, with their luminosity, show how much of it the
skims cover.

Writes, per era, ``<outdir>/golden_<year><era>.json`` (with ``--tag`` also
``processed_golden_<year><era>.json``) and ``<outdir>/lumi.json``. brilcalc runs offsite with
``-c web``; ``install_brilcalc.sh`` installs it.

Usage: python golden_lumi.py --brilcalc <path> [--data-dir <skimmer dir> --tag <tag>] [--years ...]

Author(s): Zichun Hao
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from HH4b import data_lumi, hh_vars
from HH4b.utils import GOLDEN_JSON_DIR, check_selector

NORMTAG_DIR = Path("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags")
LS_SCALE = 1_000_000  # key of a lumisection: run x LS_SCALE + LS

# normtag_PHYSICS (the LUM POG recommendation) ends with 2024; 2025: BRIL's best luminometer,
# preliminary (4.6% below normtag_PHYSICS in 2024G)
NORMTAGS = dict.fromkeys(hh_vars.DATA_ERAS, NORMTAG_DIR / "normtag_PHYSICS.json")
NORMTAGS["2025"] = NORMTAG_DIR / "normtag_BRIL.json"


def md5(path: Path | str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def golden_lumisections(path: Path) -> np.ndarray:
    """Sorted keys of the lumisections of a CMS JSON."""
    keys = [
        int(run) * LS_SCALE + ls
        for run, ranges in json.loads(path.read_text()).items()
        for first, last in ranges
        for ls in range(first, last + 1)
    ]
    return np.unique(np.array(keys, dtype=np.int64))


def skim_lumisections(parquet: Path) -> np.ndarray:
    """Keys of the lumisections with events in one skim file."""
    if not pq.read_metadata(parquet).num_rows:  # empty skim files have no columns
        return np.empty(0, np.int64)
    table = pq.read_table(parquet, columns=["('run', '0')", "('luminosityBlock', '0')"])
    run, ls = (table.column(i).to_numpy().astype(np.int64) for i in range(2))
    return np.unique(run * LS_SCALE + ls)


def cms_json(keys: np.ndarray) -> dict[str, list[list[int]]]:
    """{run: [[first LS, last LS], ...]} of sorted unique lumisection keys."""
    out = {}
    for run in np.unique(keys // LS_SCALE):
        ls = keys[keys // LS_SCALE == run] % LS_SCALE
        breaks = np.flatnonzero(np.diff(ls) != 1)
        starts, ends = np.r_[ls[0], ls[breaks + 1]], np.r_[ls[breaks], ls[-1]]
        out[str(run)] = [[int(a), int(b)] for a, b in zip(starts, ends)]
    return out


def brilcalc_recorded(brilcalc: str, lumi_json: Path, normtag: str) -> tuple[float, str]:
    """Recorded luminosity (pb^-1) of the lumisections of ``lumi_json``, and the list brilcalc
    prints of those it has no luminosity for."""
    cmd = [brilcalc, "lumi", "-c", "web", "-b", "STABLE BEAMS", "--normtag", normtag, "-u", "/pb"]
    cmd += ["-i", str(lumi_json), "--output-style", "csv"]
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}  # brilconda's SQLAlchemy, not a user-site one
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env).stdout
    lines = out.splitlines()
    summary = lines[lines.index("#nfill,nrun,nls,ncms,totdelivered(/pb),totrecorded(/pb)") + 1]
    missing = next(line for line in lines if line.startswith("#(run,ls) in json but not in"))
    return float(summary.split(",")[-1]), missing.split(":", 1)[1].strip()


def write_lumi(outdir: Path, name: str, keys: np.ndarray, brilcalc: str, normtag) -> float | None:
    """Write the CMS JSON of ``keys`` and return its luminosity (pb^-1) if there is any."""
    path = outdir / f"{name}.json"
    path.write_text(json.dumps(cms_json(keys)) + "\n")
    if not len(keys):
        return None
    lumi, missing = brilcalc_recorded(brilcalc, path, normtag)
    print(f"    {name}: {len(keys)} lumisections, {lumi:.1f} pb-1, without luminosity: {missing}")
    return lumi


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--brilcalc", required=True, help="brilcalc executable")
    parser.add_argument("--normtag", help="for every year, instead of NORMTAGS")
    parser.add_argument("--years", nargs="+", default=list(hh_vars.DATA_ERAS))
    parser.add_argument("--data-dir", help="skimmer output directory (with --tag)")
    parser.add_argument("--tag", help="skimmer tag whose data skims to check")
    parser.add_argument("--outdir", type=Path, default=Path("golden_lumi"))
    parser.add_argument("--workers", type=int, default=16, help="threads reading the skims")
    args = parser.parse_args()
    if args.tag and not args.data_dir:
        parser.error("--tag needs --data-dir")

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = {"tag": args.tag, "years": {}}
    for year in args.years:
        normtag = args.normtag or NORMTAGS[year]
        golden_path = GOLDEN_JSON_DIR / hh_vars.GOLDEN_JSON[year]
        golden = golden_lumisections(golden_path)
        skim = None
        if args.tag:
            year_dir = Path(args.data_dir) / args.tag / year
            selector = hh_vars.samples_run3[year][hh_vars.data_key]
            files = [
                f
                for p in sorted(year_dir.iterdir())
                if check_selector(p.name, selector)
                for f in sorted((p / "parquet").glob("*.parquet"))
            ]
            with ThreadPoolExecutor(args.workers) as pool:
                per_file = list(pool.map(skim_lumisections, files))
            skim = np.unique(np.concatenate([np.empty(0, np.int64), *per_file]))
        print(f"{year} ({golden_path.name}, {Path(normtag).name})")
        res = {"golden_json": {golden_path.name: md5(golden_path)}}
        res |= {"normtag": {str(normtag): md5(normtag)}, "eras": {}}
        lost = data_lumi.lumi_corrections(Path(args.data_dir) / args.tag, year) if args.tag else {}
        if "lost_lumis" in lost:
            golden = np.setdiff1d(golden, data_lumi.keys_of(lost["lost_lumis"]))
            res["lost_fraction"] = lost["lost_fraction"]
            print(f"  lumisections lost by the skims left out ({lost['method']})")
        for era, (first, last) in hh_vars.DATA_ERAS[year].items():
            in_era = golden[(golden // LS_SCALE >= first) & (golden // LS_SCALE <= last)]
            name = f"golden_{year}{era}"
            era_res = {"lumi_pb": write_lumi(args.outdir, name, in_era, args.brilcalc, normtag)}
            if skim is not None:
                processed = np.intersect1d(in_era, skim)
                name = f"processed_golden_{year}{era}"
                lumi = write_lumi(args.outdir, name, processed, args.brilcalc, normtag)
                era_res["skim_lumi_pb"] = lumi or 0.0
            res["eras"][era] = era_res
        res["lumi_pb"] = sum(r["lumi_pb"] for r in res["eras"].values())
        if skim is not None:
            res["skim_lumi_pb"] = sum(r["skim_lumi_pb"] for r in res["eras"].values())
            print(f"  skims cover {res['skim_lumi_pb']:.1f} of {res['lumi_pb']:.1f} pb-1")
        result["years"][year] = res
    (args.outdir / "lumi.json").write_text(json.dumps(result, indent=1) + "\n")
    print("LUMI (pb^-1):")
    for year, res in result["years"].items():
        print(f'    "{year}": {res["lumi_pb"]:.1f},')
        for era, r in res["eras"].items():
            print(f'    "{year}{era}": {r["lumi_pb"]:.1f},')


if __name__ == "__main__":
    main()
