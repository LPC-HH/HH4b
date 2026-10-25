"""
Copy the newest full-year golden JSONs of Run 3 from the CMS CAT area on CVMFS into
``data/golden_json/``, one file per year (``Cert_Collisions<year>_<first run>_<last run>_Golden.json``).

The newest file is the one CAT links as ``Collisions<yy>/latest/golden_json_latest.json``; its
certified name is the ``Cert_Collisions<year>_*_Golden.json`` with the same content. An older file of
the same year in ``data/golden_json/`` is replaced. Files are compared by content and written with
a final newline (as the pre-commit hooks want). Without ``--update`` nothing is written.

Usage: python fetch_golden_json.py [--years 2022 2023 2024 2025] [--update]

Author(s): Zichun Hao
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CAT_DC = Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/DC")
OUT_DIR = Path(__file__).resolve().parent / "data" / "golden_json"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def newest_golden(year: str) -> tuple[Path, Path]:
    """(certified file, dated CAT folder holding it) of the newest full-year golden JSON."""
    folder = CAT_DC / f"Collisions{year[2:]}"
    pointer = md5(folder / "latest" / "golden_json_latest.json")
    named = [
        f
        for f in sorted((folder / "latest").glob(f"Cert_Collisions{year}_*_Golden.json"))
        if "_era" not in f.name and md5(f) == pointer
    ]
    if len(named) != 1:
        raise RuntimeError(f"{year}: {len(named)} certified files match golden_json_latest.json")
    dated = sorted(
        d for d in folder.iterdir() if d.name[:1].isdigit() and (d / named[0].name).is_file()
    )
    source = next(d for d in reversed(dated) if md5(d / named[0].name) == pointer)
    return source / named[0].name, source


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--years", nargs="+", default=["2022", "2023", "2024", "2025"])
    parser.add_argument("--update", action="store_true", help="write the new files")
    args = parser.parse_args()

    for year in args.years:
        source, folder = newest_golden(year)
        target = OUT_DIR / source.name
        old = [f for f in OUT_DIR.glob(f"Cert_Collisions{year}_*_Golden.json") if f != target]
        same = target.is_file() and json.loads(target.read_text()) == json.loads(source.read_text())
        if same and not old:
            print(f"{year}: {source.name} is up to date (CAT {folder.parent.name}/{folder.name})")
            continue
        replaced = [f.name for f in old]
        print(
            f"{year}: {source.name} from CAT {folder.parent.name}/{folder.name}, replaces {replaced}"
        )
        if args.update:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text().rstrip("\n") + "\n")
            for f in old:
                f.unlink()


if __name__ == "__main__":
    main()
