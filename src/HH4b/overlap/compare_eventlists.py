"""
Compare event lists from two ROOT files (e.g. current branch vs previous version).

Usage:
    python compare_eventlists.py --current path/to/current/eventlist_boostedHH4b_2022.root \\
        --previous path/to/previous/eventlist_boostedHH4b_2022.root \\
        [--years 2022 2022EE 2023]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uproot


def load_event_set(filepath: Path, tree_name: str) -> set[tuple[int, int, int]] | None:
    """Load event keys (run, lumi, event) from a tree. Returns None if tree does not exist."""
    if not filepath.exists():
        return None
    with uproot.open(filepath) as f:
        if tree_name not in f:
            return None
        tree = f[tree_name]
        run = tree["run"].array(library="np")
        lumi = tree["luminosityBlock"].array(library="np")
        evt = tree["event"].array(library="np")
    return set(zip(run.tolist(), lumi.tolist(), evt.tolist()))


def compare_trees(
    current_path: Path,
    previous_path: Path,
    tree_name: str,
    year: str,
    verbose: bool = False,
) -> dict:
    """Compare event sets for one tree. Returns summary dict."""
    curr = load_event_set(current_path, tree_name)
    prev = load_event_set(previous_path, tree_name)

    result = {
        "tree": tree_name,
        "year": year,
        "current_count": len(curr) if curr else 0,
        "previous_count": len(prev) if prev else 0,
        "only_in_current": 0,
        "only_in_previous": 0,
        "in_both": 0,
    }

    if curr is None and prev is None:
        result["status"] = "both_missing"
        return result
    if curr is None:
        result["status"] = "current_missing"
        result["only_in_previous"] = len(prev)
        return result
    if prev is None:
        result["status"] = "previous_missing"
        result["only_in_current"] = len(curr)
        return result

    only_curr = curr - prev
    only_prev = prev - curr
    both = curr & prev

    result["only_in_current"] = len(only_curr)
    result["only_in_previous"] = len(only_prev)
    result["in_both"] = len(both)
    result["status"] = "ok" if (len(only_curr) == 0 and len(only_prev) == 0) else "diff"

    if verbose and (only_curr or only_prev):
        result["sample_only_current"] = list(only_curr)[:5]
        result["sample_only_previous"] = list(only_prev)[:5]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare event lists from current and previous ROOT files."
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current event list ROOT file or directory (e.g. eventlist_boostedHH4b_2022.root)",
    )
    parser.add_argument(
        "--previous",
        type=str,
        required=True,
        help="Path to previous event list ROOT file or directory",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
        help="Years to compare",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample event keys when differences found",
    )
    args = parser.parse_args()

    current_base = Path(args.current)
    previous_base = Path(args.previous)

    # Resolve paths: if dir, use eventlist_boostedHH4b_{year}.root
    def get_path(base: Path, year: str) -> Path:
        if base.is_dir():
            return base / f"eventlist_boostedHH4b_{year}.root"
        return base

    all_ok = True
    for year in args.years:
        curr_path = get_path(current_base, year)
        prev_path = get_path(previous_base, year)

        if not curr_path.exists():
            print(f"[{year}] Current file not found: {curr_path}")
            continue
        if not prev_path.exists():
            print(f"[{year}] Previous file not found: {prev_path}")
            continue

        with uproot.open(curr_path) as f:
            trees = [k.split(";")[0] for k in f.keys()]

        print(f"\n=== {year} ===")
        for tree_name in trees:
            r = compare_trees(curr_path, prev_path, tree_name, year, verbose=args.verbose)
            status = "OK" if r["status"] in ("ok", "both_missing") else "DIFF"
            if r["status"] not in ("ok", "both_missing"):
                all_ok = False

            print(f"  {tree_name}: curr={r['current_count']} prev={r['previous_count']} "
                  f"both={r['in_both']} only_curr={r['only_in_current']} only_prev={r['only_in_previous']} [{status}]")
            if args.verbose and r.get("sample_only_current"):
                print(f"    sample only in current: {r['sample_only_current']}")
            if args.verbose and r.get("sample_only_previous"):
                print(f"    sample only in previous: {r['sample_only_previous']}")

    print("\n" + ("All trees match." if all_ok else "Differences found."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
