"""Provenance JSON written alongside event-list ROOT files (HEFT / overlap handoff)."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

_ROOT_NAME_RE = re.compile(r"eventlist_boostedHH4b_(.+)\.root$", re.IGNORECASE)


def _mass_window_list(mass_window: np.ndarray | list[float]) -> list[float]:
    return np.asarray(mass_window, dtype=float).ravel().tolist()


def build_eventlist_manifest(
    args: Namespace, mass_window: np.ndarray | list[float]
) -> dict[str, Any]:
    """Serializable settings that define how the event list was produced."""
    return {
        "schema_version": SCHEMA_VERSION,
        "mass_window_fom": _mass_window_list(mass_window),
        "mass_variable": args.mass,
        "templates_tag": args.templates_tag,
        "ntuple_tag": args.tag,
        "data_dir": args.data_dir,
        "years": list(args.years),
        "training_years": list(args.training_years) if args.training_years is not None else None,
        "bdt_model": args.bdt_model,
        "bdt_config": args.bdt_config,
        "txbb": args.txbb,
        "txbb_wps": list(args.txbb_wps),
        "bdt_wps": list(args.bdt_wps),
        "vbf_txbb_wp": float(args.vbf_txbb_wp),
        "vbf_bdt_wp": float(args.vbf_bdt_wp),
        "vbf": bool(args.vbf),
        "vbf_priority": bool(args.vbf_priority),
        "pt_first": float(args.pt_first),
        "pt_second": float(args.pt_second),
        "sig_keys": list(args.sig_keys),
        "event_list_data_only": bool(getattr(args, "event_list_data_only", False)),
        "event_list_dir": args.event_list_dir,
        "weight_ttbar_bdt": float(args.weight_ttbar_bdt),
        "correct_vbf_bdt_shape": bool(args.correct_vbf_bdt_shape),
        "bdt_disc": bool(args.bdt_disc),
        "rerun_inference": bool(args.rerun_inference),
        "scale_smear": bool(args.scale_smear),
        "dummy_txbb_sfs": bool(args.dummy_txbb_sfs),
        "blind": bool(args.blind),
        "output_files": [f"eventlist_boostedHH4b_{y}.root" for y in args.years],
    }


_MERGE_IGNORE_KEYS = frozenset({"years", "output_files", "schema_version"})


def _deep_eq(x: Any, y: Any) -> bool:
    if isinstance(x, bool) or isinstance(y, bool):
        return x is y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return float(x) == float(y)
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return False
        return all(_deep_eq(a, b) for a, b in zip(x, y, strict=True))
    return x == y


def _manifest_settings_equal(existing: dict[str, Any], new: dict[str, Any]) -> bool:
    """True if analysis settings match (excluding per-run year lists)."""
    keys = (set(existing.keys()) | set(new.keys())) - _MERGE_IGNORE_KEYS
    return all(_deep_eq(existing.get(k), new.get(k)) for k in sorted(keys))


def write_eventlist_manifest(
    manifest_path: Path,
    args: Namespace,
    mass_window: np.ndarray | list[float],
    *,
    merge: bool = True,
) -> None:
    """Write manifest JSON.

    If ``merge`` is True and a manifest already exists with the same analysis
    settings (tagger, BDT, WPs, etc.) but a different ``years`` list, ``years``
    and ``output_files`` are unioned so one folder can hold ROOT files from
    sequential single-year PostProcess runs.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_eventlist_manifest(args, mass_window)
    if merge and manifest_path.is_file():
        try:
            old = read_manifest(manifest_path)
        except json.JSONDecodeError:
            old = None
        if (
            old is not None
            and old.get("schema_version") == SCHEMA_VERSION
            and _manifest_settings_equal(old, payload)
        ):
            merged_years: list[str] = []
            seen: set[str] = set()
            for y in list(old["years"]) + list(payload["years"]):
                if y not in seen:
                    seen.add(y)
                    merged_years.append(y)
            payload["years"] = merged_years
            payload["output_files"] = [f"eventlist_boostedHH4b_{y}.root" for y in merged_years]

    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_year_from_eventlist_filename(path: Path) -> str | None:
    m = _ROOT_NAME_RE.fullmatch(path.name)
    return m.group(1) if m else None


def manifest_errors_for_root(root_path: Path, manifest: dict[str, Any]) -> list[str]:
    """Cross-check manifest vs a single `eventlist_boostedHH4b_<year>.root` path."""
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"manifest schema_version is {manifest.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )

    required = (
        "years",
        "templates_tag",
        "ntuple_tag",
        "bdt_model",
        "bdt_config",
        "txbb",
        "mass_variable",
        "mass_window_fom",
    )
    for key in required:
        if key not in manifest:
            errors.append(f"manifest missing key {key!r}")

    year = parse_year_from_eventlist_filename(root_path)
    if year is None:
        errors.append(
            f"filename {root_path.name!r} does not match eventlist_boostedHH4b_<year>.root"
        )
        return errors

    years = manifest.get("years")
    if isinstance(years, list) and year not in years:
        errors.append(f"ROOT year {year!r} not in manifest years {years!r}")

    outs = manifest.get("output_files")
    expected_name = f"eventlist_boostedHH4b_{year}.root"
    if isinstance(outs, list) and expected_name not in outs:
        errors.append(f"manifest output_files does not list {expected_name!r}")

    return errors


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_manifest_path_for_root(root_path: Path) -> Path:
    return root_path.parent / "eventlist_manifest.json"


def fom_mass_window(mass_variable: str, txbb: str) -> np.ndarray:
    """FoM mass window used in PostProcess (must match ``postprocess_run3``)."""
    fom_window_by_mass: dict[str, list[float]] = {"H2Msd": [110.0, 140.0]}
    if txbb == "pnet-legacy":
        fom_window_by_mass["H2PNetMass"] = [105.0, 150.0]
    elif txbb == "glopart-v2":
        fom_window_by_mass["H2PNetMass"] = [110.0, 155.0]
    elif txbb == "pnet-v12":
        fom_window_by_mass["H2PNetMass"] = [120.0, 150.0]
    else:
        fom_window_by_mass["H2PNetMass"] = [110.0, 155.0]
    return np.array(fom_window_by_mass[mass_variable], dtype=float)


def namespace_from_postprocess_args_txt(path: Path) -> Namespace:
    """Load a Namespace from ``templates/<tag>/args.txt`` (pprint of ``vars(args)``)."""
    text = path.read_text(encoding="utf-8")
    try:
        data = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {path} as a Python literal dict: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return Namespace(**data)


def manifest_validation_errors(
    root_path: Path,
    manifest_path: Path | None,
    *,
    require_manifest: bool,
) -> list[str]:
    if manifest_path is not None:
        mp = manifest_path
    else:
        mp = default_manifest_path_for_root(root_path)
        if not require_manifest and not mp.is_file():
            return []

    if not mp.is_file():
        return [f"Manifest not found: {mp}"]

    try:
        manifest = read_manifest(mp)
    except json.JSONDecodeError as exc:
        return [f"Manifest is not valid JSON: {exc}"]

    return manifest_errors_for_root(root_path, manifest)


def _cli_write(ns: argparse.Namespace) -> int:
    args_ns = namespace_from_postprocess_args_txt(Path(ns.from_args_txt))
    mass_window = (
        np.array(ns.mass_window, dtype=float)
        if ns.mass_window is not None
        else fom_mass_window(args_ns.mass, args_ns.txbb)
    )
    out = (
        Path(ns.output)
        if ns.output is not None
        else Path(args_ns.event_list_dir) / "eventlist_manifest.json"
    )
    write_eventlist_manifest(out, args_ns, mass_window, merge=not ns.no_merge)
    print(f"Wrote {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Event list manifest utilities. ROOT files do not contain analysis settings; "
            "use `write --from-args-txt` to recreate eventlist_manifest.json from "
            "templates/<tag>/args.txt produced by PostProcess."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="Write eventlist_manifest.json from PostProcess args.txt")
    w.add_argument(
        "--from-args-txt",
        type=Path,
        required=True,
        help="Path to templates/<templates_tag>/args.txt (pprint of vars(args))",
    )
    w.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <event_list_dir>/eventlist_manifest.json under cwd)",
    )
    w.add_argument(
        "--mass-window",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Override FoM mass window (default: same rule as PostProcess for mass + txbb)",
    )
    w.add_argument(
        "--no-merge",
        action="store_true",
        help="Replace manifest entirely instead of merging years with an existing file",
    )
    w.set_defaults(func=_cli_write)

    parsed = parser.parse_args(argv)
    return parsed.func(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
