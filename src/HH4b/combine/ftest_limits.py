"""Transfer-factor F-test decisions and expected limits, read back from the combine outputs.

``ftest CARDS_DIR``
    The strict F-test of AN-23-151 Sec. 4.5.3.1 from the outputs of ``run_ftest_hh4b.sh`` in
    ``CARDS_DIR`` (``cards/f_tests/<cardstag>``). Each ``<region>_nTF_<k>`` directory holds the
    data GoF (``higgsCombineData.GoodnessOfFit.mH125.root``), the GoF fits of the toys
    (``higgsCombineToys<name>.GoodnessOfFit.mH125.<seed>.root``) and their ``-v 9`` logs
    (``outs/GoF_toys<name>_s<seed>.txt[.gz]``). The toys of the step k -> k+1 are generated from
    the order-k B-only fit to data and fitted at orders k and k+1. With t the saturated test
    statistic and N the number of unblinded pass bins:

    - GoF p(k) = P(t_toy(k) >= t_data(k)) on the toys converged at order k, and, for
      information, GoF p(k+1) = P(t_toy(k+1) >= t_data(k+1)) on the same toys converged at k+1;
    - F = (t(k) - t(k+1)) / t(k+1) * (N - (k + 2)) and F-test p = P(F_toy >= F_data) on the
      toys converged at both orders, paired by (iSeed, iToy).

    A toy is converged when every minimisation of its block in the log ended with status 0. The
    errors are binomial, sqrt(p (1 - p) / n). Starting at order 0, order k is kept if and only if
    the F-test p and GoF p(k) are both above ``--alpha``; otherwise the chain goes to k + 1.

``limit CARD [CARD ...]``
    The expected limits of ``run_blinded_hh4b.sh --limits`` (``outs/AsymptoticLimits.txt[.gz]``):
    combine's quantiles, which are bisection values (steps of about 1%), and the median from the
    linear interpolation of the Asimov scan ("At r = x: delta(nll) = y") to
    Phi^-1(0.975)^2 / 2 = 1.920729, extrapolated from the last two points when the scan stops
    below it.

Run from src/HH4b in the analysis environment (``ftest`` needs uproot)::

    python3 combine/ftest_limits.py ftest cards/f_tests/<cardstag>
    python3 combine/ftest_limits.py limit postprocessing/cards/<card> [...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

STATUS_OR_END = re.compile(r"Minimization finished with status=(-?\d+)|Best fit test statistic")
SCAN_POINT = re.compile(r"At r = ([\d.eE+-]+):\s*delta\(nll\) = ([\d.eE+-]+)")
EXPECTED = re.compile(r"Expected\s+([\d.]+)%: r < ([\d.eE+-]+)")
OBSERVED = re.compile(r"Observed Limit: r < ([\d.eE+-]+)")
NLL_TARGET = NormalDist().inv_cdf(0.975) ** 2 / 2
REGIONS = ["passvbf", "passbin1", "passbin2", "passbin3"]  # order of CreateDatacard.py --nTF
ORDER_DIR = re.compile(r"^(pass\w+)_nTF_(\d+)$")


def warn(msg: str):
    print(f"WARNING: {msg}", file=sys.stderr)


def find_text(path: Path) -> Path | None:
    """``path`` if it exists, else ``path.gz`` if that exists, else None."""
    for p in (path, path.with_name(path.name + ".gz")):
        if p.is_file():
            return p
    return None


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as f:
            return f.read()
    return path.read_text(errors="replace")


# ------------------------------------------------------------------------------------- F-test


def read_gof(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(iSeed, iToy, t) of a combine GoodnessOfFit output, None if it is missing or unreadable."""
    import uproot  # noqa: PLC0415 (only the F-test needs it)

    if not path.is_file():
        return None
    try:
        with uproot.open(path) as f:
            a = f["limit"].arrays(["limit", "iToy", "iSeed"], library="np")
    except Exception as e:  # e.g. still being written
        warn(f"cannot read {path}: {e}")
        return None
    return a["iSeed"].astype(int), a["iToy"].astype(int), a["limit"].astype(float)


def converged(log: Path, ntoys: int) -> np.ndarray:
    """Per toy, whether every minimisation of its log block ended with status 0."""
    found = find_text(log)
    if found is None:
        warn(f"{log}[.gz] is missing, counting none of its {ntoys} toys as converged")
        return np.zeros(ntoys, bool)
    ok, statuses = [], []
    for m in STATUS_OR_END.finditer(read_text(found)):
        if m.group(1) is None:  # "Best fit test statistic" ends the block of a toy
            ok.append(bool(statuses) and all(s == 0 for s in statuses))
            statuses = []
        else:
            statuses.append(int(m.group(1)))
    if len(ok) != ntoys:
        warn(f"{found}: {len(ok)} toy blocks for {ntoys} toys, counting none as converged")
        return np.zeros(ntoys, bool)
    return np.array(ok, bool)


def find_seeds(order_dir: Path, name: str) -> list[int]:
    if not order_dir.is_dir():
        return []
    pat = re.compile(rf"higgsCombineToys{re.escape(name)}\.GoodnessOfFit\.mH125\.(\d+)\.root$")
    return sorted(
        int(m.group(1)) for p in order_dir.iterdir() if (m := pat.match(p.name)) is not None
    )


def read_toys(order_dir: Path, name: str, seeds: list[int]) -> dict[tuple[int, int], tuple]:
    """{(iSeed, iToy): (t, converged)} of the toys ``name`` fitted at the order of ``order_dir``."""
    out = {}
    for seed in seeds:
        gof = read_gof(order_dir / f"higgsCombineToys{name}.GoodnessOfFit.mH125.{seed}.root")
        if gof is None:
            continue
        ok = converged(order_dir / "outs" / f"GoF_toys{name}_s{seed}.txt", len(gof[2]))
        for s, i, t, o in zip(*gof, ok):
            out[(int(s), int(i))] = (float(t), bool(o))
    return out


def binomial(passed: np.ndarray) -> dict:
    n = len(passed)
    p = float(passed.mean()) if n else float("nan")
    return {"p": p, "err": float(np.sqrt(p * (1 - p) / n)) if n else float("nan"), "n": n}


def evaluate_step(cards: Path, region: str, k: int, args) -> dict | None:
    """GoF and F-test p-values of the step k -> k+1; None if its toys or data fits are missing."""
    lo_dir, hi_dir = cards / f"{region}_nTF_{k}", cards / f"{region}_nTF_{k + 1}"
    name = args.toys_name.format(k=k)
    seeds = args.seeds or find_seeds(lo_dir, name)
    lo, hi = read_toys(lo_dir, name, seeds), read_toys(hi_dir, name, seeds)
    data = [read_gof(d / "higgsCombineData.GoodnessOfFit.mH125.root") for d in (lo_dir, hi_dir)]
    if not lo or not hi or None in data:
        return None
    t_lo, t_hi = (float(d[2][0]) for d in data)
    dof = args.nbins - (k + 2)
    f_data = (t_lo - t_hi) / t_hi * dof

    t_k, ok_k = (np.array(x) for x in zip(*lo.values()))
    t_k1, ok_k1 = (np.array(x) for x in zip(*hi.values()))
    pairs = sorted(set(lo) & set(hi))
    a = np.array([lo[x][0] for x in pairs], float)
    b = np.array([hi[x][0] for x in pairs], float)
    both = np.array([lo[x][1] and hi[x][1] for x in pairs], bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_toy = (a - b) / b * dof
    f_sel = both & np.isfinite(f_toy)

    gof_k = binomial(t_k[ok_k & np.isfinite(t_k)] >= t_lo)
    gof_k1 = binomial(t_k1[ok_k1 & np.isfinite(t_k1)] >= t_hi)
    ftest = binomial(f_toy[f_sel] >= f_data)
    return {
        "region": region,
        "k": k,
        "seeds": seeds,
        "t_data": [t_lo, t_hi],
        "f_data": f_data,
        "gof_k": gof_k,
        "gof_k1": gof_k1,
        "ftest": ftest,
        "keep": bool(gof_k["p"] > args.alpha and ftest["p"] > args.alpha),
    }


def find_regions(cards: Path) -> list[str]:
    found = {m.group(1) for p in cards.iterdir() if p.is_dir() and (m := ORDER_DIR.match(p.name))}
    return sorted(found, key=lambda r: (r not in REGIONS, REGIONS.index(r) if r in REGIONS else r))


def ftest(args) -> int:
    cards = args.cards_dir
    regions = args.regions or find_regions(cards)
    if not regions:
        raise SystemExit(f"no <region>_nTF_<k> directories in {cards}")
    start = dict(args.start)
    print(f"F-test in {cards}: N = {args.nbins}, alpha = {args.alpha}, status-0 toys")
    print(
        f"{'region':9s} {'step':5s} {'t_data(k)':>10s} {'t_data(k+1)':>11s} {'F_data':>7s} "
        f"{'toys k/k+1/both':>16s}  {'GoF p(k)':14s}  {'GoF p(k+1)':14s}  {'F-test p':14s}  "
        "decision"
    )
    rows, selected = [], {}
    for region in regions:
        sel, broken, k0, k = None, False, start.get(region, 0), 0
        while (cards / f"{region}_nTF_{k + 1}").is_dir():
            step = evaluate_step(cards, region, k, args)
            if step is None:
                print(f"{region:9s} {k}->{k + 1}  no toys or data fits")
                broken |= k >= k0 and sel is None  # the chain cannot go past a missing step
                k += 1
                continue
            rule = f"keep order {k}" if step["keep"] else f"go to order {k + 1}"
            if k < k0 or broken:
                step["decision"] = f"outside the chain (rule: {rule})"
            elif sel is not None:
                step["decision"] = f"not reached (rule: {rule})"
            else:
                step["decision"] = rule + (", selected" if step["keep"] else "")
                sel = k if step["keep"] else None
            rows.append(step)
            g0, g1, f = step["gof_k"], step["gof_k1"], step["ftest"]
            counts = f"{g0['n']}/{g1['n']}/{f['n']}"
            print(
                f"{region:9s} {k}->{k + 1}  {step['t_data'][0]:10.3f} {step['t_data'][1]:11.3f} "
                f"{step['f_data']:7.3f} {counts:>16s}  {g0['p']:.3f} +- {g0['err']:.3f}  "
                f"{g1['p']:.3f} +- {g1['err']:.3f}  {f['p']:.3f} +- {f['err']:.3f}  "
                f"{step['decision']}"
            )
            k += 1
        selected[region] = sel
    summary = ", ".join(f"{r} {'undecided' if s is None else s}" for r, s in selected.items())
    print("selected orders:", summary)
    if all(selected.get(r) is not None for r in REGIONS):
        print("CreateDatacard.py --nTF", *(selected[r] for r in REGIONS))
    if args.json:
        out = {"cards_dir": str(cards), "nbins": args.nbins, "alpha": args.alpha}
        args.json.write_text(json.dumps({**out, "steps": rows, "selected": selected}, indent=1))
    return 0


# ------------------------------------------------------------------------------------- limits


def interpolate(points: list[tuple[float, float]]) -> tuple[float | None, bool]:
    """r where the Asimov delta(nll) crosses NLL_TARGET, and whether it was extrapolated."""
    below = [q for q in points if q[1] <= NLL_TARGET]
    above = [q for q in points if q[1] >= NLL_TARGET]
    if below and above:
        (r0, n0), (r1, n1), extrapolated = below[-1], above[0], False
    elif len(below) >= 2:
        (r0, n0), (r1, n1), extrapolated = *below[-2:], True
    else:
        return None, False
    if n1 == n0:
        return r0, extrapolated
    return r0 + (r1 - r0) * (NLL_TARGET - n0) / (n1 - n0), extrapolated


def limits_log(path: Path) -> Path:
    if path.is_file():
        return path
    for outs in (path / "outs", path / path.name / "outs"):
        log = find_text(outs / "AsymptoticLimits.txt")
        if log is not None:
            return log
    raise FileNotFoundError(f"no outs/AsymptoticLimits.txt[.gz] in {path}")


def read_limits(path: Path) -> dict:
    log = limits_log(path)
    txt = read_text(log)
    expected = {float(q): float(r) for q, r in EXPECTED.findall(txt)}
    observed = OBSERVED.search(txt)
    scan, found, _ = txt.partition("Median for expected limits")
    points = sorted({(float(r), float(n)) for r, n in SCAN_POINT.findall(scan)}) if found else []
    median, extrapolated = interpolate(points)
    return {
        "log": str(log),
        "expected": expected,
        "observed": float(observed.group(1)) if observed else None,
        "median_interp": median,
        "extrapolated": extrapolated,
    }


def limit(args) -> int:
    quantiles = [2.5, 16.0, 50.0, 84.0, 97.5]
    width = max(len(str(c)) for c in args.cards)
    print(f"{'card':{width}s}", *(f"{q:>9g}%" for q in quantiles), "  50% interp.")
    results, status = {}, 0
    for card in args.cards:
        try:
            res = read_limits(card)
        except FileNotFoundError as e:
            warn(str(e))
            status = 1
            continue
        results[str(card)] = res
        exp = ["-" if v is None else f"{v:.4f}" for v in map(res["expected"].get, quantiles)]
        interp = res["median_interp"]
        line = [f"{card!s:{width}s}", *(f"{v:>10s}" for v in exp)]
        line.append("  -" if interp is None else f"  {interp:.6g}")
        if res["extrapolated"]:
            line.append("(extrapolated)")
        if res["observed"] is not None:
            line.append(f"observed {res['observed']:.4f}")
        print(*line)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return status


def region_order(s: str) -> tuple[str, int]:
    region, _, order = s.partition("=")
    if not region or not order.isdigit():
        raise argparse.ArgumentTypeError(f"expected REGION=K, e.g. passbin3=2, not {s!r}")
    return region, int(order)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ftest", help="strict F-test decisions from run_ftest_hh4b.sh outputs")
    p.add_argument("cards_dir", type=Path, help="cards/f_tests/<cardstag> of run_ftest_hh4b.sh")
    p.add_argument("--regions", nargs="+", help="pass regions (default: all in CARDS_DIR)")
    p.add_argument(
        "--toys-name",
        default="{k}",
        help="combine -n name of the toys of the step k -> k+1 without 'Toys', with {k} for the "
        "lower order (default: '{k}', as run_ftest_hh4b.sh writes)",
    )
    p.add_argument("--seeds", nargs="+", type=int, help="toy seeds (default: all found)")
    p.add_argument(
        "--nbins",
        type=int,
        default=13,
        help="unblinded pass bins N (default: 13, i.e. 16 m(H2) bins with bins 5-7 blinded)",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="threshold of both p-values")
    p.add_argument(
        "--start",
        nargs="+",
        default=[],
        type=region_order,
        metavar="REGION=K",
        help="start the chain of REGION at order K (when the lower steps were decided before)",
    )
    p.add_argument("--json", type=Path, help="also write the steps and decisions to this file")
    p.set_defaults(func=ftest)

    lp = sub.add_parser("limit", help="expected limits and interpolated medians")
    lp.add_argument(
        "cards",
        nargs="+",
        type=Path,
        help="card directories (with outs/ or <card>/outs/) or AsymptoticLimits.txt[.gz] files",
    )
    lp.add_argument("--json", type=Path, help="also write the limits to this file")
    lp.set_defaults(func=limit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
