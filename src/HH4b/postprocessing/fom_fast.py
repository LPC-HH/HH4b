"""Vectorized FOM-scan replacement for PostProcess.scan_fom (optimizations #2 + #3).

Why this exists
---------------
`PostProcess.scan_fom` is a serial double `for xbb_cut: for bdt_cut:` loop
(PostProcess.py:1312) that re-masks the *entire* event set 2-3x per grid point.
With grids of 1.6k-6.4k points and millions of events, each bin scan is ~20-55 min
and uses ~1 core. That is the ~90% of a FOM-scan run's wall time.

Two facts make it fully vectorizable (no multiprocessing needed -> optimization #1
is unnecessary once #2 is in):

  * The signal-region selection is  (H2TXbb > xbb_cut) & (bdt_col > bdt_cut) & V,
    where the veto/anti-cut masks V are FIXED across the grid (they depend only on
    the pinned/resolved WPs of *higher-priority* bins, never on the scanned cut).
  * "sum of weights where value_x > tx[i] and value_b > tb[j]" for ALL (i, j) is a
    2D reverse-cumulative-sum of a single histogram -> O(N_events + N_grid) instead
    of O(N_grid * N_events).

The ABCD background estimate uses four region yields A,B,C,D; regions C,D use the
*anti-cut* (fixed) so they are the SAME scalar for every grid point (the serial code
recomputes them thousands of times). A,B use the scanned cut and become the two grid
histograms. This module reproduces `abcd()` (PostProcess.py:1615) and the argmin +
reliability filter (PostProcess.py:1391) exactly.

Optimization #3 (skip pinned scans) lives in `run_nested_fom_fast`: a bin whose WP is
already pinned (>= 0) is NOT scanned -- its FOM is read off a single-point evaluation.
For the "anchor" config (all WPs pinned) this collapses three grid scans into three
point evals (hours -> seconds); those scans currently run and discard their optima.

Faithfulness
------------
The fixed veto mask is taken from the ORIGINAL `get_cut` callable by neutralizing the
scanned thresholds (`get_cut(events, -1e9, -1e9)` leaves only ~veto & non-NaN), and the
anti-cut from the ORIGINAL `get_anti_cut`. So veto/region semantics are inherited, not
reimplemented -- only the threshold sweep is vectorized. `validate_against_serial`
asserts array-level agreement with `scan_fom` before you trust it.

NOTE: only the `method == "abcd"` path is vectorized (all 8-model runs use it). The
`sideband` (non-abcd) path raises; fall back to the serial `scan_fom` there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

_NEG = -1.0e9  # threshold that neutralizes a `value > cut` term (keeps only non-NaN)


# --------------------------------------------------------------------------------------
# core vectorized primitive
# --------------------------------------------------------------------------------------
def _grid_geq_sum(
    vx: np.ndarray, vb: np.ndarray, w: np.ndarray, tx: np.ndarray, tb: np.ndarray
) -> np.ndarray:
    """Return G[i, j] = sum(w over events with vx > tx[i] AND vb > tb[j]).

    `tx`, `tb` must be sorted ascending (np.arange grids are). Matches the STRICT `>`
    used by the region cut functions: an event with vx == tx[i] does not pass i.
    O(len(w) + len(tx)*len(tb)).
    """
    nx, nb = len(tx), len(tb)
    G = np.zeros((nx, nb))
    if vx.size == 0:
        return G
    # ix = #thresholds strictly < vx  -> event passes i for i in [0, ix); likewise jb.
    ix = np.searchsorted(tx, vx, side="left")  # in [0, nx]
    jb = np.searchsorted(tb, vb, side="left")  # in [0, nb]
    # deposit each event's weight at its (ix, jb) corner in an (nx+1, nb+1) grid
    M = np.zeros((nx + 1, nb + 1))
    np.add.at(M, (ix, jb), w)
    # suffix sum SS[a,b] = sum_{a'>=a, b'>=b} M[a',b'];  G[i,j] = sum_{a>i,b>j} M = SS[i+1,j+1]
    SS = np.cumsum(np.cumsum(M[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    return SS[1:, 1:]


# --------------------------------------------------------------------------------------
# vectorized FOMs (elementwise; reproduce PostProcess.fom_classic / fom_asimov)
# --------------------------------------------------------------------------------------
def _fom_classic_vec(s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2*sqrt(b)/s where s>0 and b>0, else nan (elementwise)."""
    good = (s > 0) & (b > 0)
    out = np.full(s.shape, np.nan)
    bb = np.where(good, b, 1.0)
    ss = np.where(good, s, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(good, 2.0 * np.sqrt(bb) / ss, np.nan)
    return out


def _fom_asimov_vec(s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1/Z_A with Z_A = sqrt(2[(s+b)ln(1+s/b) - s]); nan where s<=0 or b<=0 or Z_A<=0."""
    good = (s > 0) & (b > 0)
    ss = np.where(good, s, 1.0)
    bb = np.where(good, b, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        za = np.sqrt(2.0 * ((ss + bb) * np.log1p(ss / bb) - ss))
        out = np.where(good & (za > 0), 1.0 / za, np.nan)
    return out


_FOM_VEC = {"classic": _fom_classic_vec, "asimov": _fom_asimov_vec}


# --------------------------------------------------------------------------------------
# mass-window masks (reproduce get_nevents_signal / get_nevents_nosignal)
# --------------------------------------------------------------------------------------
def _window_masks(mass_vals: np.ndarray, mass_window) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(mass_window[0]), float(mass_window[1])
    in_win = (mass_vals >= lo) & (mass_vals <= hi)
    sideband = ((mass_vals >= 60) & (mass_vals <= lo)) | ((mass_vals >= hi) & (mass_vals <= 220))
    return in_win, sideband


def _fixed_allowed(get_cut: Callable, events) -> np.ndarray:
    """The scan-independent part of get_cut: ~veto & (relevant columns non-NaN).

    Obtained by neutralizing the scanned thresholds -- get_cut(events, -1e9, -1e9) leaves
    (H2TXbb > -1e9) & (bdt_col > -1e9) & ~veto == ~veto & non-NaN in the two scan columns.
    """
    return np.asarray(get_cut(events, _NEG, _NEG))


# --------------------------------------------------------------------------------------
# vectorized scan (drop-in for PostProcess.scan_fom, abcd path)
# --------------------------------------------------------------------------------------
def scan_fom_fast(
    method: str,
    events_combined: dict,
    get_cut: Callable,
    get_anti_cut: Callable,
    xbb_cuts: np.ndarray,
    bdt_cuts: np.ndarray,
    mass_window,
    bg_keys: list[str],
    sig_keys: list[str],
    fom: str,
    bdt_col: str,
    mass: str = "H2PNetMass",
) -> dict:
    """Vectorized equivalent of PostProcess.scan_fom (method == 'abcd').

    Returns a dict with the same flat arrays scan_fom saves to *_fom_arrays.npz
    (all_fom, all_b, all_b_unc, all_s, all_sideband_events, all_xbb_cuts, all_bdt_cuts,
    row-major with xbb OUTER / bdt INNER -- identical ordering to the serial loops), plus
    'best_xbb'/'best_bdt' (argmin under the reliability filter) and the 2D grids.

    `fom` is "classic" or "asimov"; `bdt_col` is the column the scanned BDT threshold
    applies to ("bdt_score_vbf" for the VBF region, else "bdt_score"). `reliability`
    toggles the sideband>=12 & b>0.5 filter (pass args.fom_reliability_filter).
    """
    if method != "abcd":
        raise NotImplementedError("fom_fast vectorizes the abcd path only; use serial scan_fom.")

    tx = np.asarray(xbb_cuts, dtype=float)
    tb = np.asarray(bdt_cuts, dtype=float)
    fom_vec = _FOM_VEC[fom]

    bg = [k for k in bg_keys if k != "qcd"]

    def region_grids(key):
        """(A_grid, B_grid, C_scalar, D_scalar) for one sample, mirroring abcd()."""
        ev = events_combined[key]
        allowed = _fixed_allowed(get_cut, ev)  # ~veto & non-NaN (fixed across grid)
        anti = np.asarray(get_anti_cut(ev))  # anti-cut (fixed; NO veto, per abcd())
        vx = np.asarray(ev["H2TXbb"])
        vb = np.asarray(ev[bdt_col])
        w = np.asarray(ev["weight"])
        in_win, sideband = _window_masks(np.asarray(ev[mass]), mass_window)
        # A = signal window under the scanned cut; B = mass sideband under the scanned cut
        mA = allowed & in_win
        mB = allowed & sideband
        A = _grid_geq_sum(vx[mA], vb[mA], w[mA], tx, tb)
        B = _grid_geq_sum(vx[mB], vb[mB], w[mB], tx, tb)
        # C, D use the anti-cut -> scalars (constant over the grid)
        C = float(np.sum(w[anti & in_win]))
        D = float(np.sum(w[anti & sideband]))
        return A, B, C, D

    # signal: region-A yield (signal window) summed over sig_keys, on the scanned cut
    s = np.zeros((len(tx), len(tb)))
    for k in sig_keys:
        s += region_grids(k)[0]

    # data regions: A is defined 0 in abcd(); B is the data mass-sideband grid
    _, B_d, C_d, D_d = region_grids("data")

    # background MC (qcd removed): accumulate A,B grids and C,D scalars
    A_bg = np.zeros_like(s)
    B_bg = np.zeros_like(s)
    C_bg = 0.0
    D_bg = 0.0
    for k in bg:
        A_k, B_k, C_k, D_k = region_grids(k)
        A_bg += A_k
        B_bg += B_k
        C_bg += C_k
        D_bg += D_k

    # data - other-bkg in each region, then A_qcd = B*C/D  (PostProcess.abcd)
    dmt_B = B_d - B_bg
    dmt_C = C_d - C_bg
    dmt_D = D_d - D_bg
    with np.errstate(invalid="ignore", divide="ignore"):
        bqcd = dmt_B * (dmt_C / dmt_D)
    background = bqcd + A_bg if bg else bqcd

    all_sideband = B_d  # == get_nevents_nosignal(data, cut): data sideband, signal cut
    fom_grid = fom_vec(s, background)

    # flatten row-major: xbb OUTER, bdt INNER (matches the serial append order)
    xg, bg_grid = np.meshgrid(tx, tb, indexing="ij")
    out = {
        "all_fom": fom_grid.ravel(),
        "all_b": background.ravel(),
        "all_b_unc": np.sqrt(np.where(background > 0, background, np.nan)).ravel(),
        "all_s": s.ravel(),
        "all_sideband_events": all_sideband.ravel(),
        "all_xbb_cuts": xg.ravel(),
        "all_bdt_cuts": bg_grid.ravel(),
        "fom_grid": fom_grid,
        "s_grid": s,
        "b_grid": background,
        "sideband_grid": all_sideband,
        "xbb_cuts": tx,
        "bdt_cuts": tb,
    }
    return out


def best_wp(scan: dict, reliability: bool = True) -> tuple[float, float]:
    """argmin(FOM) WP with the reliability filter (PostProcess.py:1391-1397)."""
    fom = scan["all_fom"]
    b = scan["all_b"]
    sb = scan["all_sideband_events"]
    finite = np.isfinite(fom) & (fom > 0)
    if reliability:
        valid = finite & (sb >= 12) & (b > 0.5)
        if valid.sum() == 0:
            valid = finite & (b > 0)
    else:
        valid = finite
    if valid.sum() == 0:
        return -1.0, -1.0
    idx = np.where(valid, fom, np.inf).argmin()
    return float(scan["all_xbb_cuts"][idx]), float(scan["all_bdt_cuts"][idx])


# --------------------------------------------------------------------------------------
# single-point evaluation (optimization #3: a pinned bin needs no scan)
# --------------------------------------------------------------------------------------
def evaluate_point(
    events_combined: dict,
    get_cut: Callable,
    get_anti_cut: Callable,
    xbb_wp: float,
    bdt_wp: float,
    mass_window,
    bg_keys: list[str],
    sig_keys: list[str],
    fom: str,
    bdt_col: str,
    mass: str = "H2PNetMass",
) -> dict:
    """s, b, sideband, FOM at ONE (xbb_wp, bdt_wp) -- a length-1 'grid' via scan_fom_fast."""
    scan = scan_fom_fast(
        "abcd",
        events_combined,
        get_cut,
        get_anti_cut,
        np.array([xbb_wp]),
        np.array([bdt_wp]),
        mass_window,
        bg_keys,
        sig_keys,
        fom,
        bdt_col,
        mass,
    )
    return {
        "xbb_wp": xbb_wp,
        "bdt_wp": bdt_wp,
        "s": float(scan["all_s"][0]),
        "b": float(scan["all_b"][0]),
        "sideband": float(scan["all_sideband_events"][0]),
        "fom": float(scan["all_fom"][0]),
    }


# --------------------------------------------------------------------------------------
# validation harness -- run BEFORE trusting the fast path
# --------------------------------------------------------------------------------------
def validate_against_serial(
    scan_fom_serial: Callable,
    method: str,
    events_combined: dict,
    get_cut: Callable,
    get_anti_cut: Callable,
    xbb_cuts: np.ndarray,
    bdt_cuts: np.ndarray,
    mass_window,
    bg_keys: list[str],
    sig_keys: list[str],
    fom_serial: Callable,
    fom_name: str,
    bdt_col: str,
    mass: str = "H2PNetMass",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    plot_dir: str = "/tmp",
) -> bool:
    """Run serial scan_fom and scan_fom_fast on the SAME inputs; assert arrays match.

    `scan_fom_serial`/`fom_serial` are PostProcess.scan_fom and the fom callable, injected
    to avoid an import cycle. scan_fom writes a *_fom_arrays.npz; we reload it and compare
    to the fast dict. Returns True on agreement, else raises AssertionError with the worst
    mismatch. Call this once per (region, model) family before switching over.
    """
    # serial run (writes npz into plot_dir)
    scan_fom_serial(
        method,
        events_combined,
        get_cut,
        get_anti_cut,
        xbb_cuts,
        bdt_cuts,
        mass_window,
        plot_dir,
        "validate_serial",
        bg_keys=bg_keys,
        sig_keys=sig_keys,
        fom=fom_serial,
        mass=mass,
    )
    npz = sorted(Path(plot_dir).glob("validate_serial_*_fom_arrays.npz"))[-1]
    serial = np.load(npz)

    fast = scan_fom_fast(
        method,
        events_combined,
        get_cut,
        get_anti_cut,
        xbb_cuts,
        bdt_cuts,
        mass_window,
        bg_keys,
        sig_keys,
        fom_name,
        bdt_col,
        mass,
    )

    for key in ["all_s", "all_b", "all_sideband_events", "all_xbb_cuts", "all_bdt_cuts"]:
        a, b = np.asarray(serial[key], float), np.asarray(fast[key], float)
        assert a.shape == b.shape, f"{key}: shape {a.shape} vs {b.shape}"
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        worst = np.nanmax(diff) if diff.size else 0.0
        assert np.allclose(
            a, b, rtol=rtol, atol=atol, equal_nan=True
        ), f"{key}: max|Δ|={worst:.3e} at flat idx {int(np.nanargmax(diff))}"

    # FOM: compare only where both finite (nan bookkeeping can differ harmlessly)
    fa, fb = np.asarray(serial["all_fom"], float), np.asarray(fast["all_fom"], float)
    m = np.isfinite(fa) & np.isfinite(fb)
    assert np.allclose(fa[m], fb[m], rtol=1e-5, atol=1e-8), "all_fom mismatch on finite entries"

    xb_s, bd_s = best_wp({k: serial[k] for k in serial.files})
    xb_f, bd_f = best_wp(fast)
    assert (xb_s, bd_s) == (
        xb_f,
        bd_f,
    ), f"argmin WP differs: serial {xb_s,bd_s} vs fast {xb_f,bd_f}"

    print(f"[fom_fast] VALIDATED: {len(fast['all_fom'])} pts, argmin WP ({xb_f:.4f}, {bd_f:.4f})")
    return True


# --------------------------------------------------------------------------------------
# nested driver with pin-skip (optimizations #2 + #3 together)
# --------------------------------------------------------------------------------------
def _pinned(a, b) -> bool:
    return a is not None and b is not None and a >= 0 and b >= 0


def _resolve(cur, opt):
    """Keep a pinned WP (>=0), else take the scan optimum -- PostProcess.py:1960."""
    return opt if (cur is None or cur < 0) else cur


def _save_npz(plot_dir, plot_name, method, mass_window, scan):
    name = f"{plot_name}_{method}_mass{mass_window[0]}-{mass_window[1]}"
    np.savez(
        Path(plot_dir) / f"{name}_fom_arrays.npz",
        all_fom=scan["all_fom"],
        all_b=scan["all_b"],
        all_b_unc=scan["all_b_unc"],
        all_s=scan["all_s"],
        all_sideband_events=scan["all_sideband_events"],
        all_xbb_cuts=scan["all_xbb_cuts"],
        all_bdt_cuts=scan["all_bdt_cuts"],
    )


def run_nested_fom_fast(
    args,
    events_combined: dict,
    get_cuts: Callable,
    get_anti_cuts: Callable,
    mass_window,
    bg_keys: list[str],
    plot_dir: str | None = None,
) -> dict:
    """Nested Bin1 -> VBF -> Bin2 optimization (mirrors PostProcess.py:1963-2043) with the
    vectorized scan, SKIPPING any bin whose WP is already pinned.

    A pinned bin (WP >= 0) is evaluated at its single fixed point instead of scanned -- for
    the anchor config (all WPs pinned) this turns three grid scans into three point evals.
    Mutates args.txbb_wps / args.bdt_wps / args.vbf_txbb_wp / args.vbf_bdt_wp in place with
    the resolved WPs, exactly like the serial driver, and returns a per-bin summary.
    Scanned bins also write the same *_fom_arrays.npz as scan_fom (so downstream Asimov
    post-hoc keeps working) when plot_dir is given.
    """
    reliability = getattr(args, "fom_reliability_filter", True)
    bin1_fom = "asimov" if getattr(args, "fom_bin1_asimov", False) else "classic"
    x = np.arange
    summary: dict = {}

    def _do(region, bdt_col, fom_name, sig_keys, xbb_grid, bdt_grid, cur_x, cur_b, plot_name):
        get_cut = get_cuts(args, region)  # built AFTER prior WPs are resolved (veto is current)
        anti = get_anti_cuts(args, region)
        pin = _pinned(cur_x, cur_b)
        scan_flag = {
            "bin1": args.fom_scan_bin1,
            "vbf": args.fom_scan_vbf,
            "bin2": args.fom_scan_bin2,
        }[region]
        # Not scanning this bin when it is pinned (#3: evaluate the fixed point instead
        # of scanning+discarding) or when its scan flag is off (the serial path skips it
        # entirely). Only evaluate a real (resolved, >=0) WP; a disabled bin still at -1 is
        # left untouched, exactly like the serial path -- never evaluate at the -1 sentinel.
        if pin or not scan_flag:
            if pin:
                pt = evaluate_point(
                    events_combined,
                    get_cut,
                    anti,
                    cur_x,
                    cur_b,
                    mass_window,
                    bg_keys,
                    sig_keys,
                    fom_name,
                    bdt_col,
                    args.mass,
                )
                summary[region] = {**pt, "scanned": False}
            return cur_x, cur_b
        scan = scan_fom_fast(
            args.method,
            events_combined,
            get_cut,
            anti,
            xbb_grid,
            bdt_grid,
            mass_window,
            bg_keys,
            sig_keys,
            fom_name,
            bdt_col,
            args.mass,
        )
        bx, bb = best_wp(scan, reliability=reliability)
        if plot_dir is not None:
            _save_npz(plot_dir, plot_name, args.method, mass_window, scan)
        # flat index of the chosen (bx, bb) grid point (values come straight from the
        # grid arrays, so exact == is safe)
        k = int(((scan["all_xbb_cuts"] == bx) & (scan["all_bdt_cuts"] == bb)).argmax())
        summary[region] = {
            "xbb_wp": bx,
            "bdt_wp": bb,
            "s": float(scan["all_s"][k]),
            "b": float(scan["all_b"][k]),
            "fom": float(scan["all_fom"][k]),
            "scanned": True,
        }
        return bx, bb

    # 1) ggF Bin 1 -- top priority, no veto
    bx, bb = _do(
        "bin1",
        "bdt_score",
        bin1_fom,
        args.fom_ggf_samples,
        x(0.9, 0.999, 0.0025),
        x(0.9, 0.999, 0.0025),
        args.txbb_wps[0],
        args.bdt_wps[0],
        "fom_bin1",
    )
    args.txbb_wps[0] = _resolve(args.txbb_wps[0], bx)
    args.bdt_wps[0] = _resolve(args.bdt_wps[0], bb)

    # 2) VBF -- vetoes Bin 1 at its resolved WP
    if args.vbf:
        vx, vb = _do(
            "vbf",
            "bdt_score_vbf",
            "classic",
            args.fom_vbf_samples,
            x(0.8, 0.999, 0.0025),
            x(0.9, 0.999, 0.0025),
            args.vbf_txbb_wp,
            args.vbf_bdt_wp,
            "fom_vbf",
        )
        args.vbf_txbb_wp = _resolve(args.vbf_txbb_wp, vx)
        args.vbf_bdt_wp = _resolve(args.vbf_bdt_wp, vb)

    # 3) ggF Bin 2 -- vetoes Bin 1 (+ VBF) at their resolved WPs
    b2x, b2b = _do(
        "bin2",
        "bdt_score",
        "classic",
        args.fom_ggf_samples,
        x(0.5, 0.9, 0.005),
        x(0.5, 0.9, 0.005),
        args.txbb_wps[1],
        args.bdt_wps[1],
        "fom_bin2",
    )
    args.txbb_wps[1] = _resolve(args.txbb_wps[1], b2x)
    args.bdt_wps[1] = _resolve(args.bdt_wps[1], b2b)

    return summary
