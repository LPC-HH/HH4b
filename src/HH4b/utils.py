"""
General utilities for postprocessing.

Author: Raghav Kansal
"""

# ruff: noqa: PTH208

from __future__ import annotations

import contextlib
import logging
import logging.config
import pickle
import re
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from os import listdir
from pathlib import Path

import hist
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import vector
from coffea.analysis_tools import PackedSelection
from coffea.processor.accumulator import accumulate
from hist import Hist

from HH4b.xsecs import xsecs

from .hh_vars import (
    LUMI,
    data_key,
    jec_shifts,
    jec_vars,
    jmsr_keys,
    jmsr_shifts,
    jmsr_vars,
    norm_preserving_weights,
    syst_keys,
    years,
)

logger = logging.getLogger("HH4b.utils")
logger.setLevel(logging.DEBUG)

MAIN_DIR = "./"
CUT_MAX_VAL = 9999.0
PAD_VAL = -99999


@dataclass
class ShapeVar:
    """Class to store attributes of a variable to make a histogram of.

    Args:
        var (str): variable name
        label (str): variable label
        bins (List[int]): bins
        reg (bool, optional): Use a regular axis or variable binning. Defaults to True.
        blind_window (List[int], optional): if blinding, set min and max values to set 0. Defaults to None.
        significance_dir (str, optional): if plotting significance, which direction to plot it in.
          See more in plotting.py:ratioHistPlot(). Options are ["left", "right", "bin"]. Defaults to "right".
        plot_args (dict, optional): dictionary of arguments for plotting. Defaults to None.
    """

    var: str = None
    label: str = None
    bins: list[int] = None
    reg: bool = True
    blind_window: list[float] = None
    significance_dir: str = "right"
    plot_args: dict = None

    def __post_init__(self):
        # create axis used for histogramming
        if self.bins is not None:
            if self.reg:
                self.axis = hist.axis.Regular(*self.bins, name=self.var, label=self.label)
            else:
                self.axis = hist.axis.Variable(self.bins, name=self.var, label=self.label)
        else:
            self.axis = None


@dataclass
class Syst:
    samples: list[str] = None
    years: list[str] = field(default_factory=lambda: years)
    label: str = None


@contextlib.contextmanager
def timer():
    old_time = time.monotonic()
    try:
        yield
    finally:
        new_time = time.monotonic()
        print(f"Time taken: {new_time - old_time} seconds")


def remove_empty_parquets(samples_dir, year):

    full_samples_list = listdir(f"{samples_dir}/{year}")
    print("Checking for empty parquets")

    for sample in full_samples_list:
        if sample == ".DS_Store":
            continue
        parquet_files = listdir(f"{samples_dir}/{year}/{sample}/parquet")
        for f in parquet_files:
            file_path = f"{samples_dir}/{year}/{sample}/parquet/{f}"
            if not len(pd.read_parquet(file_path)):
                print("Removing: ", f"{sample}/{f}")
                Path(file_path).unlink()


def get_cutflow(pickles_path, year, sample_name):
    """Accumulates cutflow over all pickles in ``pickles_path`` directory"""
    out_pickles = listdir(pickles_path)

    file_name = out_pickles[0]
    with Path(f"{pickles_path}/{file_name}").open("rb") as file:
        try:
            out_dict = pickle.load(file)
        except RuntimeError:
            print(f"Problem opening {pickles_path}/{file_name}")
        cutflow = out_dict[year][sample_name]["cutflow"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with Path(f"{pickles_path}/{file_name}").open("rb") as file:
            out_dict = pickle.load(file)
            cutflow = accumulate([cutflow, out_dict[year][sample_name]["cutflow"]])

    return cutflow


def get_nevents(pickles_path, year, sample_name):
    """Adds up nevents over all pickles in ``pickles_path`` directory"""
    try:
        out_pickles = listdir(pickles_path)
    except:
        return None

    file_name = out_pickles[0]
    with Path(f"{pickles_path}/{file_name}").open("rb") as file:
        try:
            out_dict = pickle.load(file)
        except EOFError:
            print(f"Problem opening {pickles_path}/{file_name}")
        nevents = out_dict[year][sample_name]["nevents"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with Path(f"{pickles_path}/{file_name}").open("rb") as file:
            try:
                out_dict = pickle.load(file)
            except EOFError:
                print(f"Problem opening {pickles_path}/{file_name}")
            nevents += out_dict[year][sample_name]["nevents"]

    return nevents


def get_pickles(pickles_path, year, sample_name):
    """Accumulates all pickles in ``pickles_path`` directory"""
    out_pickles = [f for f in listdir(pickles_path) if f != ".DS_Store"]

    file_name = out_pickles[0]
    with Path(f"{pickles_path}/{file_name}").open("rb") as file:
        # out = pickle.load(file)[year][sample_name]  # TODO: uncomment and delete below
        out = pickle.load(file)[year]
        sample_name = next(iter(out.keys()))
        out = out[sample_name]

    for file_name in out_pickles[1:]:
        try:
            with Path(f"{pickles_path}/{file_name}").open("rb") as file:
                out_dict = pickle.load(file)[year][sample_name]
                out = accumulate([out, out_dict])
        except:
            warnings.warn(f"Not able to open file {pickles_path}/{file_name}", stacklevel=1)
    return out


def check_selector(sample: str, selector: str | list[str]):
    if not isinstance(selector, (list, tuple)):
        selector = [selector]

    # Case-insensitive matching: the v15 skimmer is inconsistent across years
    # (e.g. ttHto2B_M-125 vs TTHto2B_M-125).  CMS sample names don't collide by
    # case alone, so lowercasing both sides is safe.
    sample_lc = sample.lower()
    for s in selector:
        s = s.lower()  # noqa: PLW2901
        if s.endswith("?"):
            if s[:-1] == sample_lc:
                return True
        elif s.startswith("*"):
            if s[1:] in sample_lc:
                return True
        else:
            if sample_lc.startswith(s):
                return True

    return False


def format_columns(columns: list):
    """
    Reformat input of (`column name`, `num columns`) into (`column name`, `idx`) format for
    reading multiindex columns
    """
    ret_columns = []
    for key, num_columns in columns:
        for i in range(num_columns):
            ret_columns.append(f"('{key}', '{i}')")
    return ret_columns


# The 2022-2023 W->qq+jets samples whose cross sections in xsecs.py are the XSDB (= GenXSecAnalyzer)
# values / 2: their MadGraph process cards generate each W charge twice, so XSDB is 2x the physical
# cross section. The skimmer bakes
# xsecs[sample] x LUMI[year] into every skim weight, and the skims made before the "/ 2" carry the
# XSDB value: ``_apply_w_xsec_correction`` detects such a skim from weight / weight_noxsec, rescales
# it to xsecs.py at load time and warns. Only in the eras that have these samples
# (Run3Summer22/22EE/23/23BPix, one set of gridpacks); the 2024 Bin-PTQQ W samples (each charge
# once) and all Z samples are correct. Any other xsecs.py or LUMI change still needs a re-skim.
W_XSEC_CORRECTION_ERAS = ("2022", "2022EE", "2023", "2023BPix")
W_XSEC_CORRECTION_SAMPLES = frozenset(
    f"Wto2Q-2Jets_PTQQ-{ptqq}_{njets}"
    for ptqq in ["100to200", "200to400", "400to600", "600"]
    for njets in ["1J", "2J"]
)
_W_XSEC_CHECK_RTOL = 1e-9


def _skim_norm_weight_columns(events: pd.DataFrame) -> list[str]:
    """Columns the skimmer multiplied by xsec x LUMI: weight, weight_*, single_weight_*,
    scale_weights, pdf_weights (not weight_noxsec, nor the unnormalised *nonorm* copies)."""
    return [
        col
        for col in events.columns.get_level_values(0).unique()
        if col in {"weight", "scale_weights", "pdf_weights"}
        or (
            col.startswith(("weight_", "single_weight_"))
            and "noxsec" not in col
            and "nonorm" not in col
        )
    ]


def _apply_w_xsec_correction(events: pd.DataFrame, year: str, sample: str) -> None:
    """Rescale a 2022-2023 W skim made with the doubled XSDB cross section to xsecs.py, in place.

    MC only (the callers skip data), before ``finalWeight`` is formed. For a sample of
    ``W_XSEC_CORRECTION_SAMPLES`` in ``W_XSEC_CORRECTION_ERAS``, the skim-time xsec x LUMI is
    weight / weight_noxsec (one value for all events). If xsecs[sample] x LUMI[year] is half of it
    (a skim made with the XSDB value, i.e. before the "/ 2" in xsecs.py), every column of
    ``_skim_norm_weight_columns`` (weight, weight_*, single_weight_*, scale_weights, pdf_weights; not
    weight_noxsec) is multiplied by exactly 0.5, with a warning (warnings.warn and logger.warning)
    each time. If it equals it (a skim made after), nothing is done; anything else raises. No-op
    for every other sample, for years outside ``W_XSEC_CORRECTION_ERAS`` and for empty frames.
    """
    if sample not in W_XSEC_CORRECTION_SAMPLES or year not in W_XSEC_CORRECTION_ERAS:
        return
    if not len(events):
        return
    if "weight_noxsec" not in events:
        raise ValueError(
            f"{sample} ({year}): weight_noxsec is needed to check its skim cross section "
            "(load_samples(load_weight_noxsec=True))"
        )
    w = events["weight"].to_numpy().reshape(len(events), -1)[:, 0]
    wn = events["weight_noxsec"].to_numpy().reshape(len(events), -1)[:, 0]
    nonzero = wn != 0
    if not nonzero.any():
        raise ValueError(f"{sample} ({year}): weight_noxsec is 0 for every event")
    ratio = w[nonzero] / wn[nonzero]
    skim_norm = float(np.median(ratio))
    if not np.allclose(ratio, skim_norm, rtol=_W_XSEC_CHECK_RTOL, atol=0):
        raise ValueError(
            f"{sample} ({year}): weight / weight_noxsec is not one value "
            f"({ratio.min()} to {ratio.max()})"
        )
    norm = xsecs[sample] * LUMI[year]
    rel = norm / skim_norm
    if abs(rel - 1.0) <= _W_XSEC_CHECK_RTOL:
        return  # skim made with the xsecs.py value
    if abs(rel - 0.5) > 0.5 * _W_XSEC_CHECK_RTOL:
        raise ValueError(
            f"{sample} ({year}): skim xsec x lumi {skim_norm:.8g} is neither 1x nor 2x the "
            f"xsecs.py x LUMI {norm:.8g} (ratio {rel:.8g}); re-skim or check xsecs.py"
        )
    factor = 0.5  # exact, so the rescaled weights are bitwise the halved skim weights
    wcols = _skim_norm_weight_columns(events)
    for col in wcols:
        events[col] = events[col].to_numpy() * factor
    msg = (
        f"{sample} ({year}): skim made with the doubled XSDB cross section (its MadGraph process "
        f"card generates each W charge twice); weights rescaled by {factor:g} to the xsecs.py value "
        "(a re-skim with the current xsecs.py needs no rescaling)"
    )
    warnings.warn(msg, stacklevel=2)
    logger.warning(msg)
    logger.debug(f"{sample} ({year}): columns scaled by {factor:g}: {wcols}")


def _normalize_weights(
    events: pd.DataFrame,
    year: str,
    totals: dict,
    sample: str,
    isData: bool,
    variations: bool = True,
    weight_shifts: dict[str, Syst] = None,
):
    """Normalize weights and all the variations"""
    # don't need any reweighting for data
    if isData:
        events["finalWeight"] = events["weight"]
        return

    # check weights are scaled
    if "weight_noxsec" in events and np.all(events["weight"] == events["weight_noxsec"]):
        warnings.warn(f"{sample} has not been scaled by its xsec and lumi!", stacklevel=0)
        events["weight"] = events["weight"].to_numpy() * xsecs[sample] * LUMI[year]
        warnings.warn(
            f"Temporarily scaling {sample} by its xsec and lumi - remember to remove after fixing in the processor!",
            stacklevel=0,
        )

        # if ("VBF" in sample) or ("GluGlutoHHto4B" in sample):
        #     warnings.warn(
        #         f"Temporarily scaling {sample} by its xsec and lumi - remember to remove after fixing in the processor!",
        #         stacklevel=0,
        #     )
        #     events["weight"] = events["weight"].to_numpy() * xsecs[sample] * LUMI[year]
        # else:
        #     raise ValueError(f"{sample} has not been scaled by its xsec and lumi!")

    # 2022-2023 W->qq+jets skims made with the doubled XSDB xsec: rescale to xsecs.py (warns)
    _apply_w_xsec_correction(events, year, sample)

    events["finalWeight"] = events["weight"] / totals["np_nominal"]

    if not variations:
        return

    if weight_shifts is None:
        raise ValueError(
            "Variations requested but no weight shifts given! Please use ``variations=False`` or provide the systematics to be normalized."
        )

    # normalize all the variations
    for wvar in weight_shifts:
        if f"weight_{wvar}Up" not in events:
            continue

        for shift in ["Up", "Down"]:
            wlabel = wvar + shift
            if wvar in norm_preserving_weights:
                # normalize by their totals
                events[f"weight_{wlabel}"] /= totals[f"np_{wlabel}"]
            else:
                # normalize by the nominal
                events[f"weight_{wlabel}"] /= totals["np_nominal"]

    # normalize scale and PDF weights
    for wkey in ["scale_weights", "pdf_weights"]:
        if wkey in events:
            # .to_numpy() makes it way faster
            weights = events[wkey].to_numpy()
            n_weights = weights.shape[1]
            events[wkey] = weights / totals[f"np_{wkey}"][:n_weights]
            if (
                "weight_noxsec" in events
                and np.all(events["weight"] == events["weight_noxsec"])
                and "VBF" in sample
            ):
                warnings.warn(
                    f"Temporarily scaling {sample} by its xsec and lumi - remember to remove after fixing in the processor!",
                    stacklevel=0,
                )
                events[wkey] = events[wkey].to_numpy() * xsecs[sample] * LUMI[year]


def _reorder_txbb(events: pd.DataFrame, txbb):
    # print(f"Reordering by {txbb}")
    """Reorder all the bbFatJet columns by given TXbb"""
    if txbb not in events:
        raise ValueError(
            f"{txbb} not found in events! Need to include that in load columns, or set reorder_legacy_txbb to False."
        )

    bbord = np.argsort(events[txbb].to_numpy(), axis=1)[:, ::-1]
    for key in np.unique(events.columns.get_level_values(0)):
        if key.startswith("bbFatJet"):
            events[key] = np.take_along_axis(events[key].to_numpy(), bbord, axis=1)


def _parquet_has_rows(parquet_file: Path) -> bool:
    """Same answer as ``not pd.read_parquet(parquet_file).empty``, from the file footer only.

    ``DataFrame.empty`` means no rows or no columns. pandas turns the index columns listed in the
    pandas metadata into the index, so those do not count as columns. The full read it replaces
    loads every column of the file (~370 in the skims) just to test for emptiness.
    """
    with pq.ParquetFile(parquet_file) as pf:
        if pf.metadata.num_rows == 0:
            return False
        schema = pf.schema_arrow
    index_columns = (schema.pandas_metadata or {}).get("index_columns", [])
    return len(schema.names) > sum(isinstance(col, str) for col in index_columns)


# 2024 (Summer24) V+jets: {W,Z}to2Q-2Jets_Bin-PTQQ-X is generated with PTQQ > X and NO upper edge,
# so the four samples overlap. By default only Bin-PTQQ-100 is loaded (hh_vars vjets selectors).
# With ``load_samples(..., ptqq_stitch=PtqqStitch(...))`` (PostProcess --vjets-stitch) all four are
# loaded and stitched below. The 2022-2023 *_PTQQ-XtoY_{1J,2J} samples are exclusive bins and do not
# match. The generator-level PTQQ (LHE V pT) is not in the skim, so GenVPt (last-copy V,
# GenSelection.py) is used, and a higher-threshold sample is trusted only from PTQQ_STITCH_MARGIN
# above its threshold, where its GenVPt turn-on has reached the plateau.
# ``PtqqStitch(mode="range")`` is the alternative for a skim with the LHE V pT (``GenVLHEPt``) and
# the per-LHE-V-pT-bin totals (skimmer 7332ed8+): each sample covers only [X_i, X_next) in LHE V pT,
# normalised to sigma_i x (its generator-weight fraction there); see ``ptqq_range_norm``.
_PTQQ_OPEN_RE = re.compile(r"^([wz])to2q-2jets_bin-ptqq-(\d+)$")
PTQQ_STITCH_MARGIN = 50.0  # GeV
PTQQ_OPEN_INCLUSIVE = 100.0  # the lowest threshold, whose sample covers the whole phase space
PTQQ_STITCH_MODES = ("effective-lumi", "range")
# open-ended samples that must not be used (range mode raises if one is selected)
PTQQ_OPEN_INVALID = {
    ("w", 600.0): "Wto2Q-2Jets_Bin-PTQQ-600 (2024) is INVALID in DAS: it was generated with the "
    "LHE V pT filter at 400 GeV, not 600 GeV",
}
# Bin edges (GeV) of the skimmer's per-LHE-V-pT-bin totals (bbbbSkimmer.LHEVPT_EDGES, copied here
# so that postprocessing does not import the processors): arrays of len(edges) + 1 = 202 entries,
# [0] underflow (< 0: no status-2 LHE W/Z), [k] = [edges[k-1], edges[k]), [-1] >= 2000 GeV. The
# thresholds 100/200/400/600 are edges; LHEPart pT is stored with a reduced mantissa, which rounds a
# value just above a threshold to exactly the threshold, never below, so a lower edge is inclusive.
LHEVPT_EDGES = np.arange(0.0, 2001.0, 10.0)
LHEVPT_COLUMN = "GenVLHEPt"


def ptqq_open_threshold(sample: str) -> tuple[str, float] | None:
    """(boson, X) for an open-ended ``{W,Z}to2Q-2Jets_Bin-PTQQ-X`` sample name, else None."""
    m = _PTQQ_OPEN_RE.match(sample.lower())
    return (m.group(1), float(m.group(2))) if m else None


@dataclass
class PtqqStitch:
    """Settings for stitching the open-ended Bin-PTQQ V+jets samples in ``load_samples``.

    Args:
        txbb_presel: load filter max(TXbb_0, TXbb_1) >= txbb_presel on every open-ended sample.
            Set it to the analysis preselection (PostProcess ``H1TXbb >= txbb_presel``, H1 being
            the higher-TXbb jet); it then drops only events that no downstream region can use.
            None: no TXbb load filter.
        txbb_min: optional extra filter min(TXbb_0, TXbb_1) >= txbb_min on the higher-threshold
            samples only, to save memory. The stitch then trusts them only in that region; outside
            it the inclusive sample carries the full weight, exactly as without stitching, so every
            selection stays unbiased. None: the higher-threshold samples are used everywhere.
            Effective-lumi mode only (in range mode no other sample would cover the cut events).
        margin: GenVPt margin (GeV) above its threshold from which a sample is trusted
            (effective-lumi mode only).
        mode: "effective-lumi" (default): GenVPt stitch with effective-luminosity weights
            (``_stitch_open_ptqq``). "range": each sample covers one LHE-V-pT range
            [X_i, X_next) (the highest sample: >= X_i), normalised to sigma_i x f_i with f_i its
            generator-weight fraction in that range (``ptqq_range_norm``); needs a skim with
            ``GenVLHEPt`` and the per-LHE-V-pT-bin totals.
    """

    txbb_presel: float | None = None
    txbb_min: float | None = None
    margin: float = PTQQ_STITCH_MARGIN
    mode: str = "effective-lumi"

    def __post_init__(self):
        if self.mode not in PTQQ_STITCH_MODES:
            raise ValueError(f"PtqqStitch mode must be one of {PTQQ_STITCH_MODES}, got {self.mode}")
        if self.mode == "range" and self.txbb_min is not None:
            raise ValueError(
                "PtqqStitch(mode='range') takes no txbb_min: each LHE V pT range comes from one "
                "sample only, so a min(TXbb) load filter would remove events nothing else covers"
            )


def _lhevpt_range_bins(lo: float, hi: float) -> slice:
    """Slice of the per-LHE-V-pT-bin totals covering [lo, hi) (hi = inf: up to the overflow)."""
    if lo not in LHEVPT_EDGES or not (np.isinf(hi) or hi in LHEVPT_EDGES) or not lo < hi:
        raise ValueError(f"LHE V pT range [{lo}, {hi}) is not on the totals' bin edges")
    # bin k is [edges[k-1], edges[k]), so the bin starting at an edge e has index searchsorted(e)
    k_lo = int(np.searchsorted(LHEVPT_EDGES, lo, side="right"))
    if np.isinf(hi):
        return slice(k_lo, len(LHEVPT_EDGES) + 1)
    return slice(k_lo, int(np.searchsorted(LHEVPT_EDGES, hi, side="right")))


def _lhevpt_array(totals: dict, key: str, sample: str) -> np.ndarray:
    """One per-LHE-V-pT-bin totals array, checked to exist and to have the skimmer's binning."""
    if key not in totals:
        raise ValueError(
            f"{sample}: the pickle totals have no '{key}': this skim predates the per-LHE-V-pT-bin "
            "totals (skimmer commit 7332ed8), which the range stitch needs; use the LHE-pT re-skim "
            "(e.g. PostProcess --override-tag 20260820_glopartv3_reskim_v15_signal)"
        )
    arr = np.asarray(totals[key], dtype=np.float64)
    if arr.shape != (len(LHEVPT_EDGES) + 1,):
        raise ValueError(f"{sample}: '{key}' has shape {arr.shape}, not the skimmer's binning")
    return arr


def ptqq_range_norm(
    totals: dict, lo: float, hi: float, syst_labels: list[str] = (), sample: str = ""
) -> dict:
    """Normalisation of one open-ended Bin-PTQQ sample restricted to LHE V pT in [lo, hi).

    The sample's cross section in the range is sigma_i x f with f the fraction of its generated
    events there, counted as the signed sum of generator weights over ALL generated events (no other
    cut): ``f = sum_range genweight_lhevpt / sum_all genweight_lhevpt`` (the denominator is the
    sample's ``nevents``, summed in float64 over the same events as the numerator).

    ``_normalize_weights`` divides the nominal by ``np_nominal``, so the events of the range carry
    sigma_i L F_nominal, with F_nominal = sum_range np_nominal_lhevpt / np_nominal; the factor
    f / F_nominal makes that sigma_i L f exactly. A norm-preserving variation s normalised by its own
    total np_s (``syst_labels``, e.g. "pileupUp") gets f / F_s, F_s = sum_range np_s_lhevpt / np_s,
    so every such variation keeps the range normalisation sigma_i L f.

    If the range holds every generated event (the highest sample, nothing below its threshold),
    f and every F are exactly 1: the sample is taken whole at its cross section, weights unchanged.

    Returns a dict with "f" (generator-weight fraction), "f_raw" (raw-count fraction, for
    information), "F" and "factor" (dicts keyed by "nominal" and each syst label), and the
    generated events / genweight fraction below ``lo`` ("n_below", "f_below").
    """
    rng = _lhevpt_range_bins(lo, hi)
    gen = _lhevpt_array(totals, "genweight_lhevpt", sample)
    nraw = _lhevpt_array(totals, "nevents_lhevpt", sample)
    whole = nraw[rng].sum() == nraw.sum()  # every generated event is in the range
    f = 1.0 if whole else gen[rng].sum() / gen.sum()
    out = {
        "f": f,
        "f_raw": nraw[rng].sum() / nraw.sum(),
        "n_below": int(nraw[: rng.start].sum()),
        "f_below": gen[: rng.start].sum() / gen.sum(),
        "F": {},
        "factor": {},
    }
    for label in ["nominal", *syst_labels]:
        per_bin = _lhevpt_array(totals, f"np_{label}_lhevpt", sample)
        total = totals[f"np_{label}"]
        big_f = 1.0 if whole else per_bin[rng].sum() / total
        if not (np.isfinite(big_f) and big_f > 0 and np.isfinite(f) and f > 0):
            raise ValueError(f"{sample}: no generated events in LHE V pT [{lo}, {hi}) ({label})")
        out["F"][label] = big_f
        out["factor"][label] = f / big_f
    return out


def _own_total_variations(
    events: pd.DataFrame, variations: bool, weight_shifts: dict[str, Syst] | None
) -> list[str]:
    """The ``weight_<label>`` variations ``_normalize_weights`` divides by their own np_<label>."""
    if not variations or weight_shifts is None:
        return []
    return [
        wvar + shift
        for wvar in weight_shifts
        if f"weight_{wvar}Up" in events and wvar in norm_preserving_weights
        for shift in ["Up", "Down"]
    ]


def _stitch_weight_columns(events: pd.DataFrame) -> list[str]:
    """Weight columns a stitch factor multiplies (not the unnormalised ``*noxsec*``/``*nonorm*``)."""
    return [
        col
        for col in events.columns.get_level_values(0).unique()
        if col in {"weight", "finalWeight", "scale_weights", "pdf_weights"}
        or (col.startswith("weight_") and "noxsec" not in col and "nonorm" not in col)
    ]


def _ptqq_range_ranges(samples: list[str], selector) -> dict[str, tuple[float, float]]:
    """{sample: (lo, hi)} LHE V pT range of every open-ended sample the selector matches.

    Per boson the thresholds X_1 < ... < X_n of the matched samples split the axis: sample k keeps
    [X_k, X_k+1), the highest one [X_n, inf). Raises for an invalid sample (``PTQQ_OPEN_INVALID``).
    """
    by_boson: dict[str, list[tuple[float, str]]] = {}
    for sample in samples:
        thr = ptqq_open_threshold(sample) if check_selector(sample, selector) else None
        if thr is None:
            continue
        if thr in PTQQ_OPEN_INVALID:
            raise ValueError(f"PTQQ range stitch: {sample} selected, but {PTQQ_OPEN_INVALID[thr]}")
        by_boson.setdefault(thr[0], []).append((thr[1], sample))
    ranges = {}
    for boson, members in by_boson.items():
        members.sort()
        xs = [x for x, _ in members]
        if xs[0] != PTQQ_OPEN_INCLUSIVE:
            raise ValueError(
                f"PTQQ range stitch ({boson.upper()}): the lowest sample must be Bin-PTQQ-"
                f"{PTQQ_OPEN_INCLUSIVE:.0f} (matched: {[s for _, s in members]})"
            )
        for k, (x, sample) in enumerate(members):
            ranges[sample] = (x, xs[k + 1] if k + 1 < len(xs) else np.inf)
    return ranges


def _ptqq_range_filters(filters: list | None, lo: float, hi: float) -> list:
    """``filters`` AND lo <= GenVLHEPt (< hi), as a pyarrow DNF filter."""
    col = f"('{LHEVPT_COLUMN}', '0')"
    clause = [(col, ">=", lo)] + ([] if np.isinf(hi) else [(col, "<", hi)])
    return _and_dnf(filters, [clause])


def _check_range_sample(parquet_path: Path, pickles_path: Path, sample: str, lo, hi) -> None:
    """Raise if a range-stitch sample has no events at all (its LHE V pT range would be missing)
    or a non-empty parquet file without GenVLHEPt (a skim before 7e8dca7); warn if its parquet and
    pickle jobs differ (the normalisation assumes they cover the same generated events)."""
    col = f"('{LHEVPT_COLUMN}', '0')"
    files = sorted(parquet_path.glob("*.parquet")) if parquet_path.is_dir() else []
    files = [parquet_file for parquet_file in files if _parquet_has_rows(parquet_file)]
    if not files:
        raise ValueError(
            f"PTQQ range stitch: {sample} has no events in {parquet_path}, so its LHE V pT range "
            f"[{lo:g}, {hi:g}) would be missing"
        )
    for parquet_file in files:
        if col not in pq.read_schema(parquet_file).names:
            raise ValueError(
                f"{sample}: {parquet_file} has no {LHEVPT_COLUMN} column: this skim predates the "
                "LHE V pT (skimmer 7e8dca7/7332ed8), which the range stitch needs; use the LHE-pT "
                "re-skim (e.g. PostProcess --override-tag 20260820_glopartv3_reskim_v15_signal)"
            )
    jobs_pq = {p.stem for p in parquet_path.glob("*.parquet")}
    jobs_pk = {p.stem for p in pickles_path.glob("*.pkl")}
    if jobs_pq != jobs_pk:
        warnings.warn(
            f"{sample}: parquet and pickle jobs differ (parquet only: {sorted(jobs_pq - jobs_pk)}, "
            f"pickles only: {sorted(jobs_pk - jobs_pq)}); the range normalisation assumes they "
            "cover the same generated events",
            stacklevel=2,
        )


def _apply_ptqq_range(
    events: pd.DataFrame,
    sample: str,
    totals: dict,
    lo: float,
    hi: float,
    own_total: list[str],
) -> pd.DataFrame:
    """Range stitch of one open-ended sample: keep lo <= GenVLHEPt < hi and rescale its weights.

    ``weight_<s>`` for s in ``own_total`` (normalised by np_s) gets f / F_s, every other weight
    column f / F_nominal (``ptqq_range_norm``). Returns the (possibly row-reduced) frame.
    """
    norm = ptqq_range_norm(totals, lo, hi, own_total, sample)
    v = events[LHEVPT_COLUMN].to_numpy().reshape(len(events), -1)[:, 0]
    keep = (v >= lo) & (v < hi)
    n_out = int(np.sum(~keep))
    if n_out:
        # the load filter already keeps only the range; this is a safety net (e.g. no pushdown)
        events = events.loc[keep].reset_index(drop=True)
    before = events["finalWeight"].to_numpy().sum()
    for col in _stitch_weight_columns(events):
        label = col[len("weight_") :] if col.startswith("weight_") else None
        factor = norm["factor"][label if label in own_total else "nominal"]
        events[col] = events[col].to_numpy() * factor
    if norm["n_below"]:
        warnings.warn(
            f"{sample}: {norm['n_below']} generated events ({norm['f_below']:.3g} of the genweight) "
            f"have LHE V pT below the sample threshold {lo:g} GeV; they are not used",
            stacklevel=1,
        )
    logger.info(
        f"PTQQ range stitch {sample}: LHE V pT [{lo:g}, {hi:g}), f={norm['f']:.6g} "
        f"(raw-count {norm['f_raw']:.6g}), F_nominal={norm['F']['nominal']:.6g}, "
        f"factor={norm['factor']['nominal']:.6g}, {len(events)} events ({n_out} outside the "
        f"range dropped after the load filter), sum finalWeight {before:.6g} -> "
        f"{events['finalWeight'].to_numpy().sum():.6g}"
    )
    return events


def _and_dnf(filters: list | None, clauses: list[list[tuple]]) -> list[list[tuple]]:
    """``filters AND (clauses[0] OR clauses[1] OR ...)`` as a pyarrow DNF filter (OR of AND-lists)."""
    if not filters:
        base = [[]]
    elif isinstance(filters[0], tuple):  # a single flat AND-list
        base = [list(filters)]
    else:
        base = [list(clause) for clause in filters]
    return [b + c for b in base for c in clauses]


def _ptqq_stitch_filters(
    filters: list | None, txbb_str: str, stitch: PtqqStitch, higher: bool
) -> list | None:
    """The caller's filters plus the ``PtqqStitch`` TXbb filters for one open-ended sample."""
    tx0, tx1 = (f"('{txbb_str}', '{i}')" for i in range(2))
    if stitch.txbb_presel is not None:
        cut = stitch.txbb_presel
        filters = _and_dnf(filters, [[(tx0, ">=", cut)], [(tx1, ">=", cut)]])
    if higher and stitch.txbb_min is not None:
        cut = stitch.txbb_min
        filters = _and_dnf(filters, [[(tx0, ">=", cut), (tx1, ">=", cut)]])
    return filters


def _stitch_open_ptqq(
    loaded: list[tuple[str, pd.DataFrame]], txbb_str: str, stitch: PtqqStitch
) -> None:
    """Effective-luminosity stitching of the open-ended Bin-PTQQ samples, in place.

    For an event with GenVPt v, the trusted samples of its boson are the inclusive (lowest
    threshold) one, plus every sample with v >= X + margin when the event is in the stitch region
    (min(TXbb_0, TXbb_1) >= ``stitch.txbb_min`` if set, else everywhere). An event from trusted
    sample i has all its weights scaled by c_i / sum_j c_j over the trusted samples, with
    c = 1 / median|finalWeight| (the inverse per-event weight, i.e. an effective luminosity); an
    event from an untrusted sample gets 0. The factors sum to 1 at every (v, region), so every
    yield is unbiased, and the low-weight high-threshold samples dominate the high-pT tail.
    """
    groups: dict[str, list[tuple[float, str, pd.DataFrame]]] = {}
    for sample, events in loaded:
        thr = ptqq_open_threshold(sample)
        if thr is not None:
            groups.setdefault(thr[0], []).append((thr[1], sample, events))

    for boson, members in groups.items():
        xs = np.array([x for x, _, _ in members])
        if xs.min() != PTQQ_OPEN_INCLUSIVE:
            raise ValueError(
                f"PTQQ stitch ({boson.upper()}): the inclusive Bin-PTQQ-{PTQQ_OPEN_INCLUSIVE:.0f} "
                f"sample was not loaded (loaded: {[s for _, s, _ in members]})"
            )
        cs = np.array(
            [1.0 / np.median(np.abs(ev["finalWeight"].to_numpy())) for _, _, ev in members]
        )
        if not np.all(np.isfinite(cs) & (cs > 0)):
            raise ValueError(f"PTQQ stitch ({boson.upper()}): bad effective luminosities {cs}")

        for i, (_, sample, events) in enumerate(members):
            v = events["GenVPt"].to_numpy().reshape(len(events), -1)[:, 0]
            if not np.all(np.isfinite(v)):
                warnings.warn(
                    f"{sample}: {np.sum(~np.isfinite(v))} events without a finite GenVPt; only "
                    "the inclusive sample is trusted for them",
                    stacklevel=1,
                )
            higher_trusted = v[:, None] >= xs[None, :] + stitch.margin
            if stitch.txbb_min is not None:
                txbb = events[txbb_str].to_numpy()[:, :2]
                higher_trusted &= (np.min(txbb, axis=1) >= stitch.txbb_min)[:, None]
            trusted = (xs[None, :] == xs.min()) | higher_trusted
            factor = np.where(trusted[:, i], cs[i] / (trusted * cs[None, :]).sum(axis=1), 0.0)

            before = events["finalWeight"].to_numpy().sum()
            wcols = [
                col
                for col in events.columns.get_level_values(0).unique()
                if col in {"weight", "finalWeight", "scale_weights", "pdf_weights"}
                or (col.startswith("weight_") and "noxsec" not in col and "nonorm" not in col)
            ]
            for col in wcols:
                vals = events[col].to_numpy()
                events[col] = vals * (factor[:, None] if vals.ndim == 2 else factor)
            logger.info(
                f"PTQQ stitch {sample}: c={cs[i]:.4g}, {len(events)} events "
                f"({np.sum(factor == 0)} untrusted), sum finalWeight {before:.4g} -> "
                f"{events['finalWeight'].to_numpy().sum():.4g} "
                f"(txbb_presel={stitch.txbb_presel}, txbb_min={stitch.txbb_min})"
            )


def load_samples(
    data_dir: Path,
    samples: dict[str, str],
    year: str,
    filters: list = None,
    columns: list = None,
    variations: bool = True,
    weight_shifts: dict[str, Syst] = None,
    reorder_txbb: bool = False,  # temporary fix for sorting by given Txbb
    txbb_str: str = "bbFatJetPNetTXbbLegacy",
    load_weight_noxsec: bool = True,
    ptqq_stitch: PtqqStitch | None = None,
    override_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Loads events with an optional filter.
    Divides MC samples by the total pre-skimming, to take the acceptance into account.

    Args:
        data_dir (str): path to data directory.
        samples (Dict[str, str]): dictionary of samples and selectors to load.
        year (str): year.
        filters (List): Optional filters when loading data.
        columns (List): Optional columns to load.
        variations (bool): Normalize variations as well (saves time to not do so). Defaults to True.
        weight_shifts (Dict[str, Syst]): dictionary of weight shifts to consider.
        ptqq_stitch (PtqqStitch): opt-in stitching of the overlapping open-ended 2024 V+jets
            ``{W,Z}to2Q-2Jets_Bin-PTQQ-X`` samples selected under one label (see ``PtqqStitch``,
            ``_stitch_open_ptqq`` and, for mode "range", ``ptqq_range_norm``). None (default):
            every sample is loaded as it is.
        override_dir (Path): optional second skim directory (a skimmer tag, like ``data_dir``):
            every sample directory in ``override_dir / year`` replaces the same-named one of
            ``data_dir / year`` (e.g. the LHE-pT re-skim of the 2024 V+jets and SM ggHH). None
            (default), or no ``year`` directory there: everything is read from ``data_dir``.

    Returns:
        Dict[str, pd.DataFrame]: ``events_dict`` dictionary of events dataframe for each sample.

    """
    events_dict = {}

    data_dir = Path(data_dir) / year
    full_samples_list = listdir(data_dir)  # get all directories in data_dir
    sample_dirs = {}  # samples read from override_dir instead of data_dir
    if override_dir is not None and (Path(override_dir) / year).is_dir():
        for sample in sorted(listdir(Path(override_dir) / year)):
            sample_dirs[sample] = Path(override_dir) / year / sample
            if sample not in full_samples_list:
                full_samples_list.append(sample)
        logger.info(f"Samples read from {Path(override_dir) / year}: {sorted(sample_dirs)}")

    logger.debug(f"Full list of directories in {data_dir}: {full_samples_list}")
    logger.debug(f"Samples to load {samples}")

    # label - key of sample in events_dict
    # selector - string used to select directories to load in for this sample
    for label, selector in samples.items():
        # important to check that samples have been normalized properly
        load_columns = columns
        if label != "data" and load_weight_noxsec:
            load_columns = columns + format_columns([("weight_noxsec", 1)])

        # lowest open-ended Bin-PTQQ threshold per boson among this label's samples (for the stitch)
        stitch_lowest = {}
        if ptqq_stitch is not None:
            for sample in full_samples_list:
                thr = ptqq_open_threshold(sample) if check_selector(sample, selector) else None
                if thr is not None:
                    stitch_lowest[thr[0]] = min(thr[1], stitch_lowest.get(thr[0], np.inf))
        range_mode = ptqq_stitch is not None and ptqq_stitch.mode == "range"
        # range stitch: the LHE V pT range [lo, hi) of every matched open-ended sample
        ranges = _ptqq_range_ranges(full_samples_list, selector) if range_mode else {}
        # columns read only for the stitch (dropped after it unless requested)
        stitch_columns = format_columns(
            [("GenVLHEPt" if range_mode else "GenVPt", 1), (txbb_str, 2)]
        )

        events_dict[label] = []  # list of directories we load in for this sample
        loaded_names = []  # sample (directory) name of each entry in events_dict[label]
        for sample in full_samples_list:
            # check if this directory passes our selector string
            if not check_selector(sample, selector):
                continue

            # open-ended Bin-PTQQ samples: GenVPt (GenVLHEPt in range mode) + TXbb for the stitch,
            # and its TXbb (and LHE V pT range) load filters
            sample_columns, sample_filters = load_columns, filters
            thr = ptqq_open_threshold(sample) if ptqq_stitch is not None else None
            if thr is not None:
                if load_columns is not None:
                    sample_columns = load_columns + [
                        col for col in stitch_columns if col not in load_columns
                    ]
                sample_filters = _ptqq_stitch_filters(
                    filters, txbb_str, ptqq_stitch, higher=thr[1] > stitch_lowest[thr[0]]
                )
                if range_mode:
                    sample_filters = _ptqq_range_filters(sample_filters, *ranges[sample])

            sample_path = sample_dirs.get(sample, data_dir / sample)
            parquet_path, pickles_path = sample_path / "parquet", sample_path / "pickles"
            if range_mode and sample in ranges:
                _check_range_sample(parquet_path, pickles_path, sample, *ranges[sample])

            # no parquet directory?
            if not parquet_path.exists():
                warnings.warn(f"No parquet directory for {sample}!", stacklevel=1)
                continue

            logger.debug(f"Loading {sample}")
            try:
                non_empty_passed_list = []
                for parquet_file in parquet_path.glob("*.parquet"):
                    if _parquet_has_rows(parquet_file):
                        df_sample = pd.read_parquet(
                            parquet_file, filters=sample_filters, columns=sample_columns
                        )
                        non_empty_passed_list.append(df_sample)
                if not non_empty_passed_list:
                    warnings.warn(f"No events after filtering for {sample}!", stacklevel=1)
                    continue
                events = pd.concat(non_empty_passed_list, ignore_index=True)
            except Exception:
                warnings.warn(
                    f"Can't read file with requested columns/filters for {sample}!", stacklevel=1
                )
                non_empty_passed_list = []
                for parquet_file in parquet_path.glob("*.parquet"):
                    if _parquet_has_rows(parquet_file):
                        df_sample = pd.read_parquet(
                            parquet_file, filters=sample_filters, columns=sample_columns
                        )
                        non_empty_passed_list.append(df_sample)
                if not non_empty_passed_list:
                    warnings.warn(f"No events after filtering for {sample}!", stacklevel=1)
                    continue
                events = pd.concat(non_empty_passed_list, ignore_index=True)
                continue

            # no events?
            if not len(events):
                warnings.warn(f"No events for {sample}!", stacklevel=1)
                continue

            if reorder_txbb:
                _reorder_txbb(events, txbb_str)

            # normalize by total events
            pickles = get_pickles(pickles_path, year, sample)
            if "totals" in pickles:
                totals = pickles["totals"]
                _normalize_weights(
                    events,
                    year,
                    totals,
                    sample,
                    isData=label == data_key,
                    variations=variations,
                    weight_shifts=weight_shifts,
                )
            elif range_mode and sample in ranges:
                raise ValueError(
                    f"{sample}: the pickles have no totals, which the PTQQ range stitch needs"
                )
            else:
                if label == data_key:
                    events["finalWeight"] = events["weight"]
                else:
                    # the 2022-2023 W xsec correction of _normalize_weights (weight_nonorm too)
                    _apply_w_xsec_correction(events, year, sample)
                    n_events = get_nevents(pickles_path, year, sample)
                    events["weight_nonorm"] = events["weight"]
                    events["finalWeight"] = events["weight"] / n_events

            if range_mode and sample in ranges:
                # keep only this sample's LHE V pT range, normalised to sigma x f (after
                # _normalize_weights, whose np_* totals the factors correct)
                events = _apply_ptqq_range(
                    events,
                    sample,
                    totals,
                    *ranges[sample],
                    _own_total_variations(events, variations, weight_shifts),
                )

            events_dict[label].append(events)
            loaded_names.append(sample)
            logger.info(f"Loaded {sample: <50}: {len(events)} entries")

        stitch_only_columns = set()  # parquet names read only for the stitch, dropped after it
        if ptqq_stitch is not None and len(events_dict[label]):
            if not range_mode:
                # remove the overlap between the open-ended Bin-PTQQ V+jets samples (2024 MC)
                _stitch_open_ptqq(
                    list(zip(loaded_names, events_dict[label])), txbb_str, ptqq_stitch
                )
            if load_columns is not None:
                stitch_only_columns = {col for col in stitch_columns if col not in load_columns}

        if len(events_dict[label]):
            events_dict[label] = pd.concat(events_dict[label])
            # Deduplicate columns that can arise when concatenating multiple sub-samples
            keep = ~events_dict[label].columns.duplicated()
            if stitch_only_columns:
                # match by the parquet name "('name', 'i')": pandas restores the second column
                # level from the parquet metadata (int64 in the skims), so tuples of str miss it
                keep &= ~np.array(
                    [
                        isinstance(col, tuple)
                        and f"('{col[0]}', '{col[1]}')" in stitch_only_columns
                        for col in events_dict[label].columns
                    ]
                )
            events_dict[label] = events_dict[label].loc[:, keep]
        else:
            del events_dict[label]

    return events_dict


def add_to_cutflow(
    events_dict: dict[str, pd.DataFrame],
    key: str,
    weight_key: str,
    cutflow: pd.DataFrame,
):
    cutflow[key] = [
        np.sum(events_dict[sample][weight_key]).squeeze() for sample in list(cutflow.index)
    ]


def get_key_index(h: Hist, axis_name: str):
    """Get the index of a key in a Hist's first axis"""
    return np.where(np.array(list(h.axes[0])) == axis_name)[0][0]


def getParticles(particle_list, particle_type):
    """
    Finds particles in `particle_list` of type `particle_type`

    Args:
        particle_list: array of particle pdgIds
        particle_type: can be 1) string: 'b', 'V' currently, or TODO: 2) pdgID, 3) list of pdgIds
    """

    B_PDGID = 5
    Z_PDGID = 23
    W_PDGID = 24

    if particle_type == "b":
        return abs(particle_list) == B_PDGID

    if particle_type == "V":
        return (abs(particle_list) == W_PDGID) + (abs(particle_list) == Z_PDGID)

    raise NotImplementedError


# check if string is an int
def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_feat(events: pd.DataFrame, feat: str):
    if feat in events:
        return np.nan_to_num(events[feat].to_numpy().squeeze(), -1)

    if _is_int(feat[-1]):
        return np.nan_to_num(events[feat[:-1]].to_numpy()[:, int(feat[-1])].squeeze(), -1)

    return None


def tau32FittedSF_4(events: pd.DataFrame):
    tau32 = {"ak8FatJetTau3OverTau20": get_feat(events, "ak8FatJetTau3OverTau20")}[
        "ak8FatJetTau3OverTau20"
    ]
    return np.where(
        tau32 < 0.5,
        18.4912 - 235.086 * tau32 + 1098.94 * tau32**2 - 2163 * tau32**3 + 1530.59 * tau32**4,
        1,
    )


def makeHH(events: pd.DataFrame, key: str, mass: str):

    h1 = vector.array(
        {
            "pt": events[key]["bbFatJetPt"].to_numpy()[:, 0],
            "phi": events[key]["bbFatJetPhi"].to_numpy()[:, 0],
            "eta": events[key]["bbFatJetEta"].to_numpy()[:, 0],
            "M": events[key][mass].to_numpy()[:, 0],
        }
    )
    h2 = vector.array(
        {
            "pt": events[key]["bbFatJetPt"].to_numpy()[:, 1],
            "phi": events[key]["bbFatJetPhi"].to_numpy()[:, 1],
            "eta": events[key]["bbFatJetEta"].to_numpy()[:, 1],
            "M": events[key][mass].to_numpy()[:, 1],
        }
    )
    mask_h1 = h1.pt < 0
    mask_h2 = h2.pt < 0
    mask_invalid = mask_h1 | mask_h2

    hh = h1 + h2
    # Convert vectors to numpy arrays for conditional manipulation
    hh_pt = hh.pt
    hh_phi = hh.phi
    hh_eta = hh.eta
    hh_M = hh.M

    # Apply pad value
    hh_pt[mask_invalid] = -PAD_VAL
    hh_phi[mask_invalid] = -PAD_VAL
    hh_eta[mask_invalid] = -PAD_VAL
    hh_M[mask_invalid] = -PAD_VAL

    # Re-make the vector with padded entries
    hh = vector.array({"pt": hh_pt, "phi": hh_phi, "eta": hh_eta, "M": hh_M})
    return hh


def get_feat_first(events: pd.DataFrame, feat: str):
    return events[feat][0].to_numpy().squeeze()


def make_vector(events: dict, name: str, mask=None, mstring="Mass"):
    """
    Creates Lorentz vector from input events and beginning name, assuming events contain
      {name}Pt, {name}Phi, {name}Eta, {Name}Msd variables
    Optional input mask to select certain events

    Args:
        events (dict): dict of variables and corresponding numpy arrays
        name (str): object string e.g. ak8FatJet
        mask (bool array, optional): array selecting desired events
    """
    if mask is None:
        return vector.array(
            {
                "pt": get_feat(events, f"{name}Pt"),
                "phi": get_feat(events, f"{name}Phi"),
                "eta": get_feat(events, f"{name}Eta"),
                "M": get_feat(events, f"{name}{mstring}"),
            }
        )

    return vector.array(
        {
            "pt": get_feat(events, f"{name}Pt")[mask],
            "phi": get_feat(events, f"{name}Phi")[mask],
            "eta": get_feat(events, f"{name}Eta")[mask],
            "M": get_feat(events, f"{name}{mstring}")[mask],
        }
    )


# TODO: extend to multi axis using https://stackoverflow.com/a/47859801/3759946 for 2D blinding
def blindBins(h: Hist, blind_region: list, blind_sample: str | None = None, axis=0):
    """
    Blind (i.e. zero) bins in histogram ``h``.
    If ``blind_sample`` specified, only blind that sample, else blinds all.
    """
    if axis > 0:
        raise Exception("not implemented > 1D blinding yet")

    bins = h.axes[axis + 1].edges
    lv = int(np.searchsorted(bins, blind_region[0], "right"))
    rv = int(np.searchsorted(bins, blind_region[1], "left") + 1)

    if blind_sample is not None:
        data_key_index = np.where(np.array(list(h.axes[0])) == blind_sample)[0][0]
        h.view(flow=True)[data_key_index][lv:rv].value = 0
        h.view(flow=True)[data_key_index][lv:rv].variance = 0
    else:
        h.view(flow=True)[:, lv:rv].value = 0
        h.view(flow=True)[:, lv:rv].variance = 0


def singleVarHist(
    events_dict: dict[str, pd.DataFrame],
    shape_var: ShapeVar,
    weight_key: str = "finalWeight",
    selection: dict | None = None,
) -> Hist:
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format
          {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        shape_var (ShapeVar): ShapeVar object specifying the variable, label, binning, and (optionally) a blinding window.
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut].
          Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for
          each sample
    """
    samples = list(events_dict.keys())

    h = Hist(
        hist.axis.StrCategory(samples, name="Sample"),
        shape_var.axis,
        storage="weight",
    )

    var = shape_var.var

    for sample in samples:
        events = events_dict[sample]
        if sample == "data" and var.endswith(("_up", "_down")):
            fill_var = "_".join(var.split("_")[:-2])
        else:
            fill_var = var

        fill_data = {var: get_feat(events, fill_var)}
        weight = events[weight_key].to_numpy().squeeze()

        if selection is not None:
            sel = selection[sample]
            fill_data[var] = fill_data[var][sel]
            weight = weight[sel]

        # if sf is not None and year is not None and sample == "ttbar" and apply_tt_sf:
        #     weight = weight   * tau32FittedSF_4(events) * ttbar_pTjjSF(year, events)

        if fill_data[var] is not None:
            h.fill(Sample=sample, **fill_data, weight=weight)

    if shape_var.blind_window is not None:
        blindBins(h, shape_var.blind_window, data_key)

    return h


def singleVarHistSel(
    events_dict: dict[str, pd.DataFrame],
    shape_var: ShapeVar,
    samples: list[str],
    weight_key: str = "finalWeight",
    selection: dict | None = None,
) -> Hist:
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format
          {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        shape_var (ShapeVar): ShapeVar object specifying the variable, label, binning, and (optionally) a blinding window.
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut].
          Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for
          each sample
    """

    h = Hist(
        hist.axis.StrCategory(samples, name="Sample"),
        shape_var.axis,
        storage="weight",
    )

    var = shape_var.var

    for sample in samples:
        events = events_dict[sample]
        if sample == "data" and var.endswith(("_up", "_down")):
            fill_var = "_".join(var.split("_")[:-2])
        else:
            fill_var = var

        # TODO: add b1, b2 assignment if needed
        fill_data = {var: get_feat(events, fill_var)}
        weight = events[weight_key].to_numpy().squeeze()

        if selection is not None:
            sel = selection[sample]
            fill_data[var] = fill_data[var][sel]
            weight = weight[sel]

        if len(fill_data[var]):
            h.fill(Sample=sample, **fill_data, weight=weight)

    if shape_var.blind_window is not None:
        blindBins(h, shape_var.blind_window, data_key)

    return h


def singleVarHistNoMask(
    events_dict: dict[str, pd.DataFrame],
    var: str,
    bins: list,
    label: str,
    weight_key: str = "finalWeight",
    blind_region: list | None = None,
    selection: dict | None = None,
) -> Hist:
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format
          {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        var (str): variable inside the events dict to make a histogram of
        bins (list): bins in Hist format i.e. [num_bins, min_value, max_value]
        label (str): label for variable (shows up when plotting)
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut].
          Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for
          each sample
    """
    samples = list(events_dict.keys())

    h = Hist.new.StrCat(samples, name="Sample").Reg(*bins, name=var, label=label).Weight()

    for sample in samples:
        events = events_dict[sample]
        fill_data = {var: get_feat_first(events, var)}
        weight = events[weight_key].to_numpy().squeeze()

        if selection is not None:
            sel = selection[sample]
            fill_data[var] = fill_data[var][sel]
            weight = weight[sel]

        h.fill(Sample=sample, **fill_data, weight=weight)

    if blind_region is not None:
        blindBins(h, blind_region, data_key)

    return h


def add_selection(name, sel, selection, cutflow, events, weight_key):
    """Adds selection to PackedSelection object and the cutflow"""
    selection.add(name, sel)
    weight = get_feat(events, weight_key)
    if cutflow is not None:
        cutflow[name] = np.sum(weight[selection.all(*selection.names)])


def check_get_jec_var(var, jshift):
    """Checks if var is affected by the JEC / JMSR and if so, returns the shifted var name"""

    if jshift in jec_shifts and var in jec_vars:
        return var + "_" + jshift

    if jshift in jmsr_shifts and var in jmsr_vars:
        return var + "_" + jshift

    return var


def get_var_mapping(jshift):
    """Returns function that maps var to shifted var for a given systematic shift [JES|JER|JMS|JMR]_[up|down]"""

    def var_mapping(var):
        return check_get_jec_var(var, jshift)

    return var_mapping


def _var_selection(
    events: pd.DataFrame,
    var: str,
    brange: list[float],
    sample: str,
    jshift: str,
    MAX_VAL: float = CUT_MAX_VAL,
):
    """get selection for a single cut, including logic for OR-ing cut on two vars"""
    rmin, rmax = brange
    cut_vars = var.split("+")

    sels = []
    selstrs = []

    # OR the different vars
    for cutvar in cut_vars:
        if (jshift in jmsr_shifts and sample in jmsr_keys) or (
            jshift in jec_shifts and sample in syst_keys
        ):
            var = check_get_jec_var(cutvar, jshift)
        else:
            var = cutvar

        vals = get_feat(events, var)

        if rmin == -MAX_VAL:
            sels.append(vals < rmax)
            selstrs.append(f"{var} < {rmax}")
        elif rmax == MAX_VAL:
            sels.append(vals >= rmin)
            selstrs.append(f"{var} >= {rmin}")
        else:
            sels.append((vals >= rmin) & (vals < rmax))
            selstrs.append(f"{rmin} ≤ {var} < {rmax}")

    sel = np.sum(sels, axis=0).astype(bool)
    selstr = " or ".join(selstrs)

    return sel, selstr


def make_selection(
    var_cuts: dict[str, list[float]],
    events_dict: dict[str, pd.DataFrame],
    weight_key: str = "finalWeight",
    prev_cutflow: dict = None,
    selection: dict[str, np.ndarray] = None,
    jshift: str = "",
    MAX_VAL: float = CUT_MAX_VAL,
):
    """
    Makes cuts defined in `var_cuts` for each sample in `events`.

    Selection syntax:

    Simple cut:
    "var": [lower cut value, upper cut value]

    OR cut on `var`:
    "var": [[lower cut1 value, upper cut1 value], [lower cut2 value, upper cut2 value]] ...

    OR same cut(s) on multiple vars:
    "var1+var2": [lower cut value, upper cut value]

    TODO: OR more general cuts

    Args:
        var_cuts (dict): a dict of cuts, with each (key, value) pair = {var: [lower cut value, upper cut value], ...}.
        events (dict): a dict of events of format {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        weight_key (str): key to use for weights. Defaults to 'finalWeight'.
        prev_cutflow (dict): cutflow from previous cuts, if any. Defaults to None.
        selection (dict): previous selection, if any. Defaults to None.
        MAX_VAL (float): if abs of one of the cuts equals or exceeds this value it will be ignored. Defaults to 9999.

    Returns:
        selection (dict): dict of each sample's cut boolean arrays.
        cutflow (dict): dict of each sample's yields after each cut.
    """
    selection = {} if selection is None else deepcopy(selection)

    cutflow = {}

    for sample, events in events_dict.items():
        if sample not in cutflow:
            cutflow[sample] = {}

        if sample in selection:
            new_selection = PackedSelection()
            new_selection.add("Previous selection", selection[sample])
            selection[sample] = new_selection
        else:
            selection[sample] = PackedSelection()

        for cutvar, branges in var_cuts.items():
            if isinstance(branges[0], list):
                cut_vars = cutvar.split("+")
                if len(cut_vars) > 1:
                    assert len(cut_vars) == len(
                        branges
                    ), "If OR-ing different variables' cuts, num(cuts) must equal num(vars)"

                # OR the cuts
                sels = []
                selstrs = []
                for i, brange in enumerate(branges):
                    cvar = cut_vars[i] if len(cut_vars) > 1 else cut_vars[0]
                    sel, selstr = _var_selection(events, cvar, brange, sample, jshift, MAX_VAL)
                    sels.append(sel)
                    selstrs.append(selstr)

                sel = np.sum(sels, axis=0).astype(bool)
                selstr = " or ".join(selstrs)
            else:
                sel, selstr = _var_selection(events, cutvar, branges, sample, jshift, MAX_VAL)

            add_selection(
                selstr,
                sel,
                selection[sample],
                cutflow[sample],
                events,
                weight_key,
            )

        selection[sample] = selection[sample].all(*selection[sample].names)

    cutflow = pd.DataFrame.from_dict(list(cutflow.values()))
    cutflow.index = list(events_dict.keys())

    if prev_cutflow is not None:
        cutflow = pd.concat((prev_cutflow, cutflow), axis=1)

    return selection, cutflow


def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


# from https://gist.github.com/kdlong/d697ee691c696724fc656186c25f8814
# temp function until something is merged into hist https://github.com/scikit-hep/hist/issues/345
def rebin_hist(h, axis_name, edges):
    if isinstance(edges, int):
        return h[{axis_name: hist.rebin(edges)}]

    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all(np.isclose(x, ax.edges).any() for x in edges):
        raise ValueError(
            f"Cannot rebin histogram due to incompatible edges for axis '{ax.name}'\n"
            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}"
        )

    # If you rebin to a subset of initial range, keep the overflow and underflow
    overflow = ax.traits.overflow or (
        edges[-1] < ax.edges[-1] and not np.isclose(edges[-1], ax.edges[-1])
    )
    underflow = ax.traits.underflow or (
        edges[0] > ax.edges[0] and not np.isclose(edges[0], ax.edges[0])
    )
    flow = overflow or underflow
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax

    hnew = hist.Hist(*axes, name=h.name, storage=h._storage_type())

    # Offset from bin edge to avoid numeric issues
    offset = 0.5 * np.min(ax.edges[1:] - ax.edges[:-1])
    edges_eval = edges + offset
    edge_idx = ax.index(edges_eval)
    # Avoid going outside the range, reduceat will add the last index anyway
    if edge_idx[-1] == ax.size + ax.traits.overflow:
        edge_idx = edge_idx[:-1]

    if underflow:
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. Add tolerance of 1/2 min bin width to avoid numeric issues
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx, axis=ax_idx).take(
        indices=range(new_ax.size + underflow + overflow), axis=ax_idx
    )
    if hnew._storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.add.reduceat(
            h.variances(flow=flow), edge_idx, axis=ax_idx
        ).take(indices=range(new_ax.size + underflow + overflow), axis=ax_idx)

    return hnew


def remove_hist_overflow(h: Hist):
    hnew = Hist(*h.axes, name=h.name, storage=h._storage_type())
    hnew.values()[...] = h.values()
    return hnew


def multi_rebin_hist(h: Hist, axes_edges: dict[str, list[float]], flow: bool = True) -> Hist:
    """Wrapper around rebin_hist to rebin multiple axes at a time.

    Args:
        h (Hist): Hist to rebin
        axes_edges (dict[str, list[float]]): dictionary of {axis: edges}
    """
    for axis_name, edges in axes_edges.items():
        h = rebin_hist(h, axis_name, edges)

    if not flow:
        h = remove_hist_overflow(h)

    return h


def discretize_var(var_array, bins=None):

    if bins is None:
        bins = [0, 0.8, 0.9, 0.94, 0.97, 0.99, 1]

    # discretize the variable into len(bins)-1  integer categories
    bin_indices = np.digitize(var_array, bins)

    # clip just to be safe
    bin_indices = np.clip(bin_indices, 1, len(bins) - 1)

    return bin_indices
