"""
General utilities for postprocessing.

Author: Raghav Kansal
"""

from __future__ import annotations

import contextlib
import logging
import logging.config
import pickle
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import hist
import numpy as np
import pandas as pd
import vector
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

    full_samples_list = [str(p.name) for p in Path(f"{samples_dir}/{year}").iterdir()]
    print("Checking for empty parquets")

    for sample in full_samples_list:
        if sample == ".DS_Store":
            continue
        parquet_files = [
            str(p.name) for p in Path(f"{samples_dir}/{year}/{sample}/parquet").iterdir()
        ]
        for f in parquet_files:
            file_path = f"{samples_dir}/{year}/{sample}/parquet/{f}"
            if not len(pd.read_parquet(file_path)):
                print("Removing: ", f"{sample}/{f}")
                Path(file_path).unlink()


def get_cutflow(pickles_path, year, sample_name):
    """Accumulates cutflow over all pickles in ``pickles_path`` directory"""
    from coffea.processor.accumulator import accumulate

    out_pickles = [str(p.name) for p in Path(pickles_path).iterdir()]

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
        out_pickles = [str(p.name) for p in Path(pickles_path).iterdir()]
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
    from coffea.processor.accumulator import accumulate

    out_pickles = [str(p.name) for p in Path.iterdir(pickles_path) if str(p.name) != ".DS_Store"]

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

    for s in selector:
        if s.endswith("?"):
            if s[:-1] == sample:
                return True
        elif s.startswith("*"):
            if s[1:] in sample:
                return True
        else:
            if sample.startswith(s):
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

        if "VBF" in sample:
            warnings.warn(
                f"Temporarily scaling {sample} by its xsec and lumi - remember to remove after fixing in the processor!",
                stacklevel=0,
            )
            events["weight"] = events["weight"].to_numpy() * xsecs[sample] * LUMI[year]
        else:
            raise ValueError(f"{sample} has not been scaled by its xsec and lumi!")

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

    Returns:
        Dict[str, pd.DataFrame]: ``events_dict`` dictionary of events dataframe for each sample.

    """
    events_dict = {}

    data_dir = Path(data_dir) / year
    full_samples_list = [
        str(p.name) for p in Path.iterdir(data_dir)
    ]  # get all directories in data_dir

    logger.debug(f"Full list of directories in {data_dir}: {full_samples_list}")
    logger.debug(f"Samples to load {samples}")

    # label - key of sample in events_dict
    # selector - string used to select directories to load in for this sample
    for label, selector in samples.items():
        # important to check that samples have been normalized properly
        load_columns = columns
        if label != "data" and load_weight_noxsec:
            load_columns = columns + format_columns([("weight_noxsec", 1)])

        events_dict[label] = []  # list of directories we load in for this sample
        for sample in full_samples_list:
            # check if this directory passes our selector string
            if not check_selector(sample, selector):
                continue

            sample_path = data_dir / sample
            parquet_path, pickles_path = sample_path / "parquet", sample_path / "pickles"

            # no parquet directory?
            if not parquet_path.exists():
                warnings.warn(f"No parquet directory for {sample}!", stacklevel=1)
                continue

            logger.debug(f"Loading {sample}")
            try:
                non_empty_passed_list = []
                for parquet_file in parquet_path.glob("*.parquet"):
                    if not pd.read_parquet(parquet_file).empty:
                        df_sample = pd.read_parquet(
                            parquet_file, filters=filters, columns=load_columns
                        )
                        non_empty_passed_list.append(df_sample)
                events = pd.concat(non_empty_passed_list)
            except Exception:
                warnings.warn(
                    f"Can't read file with requested columns/filters for {sample}!", stacklevel=1
                )
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
            else:
                if label == data_key:
                    events["finalWeight"] = events["weight"]
                else:
                    n_events = get_nevents(pickles_path, year, sample)
                    events["weight_nonorm"] = events["weight"]
                    events["finalWeight"] = events["weight"] / n_events

            events_dict[label].append(events)
            logger.info(f"Loaded {sample: <50}: {len(events)} entries")

        if len(events_dict[label]):
            events_dict[label] = pd.concat(events_dict[label])
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
    import vector

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
    from coffea.analysis_tools import PackedSelection

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
