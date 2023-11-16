"""
Common functions for processors.

Author(s): Raghav Kansal
"""
from __future__ import annotations

import os
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from coffea.analysis_tools import PackedSelection

P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}


PAD_VAL = -99999


def pad_val(
    arr: ak.Array,
    target: int,
    value: float = PAD_VAL,
    axis: int = 0,
    to_numpy: bool = True,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=axis)
    return ret.to_numpy() if to_numpy else ret


def add_selection(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
    cutflow: dict,
    isData: bool,
    genWeights: ak.Array = None,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    if isinstance(sel, ak.Array):
        sel = sel.to_numpy()

    selection.add(name, sel.astype(bool))
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up genWeights for MC
        else np.sum(genWeights[selection.all(*selection.names)])
    )


def add_selection_no_cutflow(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
):
    """adds selection to PackedSelection object"""
    selection.add(name, ak.fill_none(sel, False))


def concatenate_dicts(dicts_list: list[dict[str, np.ndarray]]):
    """given a list of dicts of numpy arrays, concatenates the numpy arrays across the lists"""
    if len(dicts_list) > 1:
        return {
            key: np.concatenate(
                [
                    dicts_list[i][key].reshape(dicts_list[i][key].shape[0], -1)
                    for i in range(len(dicts_list))
                ],
                axis=1,
            )
            for key in dicts_list[0]
        }

    return dicts_list[0]


def select_dicts(dicts_list: list[dict[str, np.ndarray]], sel: np.ndarray):
    """given a list of dicts of numpy arrays, select the entries per array across the lists according to ``sel``"""
    return {
        key: np.stack(
            [
                dicts_list[i][key].reshape(dicts_list[i][key].shape[0], -1)
                for i in range(len(dicts_list))
            ],
            axis=1,
        )[sel]
        for key in dicts_list[0]
    }


def to_pandas(events: dict[str, np.array]):
    """
    Convert our dictionary of numpy arrays into a pandas data frame
    Uses multi-index columns for numpy arrays with >1 dimension
    (e.g. FatJet arrays with two columns)
    """
    return pd.concat(
        [pd.DataFrame(v) for k, v in events.items()],
        axis=1,
        keys=list(events.keys()),
    )


def dump_table(pddf: pd.DataFrame, fname: str, odir_str: str | None = None) -> None:
    """
    Saves pandas dataframe events to './outparquet'
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    local_dir = (Path() / "outparquet").resolve()
    if odir_str:
        local_dir += odir_str
    os.system(f"mkdir -p {local_dir}")

    # need to write with pyarrow as pd.to_parquet doesn't support different types in
    # multi-index column names
    table = pa.Table.from_pandas(pddf)
    pq.write_table(table, f"{local_dir}/{fname}")
