"""Model-independent per-year slim cache for the FOM-scan pipeline.

The FOM scan over many years OOMs because PostProcess holds every year's full
feature frame in memory at once.  This cache breaks that: each year is loaded +
built ONCE (the expensive step), slimmed, and written to disk *without* the
model-specific BDT score (only the input features + analysis columns are kept).

A FOM run for any model then reads the slim per-year cache (cheap, one year of
memory at a time) and re-runs only that model's inference to add the score.
Adding a new year (e.g. 2025) only builds that year's cache; everything else is
reused.  The cache is keyed by (data tag, txbb version) and is model-independent,
so all candidate BDTs share it -> "slim once".
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pandas as pd

# columns that depend on the BDT model -> never cached (recomputed per model)
_SCORE_PREFIXES = ("bdt_score",)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def _root(cache_dir: str, tag: str, txbb: str) -> Path:
    return Path(cache_dir) / _safe(f"{tag}__{txbb}")


def year_dir(cache_dir: str, tag: str, txbb: str, year: str) -> Path:
    return _root(cache_dir, tag, txbb) / str(year)


def exists(cache_dir: str, tag: str, txbb: str, year: str) -> bool:
    """A year is cached iff it holds at least one sample parquet."""
    d = year_dir(cache_dir, tag, txbb, year)
    return d.is_dir() and any(d.glob("*.parquet"))


def save(
    events: dict, cutflow, cache_dir: str, tag: str, txbb: str, year: str, merge: bool = False
) -> int:
    """Write per-sample slim parquet (scores dropped) + a per-chunk cutflow.

    Concurrency-safe: each sample is its own file (disjoint across chunks), the
    cutflow is written to a chunk-unique file, and the manifest is rebuilt by
    globbing (best-effort, for humans only).  So two chunks of the SAME year can
    be built on different nodes at once without a shared-file write race.  The
    `merge` arg is accepted for back-compat and ignored.
    """
    del merge  # behaviour is always-accumulate now (disjoint per-sample files)
    d = year_dir(cache_dir, tag, txbb, year)
    d.mkdir(parents=True, exist_ok=True)
    for key, df in events.items():
        drop = [c for c in df.columns if any(str(c).startswith(p) for p in _SCORE_PREFIXES)]
        slim = df.drop(columns=drop, errors="ignore")
        slim.to_parquet(d / (_safe(key) + ".parquet"))
    if cutflow is not None and len(events):
        # chunk-unique name keyed by the first sample -> no concurrent overwrite
        with open(d / f"cutflow__{_safe(sorted(events)[0])}.pkl", "wb") as f:  # noqa: PTH123
            pickle.dump(cutflow, f)
    # best-effort human-readable manifest (glob-derived; load() does not rely on it)
    files = sorted(p.name for p in d.glob("*.parquet"))
    (d / "manifest.json").write_text(json.dumps({fn: fn[:-8] for fn in files}, indent=0))
    return len(files)


def load(cache_dir: str, tag: str, txbb: str, year: str):
    """Return (events_dict, cutflow) from the slim cache (no scores yet).

    Sample list comes from globbing the parquet files (robust to a stale/partial
    manifest); cutflows from all per-chunk cutflow files are merged.
    """
    d = year_dir(cache_dir, tag, txbb, year)
    events = {p.stem: pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))}
    cf_parts = []
    for c in sorted(d.glob("cutflow*.pkl")):
        try:
            with open(c, "rb") as f:  # noqa: PTH123
                cf_parts.append(pickle.load(f))
        except Exception:
            pass
    cutflow = None
    if cf_parts:
        try:
            cutflow = pd.concat(cf_parts)
            cutflow = cutflow[~cutflow.index.duplicated(keep="last")]
        except Exception:
            cutflow = cf_parts[-1]
    return events, cutflow
