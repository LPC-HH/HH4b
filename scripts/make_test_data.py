"""Create a small fixture dataset for integration testing.

Reads N rows from each sample's parquet files in an existing skimmer output
directory and writes them to a self-contained fixture directory that can be
committed to the repository (or stored on EOS/ceph alongside the code).

The resulting directory has the same structure as a real skimmer output:

    <out-dir>/<tag>/<year>/<sample>/parquet/out_0.parquet
    <out-dir>/<tag>/<year>/<sample>/pickles/out_0.pkl

Usage
-----
    python tests/make_test_data.py \\
        --data-dir /ceph/cms/store/user/<user>/bbbb/skimmer \\
        --tag 25May9_v12v2_private_signal \\
        --out-dir tests/fixtures/skimmer \\
        --years 2022 \\
        --samples hh4b qcd data \\
        --n-events 200

Running this script once is sufficient; the fixture is then committed and reused
by test_postprocess_integration.py without touching the full dataset.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from HH4b import hh_vars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Samples that are loaded during a standard postprocess_run3 run.
# Override with --samples if you only want a subset.
DEFAULT_SAMPLES = ["hh4b", "vbfhh4b", "qcd", "ttbar", "data"]


def _find_sample_dir(year_dir: Path, selector: str | list) -> list[Path]:
    """Return all subdirs of year_dir whose name matches the selector."""
    from HH4b.utils import check_selector

    return [d for d in year_dir.iterdir() if d.is_dir() and check_selector(d.name, selector)]


def _sample_parquet(src_parquet_dir: Path, dst_parquet_dir: Path, n: int) -> int:
    """Copy at most n rows from all parquets in src into a single out_0.parquet in dst."""
    dst_parquet_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total = 0
    for pq_file in sorted(src_parquet_dir.glob("*.parquet")):
        if total >= n:
            break
        df = pd.read_parquet(pq_file)
        need = n - total
        frames.append(df.iloc[:need])
        total += min(len(df), need)

    if not frames:
        return 0

    combined = pd.concat(frames, ignore_index=True)
    # Reset row index before saving but do NOT pass index=False — that flag also
    # strips the column MultiIndex metadata, breaking the parquet schema that the
    # postprocessing loaders expect.
    combined.reset_index(drop=True).to_parquet(dst_parquet_dir / "out_0.parquet")
    return len(combined)


def _copy_pickles(src_pickles_dir: Path, dst_pickles_dir: Path) -> None:
    """Copy the first pickle file (which has the totals/cutflow) to the fixture."""
    dst_pickles_dir.mkdir(parents=True, exist_ok=True)
    pkl_files = sorted(src_pickles_dir.glob("*.pkl"))
    if not pkl_files:
        logger.warning("No pkl files in %s", src_pickles_dir)
        return
    # All pkl files for a given sample contain the same totals dict; one is enough.
    shutil.copy(pkl_files[0], dst_pickles_dir / "out_0.pkl")


def build_fixture(
    data_dir: Path,
    tag: str,
    out_dir: Path,
    years: list[str],
    sample_keys: list[str],
    n_events: int,
) -> None:
    src_tag_dir = data_dir / tag
    dst_tag_dir = out_dir / tag

    for year in years:
        src_year_dir = src_tag_dir / year
        if not src_year_dir.exists():
            logger.warning("Year directory not found, skipping: %s", src_year_dir)
            continue

        for sample_key in sample_keys:
            # Resolve selector from hh_vars
            samples_year = hh_vars.samples_run3.get(year, {})
            if sample_key not in samples_year:
                logger.warning("Sample key '%s' not in hh_vars for %s, skipping", sample_key, year)
                continue
            selector = samples_year[sample_key]
            matched_dirs = _find_sample_dir(src_year_dir, selector)

            if not matched_dirs:
                logger.warning(
                    "No directories matched selector '%s' for %s/%s", selector, year, sample_key
                )
                continue

            # Use only the first matched directory to keep the fixture small.
            src_sample_dir = matched_dirs[0]
            dst_sample_dir = dst_tag_dir / year / src_sample_dir.name

            src_parquet = src_sample_dir / "parquet"
            src_pickles = src_sample_dir / "pickles"

            if not src_parquet.exists():
                logger.warning("No parquet dir for %s, skipping", src_sample_dir)
                continue

            n_written = _sample_parquet(src_parquet, dst_sample_dir / "parquet", n_events)
            logger.info(
                "  %s / %s / %s → %d rows", year, sample_key, src_sample_dir.name, n_written
            )

            if src_pickles.exists():
                _copy_pickles(src_pickles, dst_sample_dir / "pickles")
            else:
                logger.warning("No pickles dir for %s", src_sample_dir)

    logger.info("Fixture written to %s", dst_tag_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Root skimmer output directory (e.g. /ceph/cms/store/user/<user>/bbbb/skimmer/)",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Skimmer tag subdirectory (e.g. 25May9_v12v2_private_signal)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/fixtures/skimmer"),
        help="Destination directory for the fixture (default: tests/fixtures/skimmer)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022"],
        choices=list(hh_vars.samples_run3.keys()),
        help="Years to include in the fixture (default: 2022)",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=DEFAULT_SAMPLES,
        help=f"Sample keys to include (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=200,
        help="Maximum rows to copy per sample-directory (default: 200)",
    )
    args = parser.parse_args()

    build_fixture(
        data_dir=args.data_dir,
        tag=args.tag,
        out_dir=args.out_dir,
        years=args.years,
        sample_keys=args.samples,
        n_events=args.n_events,
    )


if __name__ == "__main__":
    main()
