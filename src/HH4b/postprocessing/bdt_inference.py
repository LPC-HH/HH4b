"""
BDT Inference Script

Runs BDT inference on HH4b samples, processing each sample in a subprocess
to avoid memory leaks. Results are cached to disk and aggregated by year.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import logging.config
import multiprocessing as mp
import pickle
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b import hh_vars
from HH4b.hh_vars import mreg_strings, samples_run3
from HH4b.log_utils import log_config
from HH4b.postprocessing import load_run3_samples
from HH4b.utils import get_var_mapping

# Setup logging
log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)

HH4B_DIR = Path(__file__).resolve().parents[3]


def load_model(model_name: str) -> xgb.XGBClassifier:
    """Load trained BDT model."""
    path = HH4B_DIR / f"src/HH4b/boosted/bdt_trainings_run3/{model_name}/trained_bdt.model"
    model = xgb.XGBClassifier()
    model.load_model(fname=str(path))
    return model


def compute_scores(
    events: pd.DataFrame,
    preds: np.ndarray,
    jshift: str = "",
    weight_ttbar: float = 1.0,
    use_disc: bool = True,
):
    """
    Add BDT scores to events DataFrame.

    Handles different BDT configurations:
    - 2 classes: binary BDT
    - 3 classes: ggF HH, QCD, ttbar
    - 4 classes: ggF HH, VBF HH, QCD, ttbar
    - 5 classes: ggF HH, VBF HH(K2V=0), VBF HH(K2V=1), QCD, ttbar
    """
    suffix = "" if jshift == "" else f"_{jshift}"
    n_classes = preds.shape[1]

    if n_classes == 2:
        # Binary BDT
        events[f"bdt_score{suffix}"] = preds[:, 1]
    elif n_classes == 3:
        # Multi-class: ggF HH, QCD, ttbar
        events[f"bdt_score{suffix}"] = preds[:, 0]
    elif n_classes == 4:
        # Multi-class: ggF HH, VBF HH, QCD, ttbar
        bg = preds[:, 2] + preds[:, 3]
        events[f"bdt_score{suffix}"] = preds[:, 0] / (preds[:, 0] + bg) if use_disc else preds[:, 0]
        events[f"bdt_score_vbf{suffix}"] = (
            preds[:, 1] / (preds[:, 1] + preds[:, 2] + weight_ttbar * preds[:, 3])
            if use_disc
            else preds[:, 1]
        )
    elif n_classes == 5:
        # Multi-class: ggF HH, VBF HH(K2V=0), VBF HH(K2V=1), QCD, ttbar
        bg = preds[:, 3] + preds[:, 4]
        events[f"bdt_score{suffix}"] = preds[:, 0] / (preds[:, 0] + bg) if use_disc else preds[:, 0]
        vbf_sig = preds[:, 1] + preds[:, 2]
        events[f"bdt_score_vbf{suffix}"] = (
            vbf_sig / (vbf_sig + preds[:, 3] + weight_ttbar * preds[:, 4]) if use_disc else vbf_sig
        )


def run_inference(model, events, bdt_dataframe_fn, jshifts, args):
    """Run BDT inference for all JEC/JMR shifts and combine results."""
    results = []
    for jshift in jshifts:
        df = bdt_dataframe_fn(events, get_var_mapping(jshift))
        preds = model.predict_proba(df)
        compute_scores(df, preds, jshift, args.weight_ttbar_bdt, args.bdt_disc)
        results.append(df)
        del preds

    combined = pd.concat(results, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()].copy()


def process_sample(year: str, sample: str, args_dict: dict, cache_dir: str) -> bool:
    """
    Process a single sample (runs in subprocess).

    Loads data, runs inference, saves to cache. Each sample runs in its own
    subprocess to ensure complete memory cleanup between samples.
    """

    # Re-setup logging in subprocess
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(__name__)

    args = argparse.Namespace(**args_dict)
    cache_dir = Path(cache_dir)
    sample_dir = cache_dir / year / sample
    sample_dir.mkdir(parents=True, exist_ok=True)

    if year == "2025" and sample != "data":
        logger.warning("Only data sample is available for 2025; Using 2024 MC.")
        load_year = "2024"
    else:
        load_year = year

    try:
        logger.info(f"Processing {sample}")

        # Load model and BDT dataframe function
        model = load_model(args.bdt_model)
        bdt_dataframe_fn = importlib.import_module(
            f".{args.bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
        ).bdt_dataframe

        # Load sample data
        events = load_run3_samples(
            f"{args.data_dir}/{args.tag}",
            load_year,
            {load_year: {sample: samples_run3[load_year][sample]}},
            reorder_txbb=True,
            load_systematics=args.load_systematics,
            txbb_version=args.txbb,
            scale_and_smear=False,
            mass_str=mreg_strings[args.txbb],
            bdt_version=args.bdt_model,
        )[sample]

        n_events = len(events)
        logger.info(f"Loaded {n_events} events")
        logger.info(f"Columns: {events.columns.tolist()}")

        # Determine JEC/JMR shifts to process
        jshifts = [""]
        if args.include_systematics:
            if sample in hh_vars.syst_keys:
                jshifts += hh_vars.jec_shifts
            if sample in hh_vars.jmsr_keys:
                jshifts += hh_vars.jmsr_shifts

        # Process with optional chunking
        if args.chunk_size > 0 and n_events > args.chunk_size:
            n_chunks = (n_events + args.chunk_size - 1) // args.chunk_size
            logger.info(f"Processing {n_chunks} chunks of size {args.chunk_size}")

            for i in range(n_chunks):
                start = i * args.chunk_size
                end = min((i + 1) * args.chunk_size, n_events)
                chunk = events.iloc[start:end].copy()

                result = run_inference(model, chunk, bdt_dataframe_fn, jshifts, args)
                result["year"] = year
                result["finalWeight"] = chunk["finalWeight"].to_numpy()
                result.to_parquet(sample_dir / f"chunk_{i:04d}.parquet")

                del chunk, result
                gc.collect()
        else:
            # Process entire sample at once
            result = run_inference(model, events, bdt_dataframe_fn, jshifts, args)
            result["year"] = year
            result["finalWeight"] = events["finalWeight"].to_numpy()
            result.to_parquet(sample_dir / "data.parquet")

        logger.info(f"Saved {sample}")
        return True

    except Exception as e:
        logger.error(f"Error processing {sample}: {e}")

        traceback.print_exc()
        return False


def run_in_subprocess(year: str, sample: str, args: argparse.Namespace, cache_dir: Path) -> bool:
    """
    Run sample processing in isolated subprocess.

    Uses 'spawn' context to create a fresh Python interpreter,
    ensuring complete memory cleanup when subprocess exits.
    """
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=process_sample, args=(year, sample, vars(args), str(cache_dir)))
    p.start()
    p.join()
    return p.exitcode == 0


def aggregate_year(cache_dir: Path, year: str, output_dir: Path, fmt: str):
    """Aggregate all cached samples for a year into final output."""
    year_dir = cache_dir / year
    if not year_dir.exists():
        return

    sample_dirs = [d for d in year_dir.iterdir() if d.is_dir()]
    logger.info(f"Aggregating {len(sample_dirs)} samples for {year}")

    results = {}
    for sample_dir in sample_dirs:
        sample = sample_dir.name

        # Check for chunks or single data file
        chunks = sorted(sample_dir.glob("chunk_*.parquet"))
        data_file = sample_dir / "data.parquet"

        if chunks:
            df = pd.concat([pd.read_parquet(f) for f in chunks], ignore_index=True)
        elif data_file.exists():
            df = pd.read_parquet(data_file)
        else:
            continue

        results[sample] = df
        logger.info(f"  {sample}: {len(df)} events")

    # Save aggregated results
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "pkl":
        save_path = output_dir / f"{year}_bdt_scores.pkl"
        with save_path.open("wb") as f:
            pickle.dump(results, f)
    else:  # parquet
        year_out = output_dir / year
        year_out.mkdir(exist_ok=True)
        for sample, df in results.items():
            df.to_parquet(year_out / f"{sample}_bdt_scores.parquet")

    logger.info(f"Saved {year} results")


def main():
    parser = argparse.ArgumentParser(description="Run BDT inference on HH4b samples")

    # Model configuration
    parser.add_argument(
        "--bdt-model",
        default="25Feb5_v13_glopartv2_rawmass",
        help="Name of the BDT model directory",
    )
    parser.add_argument("--bdt-config", default="v13_glopartv3", help="BDT config module name")

    # Data configuration
    parser.add_argument(
        "--data-dir",
        default="/ceph/cms/store/user/zichun/bbbb/skimmer/",
        help="Base directory for input ntuples",
    )
    parser.add_argument(
        "--tag", default="nanov15_20251202_v15_signal", help="Tag for input ntuples"
    )
    parser.add_argument(
        "--years", nargs="+", default=hh_vars.years, choices=hh_vars.years, help="Years to process"
    )
    parser.add_argument(
        "--samples", nargs="+", default=None, help="Specific samples to process (default: all)"
    )
    parser.add_argument(
        "--txbb",
        default="glopart-v3",
        choices=["pnet-legacy", "pnet-v12", "glopart-v2", "glopart-v3"],
        help="Version of TXbb tagger to use",
    )

    # BDT configuration
    parser.add_argument(
        "--weight-ttbar-bdt",
        type=float,
        default=1.0,
        help="Weight for ttbar class in VBF BDT discriminant",
    )
    parser.add_argument(
        "--bdt-disc",
        action="store_true",
        default=True,
        help="Use BDT discriminant P_sig/(P_sig+P_bkg)",
    )

    # Processing configuration
    parser.add_argument(
        "--chunk-size", type=int, default=-1, help="Events per chunk (-1 = no chunking)"
    )
    parser.add_argument(
        "--include-systematics",
        action="store_true",
        default=False,
        help="Include JEC/JMR systematic variations",
    )
    parser.add_argument(
        "--load-systematics",
        action="store_true",
        default=False,
        help="Load systematic variations from input files",
    )
    parser.add_argument(
        "--abort-on-fail",
        action="store_true",
        default=False,
        help="Abort entire job if any sample fails",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="/home/users/zichun/ceph/bbbb/bdt_inference/",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--output-format", default="pkl", choices=["pkl", "parquet"], help="Output file format"
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        default=False,
        help="Keep intermediate cache files after aggregation",
    )

    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir) / args.tag
    cache_dir = output_dir / ".cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BDT Inference")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info("=" * 60)

    # Step 1: Process each sample in subprocess
    for year in args.years:
        logger.info(f"\n{'='*40}\nYear: {year}\n{'='*40}")

        all_samples = list(samples_run3[year].keys())
        samples = [s for s in args.samples if s in all_samples] if args.samples else all_samples

        for sample in samples:
            success = run_in_subprocess(year, sample, args, cache_dir)
            status = "✓" if success else "✗"
            if success:
                logger.info(f"{status} {sample}")
            else:
                logger.error(f"{status} {sample}")
            if not success and args.abort_on_fail:
                raise RuntimeError(
                    f"Failed to process sample {sample} for year {year}. Job aborted."
                )

    # Step 2: Aggregate cached results by year
    logger.info(f"\n{'='*40}\nAggregating results\n{'='*40}")
    for year in args.years:
        aggregate_year(cache_dir, year, output_dir, args.output_format)

    # Cleanup cache
    if not args.keep_cache:
        shutil.rmtree(cache_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
