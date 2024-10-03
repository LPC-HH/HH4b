from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option(
    "--base_dir",
    "-b",
    type=click.Path(exists=True),
    prompt="Enter the base directory",
    help="Path to the base directory containing the output directories.",
)
@click.option(
    "--output_file",
    "-o",
    type=click.Path(),
    prompt="Enter the output file path",
    help="Path to save the DataFrame containing the results in JSON format.",
)
def process_directory(base_dir, output_file):
    """Process all subdirectories in the base directory to collect BDT training results."""

    def parse_evals_file(filepath):
        """Parse the evals_result.txt file to extract the best validation_1 mlogloss."""
        with Path(filepath).open() as file:
            content = file.read()
            evals_result = eval(content, {"OrderedDict": OrderedDict})  # Pass OrderedDict to eval
            val_1_mlogloss = evals_result["validation_1"]["mlogloss"]
            best_val_1_mlogloss = min(val_1_mlogloss)
        return best_val_1_mlogloss

    def extract_params_from_dirname(dirname):
        """Extract learning rate and max_depth from the directory name."""
        lr_match = re.search(r"lr_(\d+\.\d+)", dirname)
        depth_match = re.search(r"max_depth_(\d+)", dirname)
        lr = float(lr_match.group(1)) if lr_match else None
        max_depth = int(depth_match.group(1)) if depth_match else None
        return lr, max_depth

    results = []
    base_path = Path(base_dir)
    for dirname in base_path.iterdir():
        if dirname.is_dir():
            evals_file = dirname / "evals_result.txt"
            if evals_file.exists():
                best_val_1 = parse_evals_file(evals_file)
                lr, max_depth = extract_params_from_dirname(dirname.name)
                results.append([lr, max_depth, best_val_1])

    # Create a DataFrame
    results_df = pd.DataFrame(
        results, columns=["Learning Rate", "Max Depth", "Best Validation_1 Mlogloss"]
    )
    print(results_df)

    # Print the row with the lowest Best Validation_1 Mlogloss
    if not results_df.empty:
        best_row = results_df.loc[results_df["Best Validation_1 Mlogloss"].idxmin()]
        print("\nRow with the lowest Best Validation_1 Mlogloss:")
        print(best_row)

    # Save the DataFrame to the specified output file in JSON format
    results_df.to_json(output_file, orient="records", lines=True)
    print(f"DataFrame saved to {output_file} in JSON format")


if __name__ == "__main__":
    process_directory()
