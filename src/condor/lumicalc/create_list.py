from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"


def load_yaml_config(yaml_path):
    """Load YAML configuration file."""
    with Path(yaml_path).open("r") as f:
        return yaml.safe_load(f)


def load_nanoindex_data(nano_version):
    """Load the nanoindex JSON file based on nano_version."""
    json_path = DATA_DIR / f"nanoindex_{nano_version}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Nanoindex file not found: {json_path}")

    with json_path.open("r") as f:
        return json.load(f)


def extract_file_paths(nanoindex_data, year, samples):
    """Extract file paths for given year and samples."""
    file_paths = []

    if year not in nanoindex_data:
        print(f"Warning: Year '{year}' not found in nanoindex data")
        return file_paths

    year_data = nanoindex_data[year]

    for sample in samples:
        if sample in year_data:
            sample_data = year_data[sample]
            # Iterate through all datasets in this sample
            for dataset_name, file_list in sample_data.items():
                if isinstance(file_list, list):
                    file_paths.extend(file_list)
                else:
                    print(
                        f"Warning: Expected list for {sample}/{dataset_name}, got {type(file_list)}"
                    )
        else:
            print(f"Warning: Sample '{sample}' not found in year '{year}' data")

    return file_paths


def write_list_file(output_dir, year, file_paths):
    """Write file paths to a .list file."""
    output_path = Path(output_dir) / f"{year}.list"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for file_path in file_paths:
            f.write(f"{file_path}\n")

    print(f"Written {len(file_paths)} file paths to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate list files from YAML configuration")
    parser.add_argument("-c", "--config", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load YAML configuration
    config = load_yaml_config(args.config)

    # Extract configuration parameters
    nano_version = config["nano_version"]
    output_dir = config["output_dir"]

    # Load nanoindex data
    try:
        nanoindex_data = load_nanoindex_data(nano_version)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Process each year from the data section
    data_config = config.get("data", {})
    for year, samples in data_config.items():

        print(f"Processing year: {year} with samples: {samples}")

        # Extract file paths for this year and samples
        file_paths = extract_file_paths(nanoindex_data, year, samples)

        if file_paths:
            # Write to list file
            write_list_file(output_dir, year, file_paths)
        else:
            print(f"No file paths found for year '{year}' with samples {samples}")


if __name__ == "__main__":
    main()
