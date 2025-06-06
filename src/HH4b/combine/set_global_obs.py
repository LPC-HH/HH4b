#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_json_file(file_path):
    """
    Loads JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the data from the JSON file, or None if an error occurs.
    """
    try:
        with Path(file_path).open("r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# Check if the correct number of arguments is provided
if len(sys.argv) != 3:
    print("Usage: set_global_obs.py inject_original.json inject.json")
    sys.exit(1)


original_file_path = sys.argv[1]
file_path = sys.argv[2]

print(f"Loading {original_file_path}")
loaded_data = load_json_file(original_file_path)

for param in loaded_data:
    if "_In" in param:
        global_obs = param
        nuisance = param.split("_In")[0]
        val = loaded_data[nuisance]["value"]
        print(f"Setting global observable {global_obs}={val}")
        loaded_data[global_obs]["value"] = val

print(f"Creating {file_path}")
with Path(file_path).open("w") as json_file:
    json.dump(loaded_data, json_file, indent=4)
