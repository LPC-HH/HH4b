#!/usr/bin/env python3
import json
import sys

def load_json_file(file_path):
    """
    Loads JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the data from the JSON file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
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
    print("Usage: python set_global_obs.py inject_original.json inject.json")
    sys.exit(1)


original_file_path = sys.argv[1]
file_path = sys.argv[2]
loaded_data = load_json_file(file_path)

for param in loaded_data:
    if "_In" in param:
        global_obs = param 
        nuisance = param.split("_In")[0]
        loaded_data[global_obs]["value"] = loaded_data[nuisance]["value"]

with open(file_path, 'w') as json_file:
    json.dump(loaded_data, json_file, indent=4)
