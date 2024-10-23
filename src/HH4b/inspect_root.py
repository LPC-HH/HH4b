from __future__ import annotations

import argparse

import uproot


def inspect_root_file(file_path):
    """Inspect the contents of a ROOT file."""
    try:
        # Open the ROOT file using uproot
        with uproot.open(file_path) as file:
            print(f"\nInspecting ROOT file: {file_path}\n")

            # List all available keys (objects) in the file
            print("Available keys in the ROOT file:")
            for key in file:
                print(f"  - {key}")

            # Check for any trees and print their branches
            print("\nAvailable Trees and their Branches:")
            for name, tree in file.items():
                if isinstance(tree, uproot.behaviors.TTree.TTree):
                    print(f"\nTree: {name}")
                    print(f"{'-' * 30}")
                    for branch_name in tree:
                        print(f"  Branch: {branch_name}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Inspect the contents of a ROOT file.")
    parser.add_argument("file", type=str, help="Path to the ROOT file to inspect")
    args = parser.parse_args()

    # Inspect the given ROOT file
    inspect_root_file(args.file)


if __name__ == "__main__":
    main()
