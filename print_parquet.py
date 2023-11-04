import pandas as pd
import click
import numpy as np

np.set_printoptions(threshold=np.inf)


@click.command()
@click.argument("filename")
@click.option("-b", "--branches", multiple=True, default=[])
def print_parquet(filename, branches):
    events = pd.read_parquet(filename)
    print(np.array(events.columns))
    if len(branches) > 0:
        for b in branches:
            if b in events.columns:
                print(b, np.array(events[b]))
            else:
                print(f"Branch {b} not found")


if __name__ == "__main__":
    print_parquet()
