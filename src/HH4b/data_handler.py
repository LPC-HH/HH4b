from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

from HH4b.utils import format_columns, load_samples

logger = logging.getLogger(__name__)


class DataConfig:
    """
    Holds configuration for data loading, such as:
    - data_path: where to find the data
    - columns: which columns to load
    - filters: which event-level filters to apply
    - samples: dictionary of processes and their corresponding datasets
    """

    def __init__(
        self, data_path: str, columns: list[tuple[str, int]], filters: list, samples: dict
    ):
        self.data_path = data_path
        self.columns = columns
        self.filters = filters
        self.samples = samples  # e.g. {"qcd": ["QCD_HT-600to800", ...], "hh4b": ["GluGluToHH..."]}


class SampleSet:
    """
    Represents a named set of samples (datasets) to be loaded.
    For example:
      name = "qcd"
      dataset_list = ["QCD_HT-600to800", "QCD_HT-800to1000"]
    """

    def __init__(self, name: str, dataset_list: list[str]):
        self.name = name
        self.dataset_list = dataset_list


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Different implementations might load from parquet, ROOT, CSV, etc.
    """

    @abstractmethod
    def load_data(
        self, sample: SampleSet, year: str, reorder_txbb: bool, txbb_str: str
    ) -> dict[str, pd.DataFrame]:
        pass


class ParquetDataLoader(BaseDataLoader):
    """
    Loads data from parquet files using the HH4b utilities.
    """

    def __init__(self, data_config: DataConfig):
        self.data_config = data_config

    def load_data(
        self, sample: SampleSet, year: str, reorder_txbb: bool, txbb_str: str
    ) -> dict[str, pd.DataFrame]:
        logger.info(f"Loading data for {sample.name} in year {year}")
        events_dict = load_samples(
            path_to_dir=self.data_config.data_path,
            sample_dirs={sample.name: sample.dataset_list},
            year=year,
            filters=self.data_config.filters,
            columns=format_columns(self.data_config.columns),
            reorder_txbb=reorder_txbb,
            txbb_str=txbb_str,
            variations=False,
        )
        return events_dict
