"""
   BDT: A helper class for loading a trained BDT and running inference

   Usage:
       from HH4b.boosted.BDT import BDT

       # Initialize the BDT
       bdt = BDT(
           model_name="name_of_trained_model",
           model_tag="my_trained_model_dir",
           config_name="my_bdt_config",
           base_model_dir="../some_base_directory",
           jlabel="some_label"
       )

       # Use the class method to get inference
       events_dict = bdt.infer(events_dict)

   Author: Daniel Primosch

   TODO: integrate logging
"""

from __future__ import annotations

import importlib

import numpy as np
import xgboost as xgb

from HH4b.utils import get_var_mapping


class BDT:

    def __init__(
        self,
        model_name: str,
        model_tag: str,
        config_name: str,
        mass_str: str = "bbFatJetParTmassVis",
        base_model_dir: str = "../boosted/bdt_trainings_run3",
        jlabel: str = "",
    ):
        """
        Initialize the BDT class by loading the model and config.

        Args:
            model_name (str): The descriptive name of the trained BDT model
            model_tag (str): The folder name of the trained BDT model
                              (e.g., "24May31_lr_0p02_md_8_AK4Away").
            config_name (str): The Python module to load (e.g., "v5_glopartv2")
                               which contains the `bdt_dataframe` function.
            base_model_dir (str): Directory where models are stored.
            jlabel (str): A label passed to `get_var_mapping` to build the BDT input.
        """

        self.model_name = model_name
        self.model_tag = model_tag
        self.config_name = config_name
        self.base_model_dir = base_model_dir
        self.mass_str = mass_str
        self.jlabel = jlabel

        # Load the trained XGBoost model.
        self.bdt_model = xgb.XGBClassifier()
        model_path = f"{self.base_model_dir}/{self.model_tag}/trained_bdt.model"
        self.bdt_model.load_model(fname=model_path)

        # Dynamically import the config module that has the function to build the BDT dataframe.
        self.make_bdt_dataframe = importlib.import_module(
            f".{self.config_name}", package="HH4b.boosted.bdt_trainings_run3"
        )

    def infer(self, events_dict):
        """
        Run inference on the provided events_dict.

        Args:
            events_dict (dict): Dictionary of events (DataFrame-like) on which to run inference.

        Returns:
            events_dict (dict): The same dictionary with the BDT scores added.
        """
        # ----------------------------------------------
        # Check events_dict input
        # ----------------------------------------------
        if not isinstance(events_dict, dict):
            raise ValueError(f"Expected events_dict to be a dictionary, got {type(events_dict)}")

        if not events_dict:
            # e.g., if events_dict is empty or None
            raise ValueError("events_dict is empty. Cannot run inference on empty data.")

        # TODO: remove once done in pre-processing!
        if self.mass_str == "bbFatJetParTmassVis":
            events_dict = self.correct_mass(events_dict, self.mass_str)

        # Build the dataframes using the loaded configs bdt_dataframe function.
        # bdt_dfs = {}
        preds = {}
        for key in events_dict:
            bdt_df = self.make_bdt_dataframe.bdt_dataframe(
                events_dict[key], get_var_mapping(self.jlabel)
            )

            # run inference on the BDT model with given events
            preds = self.bdt_model.predict_proba(bdt_df)

            bdt_score = None
            bdt_score_vbf = None

            if preds.shape[1] == 2:
                # Binary classification: signal vs. background
                bdt_score = preds[:, 1]
            elif preds.shape[1] == 3:
                # 3-class: e.g. [ggF HH, QCD, ttbar]
                bdt_score = preds[:, 0]
            elif preds.shape[1] == 4:
                # 4-class: e.g. [ggF HH, VBF HH, QCD, ttbar]
                bg_tot = np.sum(preds[:, 2:], axis=1)
                bdt_score = preds[:, 0] / (preds[:, 0] + bg_tot)
                bdt_score_vbf = preds[:, 1] / (preds[:, 1] + preds[:, 2] + preds[:, 3])

            # return bdt_score, bdt_score_vbf
            events_dict[key][f"bdtscore_{self.model_name}"] = (
                bdt_score if bdt_score is not None else np.ones(events_dict[key]["weight"])
            )
            events_dict[key][f"bdtscoreVBF_{self.model_name}"] = (
                bdt_score_vbf if bdt_score_vbf is not None else np.ones(events_dict[key]["weight"])
            )
            # bdt_scores.extend([f"bdtscore_{self.config_name}", f"bdtscoreVBF_{self.config_name}"])
        return events_dict

    # remove once done in pre-processing!
    def correct_mass(self, events_dict, mass_str):
        for key in events_dict:
            events_dict[key][(mass_str, 0)] = events_dict[key][(mass_str, 0)] * (
                1 - events_dict[key][("bbFatJetrawFactor", 0)]
            )
            events_dict[key][(mass_str, 1)] = events_dict[key][(mass_str, 1)] * (
                1 - events_dict[key][("bbFatJetrawFactor", 1)]
            )

        return events_dict
