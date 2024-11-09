from __future__ import annotations

import argparse
import importlib
import logging
import logging.config
import pickle
from collections import OrderedDict
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from HH4b import hh_vars, plotting
from HH4b.hh_vars import samples_run3
from HH4b.log_utils import log_config
from HH4b.postprocessing import (
    get_evt_testing,
    load_run3_samples,
)
from HH4b.run_utils import add_bool_arg
from HH4b.utils import ShapeVar

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("TrainBDT")

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update(
    {
        "font.size": 12,
        "lines.linewidth": 2,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.5,
        "figure.edgecolor": "none",
    }
)
plt.style.use(hep.style.CMS)

bdt_axis = hist.axis.Regular(40, 0, 1, name="bdt", label=r"BDT")
cat_axis = hist.axis.StrCategory([], name="cat", label="cat", growth=True)
cut_axis = hist.axis.StrCategory([], name="cut", label="cut", growth=True)
h2_msd_axis = hist.axis.Regular(18, 40, 220, name="mass", label=r"Higgs 2 m$_{SD}$ [GeV]")
h2_mass_axis = hist.axis.Regular(18, 40, 220, name="mass", label=r"Higgs 2 m$_{reg}$ [GeV]")

bdt_cuts = [0, 0.03, 0.88, 0.95, 0.98]
txbb_cuts = [0, 0.8, 0.9, 0.98]

txbb_preselection = {
    "bbFatJetPNetTXbb": 0.3,
    "bbFatJetPNetTXbbLegacy": 0.8,
    "bbFatJetParTTXbb": 0.3,
}
msd1_preselection = {
    "bbFatJetPNetTXbb": 40,
    "bbFatJetPNetTXbbLegacy": 40,
    "bbFatJetParTTXbb": 40,
}
msd2_preselection = {
    "bbFatJetPNetTXbb": 30,
    "bbFatJetPNetTXbbLegacy": 0,
    "bbFatJetParTTXbb": 30,
}

control_plot_vars = [
    ShapeVar(var="H1Msd", label=r"$m_{SD}^{1}$ (GeV)", bins=[30, 0, 300]),
    ShapeVar(var="H2Msd", label=r"$m_{SD}^{2}$ (GeV)", bins=[30, 0, 300]),
    ShapeVar(var="H1TXbb", label=r"Xbb$^{1}$", bins=[30, 0, 1]),
    ShapeVar(var="H2TXbb", label=r"Xbb$^{2}$", bins=[30, 0, 1]),
    ShapeVar(var="H1Xbb", label=r"Xbb$^{1}$", bins=[30, 0, 1]),
    ShapeVar(var="H1PNetMass", label=r"$m_{reg}^{1}$ (GeV)", bins=[30, 0, 300]),
    ShapeVar(var="H2PNetMass", label=r"$m_{reg}^{2}$ (GeV)", bins=[30, 0, 300]),
    ShapeVar(var="H1Mass", label=r"$m_{reg}^{1}$ (GeV)", bins=[30, 0, 300]),
    ShapeVar(var="HHPt", label=r"HH $p_{T}$ (GeV)", bins=[30, 0, 4000]),
    ShapeVar(var="HHEta", label=r"HH $\eta$", bins=[30, -5, 5]),
    ShapeVar(var="HHMass", label=r"HH mass (GeV)", bins=[30, 0, 1500]),
    ShapeVar(var="MET", label=r"MET (GeV)", bins=[30, 0, 600]),
    ShapeVar(var="H1T32", label=r"$\tau_{32}^{0}$", bins=[30, 0, 1]),
    ShapeVar(var="H2T32", label=r"$\tau_{32}^{1}$", bins=[30, 0, 1]),
    ShapeVar(var="H1Pt", label=r"H $p_{T}^{0}$ (GeV)", bins=[30, 200, 1000]),
    ShapeVar(var="H2Pt", label=r"H $p_{T}^{1}$ (GeV)", bins=[30, 200, 1000]),
    ShapeVar(var="H1Eta", label=r"H $\eta^{0}$", bins=[30, -4, 4]),
    ShapeVar(var="H1QCDb", label=r"QCDb$^{1}$", bins=[30, 0, 0.2]),
    ShapeVar(var="H1QCDbb", label=r"QCDbb$^{1}$", bins=[30, 0, 0.2]),
    ShapeVar(var="H1QCDothers", label=r"QCDothers$^{1}$", bins=[30, 0, 0.2]),
    ShapeVar(var="H1Pt_HHmass", label=r"H$^0$ $p_{T}/mass$", bins=[30, 0, 1]),
    ShapeVar(var="H2Pt_HHmass", label=r"H$^1$ $p_{T}/mass$", bins=[30, 0, 0.7]),
    ShapeVar(var="H1Pt_H2Pt", label=r"H$^0$/H$^1$ $p_{T}$ (GeV)", bins=[30, 0.5, 1]),
    ShapeVar(var="VBFjjMass", label=r"VBF jj mass (GeV)", bins=[30, 0.0, 1000]),
    ShapeVar(var="VBFjjDeltaEta", label=r"VBF jj $\Delta \eta$", bins=[30, 0, 5]),
    ShapeVar(var="H1dRAK4", label=r"$\Delta R$(H1,J1)", bins=[30, 0, 5]),
    ShapeVar(var="H2dRAK4", label=r"$\Delta R$(H2,J2)", bins=[30, 0, 5]),
    ShapeVar(var="H1AK4mass", label=r"(H1 + J1) mass (GeV)", bins=[30, 80, 600]),
    ShapeVar(var="H2AK4mass", label=r"(H2 + J2) mass (GeV)", bins=[30, 80, 600]),
]

# do not include small qcd bins
for year in samples_run3:
    samples_run3[year]["qcd"] = [
        "QCD_HT-100to200",
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
        "QCD_HT-2000",
        "QCD_HT-200to400",
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
    ]
    samples_run3[year]["ttbar"] = [
        "TTto2L2Nu",
        "TTto4Q",
        "TTtoLNu2Q",
    ]
    samples_run3[year]["diboson"] = [
        "WW",
        "WZ",
        "ZZ",
    ]


def get_legtitle(txbb_str):
    title = r"FatJet p$_T^{(0,1)}$ > 250 GeV" + "\n"
    title += "$T_{Xbb}^{0}$ >" + f"{txbb_preselection[txbb_str]}"
    title += "$T_{Xbb}^{0}$ >" + f"{txbb_preselection[txbb_str]}"

    if "Legacy" in txbb_str:
        title += "\n" + "PNet Legacy"
    elif "ParT" in txbb_str:
        title += "\n" + "GloParTv2"
    else:
        title += "\n" + "PNet 103X"

    title += "\n" + r"m$_{reg}$ > 50 GeV"
    if "Legacy" not in txbb_str:
        title += "\n" + r"m$_{SD}^{0}$ > " + f"{msd1_preselection[txbb_str]} GeV"
        title += "\n" + r"m$_{SD}^{1}$ > " + f"{msd2_preselection[txbb_str]} GeV"
        title += "\n" + "PNet 103X"

    return title


def apply_cuts(events_dict, txbb_str, mass_str):
    """
    Apply cuts
    Skimmer selection already includes
    - 2 AK8 jets pT > 250 GeV, mSD > 60 or mReg > 60
    - HLT OR
    - Here we apply pT(1,2)> 250, mReg(1,2)>50 and a TXbb(1) and mSD(1,2) preselection
    - Here we apply pT(1,2)> 250, mReg(1,2)>50 and a TXbb(1) and mSD(1,2) preselection
    """
    for key in events_dict:
        msd1 = events_dict[key]["bbFatJetMsd"][0]
        msd2 = events_dict[key]["bbFatJetMsd"][1]
        pt1 = events_dict[key]["bbFatJetPt"][0]
        pt2 = events_dict[key]["bbFatJetPt"][1]
        txbb1 = events_dict[key][txbb_str][0]
        mass1 = events_dict[key][mass_str][0]
        mass2 = events_dict[key][mass_str][1]
        # add msd > 40 cut for the first jet FIXME: replace this by the trigobj matched jet
        events_dict[key] = events_dict[key][
            (pt1 > 250)
            & (pt2 > 250)
            & (txbb1 > txbb_preselection[txbb_str])
            & (msd1 > msd1_preselection[txbb_str])
            & (msd2 > msd2_preselection[txbb_str])
            & (mass1 > 50)
            & (mass2 > 50)
        ].copy()
        txbb1 = events_dict[key][txbb_str][0]
        mass1 = events_dict[key][mass_str][0]
        mass2 = events_dict[key][mass_str][1]
        # add msd > 40 cut for the first jet FIXME: replace this by the trigobj matched jet

    return events_dict


def preprocess_data(
    events_dict: dict,
    train_keys: list[str],
    config_name: str,
    test_size: float,
    seed: int,
    multiclass: bool,
    equalize_weights: bool,
    run2_wapproach: bool,
    label_encoder,
):

    # dataframe function
    make_bdt_dataframe = importlib.import_module(
        f".{config_name}", package="HH4b.boosted.bdt_trainings_run3"
    )

    training_keys = train_keys.copy()
    for key in training_keys:
        if key not in events_dict:
            logger.info(f"{key} not in events_dict, removing...")
            training_keys.remove(key)

    logger.info(f"Keys in events_dict {events_dict.keys()}")
    logger.info(f"Training keys {training_keys}")

    events_dict_bdt = {}
    weights_bdt = {}
    events_bdt = {}
    for key in training_keys:
        events_dict_bdt[key] = make_bdt_dataframe.bdt_dataframe(events_dict[key])
        # get absolute weights (no negative weights)
        weights_bdt[key] = np.abs(events_dict[key]["finalWeight"].to_numpy())
        events_bdt[key] = events_dict[key]["event"].to_numpy()[:, 0]

    events = pd.concat(
        [events_dict_bdt[key] for key in training_keys],
        keys=training_keys,
    )

    for key in weights_bdt:
        logger.info(f"Total {key} pre-normalization: {np.sum(weights_bdt[key]):.3f}")
        logger.info(f"Total {key} pre-normalization: {np.sum(weights_bdt[key]):.3f}")

    # weights
    if run2_wapproach:
        logger.info("Normalize weights, using Run-2 approach")
        bkg_weight = np.concatenate([weights_bdt[key] for key in args.bg_keys])
        bkg_weight_min = np.amin(np.absolute(bkg_weight))
        bkg_weight_rescale = 1.0 / np.absolute(bkg_weight_min)
        logger.info(f"Background weight rescale {bkg_weight_rescale}")
        for key in args.bg_keys:
            weights_bdt[key][weights_bdt[key] < 0] = 0
            events.loc[key, "weight"] = weights_bdt[key] * bkg_weight_rescale

        for sig_key in args.sig_keys:
            sig_weight = weights_bdt[sig_key]
            sig_weight_min = np.amin(np.absolute(sig_weight))
            sig_weight_rescale = 1.0 / np.absolute(sig_weight_min)
            logger.info(f"Signal weight rescale {sig_weight_rescale}")
            sig_weight[sig_weight < 0] = 0
            events.loc[sig_key, "weight"] = sig_weight * sig_weight_rescale

        for key in training_keys:
            logger.info(f"Total {key} post-normalization: {events.loc[key, 'weight'].sum():.3f}")

    if equalize_weights:
        logger.info("Equalize signal weights so that total signal = total bkg")
        bkg_total = np.sum([np.sum(weights_bdt[key]) for key in args.bg_keys])

        num_sigs = len(args.sig_keys)
        for sig_key in args.sig_keys:
            if sig_key not in weights_bdt:
                continue
            sig_total = np.sum(weights_bdt[sig_key])
            logger.info(f"Scaling {sig_key} by {bkg_total / sig_total} / {num_sigs} signal(s).")
            events.loc[sig_key, "weight"] = (
                weights_bdt[sig_key] * (bkg_total / sig_total) / num_sigs
            )

        for key in args.bg_keys:
            events.loc[key, "weight"] = weights_bdt[key]

        for key in training_keys:
            logger.info(f"Total {key} post-normalization: {events.loc[key, 'weight'].sum():.3f}")

    # Define target
    events["target"] = 0  # Default to 0 (background)
    for key in args.sig_keys:
        if key in training_keys:
            events.loc[key, "target"] = 1  # Set to 1 for 'hh4b' samples (signal)

    target = events["target"]
    simple_target = events["target"]

    if multiclass:
        logger.info("MultiClass labeling")
        target = label_encoder.transform(list(events.index.get_level_values(0)))

    # Define event number
    events["event"] = 1
    for key in training_keys:
        events.loc[key, "event"] = events_bdt[key]
    event_num = events["event"]

    # Define features
    features = events.drop(columns=["target", "event"])

    # Split the (bdt dataframe) dataset
    X_train, X_test, y_train, y_test, yt_train, yt_test, ev_train, ev_test = train_test_split(
        features,
        target,
        simple_target,
        event_num,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # drop weights from features
    weights_train = X_train["weight"].copy()
    X_train = X_train.drop(columns=["weight"])
    weights_test = X_test["weight"].copy()
    X_test = X_test.drop(columns=["weight"])

    for key in training_keys:
        logger.info(f"Total training {key} after splitting: {weights_train.loc[key].sum():.3f}")

    for key in training_keys:
        logger.info(f"Total testing {key} after splitting: {weights_test.loc[key].sum():.3f}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        weights_train,
        weights_test,
        yt_train,
        yt_test,
        ev_train,
        ev_test,
    )


def plot_losses(evals_result: dict, model_dir: Path, multiclass: bool):
    loss_key = "logloss" if not multiclass else "mlogloss"

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(["Train", "Test"]):
        plt.plot(evals_result[f"validation_{i}"][loss_key], label=label, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_dir / "losses.pdf", bbox_inches="tight")
    plt.close()

    logger.info("Loss saved as {model_dir.resolve()}/losses.pdf")


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    weights_train: np.ndarray,
    weights_test: np.ndarray,
    training_keys: list[str],
    model_dir: Path,
    **classifier_params,
):
    """Trains BDT. ``classifier_params`` are hyperparameters for the classifier"""
    early_stopping_callback = xgb.callback.EarlyStopping(rounds=5, min_delta=0.0)
    classifier_params = {**classifier_params, "callbacks": [early_stopping_callback]}

    logger.info(f"Training model with features {list(X_train.columns)}")
    model = xgb.XGBClassifier(**classifier_params)

    for key in training_keys:
        logger.info(f"Number of training {key} events: {X_train.loc[key].shape[0]}")

    for key in training_keys:
        logger.info(f"Number of testing {key} events: {X_test.loc[key].shape[0]}")

    trained_model = model.fit(
        X_train,
        y_train,
        sample_weight=weights_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        sample_weight_eval_set=[weights_train, weights_test],
        verbose=True,
    )

    trained_model.save_model(model_dir / "trained_bdt.model")

    evals_result = trained_model.evals_result()

    with (model_dir / "evals_result.txt").open("w") as f:
        f.write(str(evals_result))

    return model, evals_result


def evaluate_model(
    config_name: str,
    events_dict_years: dict,
    model: xgb.XGBClassifier,
    model_dir: Path,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    yt_test: pd.DataFrame,
    weights_test: np.ndarray,
    multiclass: bool,
    sig_keys: list[str],
    bg_keys: list[str],
    training_keys: list[str],
    txbb_plots: bool,
    txbb_str: str,
    mass_str: str,
):
    """
    1) Makes ROC curves for testing data
    2) Prints Sig efficiency at Bkg efficiency
    """
    plot_dir = model_dir / "evaluation"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # sorting by importance
    importances = model.feature_importances_
    feature_importances = sorted(
        zip(list(X_test.columns), importances), key=lambda x: x[1], reverse=True
    )
    feature_importance_df = pd.DataFrame.from_dict({"Importance": feature_importances})
    feature_importance_df.to_markdown(f"{model_dir}/feature_importances.md")

    # make and save ROCs for testing data
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    y_scores = model.predict_proba(X_test)
    y_scores = _get_bdt_scores(y_scores, sig_keys, multiclass)

    for i, sig_key in enumerate(sig_keys):
        (plot_dir / sig_key).mkdir(exist_ok=True, parents=True)

        logger.info(f"Evaluating {sig_key} performance")

        if multiclass:
            # selecting only this signal + BGs for ROC curves
            bgs = y_test >= len(sig_keys)
            sigs = y_test == i
            sel = np.logical_or(sigs, bgs).squeeze()
        else:
            sel = np.ones(len(y_test), dtype=bool)

        logger.info("Test ROC with sample weights")
        fpr, tpr, thresholds = roc_curve(
            yt_test[sel], y_scores[sel][:, i], sample_weight=weights_test[sel]
        )

        roc_info = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
        with (plot_dir / sig_key / "roc_dict.pkl").open("wb") as f:
            pickle.dump(roc_info, f)

        # print FPR, TPR for a couple of tprs
        for tpr_val in [0.10, 0.12, 0.15]:
            idx = find_nearest(tpr, tpr_val)
            logger.info(
                f"Signal efficiency: {tpr[idx]:.4f}, Background efficiency: {fpr[idx]:.5f}, BDT Threshold: {thresholds[idx]}"
            )

        # plot BDT scores for test samples
        make_bdt_dataframe = importlib.import_module(
            f".{config_name}", package="HH4b.boosted.bdt_trainings_run3"
        )

        logger.info("Performing inference on all samples")
        # get scores from full dataframe, but only use testing indices
        scores = {}
        weights = {}
        mass_dict = {}
        msd_dict = {}
        txbb_dict = {}

        for key in training_keys:
            score = []
            weight = []
            mass = []
            msd = []
            txbb = []
            for year in events_dict_years:
                evt_list = get_evt_testing(f"{model_dir}/inferences/{year}", key)
                if evt_list is None:
                    continue

                events = events_dict_years[year][key]
                bdt_events = make_bdt_dataframe.bdt_dataframe(events)
                test_bdt_dataframe = bdt_events.copy()
                bdt_events["event"] = events["event"].to_numpy()[:, 0]
                bdt_events["finalWeight"] = events["finalWeight"]
                bdt_events["mass"] = events[mass_str][1]
                bdt_events["msd"] = events["bbFatJetMsd"][1]
                bdt_events["txbb"] = events[txbb_str][1]
                mask = bdt_events["event"].isin(evt_list)
                test_dataset = bdt_events[mask]

                test_bdt_dataframe = test_bdt_dataframe[mask]
                test_preds = model.predict_proba(test_bdt_dataframe)

                score.append(_get_bdt_scores(test_preds, sig_keys, multiclass)[:, i])
                weight.append(test_dataset["finalWeight"])
                mass.append(test_dataset["mass"])
                msd.append(test_dataset["msd"])
                txbb.append(test_dataset["txbb"])

            scores[key] = np.concatenate(score)
            weights[key] = np.concatenate(weight)
            mass_dict[key] = np.concatenate(mass)
            msd_dict[key] = np.concatenate(msd)
            txbb_dict[key] = np.concatenate(txbb)

        for key in events_dict_years[year]:
            if key in training_keys:
                continue
            score = []
            weight = []
            txbb = []
            for year in events_dict_years:
                preds = model.predict_proba(
                    make_bdt_dataframe.bdt_dataframe(events_dict_years[year][key])
                )
                score.append(_get_bdt_scores(preds, sig_keys, multiclass)[:, i])
                weight.append(events_dict_years[year][key]["finalWeight"])
                txbb.append(events_dict_years[year][key][txbb_str][1])
                msd.append(events_dict_years[year][key]["bbFatJetMsd"][1])
                mass.append(events_dict_years[year][key][mass_str][1])
            scores[key] = np.concatenate(score)
            weights[key] = np.concatenate(weight)
            txbb_dict[key] = np.concatenate(txbb)
            msd_dict[key] = np.concatenate(msd)
            mass_dict[key] = np.concatenate(mass)

        logger.info("Making BDT shape plots")

        legtitle = get_legtitle(txbb_str)

        h_bdt = hist.Hist(bdt_axis, cat_axis)
        h_bdt_weight = hist.Hist(bdt_axis, cat_axis)
        for key in scores:
            h_bdt.fill(bdt=scores[key], cat=key)
            h_bdt_weight.fill(scores[key], key, weight=weights[key])

        hists = {
            "weight": h_bdt_weight,
            "no_weight": h_bdt,
        }
        for h_key, h in hists.items():
            colors = plotting.color_by_sample
            legends = plotting.label_by_sample

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            for key in scores:
                hep.histplot(
                    h[{"cat": key}],
                    ax=ax,
                    label=f"{legends[key]}",
                    histtype="step",
                    linewidth=1,
                    color=colors[key],
                    density=True,
                    flow="none",
                )
            ax.set_yscale("log")
            ax.legend(
                title=legtitle,
                bbox_to_anchor=(1.03, 1),
                loc="upper left",
            )
            ax.set_ylabel("Density")
            ax.set_title("Pre-Selection")
            ax.xaxis.grid(True, which="major")
            ax.yaxis.grid(True, which="major")
            fig.tight_layout()
            fig.savefig(plot_dir / sig_key / f"bdt_shape_{h_key}.png")
            fig.savefig(plot_dir / sig_key / f"bdt_shape_{h_key}.pdf", bbox_inches="tight")
            plt.close()

        logger.info("Making ROC Curves")
        # Plot and save ROC figure
        for log, logstr in [(False, ""), (True, "_log")]:
            fig, ax = plt.subplots(1, 1, figsize=(18, 12))
            bkg_colors = {**plotting.color_by_sample, "merged": "orange"}
            legends = {**plotting.label_by_sample, "merged": "Total Background"}
            plot_thresholds = bdt_cuts
            th_colours = ["#9381FF", "#1f78b4", "#a6cee3", "cyan", "blue"]

            for bkg in [*bg_keys, "merged"]:
                if bkg != "merged":
                    scores_roc = np.concatenate([scores[sig_key], scores[bkg]])
                    sig_jets_score = scores[sig_key]
                    bkg_jets_score = scores[bkg]
                    scores_true = np.concatenate(
                        [
                            np.ones(len(sig_jets_score)),
                            np.zeros(len(bkg_jets_score)),
                        ]
                    )
                    scores_weights = np.concatenate([weights[sig_key], weights[bkg]])
                    fpr, tpr, thresholds = roc_curve(
                        scores_true, scores_roc, sample_weight=scores_weights
                    )
                    # save background roc curves
                    roc_info_bg = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                    }
                    with (plot_dir / sig_key / f"roc_dict_{bkg}.pkl").open("wb") as f:
                        pickle.dump(roc_info_bg, f)
                else:
                    scores_roc = np.concatenate(
                        [scores[sig_key]] + [scores[bg_key] for bg_key in bg_keys]
                    )
                    sig_jets_score = scores[sig_key]
                    bkg_jets_score = np.concatenate([scores[bg_key] for bg_key in bg_keys])
                    scores_true = np.concatenate(
                        [
                            np.ones(len(sig_jets_score)),
                            np.zeros(len(bkg_jets_score)),
                        ]
                    )
                    scores_weights = np.concatenate(
                        [weights[sig_key]] + [weights[bg_key] for bg_key in bg_keys]
                    )
                    fpr, tpr, thresholds = roc_curve(
                        scores_true, scores_roc, sample_weight=scores_weights
                    )
                    # save background roc curves
                    roc_info_bg = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                    }
                    with (plot_dir / sig_key / f"roc_dict_{bkg}.pkl").open("wb") as f:
                        pickle.dump(roc_info_bg, f)

                ax.plot(tpr, fpr, linewidth=2, color=bkg_colors[bkg], label=legends[bkg])

                pths = {th: [[], []] for th in plot_thresholds}
                for th in plot_thresholds:
                    idx = find_nearest(thresholds, th)
                    pths[th][0].append(tpr[idx])
                    pths[th][1].append(fpr[idx])

                if bkg == "merged":
                    for k, th in enumerate(plot_thresholds):
                        plt.scatter(
                            *pths[th],
                            marker="o",
                            s=40,
                            label=rf"BDT > {th}",
                            color=th_colours[k],
                            zorder=100,
                        )

                        plt.vlines(
                            x=pths[th][0],
                            ymin=0,
                            ymax=pths[th][1],
                            color=th_colours[k],
                            linestyles="dashed",
                            alpha=0.5,
                        )

                        plt.hlines(
                            y=pths[th][1],
                            xmin=0,
                            xmax=pths[th][0],
                            color=th_colours[k],
                            linestyles="dashed",
                            alpha=0.5,
                        )

            ax.set_title(f"{plotting.label_by_sample[sig_key]} BDT ROC Curve")
            ax.set_xlabel("Signal efficiency")
            ax.set_ylabel("Background efficiency")

            if log:
                ax.set_xlim([0.0, 0.6])
                ax.set_ylim([1e-5, 1e-1])
                ax.set_yscale("log")
            else:
                ax.set_xlim([0.0, 0.7])
                ax.set_ylim([0, 0.08])

            ax.xaxis.grid(True, which="major")
            ax.yaxis.grid(True, which="major")
            ax.legend(
                title=legtitle,
                bbox_to_anchor=(1.03, 1),
                loc="upper left",
            )
            fig.tight_layout()
            fig.savefig(plot_dir / sig_key / f"roc_weights{logstr}.png")
            fig.savefig(plot_dir / sig_key / f"roc_weights{logstr}.pdf", bbox_inches="tight")
            plt.close()

    if not txbb_plots:
        return

    # TXbb ROC
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    plot_thresholds = [0.8, 0.9]
    th_colours = ["#9381FF", "#1f78b4", "#a6cee3"]
    for bkg in ["qcd", "ttbar", "merged"]:
        if bkg != "merged":
            scores_roc = np.concatenate([txbb_dict["hh4b"], txbb_dict[bkg]])
            sig_jets_score = txbb_dict["hh4b"]
            bkg_jets_score = txbb_dict[bkg]
            scores_true = np.concatenate(
                [
                    np.ones(len(sig_jets_score)),
                    np.zeros(len(bkg_jets_score)),
                ]
            )
            scores_weights = np.concatenate([weights["hh4b"], weights[bkg]])
            fpr, tpr, thresholds = roc_curve(scores_true, scores_roc, sample_weight=scores_weights)
        else:
            scores_roc = np.concatenate([txbb_dict["hh4b"], txbb_dict["qcd"], txbb_dict["ttbar"]])
            sig_jets_score = txbb_dict["hh4b"]
            bkg_jets_score = np.concatenate([txbb_dict["qcd"], txbb_dict["ttbar"]])
            scores_true = np.concatenate(
                [
                    np.ones(len(sig_jets_score)),
                    np.zeros(len(bkg_jets_score)),
                ]
            )
            scores_weights = np.concatenate([weights["hh4b"], weights["qcd"], weights["ttbar"]])
            fpr, tpr, thresholds = roc_curve(scores_true, scores_roc, sample_weight=scores_weights)

        ax.plot(tpr, fpr, linewidth=2, color=bkg_colors[bkg], label=legends[bkg])

        pths = {th: [[], []] for th in plot_thresholds}
        for th in plot_thresholds:
            idx = find_nearest(thresholds, th)
            pths[th][0].append(tpr[idx])
            pths[th][1].append(fpr[idx])

        if bkg == "merged":
            for k, th in enumerate(plot_thresholds):
                plt.scatter(
                    *pths[th],
                    marker="o",
                    s=40,
                    label=rf"BDT > {th}",
                    color=th_colours[k],
                    zorder=100,
                )

                plt.vlines(
                    x=pths[th][0],
                    ymin=0,
                    ymax=pths[th][1],
                    color=th_colours[k],
                    linestyles="dashed",
                    alpha=0.5,
                )

                plt.hlines(
                    y=pths[th][1],
                    xmin=0,
                    xmax=pths[th][0],
                    color=th_colours[k],
                    linestyles="dashed",
                    alpha=0.5,
                )

    ax.set_title("ggF HH4b TXbb ROC Curve")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_xlim([0.0, 0.7])
    ax.set_ylim([0, 0.08])
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    ax.legend(
        title=legtitle,
        bbox_to_anchor=(1.03, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(model_dir / "roc_txbb_weights.png")
    fig.savefig(model_dir / "roc_txbb_weights.png")
    plt.close()

    (model_dir / "validation_mass").mkdir(exist_ok=True, parents=True)
    """
    # mass sculpting with TXbb
    for txbb_cut in txbb_cuts:
        hist_h2 = hist.Hist(h2_mass_axis, cut_axis, cat_axis)
        hist_h2_msd = hist.Hist(h2_msd_axis, cut_axis, cat_axis)
        for key in txbb_dict:
            if key not in training_keys:
                continue
            h2_mass = mass_dict[key]
            h2_msd = msd_dict[key]
            h2_txbb = txbb_dict[key]
            for cut in bdt_cuts:
                mask = (scores[key] >= cut) & (h2_txbb >= txbb_cut)
                hist_h2.fill(h2_mass[mask], str(cut), key)
                hist_h2_msd.fill(h2_msd[mask], str(cut), key)

        hists = {
            "msd": hist_h2_msd,
            "mreg": hist_h2,
        }
        for key in txbb_dict:
            if key not in training_keys:
                continue
            for hkey, h in hists.items():
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                for cut in bdt_cuts:
                    hep.histplot(
                        h[{"cat": key, "cut": str(cut)}],
                        lw=2,
                        label=f"BDT > {cut}",
                        density=True,
                        flow="none",
                    )
                ax.legend()
                ax.set_ylabel("Density")
                ax.set_title(f"{legends[key]} TXbb > {txbb_cut}")
                ax.xaxis.grid(True, which="major")
                ax.yaxis.grid(True, which="major")
                fig.tight_layout()
                fig.savefig(model_dir / "validation_mass" / f"{hkey}2_{key}_txbbcut{txbb_cut}.png")
                fig.savefig(model_dir / "validation_mass" / f"{hkey}2_{key}_txbbcut{txbb_cut}.png")
                plt.close()
    """


def plot_allyears(
    events_dict,
    model,
    model_dir,
    config_name,
    multiclass,
    sig_keys: list[str],
    bg_keys: list[str],
    txbb_str: str,
    mass_str: str,
):
    make_bdt_dataframe = importlib.import_module(
        f".{config_name}", package="HH4b.boosted.bdt_trainings_run3"
    )

    for i, sig_key in enumerate(sig_keys):
        for txbb_cut in txbb_cuts:
            for key in bg_keys:
                hist_h2 = hist.Hist(h2_mass_axis, cut_axis, cat_axis)
                hist_h2_msd = hist.Hist(h2_msd_axis, cut_axis, cat_axis)

                for year in events_dict:
                    (model_dir / year).mkdir(exist_ok=True, parents=True)

                    preds = model.predict_proba(
                        make_bdt_dataframe.bdt_dataframe(events_dict[year][key])
                    )
                    scores = _get_bdt_scores(preds, sig_keys, multiclass)[:, i]

                    weights = events_dict[year][key]["finalWeight"]
                    h2_mass = events_dict[year][key][mass_str].to_numpy()[:, 1]
                    h2_msd = events_dict[year][key]["bbFatJetMsd"].to_numpy()[:, 1]
                    h2_txbb = events_dict[year][key][txbb_str].to_numpy()[:, 1]
                    cuts_filled = []
                    for cut in bdt_cuts:
                        mask = (scores >= cut) & (h2_txbb >= txbb_cut)
                        if np.any(mask):
                            cuts_filled.append(cut)
                        hist_h2.fill(h2_mass[mask], str(cut), year, weight=weights[mask])
                        hist_h2_msd.fill(h2_msd[mask], str(cut), year, weight=weights[mask])

                    hists = {
                        "msd": hist_h2_msd,
                        "mreg": hist_h2,
                    }

                    for hkey, h in hists.items():
                        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                        for cut in cuts_filled:
                            hep.histplot(
                                h[{"cat": year, "cut": str(cut)}],
                                lw=2,
                                label=f"BDT > {cut}",
                                density=True,
                                flow="none",
                            )
                        ax.legend()
                        ax.set_ylabel("Density")
                        ax.set_title(f"{year}, TXbb > {txbb_cut}")
                        ax.xaxis.grid(True, which="major")
                        ax.yaxis.grid(True, which="major")
                        fig.tight_layout()
                        fig.savefig(
                            model_dir
                            / year
                            / f"{sig_key}_{hkey}2_{key}_txbbcut{txbb_cut}_{year}.png"
                        )
                        fig.savefig(
                            model_dir
                            / year
                            / f"{sig_key}_{hkey}2_{key}_txbbcut{txbb_cut}_{year}.pdf",
                            bbox_inches="tight",
                        )
                        plt.close()


def _get_bdt_scores(preds, sig_keys, multiclass):
    # Helper function to calculate which BDT outputs to use
    if not multiclass:
        return preds[:, 1:]
    else:
        if len(sig_keys) == 1:
            return preds[:, :1]
        else:
            # Relevant score is signal score / (signal score + all background scores)
            bg_tot = np.sum(preds[:, len(sig_keys) :], axis=1, keepdims=True)
            return preds[:, : len(sig_keys)] / (preds[:, : len(sig_keys)] + bg_tot)


def plot_train_test(
    X_train,
    y_train,
    yt_train,
    weights_train,
    X_test,
    y_test,
    yt_test,
    weights_test,
    model,
    multiclass,
    sig_keys,
    training_keys,
    model_dir,
    txbb_str,
):
    plot_dir = model_dir / "train_test_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    colors = plotting.color_by_sample
    legends = plotting.label_by_sample

    (plot_dir / "inputs").mkdir(exist_ok=True, parents=True)
    ########### Plot training BDT Inputs ############
    for shape_var in control_plot_vars:
        # print("shape ", shape_var.var, X_train.columns)
        if shape_var.var in X_train.columns:
            h = hist.Hist(cat_axis, shape_var.axis)
            for key in training_keys:
                h.fill(key, X_train.loc[key, shape_var.var], weight=weights_train[key])

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            for key in training_keys:
                hep.histplot(
                    h[{"cat": key}],
                    ax=ax,
                    label=f"{legends[key]}",
                    histtype="step",
                    linewidth=1,
                    color=colors[key],
                    density=True,
                    flow="none",
                )
                ax.legend()
                ax.set_ylabel("Density")
                ax.xaxis.grid(True, which="major")
                ax.yaxis.grid(True, which="major")
                fig.tight_layout()
                shape_var.var = shape_var.var.replace("/", "_")
                fig.savefig(plot_dir / "inputs" / f"{shape_var.var}.png")

    for i, sig_key in enumerate(sig_keys):

        ########## Inference and ROC Curves ############
        rocs = {}
        for key, X, y, yt, weights in [
            ("train", X_train, y_train, yt_train, weights_train),
            ("test", X_test, y_test, yt_test, weights_test),
        ]:
            if multiclass:
                # selecting only this signal + BGs for ROC curves
                bgs = y >= len(sig_keys)
                sigs = y == i
                sel = np.logical_or(sigs, bgs).squeeze()
            else:
                sel = np.ones(len(y), dtype=bool)

            y_scores = model.predict_proba(X)
            y_scores = _get_bdt_scores(y_scores, sig_keys, multiclass)[:, i]

            fpr, tpr, thresholds = roc_curve(yt[sel], y_scores[sel], sample_weight=weights[sel])

            rocs[key] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "label": key,
            }

        ########### Plot ROC Curve ############
        for log, logstr in [(False, ""), (True, "_log")]:
            fig, ax = plt.subplots(1, 1, figsize=(18, 12))
            ax.plot(
                rocs["train"]["tpr"],
                rocs["train"]["fpr"],
                linewidth=2,
                label="Train Dataset",
            )
            ax.plot(
                rocs["test"]["tpr"],
                rocs["test"]["fpr"],
                linewidth=2,
                label="Test Dataset",
            )
            ax.set_title(f"{plotting.label_by_sample[sig_key]} BDT ROC Curve from Training")
            ax.set_xlabel("Signal efficiency")
            ax.set_ylabel("Background efficiency")

            if log:
                ax.set_xlim([0.0, 0.6])
                ax.set_ylim([1e-5, 1e-1])
                ax.set_yscale("log")
            else:
                ax.set_xlim([0.0, 0.7])
                ax.set_ylim([0, 0.08])

            ax.xaxis.grid(True, which="major")
            ax.yaxis.grid(True, which="major")

            legtitle = get_legtitle(txbb_str)

            ax.legend(
                title=legtitle,
                bbox_to_anchor=(1.03, 1),
                loc="upper left",
            )
            fig.tight_layout()
            fig.savefig(plot_dir / f"{sig_key}_roc_train_test{logstr}.png")
            fig.savefig(plot_dir / f"{sig_key}_roc_train_test{logstr}.pdf", bbox_inches="tight")

        ########### Plot BDT Shape ############
        h_bdt_weight = hist.Hist(bdt_axis, cat_axis)
        for key in training_keys:
            scores = model.predict_proba(X_test.loc[key])
            scores = _get_bdt_scores(scores, sig_keys, multiclass)[:, i]
            h_bdt_weight.fill(scores, key, weight=weights_test.loc[key])

        for key in training_keys:
            scores = model.predict_proba(X_train.loc[key])
            scores = _get_bdt_scores(scores, sig_keys, multiclass)[:, i]
            h_bdt_weight.fill(scores, key + "train", weight=weights_train.loc[key])

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        for key in training_keys:
            hep.histplot(
                h_bdt_weight[{"cat": key}],
                ax=ax,
                label=f"{legends[key]}",
                histtype="step",
                linewidth=1,
                color=colors[key],
                density=True,
                flow="none",
            )
            hep.histplot(
                h_bdt_weight[{"cat": key + "train"}],
                ax=ax,
                label=f"{legends[key]} Train",
                histtype="step",
                linewidth=1,
                linestyle="dashed",
                color=colors[key],
                density=True,
                flow="none",
            )
            ax.set_yscale("log")
            ax.legend(
                title=legtitle,
                bbox_to_anchor=(1.03, 1),
                loc="upper left",
            )
            ax.set_ylabel("Density")
            ax.set_title("Pre-Selection")
            ax.xaxis.grid(True, which="major")
            ax.yaxis.grid(True, which="major")

        fig.tight_layout()
        fig.savefig(plot_dir / "bdt_shape_traintest.png")
        fig.savefig(plot_dir / "bdt_shape_traintest.pdf", bbox_inches="tight")


def get_combined(data_dict):
    """
    Combine all years
    """
    X = []
    is_pandas = True
    for _year, data in data_dict.items():
        if isinstance(data, np.ndarray):
            is_pandas = False
        X.append(data)

    if is_pandas:
        return pd.concat(X, axis=0)
    else:
        return np.concatenate(X, axis=0)


def main(args):
    training_keys = args.sig_keys + args.bg_keys  # default: ["hh4b", "ttbar", "qcd"]

    model_dir = Path(f"./bdt_trainings_run3/{args.model_name}/")
    model_dir.mkdir(exist_ok=True, parents=True)

    if args.year == "2022-2023":
        years = ["2022", "2022EE", "2023", "2023BPix"]
    else:
        years = args.year

    X_train = OrderedDict()
    X_test = OrderedDict()
    y_train = OrderedDict()
    y_test = OrderedDict()
    weights_train = OrderedDict()
    weights_test = OrderedDict()
    yt_train = OrderedDict()
    yt_test = OrderedDict()

    events_dict_years = {}

    aux_keys = ["vhtobb", "tthtobb", "diboson", "vjets"]
    all_years = list(samples_run3.keys())
    for year in all_years:
        lkeys = list(samples_run3[year].keys())
        if year not in years:
            for key in lkeys:
                samples_run3[year].pop(key)
        else:
            for key in lkeys:
                if key not in training_keys + aux_keys:
                    logger.info(f"Removing {key}")
                    samples_run3[year].pop(key)

    logger.info(f"Samples to load {samples_run3}")
    for year in years:
        logger.info(f"Loading {year}, with txbb {args.txbb_str}")
        events_dict_years[year] = load_run3_samples(
            args.data_path,
            year,
            samples_run3,
            reorder_txbb=True,
            txbb_str=args.txbb_str,
            load_systematics=False,
            txbb_version=args.txbb,
            scale_and_smear=False,  # TODO: train with scale and smear corrections
            mass_str=args.mass_str,
        )

        if args.apply_cuts:
            # apply cuts
            events_dict_years[year] = apply_cuts(
                events_dict_years[year],
                args.txbb_str,
                args.mass_str,
            )

        # concatenate data
        # if doing multiclass classification, encode each process separately
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(
            training_keys
        )  # need this to maintain training keys order

        # pre-process data
        (
            X_train[year],
            X_test[year],
            y_train[year],
            y_test[year],
            weights_train[year],
            weights_test[year],
            yt_train[year],
            yt_test[year],
            _,
            ev_test,
        ) = preprocess_data(
            events_dict_years[year],
            training_keys,
            args.config_name,
            args.test_size,
            args.seed,
            args.multiclass,
            args.equalize_weights,
            args.run2_wapproach,
            label_encoder,
        )

        (model_dir / "inferences" / year).mkdir(exist_ok=True, parents=True)
        for key in training_keys:
            if key in events_dict_years[year]:
                np.save(f"{model_dir}/inferences/{year}/evt_{key}.npy", ev_test.loc[key])

    X_train_combined = get_combined(X_train)
    X_test_combined = get_combined(X_test)
    y_train_combined = get_combined(y_train)
    y_test_combined = get_combined(y_test)
    yt_train_combined = get_combined(yt_train)
    yt_test_combined = get_combined(yt_test)
    weights_train_combined = get_combined(weights_train)
    weights_test_combined = get_combined(weights_test)

    classifier_params = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": 5000,
        "verbosity": 2,
        "reg_lambda": 1.0,
    }

    if args.evaluate_only:
        model = xgb.XGBClassifier()
        model.load_model(model_dir / "trained_bdt.model")
        print("Loaded model", model)

        with (model_dir / "evals_result.txt").open("r") as f:
            evals_result = eval(f.read())
    else:
        model, evals_result = train_model(
            X_train_combined,
            X_test_combined,
            y_train_combined,
            y_test_combined,
            weights_train_combined,
            weights_test_combined,
            training_keys,
            model_dir,
            **classifier_params,
        )

        print(args.multiclass)
        plot_losses(evals_result, model_dir, args.multiclass)

    if not args.evaluate_only:
        plot_train_test(
            X_train_combined,
            y_train_combined,
            yt_train_combined,
            weights_train_combined,
            X_test_combined,
            y_test_combined,
            yt_test_combined,
            weights_test_combined,
            model,
            args.multiclass,
            args.sig_keys,
            training_keys,
            model_dir,
            args.txbb_str,
        )

    evaluate_model(
        args.config_name,
        events_dict_years,
        model,
        model_dir,
        X_test_combined,
        y_test_combined,
        yt_test_combined,
        weights_test_combined,
        args.multiclass,
        args.sig_keys,
        args.bg_keys,
        training_keys,
        args.txbb_plots,
        args.txbb_str,
        args.mass_str,
    )

    # test in other years
    events_dict = {}
    years_test = ["2022", "2022EE", "2023", "2023BPix"]
    for year in years_test:
        events_dict[year] = load_run3_samples(
            args.data_path,
            year,
            samples_run3,
            reorder_txbb=True,
            txbb_str=args.txbb_str,
            load_systematics=False,
            txbb_version=args.txbb,
            scale_and_smear=False,
            mass_str=args.mass_str,
        )
        if args.apply_cuts:
            events_dict[year] = apply_cuts(
                events_dict[year],
                args.txbb_str,
                args.mass_str,
            )

    if args.plot_allyears:
        plot_allyears(
            events_dict,
            model,
            model_dir,
            args.config_name,
            args.multiclass,
            args.sig_keys,
            args.bg_keys,
            args.txbb_str,
            args.mass_str,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        nargs="+",
        type=str,
        default=["2022EE"],
        choices=[hh_vars.years, "2022-2023"],
        help="years to train on",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="path to training data",
        type=str,
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="model name",
        type=str,
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="config name in case model name is different",
        type=str,
    )
    parser.add_argument("--test-size", default=0.4, help="testing/training split", type=float)
    parser.add_argument("--seed", default=42, help="seed for testing/training split", type=int)

    parser.add_argument(
        "--sig-keys", default=["hh4b"], help="which signals to train on", type=str, nargs="+"
    )
    parser.add_argument(
        "--bg-keys",
        default=["qcd", "ttbar"],
        help="which backgrounds to train on",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--txbb",
        choices=["pnet-v12", "pnet-legacy", "glopart-v2"],
        help="txbb version to be used for cuts and txbb performance comparison",
        required=True,
    )
    parser.add_argument(
        "--mass",
        choices=[
            "bbFatJetPNetMass",
            "bbFatJetPNetMassLegacy",
            "bbFatJetMsd",
            "bbFatJetParTmassVis",
        ],
        help="txbb mass",
        required=True,
    )
    parser.add_argument(
        "--learning-rate",
        default=0.1,
        help="BDT's learning rate",
        type=float,
    )
    parser.add_argument(
        "--max-depth",
        default=3,
        help="BDT's maximum depth",
        type=int,
    )

    add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)
    add_bool_arg(parser, "multiclass", "Classify each background separately", default=True)
    add_bool_arg(
        parser, "equalize-weights", "Equalise total signal and background weights", default=True
    )
    add_bool_arg(parser, "run2-wapproach", "Run2 weight approach", default=False)
    add_bool_arg(parser, "txbb-plots", "Make TXbb plots", default=True)
    add_bool_arg(parser, "apply-cuts", "Apply cuts", default=True)
    add_bool_arg(parser, "plot-allyears", "Plot histograms for all years", default=False)

    args = parser.parse_args()
    args.txbb_str = {
        "pnet-v12": "bbFatJetPNetTXbb",
        "pnet-legacy": "bbFatJetPNetTXbbLegacy",
        "glopart-v2": "bbFatJetParTTXbb",
    }[args.txbb]
    args.mass_str = args.mass

    if args.config_name is None:
        args.config_name = args.model_name

    main(args)
