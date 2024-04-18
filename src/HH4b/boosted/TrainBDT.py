from __future__ import annotations

import argparse
import importlib
import pickle
import sys
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

from HH4b.run_utils import add_bool_arg
from HH4b.utils import format_columns, load_samples

sys.path.append("./bdt_trainings_run3/")

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 12})
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.color"] = "#CCCCCC"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["figure.edgecolor"] = "none"
plt.style.use(hep.style.CMS)

training_keys = ["hh4b", "qcd", "ttbar"]


def load_data(data_path: str, year: str, legacy: bool):
    """
    Load samples
    """

    samples = {
        "2022": {
            #    "hh4b": "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_Private",
            "qcd": [
                "QCD_HT-400to600",
                "QCD_HT-600to800",
                "QCD_HT-800to1000",
                "QCD_HT-1000to1200",
                "QCD_HT-1200to1500",
                "QCD_HT-1500to2000",
                "QCD_HT-2000",
            ],
            "ttbar": [
                "TTto4Q",
            ],
        },
        "2022EE": {
            "qcd": [
                "QCD_HT-400to600",
                "QCD_HT-600to800",
                "QCD_HT-800to1000",
                "QCD_HT-1000to1200",
                "QCD_HT-1200to1500",
                "QCD_HT-1500to2000",
                "QCD_HT-2000",
            ],
            "ttbar": [
                "TTto4Q",
            ],
            "ttlep": [
                "TTtoLNu2Q",
            ],
            "hh4b": [
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV",
            ],
            "vhtobb": [
                "WminusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_Wto2Q_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2Q_M-125",
            ],
            "vjets": [
                "Wto2Q-3Jets_HT-200to400",
                "Wto2Q-3Jets_HT-400to600",
                "Wto2Q-3Jets_HT-600to800",
                "Wto2Q-3Jets_HT-800",
                "Zto2Q-4Jets_HT-200to400",
                "Zto2Q-4Jets_HT-400to600",
                "Zto2Q-4Jets_HT-600to800",
                "Zto2Q-4Jets_HT-800",
            ],
        },
        "2023": {
            "qcd": [
                "QCD_HT-400to600",
                "QCD_HT-600to800",
                "QCD_HT-800to1000",
                "QCD_HT-1000to1200",
                "QCD_HT-1200to1500",
                "QCD_HT-1500to2000",
                "QCD_HT-2000",
            ],
            "ttbar": [
                "TTto4Q",
            ],
        },
        "2023BPix": {
            "qcd": [
                "QCD_HT-400to600",
                "QCD_HT-600to800",
                "QCD_HT-800to1000",
                "QCD_HT-1000to1200",
                "QCD_HT-1200to1500",
                "QCD_HT-1500to2000",
                "QCD_HT-2000",
            ],
            "ttbar": [
                "TTto4Q",
            ],
        },
    }

    dirs = {data_path: samples}

    if legacy:
        filters = [
            [
                ("('bbFatJetPt', '0')", ">=", 300),
                ("('bbFatJetPt', '1')", ">=", 300),
            ]
        ]
    else:
        filters = [
            [
                ("('bbFatJetPt', '0')", ">=", 300),
                ("('bbFatJetPt', '1')", ">=", 300),
                ("('bbFatJetMsd', '0')", "<=", 250),
                ("('bbFatJetMsd', '1')", "<=", 250),
                ("('bbFatJetMsd', '0')", ">=", 30),
                ("('bbFatJetMsd', '1')", ">=", 30),
            ],
        ]

    load_columns = [
        ("weight", 1),
        ("event", 1),
        ("MET_pt", 1),
        ("bbFatJetPt", 2),
        ("bbFatJetEta", 2),
        ("bbFatJetPhi", 2),
        ("bbFatJetMsd", 2),
        ("bbFatJetPNetMass", 2),
        ("bbFatJetPNetXbb", 2),
        ("bbFatJetTau3OverTau2", 2),
        ("bbFatJetPNetQCD0HF", 2),
        ("bbFatJetPNetQCD1HF", 2),
        ("bbFatJetPNetQCD2HF", 2),
    ]
    if legacy:
        load_columns += [("bbFatJetPNetMassLegacy", 2), ("bbFatJetPNetXbbLegacy", 2)]

    events_dict = {}
    for input_dir, samples in dirs.items():
        events_dict = {
            **events_dict,
            **load_samples(
                input_dir,
                samples[year],
                year,
                filters=filters,
                variations=False,
                columns=format_columns(load_columns),
            ),
        }

    print(events_dict.keys())

    return events_dict


def preprocess_data(
    events_dict: dict, config_name: str, test_size: float, seed: int, multiclass: bool
):

    # dataframe function
    make_bdt_dataframe = importlib.import_module(f"{config_name}")
    events_dict_bdt = {}
    weights_bdt = {}
    events_bdt = {}
    for key in training_keys:
        events_dict_bdt[key] = make_bdt_dataframe.bdt_dataframe(events_dict[key])
        # get absolute weights (no negative weights)
        weights_bdt[key] = np.abs(events_dict[key]["finalWeight"].to_numpy())
        events_bdt[key] = events_dict[key]["event"].to_numpy()[:, 0]

    # concatenate data
    # if doing multiclass classification, encode each process separately
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(training_keys)  # need this to maintain training keys order

    events = pd.concat(
        [events_dict_bdt["hh4b"], events_dict_bdt["qcd"], events_dict_bdt["ttbar"]],
        keys=["hh4b", "qcd", "ttbar"],
    )
    events["target"] = 0  # Default to 0 (background)
    events.loc["hh4b", "target"] = 1  # Set to 1 for 'hh4b' samples (signal)

    # weights
    equalize_weights = True
    if equalize_weights:
        # scales signal such that total signal = total background
        sig_total = np.sum(weights_bdt["hh4b"])
        bkg_total = np.sum(np.concatenate((weights_bdt["qcd"], weights_bdt["ttbar"])))
        print(f"Scale signal by {bkg_total / sig_total}")

        events.loc["hh4b", "weight"] = weights_bdt["hh4b"] * (bkg_total / sig_total)
        events.loc["qcd", "weight"] = weights_bdt["qcd"]
        events.loc["ttbar", "weight"] = weights_bdt["ttbar"]

    # Define target
    target = events["target"]
    simple_target = events["target"]
    if multiclass:
        target = label_encoder.transform(list(events.index.get_level_values(0)))

    # Define event number
    events["event"] = 1
    events.loc["hh4b", "event"] = events_bdt["hh4b"]
    events.loc["qcd", "event"] = events_bdt["qcd"]
    events.loc["ttbar", "event"] = events_bdt["ttbar"]
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
        # shuffle=False
    )
    # drop weights from features
    weights_train = X_train["weight"].copy()
    X_train = X_train.drop(columns=["weight"])
    weights_test = X_test["weight"].copy()
    X_test = X_test.drop(columns=["weight"])

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


def plot_losses(trained_model: xgb.XGBClassifier, model_dir: Path):
    evals_result = trained_model.evals_result()

    with (model_dir / "evals_result.txt").open("w") as f:
        f.write(str(evals_result))

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(["Train", "Test"]):
        plt.plot(evals_result[f"validation_{i}"]["logloss"], label=label, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_dir / "losses.pdf", bbox_inches="tight")
    plt.close()


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    weights_train: np.ndarray,
    weights_test: np.ndarray,
    model_dir: Path,
    **classifier_params,
):
    """Trains BDT. ``classifier_params`` are hyperparameters for the classifier"""

    early_stopping_callback = xgb.callback.EarlyStopping(rounds=5, min_delta=0.0)
    classifier_params = {**classifier_params, "callbacks": [early_stopping_callback]}

    print("Training model")
    model = xgb.XGBClassifier(**classifier_params)
    print("Training features: ", list(X_train.columns))

    num_hh4b = X_train.loc["hh4b"].shape[0]
    num_qcd = X_train.loc["qcd"].shape[0]
    num_tt = X_train.loc["ttbar"].shape[0]
    print("Number of events training (hh4b,qcd,tt): ", num_hh4b, num_qcd, num_tt)
    num_hh4b = X_test.loc["hh4b"].shape[0]
    num_qcd = X_test.loc["qcd"].shape[0]
    num_tt = X_test.loc["ttbar"].shape[0]
    print("Number of events testing (hh4b,qcd,tt): ", num_hh4b, num_qcd, num_tt)

    trained_model = model.fit(
        X_train,
        y_train,
        sample_weight=weights_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        sample_weight_eval_set=[weights_train, weights_test],
        verbose=True,
    )
    trained_model.save_model(model_dir / "trained_bdt.model")
    plot_losses(trained_model, model_dir)

    # sorting by importance
    importances = model.feature_importances_
    feature_importances = sorted(
        zip(list(X_train.columns), importances), key=lambda x: x[1], reverse=True
    )
    feature_importance_df = pd.DataFrame.from_dict({"Importance": feature_importances})
    feature_importance_df.to_markdown(f"{model_dir}/feature_importances.md")

    return model


def evaluate_model(
    config_name: str,
    events_dict: dict,
    model: xgb.XGBClassifier,
    model_dir: Path,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,  # noqa: ARG001
    yt_test: pd.DataFrame,
    weights_test: np.ndarray,
    # test_size: float, seed: int,
    # year: str,
    multiclass: bool,
    legacy: bool,
):
    """
    1) Makes ROC curves for testing data
    2) Prints Sig efficiency at Bkg efficiency
    """

    pnet_xbb_str = "bbFatJetPNetXbb" if not legacy else "bbFatJetPNetXbbLegacy"
    pnet_mass_str = "bbFatJetPNetMass" if not legacy else "bbFatJetPNetMassLegacy"

    # make and save ROCs for testing data
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    y_scores = model.predict_proba(X_test)
    y_scores = y_scores[:, 0] if multiclass else y_scores[:, 1]

    # WARNING, if using multiclass should use y_Test?

    print("Test ROC with sample weights")
    fpr, tpr, thresholds = roc_curve(yt_test, y_scores, sample_weight=weights_test)

    roc_info = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
    with (model_dir / "roc_dict.pkl").open("wb") as f:
        pickle.dump(roc_info, f)

    # print FPR, TPR for a couple of tprs
    for tpr_val in [0.10, 0.12, 0.15]:
        idx = find_nearest(tpr, tpr_val)
        print(
            f"Signal efficiency: {tpr[idx]:.4f}, Background efficiency: {fpr[idx]:.5f}, BDT Threshold: {thresholds[idx]}"
        )

    # ROC w/o weights
    print("Test ROC without sample weights")
    fpr, tpr, thresholds = roc_curve(yt_test, y_scores)

    # print FPR, TPR for a couple of tprs
    for tpr_val in [0.10, 0.12, 0.15]:
        idx = find_nearest(tpr, tpr_val)
        print(
            f"Signal efficiency: {tpr[idx]:.4f}, Background efficiency: {fpr[idx]:.5f}, BDT Threshold: {thresholds[idx]}"
        )

    # plot BDT scores for test samples
    make_bdt_dataframe = importlib.import_module(f"{config_name}")

    print("Perform inference on test signal sample")
    # x_train, x_test = train_test_split(events_dict["hh4b"], test_size=test_size, random_state=seed)
    # hh4b_indices = x_test.index

    # get scores from full dataframe, but only use testing indices
    scores = {}
    weights = {}
    mass_dict = {}
    msd_dict = {}
    xbb_dict = {}
    for key in ["hh4b", "qcd", "ttbar"]:
        indices = X_test[X_test.index.get_level_values(0) == key].index.get_level_values(1)
        test_dataset = events_dict[key].loc[indices]
        test_bdt_dataframe = make_bdt_dataframe.bdt_dataframe(test_dataset)
        test_preds = model.predict_proba(test_bdt_dataframe)
        scores[key] = test_preds[:, 0] if multiclass else test_preds[:, 1]
        weights[key] = test_dataset["finalWeight"]
        mass_dict[key] = test_dataset[pnet_mass_str].to_numpy()[:, 1]
        msd_dict[key] = test_dataset["bbFatJetMsd"].to_numpy()[:, 1]
        xbb_dict[key] = test_dataset[pnet_xbb_str].to_numpy()[:, 1]

    for key in ["vhtobb", "vjets", "ttlep"]:
        preds = model.predict_proba(make_bdt_dataframe.bdt_dataframe(events_dict[key]))
        scores[key] = preds[:, 0] if multiclass else preds[:, 1]
        weights[key] = events_dict[key]["finalWeight"]
        xbb_dict[key] = events_dict[key][pnet_xbb_str].to_numpy()[:, 1]

    # save scores and indices for testing dataset
    # TODO: add shifts (e.g. JECs etc)
    # (model_dir / "inferences" / year).mkdir(exist_ok=True, parents=True)
    # np.save(f"{model_dir}/inferences/{year}/preds.npy", scores["hh4b"])
    # np.save(f"{model_dir}/inferences/{year}/indices.npy", hh4b_indices)

    # print("Scores ", scores)

    legtitle = r"FatJet p$_T^{(0,1)}$ > 300 GeV" + "\n" + "Xbb$^{0}$>0.8"
    if not legacy:
        legtitle += "\n" + r"m$_{SD}^{(0,1)}$:[30-250] GeV"
            
    bdt_axis = hist.axis.Regular(40, 0, 1, name="bdt", label=r"BDT")
    cat_axis = hist.axis.StrCategory([], name="cat", growth=True)
    h_bdt = hist.Hist(bdt_axis, cat_axis)
    h_bdt_weight = hist.Hist(bdt_axis, cat_axis)
    for key in events_dict:
        h_bdt.fill(bdt=scores[key], cat=key)
        h_bdt_weight.fill(bdt=scores[key], cat=key, weight=weights[key])

    hists = {
        "weight": h_bdt_weight,
        "no_weight": h_bdt,
    }
    for h_key, h in hists.items():
        colors = {
            "ttbar": "b",
            "hh4b": "k",
            "qcd": "r",
            "vhtobb": "g",
            "vjets": "pink",
            "ttlep": "violet",
        }
        legends = {
            "ttbar": r"$t\bar{t}$ + Jets",
            "hh4b": "ggHH(4b)",
            "qcd": "Multijet",
            "vhtobb": "VH(bb)",
            "vjets": r"W/Z$(qq)$ + Jets",
            "ttlep": r"$t\bar{t}$ (Lep.) + Jets",
        }
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        for key in events_dict:
            hep.histplot(
                h[{"cat": key}],
                ax=ax,
                label=f"{legends[key]}",
                histtype="step",
                linewidth=1,
                color=colors[key],
                density=True,
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
        fig.savefig(model_dir / f"bdt_shape_{h_key}.png")
        plt.close()

    # Plot and save ROC figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    bkg_colors = {
        "qcd": "r",
        "ttbar": "blue",
        "merged": "orange",
    }
    legends = {
        "qcd": "Multijet",
        "ttbar": r"$t\bar{t}$ + Jets",
        "merged": "TT + QCD",
    }
    plot_thresholds = [0.7, 0.9, 0.97]
    th_colours = ["#9381FF", "#1f78b4", "#a6cee3"]

    for bkg in ["qcd", "ttbar", "merged"]:
        if bkg != "merged":
            scores_roc = np.concatenate([scores["hh4b"], scores[bkg]])
            sig_jets_score = scores["hh4b"]
            bkg_jets_score = scores[bkg]
            scores_true = np.concatenate(
                [
                    np.ones(len(sig_jets_score)),
                    np.zeros(len(bkg_jets_score)),
                ]
            )
            scores_weights = np.concatenate([weights["hh4b"], weights[bkg]])
            fpr, tpr, thresholds = roc_curve(scores_true, scores_roc, sample_weight=scores_weights)
        else:
            # fpr, tpr, thresholds = roc_info["fpr"], roc_info["tpr"], roc_info["thresholds"]
            scores_roc = np.concatenate([scores["hh4b"], scores["qcd"], scores["ttbar"]])
            sig_jets_score = scores["hh4b"]
            bkg_jets_score = np.concatenate([scores["qcd"], scores["ttbar"]])
            scores_true = np.concatenate(
                [
                    np.ones(len(sig_jets_score)),
                    np.zeros(len(bkg_jets_score)),
                ]
            )
            scores_weights = np.concatenate([weights["hh4b"], weights["qcd"], scores["ttbar"]])
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

    ax.set_title("ggF HH4b BDT ROC Curve")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_xlim([0.0, 0.7])
    # ax.set_ylim([0, 0.002])
    ax.set_ylim([0, 0.08])
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    ax.legend(
        title=legtitle,
        bbox_to_anchor=(1.03, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(model_dir / "roc_weights.png")
    plt.close()

    # look into mass sculpting
    cat_axis = hist.axis.StrCategory([], name="Sample", growth=True)
    cut_axis = hist.axis.StrCategory([], name="Cut", growth=True)
    h2_msd_axis = hist.axis.Regular(40, 0, 300, name="mass", label=r"Higgs 2 m$_{SD}$ [GeV]")
    h2_mass_axis = hist.axis.Regular(40, 0, 300, name="mass", label=r"Higgs 2 m$_{reg}$ [GeV]")

    hist_h2 = hist.Hist(h2_mass_axis, cut_axis, cat_axis)
    hist_h2_msd = hist.Hist(h2_msd_axis, cut_axis, cat_axis)
    bdt_cuts = [0, 0.03, 0.7, 0.9, 0.97]

    legends = {
        "ttbar": r"$t\bar{t}$ + Jets",
        "hh4b": "ggHH(4b)",
        "qcd": "Multijet",
        "vhtobb": "VH(bb)",
        "vjets": r"W/Z$(qq)$ + Jets",
        "ttlep": r"$t\bar{t}$ (Lep.) + Jets",
    }
    for key in ["qcd", "ttbar", "vhtobb", "vjets", "hh4b"]:
        events = events_dict[key]
        if key in msd_dict:
            h2_mass = mass_dict[key]
            h2_msd = msd_dict[key]
        else:
            h2_mass = events[pnet_mass_str].to_numpy()[:, 1]
            h2_msd = events["bbFatJetMsd"].to_numpy()[:, 1]

        for cut in bdt_cuts:
            mask = scores[key] >= cut
            hist_h2.fill(h2_mass[mask], str(cut), key)
            hist_h2_msd.fill(h2_msd[mask], str(cut), key)

    for key in ["qcd", "ttbar", "vhtobb", "vjets", "hh4b"]:
        hists = {
            "msd": hist_h2_msd,
            "mreg": hist_h2,
        }
        for hkey, h in hists.items():
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            for cut in bdt_cuts:
                hep.histplot(
                    h[{"Sample": key, "Cut": str(cut)}], lw=2, label=f"BDT > {cut}", density=True
                )
            ax.legend()
            ax.set_ylabel("Density")
            ax.set_title(f"{legends[key]}")
            ax.xaxis.grid(True, which="major")
            ax.yaxis.grid(True, which="major")
            fig.tight_layout()
            fig.savefig(model_dir / f"{hkey}2_{key}.png")
            plt.close()

    # mass sculpting with Xbb
    hist_h2 = hist.Hist(h2_mass_axis, cut_axis, cat_axis)
    hist_h2_msd = hist.Hist(h2_msd_axis, cut_axis, cat_axis)
    bdt_cuts = [0, 0.03, 0.7, 0.9, 0.97]
    xbb_cuts = [0, 0.8, 0.9, 0.92]
    for xbb_cut in xbb_cuts:
        for key in ["qcd", "ttbar", "vhtobb", "vjets", "hh4b"]:
            events = events_dict[key]
            if key in msd_dict:
                h2_mass = mass_dict[key]
                h2_msd = msd_dict[key]
                h2_xbb = xbb_dict[key]
            else:
                h2_mass = events[pnet_mass_str].to_numpy()[:, 1]
                h2_msd = events["bbFatJetMsd"].to_numpy()[:, 1]
                h2_xbb = events[pnet_xbb_str].to_numpy()[:, 1]

            for cut in bdt_cuts:
                mask = (scores[key] >= cut) & (h2_xbb > xbb_cut)
                hist_h2.fill(h2_mass[mask], str(cut), key)
                hist_h2_msd.fill(h2_msd[mask], str(cut), key)
        for key in ["qcd", "ttbar", "vhtobb", "vjets", "hh4b"]:
            hists = {
                "msd": hist_h2_msd,
                "mreg": hist_h2,
            }
            for hkey, h in hists.items():
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                for cut in bdt_cuts:
                    hep.histplot(
                        h[{"Sample": key, "Cut": str(cut)}],
                        lw=2,
                        label=f"BDT > {cut}",
                        density=True,
                    )
                ax.legend()
                ax.set_ylabel("Density")
                ax.set_title(f"{legends[key]} Xbb > {xbb_cut}")
                ax.xaxis.grid(True, which="major")
                ax.yaxis.grid(True, which="major")
                fig.tight_layout()
                fig.savefig(model_dir / f"{hkey}2_{key}_xbbcut{xbb_cut}.png")
                plt.close()


def plot_train_test(
    X_train, yt_train, weights_train, X_test, yt_test, weights_test, model, multiclass, model_dir, legacy
):
    y_scores_train = model.predict_proba(X_train)
    y_scores_train = y_scores_train[:, 0] if multiclass else y_scores_train[:, 1]
    fpr_train, tpr_train, thresholds_train = roc_curve(
        yt_train, y_scores_train, sample_weight=weights_train
    )

    y_scores_test = model.predict_proba(X_test)
    y_scores_test = y_scores_test[:, 0] if multiclass else y_scores_test[:, 1]
    fpr, tpr, thresholds = roc_curve(yt_test, y_scores_test, sample_weight=weights_test)

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.plot(tpr_train, fpr_train, linewidth=2, color="orange", label="Train Dataset")
    ax.plot(tpr, fpr, linewidth=2, color="orange", label="Test Dataset")
    ax.set_title("ggF HH4b BDT ROC Curve from Training")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_xlim([0.0, 0.7])
    ax.set_ylim([0, 0.07])
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    legtitle = r"FatJet p$_T^{(0,1)}$ > 300 GeV" + "\n" + "Xbb$^{0}$>0.8"
    if not legacy:
        legtitle += "\n" + r"m$_{SD}^{(0,1)}$:[30-250] GeV"
    ax.legend(
        title=legtitle,
        bbox_to_anchor=(1.03, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(model_dir / "roc_train_test.png")

    bdt_axis = hist.axis.Regular(40, 0, 1, name="bdt", label=r"BDT")
    cat_axis = hist.axis.StrCategory([], name="cat", growth=True)
    h_bdt_weight = hist.Hist(bdt_axis, cat_axis)
    for key in ["qcd", "ttbar", "hh4b"]:
        scores = model.predict_proba(X_test.loc[key])
        scores = scores[:, 0] if multiclass else scores[:, 1]
        print(weights_test.loc[key])
        h_bdt_weight.fill(bdt=scores, cat=key, weight=weights_test.loc[key])
    for key in ["qcd", "ttbar", "hh4b"]:
        scores = model.predict_proba(X_train.loc[key])
        scores = scores[:, 0] if multiclass else scores[:, 1]
        h_bdt_weight.fill(bdt=scores, cat=key + "train", weight=weights_train.loc[key])

    colors = {
        "ttbar": "b",
        "hh4b": "k",
        "qcd": "r",
        "vhtobb": "g",
        "vjets": "pink",
        "ttlep": "violet",
    }
    legends = {
        "ttbar": r"$t\bar{t}$ + Jets",
        "hh4b": "ggHH(4b)",
        "qcd": "Multijet",
        "vhtobb": "VH(bb)",
        "vjets": r"W/Z$(qq)$ + Jets",
        "ttlep": r"$t\bar{t}$ (Lep.) + Jets",
    }
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for key in ["qcd", "ttbar", "hh4b"]:
        hep.histplot(
            h_bdt_weight[{"cat": key}],
            ax=ax,
            label=f"{legends[key]}",
            histtype="step",
            linewidth=1,
            color=colors[key],
            density=True,
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
    fig.savefig(model_dir / "bdt_shape_traintest.png")


def main(args):
    # for year in ["2022", "2022EE", "2023", "2023BPix"]:
    year = "2022EE"
    events_dict = load_data(args.data_path, year, args.legacy)

    (
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
    ) = preprocess_data(events_dict, args.config_name, args.test_size, args.seed, args.multiclass)

    model_dir = Path(f"./bdt_trainings_run3/{args.model_name}/")
    model_dir.mkdir(exist_ok=True, parents=True)

    (model_dir / "inferences" / year).mkdir(exist_ok=True, parents=True)
    ev_hh4b = ev_test.loc["hh4b"]
    ev_qcd = ev_test.loc["qcd"]
    ev_tt = ev_test.loc["ttbar"]
    np.save(f"{model_dir}/inferences/{year}/evt_hh4b.npy", ev_hh4b)
    np.save(f"{model_dir}/inferences/{year}/evt_qcd.npy", ev_qcd)
    np.save(f"{model_dir}/inferences/{year}/evt_ttbar.npy", ev_tt)

    classifier_params = {
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 1000,
        "verbosity": 2,
        "reg_lambda": 1.0,
    }

    if args.evaluate_only:
        model = xgb.XGBClassifier()
        model.load_model(model_dir / "trained_bdt.model")
        print(model)
    else:
        model = train_model(
            X_train,
            X_test,
            y_train,
            y_test,
            weights_train,
            weights_test,
            model_dir,
            **classifier_params,
        )
        plot_train_test(
            X_train,
            yt_train,
            weights_train,
            X_test,
            yt_test,
            weights_test,
            model,
            args.multiclass,
            model_dir,
            args.legacy,
        )

    evaluate_model(
        args.config_name,
        events_dict,
        model,
        model_dir,
        X_test,
        y_test,
        yt_test,
        weights_test,
        args.multiclass,
        args.legacy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)
    add_bool_arg(parser, "multiclass", "Classify each background separately", default=True)
    add_bool_arg(parser, "legacy", "Legacy PNet versions", default=False)

    args = parser.parse_args()

    if args.config_name is None:
        args.config_name = args.model_name

    main(args)
