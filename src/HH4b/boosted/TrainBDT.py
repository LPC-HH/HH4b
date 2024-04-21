from __future__ import annotations

import argparse
import importlib
import pickle
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

from HH4b import plotting
from HH4b.run_utils import add_bool_arg
from HH4b.utils import format_columns, load_samples

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 12})
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.color"] = "#CCCCCC"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["figure.edgecolor"] = "none"
plt.style.use(hep.style.CMS)


bdt_axis = hist.axis.Regular(40, 0, 1, name="bdt", label=r"BDT")
cat_axis = hist.axis.StrCategory([], name="cat", label="cat", growth=True)
cut_axis = hist.axis.StrCategory([], name="cut", label="cut", growth=True)
h2_msd_axis = hist.axis.Regular(18, 40, 220, name="mass", label=r"Higgs 2 m$_{SD}$ [GeV]")
h2_mass_axis = hist.axis.Regular(18, 40, 220, name="mass", label=r"Higgs 2 m$_{reg}$ [GeV]")

bdt_cuts = [0, 0.03, 0.7, 0.9, 0.92]
xbb_cuts = [0, 0.8, 0.9, 0.92]


def _get_title(legacy: bool):
    title = r"FatJet p$_T^{(0,1)}$ > 300 GeV" + "\n" + "Xbb$^{0}$>0.8"

    if not legacy:
        title += "\n" + r"m$_{SD}^{(0,1)}$:[30-250] GeV"
    else:
        title += "\n" + r"m$_{Reg}^{(0,1)}$:[60-250] GeV"

    return title


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
            "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
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
            "vbfhh4b-k2v0": ["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"],
            # "vhtobb": [
            #     "WminusH_Hto2B_Wto2Q_M-125",
            #     "WplusH_Hto2B_Wto2Q_M-125",
            #     "ZH_Hto2B_Zto2Q_M-125",
            #     "ggZH_Hto2B_Zto2Q_M-125",
            # ],
            # "vjets": [
            #     "Wto2Q-3Jets_HT-200to400",
            #     "Wto2Q-3Jets_HT-400to600",
            #     "Wto2Q-3Jets_HT-600to800",
            #     "Wto2Q-3Jets_HT-800",
            #     "Zto2Q-4Jets_HT-200to400",
            #     "Zto2Q-4Jets_HT-400to600",
            #     "Zto2Q-4Jets_HT-600to800",
            #     "Zto2Q-4Jets_HT-800",
            # ],
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
            "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG"],
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
            "hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG"],
        },
    }

    dirs = {data_path: samples}

    if legacy:
        # mass_key = "bbFatJetMsd"
        mass_key = "bbFatJetPNetMassLegacy"
        # both jets pT > 300, both jets mass [60, 250], at least one jet's TXbb legacy > 0.8
        # (need to do an OR since ordering is based on v12 TXbb, not legacy for now)
        filters = [
            [
                ("('bbFatJetPt', '0')", ">=", 300),
                ("('bbFatJetPt', '1')", ">=", 300),
                # added
                (f"('{mass_key}', '0')", "<=", 250),
                (f"('{mass_key}', '1')", "<=", 250),
                (f"('{mass_key}', '0')", ">=", 60),
                (f"('{mass_key}', '1')", ">=", 60),
                ("('bbFatJetPNetTXbbLegacy', '0')", ">=", 0.8),
            ],
            [
                ("('bbFatJetPt', '0')", ">=", 300),
                ("('bbFatJetPt', '1')", ">=", 300),
                # added
                (f"('{mass_key}', '0')", "<=", 250),
                (f"('{mass_key}', '1')", "<=", 250),
                (f"('{mass_key}', '0')", ">=", 60),
                (f"('{mass_key}', '1')", ">=", 60),
                ("('bbFatJetPNetTXbbLegacy', '1')", ">=", 0.8),
            ],
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
        ("bbFatJetTau3OverTau2", 2),
    ]

    if legacy:
        load_columns += [
            ("bbFatJetPNetTXbbLegacy", 2),
            ("bbFatJetPNetPXbbLegacy", 2),
            ("bbFatJetPNetPQCDbLegacy", 2),
            ("bbFatJetPNetPQCDbbLegacy", 2),
            ("bbFatJetPNetPQCDothersLegacy", 2),
            ("bbFatJetPNetMassLegacy", 2),
        ]
    else:
        load_columns += [
            ("bbFatJetPNetTXbb", 2),
            ("bbFatJetPNetMass", 2),
            ("bbFatJetPNetQCD0HF", 2),
            ("bbFatJetPNetQCD1HF", 2),
            ("bbFatJetPNetQCD2HF", 2),
        ]

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
                reorder_legacy_txbb=legacy,
                columns=format_columns(load_columns),
            ),
        }

    # apply mask (maybe this is not needed once ordering is fixed in processor)
    for key in events_dict:
        if legacy:
            xbb_0 = events_dict[key]["bbFatJetPNetTXbbLegacy"].to_numpy()[:, 0]
            mask = (xbb_0 >= 0.8)
            events_dict[key] = events_dict[key][mask]
        
    return events_dict


def preprocess_data(
    events_dict: dict,
    training_keys: list[str],
    config_name: str,
    test_size: float,
    seed: int,
    multiclass: bool,
):

    # dataframe function
    make_bdt_dataframe = importlib.import_module(
        f".{config_name}", package="HH4b.boosted.bdt_trainings_run3"
    )

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
        [events_dict_bdt[key] for key in training_keys],
        keys=training_keys,
    )

    # weights
    equalize_weights = True
    if equalize_weights:
        # scales signal such that total signal = total background
        bkg_total = np.sum([np.sum(weights_bdt[key]) for key in args.bg_keys])

        num_sigs = len(args.sig_keys)
        for sig_key in args.sig_keys:
            sig_total = np.sum(weights_bdt[sig_key])
            print(f"Scale {sig_key} by {bkg_total / sig_total} / {num_sigs} signal(s).")
            events.loc[sig_key, "weight"] = (
                weights_bdt[sig_key] * (bkg_total / sig_total) / num_sigs
            )

        for key in args.bg_keys:
            events.loc[key, "weight"] = weights_bdt[key]

    # Define target
    events["target"] = 0  # Default to 0 (background)
    for key in args.sig_keys:
        events.loc[key, "target"] = 1  # Set to 1 for 'hh4b' samples (signal)

    target = events["target"]
    simple_target = events["target"]

    if multiclass:
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
    """Plot Losses"""
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
    training_keys: list[str],
    model_dir: Path,
    **classifier_params,
):
    """Trains BDT. ``classifier_params`` are hyperparameters for the classifier"""

    early_stopping_callback = xgb.callback.EarlyStopping(rounds=5, min_delta=0.0)
    classifier_params = {**classifier_params, "callbacks": [early_stopping_callback]}

    print("Training model")
    model = xgb.XGBClassifier(**classifier_params)
    print("Training features: ", list(X_train.columns))

    for key in training_keys:
        print(f"Number of training {key} events: {X_train.loc[key].shape[0]}")

    for key in training_keys:
        print(f"Number of testing {key} events: {X_test.loc[key].shape[0]}")

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
    y_test: pd.DataFrame,
    yt_test: pd.DataFrame,
    weights_test: np.ndarray,
    multiclass: bool,
    sig_keys: list[str],
    bg_keys: list[str],
    training_keys: list[str],
    legacy: bool,
):
    """
    1) Makes ROC curves for testing data
    2) Prints Sig efficiency at Bkg efficiency
    """

    pnet_xbb_str = "bbFatJetPNetTXbb" if not legacy else "bbFatJetPNetTXbbLegacy"
    pnet_mass_str = "bbFatJetPNetMass" if not legacy else "bbFatJetPNetMassLegacy"

    # make and save ROCs for testing data
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    y_scores = model.predict_proba(X_test)
    y_scores = _get_bdt_scores(y_scores, sig_keys, multiclass)

    for i, sig_key in enumerate(sig_keys):
        (model_dir / sig_key ).mkdir(exist_ok=True, parents=True)

        if multiclass:
            # selecting only this signal + BGs for ROC curves
            bgs = y_test >= len(sig_keys)
            sigs = y_test == i
            sel = np.logical_or(sigs, bgs).to_numpy().squeeze()
        else:
            sel = np.ones(len(y_test), dtype=bool)

        print("Test ROC with sample weights")
        fpr, tpr, thresholds = roc_curve(
            yt_test[sel], y_scores[sel][:, i], sample_weight=weights_test[sel]
        )

        roc_info = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
        with (model_dir / sig_key / "roc_dict.pkl").open("wb") as f:
            pickle.dump(roc_info, f)

        # print FPR, TPR for a couple of tprs
        for tpr_val in [0.10, 0.12, 0.15]:
            idx = find_nearest(tpr, tpr_val)
            print(
                f"Signal efficiency: {tpr[idx]:.4f}, Background efficiency: {fpr[idx]:.5f}, BDT Threshold: {thresholds[idx]}"
            )

        # plot BDT scores for test samples
        make_bdt_dataframe = importlib.import_module(
            f".{config_name}", package="HH4b.boosted.bdt_trainings_run3"
        )

        print("Perform inference on test signal sample")
        # get scores from full dataframe, but only use testing indices
        scores = {}
        weights = {}
        mass_dict = {}
        msd_dict = {}
        xbb_dict = {}

        for key in training_keys:
            indices = X_test[X_test.index.get_level_values(0) == key].index.get_level_values(1)
            test_dataset = events_dict[key].loc[indices]
            test_bdt_dataframe = make_bdt_dataframe.bdt_dataframe(test_dataset)
            test_preds = model.predict_proba(test_bdt_dataframe)
            scores[key] = _get_bdt_scores(test_preds, sig_keys, multiclass)[:, i]
            weights[key] = test_dataset["finalWeight"]
            mass_dict[key] = test_dataset[pnet_mass_str].to_numpy()[:, 1]
            msd_dict[key] = test_dataset["bbFatJetMsd"].to_numpy()[:, 1]
            xbb_dict[key] = test_dataset[pnet_xbb_str].to_numpy()[:, 1]

        other_keys = ["ttlep"] # + ["vjets", "vhtobb"]
        for key in other_keys:
            preds = model.predict_proba(make_bdt_dataframe.bdt_dataframe(events_dict[key]))
            scores[key] = _get_bdt_scores(preds, sig_keys, multiclass)[:, i]
            weights[key] = events_dict[key]["finalWeight"]
            xbb_dict[key] = events_dict[key][pnet_xbb_str].to_numpy()[:, 1]

        legtitle = _get_title(legacy)

        h_bdt = hist.Hist(bdt_axis, cat_axis)
        h_bdt_weight = hist.Hist(bdt_axis, cat_axis)
        for key in events_dict:
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
            fig.savefig(model_dir / sig_key / f"bdt_shape_{h_key}.png")
            fig.savefig(model_dir / sig_key / f"bdt_shape_{h_key}.pdf", bbox_inches="tight")
            plt.close()

        # Plot and save ROC figure
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        bkg_colors = {**plotting.color_by_sample, "merged": "orange"}
        legends = {**plotting.label_by_sample, "merged": "Total Background"}
        plot_thresholds = [0.68, 0.9, 0.92]
        th_colours = ["#9381FF", "#1f78b4", "#a6cee3"]

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
        ax.set_xlim([0.0, 0.7])
        ax.set_ylim([1e-6, 1e-1])
        ax.set_yscale("log")
        ax.xaxis.grid(True, which="major")
        ax.yaxis.grid(True, which="major")
        ax.legend(
            title=legtitle,
            bbox_to_anchor=(1.03, 1),
            loc="upper left",
        )
        fig.tight_layout()
        fig.savefig(model_dir / sig_key / "roc_weights.png")
        fig.savefig(model_dir / sig_key / "roc_weights.pdf", bbox_inches="tight")
        plt.close()

        # look into mass sculpting

        hist_h2 = hist.Hist(h2_mass_axis, cut_axis, cat_axis)
        hist_h2_msd = hist.Hist(h2_msd_axis, cut_axis, cat_axis)

        for key in training_keys + other_keys:
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

        for key in training_keys + other_keys:
            hists = {
                "msd": hist_h2_msd,
                "mreg": hist_h2,
            }
            for hkey, h in hists.items():
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                for cut in bdt_cuts:
                    hep.histplot(
                        h[{"cat": key, "cut": str(cut)}], lw=2, label=f"BDT > {cut}", density=True
                    )
                ax.legend()
                ax.set_ylabel("Density")
                ax.set_title(f"{legends[key]}")
                ax.xaxis.grid(True, which="major")
                ax.yaxis.grid(True, which="major")
                fig.tight_layout()
                fig.savefig(model_dir / sig_key / f"{hkey}2_{key}.png")
                fig.savefig(model_dir / sig_key / f"{hkey}2_{key}.pdf", bbox_inches="tight")
                plt.close()

    # PNetXbb ROC
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    plot_thresholds = [0.8, 0.9]
    th_colours = ["#9381FF", "#1f78b4", "#a6cee3"]
    for bkg in ["qcd", "ttbar", "merged"]:
        if bkg != "merged":
            scores_roc = np.concatenate([xbb_dict["hh4b"], xbb_dict[bkg]])
            sig_jets_score = xbb_dict["hh4b"]
            bkg_jets_score = xbb_dict[bkg]
            scores_true = np.concatenate(
                [
                    np.ones(len(sig_jets_score)),
                    np.zeros(len(bkg_jets_score)),
                ]
            )
            scores_weights = np.concatenate([weights["hh4b"], weights[bkg]])
            fpr, tpr, thresholds = roc_curve(scores_true, scores_roc, sample_weight=scores_weights)
        else:
            scores_roc = np.concatenate([xbb_dict["hh4b"], xbb_dict["qcd"], xbb_dict["ttbar"]])
            sig_jets_score = xbb_dict["hh4b"]
            bkg_jets_score = np.concatenate([xbb_dict["qcd"], xbb_dict["ttbar"]])
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

    ax.set_title("ggF HH4b PNetXbb ROC Curve")
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
    fig.savefig(model_dir / "roc_pnetxbb_weights.png")
    plt.close()

    # mass sculpting with Xbb
    for xbb_cut in xbb_cuts:
        hist_h2 = hist.Hist(h2_mass_axis, cut_axis, cat_axis)
        hist_h2_msd = hist.Hist(h2_msd_axis, cut_axis, cat_axis)
        for key in training_keys + other_keys:
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
                mask = (scores[key] >= cut) & (h2_xbb >= xbb_cut)
                hist_h2.fill(h2_mass[mask], str(cut), key)
                hist_h2_msd.fill(h2_msd[mask], str(cut), key)
        for key in training_keys + other_keys:
            hists = {
                "msd": hist_h2_msd,
                "mreg": hist_h2,
            }
            for hkey, h in hists.items():
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                for cut in bdt_cuts:
                    hep.histplot(
                        h[{"cat": key, "cut": str(cut)}],
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


def plot_allyears(
    events_dict,
    model,
    model_dir,
    config_name,
    multiclass,
    sig_keys: list[str],
    bg_keys: list[str],
    legacy,
):

    pnet_xbb_str = "bbFatJetPNetTXbb" if not legacy else "bbFatJetPNetTXbbLegacy"
    pnet_mass_str = "bbFatJetPNetMass" if not legacy else "bbFatJetPNetMassLegacy"
    make_bdt_dataframe = importlib.import_module(
        f".{config_name}", package="HH4b.boosted.bdt_trainings_run3"
    )

    for i, sig_key in enumerate(sig_keys):
        (model_dir / sig_key ).mkdir(exist_ok=True, parents=True)

        for xbb_cut in xbb_cuts:
            for key in bg_keys:
                hist_h2 = hist.Hist(h2_mass_axis, cut_axis, cat_axis)
                hist_h2_msd = hist.Hist(h2_msd_axis, cut_axis, cat_axis)

                for year in events_dict:
                    (model_dir / sig_key / year).mkdir(exist_ok=True, parents=True)

                    preds = model.predict_proba(
                        make_bdt_dataframe.bdt_dataframe(events_dict[year][key])
                    )
                    scores = _get_bdt_scores(preds, sig_keys, multiclass)[:, i]

                    weights = events_dict[year][key]["finalWeight"]
                    h2_mass = events_dict[year][key][pnet_mass_str].to_numpy()[:, 1]
                    h2_msd = events_dict[year][key]["bbFatJetMsd"].to_numpy()[:, 1]
                    h2_xbb = events_dict[year][key][pnet_xbb_str].to_numpy()[:, 1]
                    for cut in bdt_cuts:
                        mask = (scores >= cut) & (h2_xbb >= xbb_cut)
                        hist_h2.fill(h2_mass[mask], str(cut), year, weight=weights[mask])
                        hist_h2_msd.fill(h2_msd[mask], str(cut), year, weight=weights[mask])

                hists = {
                    "msd": hist_h2_msd,
                    "mreg": hist_h2,
                }
                for year in events_dict:
                    for hkey, h in hists.items():
                        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                        for cut in bdt_cuts:
                            hep.histplot(
                                h[{"cat": year, "cut": str(cut)}],
                                lw=2,
                                label=f"BDT > {cut}",
                                density=True,
                            )
                        ax.legend()
                        ax.set_ylabel("Density")
                        ax.set_title(f"{year}, Xbb > {xbb_cut}")
                        ax.xaxis.grid(True, which="major")
                        ax.yaxis.grid(True, which="major")
                        fig.tight_layout()
                        fig.savefig(
                            model_dir / sig_key / year / f"{hkey}2_{key}_xbbcut{xbb_cut}_{year}.png"
                        )
                        fig.savefig(
                            model_dir / sig_key / year
                            / f"{hkey}2_{key}_xbbcut{xbb_cut}_{year}.pdf",
                            bbox_inches="tight",
                        )
                        plt.close()


def _get_bdt_scores(preds, sig_keys, multiclass):
    """Helper function to calculate which BDT outputs to use"""
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
    legacy,
):
    for i, sig_key in enumerate(sig_keys):

        (model_dir / sig_key ).mkdir(exist_ok=True, parents=True)
                
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
                sel = np.logical_or(sigs, bgs).to_numpy().squeeze()
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
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        ax.plot(
            rocs["train"]["tpr"],
            rocs["train"]["fpr"],
            linewidth=2,
            color="orange",
            label="Train Dataset",
        )
        ax.plot(
            rocs["test"]["tpr"],
            rocs["test"]["fpr"],
            linewidth=2,
            color="orange",
            label="Test Dataset",
        )
        ax.set_title(f"{plotting.label_by_sample[sig_key]} BDT ROC Curve from Training")
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel("Background efficiency")
        ax.set_xlim([0.0, 0.7])
        ax.set_ylim([1e-6, 1e-1])
        ax.set_yscale("log")
        ax.xaxis.grid(True, which="major")
        ax.yaxis.grid(True, which="major")

        legtitle = _get_title(legacy)

        ax.legend(
            title=legtitle,
            bbox_to_anchor=(1.03, 1),
            loc="upper left",
        )
        fig.tight_layout()
        fig.savefig(model_dir / sig_key / "roc_train_test.png")
        fig.savefig(model_dir / sig_key / "roc_train_test.pdf", bbox_inches="tight")

        h_bdt_weight = hist.Hist(bdt_axis, cat_axis)
        for key in training_keys:
            scores = model.predict_proba(X_test.loc[key])
            scores = _get_bdt_scores(scores, sig_keys, multiclass)[:, i]
            h_bdt_weight.fill(scores, key, weight=weights_test.loc[key])

        for key in training_keys:
            scores = model.predict_proba(X_train.loc[key])
            scores = _get_bdt_scores(scores, sig_keys, multiclass)[:, i]
            h_bdt_weight.fill(scores, key + "train", weight=weights_train.loc[key])

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
        for key in training_keys:
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
        fig.savefig(model_dir / "bdt_shape_traintest.pdf", bbox_inches="tight")


def main(args):
    # for year in ["2022", "2022EE", "2023", "2023BPix"]:
    year = "2022EE"
    events_dict = load_data(args.data_path, year, args.legacy)

    training_keys = args.sig_keys + args.bg_keys  # default: ["hh4b", "ttbar", "qcd"]

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
    ) = preprocess_data(
        events_dict, training_keys, args.config_name, args.test_size, args.seed, args.multiclass
    )

    model_dir = Path(f"./bdt_trainings_run3/{args.model_name}/")
    model_dir.mkdir(exist_ok=True, parents=True)

    (model_dir / "inferences" / year).mkdir(exist_ok=True, parents=True)

    for key in training_keys:
        np.save(f"{model_dir}/inferences/{year}/evt_{key}.npy", ev_test.loc[key])

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
            training_keys,
            model_dir,
            **classifier_params,
        )

        plot_train_test(
            X_train,
            y_train,
            yt_train,
            weights_train,
            X_test,
            y_test,
            yt_test,
            weights_test,
            model,
            args.multiclass,
            args.sig_keys,
            training_keys,
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
        args.sig_keys,
        args.bg_keys,
        training_keys,
        args.legacy,
    )

    # test in other years
    events_dict = {}
    years_test = ["2022EE"] if args.legacy else ["2022", "2022EE", "2023", "2023BPix"]
    for year in years_test:
        events_dict[year] = load_data(args.data_path, year, args.legacy)

    plot_allyears(
        events_dict,
        model,
        model_dir,
        args.config_name,
        args.multiclass,
        args.sig_keys,
        args.bg_keys,
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

    add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)
    add_bool_arg(parser, "multiclass", "Classify each background separately", default=True)
    add_bool_arg(parser, "legacy", "Legacy PNet versions", default=False)

    args = parser.parse_args()

    if args.config_name is None:
        args.config_name = args.model_name

    main(args)
