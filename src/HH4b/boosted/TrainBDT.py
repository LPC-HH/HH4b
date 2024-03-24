import argparse
import numpy as np
import xgboost as xgb
import importlib
import pandas as pd
from pathlib import Path
import pickle
import hist

import mplhep as hep
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12})
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.color"] = "#CCCCCC"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["figure.edgecolor"] = "none"

import sys
sys.path.append("./bdt_trainings_run3/")

from HH4b.utils import load_samples, format_columns
from HH4b.run_utils import add_bool_arg

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc


def load_data(data_path: str, year: str):
    """
    Load samples
    """

    # TODO: integrate all years

    samples = {
        "2022EE": {
            "qcd": [
                "QCD_HT-200to400",
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
                "TTtoLNu2Q",
            ],
            "hh4b": [
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV",
            ],
        }
    }

    dirs = {data_path: samples}

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

    events_dict = {}
    for input_dir, samples in dirs.items():
        events_dict = {
            **events_dict,
            **load_samples(
                input_dir,
                samples[year],
                year,
                filters=filters,
                columns_mc=format_columns(load_columns),
            ),
        }

    return events_dict


def preprocess_data(events_dict: dict, model_name: str, test_size: float, seed: int):
    # dataframe function
    make_bdt_dataframe = importlib.import_module(f"{model_name}")
    events_dict_bdt = {}
    weights_bdt = {}
    for key in events_dict:
        events_dict_bdt[key] = make_bdt_dataframe.bdt_dataframe(events_dict[key])
        # get absolute weights (no negative weights)
        weights_bdt[key] = np.abs(events_dict[key]["weight"].to_numpy()[:, 0])

    # concatenate data
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
    # Define weights
    weights_to_use = events["weight"]
    # Define features
    features = events
    features.drop(columns=["target"], inplace=True)

    # Split the (bdt dataframe) dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        # shuffle=False
    )
    weights_train = X_train["weight"].copy()
    X_train.drop(columns=["weight"], inplace=True)
    weights_test = X_test["weight"].copy()
    X_test.drop(columns=["weight"], inplace=True)

    return X_train, X_test, y_train, y_test, weights_train, weights_test


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

    print("Training model")
    model = xgb.XGBClassifier(**classifier_params)
    print("Training features: ", list(X_train.columns))
    
    trained_model = model.fit(
        X_train,
        y_train,
        sample_weight=weights_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        sample_weight_eval_set=[weights_train, weights_test],
        verbose=True,
    )
    trained_model.save_model(model_dir / "trained_bdt.model")

    # sorting by importance
    importances = model.feature_importances_
    feature_importances = sorted(zip(list(X_train.columns), importances), key=lambda x: x[1], reverse=True)
    feature_importance_df = pd.DataFrame.from_dict({"Importance": feature_importances})
    feature_importance_df.to_markdown(f"{model_dir}/feature_importances.md")

    return model

def evaluate_model(
    model_name: str,
    events_dict: dict,
    model: xgb.XGBClassifier,
    model_dir: Path,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    weights_test: np.ndarray,
    test_size: float, seed: int, year: str
):
    """
    1) Makes ROC curves for testing data
    2) Prints Sig efficiency at Bkg efficiency
    """
    # make and save ROCs for testing data
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    y_scores = model.predict_proba(X_test)[:, 1]

    print("Test ROC with sample weights")
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, sample_weight=weights_test)

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
        print(f"Signal efficiency: {tpr[idx]:.4f}, Background efficiency: {fpr[idx]:.5f}, BDT Threshold: {thresholds[idx]}")

    # ROC w/o weights
    print("Test ROC without sample weights")
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # print FPR, TPR for a couple of tprs
    for tpr_val in [0.10, 0.12, 0.15]:
        idx = find_nearest(tpr, tpr_val)
        print(f"Signal efficiency: {tpr[idx]:.4f}, Background efficiency: {fpr[idx]:.5f}, BDT Threshold: {thresholds[idx]}")

    # plot BDT scores for test samples
    make_bdt_dataframe = importlib.import_module(f"{model_name}")

    print("Perform inference on test signal sample")
    # get hh4b scores from full dataframe
    scores = {}
    hh4b_indices = X_test[X_test.index.get_level_values(0) == "hh4b"].index.get_level_values(1)
    # x_train, x_test = train_test_split(events_dict["hh4b"], test_size=test_size, random_state=seed)
    # hh4b_indices = x_test.index
    hh4b_test = events_dict["hh4b"].loc[hh4b_indices]
    hh4b_bdt_dataframe = make_bdt_dataframe.bdt_dataframe(hh4b_test)
    scores["hh4b"] = model.predict_proba(hh4b_bdt_dataframe)[:, 1]
    for key in ["qcd", "ttbar"]:
        scores[key] = model.predict_proba(make_bdt_dataframe.bdt_dataframe(events_dict[key]))[:, 1]
    print("Scores ", scores)
    print("HH4b  indices", hh4b_indices)

    # save scores and indices for testing dataset
    # TODO: add shifts (e.g. JECs etc)
    (model_dir / "inferences" / year).mkdir(exist_ok=True, parents=True)
    np.save(f"{model_dir}/inferences/{year}/preds.npy", scores["hh4b"])
    np.save(f"{model_dir}/inferences/{year}/indices.npy", hh4b_indices)

    bdt_axis = hist.axis.Regular(40, 0, 1, name="bdt", label=r"BDT")
    cat_axis = hist.axis.StrCategory([], name="cat", growth=True)
    h_bdt = hist.Hist(bdt_axis, cat_axis)
    for key in ["hh4b", "qcd", "ttbar"]:
        # TODO: add sample weight here...
        h_bdt.fill(bdt=scores[key], cat=key)

    colors = {
        "ttbar": "b",
        "hh4b": "k",
        "qcd": "r"
    }
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for key in ["hh4b", "qcd", "ttbar"]:
        hep.histplot(h_bdt[{"cat": key}], ax=ax, label=f"{key}", histtype="step", linewidth=1,
            color=colors[key],
            density=True,
        )
    ax.set_yscale('log')
    ax.legend(title=r"FatJets $p_T>$300, m$_{SD}$:[50-250] GeV")
    ax.set_ylabel("Density")
    ax.set_title("Pre-Selection")
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    fig.savefig(model_dir / "bdt_shape.png")


def main(args):
    events_dict = load_data(args.data_path, args.year)

    X_train, X_test, y_train, y_test, weights_train, weights_test = preprocess_data(
        events_dict, args.model_name, args.test_size, args.seed
    )

    model_dir = Path(f"./bdt_trainings_run3/{args.model_name}/")
    model_dir.mkdir(exist_ok=True, parents=True)

    classifier_params = {
        "max_depth": 17,
        "learning_rate": 0.1,
        "n_estimators": 196,
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

    evaluate_model(
        args.model_name,
        events_dict,
        model,
        model_dir,
        X_test,
        y_test,
        weights_test,
        args.test_size, args.seed, args.year
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
        "--year",
        required=True,
        help="year",
        type=str,
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="model name",
        type=str,
    )
    parser.add_argument("--test-size", default=0.4, help="testing/training split", type=float)
    parser.add_argument("--seed", default=42, help="seed for testing/training split", type=int)
    add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)
    args = parser.parse_args()

    main(args)
