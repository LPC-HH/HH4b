from __future__ import annotations

import argparse
import importlib
import pickle
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from HH4b import hh_vars, plotting
from HH4b.boosted.TrainBDT import (
    _get_bdt_scores,
    evaluate_model,
    get_legtitle,
    roc_curve,
)
from HH4b.hh_vars import samples_run3
from HH4b.postprocessing import (
    get_evt_testing,
    load_run3_samples,
)

# definitions
bdt_axis = hist.axis.Regular(40, 0, 1, name="bdt", label=r"BDT")


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
    print("Training keys ", training_keys)
    for key in training_keys:
        if key not in events_dict:
            print(f"removing {key}")
            training_keys.remove(key)

    print("events dict ", events_dict.keys())

    print("Train keys ", training_keys)

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
        print(f"Total {key} pre-normalization: {np.sum(weights_bdt[key]):.3f}")

    # weights
    if run2_wapproach:
        print("Norm weights")
        bkg_weight = np.concatenate([weights_bdt[key] for key in args.bg_keys])
        bkg_weight_min = np.amin(np.absolute(bkg_weight))
        bkg_weight_rescale = 1.0 / np.absolute(bkg_weight_min)
        print("Background weight rescale ", bkg_weight_rescale)
        for key in args.bg_keys:
            weights_bdt[key][weights_bdt[key] < 0] = 0
            events.loc[key, "weight"] = weights_bdt[key] * bkg_weight_rescale

        for sig_key in args.sig_keys:
            sig_weight = weights_bdt[sig_key]
            sig_weight_min = np.amin(np.absolute(sig_weight))
            sig_weight_rescale = 1.0 / np.absolute(sig_weight_min)
            print("Signal weight rescale ", sig_weight_rescale)
            sig_weight[sig_weight < 0] = 0
            events.loc[sig_key, "weight"] = sig_weight * sig_weight_rescale

        for key in training_keys:
            print(f"Total {key} post-normalization: {events.loc[key, 'weight'].sum():.3f}")

    if equalize_weights:
        print("Equalize weights")
        # scales signal such that total signal = total background
        bkg_total = np.sum([np.sum(weights_bdt[key]) for key in args.bg_keys])

        num_sigs = len(args.sig_keys)
        for sig_key in args.sig_keys:
            if sig_key not in weights_bdt:
                continue
            sig_total = np.sum(weights_bdt[sig_key])
            print(f"Scaling {sig_key} by {bkg_total / sig_total} / {num_sigs} signal(s).")
            events.loc[sig_key, "weight"] = (
                weights_bdt[sig_key] * (bkg_total / sig_total) / num_sigs
            )

        for key in args.bg_keys:
            events.loc[key, "weight"] = weights_bdt[key]

        for key in training_keys:
            print(f"Total {key} post-normalization: {events.loc[key, 'weight'].sum():.3f}")

    # Define target
    events["target"] = 0  # Default to 0 (background)
    for key in args.sig_keys:
        if key in training_keys:
            events.loc[key, "target"] = 1  # Set to 1 for 'hh4b' samples (signal)

    target = events["target"]
    simple_target = events["target"]

    print("multiclass labeling")
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

    for key in training_keys:
        print(f"Total training {key} after splitting: {weights_train.loc[key].sum():.3f}")

    for key in training_keys:
        print(f"Total testing {key} after splitting: {weights_test.loc[key].sum():.3f}")

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


# following may not be necessary
"""
model_dir = (
    "/home/users/dprimosc/HH4b/src/HH4b/boosted/bdt_trainings_run3/24May31_lr_0p02_md_8_AK4Away"
)
config_name = "24May31_lr_0p02_md_8_AK4Away"


def load_model(model_path: Path) -> xgb.XGBClassifier:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model(model_dir)

"""


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
    pnet_plots: bool,
    legacy: bool,
    pnet_xbb_str: str,
    pnet_mass_str: str,
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

        print(f"Evaluating {sig_key} performance")

        if multiclass:
            # selecting only this signal + BGs for ROC curves
            bgs = y_test >= len(sig_keys)
            sigs = y_test == i
            sel = np.logical_or(sigs, bgs).squeeze()
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
        with (plot_dir / sig_key / "roc_dict.pkl").open("wb") as f:
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

        print("Performing inference on all samples")
        # get scores from full dataframe, but only use testing indices
        scores = {}
        weights = {}
        mass_dict = {}
        msd_dict = {}
        xbb_dict = {}

        for key in training_keys:
            score = []
            weight = []
            mass = []
            msd = []
            xbb = []
            for year in events_dict_years:
                evt_list = get_evt_testing(f"{model_dir}/inferences/{year}", key)
                if evt_list is None:
                    continue

                events = events_dict_years[year][key]
                bdt_events = make_bdt_dataframe.bdt_dataframe(events)
                test_bdt_dataframe = bdt_events.copy()
                bdt_events["event"] = events["event"].to_numpy()[:, 0]
                bdt_events["finalWeight"] = events["finalWeight"]
                bdt_events["mass"] = events[pnet_mass_str][1]
                bdt_events["msd"] = events["bbFatJetMsd"][1]
                bdt_events["xbb"] = events[pnet_xbb_str][1]
                mask = bdt_events["event"].isin(evt_list)
                test_dataset = bdt_events[mask]

                test_bdt_dataframe = test_bdt_dataframe[mask]
                test_preds = model.predict_proba(test_bdt_dataframe)

                score.append(_get_bdt_scores(test_preds, sig_keys, multiclass)[:, i])
                weight.append(test_dataset["finalWeight"])
                mass.append(test_dataset["mass"])
                msd.append(test_dataset["msd"])
                xbb.append(test_dataset["xbb"])

            scores[key] = np.concatenate(score)
            weights[key] = np.concatenate(weight)
            mass_dict[key] = np.concatenate(mass)
            msd_dict[key] = np.concatenate(msd)
            xbb_dict[key] = np.concatenate(xbb)

        for key in events_dict_years[year]:
            if key in training_keys:
                continue
            score = []
            weight = []
            xbb = []
            for year in events_dict_years:
                preds = model.predict_proba(
                    make_bdt_dataframe.bdt_dataframe(events_dict_years[year][key])
                )
                score.append(_get_bdt_scores(preds, sig_keys, multiclass)[:, i])
                weight.append(events_dict_years[year][key]["finalWeight"])
                xbb.append(events_dict_years[year][key][pnet_xbb_str][1])
                msd.append(events_dict_years[year][key]["bbFatJetMsd"][1])
                mass.append(events_dict_years[year][key][pnet_mass_str][1])
            scores[key] = np.concatenate(score)
            weights[key] = np.concatenate(weight)
            xbb_dict[key] = np.concatenate(xbb)
            msd_dict[key] = np.concatenate(msd)
            mass_dict[key] = np.concatenate(mass)

        print("Making BDT shape plots")

        legtitle = get_legtitle(legacy, pnet_xbb_str)

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

        print("Making ROC Curves")

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


def main(args):
    training_keys = args.sig_keys + args.bg_keys  # default: ["hh4b", "ttbar", "qcd"]

    model_dir = Path(f"./bdt_trainings_run3/{args.model_name}/")
    model_dir.mkdir(exist_ok=True, parents=True)

    if args.year == "2022-2023":
        years = ["2022", "2022EE", "2023", "2023BPix"]
    else:
        years = args.year

    print(years)

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
    for year in years:
        for key in list(samples_run3[year].keys()):
            if key not in training_keys + aux_keys:
                samples_run3[year].pop(key)

    for year in years:
        print("loading ", year)
        events_dict_years[year] = load_run3_samples(
            args.data_path,
            year,
            args.legacy,
            samples_run3,
            reorder_txbb=True,
            txbb=args.pnet_xbb_str,
        )

        if args.apply_cuts:
            # apply cuts
            events_dict_years[year] = apply_cuts(
                events_dict_years[year], args.pnet_xbb_str, args.pnet_mass_str, args.legacy
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
            args.legacy,
            args.pnet_xbb_str,
        )

    # no combination for now
    # events_dict, _ = combine_run3_samples(events_dict_years, training_keys, scale_processes = {"hh4b": ["2022EE", "2023"]}, years_run3=years)

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
        args.pnet_plots,
        args.legacy,
        args.pnet_xbb_str,
        args.pnet_mass_str,
    )

    # test in other years
    events_dict = {}
    years_test = ["2022", "2022EE", "2023", "2023BPix"]
    for year in years_test:
        events_dict[year] = load_run3_samples(
            args.data_path,
            year,
            args.legacy,
            samples_run3,
            reorder_txbb=True,
            txbb=args.pnet_xbb_str,
        )
        if args.apply_cuts:
            events_dict[year] = apply_cuts(
                events_dict[year], args.pnet_xbb_str, args.pnet_mass_str, args.legacy
            )

    plot_allyears(
        events_dict,
        model,
        model_dir,
        args.config_name,
        args.multiclass,
        args.sig_keys,
        args.bg_keys,
        args.pnet_xbb_str,
        args.pnet_mass_str,
    )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        nargs="+",
        type=str,
        default=["2022EE"],
        choices=hh_vars.years,
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
        "--xbb",
        choices=["bbFatJetPNetTXbb", "bbFatJetPNetTXbbLegacy"],
        help="xbb branch",
        required=True,
    )
    parser.add_argument(
        "--mass",
        choices=["bbFatJetPNetMass", "bbFatJetPNetMassLegacy", "bbFatJetMsd"],
        help="xbb pnet mass",
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

    add_bool_arg(parser, "legacy", "Legacy PNet versions", default=False)
    add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)
    add_bool_arg(parser, "multiclass", "Classify each background separately", default=True)
    add_bool_arg(
        parser, "equalize-weights", "Equalise total signal and background weights", default=True
    )
    add_bool_arg(parser, "run2-wapproach", "Run2 weight approach", default=False)
    add_bool_arg(parser, "pnet-plots", "Make PNet plots", default=True)
    add_bool_arg(parser, "apply-cuts", "Apply cuts", default=True)

    args = parser.parse_args()
    args.pnet_xbb_str = args.xbb
    args.pnet_mass_str = args.mass

    if args.config_name is None:
        args.config_name = args.model_name

    main(args)
