import os
import pandas as pd
import numpy as np
import random
import time
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
from optuna.integration.mlflow import MLflowCallback

import logging
import joblib
import s3fs

from utils import set_seed, upsample, print_feature_importances

RANDOM_STATE = 42
set_seed(RANDOM_STATE)

remote_server_uri = "http://localhost:5000"  # set server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflc = MLflowCallback(tracking_uri=remote_server_uri, metric_name="recall valid")

logger = logging.getLogger("water-potability")


@mlflc.track_in_mlflow()
def objective(trial):
    """hyperparamter tuning"""

    global best_model
    global max_score

    print("START TRAINING")
    current = time.time()

    if args.clf == "rf":
        n_estimators = trial.suggest_int(
            "n_estimators", args.n_estimators[0], args.n_estimators[1], log=True
        )
        max_depth = trial.suggest_categorical("max_depth", args.max_depth)
        min_samples_leaf = trial.suggest_categorical(
            "min_samples_leaf", args.min_samples_leaf
        )
        min_samples_split = trial.suggest_categorical(
            "min_samples_split", args.min_samples_split
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif args.clf == "xgb":
        n_estimators = trial.suggest_int(
            "n_estimators", args.n_estimators[0], args.n_estimators[1], log=True
        )
        eta = trial.suggest_float("eta", args.eta[0], args.eta[1], log=True)
        gamma = trial.suggest_float("gamma", args.gamma[0], args.gamma[1])
        alpha = trial.suggest_float("alpha", args.alpha[0], args.alpha[1])
        max_depth = trial.suggest_categorical("max_depth", args.max_depth)
        min_child_weight = trial.suggest_categorical(
            "min_child_weight", args.min_child_weight
        )
        colsample_bytree = trial.suggest_float(
            "colsample_bytree", args.colsample_bytree[0], args.colsample_bytree[1]
        )
        subsample = trial.suggest_float(
            "subsample", args.subsample[0], args.subsample[1]
        )

        parameters = {
            "objective": "binary:logistic",
            "n_estimators": n_estimators,
            "eta": eta,
            "gamma": gamma,
            "alpha": alpha,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "verbosity": 1,
        }
        model = xgb.XGBClassifier(**parameters)

    print(f"training {model.__class__.__name__}")
    model.fit(Xt, yt)
    pred_train = model.predict(Xt)
    pred_valid = model.predict(Xv)

    acc_train = accuracy_score(pred_train, yt)
    rec_train = recall_score(pred_train, yt)
    prec_train = precision_score(pred_train, yt)
    acc_valid = accuracy_score(pred_valid, yv)
    rec_valid = recall_score(pred_valid, yv)
    prec_valid = precision_score(pred_valid, yv)
    print(f"Accuracy on training set {acc_train:.4f}")
    print(f"Recall on training set {rec_train:.4f}")
    print(f"Precision on training set {prec_train:.4f}")
    print(f"Accuracy on validation set {acc_valid:.4f}")
    print(f"Recall on validation set {rec_valid:.4f}")
    print(f"Precision on validation set {prec_valid:.4f}")

    valid_score = rec_valid

    if valid_score < max_score:
        max_score = valid_score
        best_model = model

    print("END TRAINING")
    print(f"training took {time.time() - current:.3f}s")

    return valid_score


def apply_best_model(study, args):
    """apply best model from hyperparameter tuning to training an dvalidation data"""

    final_model = args.clf

    if final_model == "rf":
        best_model = RandomForestClassifier(
            n_estimators=study.best_params["n_estimators"],
            max_depth=study.best_params["max_depth"],
            min_samples_leaf=study.best_params["min_samples_leaf"],
            min_samples_split=study.best_params["min_samples_split"],
            n_jobs=-1,
        )

    if final_model == "xgb":
        parameters = {
            "objective": "reg:squarederror",
            "n_estimators": study.best_params["n_estimators"],
            "eta": study.best_params["eta"],
            "gamma": study.best_params["gamma"],
            "alpha": study.best_params["alpha"],
            "max_depth": study.best_params["max_depth"],
            "min_child_weight": study.best_params["min_child_weight"],
            "colsample_bytree": study.best_params["colsample_bytree"],
            "subsample": study.best_params["subsample"],
            "verbosity": 1,
        }

        best_model = xgb.XGBClassifier(**parameters)

    best_model.fit(Xt, yt)
    pred_train = best_model.predict(Xt)
    pred_valid = best_model.predict(Xv)

    # print feature importances if rf is used
    if final_model == "rf":
        print("\nfeature importances:")
        feature_names = args.input_data
        importances = best_model.feature_importances_
        print_feature_importances(feature_names, importances)

    if args.save:
        model_path = os.path.join(args.out_path, f"{best_model.__class__.__name__}.bin")
        joblib.dump(best_model, model_path)
    if args.log_model:
        mlflow.sklearn.log_model(best_model, "models")

    return pred_train, pred_valid


def main(args):
    start_time = time.time()
    print("read and prepare data ..")
    df = pd.read_csv(args.path)

    # fill nans
    df["ph"].fillna(value=df["ph"].mean(), inplace=True)
    df["Sulfate"].fillna(value=df["Sulfate"].mean(), inplace=True)
    df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].mean(), inplace=True)
    # train-valid-test split
    X = df.iloc[:, :-1]
    y = df["Potability"]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE, shuffle=True
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.11,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(f"Length training data: {len(X_train)}")
    print(f"Length validation data: {len(X_valid)}")
    print(f"Length test data: {len(X_test)}")

    if args.up:
        X_train, y_train = upsample(X_train, y_train)

    if args.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    print(f"reading and preparing data took: {time.time() - start_time:.4f} s")
    current = time.time()

    global best_model
    best_model = None
    global max_score
    max_score = -np.inf
    global Xt
    Xt = X_train
    global yt
    yt = y_train
    global Xv
    Xv = X_valid
    global yv
    yv = y_valid

    study = optuna.create_study(sampler=TPESampler(), direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=args.n_trials, callbacks=[mlflc])

    print("make predictions and calculate metrics for best model...")
    pred_train, pred_valid = apply_best_model(study, args)

    acc_train = accuracy_score(pred_train, y_train)
    rec_train = recall_score(pred_train, y_train)
    prec_train = precision_score(pred_train, y_train)
    acc_valid = accuracy_score(pred_valid, y_valid)
    rec_valid = recall_score(pred_valid, y_valid)
    prec_valid = precision_score(pred_valid, y_valid)
    print(f"Accuracy on training set {acc_train:.4f}")
    print(f"Recall on training set {rec_train:.4f}")
    print(f"Precision on training set {prec_train:.4f}")
    print(f"Accuracy on validation set {acc_valid:.4f}")
    print(f"Recall on validation set {rec_valid:.4f}")
    print(f"Precision on validation set {prec_valid:.4f}")

    mlflow.log_metric("acc_train", float(acc_train))
    mlflow.log_metric("rec_train", float(rec_train))
    mlflow.log_metric("prec_train", float(prec_train))
    mlflow.log_metric("acc_valid", float(acc_valid))
    mlflow.log_metric("rec_valid", float(rec_valid))
    mlflow.log_metric("prec_valid", float(prec_valid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/home/frauke/waterplan/workflow-example/data/water_potability.csv",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="output",  # "s3://model-development/example/",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default=[
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ],
        choices=[
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ],
    )
    parser.add_argument("--study-name", type=str, default="RandomForest")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--log-model", action="store_true", default=False)
    parser.add_argument("--up", action="store_true", default=False)
    parser.add_argument("--scale", action="store_true", default=False)
    parser.add_argument(
        "--n-trials", type=int, default=5, help="nr of trials for hyperparameter tuning"
    )
    parser.add_argument("--clf", type=str, default="rf", choices=["rf", "xgb"])
    parser.add_argument("--n-estimators", type=int, nargs="+", default=[8, 10])
    parser.add_argument("--max-depth", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--max-depth-none", action="store_true", default=False)
    parser.add_argument("--min-samples-leaf", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--min-samples-split", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--eta", type=float, nargs="+", default=[0.1, 0.5]
    )  # default 0.3
    parser.add_argument("--gamma", type=float, nargs="+", default=[0, 0])  # default=0
    parser.add_argument("--alpha", type=float, nargs="+", default=[0, 0])  # default=0
    parser.add_argument("--min-child_weight", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        nargs="+",
        default=[1, 1],
        help="lower to reduce overfitting",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        nargs="+",
        default=[1, 1],
        help="between 0 and 1, lower to reduce overfitting",
    )

    args = parser.parse_args()
    print("BEGIN argparse key - value pairs")
    params_dict = vars(args)
    for key, value in params_dict.items():
        print(f"{key}: {value}")
    print("END argparse key - value pairs")
    print()
    # None is added to max-depth (cannot be done directly -> type error)
    if args.max_depth_none:
        args.max_depth = args.max_depth + [None]

    #mlflow.set_experiment("RandomForest")
    #with mlflow.start_run(run_name="optuna-test") as run:
    #    mlflow.log_params(params_dict)
    #    main(args)
    main(args)
