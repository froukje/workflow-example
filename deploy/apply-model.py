import os
import argparse
import pandas as pd
import sklearn
import mlflow
from mlflow import MlflowClient

remote_server_uri = "http://localhost:5000"  # set server URI
mlflow.set_tracking_uri(remote_server_uri)

client = MlflowClient()

def main(args):
    model_name = args.model_name
    model_version = args.version
    run_id = args.run_id
    model_uri = f"{args.model_uri}/{run_id}/artifacts/models/"

    print("load model ...")
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    print("load data ...")
    df = pd.read_csv(args.path)
    # fill nans
    df["ph"].fillna(value=df["ph"].mean(), inplace=True)
    df["Sulfate"].fillna(value=df["Sulfate"].mean(), inplace=True)
    df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].mean(), inplace=True)
    X = df.iloc[:, :-1]
    print("make predictions ...")
    y_pred = model.predict(X)
    print(y_pred)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/home/frauke/waterplan/workflow-example/data/water_potability.csv",
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
    parser.add_argument(
        "--run-id", type=str, default="421eee19260b4da58c7080c7804706a1"
    )
    parser.add_argument(
        "--model-uri", type=str, default=f"s3://model-development/example/1"
    )
    parser.add_argument("--model-name", type=str, default="sk-learn-random-forest-reg-model")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    print("BEGIN argparse key - value pairs")
    params_dict = vars(args)
    for key, value in params_dict.items():
        print(f"{key}: {value}")
    print("END argparse key - value pairs")
    print()

    main(args)
