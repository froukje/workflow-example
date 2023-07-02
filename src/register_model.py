import argparse
import mlflow
from mlflow.tracking import MlflowClient

remote_server_uri = "http://localhost:5000"
client = MlflowClient(remote_server_uri)


def main(args):
    run_id = args.run_id
    model_uri = f"{args.model_uri}/{run_id}/artifacts/models/"

    if args.register:
        client.create_registered_model(args.model_name)
        result = client.create_model_version(
            name=args.model_name,
            source=model_uri,
            run_id=run_id,
        )

    print("Registered Models:")
    print(client.get_registered_model(args.model_name))

    if args.change_stage:
        client.transition_model_version_stage(
            name=args.model_name, version=args.version, stage=args.stage
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", default=False)
    parser.add_argument("--change-stage", action="store_true", default=False)
    parser.add_argument(
        "--run-id", type=str, default="421eee19260b4da58c7080c7804706a1"
    )
    parser.add_argument(
        "--model-uri", type=str, default=f"s3://model-development/example/1/"
    )
    parser.add_argument("--model-name", type=str, default="random-forest")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=["None", "Staging", "Production"],
    )
    args = parser.parse_args()
    print("BEGIN argparse key - value pairs")
    params_dict = vars(args)
    for key, value in params_dict.items():
        print(f"{key}: {value}")
    print("END argparse key - value pairs")
    main(args)
