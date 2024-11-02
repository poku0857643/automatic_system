import argparse
import os

import mlflow

def main():
    parser = argparse.ArgumentParser(
        description="Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--commit_hash",
        type=str,
        default="000000",
        help="commit hash",
    )

    parser.add_argument(
        "--preprocess_data",
        type=str,
        default="mimiciv_waveform_0.1.0",
        help="mimiciv_waveform_0.1.0",
    )
    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="./preprocess/data/preprocess",
        help="preprocess downstream directory",
    )
    parser.add_argument(
        "--preprocess_cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )
    parser.add_argument(
        "--train_upstream",
        type=str,
        default="./preprocess/data/preprocess",
        help="upstream directory",
    )
    parser.add_argument(
        "--train_downstream",
        type=str,
        default="./train/data/model/",
        help="downstream directory",
    )
    parser.add_argument(
        "--train_tensorboard",
        type=str,
        default="./train/data/tensorboard/",
        help="tensrboard directory",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--train_num_workers",
        type=int,
        default=4,
        help="number of workers",
    )
    parser.add_argument(
        "--train_learning_rate",
        type=float,
        default=0.0001,
        help="learning rate",
    )
    parser.add_argument(
        "--train_model_type",
        type=str,
        default="randomforest",
        choices=["randomforest", "LSTM", "LLM"],
        help="random forest model, LSTM or LLM",
    )
    parser.add_argument(
        "--signal_type",
        type=str,
        choices=["ecg", "ppg", "both"],
        default="both",
        help="Type of signals to process: ech, ppg, or both",
    )

    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=62.5,
        help="sampling rate (Hz) of ECG and PPG",
    )
    parser.add_argument(
        "--signal_length",
        type=int,
        choices=[5, 10, 30, 60],
        default=30,
        help="Length of the signal segments in seconds",
    )
    parser.add_argument(
        "--building_dockerfile_path",
        type=str,
        default="./Dockerfile",
        help="building Dockerfile path",
    )
    parser.add_argument(
        "--building_model_filename",
        type=str,
        default="mimicivwvform010.onnx",
        help="building model file name",
    )
    parser.add_argument(
        "--building_entrypoint_path",
        type=str,
        default="./onnx_runtime_server_entrypoint.sh",
        help="building entrypoint path",
    )

    parser.add_argument(
        "--evaluate_downstream",
        type=str,
        default="./data/evaluate/",
        help="evaluate downstream directory",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    with mlflow.start_run() as r:
        preprocess_run = mlflow.run(
            uri="./preprocess",
            entry_point="preprocess",
            backend="local",
            parameters={
                "data": args.preprocess_data,
                "downstream": args.preprocess_downstream,
                "cached_data_id": args.preprocess_cached_data_id,
            },
        )
        preprocess_run = mlflow.tracking.MLflowClient().get_run(preprocess_run.run_id)

        dataset = os.path.join(
            "tmp/mlruns/",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/downstream_directory",
        )

        train_run = mlflow.run(
            uri="./train",
            entry_point="train",
            backend="local",
            parameters={
                "upstream": dataset,
                "downstream": args.train_downstream,
                "tensorboard": args.train_tensorboard,
                "epochs":args.train_epochs,
                "batch_size":args.train_batch_size,
                "num_workers": args.train_num_workers,
                "learning_rate": args.train_learning_rate,
                "model_type": args.train_model_type,
                "signal_type": args.signal_type,
                "sampling_rate": args.sampling_rate,
                "signal_length": args.signal_length,
            },
        )
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)

        building_run = mlflow.run(
            uri="./building",
            entry_point="building",
            backend="local",
            parameters={
                "dockerfile_path": args.building_dockerfile_path,
                "model_filename": args.building_model_filename,
                "model_directory": os.path.join(
                    "mlruns/",
                    str(mlflow_experiment_id),
                    train_run.info.run_id,
                    "artifacts",
                ),
                "entrypoint_path": args.building_entrypoint_path,
                "dockerimage": f"eshan75/automatic_system: training_signal_mimiciv010_evaluate_{mlflow_experiment_id}",
            },
        )
        building_run = mlflow.tracking.MlflowClient().get_run(building_run,run_id)

        evaluate_run = mlflow.run(
            uri="./evaluate",
            entry_point="evaluate",
            backend="local",
            parameters={
                "upstream": os.path.join(
                    "../mlruns/",
                    str(mlflow_experiment_id),
                    train_run.info.run_id,
                    "artifacts",
                ),
                "downstream": args.evaluate_downstream,
                "testdata_directory": os.path.join(
                    "../mlruns/",
                    str(mlflow_experiment_id),
                    preprocess_run.info.run_id,
                    "artifacts/downstream_directory/test",
                ),
                "dockerimage": f"eshan75/automatic_system: training_signal_mimiciv010_evaluate_{mlflow_experiment_id}",
                "container_name": f"training_signal_mimiciv010_evaluate_{mlflow_experiment_id}",
            },
        )
        evaluate_run = mlflow.tracking.MlflowClient().get_run(evaluate_run.run_id)