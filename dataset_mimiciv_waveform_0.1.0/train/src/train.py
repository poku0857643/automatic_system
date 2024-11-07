import argparse
import logging
import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from constants import MODEL_ENUM
from model import randomforest, LSTM, llama3, gllm, evalulte, train
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_run(
        mlflow_experiment_id: str,
        upstream_directory: str,
        downstream_directory: str,
        tensorboard_directory: str,
        batch_size: int,
        num_workers: int,
        epochs: int,
        learning_rate: float,
        model_type: str,
):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )
        writer = SummaryWriter(log_dir=tensorboard_directory)

        transform = transforms.Compose( # modify to data preprocessing steps
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_dataset = mimiciv_wvformDataset(
            data_directory=os.path.join(upstream_directory, "train"),
            transform = transform, # modify to data preprocessing steps
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        test_dataset = mimiciv_wvformDataset(
            data_directory=os.path.join(upstream_directory, "test"),
            transform = transform, # modifu to data preprocessing steps
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
#
#         if model_type == MODEL_ENUM.randomforest.value:
#             model = randomforest().to.device()
#         elif model_type == MODEL_ENUM.lstm.value:
#             model = LSTM().to.device()
#         elif model_type == MODEL_ENUM.llama3.value:
#             model = llama3().to.device()
#         elif model_type == MODEL_ENUM.gllm.value:
#             model = gllm().to.device()
#         else:
#             raise ValueError("Unknown model type")
#
#         mlflow.pytorch.log_model(model, "model")
#
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#         train(
#             model=model,
#             train_dataloader=train_dataloader,
#             test_dataloader=test_dataloader,
#             criterion=criterion,
#             optimizer=optimizer,
#             epochs=epochs,
#             device=device,
#             writer=writer,
#             checkpoints_directory=downstream_directory,
#         )
#
#         accuracy, loss = evaluate(
#             model=model,
#             device=device,
#             test_dataloader=test_dataloader,
#             writer=writer,
#             epochs=epochs + 1,
#             criterion=criterion,
#         )
#         logger.info(f"Latest performance: Accuracy: {accuracy}, Loss: {loss}")
#
#         writer.close()
#
#         model_file_name = os.path.join(
#             downstream_directory,
#             f"mimiciv_wvform010_{mlflow_experiment_id}.pth"
#         )
#         onnx_file_name = os.path.join(
#             downstream_directory,
#             f"mimiciv_wvform010_{mlflow_experiment_id}.onnx"
#         )
#
#         torch.save(model.state_dict(), model_file_name)
#
#         dummy_input = torch.randn(1,3,32,32)
#         torch.onnx.export(
#             model, dummy_input,
#             onnx_file_name,
#             verbose=True,
#             input_names=["input"],
#             output_names=["output"],
#         )
#
#         mlflow.log_param("optimizer", "Adam")
#         mlflow.log_param(
#             "preprocess",
#             "Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))",
#         )
#         mlflow.log_param("learning_rate", learning_rate)
#         mlflow.log_param("batch_size", batch_size)
#         mlflow.log_param("epochs", epochs)
#         mlflow.log_param("device", device)
#         mlflow.log_param("num_workers", num_workers)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("loss", loss)
#         mlflow.log_artifact(model_file_name)
#         mlflow.log_artifact(onnx_file_name)
#         mlflow.log_artifacts(tensorboard_directory, artifact_path="tensorboard")
#
        # Model selection and setup
        if model_type == MODEL_ENUM.randomforest.value:
            model = randomforest()
            model.fit(train_dataset.data, train_dataset.labels)
            accuracy, loss = evaluate_random_forest(
                model=model,
                test_data=test_dataset.data,
                test_labels=test_dataset.labels,
                writer=writer,
                epoch=epochs + 1
            )

        elif model_type == MODEL_ENUM.lstm.value:
            model = LSTM().to(device)
            criterion = nn.MSELoss()  # Assuming regression
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_lstm(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                device=device,
                writer=writer,
                checkpoints_directory=downstream_directory,
            )

            accuracy, loss = evaluate_lstm(
                model=model,
                test_dataloader=test_dataloader,
                criterion=criterion,
                writer=writer,
                epoch=epochs + 1,
                device=device,
            )

        elif model_type == MODEL_ENUM.llama3.value:
            model = llama3().to(device)
            criterion = nn.MSELoss()  # Assuming regression
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

            train_llm(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                device=device,
                writer=writer,
                checkpoints_directory=downstream_directory,
            )

            accuracy, loss = evaluate_llm(
                model=model,
                test_dataloader=test_dataloader,
                criterion=criterion,
                writer=writer,
                epoch=epochs + 1,
                device=device,
            )

        elif model_type == MODEL_ENUM.gllm.value:
            model = gllm().to(device)
            criterion = nn.MSELoss()  # Assuming regression
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

            train_llm(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                device=device,
                writer=writer,
                checkpoints_directory=downstream_directory,
            )

            accuracy, loss = evaluate_llm(
                model=model,
                test_dataloader=test_dataloader,
                criterion=criterion,
                writer=writer,
                epoch=epochs + 1,
                device=device,
            )

        else:
            raise ValueError("Unknown model type")

        logger.info(f"Latest performance: Accuracy: {accuracy}, Loss: {loss}")

        writer.close()

        # Saving model artifacts based on model type
        if model_type != MODEL_ENUM.randomforest.value:
            model_file_name = os.path.join(
                downstream_directory,
                f"mimiciv_wvform010_{mlflow_experiment_id}.pth"
            )
            torch.save(model.state_dict(), model_file_name)
            mlflow.pytorch.log_model(model, "model")

            # Export model to ONNX
            dummy_input = torch.randn(1, 3, 32, 32)  # Update as per input shape
            onnx_file_name = os.path.join(
                downstream_directory,
                f"mimiciv_wvform010_{mlflow_experiment_id}.onnx"
            )
            torch.onnx.export(
                model, dummy_input,
                onnx_file_name,
                verbose=True,
                input_names=["input"],
                output_names=["output"],
            )
            mlflow.log_artifact(model_file_name)
            mlflow.log_artifact(onnx_file_name)

        # Log parameters and metrics for MLflow
        mlflow.log_param("optimizer",
                         "AdamW" if model_type in [MODEL_ENUM.llama3.value, MODEL_ENUM.gllm.value] else "Adam")
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("device", device)
        mlflow.log_param("num_workers", num_workers)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)
        mlflow.log_artifacts(tensorboard_directory, artifact_path="tensorboard")

#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Train mimiciv-waveform010",
#         formatter_class=argparse.RawTextHelpFormatter,
#     )
#     parser.add_argument(
#         "--upstream",
#         type=str,
#         default="/opt/data/preprocess",
#         help="upstream directory",
#     )
#     parser.add_argument(
#         "--downstream",
#         type=str,
#         default="/opt/mimiciv-waveform010/model",
#         help="downstream directory",
#     )
#     parser.add_argument(
#         "--tensorboard",
#         type=str,
#         default="/opt/mimiciv-waveform010/tensorboard",
#         help="tensorboard directory",
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=10,
#         help="number of epochs",
#     )
#     parser.add_argument(
#         "--learning_rate",
#         type=float,
#         default=0.001,
#         help="learning rate",
#     )
#     parser.add_argument(
#         "--model_type",
#         type=str,
#         default=MODEL_ENUM.ranodmforest.value,
#         choices=[MODEL_ENUM.randomforest.value, MODEL_ENUM.llama3.value,
#                  MODEL_ENUM.gllm.value, MODEL_ENUM.LSTM.value],
#         help="model type: randomforest, llama3, gllm, lstm",
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=32,
#         help="batch size",
#     )
#     parser.add_argument(
#         "--num_workers",
#         type=int,
#         default=4,
#         help="number of workers",
#     )
#     args = parser.parse_args()
#     mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))
#
#     upstream_directory = args.upstream
#     downstream_directory  = args.downstream
#     tensorboard_directory = args.tensorboard
#     os.makedirs(tensorboard_directory, exist_ok=True)
#     os.makedirs(downstream_directory, exist_ok=True)
#
#     start_run(
#         mlflow_experiment_id=mlflow_experiment_id,
#         upstream_directory=upstream_directory,
#         downstream_directory=downstream_directory,
#         tensorboard_directory=tensorboard_directory,
#         batch_size=args.batch_size,
#         epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         model_type=args.model_type,
#         num_workers=args.num_workers,
#     )

def main():
    parser = argparse.ArgumentParser(
        description="Train mimiciv-waveform010",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--upstream",
        type=str,
        default="/opt/data/preprocess",
        help="upstream directory",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/mimiciv-waveform010/model",
        help="downstream directory",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default="/opt/mimiciv-waveform010/tensorboard",
        help="tensorboard directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_ENUM.randomforest.value,  # Corrected typo here
        choices=[
            MODEL_ENUM.randomforest.value,
            MODEL_ENUM.llama3.value,
            MODEL_ENUM.gllm.value,
            MODEL_ENUM.lstm.value
        ],
        help="model type: randomforest, llama3, gllm, lstm",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    # Setup directories
    upstream_directory = args.upstream
    downstream_directory = args.downstream
    tensorboard_directory = args.tensorboard
    os.makedirs(tensorboard_directory, exist_ok=True)
    os.makedirs(downstream_directory, exist_ok=True)

    # Start the run with the parsed arguments
    start_run(
        mlflow_experiment_id=mlflow_experiment_id,
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        tensorboard_directory=tensorboard_directory,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
