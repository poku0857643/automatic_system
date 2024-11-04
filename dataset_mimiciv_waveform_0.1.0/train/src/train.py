import argparse
import os
import logging

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from src.constants import MODEL_ENUM
from src.model import (randomforest, mimiciv_wvform_dataset010, SimpleModel,evaluate, train)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from torchvision import transformers

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=tensorboard_directory)

    # transformer = transforms.Compose(  # image transformers
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    train_dataset = mimiciv_wvform_dataset010(
        data_dir=os.path.join(upstream_directory, "data"),
        # transform=transform,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = mimiciv_wvform_dataset010(
        data_dir=os.path.join(downstream_directory, "data"),
        # transform=transform,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if model_type == MODEL_ENUM.randomforest.value:
        model = randomforest.RandomForest().to(device)
    elif model_type == MODEL_ENUM.LSTM.value:
        model = nn.LSTM().to(device)
    elif model_type == MODEL_ENUM.LLM.value:
        model = LLM(dim=1)
    else:
        raise ValueError("Unknown model or model type not supported.")
    model.eval()

    mlflow.pytorch.log_model(model, "model")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=epochs,
        device=device,
        writer=writer,
        checkpoints_directory = downstream_directory,
    )

    accuracy, loss = evaluate(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        criterion=criterion,
        writer=writer,
        epochs=epochs+ 1
    )
    logger.info(f"Latest performance: Accuracy: {accuracy}, Loss: {loss}")

    writer.close()

    model_file_name = os.path.join(
        downstream_directory, f"mimiciv_waveform_dataset010_{mlflow_experiment_id}.pth",
    )
    onnx_file_name = os.path.join(
        downstream_directory, f"mimiciv_waveform_dataset010_{mlflow_experiment_id}.onnx"
    )

    torch.save(model.state_dict(), model_file_name)

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_name,
        verbose=True,
        input_names=["input"],
        output_names=["output"]
    )

    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param(
        "preprocess",
        "Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))",
    )
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_workers", num_workers)
    mlflow.log_param("device", device)
    mlflow.log_param("model_type", model_type)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)
    mlflow.log_artifact(onnx_file_name)
    mlflow.log_artifact(model_file_name)
    mlflow.log_artifacts(tensorboard_directory, artifact_path="tensorboard")

def main():
    parser = argparse.ArgumentParser(
        description="Train mimiciv_waveform_0.1.0",
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
        default="/opt/mimiciv_waveform_0.1.0/",
        help="downstream directory",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default="/opt/mimiciv_waveform_0.1.0/tensorboard",
        help="tensorboard directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_ENUM.randomforest.value,
        choices=[
            MODEL_ENUM.randomforest.value,
            MODEL_ENUM.LSTM.value,
            MODEL_ENUM.LLM.value,
        ],
        help="model type: randomforest, lstm, llm",
    )
    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = args.upstream
    downstream_directory = args.downstream
    tensorboard_directory = args.tensorboard
    os.makedirs(downstream_directory, exist_ok=True)
    os.makedirs(upstream_directory, exist_ok=True)

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
