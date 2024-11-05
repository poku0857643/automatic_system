import logging
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class mimiciv_wvformDataset(Dataset):
    def __init__(self, data_directory, transform):
        super().__init__()
        self.data_directory = data_directory
        self.transform = transform ##
        self.signal_array_list = []
        self.label_list = []
        self.__load_signal_files_and_labels()  ##

    def __len__(self):
        return len(self.signal_array_list)

    def __getitem__(self, index):
        signal_array = self.signal_array_list[index]
        signal_tensor = self.transform(signal_array) # transform
        label = self.label_list[index]

        return signal_tensor, label

    def __load_signal_files_and_labels(self):
        # Assuming subdirectories correspond to different sets of signal files
        # and that each set has associated SBP and DBP values as targets
        target_directories = [i for i in os.listdir(self.data_directory) if i.isdecimal()]
        filepath_list = []

        for d in target_directories:
            _d = os.path.join(self.data_directory, d)
            filepath_list.extend([os.path.join(_d, f) for f in os.listdir(_d)])

            # Extend target_list with SBP and DBP values loaded from metadata (modify as needed)
            # For example, assume SBP and DBP values are stored in a .txt file in each directory:
            for _ in os.listdir(_d):
                sbp, dbp = self._load_targets_for_directory(_d)  # implement this helper method as needed
                self.label_list.append((sbp, dbp))

        # Process each signal file and add it to signal_array_list
        for fp in filepath_list:
            # Here, assume each file contains ECG and PPG signals in a suitable format (e.g., .npy or .csv)
            signal_data = np.load(fp)  # or use an appropriate loading function for your signal data format
            self.signal_array_list.append(signal_data)

        logger.info(f"Loaded: {len(self.label_list)} data entries for regression")

    def _load_targets_for_directory(self, directory):

class ramdomforest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=0, regression=True):
        # Initialize a random forest model; can be used for regression or classification
        if regression:
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        else:
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Define the output layer (fully connected)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the last hidden state
        out = self.fc(out[:, -1, :])  # Only take the output from the last time step
        return out


class LLaMA32RegressionModel(nn.Module):
    def __init__(self, model_name="llama3.2"):
        super(LLaMA32RegressionModel, self).__init__()

        # Load the LLaMA 3.2 pretrained model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Modify the output layer for regression (SBP and DBP)
        # Assuming the hidden size of LLaMA's last layer is model.config.hidden_size
        self.regression_head = nn.Linear(self.model.config.hidden_size, 2)  # Predicts SBP and DBP

    def forward(self, input_ids, attention_mask=None):
        # Get the hidden states from the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, 0, :]  # Using CLS token representation

        # Pass the hidden state through regression head
        return self.regression_head(last_hidden_state)


class GemmaRegressionModel(nn.Module):
    def __init__(self, model_name="gemma"):
        super(GemmaRegressionModel, self).__init__()

        # Load the Gemini pretrained model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Modify the output layer for regression (SBP and DBP)
        self.regression_head = nn.Linear(self.model.config.hidden_size, 2)  # Predicts SBP and DBP

    def forward(self, input_ids, attention_mask=None):
        # Get the hidden states from the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, 0, :]  # Using CLS token representation

        # Pass the hidden state through regression head
        return self.regression_head(last_hidden_state)


from sklearn.metrics import mean_squared_error, accuracy_score


def train_random_forest(model, train_data, train_labels):
    model.fit(train_data, train_labels)
    logger.info("Random Forest model trained.")


def evaluate_random_forest(model, test_data, test_labels, writer, epoch):
    predictions = model.predict(test_data)
    loss = mean_squared_error(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions.round())

    writer.add_scalar("Loss/test", loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy * 100, epoch)
    logger.info(f"Accuracy: {accuracy * 100}%, Loss: {loss}")
    return accuracy * 100, loss


def evaluate_lstm(model, test_dataloader, criterion, writer, epoch, device="cpu"):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            sequences, labels = data[0].to(device), data[1].to(device)
            outputs = model(sequences)
            total += labels.size(0)
            total_loss += criterion(outputs, labels)

    loss = total_loss / total
    writer.add_scalar("Loss/test", loss, epoch)
    logger.info(f"Test Loss: {loss}")
    return None, float(loss)  # LSTM regression; returning loss only


def train_lstm(model, train_dataloader, test_dataloader, criterion, optimizer, writer, epochs=10,
               checkpoints_directory="/opt/lstm/model/", device="cpu"):
    logger.info("start training...")
    for epoch in range(epochs):
        running_loss = 0.0
        logger.info(f"starting epoch: {epoch}")
        for i, data in enumerate(train_dataloader, 0):
            sequences, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar("Loss/train", running_loss / (i + 1), epoch * len(train_dataloader) + i)
        _, loss = evaluate_lstm(model, test_dataloader, criterion, writer, epoch, device)
        torch.save(model.state_dict(), os.path.join(checkpoints_directory, f"epoch_{epoch}_loss_{loss:.4f}.pth"))


def evaluate_llm(model, test_dataloader, criterion, writer, epoch, device="cpu"):
    total_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data['input_ids'].to(device), data['labels'].to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels)

    loss = total_loss / len(test_dataloader)
    writer.add_scalar("Loss/test", loss, epoch)
    logger.info(f"Test Loss: {loss}")
    return None, float(loss)  # For regression tasks


def train_llm(model, train_dataloader, test_dataloader, criterion, optimizer, writer, epochs=10,
              checkpoints_directory="/opt/llm/model/", device="cpu"):
    logger.info("start training...")
    for epoch in range(epochs):
        running_loss = 0.0
        logger.info(f"starting epoch: {epoch}")
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data['input_ids'].to(device), data['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar("Loss/train", running_loss / (i + 1), epoch * len(train_dataloader) + i)
        _, loss = evaluate_llm(model, test_dataloader, criterion, writer, epoch, device)
        torch.save(model.state_dict(), os.path.join(checkpoints_directory, f"epoch_{epoch}_loss_{loss:.4f}.pth"))
