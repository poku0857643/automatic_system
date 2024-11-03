import os
import pickle
from typing import List, Tuple, Dict
import json

import numpy as np
import wfdb  # Use wfdb for reading MIMIC-IV waveform data


def unpickle(file: str) -> Dict:
    with open(file, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return data_dict


def calculate_features(ecg_signal: np.ndarray, ppg_signal: np.ndarray, abp_signal: np.ndarray) -> Dict[str, float]:
    # Example feature calculations based on the signals
    features = {
        "ecg_mean": np.mean(ecg_signal),
        "ecg_std": np.std(ecg_signal),
        "ppg_mean": np.mean(ppg_signal),
        "ppg_std": np.std(ppg_signal),
        "abp_mean": np.mean(abp_signal),
        "abp_std": np.std(abp_signal),
    }
    return features


def calculate_sbp_dbp(abp_signal: np.ndarray) -> Tuple[float, float]:
    # Calculate SBP (systolic) and DBP (diastolic) from the ABP signal
    sbp = np.max(abp_signal)
    dbp = np.min(abp_signal)
    return sbp, dbp


def parse_signals(rawdata: dict, rootdir: str) -> Tuple[List[List[str]], List[float], List[float]]:
    # Create directories for processed ECG, PPG, and ABP data
    os.makedirs(rootdir, exist_ok=True)

    class_to_filename_list = []
    sbp_values = []
    dbp_values = []

    for i in range(len(rawdata["filenames"])):
        filename = rawdata["filenames"][i]
        ecg_signal = rawdata["ecg"][i]  # Placeholder for ECG signal data
        ppg_signal = rawdata["ppg"][i]  # Placeholder for PPG signal data
        abp_signal = rawdata["abp"][i]  # Placeholder for ABP signal data

        # Calculate features and SBP/DBP values
        features = calculate_features(ecg_signal, ppg_signal, abp_signal)
        sbp, dbp = calculate_sbp_dbp(abp_signal)

        # Store the feature data in a text file or JSON
        feature_file = os.path.join(rootdir, f"{filename}_features.json")
        with open(feature_file, "w") as f:
            json.dump(features, f)

        # Store the calculated SBP and DBP values
        sbp_values.append(sbp)
        dbp_values.append(dbp)

        # Track filenames by class for metadata
        class_to_filename_list.append([filename, feature_file])

    return class_to_filename_list, sbp_values, dbp_values



