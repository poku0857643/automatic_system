import argparse
import json
import os
from distutils.dir_util import copy_tree

import mlflow
import wfdb

from src.configurations import PreprocessConfigurations
from src.extract_data import parse_pickle, unpickle


def main():
    parser = argparse.ArgumentParser(
        description="Process MIMIC-IV waveform dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="mimic_iv_waveform010",
        help="Dataset name; default is mimic_iv",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/mimic_iv/preprocess/",
        help="Directory to store processed data",
    )
    parser.add_argument(
        "--cached_data_id",
        type=str,
        default="",
        help="Previous run ID for cache",
    )
    args = parser.parse_args()

    downstream_directory = args.downstream

    # Check for cached data
    if args.cached_data_id:
        cached_artifact_directory = os.path.join(
            "/tmp/mlruns/0",
            args.cached_data_id,
            "artifacts/downstream_directory",
        )
        copy_tree(
            cached_artifact_directory,
            downstream_directory,
        )
        else:
        # Define output directories for each signal type and blood pressure values
        ecg_output_destination = os.path.join(downstream_directory, "ecg")
        ppg_output_destination = os.path.join(downstream_directory, "ppg")
        sbp_output_destination = os.path.join(downstream_directory, "sbp")
        dbp_output_destination = os.path.join(downstream_directory, "dbp")

        os.makedirs(downstream_directory, exist_ok=True)
        os.makedirs(ecg_output_destination, exist_ok=True)
        os.makedirs(ppg_output_destination, exist_ok=True)
        os.makedirs(sbp_output_destination, exist_ok=True)
        os.makedirs(dbp_output_destination, exist_ok=True)

        # Load and process MIMIC-IV waveform data
        signal_data = process_signals(
            source_dir=Configurations.mimic_iv_waveform_path,
            ecg_dest=ecg_output_destination,
            ppg_dest=ppg_output_destination,
            sbp_dest=sbp_output_destination,
            dbp_dest=dbp_output_destination,
        )

        # Meta information for processed data
        meta_ecg, meta_ppg, meta_sbp, meta_dbp = {}, {}, {}, {}
        for subject_id, signals in signal_data.items():
            ecg_file, ppg_file, sbp_file, dbp_file = signals
            meta_ecg[subject_id] = ecg_file
            meta_ppg[subject_id] = ppg_file
            meta_sbp[subject_id] = sbp_file
            meta_dbp[subject_id] = dbp_file

        # Save metadata to JSON files
        meta_ecg_filepath = os.path.join(downstream_directory, "meta_ecg.json")
        meta_ppg_filepath = os.path.join(downstream_directory, "meta_ppg.json")
        meta_sbp_filepath = os.path.join(downstream_directory, "meta_sbp.json")
        meta_dbp_filepath = os.path.join(downstream_directory, "meta_dbp.json")

        with open(meta_ecg_filepath, "w") as f:
            json.dump(meta_ecg, f)
        with open(meta_ppg_filepath, "w") as f:
            json.dump(meta_ppg, f)
        with open(meta_sbp_filepath, "w") as f:
            json.dump(meta_sbp, f)
        with open(meta_dbp_filepath, "w") as f:
            json.dump(meta_dbp, f)

        # Log processed data to MLflow
    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="downstream_directory",
    )


if __name__ == "__main__":
    main()
