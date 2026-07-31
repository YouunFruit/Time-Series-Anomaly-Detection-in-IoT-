# processing.py

import ast
import os

import numpy as np
import pandas as pd


class DataProcessor:
    """
    Handles loading and preprocessing of the SMAP/MSL datasets.
    """

    def __init__(
        self,
        data_dir="data",
        anomaly_file="data/labeled_anomalies.csv",
    ):
        self.data_dir = data_dir
        self.anomaly_file = anomaly_file

        self.mean = None
        self.std = None
        self.valid_features = None

    ####################################################################
    # DATA LOADING
    ####################################################################

    def load_channel(self, channel_id):
        """
        Loads one SMAP/MSL channel together with its anomaly labels.
        """

        if not os.path.exists(self.anomaly_file):
            raise FileNotFoundError(
                f"Cannot find {self.anomaly_file}"
            )

        labels = pd.read_csv(self.anomaly_file)

        if channel_id not in labels["chan_id"].values:
            raise ValueError(f"Unknown channel '{channel_id}'")

        train = np.load(
            os.path.join(self.data_dir, "train", f"{channel_id}.npy")
        ).astype(np.float32)

        test = np.load(
            os.path.join(self.data_dir, "test", f"{channel_id}.npy")
        ).astype(np.float32)

        row = labels.loc[labels["chan_id"] == channel_id].iloc[0]

        point_labels = np.zeros(len(test), dtype=np.int32)

        anomaly_ranges = ast.literal_eval(row["anomaly_sequences"])

        for start, end in anomaly_ranges:
            point_labels[start : end + 1] = 1

        return train, test, point_labels

    ####################################################################
    # FEATURE PROCESSING
    ####################################################################

    def remove_constant_features(self, train, test):
        """
        Removes features that never change in the training set.
        """

        std = train.std(axis=0)

        self.valid_features = std > 1e-8

        train = train[:, self.valid_features]
        test = test[:, self.valid_features]

        print(
            f"Removed {(~self.valid_features).sum()} constant features."
        )

        print(
            f"Remaining features: {train.shape[1]}"
        )

        return train, test

    ####################################################################
    # NORMALIZATION
    ####################################################################

    def normalize(self, train, test):
        """
        Standardize using ONLY training statistics.
        """

        self.mean = train.mean(axis=0)

        self.std = train.std(axis=0)

        self.std[self.std < 1e-8] = 1.0

        train = (train - self.mean) / self.std
        test = (test - self.mean) / self.std

        return train.astype(np.float32), test.astype(np.float32)

    ####################################################################
    # WINDOWING
    ####################################################################

    @staticmethod
    def create_windows(data, window_size, stride):
        """
        Creates overlapping windows.

        Shape:
            (num_windows, window_size, num_features)
        """

        windows = []

        for start in range(
            0,
            len(data) - window_size + 1,
            stride,
        ):
            windows.append(
                data[start : start + window_size]
            )

        return np.asarray(windows, dtype=np.float32)

    @staticmethod
    def create_window_labels(labels, window_size, stride):
        """
        A window is anomalous if ANY point inside it is anomalous.
        """

        y = []

        for start in range(
            0,
            len(labels) - window_size + 1,
            stride,
        ):
            window = labels[start : start + window_size]

            y.append(int(window.max()))

        return np.asarray(y, dtype=np.int32)

    ####################################################################
    # COMPLETE PIPELINE
    ####################################################################

    def prepare_channel(
        self,
        channel_id,
        window_size=30,
        stride=1,
    ):
        """
        Complete preprocessing pipeline.
        """

        train, test, point_labels = self.load_channel(channel_id)

        print("=" * 60)
        print(f"Channel: {channel_id}")
        print("=" * 60)

        print("Original train:", train.shape)
        print("Original test :", test.shape)

        train, test = self.remove_constant_features(
            train,
            test,
        )

        train, test = self.normalize(train, test)

        train_windows = self.create_windows(
            train,
            window_size,
            stride,
        )

        test_windows = self.create_windows(
            test,
            window_size,
            stride,
        )

        window_labels = self.create_window_labels(
            point_labels,
            window_size,
            stride,
        )

        print()
        print("Train windows:", train_windows.shape)
        print("Test windows :", test_windows.shape)

        print(
            f"Window anomaly rate: {window_labels.mean():.2%}"
        )

        return (
            train_windows,
            test_windows,
            point_labels,
            window_labels,
        )
