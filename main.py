import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

channel_ID = "P-1"
stride = 1
data_dir = "data"
data_anomaly_sheet = "data/labeled_anomalies.csv"

#select a channel from the anomaly file
def select_channel(channel):
    if not os.path.exists(data_anomaly_sheet):
        raise FileNotFoundError("No anomaly sheet found. download it first from https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv")
    labels_df = pd.read_csv(data_anomaly_sheet)
    if channel_ID not in labels_df["chan_id"].values:
        raise ValueError("channel")
#trying to do the data preparation
    train = np.load(os.path.join(data_dir, "train", f"{channel_ID}.npy")).astype(np.float32)
    test = np.load(os.path.join(data_dir, "test", f"{channel_ID}.npy")).astype(np.float32)
    row = labels_df[labels_df["chan_id"] == channel_ID].iloc[0]
    test_labels = np.zeros(len(test), dtype=int)
    for start, end in eval(row["anomaly_sequences"]):
        test_labels[start:end + 1] = 1
    return train, test, test_labels

# windows for looking for anomalies
def make_windows(arr, window_size, stride):
    n = (len(arr) - window_size) // stride + 1
    return np.stack([arr[i:i + window_size] for i in range(0, n * stride, stride)])


def window_labels(labels, window_size, stride):
    n = (len(labels) - window_size) // stride + 1
    return np.array([int(labels[i:i + window_size].max()) for i in range(0, n * stride, stride)])


select_channel("P-1")
