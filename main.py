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

def select_channel(channel):
    if not os.path.exists(data_anomaly_sheet):
        raise FileNotFoundError("No anomaly sheet found. download it first from https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv")
    labels_df = pd.read_csv(data_anomaly_sheet)
    if channel_ID not in labels_df["chan_id"].values:
        raise ValueError("channel")

    train = np.load(os.path.join(data_dir, "train", f"{channel_ID}.npy")).astype(np.float32)

select_channel("P-1")
