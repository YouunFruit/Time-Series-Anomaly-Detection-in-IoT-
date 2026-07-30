import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

channel_ID = "P-1"
jump = 1
data_dir = "data"
hidden_dim = 10
output_dim = 25
window_size = 10
layer_dim = 10
treshold = 99
LR = 1e-3
epochs = 10
batch_size = 10
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

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        latent = self.to_latent(h[-1])
        dec_in = self.from_latent(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        return self.output(dec_out)

    def recon_error(self, x):
        return ((self.forward(x) - x) ** 2).mean(dim=(1, 2))

def train_lstm(train_windows, device):
    model = LSTMModel(train_windows.shape[-1], hidden_dim, layer_dim).to(device)
    X = torch.tensor(train_windows, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), ):
            batch = X[perm[i:i + batch_size]].to(device)
            opt.zero_grad()
            predictions, *rest = model(batch) 
    
            loss = loss_fn(predictions, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}  train_MSE={total_loss / len(X):.5f}")
    return model

def evaluate(name, y_true, y_pred, scores):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = (fp / (tn + fp)) * 1000 if (tn + fp) > 0 else 0.0
    return {"model": name, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1_score": round(f1, 4), "auc_roc": round(auc, 4) if not np.isnan(auc) else "N/A","false_alarm_rate_per_1000": round(far, 2)}

def main():
    device = "cpu"
    print(f"Device: {device}")

    train, test, test_labels = select_channel(channel_ID)
    print(f"train={train.shape} test={test.shape} test_anomaly_rate={test_labels.mean():.3%}")

    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-8
    train_norm, test_norm = (train - mean) / std, (test - mean) / std

    train_win = make_windows(train_norm, window_size, jump)
    test_win = make_windows(test_norm, window_size, jump)
    y_true = window_labels(test_labels, window_size, jump)

    results = []
    os.makedirs("results", exist_ok=True)

    #LSTM model
    print("\nTraining LSTM autoencoder...")
    model = train_lstm(train_win, device)
    model.eval()
    with torch.no_grad():
        train_err = model.recon_error(torch.tensor(train_win, dtype=torch.float32).to(device)).cpu().numpy()
        test_err = model.recon_error(torch.tensor(test_win, dtype=torch.float32).to(device)).cpu().numpy()
    thresh = np.percentile(train_err, treshold)
    results.append(evaluate("LSTM Autoencoder", y_true, (test_err > thresh).astype(int), test_err))

    
if __name__ == "__main__":
    main()
