import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(train_windows, cfg, device):
    c = cfg["lstm"]
    X = torch.tensor(train_windows[:, :-1, :], dtype=torch.float32)
    y = torch.tensor(train_windows[:, -1, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y), batch_size=c["batch"], shuffle=True)

    model = LSTMForecaster(train_windows.shape[-1], c["hidden"], c["layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(c["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def score_lstm(model, windows, device):
    model.eval()
    X = torch.tensor(windows[:, :-1, :], dtype=torch.float32).to(device)
    y = torch.tensor(windows[:, -1, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        err = ((model(X) - y) ** 2).mean(dim=1)
    return err.cpu().numpy()


def flatten_windows(windows):
    return windows.reshape(windows.shape[0], -1)


def fit_isolation_forest(train_windows, cfg):
    c = cfg["iforest"]
    model = IsolationForest(
        n_estimators=c["n_estimators"],
        contamination=c.get("contamination", "auto"),
        max_samples=c.get("max_samples", "auto"),
        max_features=c.get("max_features", 1.0),
        random_state=cfg["seed"],
    )
    model.fit(flatten_windows(train_windows))
    return model


def score_isolation_forest(model, windows):
    return -model.score_samples(flatten_windows(windows))


def fit_one_class_svm(train_windows, cfg):
    c = cfg["svm"]
    model = OneClassSVM(kernel=c["kernel"], nu=c.get("nu", 0.5), gamma=c.get("gamma", "scale"))
    model.fit(flatten_windows(train_windows))
    return model


def score_one_class_svm(model, windows):
    return -model.decision_function(flatten_windows(windows))
