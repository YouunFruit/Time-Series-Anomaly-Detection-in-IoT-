import json
import os
import random

import numpy as np
import torch

from processing import DataProcessor
from models import (train_lstm, score_lstm, fit_isolation_forest, score_isolation_forest,
                     fit_one_class_svm, score_one_class_svm)
from evaluate import run_evaluation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.zeros(1, device=device)
    except RuntimeError:
        device = torch.device("cpu")
    return device


def main():
    with open("config/config.json") as f:
        cfg = json.load(f)

    set_seed(cfg["seed"])
    device = get_device()
    channel_id = cfg["channel_id"]

    dp = DataProcessor(data_dir=cfg["data"]["data_dir"], anomaly_file=cfg["data"]["anomaly_file"])

    def windows_for(model_key):
        ws = cfg[model_key]["window_size"]
        stride = cfg[model_key]["stride"]
        train_w, test_w, point_labels, _ = dp.prepare_channel(channel_id, window_size=ws, stride=stride)
        ts_idx = np.arange(ws - 1, ws - 1 + stride * test_w.shape[0], stride)
        return train_w, test_w, point_labels, ts_idx

    lstm_train_w, lstm_test_w, point_labels, lstm_ts = windows_for("lstm")
    lstm_model = train_lstm(lstm_train_w, cfg, device)
    lstm_train_scores = score_lstm(lstm_model, lstm_train_w, device)
    lstm_test_scores = score_lstm(lstm_model, lstm_test_w, device)

    if_train_w, if_test_w, _, if_ts = windows_for("iforest")
    iforest = fit_isolation_forest(if_train_w, cfg)
    if_train_scores = score_isolation_forest(iforest, if_train_w)
    if_test_scores = score_isolation_forest(iforest, if_test_w)

    svm_train_w, svm_test_w, _, svm_ts = windows_for("svm")
    svm = fit_one_class_svm(svm_train_w, cfg)
    svm_train_scores = score_one_class_svm(svm, svm_train_w)
    svm_test_scores = score_one_class_svm(svm, svm_test_w)

    models = {
        "lstm": {"timestamp_idx": lstm_ts, "train_scores": lstm_train_scores, "test_scores": lstm_test_scores},
        "iforest": {"timestamp_idx": if_ts, "train_scores": if_train_scores, "test_scores": if_test_scores},
        "svm": {"timestamp_idx": svm_ts, "train_scores": svm_train_scores, "test_scores": svm_test_scores},
    }

    out_dir = os.path.join("results", channel_id)
    os.makedirs(out_dir, exist_ok=True)
    metrics = run_evaluation(cfg, point_labels, models, out_dir=out_dir, summary_path="results/metrics_summary.csv")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
