import ast
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def build_anomaly_class_array(anomaly_file, channel_id, test_len):
    df = pd.read_csv(anomaly_file)
    row = df.loc[df["chan_id"] == channel_id].iloc[0]
    sequences = ast.literal_eval(row["anomaly_sequences"])
    classes = ast.literal_eval(re.sub(r"([A-Za-z_]+)", r"'\1'", row["class"]))

    arr = np.full(test_len, "normal", dtype=object)
    for (start, end), cls in zip(sequences, classes):
        arr[start:end + 1] = cls
    return arr


def point_adjust(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred).copy()
    i, n = 0, len(y_true)
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            if y_pred[i:j].any():
                y_pred[i:j] = 1
            i = j
        else:
            i += 1
    return y_pred


def false_alarm_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (tn + fp) * 1000) if (tn + fp) > 0 else 0.0


def evaluate_model(y_true, train_scores, test_scores, percentile=95):
    thresh = float(np.percentile(train_scores, percentile))
    y_pred = (test_scores > thresh).astype(int)
    y_pred_pa = point_adjust(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    precision_pa, recall_pa, f1_pa, _ = precision_recall_fscore_support(y_true, y_pred_pa, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, test_scores)
    except ValueError:
        auc = None

    return {
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "auc_roc": round(float(auc), 4) if auc is not None else None,
        "false_alarm_rate_per_1000": round(false_alarm_rate(y_true, y_pred), 2),
        "threshold": round(thresh, 6),
        "f1_pa": round(float(f1_pa), 4),
        "precision_pa": round(float(precision_pa), 4),
        "recall_pa": round(float(recall_pa), 4),
        "false_alarm_rate_per_1000_pa": round(false_alarm_rate(y_true, y_pred_pa), 2),
    }


def dominant_anomaly_class(anomaly_classes):
    labels = anomaly_classes[anomaly_classes != "normal"]
    if len(labels) == 0:
        return "none"
    vals, counts = np.unique(labels, return_counts=True)
    return str(vals[np.argmax(counts)])


def append_metrics_summary(rows, path):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        channel_id, models_here = df["channel_id"].iloc[0], set(df["model"])
        keep = ~((existing["channel_id"] == channel_id) & (existing["model"].isin(models_here)))
        df = pd.concat([existing[keep], df], ignore_index=True)
    df.to_csv(path, index=False)


def export_predictions_csv(path, point_labels, anomaly_classes, models):
    n = len(point_labels)
    df = pd.DataFrame({
        "timestamp_idx": np.arange(n),
        "ground_truth": point_labels,
        "anomaly_class": anomaly_classes,
    })
    for name, m in models.items():
        col = np.full(n, np.nan)
        col[m["timestamp_idx"]] = m["test_scores"]
        df[f"{name}_score"] = col
    df.to_csv(path, index=False)


def run_evaluation(cfg, point_labels, models, out_dir="results", summary_path=None):
    """models: {name: {"timestamp_idx": arr, "train_scores": arr, "test_scores": arr}}"""
    anomaly_classes = build_anomaly_class_array(cfg["data"]["anomaly_file"], cfg["channel_id"], len(point_labels))
    dom_class = dominant_anomaly_class(anomaly_classes)

    metrics = {}
    summary_rows = []
    for name, m in models.items():
        y_true = point_labels[m["timestamp_idx"]]
        result = evaluate_model(y_true, m["train_scores"], m["test_scores"], percentile=cfg["threshold_percentile"])
        metrics[name] = result
        with open(f"{out_dir}/metrics_{name}.json", "w") as f:
            json.dump(result, f, indent=2)

        row = {
            "channel_id": cfg["channel_id"],
            "model": name,
            "anomaly_type": dom_class,
            "n_test_windows": len(y_true),
            "test_anomaly_rate": round(float(y_true.mean()), 4),
        }
        row.update(result)
        summary_rows.append(row)

    export_predictions_csv(f"{out_dir}/predictions_comparison.csv", point_labels, anomaly_classes, models)
    append_metrics_summary(summary_rows, summary_path or f"{out_dir}/metrics_summary.csv")
    return metrics
