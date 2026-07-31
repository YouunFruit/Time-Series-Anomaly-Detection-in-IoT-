# Time-Series Anomaly Detection in IoT

Compares a PyTorch LSTM forecaster against Isolation Forest and One-Class SVM baselines on NASA's SMAP/MSL spacecraft telemetry, isolating where and why static (non-sequential) models fail to capture memory-dependent anomalies. See [draft.md](draft.md) for the full writeup.

## Setup

1. Download the SMAP/MSL dataset from Kaggle and place it under `data/`, so that `data/train/`, `data/test/`, and `data/labeled_anomalies.csv` exist.
2. Install dependencies and run:

```bash
pip install -r requirements.txt
python3 main.py         # runs the channel set in config/config.json
```

Edit `config/config.json` to change the target channel (`channel_id`) or any model hyperparameter — the whole pipeline (windowing, training, thresholding, evaluation) is config-driven; no code changes needed to re-run on a different channel.

## Running on Google Colab

```python
!git clone <this-repo-url> tsad && %cd tsad
!pip install -r requirements.txt
# upload/download the Kaggle dataset into data/ here
!python3 main.py
```

`main.py` probes for CUDA and falls back to CPU automatically, so this runs unmodified on Colab's GPU or CPU runtimes alike.

## Outputs

Each run writes to `results/<channel_id>/`: per-model metrics (`metrics_{lstm,iforest,svm}.json`) and a per-timestamp diagnostic CSV (`predictions_comparison.csv`, ground truth + anomaly type + all three models' scores). A cross-channel `results/metrics_summary.csv` accumulates one row per `(channel, model)` across runs, updating in place on re-run rather than duplicating.

## Layout

- `processing.py` — `DataProcessor`: channel loading, constant-feature removal, normalization, windowing.
- `models.py` — LSTM forecaster + Isolation Forest / One-Class SVM baselines (flattened-window input).
- `evaluate.py` — thresholding, point-adjustment protocol, metrics/CSV export.
- `main.py` — orchestration, seeding, device selection.
- `config/config.json` — every tunable parameter (per-model window size/stride, hyperparameters, seed, threshold percentile).
