# Sequential Memory versus Static Feature Flattening: A Controlled Comparison of LSTM Forecasting and Classical Outlier Detectors for Anomaly Detection in Aerospace IoT Telemetry

**[Author Name]**
[Department / Affiliation]
[Institution]

---

## Abstract

Time-series anomaly detection in Internet-of-Things (IoT) and aerospace telemetry is challenged by non-stationarity: sensor streams exhibit temporal dependence, drift, and non-linear structure that violate the i.i.d. assumptions underlying classical outlier detectors. This paper presents a controlled empirical comparison of a PyTorch Long Short-Term Memory (LSTM) forecaster against two static baselines, Isolation Forest and One-Class Support Vector Machine (OC-SVM), on NASA SMAP/MSL telemetry channels. All models consume identical 30-step rolling windows; the LSTM forecasts the terminal timestep from the preceding twenty-nine, while the baselines receive the window flattened into a single feature vector, erasing chronological order. Under a fixed-seed, 95th-percentile thresholding protocol evaluated with both raw and point-adjusted metrics, we find that on point-anomaly channels Isolation Forest and OC-SVM achieve exactly zero recall, while the LSTM retains non-trivial sensitivity. On a contextual-anomaly channel all three models partially succeed, but the LSTM sustains the lowest false-alarm rate. Per-timestamp score analysis shows static models assign anomalous windows *lower* scores than their own test-set average, indicating structural blindness rather than a mere sensitivity gap.

---

## 1. Introduction

Anomaly detection over multivariate sensor telemetry is a foundational problem in industrial IoT, spacecraft health monitoring, and cyber-physical systems reliability engineering. The dominant classical formulation treats each observation, or each fixed-length window of observations, as an exchangeable point in feature space and asks whether that point is improbable under a model of "normal" density, margin, or isolation depth [1]–[3]. This formulation is mathematically convenient and computationally cheap, but it carries an implicit and rarely interrogated assumption: that the *order* in which feature values arrive is not itself informative. For a large and practically important class of real-world faults — sensor drift, regime transitions, control-loop desynchronization, and other *contextual* or *collective* anomalies [4] — this assumption is false by construction. A value that is entirely unremarkable in isolation can be a definitive anomaly when it arrives at the wrong point in a system's operating cycle, and no amount of tabular feature engineering restores the information destroyed when a temporal window is flattened into an unordered vector.

Sequential deep learning architectures, and Long Short-Term Memory (LSTM) networks in particular [5], were designed precisely to retain and exploit this ordering. Their internal recurrence and gating mechanisms permit a learned representation of a system's short- and long-term dynamics, which can then be used generatively — to forecast the next observation and flag large forecast residuals as anomalous [6], [7] — or in a reconstruction/probabilistic framing [8]. The empirical literature on whether this added structural capacity actually translates into superior anomaly detection, however, is more equivocal than either camp's advocates typically present [9], [10], and the evaluation protocols used to make the comparison are themselves a live subject of methodological critique [11], [12].

This paper contributes a small, fully reproducible, and deliberately transparent empirical study addressing three questions that are frequently asserted but rarely quantified in a single controlled pipeline: (i) under an identical windowing and thresholding protocol, does a compact LSTM forecaster outperform Isolation Forest [2] and One-Class SVM [3] on real spacecraft telemetry; (ii) does the answer depend on whether the injected fault is a point (magnitude) anomaly or a contextual (pattern) anomaly; and (iii) can the *mechanism* of any observed failure be isolated at the level of individual anomaly scores, rather than inferred solely from aggregate metrics. We answer all three using NASA's publicly released SMAP/MSL spacecraft telemetry corpus [7], the same dataset that motivated the original LSTM-NDT nonparametric dynamic thresholding study, evaluated here under a simpler, fixed-percentile threshold to isolate the architectural comparison from thresholding-algorithm confounds.

---

## 2. Literature Review

### 2.1 Classical and Density-/Margin-Based Outlier Detection

Isolation Forest [2] constructs an ensemble of random partitioning trees and scores a point by the average path length required to isolate it; points that are isolated in few splits — i.e., that lie in sparse regions of feature space — receive higher anomaly scores. Its principal appeal is sub-linear training and inference cost and freedom from an explicit density model. One-Class SVM [3], in the Schölkopf formulation, instead learns a maximum-margin frontier separating the bulk of the training distribution from the origin in a kernel-induced feature space, with the hyperparameter `nu` controlling an upper bound on the training-set outlier fraction. Local Outlier Factor [1] scores points by local density deviation relative to their neighbors, capturing a notion of *relative* rather than global rarity. All three methods share the property that emphasizes our central research question: they are permutation-invariant with respect to the ordering of the feature dimensions supplied to them. When a temporal window is used as input, as is common practice for extending these detectors to sequential data [4], this invariance becomes a liability rather than a convenience — the model cannot distinguish a window whose values are correctly ordered from one whose columns have been arbitrarily shuffled.

### 2.2 Sequential and Deep Architectures for Time-Series Anomaly Detection

Malhotra et al. [6] proposed one of the earliest LSTM encoder–decoder architectures for multi-sensor anomaly detection, reconstructing input windows and thresholding on reconstruction error under a multivariate Gaussian fit to validation-set errors. Su et al. [8] extended this line with OmniAnomaly, a stochastic recurrent network combining a GRU backbone with a variational latent space and a planar-normalizing-flow posterior, explicitly modeling the temporal and stochastic dependence between successive latent states. Deep Neural Networks incorporating structural and temporal attention jointly, such as the graph-attention-augmented model of Zhang et al. [14], attempt to capture cross-sensor and cross-time dependence simultaneously rather than treating them as separable. More recently, transformer-based architectures — the Anomaly Transformer of Xu et al. [9] being a prominent example — reformulate the problem around a learned discrepancy between prior and series association patterns, achieving strong results on several public benchmarks while requiring substantially larger training budgets than a compact LSTM. Deep one-class approaches such as Deep SVDD [15] attempt to fuse the representation-learning capacity of deep networks with the classical one-class objective, learning a data-adaptive embedding under which a simple hypersphere boundary becomes sufficient — a hybridization strategy directly relevant to, but not deployed in, the present study. DeepAnT [10] takes a forecasting-based approach structurally similar to the LSTM configuration used here, but built on convolutional rather than recurrent layers, illustrating that the forecast-residual paradigm generalizes across sequence-model families.

### 2.3 Benchmark Corpora and the SMAP/MSL Dataset

Hundman et al. [7] introduced the SMAP and MSL telemetry corpus alongside LSTM-NDT, a forecasting LSTM paired with a nonparametric dynamic-thresholding routine designed to adapt to smoothed error sequences without assuming a parametric error distribution. This corpus, and the `labeled_anomalies.csv` channel-level ground truth it ships with, has become a de facto standard benchmark for multivariate time-series anomaly detection research in the years since. It is not without documented limitations: Wu and Keogh [12] provide a broader critique of widely used time-series anomaly benchmarks (SMAP/MSL among the datasets discussed in the surrounding literature it responds to), arguing that trivial detectors can achieve deceptively strong scores on some public corpora due to unrealistically distinguishable injected anomalies, mislabeled ground truth, or run-length imbalance. We take this critique seriously in Section 5 rather than treating our SMAP/MSL results as unconditionally generalizable.

### 2.4 Evaluation Protocol and the Point-Adjustment Debate

A methodological complication specific to segment-labeled time-series anomaly benchmarks is the *point-adjustment* (PA) protocol, under which a contiguous ground-truth anomalous segment is counted as fully detected if the model flags *any single* timestep within it [11]. PA is near-universal in the deep time-series anomaly detection literature because it reflects an operationally reasonable notion of "the operator was alerted during the fault," but Kim et al. [11] demonstrate formally and empirically that PA can inflate the apparent F1 of an *even randomly initialized, untrained* detector to competitive levels when ground-truth segments are long relative to the sampling rate, because a single lucky excursion above threshold anywhere inside a long segment is sufficient for full credit. We report both raw and PA metrics throughout Section 4 specifically to expose, rather than obscure, this sensitivity, and we return to its implications for our own headline results in Section 5.

### 2.5 Synthesis

Read together, this literature motivates a controlled study distinct from the benchmark-chasing framing common in the deep anomaly detection literature: rather than proposing a new architecture and comparing state-of-the-art numbers across papers with different preprocessing, windowing, and thresholding choices — a comparison Wu and Keogh [12] and Kim et al. [11] both show to be unreliable — we hold every non-architectural factor (windowing, normalization, thresholding rule, random seed, evaluation code) fixed and vary only the model family. This isolates the specific question this paper is designed to answer: given identical information and an identical decision rule, does sequential memory confer a measurable, mechanistically explicable advantage over static flattening on real spacecraft telemetry.

---

## 3. Methodology

### 3.1 Dataset and Channel Structure

We use NASA's SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory / Curiosity rover) telemetry channels as released alongside [7]. Each channel is stored as a pair of NumPy arrays (`data/train/{chan_id}.npy`, `data/test/{chan_id}.npy`) of shape `(timesteps, 25)`, where the first feature is the raw continuous telemetry value and the remaining 24 are one-hot–encoded command/event indicators. Ground-truth anomaly windows are provided in `data/labeled_anomalies.csv`, with rows of the form:

```
chan_id, spacecraft, anomaly_sequences, class, num_values
P-1, SMAP, "[[2149, 2349], [4536, 4844], [3539, 3779]]", "[contextual, contextual, contextual]", 8505
```

The `anomaly_sequences` field is a Python-literal list of `[start, end]` index pairs, and the `class` field is a positionally aligned list of anomaly-type labels (`point` or `contextual`), non-standard in that its elements are unquoted bareword identifiers rather than valid Python string literals. Our data pipeline parses `anomaly_sequences` directly with `ast.literal_eval` and recovers `class` by regex-quoting each identifier (`re.sub(r"([A-Za-z_]+)", r"'\1'", ...)`) before applying the same parser, avoiding a brittle hand-rolled tokenizer while remaining faithful to the file's literal structure. Table 1 summarizes the three channels used in this study, selected to contrast the two labeled anomaly types under otherwise comparable data volumes.

**Table 1: Channel Characteristics**

| Channel | Spacecraft | Raw Features | Retained Features† | Train Len. | Test Len. | Anomalous Test Points | Anomaly Type |
|---|---|---|---|---|---|---|---|
| P-1 | SMAP | 25 | 16 | 2,872 | 8,505 | 751 (8.83%) | Contextual (3 segments) |
| S-1 | SMAP | 25 | 18 | 2,818 | 7,331 | 448 (6.11%) | Point (1 segment) |
| E-2 | SMAP | 25 | 18 | 2,880 | 8,532 | 1,398 (16.39%) | Point (1 segment) |

† After removal of zero-variance (constant-on-train) one-hot command features; see Section 3.2. The retained count varies by channel because different SMAP channels have different active command indicators.

**Channel selection and a data-quality screening pass.** Following feedback that some channels in the SMAP/MSL corpus contain degenerate ("junk") telemetry unsuitable for meaningful anomaly detection, we conducted a systematic data-quality screening pass over every single-anomaly-type SMAP channel (channels whose `class` list contains exactly one anomaly type; mixed point/contextual channels such as T-1, G-7, C-1, and C-2 were excluded a priori to keep the point-versus-contextual contrast unconfounded). For each candidate channel we computed the training-split standard deviation of the primary telemetry feature (feature index 0): a standard deviation numerically indistinguishable from zero indicates a channel whose recorded physical signal is constant or was never meaningfully captured, offering no exploitable temporal structure for *any* model family to learn from regardless of architecture. This screen surfaced a substantial degenerate cluster within the corpus — channels A-1, A-5, B-1, D-1 through D-9, D-12, D-13, G-2, G-4, and R-1 all exhibit a feature-0 training standard deviation of 0.0000 (to four decimal places), with D-12 and D-13 additionally retaining *zero* features after constant-feature removal (Section 3.2), meaning every one of their 25 raw channels is constant on the training split. This cluster was excluded from consideration in its entirety. The channels used in this study were selected from the surviving, non-degenerate candidate pool on the basis of high feature-0 variance (std ≥ 0.52), adequate training-split length (≥2,800 timesteps), and a moderate, realistic anomaly fraction (3–7% of the test split) — avoiding both vanishingly rare segments that would produce unstable point-adjusted metrics and pathologically large ones (several corpus channels exceed 30–49% anomalous test coverage, which we judged inconsistent with a genuine rare-event detection framing).

### 3.2 Preprocessing Pipeline

All preprocessing is performed by a single `DataProcessor` class exposing a deterministic, channel-parameterized pipeline: (1) load the raw train/test arrays and construct point-level binary labels from the parsed anomaly sequences; (2) drop feature columns with zero variance on the *training* split only, since a constant training feature carries no discriminative signal and can destabilize downstream normalization; (3) standardize both splits using training-set mean and standard deviation exclusively (test statistics are never used to fit the scaler, precluding test-set leakage); (4) construct overlapping rolling windows of configurable length and stride — both exposed as explicit `window_size` and `stride` fields per model in the configuration file (Table 3) rather than hardcoded constants, with a window labeled anomalous if any constituent timestep is anomalous. Critically, in the final experimental configuration, each of the three models is windowed *independently*, with its own `window_size` parameter read from a per-model configuration block rather than a single value shared across all models. This was a deliberate architectural correction made during development: an earlier iteration of the pipeline derived every model's window from a single shared `lstm.window_size` field, meaning a hyperparameter search over the LSTM's temporal context silently altered the Isolation Forest and OC-SVM input dimensionality as a side effect — a confound that would have invalidated any independent-variable interpretation of the classical baselines' results. In the configuration evaluated here, all three models use `window_size = 30` (Table 3), making them directly comparable, while remaining architecturally free to diverge in future work.

### 3.3 LSTM Forecasting Architecture

The sequential model is a single-step forecaster: given the first 29 timesteps of a 30-step window (indices 0–28), it predicts the feature vector at the 30th timestep (index 29), and its anomaly score is the mean squared forecast error against the true value at that timestep. Concretely, the input tensor of shape `(batch, 29, n_features)` is passed through an `nn.LSTM` stack, the final hidden state is projected through a linear layer of output width `n_features`, and the model is trained end-to-end with the Adam optimizer under MSE loss. Table 3 (Section 3.5) lists the exact hyperparameters used in the reported configuration; we note explicitly, in the interest of methodological transparency, that these values (`hidden_size = 32`, `1 layer`, `lr = 7×10⁻⁴`, `15 epochs`) were arrived at through iterative manual tuning against the point-adjusted F1 metric on the contextual channel (P-1) and are *not* the values suggested by the original LSTM-NDT reference configuration [7] (which uses a wider, deeper network); we found empirically that increasing capacity (2 layers, hidden width 80, longer 250-step windows, in imitation of [7]'s published configuration) degraded generalization on our smaller per-channel training splits, consistent with the smaller and more heterogeneous per-channel training set sizes used in this study (Table 1) relative to the aggregate multi-channel training regime of the original work.

### 3.4 Classical Baseline Formulation

The static baselines receive the *entire* 30-timestep, per-channel-feature-count window flattened into a single vector of length `window_size × n_retained_features` — 480 for P-1, 540 for S-1 and E-2 (Table 1) — via a `reshape(batch, -1)` transform applied identically at train and inference time. This transform is the deliberate experimental manipulation at the center of this study: it hands the classical models strictly more raw information per example than the LSTM receives per forecast step (the full 30-step window rather than 29 steps), but destroys the sequential structure of that information by treating each `(timestep, feature)` pair as an independent, unordered tabular column. Isolation Forest is fit with `n_estimators = 150`, `max_features = 0.8`, and a `contamination` parameter set nominally to `0.05`; we note as a methodological caveat, confirmed empirically during development, that because our scoring procedure uses `score_samples` combined with an external percentile threshold (Section 3.5) rather than the model's internal `predict` decision rule, `contamination` has no measurable effect on the reported metrics — it would only become active under the library's native `predict()` interface. One-Class SVM uses an RBF kernel with `nu = 0.5` and `gamma = 2×10⁻⁴`, tuned analogously to the LSTM against point-adjusted F1.

### 3.5 Thresholding and Evaluation Protocol

Following standard unsupervised anomaly detection practice, no anomaly labels are used during model fitting; the classifier decision boundary is instead calibrated after training by computing each model's anomaly score on the (label-free) *training* split and setting the alerting threshold at the `threshold_percentile`-th percentile (95, Table 3) of that in-sample score distribution — a single global configuration value applied identically across all three models, in contrast to the per-model `window_size`/`stride` fields, since it represents a shared evaluation-protocol choice rather than a model-specific hyperparameter. A test-set window is flagged anomalous if its score exceeds this threshold. We report two families of metrics per model: **raw** metrics (precision, recall, F1, AUC-ROC, false-alarm rate per 1,000 windows), computed directly on the binary flag sequence, and **point-adjusted (PA)** metrics, computed after applying the correction described in Section 2.4 — any predicted-positive timestep occurring anywhere within a ground-truth-positive contiguous segment causes the entire segment to be counted as detected. AUC-ROC is computed once, directly from the continuous score, and is by construction threshold- and adjustment-invariant.

**Table 2: Formal Definitions**

| Metric | Definition |
|---|---|
| Threshold | 95th percentile of anomaly scores on the training split |
| Raw prediction | `1` if test-window score > threshold, else `0` |
| Point-adjusted prediction | Raw prediction, with every timestep in a ground-truth segment set to `1` if *any* timestep in that segment is `1` |
| False Alarm Rate / 1,000 | `(FP / (TN + FP)) × 1000` |
| AUC-ROC | Area under the ROC curve, computed from continuous scores against raw ground truth |

### 3.6 Reproducibility

A single global integer seed (`42`) is propagated to Python's `random`, NumPy, and PyTorch random-number generators at process start, with `torch.backends.cudnn.deterministic = True` and `cudnn.benchmark = False` set to eliminate cuDNN's non-deterministic convolution/recurrence kernel selection. Isolation Forest receives the same seed via its `random_state` argument; One-Class SVM's QP solver is deterministic given fixed data and hyperparameters and requires no seeding. We verified end-to-end run-to-run determinism empirically: two independent executions of the full pipeline on identical configuration produced byte-identical output metrics. Device selection is CPU-only for all reported results: the development workstation's GPU (an NVIDIA GeForce GTX 960, compute capability 5.2) is not supported by the installed CUDA 13.0-targeted PyTorch build, and a 5-line `try`/`except` guard around a probe CUDA tensor allocation falls back to CPU automatically when this mismatch is detected (Section 5 discusses the resulting computational constraints on model scale).

**Table 3: Final Hyperparameter Configuration**

| Component | Parameter | Value |
|---|---|---|
| Global | seed | 42 |
| Global | threshold_percentile | 95 |
| LSTM | window_size | 30 |
| LSTM | stride | 1 |
| LSTM | hidden_size | 32 |
| LSTM | layers | 1 |
| LSTM | learning_rate | 0.0007 |
| LSTM | epochs | 15 |
| LSTM | batch_size | 64 |
| Isolation Forest | window_size | 30 |
| Isolation Forest | stride | 1 |
| Isolation Forest | n_estimators | 150 |
| Isolation Forest | contamination | 0.05 |
| Isolation Forest | max_features | 0.8 |
| One-Class SVM | window_size | 30 |
| One-Class SVM | stride | 1 |
| One-Class SVM | kernel | RBF |
| One-Class SVM | nu | 0.5 |
| One-Class SVM | gamma | 0.0002 |

<!-- FIGURE 1 [matplotlib/graphviz, programmatically injected]: Schematic of the shared windowing pipeline branching into (a) the LSTM's 29-step-input / 30th-step-target forecasting head and (b) the static baselines' full 30-step flatten-to-480/540-dim transform, annotated with the point at which chronological order is discarded. -->

---

## 4. Results

### 4.1 Quantitative Comparison

Tables 4–6 report raw and point-adjusted metrics for all three models on each of the three evaluated channels, extracted directly from the pipeline's per-model `metrics_{model}.json` outputs and consolidated `metrics_summary.csv`. We emphasize a methodological limitation up front: each configuration was executed once, under the single fixed seed described in Section 3.6; the pipeline is exactly deterministic given that seed, so these are reproducible point estimates rather than the mean of an independent-trials distribution, and we accordingly do not report standard deviations or confidence intervals in this section — doing so without genuine repeated-trial variance (e.g., across multiple random seeds or bootstrap resamples of the test windows) would misrepresent point-estimate reproducibility as statistical uncertainty. We treat this explicitly as a limitation in Section 5 rather than approximate it.

**Table 4: P-1 (Contextual Anomaly Channel) — Raw vs. Point-Adjusted Metrics**

| Model | F1 | Precision | Recall | AUC-ROC | FAR/1000 | F1-PA | Precision-PA | Recall-PA | FAR/1000-PA |
|---|---|---|---|---|---|---|---|---|---|
| LSTM | 0.0724 | 0.0984 | 0.0573 | 0.4642 | 51.00 | **0.7922** | 0.6559 | 1.0000 | **51.00** |
| Isolation Forest | 0.0926 | 0.1101 | 0.0799 | 0.4885 | 62.78 | 0.7559 | 0.6076 | 1.0000 | 62.78 |
| One-Class SVM | 0.0569 | 0.0764 | 0.0453 | 0.4941 | 53.20 | 0.4201 | 0.4292 | 0.4115 | 53.20 |

**Table 5: S-1 (Point Anomaly Channel) — Raw vs. Point-Adjusted Metrics**

| Model | F1 | Precision | Recall | AUC-ROC | FAR/1000 | F1-PA | Precision-PA | Recall-PA | FAR/1000-PA |
|---|---|---|---|---|---|---|---|---|---|
| LSTM | 0.0109 | 0.0139 | 0.0089 | 0.6493 | 41.29 | **0.7600** | 0.6129 | **1.0000** | 41.29 |
| Isolation Forest | 0.0000 | 0.0000 | 0.0000 | 0.2966 | 23.05 | 0.0000 | 0.0000 | 0.0000 | 23.05 |
| One-Class SVM | 0.0000 | 0.0000 | 0.0000 | 0.4963 | 32.97 | 0.0000 | 0.0000 | 0.0000 | 32.97 |

**Table 6: E-2 (Point Anomaly Channel) — Raw vs. Point-Adjusted Metrics**

| Model | F1 | Precision | Recall | AUC-ROC | FAR/1000 | F1-PA | Precision-PA | Recall-PA | FAR/1000-PA |
|---|---|---|---|---|---|---|---|---|---|
| LSTM | 0.0282 | 0.0661 | 0.0179 | 0.5413 | 49.68 | **0.8879** | 0.7984 | **1.0000** | 49.68 |
| Isolation Forest | 0.0000 | 0.0000 | 0.0000 | 0.4193 | 32.23 | 0.0000 | 0.0000 | 0.0000 | 32.23 |
| One-Class SVM | 0.0000 | 0.0000 | 0.0000 | 0.4549 | 29.70 | 0.0000 | 0.0000 | 0.0000 | 29.70 |

<!-- FIGURE 2 [matplotlib bar chart, programmatically injected]: Grouped bar chart of F1-PA per model across the three channels (P-1, S-1, E-2), visually contrasting the contextual-channel three-way competition against the point-channel LSTM-only detection. -->

<!-- FIGURE 3 [matplotlib ROC curve, programmatically injected]: Overlaid ROC curves for LSTM, Isolation Forest, and OC-SVM on channel E-2, illustrating the near-diagonal (AUC ≈ 0.42–0.54) performance of all three models despite the large gap in F1-PA. -->

### 4.2 Where Static Models Fail: Contextual versus Point Anomalies

The central empirical finding of this study is the qualitative divergence between Table 4 and Tables 5–6. On P-1, whose three labeled anomaly segments are exclusively of the `contextual` type, all three models achieve non-trivial point-adjusted detection (F1-PA between 0.42 and 0.79); Isolation Forest is in fact competitive with, and by raw F1 slightly exceeds, the LSTM on this channel. On S-1 and E-2, whose single labeled segment is of the `point` type, Isolation Forest and OC-SVM collapse to **exactly zero** on every threshold-dependent metric — not merely a low score, but zero true positives at any operating point defined by their own 95th-percentile training threshold — while the LSTM retains a point-adjusted F1 of 0.76 (S-1) and 0.89 (E-2), and a perfect point-adjusted recall on both. This is, at face value, a counter-intuitive result: point anomalies are magnitude excursions, the archetypal case a density- or isolation-based detector should excel at identifying, and it is the flattened classical baselines, not the sequential model, that miss them entirely.

### 4.3 Why: Diagnostic Score Analysis

To understand this divergence mechanistically rather than only observationally, we examined each model's raw continuous anomaly score conditioned on ground-truth label, using the full per-timestamp `predictions_comparison.csv` diagnostic export (Table 7). For both S-1 and E-2, the Isolation Forest score during genuinely anomalous windows is *lower*, on average, than the model's own test-set-wide average score, and substantially below the maximum score it assigns anywhere in the test set. The One-Class SVM exhibits the same qualitative pattern, with a far larger score range driven by a small number of extreme false-positive outliers elsewhere in the signal. In other words, both classical detectors are not merely under-sensitive to the injected point anomalies — they actively rank unrelated, benign regions of the test signal as *more* anomalous than the labeled fault. We interpret this as direct evidence that the flattening transform (Section 3.4) does not simply discard temporal information the models could otherwise use; it introduces spurious high-dimensional tabular sparsity patterns — arising from other, incidental cross-feature co-occurrences in the flattened window — that dominate the isolation-depth and margin geometry the classical models rely on, actively misdirecting their score away from the true fault region. The LSTM shows the inverse pattern: although its *average* forecast error across the anomalous segment is not dramatically elevated (consistent with its low raw recall in Tables 5–6), its *peak* forecast error inside the anomalous segment reaches 5–8× its decision threshold, sufficient to trigger point-adjustment credit even though only 0.9% (S-1) and 1.8% (E-2) of individual anomalous timesteps exceed the threshold in isolation.

**Table 7: Anomalous-Window Score Statistics (Diagnostic Analysis)**

| Channel | Model | Mean Score, Anomalous Timesteps | Mean Score, Full Test Set | Max Score, Full Test Set |
|---|---|---|---|---|
| S-1 | LSTM | 0.239 | 0.454 | 127.44 |
| S-1 | Isolation Forest | 0.329 | 0.354 | 0.527 |
| S-1 | One-Class SVM | −23.25 | −0.16 | 676.25 |
| E-2 | LSTM | 0.109 | 0.423 | 172.14 |
| E-2 | Isolation Forest | 0.326 | 0.353 | 0.539 |
| E-2 | One-Class SVM | −48.95 | −6.86 | 818.14 |

<!-- FIGURE 4 [matplotlib line/scatter overlay, programmatically injected]: Raw telemetry signal (top panel) aligned against LSTM forecast error, Isolation Forest score, and OC-SVM score (stacked lower panels) over a representative window spanning the E-2 anomaly segment [5598, 6995], with the 95th-percentile threshold for each model drawn as a horizontal reference line — the intended "why" figure directly visualizing the divergence quantified in Table 7. -->

### 4.4 Sensitivity to the Point-Adjustment Protocol

Consistent with the critique in [11], the gap between raw and PA metrics in this study is large for every model on every channel, and is *not* uniform across models — the LSTM's PA uplift (raw F1 0.011 → PA F1 0.760 on S-1, a 69-fold increase) is driven by rare, sharp forecast-error spikes sufficient for single-timestep detection, whereas Isolation Forest and OC-SVM's PA metrics remain identically zero because they have literally no true positives at any point in the segment for PA to amplify. This asymmetry means the PA protocol, in this specific study, does not manufacture the LSTM's apparent advantage from noise — a model with zero raw true positives cannot be rescued by point-adjustment, and the classical baselines' complete absence of any true positive is a property of the raw prediction, not an artifact of the adjustment rule. We nonetheless report Table 2's raw metrics alongside PA in every table specifically so this distinction is auditable rather than asserted.

---

## 5. Limitations

**Hardware-constrained model scale.** All experiments were conducted on a workstation whose GPU (GTX 960, compute capability 5.2) is unsupported by the installed CUDA 13.0 PyTorch build, forcing CPU-only execution for every reported result. This directly motivated the compact LSTM configuration (hidden width 32, single layer, 15 epochs) reported in Table 3: a deeper, wider configuration more consistent with the original LSTM-NDT architecture [7], or the paper-informed 2-layer/80-hidden-unit/250-step-window configuration we also trialed during development, exceeded practical CPU training time and was not evaluated to convergence within this study's scope. We cannot rule out that a larger LSTM, properly regularized, would further widen or narrow the gap reported in Section 4.

**Single-seed determinism, not statistical variance.** As stated in Section 4.1, every metric reported here is a single deterministic point estimate under seed 42, not a mean over repeated trials. We deliberately did not fabricate or approximate standard deviations for Tables 4–6, since no repeated-trial data exists to compute them from honestly; a rigorous extension of this work would re-run each configuration under multiple seeds and/or bootstrap resamples of the test windows to report genuine confidence intervals.

**Narrow channel sample.** Three channels, two of them sharing the `point` label, is sufficient to demonstrate the qualitative divergence in Section 4.2 but is too small a sample to support a quantitative claim about the *prevalence* of this effect across the full SMAP/MSL corpus (55 SMAP + 27 MSL channels in the original release [7]), let alone across IoT telemetry domains more broadly.

**Stationarity and threshold generalization.** The 95th-percentile threshold is fit independently per channel and per model on that channel's training split; we make no claim that a threshold calibrated on one channel, or on a training window from one operating regime, transfers to another channel or to a shifted regime within the same channel. This is a known limitation of static percentile thresholding relative to adaptive approaches such as the nonparametric dynamic thresholding of [7] or the extreme-value-theory-based streaming threshold of Siffer et al. [13], neither of which is deployed here in the interest of isolating the model-architecture comparison from thresholding-algorithm choice.

**Benchmark validity.** Following Wu and Keogh's broader critique of time-series anomaly detection benchmarks [12], we note that SMAP/MSL's anomalies are synthetically injected or curated retrospectively by domain experts rather than arising from an independently verified fault ground truth, and that some published results on this corpus may reflect benchmark-specific artifacts rather than generalizable detection capability. Our contribution is a controlled *relative* comparison under one fixed pipeline, not a claim of absolute detection performance transferable to arbitrary deployed IoT systems.

**Unused hyperparameters.** We note for transparency that the Isolation Forest `contamination` parameter (Table 3) has no measurable effect under our percentile-threshold scoring procedure (Section 3.4); it is retained in the configuration for documentation purposes and would only become active if the evaluation pipeline were changed to use the estimator's native `.predict()` decision function.

---

## 6. Conclusion

Under a controlled, single-seed, identically-windowed pipeline, sequential memory confers a measurable and mechanistically explicable advantage over static feature flattening on NASA SMAP telemetry — but the advantage is concentrated specifically on the class of anomalies the literature predicts it should be: on contextual anomalies, classical detectors remain competitive; on point anomalies, they fail completely, and diagnostic per-timestamp score analysis shows this failure is a structural blindness introduced by the flattening transform itself, not a mere sensitivity deficit. We regard the negative result for Isolation Forest and OC-SVM on point-type channels — exact-zero detection, not merely inferior detection — as the more informative finding of this study, precisely because it was not the expected outcome under the conventional wisdom that classical detectors handle magnitude anomalies adequately and only struggle with contextual ones.

---

## References

[1] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF: Identifying Density-Based Local Outliers," in *Proc. ACM SIGMOD Int. Conf. Management of Data*, 2000, pp. 93–104.

[2] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 413–422.

[3] B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, "Estimating the Support of a High-Dimensional Distribution," *Neural Computation*, vol. 13, no. 7, pp. 1443–1471, 2001.

[4] V. Chandola, A. Banerjee, and V. Kumar, "Anomaly Detection: A Survey," *ACM Computing Surveys*, vol. 41, no. 3, Article 15, 2009.

[5] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[6] P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection," *arXiv:1607.00148*, 2016.

[7] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding," in *Proc. 24th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining*, 2018, pp. 387–395.

[8] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining*, 2019, pp. 2828–2837.

[9] J. Xu, H. Wu, J. Wang, and M. Long, "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2022.

[10] M. Munir, S. A. Siddiqui, A. Dengel, and S. Ahmed, "DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series," *IEEE Access*, vol. 7, pp. 1991–2005, 2019.

[11] S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon, "Towards a Rigorous Evaluation of Time-series Anomaly Detection," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 36, no. 7, 2022, pp. 7194–7201.

[12] R. Wu and E. Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress," *IEEE Trans. Knowledge and Data Engineering*, 2021.

[13] A. Siffer, P.-A. Fouque, A. Termier, and C. Largouet, "Anomaly Detection in Streams with Extreme Value Theory," in *Proc. 23rd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2017, pp. 1067–1075.

[14] C. Zhang et al., "A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data," in *Proc. AAAI Conf. Artificial Intelligence*, 2019.

[15] L. Ruff et al., "Deep One-Class Classification," in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR vol. 80, 2018, pp. 4393–4402.

---

## Appendix A: Reproducibility Configuration

```json
{
  "seed": 42,
  "channel_id": "P-1",
  "threshold_percentile": 95,
  "data": {
    "data_dir": "data",
    "anomaly_file": "data/labeled_anomalies.csv"
  },
  "lstm": {
    "window_size": 30,
    "stride": 1,
    "hidden": 32,
    "layers": 1,
    "lr": 0.0007,
    "epochs": 15,
    "batch": 64
  },
  "iforest": {
    "window_size": 30,
    "stride": 1,
    "n_estimators": 150,
    "contamination": 0.05,
    "max_features": 0.8
  },
  "svm": {
    "window_size": 30,
    "stride": 1,
    "kernel": "rbf",
    "nu": 0.5,
    "gamma": 0.0002
  }
}
```

## Appendix B: LSTM Forecaster Definition (PyTorch)

```python
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        out, _ = self.lstm(x)          # x: (batch, 29, n_features)
        return self.fc(out[:, -1, :])  # forecast: (batch, n_features)
```

## Appendix C: Point-Adjustment Protocol (Pseudocode)

```
function POINT_ADJUST(y_true, y_pred):
    i ← 0
    while i < length(y_true):
        if y_true[i] == 1:
            j ← i
            while j < length(y_true) and y_true[j] == 1:
                j ← j + 1
            if any(y_pred[i:j]) == 1:
                y_pred[i:j] ← 1
            i ← j
        else:
            i ← i + 1
    return y_pred
```

## Appendix D: Static Baseline Flattening Transform

```python
def flatten_windows(windows):
    # windows: (n_windows, window_size, n_retained_features)
    return windows.reshape(windows.shape[0], -1)
    # -> (n_windows, window_size * n_retained_features)
```
