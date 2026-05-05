# Semiconductor Yield & Process Analytics Platform

A full-stack data engineering and machine learning project demonstrating two core competencies
in semiconductor manufacturing analytics:

1. **Wafer Map Defect Classification** — CNN-based classification of 9 defect pattern types on the
   public WM-811K dataset, with class imbalance handling, Grad-CAM explainability, and an
   interactive Streamlit demo.
2. **SPC + Process Anomaly Detection** — Statistical Process Control pipeline implementing
   Western Electric Rules combined with ML-based anomaly detection (Isolation Forest + Autoencoder),
   visualized in an interactive dashboard.

> **Data disclosure:** This project uses the WM-811K public dataset and simulated process data.
> All reported metrics are on these datasets only and do not represent real fab production performance.

---

## Architecture

```
semiconductor-yield-analytics/
├── src/semiconductor_yield/      # Core Python package
│   ├── config.py                 # Central path & parameter config
│   ├── data/                     # Data loading utilities
│   ├── wafer/                    # Module A: wafer map classification
│   ├── process/                  # Module B: SPC & anomaly detection
│   ├── models/                   # CNN, Autoencoder, IsolationForest wrapper
│   ├── dashboard/                # Streamlit UI components
│   └── utils/                    # Shared utilities
├── scripts/                      # CLI entry points (train, evaluate, generate)
├── app/                          # Streamlit application
├── configs/                      # YAML hyperparameter configs
├── tests/                        # pytest test suite
├── notebooks/                    # EDA notebooks (read-only, not production code)
└── docs/                         # Design docs, data contract, interview notes
```

---

## Module A: Wafer Map Defect Classification

### Background

In semiconductor fabs, every wafer undergoes electrical testing (E-Test) after fabrication.
The spatial pattern of failing dies on a wafer map is a key diagnostic signal — different
failure patterns (e.g., edge ring, scratch, center cluster) correspond to different root causes
in the process flow (lithography, CMP, etch, contamination, etc.).

Manual pattern classification is time-consuming and inconsistent across engineers.
An automated classifier accelerates root-cause analysis and enables real-time yield monitoring.

### Dataset

- **WM-811K** (MiraCle research group): 811,457 wafer maps, 9 defect pattern classes
- Severe class imbalance: `none` class ~79%; `Near-Full` class ~0.08%
- Public dataset — not proprietary fab data

### Approach

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | WaferCNN (3 conv blocks + GAP, ~94K params) | Lightweight baseline; runs on CPU; no pretrained weights needed |
| Loss function | CrossEntropyLoss | Standard multi-class loss; class imbalance handled by sampler |
| Sampling | `WeightedRandomSampler` | Equalises class frequency per batch; preferred over loss-weighting at extreme imbalance |
| Augmentation | Random 90° rotation, horizontal/vertical flip | Exploits wafer rotational symmetry; preserves discrete {0,1,2} values |
| Primary metric | Macro F1 | Equal weight across all 9 classes; accuracy is misleading at 79% imbalance |
| LR schedule | CosineAnnealingLR | Smooth decay without manual step tuning |

### Data Preparation

**Step 1 — Download WM-811K**

The dataset is not bundled with this repository. Download manually from Kaggle:

```
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
```

Place the file at `data/raw/wm811k/LSWMD.pkl` (~350 MB). If the file is missing,
all Module A scripts will print a clear download instruction and exit.

**Step 2 — Run EDA**

```bash
python scripts/run_wafer_eda.py
```

Outputs saved to `outputs/reports/wafer/`:

| File | Description |
|------|-------------|
| `class_distribution.csv` | Count and percentage per defect class |
| `class_distribution.png` | Horizontal bar chart (labeled wafers only) |
| `imbalance_report.csv` | Per-class count, imbalance ratio, and balanced weight |
| `sample_wafer_maps.png` | 9 × 3 grid of example wafer maps per class |
| `wafer_map_sizes.csv` | Distribution of raw wafer map dimensions |

For a fast smoke run without loading all 172k samples:

```bash
python scripts/run_wafer_eda.py --max-samples 5000
```

**Class imbalance (labeled subset ~172k wafers)**

| Class | Count | % |
|-------|-------|---|
| none | ~147k | 79% |
| Edge-Ring | ~9.7k | 5.2% |
| Edge-Loc | ~5.2k | 2.8% |
| … | … | … |
| Near-Full | ~149 | 0.08% |

Macro F1 is used as the primary training metric because accuracy would be 79%
even for a degenerate "predict none" classifier.

### Training and Evaluation

**Train (requires `data/raw/wm811k/LSWMD.pkl`):**

```bash
# Default: 30 epochs, batch 64, Adam lr=1e-3, WeightedRandomSampler enabled
python scripts/train_wafer_cnn.py

# Faster smoke run (subsamples 5 000 labeled wafers)
python scripts/train_wafer_cnn.py --max-samples 5000 --epochs 5

# Longer run
python scripts/train_wafer_cnn.py --epochs 50 --batch-size 128
```

Outputs: `outputs/models/wafer_cnn_best.pth`, `outputs/reports/wafer/training_metrics.json`,
`outputs/reports/wafer/confusion_matrix_val.png`.

**Evaluate on held-out test split:**

```bash
python scripts/evaluate_wafer_cnn.py
```

Outputs: `outputs/reports/wafer/evaluation_metrics.json`, `confusion_matrix_test.png`,
`misclassified.png`.

### Results

> Metrics on **WM-811K public dataset** only.
> This is a portfolio project — numbers reflect dataset properties, not real fab deployment performance.

| Split | Macro F1 | Notes |
|-------|----------|-------|
| Validation | TBD | Run `train_wafer_cnn.py` to populate |
| Test | TBD | Run `evaluate_wafer_cnn.py` to populate |

---

## Module B: SPC + Process Anomaly Detection

### Background

Statistical Process Control (SPC) is a mandatory tool in semiconductor manufacturing, required
by SEMI and ISO standards. Control charts monitor process parameters in real time; violations
of control rules (Western Electric Rules) trigger engineer investigation.

Modern fabs generate thousands of process parameters per tool per lot. SPC covers univariate
monitoring well, but multivariate anomalies — where no single parameter violates its limits
but the combination is abnormal — require ML-based detection.

This module implements both layers: rule-based SPC and ML-based anomaly detection, with an
ensemble combining their outputs.

### Data

Synthetic process data generated by `scripts/generate_synthetic_data.py`, simulating 8
process parameters (etch rate, deposition thickness, chamber temperature, pressure, RF power,
gas flow, bias voltage, substrate rotation) with controlled anomaly injection.

Optional: UCI SECOM dataset (1,567 samples, 590 features, real semiconductor process,
binary pass/fail labels).

### Approach

| Component | Implementation | Rationale |
|-----------|---------------|-----------|
| SPC | 4 Western Electric Rules (Rules 1–4) | Interpretable, per-parameter, ISO 7870 compliant |
| Anomaly detection | Isolation Forest (sklearn) | Multivariate, unsupervised, fast inference |
| Anomaly detection | Autoencoder (PyTorch MLP) | Reconstruction-error-based; catches correlated shifts |
| Ensemble | `any` / `all` voting | Tune recall vs precision trade-off for false-alarm cost |

### SPC vs ML: Complementary Layers

| Aspect | SPC (WE Rules) | ML (IF + Autoencoder) |
|--------|---------------|----------------------|
| Interpretability | High — exact rule and sigma zone named | Low-medium — anomaly score only |
| Scope | Univariate, one parameter at a time | Multivariate, all features jointly |
| Training data needed | No — statistical formulas | Yes — requires process-baseline data |
| Detects subtle correlated shifts | No | Yes |
| Regulatory / audit trail | Yes (ISO 7870, SEMI E10) | Requires additional documentation |

In practice, SPC and ML run in parallel: SPC for univariate rule-based alerts and regulatory
compliance; ML for multivariate anomaly patterns that no single parameter's chart would catch.
A violation from either layer triggers engineer investigation — neither is a confirmed defect.

### Feature Sets and Leakage Boundaries

Two named feature sets are defined to make the model's information horizon explicit and prevent
temporal leakage:

| Feature set | CLI flag | Features | When available |
|-------------|----------|----------|----------------|
| `process_only` | `--feature-set process_only` | temperature, pressure, gas\_flow, rf\_power, exposure\_dose | During step execution (in-situ, real-time) |
| `full` | `--feature-set full` (default) | all 8 features — adds film\_thickness, overlay\_error, defect\_density | After offline inspection (post-process only) |

**`process_only`** avoids label, outcome, and metrology leakage entirely. All five features
are reported by the process tool in real time during step execution — no post-process
inspection result is included. This is the correct set for early detection: flagging issues
while the lot is still on the tool, before any offline measurement is scheduled.

**`full`** adds the three offline inspection results (film thickness from ellipsometer,
overlay error from metrology tool, defect density from wafer inspection scanner). These are
only available after the process step completes and the wafer is routed to an inspection
station. Use this set exclusively for post-process quality monitoring or retrospective
anomaly analysis — **not** for real-time or inline detection models.

`yield_rate`, `anomaly_label`, `anomaly_type`, and `suspected_root_cause` are **never** used
as model inputs. The ground-truth labels in the synthetic dataset are generated from
controlled perturbations (e.g., step-specific temperature drift, pressure spikes) and are
used **only** to evaluate detection recall on the held-out test split — training is fully
unsupervised.

### Train / Validation / Test Split

The dataset is partitioned by `lot_id` before any model fitting, so all wafers from a given
lot stay in the same split. No wafer from a training lot appears in the test evaluation.

| Split | Lots | Rows | Purpose |
|-------|------|------|---------|
| Train | 35 (70%) | 4,375 | Unsupervised detector fitting |
| Val | 7 (14%) | 875 | Available for threshold tuning (not used in current pipeline) |
| Test | 8 (16%) | 1,000 | Held out; all reported metrics are on this split only |

The split is deterministic (seed = 42). Split metadata (which `lot_id`s belong to each split)
is saved to `outputs/models/split_info.json` alongside trained models. The evaluate script
reads this file to reconstruct the identical test set — it never touches training lots.

### Results (simulated data — held-out test split, not real fab performance)

> All metrics are on the held-out **test split** (8 lots, 1,000 rows, anomaly rate ≈ 21.8%).
> Detectors were trained only on the 35-lot training split; test lots were never seen during fitting.
> These numbers validate pipeline correctness, not real-world detection capability.

| Model | Precision | Recall | F1 | False Alarm Rate | ROC-AUC |
|-------|-----------|--------|----|-----------------|---------|
| Isolation Forest | 0.56 | 0.15 | 0.24 | 0.033 | 0.74 |
| Autoencoder | 0.42 | 0.17 | 0.24 | 0.063 | 0.65 |
| Ensemble (any) | 0.46 | 0.24 | 0.32 | 0.081 | 0.74 |
| Ensemble (all) | 0.57 | 0.07 | 0.13 | 0.015 | 0.74 |

The moderate recall reflects that the synthetic anomalies include subtle slow drifts (last 20%
of lots, +6 °C temperature) which both models partially miss — a realistic challenge in fabs.
Full evaluation report: `outputs/reports/process/anomaly_summary.json`.

### Real-World Considerations

The following topics are not implemented in this portfolio project but are documented in
`docs/interview_notes.md` as design considerations for real fab deployment:

- **Recipe drift:** Adaptive control limits via rolling-window baseline recalculation; CUSUM
  charts for gradual drift detection; input-feature drift monitoring (PSI, KS test).
- **Tool-to-tool variation:** Per-tool normalization; mixed-effects models to separate
  tool offset from true process variation; fleet-level aggregation.
- **Domain shift:** Data drift detection triggers for retraining; domain adaptation for
  new product lines.
- **False alarm cost:** In fab, a false alarm stops a tool (significant cost). The ensemble
  `all` strategy maximizes precision; threshold tuning is guided by false-alarm cost models.
- **Model monitoring:** Reconstruction error distribution tracking for Autoencoder degradation;
  periodic backtesting of SPC rules against labeled fault events.

---

## Getting Started

### Prerequisites

- Python 3.11+
- (Optional) CUDA-compatible GPU for faster CNN training

### 1. Create virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
# or
make install-dev
```

### 3. Download / prepare data

**WM-811K (required for Module A):**
1. Go to https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
2. Download `LSWMD.pkl` (~350 MB)
3. Place at `data/raw/wm811k/LSWMD.pkl`

**UCI SECOM (optional for Module B):**
```bash
python scripts/download_data.py --dataset secom
```

**Synthetic SPC data (Module B, auto-generated):**
```bash
python scripts/generate_synthetic_data.py
# or
make generate-data
```

### 4. Run EDA (Module A)

```bash
# Full EDA — requires data/raw/wm811k/LSWMD.pkl
python scripts/run_wafer_eda.py

# Fast smoke run (5 000 samples)
python scripts/run_wafer_eda.py --max-samples 5000
```

Outputs land in `outputs/reports/wafer/`. If the pkl file is missing, the script
prints clear download instructions and exits — no cryptic traceback.

### 5. Run tests

```bash
pytest tests/
# or
make test
```

### 6. Train Module A (wafer map classifier)

```bash
# Requires data/raw/wm811k/LSWMD.pkl — see step 3 above
python scripts/train_wafer_cnn.py

# Quick smoke run (5 000 samples, 5 epochs)
python scripts/train_wafer_cnn.py --max-samples 5000 --epochs 5
```

### 6b. Evaluate Module A (test split)

```bash
python scripts/evaluate_wafer_cnn.py
```

### 7. Run Module B — SPC analysis

```bash
python scripts/run_spc_analysis.py
```

Outputs: `outputs/reports/process/spc_violations.csv` + 26 control chart PNGs.

### 8. Run Module B — ML anomaly detection

```bash
# Train Isolation Forest + Autoencoder (unsupervised, no labels used)
python scripts/train_process_anomaly.py

# Evaluate against ground-truth labels and save reports
python scripts/evaluate_process_anomaly.py
```

Outputs: `outputs/reports/process/anomaly_scores.csv`, `anomaly_summary.json`,
`feature_importance.csv`.

### 9. Run Module B full pipeline (legacy)

```bash
python scripts/run_module_b_pipeline.py --stage full
# or
make run-spc
```

### 8. Launch dashboard

```bash
streamlit run app/main.py
# or
make dashboard
```

---

## Docker

```bash
# Build
docker build -t semiconductor-yield-analytics .

# Run dashboard
docker run -p 8501:8501 semiconductor-yield-analytics
```

---

## Testing

```bash
# Run all tests
pytest tests/

# With coverage report
pytest tests/ --cov=src/semiconductor_yield --cov-report=html
make test-cov
```

Coverage targets: ≥90% on SPC/rules logic, ≥80% on preprocessing, ≥70% overall.

---

## Limitations & Future Work

**Current limitations:**
- All results are on public / simulated data — not validated in a real fab environment.
- CNN training does not stratify by lot ID (ideal split would be lot-level to prevent data leakage).
- SECOM dataset has only 1,567 samples; too small for deep learning, used as pipeline demo only.
- Autoencoder threshold is set on training data; in production, continuous recalibration is needed.

**Future extensions:**
- Lot-stratified train/val/test split for Module A
- CUSUM chart implementation for Module B
- Per-tool normalization and multi-tool comparison view
- REST API wrapper for model inference (FastAPI)
- CI/CD pipeline with GitHub Actions

---

## Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| Core ML | PyTorch | ≥2.2 |
| Classical ML | scikit-learn | ≥1.4 |
| Data | pandas, numpy | ≥2.1, ≥1.26 |
| Visualization | plotly, matplotlib | ≥5.20, ≥3.8 |
| Web UI | Streamlit | ≥1.35 |
| Testing | pytest, pytest-cov | ≥7.4, ≥4.1 |
| Packaging | setuptools, pyproject.toml | ≥68 |
| Containers | Docker | — |

---

## License

MIT License. Dataset licenses:
- WM-811K: see original MiraCle research publication for terms
- UCI SECOM: UCI ML Repository terms of use
