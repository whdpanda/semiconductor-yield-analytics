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
| Architecture | ResNet-18 (pretrained, fine-tuned) | Proven baseline; fits on a single GPU |
| Loss function | Focal Loss | Penalizes easy majority-class samples, improves minority-class recall |
| Sampling | `WeightedRandomSampler` | Balances class frequency within each batch |
| Augmentation | Random 90° rotation, horizontal/vertical flip | Exploits wafer rotational symmetry |
| Primary metric | Macro F1 | Equal weight across all 9 classes; accuracy is misleading at 79% imbalance |
| Explainability | Grad-CAM | Visualizes which wafer regions drove the prediction |

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

### Results

> CNN training not yet started — results will be updated after Phase 5 completion.

| Split | Macro F1 | Notes |
|-------|----------|-------|
| Validation | TBD | |
| Test | TBD | |

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

**SPC Layer:**
- X-bar chart, R chart, EWMA chart
- All 8 Western Electric Rules implemented and unit-tested
- Per-parameter violation event log with rule ID and description

**ML Anomaly Detection:**
- Isolation Forest (scikit-learn): multivariate, unsupervised, fast inference
- Autoencoder (PyTorch MLP): reconstruction-error-based detection; threshold at 95th percentile
  of training reconstruction error
- Ensemble: configurable `any` / `majority` / `all` strategy

**Evaluation** (on simulated data with ground-truth anomaly labels):

| Model | Precision | Recall | F1 | False Alarm Rate |
|-------|-----------|--------|----|-----------------|
| SPC (WE Rules) | TBD | TBD | TBD | TBD |
| Isolation Forest | TBD | TBD | TBD | TBD |
| Autoencoder | TBD | TBD | TBD | TBD |
| Ensemble | TBD | TBD | TBD | TBD |

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
python scripts/train_module_a.py --config configs/module_a.yaml
# or
make train-wafer
```

### 7. Run Module B pipeline

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
