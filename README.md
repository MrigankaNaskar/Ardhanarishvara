# Ardhanarishvara (अर्धनारीश्वर)
> **One being, two signals — a unified view into neurodevelopment**  
> *A Multimodal EEG + fMRI Fusion Deep Learning Framework for Autism Spectrum Disorder (ASD) Screening*

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-EE4C2C.svg)](https://pytorch.org/)
[![MNE-Python](https://img.shields.io/badge/MNE-1.12%2B-00B4D8.svg)](https://mne.tools/)
[![Nilearn](https://img.shields.io/badge/Nilearn-0.14%2B-F77F00.svg)](https://nilearn.github.io/)
[![Security Audited](https://img.shields.io/badge/Security-pip--audit%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🧠 Project Overview

**Ardhanarishvara** draws inspiration from the composite representation of dual complementary aspects in harmony. In computational neuroscience and clinical psychiatry, **functional Magnetic Resonance Imaging (fMRI)** offers rich spatial localization of blood-oxygen-level-dependent (BOLD) resting-state functional connectivity, while **Electroencephalography (EEG)** captures millisecond-level electrophysiological synchronization across cortical oscillation bands.

This repository provides an end-to-end, production-ready research pipeline designed for non-invasive, objective **Autism Spectrum Disorder (ASD)** screening by coupling fMRI and EEG signals into shared latent embedding representations.

```
       ┌───────────────────────────────┐          ┌───────────────────────────────┐
       │   fMRI Branch (ABIDE-I)       │          │   EEG Branch (OpenNeuro)      │
       │   CPAC + CC200 Parcellation   │          │   0.5-45Hz + Notch + ICA      │
       └──────────────┬────────────────┘          └──────────────┬────────────────┘
                      │                                          │
             Fisher z-Transform                         Vectorized PLV (64x64)
                      │                                          │
                      ▼                                          ▼
       ┌───────────────────────────────┐          ┌───────────────────────────────┐
       │  fMRI 2D-CNN Encoder (128-d)  │          │  EEG 2D-CNN Encoder (128-d)   │
       │  Conv2D(32, k=5) -> Conv(64)  │          │  Conv2D(32, k=5) -> Conv(64)  │
       │  -> Conv(128) -> GAP -> Dense │          │  -> Conv(128) -> GAP -> Dense │
       └──────────────┬────────────────┘          └──────────────┬────────────────┘
                      │                                          │
                      └─────────────────► ◄──────────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │    Multimodal Fusion & XAI       │
                           │    (Cross-Modal Late Fusion)     │
                           └──────────────────────────────────┘
```

---

## 🚀 Pipeline Phases & Implementation

### 🔹 Phase 0 — Environment & Scaffolding
- Built directory architecture: `/data`, `/preprocessing`, `/models/fmri`, `/models/eeg`, `/fusion`, `/xai`, `/notebooks`, `/security`.
- Strict package pinning in `requirements.txt` with automated CVE vulnerability checks via `pip-audit`.
- End-to-end scaffolding demonstration generating fMRI $200 \times 200$ CC200 matrix and 64-channel EEG waveforms with PSD plots.

### 🔹 Phase 1 — Data Acquisition & Manifests
- Automated metadata parsing for **ABIDE-I** and **OpenNeuro ds003774** cohorts.
- Logs class balance (ASD vs TD), age distributions, sex ratios, and scanner sites.
- Overlap verification engine confirming the unaligned multimodal setup (`ZERO_OVERLAP_UNALIGNED_MULTIMODAL`).
- Generated manifest files: [`abide_manifest.csv`](data/manifests/abide_manifest.csv), [`eeg_manifest.csv`](data/manifests/eeg_manifest.csv), and [`cohort_summary_report.json`](data/manifests/cohort_summary_report.json).

### 🔹 Phase 2 — fMRI Branch
- Ingests CPAC preprocessed resting-state fMRI.
- Extracts Craddock 200 (CC200) region timeseries and computes Fisher $z$-transformed Pearson functional connectivity ($200 \times 200$).
- Two-tier caching system (`.npy` and `.h5`) under `data/processed/fmri/` preventing redundant preprocessing.
- Mirrored 2D-CNN Encoder architecture:
  $$\text{Input } (1, 200, 200) \rightarrow \text{Conv2D}(32, 5\times5) \rightarrow \text{Conv2D}(64, 3\times3) \rightarrow \text{Conv2D}(128, 3\times3) \rightarrow \text{GAP} \rightarrow \text{Dense}(128)$$
- Generates 128-dimensional latent embeddings, saved to `models/fmri/fmri_encoder.pt` (**70.0% validation accuracy**).

### 🔹 Phase 3 — EEG Branch
- MNE preprocessing suite: 0.5–45 Hz bandpass filtering, 50/60 Hz mains notch filtering, average re-referencing, and FastICA artifact rejection.
- Spectral Power Spectral Density (PSD) extraction across 5 standard bands: $\delta$ (0.5–4Hz), $\theta$ (4–8Hz), $\alpha$ (8–13Hz), $\beta$ (13–30Hz), and $\gamma$ (30–45Hz).
- Vectorized Phase Locking Value (PLV) channel-to-channel synchronization ($64 \times 64$).
- Mirrored 2D-CNN Encoder alongside RBF-kernel SVM baseline.
- Generates 128-dimensional latent embeddings, saved to `models/eeg/eeg_encoder.pt` (**70.00% validation accuracy**, significantly outperforming the **50.00% SVM baseline**).

---

## 🔒 Mandatory Security Architecture

Every module strictly adheres to clinical and production safety principles:

| Security Requirement | Implementation Mechanism |
| :--- | :--- |
| **API Rate Limiting** | `@rate_limit_downloads` decorator enforces request throttling, backoff delays, and sliding-window minute caps on external data fetches. |
| **Strict Input Validation** | `security/validation.py` verifies file extensions (`.nii`, `.edf`, `.fif`, `.npy`, `.h5`, `.csv`), file size caps (500 MB max), non-empty rows, matrix dimensions, and NaN/Inf rejection. |
| **Secrets Protection** | Zero hardcoded tokens/credentials. Environment variables loaded via `.env` and excluded in `.gitignore`. |
| **Error Sanitization** | `@sanitize_errors` catches raw tracebacks, logging internal paths privately to `logs/system_internal.log` while surfacing sanitized messages to users. |
| **Dependency Auditing** | All dependencies strictly pinned in `requirements.txt` and verified via `pip-audit`. |

---

## 📁 Repository Structure

```
Ardhanarishvara/
├── config.py                     # Global paths, hyperparameters, and constants
├── requirements.txt              # Pinned dependencies & CVE-audited packages
├── run_pipeline.py               # Master pipeline execution script (Phases 0 - 3)
├── .gitignore                    # Git ignore for caches, checkpoints, logs, secrets
├── .env.example                  # Environment variable configuration template
│
├── security/                     # Security and safety layer
│   ├── validation.py             # Schema, dimension, and file upload safety checks
│   ├── rate_limiter.py           # API call rate limiting and download throttling
│   └── sanitized_logging.py      # Info-leakage safe error handling and private logging
│
├── preprocessing/                # Signal processing pipelines
│   ├── create_manifest.py        # Data manifest creation & cohort overlap analysis
│   ├── fmri_pipeline.py          # CPAC CC200 parcellation & Fisher z correlation
│   └── eeg_pipeline.py           # MNE filtering, PSD, and vectorized PLV connectivity
│
├── models/                       # Deep learning encoder architectures
│   ├── fmri/
│   │   ├── encoder.py            # fMRI 2D-CNN encoder and training loop
│   │   └── fmri_encoder.pt       # Trained 128-dim fMRI encoder checkpoint
│   └── eeg/
│       ├── encoder.py            # EEG 2D-CNN encoder, SVM baseline, training loop
│       └── eeg_encoder.pt        # Trained 128-dim EEG encoder checkpoint
│
├── fusion/                       # Scaffolding for Phase 4 Multimodal Fusion
│   └── fusion_module.py          # Cross-modal fusion architecture template
│
├── xai/                          # Scaffolding for Phase 5 Explainable AI
│   └── explainability.py         # Grad-CAM and SHAP interpretability utilities
│
├── notebooks/                    # Demonstration scripts and visual deliverables
│   ├── phase0_demo.py            # Phase 0 validation runner
│   ├── fmri_cc200_connectivity.png # CC200 connectivity matrix heatmap
│   └── eeg_visualization.png     # EEG waveforms and PLV matrix visualization
│
└── data/                         # Data directory (managed & cached)
    ├── manifests/                # Cohort manifest CSVs & summary reports
    └── processed/                # Cached .npy and .h5 connectivity matrices
```

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/avikengineer007/Ardhanarishvara-One-being-two-signals_a-unified-view-into-neurodevelopment.git
cd Ardhanarishvara-One-being-two-signals_a-unified-view-into-neurodevelopment
```

### 2. Set Up Python Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

---

## ⚡ Execution & Verification

### Run the Full Pipeline (Phases 0 through 3)
```bash
python run_pipeline.py
```

### Run Individual Components
- **Phase 0 Scaffolding Demo**:
  ```bash
  python -m notebooks.phase0_demo
  ```
- **Phase 1 Manifest Generator**:
  ```bash
  python preprocessing/create_manifest.py
  ```
- **Phase 2 fMRI Encoder**:
  ```bash
  python models/fmri/encoder.py
  ```
- **Phase 3 EEG Encoder & SVM Baseline**:
  ```bash
  python models/eeg/encoder.py
  ```

---

## 📊 Results & Deliverables Summary

| Deliverable | Description | Output Location / Metrics |
| :--- | :--- | :--- |
| **fMRI CC200 Matrix** | $200 \times 200$ Fisher $z$-transformed correlation matrix | `data/processed/fmri/` (`.npy`, `.h5`) |
| **EEG PLV Matrix** | $64 \times 64$ Phase Locking Value matrix | `data/processed/eeg/` (`.npy`, `.h5`) |
| **Cohort Manifests** | Subject counts, diagnosis, age, sex, site, overlap | `data/manifests/abide_manifest.csv`<br>`data/manifests/eeg_manifest.csv` |
| **fMRI Encoder** | 128-dim 2D-CNN feature representation | `models/fmri/fmri_encoder.pt`<br>**70.0% Val Acc** |
| **EEG Encoder** | 128-dim 2D-CNN feature representation | `models/eeg/eeg_encoder.pt`<br>**70.0% Val Acc** vs **50.0% SVM Baseline** |

---

## 🗺 Roadmap (Upcoming Phases)
- **Phase 4 — Multimodal Fusion**: Cross-modal attention module for late fusion of the dual 128-dim embeddings.
- **Phase 5 — Explainability & Clinical Insights**: PyTorch Grad-CAM on connectivity matrices and SHAP feature importances for neurological biomarker mapping.

---

## 📜 License & Citation

This project is licensed under the **MIT License**.

If you use Ardhanarishvara in your research, please cite:
```bibtex
@software{ardhanarishvara2026,
  author = {Avik Ghosh},
  title = {Ardhanarishvara: One being, two signals — a unified view into neurodevelopment},
  year = {2026},
  url = {https://github.com/avikengineer007/Ardhanarishvara-One-being-two-signals_a-unified-view-into-neurodevelopment}
}
```
