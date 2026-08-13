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

**Ardhanarishvara** represents dual complementary neuroimaging modalities united in harmony. In computational neuroscience and clinical psychiatric AI:
- **Functional Magnetic Resonance Imaging (fMRI)** captures spatial localization of blood-oxygen-level-dependent (BOLD) resting-state functional connectivity.
- **Electroencephalography (EEG)** captures millisecond-level electrophysiological synchronization across cortical oscillation frequency bands ($\delta, \theta, \alpha, \beta, \gamma$).

This repository provides an end-to-end research pipeline designed for non-invasive, objective **Autism Spectrum Disorder (ASD)** screening by coupling fMRI and EEG signals into shared latent 128-dimensional embedding representations.

```
       ┌───────────────────────────────┐          ┌───────────────────────────────┐
       │   fMRI Branch (ABIDE-I)       │          │   EEG Branch (KAU/SFARI ASD)  │
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

## 🚀 Pipeline Phases & Deliverables

### 🔹 Phase 0 — Environment & Scaffolding
- Modular directory architecture: `/data`, `/preprocessing`, `/models/fmri`, `/models/eeg`, `/fusion`, `/xai`, `/notebooks`, `/security`.
- Strict environment locking in `requirements.txt` with verified packages (`torch==2.13.0`, `scikit-learn==1.8.0`, `mne==1.12.1`, `nilearn==0.14.0`, `h5py==3.16.0`, `nibabel==5.4.2`).
- Deliverables: $200 \times 200$ CC200 correlation matrix heatmap ([`notebooks/fmri_cc200_connectivity.png`](notebooks/fmri_cc200_connectivity.png)) and 64-channel EEG preprocessed waveforms ([`notebooks/eeg_visualization.png`](notebooks/eeg_visualization.png)).

### 🔹 Phase 1 — Data Acquisition & Manifests
- **ABIDE-I fMRI Cohort**: Full phenotypic table parsing (1,112 subjects: 539 ASD, 573 TD across NYU, PITT, UCLA, USM, YALE, etc.).
- **EEG ASD Benchmark Cohort**: King Abdulaziz University (KAU) / SFARI Multi-Paradigm Autism EEG cohort metadata (66 ASD, 44 TD).
- **Cohort Overlap Analysis**: Confirms unaligned multimodal setup (`ZERO_OVERLAP_UNALIGNED_MULTIMODAL`).
- Saved manifests: [`data/manifests/abide_manifest.csv`](data/manifests/abide_manifest.csv), [`data/manifests/eeg_manifest.csv`](data/manifests/eeg_manifest.csv), and [`data/manifests/cohort_summary_report.json`](data/manifests/cohort_summary_report.json).

### 🔹 Phase 2 — fMRI Branch
- Ingests authentic CPAC preprocessed resting-state fMRI CC200 timeseries directly from Nilearn / Amazon S3.
- Computes Fisher $z$-transformed Pearson functional connectivity ($200 \times 200$).
- Two-tier caching system (`.npy` and `.h5`) under `data/processed/fmri/` preventing redundant downloads.
- Mirrored 2D-CNN Encoder architecture:
  $$\text{Input } (1, 200, 200) \rightarrow \text{Conv2D}(32, 5\times5) \rightarrow \text{Conv2D}(64, 3\times3) \rightarrow \text{Conv2D}(128, 3\times3) \rightarrow \text{GAP} \rightarrow \text{Dense}(128)$$
- Saves 128-dimensional latent representations to [`models/fmri/fmri_encoder.pt`](models/fmri/fmri_encoder.pt).

### 🔹 Phase 3 — EEG Branch
- MNE preprocessing suite: 0.5–45 Hz bandpass filtering, 50/60 Hz mains notch filtering, average re-referencing, and FastICA artifact rejection.
- Spectral Power Spectral Density (PSD) extraction across 5 standard bands ($\delta, \theta, \alpha, \beta, \gamma$).
- Vectorized Phase Locking Value (PLV) channel synchronization ($64 \times 64$).
- Mirrored 2D-CNN Encoder alongside RBF-kernel SVM baseline.
- Saves 128-dimensional latent representations to [`models/eeg/eeg_encoder.pt`](models/eeg/eeg_encoder.pt) (**66.67% Validation Accuracy** vs **50.00% SVM baseline**).

---

## 🔒 Security Architecture

| Security Requirement | Implementation Mechanism |
| :--- | :--- |
| **API Rate Limiting** | `@rate_limit_downloads` decorator enforces request throttling, backoff delays, and sliding-window minute caps on external data fetches. |
| **Strict Input Validation** | `security/validation.py` verifies file extensions (`.nii`, `.edf`, `.fif`, `.npy`, `.h5`, `.csv`), file size caps (500 MB max), non-empty rows, matrix dimensions ($200 \times 200$ fMRI, $64 \times 64$ EEG), and NaN/Inf rejection. |
| **Secrets Protection** | Zero hardcoded tokens/credentials. Environment variables loaded via `.env` and excluded in `.gitignore`. |
| **Error Sanitization** | `@sanitize_errors` catches raw tracebacks, logging internal paths privately to `logs/system_internal.log` while surfacing sanitized messages to users. |

---

## 📁 Repository Structure

```
Ardhanarishvara/
├── config.py                     # Global paths, hyperparameters, and constants
├── requirements.txt              # Pinned environment package lockfile
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
git clone https://github.com/MrigankaNaskar/Ardhanarishvara-One-being-two-signals_a-unified-view-into-neurodevelopment.git
cd Ardhanarishvara-One-being-two-signals_a-unified-view-into-neurodevelopment
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Locked Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

---

## ⚡ Execution & Verification

### Run Full Pipeline (Phases 0 through 3)
```bash
python run_pipeline.py
```

### Run Individual Modules
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
- **Phase 3 EEG Encoder & Baseline**:
  ```bash
  python models/eeg/encoder.py
  ```

---

## 📊 Summary of Model Checkpoints & Deliverables

| Deliverable | Description | Output Location / Metrics |
| :--- | :--- | :--- |
| **fMRI CC200 Matrix** | $200 \times 200$ Fisher $z$-transformed correlation matrix | `data/processed/fmri/` (`.npy`, `.h5`) |
| **EEG PLV Matrix** | $64 \times 64$ Phase Locking Value matrix | `data/processed/eeg/` (`.npy`, `.h5`) |
| **Cohort Manifests** | Subject counts, diagnosis, age, sex, site, overlap | `data/manifests/abide_manifest.csv`<br>`data/manifests/eeg_manifest.csv` |
| **fMRI Encoder** | 128-dim 2D-CNN feature representation | `models/fmri/fmri_encoder.pt` |
| **EEG Encoder** | 128-dim 2D-CNN feature representation | `models/eeg/eeg_encoder.pt`<br>**66.67% Val Acc** vs **50.00% SVM Baseline** |

---

## 📜 License & Citation

This project is licensed under the **MIT License**.

```bibtex
@software{ardhanarishvara2026,
  author = {Mriganka Naskar},
  title = {Ardhanarishvara: One being, two signals — a unified view into neurodevelopment},
  year = {2026},
  url = {https://github.com/MrigankaNaskar/Ardhanarishvara-One-being-two-signals_a-unified-view-into-neurodevelopment}
}
```
