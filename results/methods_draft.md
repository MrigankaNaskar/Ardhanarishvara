# Methods

## 2.1 Data Description

### 2.1.1 fMRI Data — ABIDE-I
Resting-state functional MRI data were obtained from the Autism Brain Imaging
Data Exchange I (ABIDE-I) repository. Data were preprocessed using the
Configurable Pipeline for the Analysis of Connectomes (CPAC) pipeline.
Time series were extracted using the Craddock 200 (CC200) parcellation atlas,
yielding 200 regions of interest (ROIs) per subject.
Functional connectivity matrices (200×200) were computed via Pearson
correlation with Fisher z-transformation.

### 2.1.2 EEG Data — KAU ASD Benchmark Cohort
Electroencephalography (EEG) data were acquired based on the King Abdulaziz
University (KAU) Autism Spectrum Disorder benchmark cohort (Alhaddad et al., 2012).
Recordings used a standard 10-20 system 16-channel montage
(Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6) at 250 Hz sampling rate.
Connectivity matrices (16×16) were computed
using the Phase Locking Value (PLV) metric after bandpass, notch filtering, and artifact rejection.

### 2.1.3 Cohort Overlap
The fMRI and EEG cohorts represent separate clinical populations with
zero subject overlap (ZERO_OVERLAP_UNALIGNED_MULTIMODAL configuration).

## 2.2 Preprocessing

### 2.2.1 fMRI Preprocessing
1. CPAC pipeline with standard motion correction and normalization
2. CC200 parcellation to extract 200 ROI time series
3. Pearson correlation → Fisher z-transform → 200×200 connectivity matrix
4. Dual caching (.npy + .h5) for reproducibility

### 2.2.2 EEG Preprocessing
1. Bandpass filtering (0.5–45 Hz, FIR windowed sinc)
2. Notch filtering (50/60 Hz mains hum removal)
3. Average re-referencing
4. FastICA artifact rejection
5. Phase Locking Value (PLV) connectivity → 16×16 matrix
6. Power Spectral Density extraction across δ, θ, α, β, γ bands

## 2.3 Model Architecture

### 2.3.1 Unimodal Encoders
Both fMRI and EEG branches use mirrored 2D-CNN architectures:
- Conv2D(1→32, k=5, p=2) → BN → ReLU → MaxPool(2)
- Conv2D(32→64, k=3, p=1) → BN → ReLU → MaxPool(2)
- Conv2D(64→128, k=3, p=1) → BN → ReLU → GlobalAvgPool
- Dense(128→128) → ReLU

Each encoder produces a 128-dimensional embedding vector.

### 2.3.2 Fusion Strategies
Two fusion approaches were compared:

**ConcatMLP Fusion**: Concatenation of fMRI and EEG embeddings (256-dim)
followed by a 3-layer MLP (256→256→128→2) with BatchNorm, ReLU, and
Dropout(0.3).

**CrossAttention Fusion**: Bidirectional multi-head cross-attention
(4 heads) where fMRI queries attend to EEG and
vice versa. Gated residual connections preserve original information.
MLP classifier on concatenated attended representations.

### 2.3.3 Unpaired Population Fusion Strategy
Due to zero subject overlap between cohorts, we employ label-conditioned
random pairing: for each diagnosis class, fMRI embeddings are randomly
paired with EEG embeddings 5× to create
augmented multimodal training pairs.

## 2.4 Training Protocol
- Encoders: Adam optimizer, lr=1e-3, weight_decay=1e-4, 15 epochs
- Fusion: Adam optimizer, lr=0.0005, cosine annealing schedule,
  30 epochs, gradient clipping (max_norm=1.0)
- 80/20 train/val split, stratified by diagnosis
- Cross-entropy loss for all classifiers

## 2.5 Evaluation Protocol

### 2.5.1 Ablation Study
Four configurations evaluated across 5 seeds:
fMRI-only, EEG-only, Fusion (ConcatMLP), Fusion (CrossAttention).
Metrics: accuracy, AUC, sensitivity, specificity (mean ± std).

### 2.5.2 Statistical Significance
- McNemar's test with continuity correction for pairwise comparison
- Paired bootstrap (10000 iterations,
  95% confidence intervals)

### 2.5.3 Explainability
- Grad-CAM heatmaps on conv3 of each encoder branch
- SHAP/gradient×input feature importance on connectivity matrices
- Cross-modal convergence analysis mapping to Yeo 7-network atlas
