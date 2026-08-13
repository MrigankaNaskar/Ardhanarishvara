"""
Report Generator for Ardhanarishvara Phase 6.
Generates all Phase 6 deliverables: results tables, figures, and draft methods section.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from security.sanitized_logging import sanitize_errors, log_info


@sanitize_errors("Failed to generate results table.")
def generate_results_table(ablation_df: pd.DataFrame, save_path: str = None) -> str:
    """
    Generate formatted results table from ablation study.

    Args:
        ablation_df: Aggregated ablation results DataFrame
        save_path:   Path to save markdown table

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Results Table — Multimodal ASD Classification\n",
        "| Model | Accuracy | AUC | Sensitivity | Specificity |",
        "|:---|:---|:---|:---|:---|",
    ]

    for _, row in ablation_df.iterrows():
        lines.append(
            f"| {row['ablation_id']} | "
            f"{row['accuracy_mean']*100:.2f}±{row['accuracy_std']*100:.2f}% | "
            f"{row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
            f"{row['sensitivity_mean']*100:.1f}±{row['sensitivity_std']*100:.1f}% | "
            f"{row['specificity_mean']*100:.1f}±{row['specificity_std']*100:.1f}% |"
        )

    table = "\n".join(lines)

    if save_path is None:
        save_path = os.path.join(config.TABLES_DIR, "results_table.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    log_info(f"Saved results table to {os.path.basename(save_path)}")
    return table


@sanitize_errors("Failed to generate ablation figures.")
def generate_ablation_figures(ablation_df: pd.DataFrame, save_dir: str = None):
    """
    Generate publication-quality figures from ablation results.

    Creates:
      1. Accuracy bar chart with error bars
      2. Metric comparison radar/grouped bar chart
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    # ── Figure 1: Accuracy Bar Chart ──
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ablation_df["ablation_id"].tolist()
    accs = ablation_df["accuracy_mean"].values * 100
    stds = ablation_df["accuracy_std"].values * 100

    # Color scheme: blue for unimodal, red for fusion
    colors = ["#3498db" if "only" in m else "#e74c3c" for m in models]

    bars = ax.bar(range(len(models)), accs, yerr=stds,
                  color=colors, edgecolor="white", linewidth=1.5,
                  capsize=5, alpha=0.85)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Study: fMRI-only vs EEG-only vs Fusion", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, acc, std in zip(bars, accs, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_accuracy.png"), dpi=300)
    plt.close()
    log_info("Saved ablation accuracy figure")

    # ── Figure 2: Multi-Metric Grouped Bar Chart ──
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.2

    metrics = [
        ("Accuracy", "accuracy_mean", "accuracy_std", "#3498db"),
        ("AUC", "auc_mean", "auc_std", "#2ecc71"),
        ("Sensitivity", "sensitivity_mean", "sensitivity_std", "#e74c3c"),
        ("Specificity", "specificity_mean", "specificity_std", "#f39c12"),
    ]

    for i, (label, mean_col, std_col, color) in enumerate(metrics):
        vals = ablation_df[mean_col].values * 100
        errs = ablation_df[std_col].values * 100
        ax.bar(x + i * width, vals, width, yerr=errs,
               label=label, color=color, alpha=0.8, capsize=3, edgecolor="white")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Multi-Metric Comparison Across Ablations", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_multimetric.png"), dpi=300)
    plt.close()
    log_info("Saved multi-metric comparison figure")


@sanitize_errors("Failed to generate methods draft.")
def generate_methods_draft(save_path: str = None) -> str:
    """
    Generate a draft methods section for the paper/report.
    Covers data description, preprocessing, model architecture,
    fusion strategy, and evaluation protocol.
    """
    methods = f"""# Methods

## 2.1 Data Description

### 2.1.1 fMRI Data — ABIDE-I
Resting-state functional MRI data were obtained from the Autism Brain Imaging
Data Exchange I (ABIDE-I) repository. Data were preprocessed using the
Configurable Pipeline for the Analysis of Connectomes (CPAC) pipeline.
Time series were extracted using the Craddock 200 (CC200) parcellation atlas,
yielding {config.CC200_N_ROIS} regions of interest (ROIs) per subject.
Functional connectivity matrices (200×200) were computed via Pearson
correlation with Fisher z-transformation.

### 2.1.2 EEG Data — KAU ASD Benchmark Cohort
Electroencephalography (EEG) data were acquired based on the King Abdulaziz
University (KAU) Autism Spectrum Disorder benchmark cohort (Alhaddad et al., 2012).
Recordings used a standard 10-20 system {config.EEG_N_CHANNELS}-channel montage
(Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6) at 250 Hz sampling rate.
Connectivity matrices ({config.EEG_N_CHANNELS}×{config.EEG_N_CHANNELS}) were computed
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
5. Phase Locking Value (PLV) connectivity → {config.EEG_N_CHANNELS}×{config.EEG_N_CHANNELS} matrix
6. Power Spectral Density extraction across δ, θ, α, β, γ bands

## 2.3 Model Architecture

### 2.3.1 Unimodal Encoders
Both fMRI and EEG branches use mirrored 2D-CNN architectures:
- Conv2D(1→32, k=5, p=2) → BN → ReLU → MaxPool(2)
- Conv2D(32→64, k=3, p=1) → BN → ReLU → MaxPool(2)
- Conv2D(64→128, k=3, p=1) → BN → ReLU → GlobalAvgPool
- Dense(128→{config.EMBEDDING_DIM}) → ReLU

Each encoder produces a {config.EMBEDDING_DIM}-dimensional embedding vector.

### 2.3.2 Fusion Strategies
Two fusion approaches were compared:

**ConcatMLP Fusion**: Concatenation of fMRI and EEG embeddings (256-dim)
followed by a 3-layer MLP (256→256→128→2) with BatchNorm, ReLU, and
Dropout({config.FUSION_DROPOUT}).

**CrossAttention Fusion**: Bidirectional multi-head cross-attention
({config.ATTENTION_HEADS} heads) where fMRI queries attend to EEG and
vice versa. Gated residual connections preserve original information.
MLP classifier on concatenated attended representations.

### 2.3.3 Unpaired Population Fusion Strategy
Due to zero subject overlap between cohorts, we employ label-conditioned
random pairing: for each diagnosis class, fMRI embeddings are randomly
paired with EEG embeddings {config.N_RANDOM_PAIRINGS}× to create
augmented multimodal training pairs.

## 2.4 Training Protocol
- Encoders: Adam optimizer, lr=1e-3, weight_decay=1e-4, 15 epochs
- Fusion: Adam optimizer, lr={config.FUSION_LR}, cosine annealing schedule,
  {config.FUSION_EPOCHS} epochs, gradient clipping (max_norm=1.0)
- 80/20 train/val split, stratified by diagnosis
- Cross-entropy loss for all classifiers

## 2.5 Evaluation Protocol

### 2.5.1 Ablation Study
Four configurations evaluated across {len(config.ABLATION_SEEDS)} seeds:
fMRI-only, EEG-only, Fusion (ConcatMLP), Fusion (CrossAttention).
Metrics: accuracy, AUC, sensitivity, specificity (mean ± std).

### 2.5.2 Statistical Significance
- McNemar's test with continuity correction for pairwise comparison
- Paired bootstrap ({config.BOOTSTRAP_N_ITERATIONS} iterations,
  {config.BOOTSTRAP_CI_LEVEL*100:.0f}% confidence intervals)

### 2.5.3 Explainability
- Grad-CAM heatmaps on conv3 of each encoder branch
- SHAP/gradient×input feature importance on connectivity matrices
- Cross-modal convergence analysis mapping to Yeo 7-network atlas
"""

    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "methods_draft.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(methods)

    log_info(f"Saved methods draft to {os.path.basename(save_path)}")
    return methods


@sanitize_errors("Failed to generate full report.")
def generate_full_report(ablation_df: pd.DataFrame = None,
                         xai_results: dict = None,
                         neuro_interpretation: str = None):
    """
    Orchestrate generation of all Phase 6 deliverables.

    Args:
        ablation_df:          Ablation summary DataFrame
        xai_results:          XAI analysis results dict
        neuro_interpretation: Neurobiological interpretation markdown
    """
    log_info("=== Phase 6: Generating Full Report ===")

    # 1. Methods draft
    generate_methods_draft()

    # 2. Results table
    if ablation_df is not None:
        generate_results_table(ablation_df)
        generate_ablation_figures(ablation_df)
    else:
        log_info("Skipping results table — no ablation data provided.")

    # 3. Save neuro interpretation
    if neuro_interpretation:
        neuro_path = os.path.join(config.RESULTS_DIR, "neurobiological_interpretation.md")
        with open(neuro_path, "w", encoding="utf-8") as f:
            f.write(neuro_interpretation)
        log_info(f"Saved neurobiological interpretation to {os.path.basename(neuro_path)}")

    # 4. Summary index
    index_lines = [
        "# Ardhanarishvara — Phase 6 Results Index\n",
        "## Deliverables\n",
        f"- **Results Table**: [`results/tables/results_table.md`](tables/results_table.md)",
        f"- **Ablation Raw Data**: [`results/tables/ablation_raw_results.csv`](tables/ablation_raw_results.csv)",
        f"- **Ablation Summary**: [`results/tables/ablation_summary.csv`](tables/ablation_summary.csv)",
        f"- **Methods Draft**: [`results/methods_draft.md`](methods_draft.md)",
        f"- **Neuro Interpretation**: [`results/neurobiological_interpretation.md`](neurobiological_interpretation.md)",
        "\n## Figures\n",
        f"- **XAI Panel**: [`results/figures/xai_panel.png`](figures/xai_panel.png)",
        f"- **Ablation Accuracy**: [`results/figures/ablation_accuracy.png`](figures/ablation_accuracy.png)",
        f"- **Multi-Metric Comparison**: [`results/figures/ablation_multimetric.png`](figures/ablation_multimetric.png)",
        f"- **Training Convergence**: [`results/figures/training_convergence.png`](figures/training_convergence.png)",
    ]

    index_path = os.path.join(config.RESULTS_DIR, "INDEX.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")

    log_info(f"Generated results index at {os.path.basename(index_path)}")
    log_info("=== Phase 6 Report Generation Complete ===")
