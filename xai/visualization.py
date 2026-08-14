"""
XAI Visualization Module for Ardhanarishvara Phase 5.
Publication-quality figures for Grad-CAM heatmaps, SHAP summaries,
cross-modal convergence comparisons, and the composite multi-panel XAI figure.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from security.sanitized_logging import sanitize_errors, log_info

# ── Publication-quality defaults ──────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


@sanitize_errors("Failed to plot Grad-CAM heatmap.")
def plot_gradcam_heatmap(cam: np.ndarray, input_matrix: np.ndarray = None,
                         title: str = "Grad-CAM Heatmap",
                         save_path: str = None, ax: plt.Axes = None):
    """
    Plot Grad-CAM heatmap, optionally overlaid on the original connectivity matrix.

    Args:
        cam: (H, W) normalized Grad-CAM heatmap in [0, 1]
        input_matrix: (H, W) original connectivity matrix for overlay
        title: Plot title
        save_path: Path to save figure (if standalone)
        ax: Matplotlib axes (for subplot embedding)
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))

    if input_matrix is not None:
        ax.imshow(input_matrix, cmap="gray", aspect="equal", alpha=0.5)
        im = ax.imshow(cam, cmap="jet", aspect="equal", alpha=0.6, vmin=0, vmax=1)
    else:
        im = ax.imshow(cam, cmap="inferno", aspect="equal", vmin=0, vmax=1)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Region / Channel Index")
    ax.set_ylabel("Region / Channel Index")

    if standalone:
        plt.colorbar(im, ax=ax, label="Importance", shrink=0.8)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            log_info(f"Saved Grad-CAM heatmap to {os.path.basename(save_path)}")
        plt.close()
    else:
        plt.colorbar(im, ax=ax, label="Importance", shrink=0.8)


@sanitize_errors("Failed to plot SHAP summary.")
def plot_shap_summary(importance_values: np.ndarray, title: str = "Feature Importance",
                      n_top: int = 20, feature_labels: list = None,
                      save_path: str = None, ax: plt.Axes = None):
    """
    Plot top-N most important features as a horizontal bar chart.

    Args:
        importance_values: (H, W) mean absolute importance matrix
        title: Plot title
        n_top: Number of top features to display
        feature_labels: Optional labels for features
        save_path: Path to save figure
        ax: Matplotlib axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Get top connections from upper triangle
    H, W = importance_values.shape
    triu_r, triu_c = np.triu_indices(min(H, W), k=1)
    triu_vals = importance_values[triu_r, triu_c]

    top_idx = np.argsort(triu_vals)[::-1][:n_top]
    top_scores = triu_vals[top_idx]

    if feature_labels:
        top_labels = [f"{feature_labels[triu_r[i]]}↔{feature_labels[triu_c[i]]}"
                      for i in top_idx]
    else:
        top_labels = [f"({triu_r[i]},{triu_c[i]})" for i in top_idx]

    # Reverse for horizontal bar (top feature at top)
    y_pos = np.arange(len(top_labels))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_scores)))

    ax.barh(y_pos, top_scores[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.set_xlabel("Mean |Importance|")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            log_info(f"Saved SHAP summary to {os.path.basename(save_path)}")
        plt.close()


@sanitize_errors("Failed to plot cross-modal comparison.")
def plot_cross_modal_comparison(fmri_importance: dict, eeg_importance: dict,
                                convergence: dict, save_path: str = None,
                                ax_left: plt.Axes = None, ax_right: plt.Axes = None):
    """
    Plot side-by-side brain region importance from both modalities.

    Args:
        fmri_importance: Output from compute_fmri_roi_importance()
        eeg_importance:  Output from compute_eeg_channel_importance()
        convergence:     Output from cross_modal_convergence_analysis()
        save_path: Path to save figure
    """
    standalone = ax_left is None
    if standalone:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # fMRI network scores
    fmri_nets = fmri_importance.get("network_scores", {})
    if fmri_nets:
        nets = sorted(fmri_nets.keys())
        scores = [fmri_nets[n] for n in nets]
        short_labels = [n.split("(")[0].strip()[:15] for n in nets]
        convergent = convergence.get("convergent_networks", [])
        colors = ["#e74c3c" if n in convergent else "#3498db" for n in nets]

        ax_left.barh(range(len(nets)), scores, color=colors, edgecolor="white")
        ax_left.set_yticks(range(len(nets)))
        ax_left.set_yticklabels(short_labels, fontsize=9)
        ax_left.set_xlabel("Importance Score")
        ax_left.set_title("fMRI Network Importance", fontweight="bold")
        ax_left.grid(axis="x", alpha=0.3)

    # EEG region scores
    eeg_regs = eeg_importance.get("region_scores", {})
    if eeg_regs:
        regions = sorted(eeg_regs.keys())
        scores = [eeg_regs[r] for r in regions]
        colors = ["#e74c3c" if any(r in str(c) for c in convergence.get("convergent_networks", []))
                  else "#2ecc71" for r in regions]

        ax_right.barh(range(len(regions)), scores, color=colors, edgecolor="white")
        ax_right.set_yticks(range(len(regions)))
        ax_right.set_yticklabels(regions, fontsize=9)
        ax_right.set_xlabel("Importance Score")
        ax_right.set_title("EEG Region Importance", fontweight="bold")
        ax_right.grid(axis="x", alpha=0.3)

    if standalone:
        plt.suptitle(f"Cross-Modal Convergence (score: {convergence.get('convergence_score', 0):.2f})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            log_info(f"Saved cross-modal comparison to {os.path.basename(save_path)}")
        plt.close()


@sanitize_errors("Failed to plot attention weights heatmap.")
def plot_attention_weights_heatmap(attn_analysis: dict, save_path: str = None,
                                   ax: plt.Axes = None):
    """
    Visualize cross-attention weight distributions across heads.

    Args:
        attn_analysis: Output from compute_attention_weight_analysis()
        save_path: Path to save figure
        ax: Matplotlib axes
    """
    if attn_analysis.get("status") != "computed":
        log_info("No attention weights to visualize (concat fusion).")
        return

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    f2e_means = attn_analysis["fmri_to_eeg"]["mean_per_head"]
    e2f_means = attn_analysis["eeg_to_fmri"]["mean_per_head"]
    f2e_stds = attn_analysis["fmri_to_eeg"]["std_per_head"]
    e2f_stds = attn_analysis["eeg_to_fmri"]["std_per_head"]
    n_heads = len(f2e_means)

    x = np.arange(n_heads)
    width = 0.35

    ax.bar(x - width/2, f2e_means, width, yerr=f2e_stds,
           label="fMRI → EEG", color="#3498db", alpha=0.85, capsize=3)
    ax.bar(x + width/2, e2f_means, width, yerr=e2f_stds,
           label="EEG → fMRI", color="#2ecc71", alpha=0.85, capsize=3)

    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Cross-Attention Weights per Head", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {i+1}" for i in range(n_heads)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            log_info(f"Saved attention weights plot to {os.path.basename(save_path)}")
        plt.close()


@sanitize_errors("Failed to generate XAI figure panel.")
def generate_xai_figure_panel(fmri_cam: np.ndarray, eeg_cam: np.ndarray,
                              fmri_shap: np.ndarray, eeg_shap: np.ndarray,
                              fmri_importance: dict, eeg_importance: dict,
                              convergence: dict, attn_analysis: dict = None,
                              save_path: str = None):
    """
    Generate the composite multi-panel XAI figure — the strongest paper figure.

    Layout:
        ┌─────────────┬──────────────┐
        │ A: fMRI      │ B: EEG       │
        │ Grad-CAM     │ Grad-CAM     │
        ├─────────────┼──────────────┤
        │ C: fMRI SHAP │ D: EEG SHAP  │
        │ Top Features │ Top Features │
        ├─────────────┴──────────────┤
        │ E: Cross-Modal Convergence  │
        │    + Attention Weights      │
        └────────────────────────────┘
    """
    has_attn = attn_analysis is not None and attn_analysis.get("status") == "computed"
    n_bottom_cols = 2 if has_attn else 1

    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8],
                           hspace=0.35, wspace=0.3)

    # ── Panel A: fMRI Grad-CAM ──
    ax_a = fig.add_subplot(gs[0, 0])
    plot_gradcam_heatmap(fmri_cam, title="(A) fMRI Grad-CAM (CC200)", ax=ax_a)

    # ── Panel B: EEG Grad-CAM ──
    ax_b = fig.add_subplot(gs[0, 1])
    plot_gradcam_heatmap(eeg_cam, title="(B) EEG Grad-CAM (PLV)", ax=ax_b)

    # ── Panel C: fMRI SHAP Top Features ──
    ax_c = fig.add_subplot(gs[1, 0])
    plot_shap_summary(fmri_shap, title="(C) fMRI Top Connections", n_top=15, ax=ax_c)

    # ── Panel D: EEG SHAP Top Features ──
    ax_d = fig.add_subplot(gs[1, 1])
    plot_shap_summary(eeg_shap, title="(D) EEG Top Connections", n_top=15, ax=ax_d)

    # ── Panel E: Cross-Modal Convergence ──
    if has_attn:
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :])
        ax_e1 = fig.add_subplot(gs_bottom[0])
        ax_e2 = fig.add_subplot(gs_bottom[1])
        plot_cross_modal_comparison(fmri_importance, eeg_importance, convergence,
                                     ax_left=ax_e1, ax_right=ax_e2)
        ax_e1.set_title("(E) fMRI Network Importance", fontweight="bold")
        ax_e2.set_title("(F) EEG Region Importance", fontweight="bold")
    else:
        ax_e = fig.add_subplot(gs[2, 0])
        ax_f = fig.add_subplot(gs[2, 1])
        plot_cross_modal_comparison(fmri_importance, eeg_importance, convergence,
                                     ax_left=ax_e, ax_right=ax_f)
        ax_e.set_title("(E) fMRI Network Importance", fontweight="bold")
        ax_f.set_title("(F) EEG Region Importance", fontweight="bold")

    # ── Title and save ──
    conv_score = convergence.get("convergence_score", 0)
    fig.suptitle(f"Ardhanarishvara XAI Analysis — Cross-Modal Convergence: {conv_score:.2f}",
                 fontsize=15, fontweight="bold", y=0.98)

    if save_path is None:
        save_path = os.path.join(config.FIGURES_DIR, "xai_panel.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    log_info(f"Saved composite XAI panel figure to {os.path.basename(save_path)}")


@sanitize_errors("Failed to plot training convergence.")
def plot_training_convergence(history: dict, save_path: str = None):
    """Plot fusion training loss and 3-way accuracy convergence curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["fusion_train_loss"]) + 1)

    # Loss curve
    ax1.plot(epochs, history["fusion_train_loss"], "b-", linewidth=2, label="Fusion Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy comparison
    ax2.plot(epochs, [a * 100 for a in history["fusion_val_acc"]],
             "r-", linewidth=2, label=f"Fusion ({history['fusion_type']})")
    ax2.axhline(y=history["fmri_only_acc"] * 100, color="blue",
                linestyle="--", linewidth=1.5, label="fMRI-only")
    ax2.axhline(y=history["eeg_only_acc"] * 100, color="green",
                linestyle="--", linewidth=1.5, label="EEG-only")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("3-Way Accuracy Comparison", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURES_DIR, "training_convergence.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    log_info(f"Saved training convergence plot to {os.path.basename(save_path)}")
