"""
Cross-Modal XAI Analysis for Ardhanarishvara Phase 5.
Compares what each modality's XAI flags, checks for convergence across
fMRI ROIs and EEG channels, and analyzes cross-attention weights.
"""

import numpy as np

import config
from security.sanitized_logging import sanitize_errors, log_info


# ── Approximate CC200 ROI → Brain Network Mapping ──────────────────────────
# Based on Craddock et al. (2012) parcellation and Yeo 7-network atlas overlap.
# ROI ranges are approximate; precise mapping depends on the exact atlas version.
CC200_NETWORK_MAP = {
    "Default Mode Network (DMN)":     list(range(0, 20)) + list(range(160, 175)),
    "Frontoparietal Control":         list(range(20, 45)),
    "Salience / Ventral Attention":   list(range(45, 65)),
    "Dorsal Attention":               list(range(65, 85)),
    "Somatomotor":                    list(range(85, 110)),
    "Visual":                         list(range(110, 140)),
    "Limbic / Temporal":              list(range(140, 160)),
    "Subcortical / Cerebellum":       list(range(175, 200)),
}

# ── EEG Channel → Scalp Region Mapping (Standard 16-ch 10-20 KAU Montage) ────
# Channels: Fp1(0), Fp2(1), F3(2), F4(3), C3(4), C4(5), P3(6), P4(7),
#           O1(8), O2(9), F7(10), F8(11), T3(12), T4(13), T5(14), T6(15)
EEG_CHANNELS_16 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"
]

EEG_REGION_MAP = {
    "Frontal":    [0, 1, 2, 3, 10, 11],  # Fp1, Fp2, F3, F4, F7, F8
    "Central":    [4, 5],                # C3, C4
    "Parietal":   [6, 7],                # P3, P4
    "Temporal":   [12, 13, 14, 15],      # T3, T4, T5, T6
    "Occipital":  [8, 9],                # O1, O2
}


def _get_roi_network(roi_idx: int) -> str:
    """Map a CC200 ROI index to its brain network."""
    for network, rois in CC200_NETWORK_MAP.items():
        if roi_idx in rois:
            return network
    return "Unknown"


def _get_channel_region(ch_idx: int) -> str:
    """Map an EEG channel index to its scalp region."""
    for region, channels in EEG_REGION_MAP.items():
        if ch_idx in channels:
            return region
    return "Unknown"


@sanitize_errors("Failed to compute fMRI ROI importance.")
def compute_fmri_roi_importance(gradcam_maps: list, shap_values: np.ndarray = None) -> dict:
    """
    Compute per-ROI importance from Grad-CAM heatmaps and/or SHAP values.

    Args:
        gradcam_maps: List of (200, 200) Grad-CAM heatmaps
        shap_values:  (N, 200, 200) SHAP importance values (optional)

    Returns:
        Dict with ROI rankings and network-level aggregation:
        {
            'roi_scores': [(roi_idx, score), ...],  # sorted descending
            'network_scores': {network_name: score, ...},
            'top_connections': [(roi_i, roi_j, score), ...],
        }
    """
    n_rois = config.CC200_N_ROIS

    # Aggregate Grad-CAM across samples → mean heatmap
    if gradcam_maps:
        mean_cam = np.mean(np.stack(gradcam_maps), axis=0)  # (200, 200)
    else:
        mean_cam = np.zeros((n_rois, n_rois))

    # Aggregate SHAP across samples
    if shap_values is not None and len(shap_values) > 0:
        mean_shap = np.abs(shap_values).mean(axis=0)  # should be (H, W)
        while mean_shap.ndim > 2:
            mean_shap = mean_shap.mean(axis=-1) if mean_shap.shape[-1] == 2 else mean_shap.squeeze()
    else:
        mean_shap = np.zeros((n_rois, n_rois))

    # Combined importance: average of normalized Grad-CAM and SHAP
    cam_norm = mean_cam / (mean_cam.max() + 1e-10)
    shap_norm = mean_shap / (mean_shap.max() + 1e-10)
    combined = (cam_norm + shap_norm) / 2.0

    # Per-ROI importance: sum of row (connections involving this ROI)
    roi_importance = combined.sum(axis=1) + combined.sum(axis=0)  # (200,)
    roi_importance /= (roi_importance.max() + 1e-10)

    # Rank ROIs
    roi_ranking = sorted(enumerate(roi_importance), key=lambda x: x[1], reverse=True)
    roi_scores = [(int(idx), float(score)) for idx, score in roi_ranking]

    # Network-level aggregation
    network_scores = {}
    for network, rois in CC200_NETWORK_MAP.items():
        valid_rois = [r for r in rois if r < n_rois]
        if valid_rois:
            network_scores[network] = float(np.mean([roi_importance[r] for r in valid_rois]))

    # Top connections (upper triangle)
    triu_r, triu_c = np.triu_indices(n_rois, k=1)
    conn_scores = combined[triu_r, triu_c]
    top_conn_idx = np.argsort(conn_scores)[::-1][:20]
    top_connections = [
        (int(triu_r[i]), int(triu_c[i]), float(conn_scores[i]),
         f"{_get_roi_network(triu_r[i])} ↔ {_get_roi_network(triu_c[i])}")
        for i in top_conn_idx
    ]

    log_info(f"fMRI ROI importance: Top ROI={roi_scores[0][0]} "
             f"(score={roi_scores[0][1]:.4f}), "
             f"Top Network={max(network_scores, key=network_scores.get)}")

    return {
        "roi_scores": roi_scores,
        "network_scores": network_scores,
        "top_connections": top_connections,
        "combined_importance_matrix": combined,
    }


@sanitize_errors("Failed to compute EEG channel importance.")
def compute_eeg_channel_importance(gradcam_maps: list, shap_values: np.ndarray = None) -> dict:
    """
    Compute per-channel importance from Grad-CAM heatmaps and/or SHAP values.

    Args:
        gradcam_maps: List of (64, 64) Grad-CAM heatmaps
        shap_values:  (N, 64, 64) SHAP importance values (optional)

    Returns:
        Dict with channel rankings and region-level aggregation.
    """
    n_channels = config.EEG_N_CHANNELS

    if gradcam_maps:
        mean_cam = np.mean(np.stack(gradcam_maps), axis=0)
    else:
        mean_cam = np.zeros((n_channels, n_channels))

    if shap_values is not None and len(shap_values) > 0:
        mean_shap = np.abs(shap_values).mean(axis=0)
        while mean_shap.ndim > 2:
            mean_shap = mean_shap.mean(axis=-1) if mean_shap.shape[-1] == 2 else mean_shap.squeeze()
    else:
        mean_shap = np.zeros((n_channels, n_channels))

    # Combined importance
    cam_norm = mean_cam / (mean_cam.max() + 1e-10)
    shap_norm = mean_shap / (mean_shap.max() + 1e-10)
    combined = (cam_norm + shap_norm) / 2.0

    # Per-channel importance
    ch_importance = combined.sum(axis=1) + combined.sum(axis=0)
    ch_importance /= (ch_importance.max() + 1e-10)

    ch_ranking = sorted(enumerate(ch_importance), key=lambda x: x[1], reverse=True)
    ch_scores = [(int(idx), float(score)) for idx, score in ch_ranking]

    # Region-level aggregation
    region_scores = {}
    for region, channels in EEG_REGION_MAP.items():
        valid_ch = [c for c in channels if c < n_channels]
        if valid_ch:
            region_scores[region] = float(np.mean([ch_importance[c] for c in valid_ch]))

    # Top connections
    triu_r, triu_c = np.triu_indices(n_channels, k=1)
    conn_scores = combined[triu_r, triu_c]
    top_conn_idx = np.argsort(conn_scores)[::-1][:20]
    top_connections = [
        (int(triu_r[i]), int(triu_c[i]), float(conn_scores[i]),
         f"{_get_channel_region(triu_r[i])} ↔ {_get_channel_region(triu_c[i])}")
        for i in top_conn_idx
    ]

    log_info(f"EEG channel importance: Top Ch={ch_scores[0][0]} "
             f"(score={ch_scores[0][1]:.4f}), "
             f"Top Region={max(region_scores, key=region_scores.get)}")

    return {
        "channel_scores": ch_scores,
        "region_scores": region_scores,
        "top_connections": top_connections,
        "combined_importance_matrix": combined,
    }


@sanitize_errors("Failed to compute cross-modal convergence analysis.")
def cross_modal_convergence_analysis(fmri_importance: dict, eeg_importance: dict) -> dict:
    """
    Compare brain regions flagged by both modalities for convergence.

    Maps fMRI ROIs to brain networks and EEG channels to scalp regions,
    then checks which brain areas are consistently highlighted by both
    modalities' XAI methods.

    Args:
        fmri_importance: Output from compute_fmri_roi_importance()
        eeg_importance:  Output from compute_eeg_channel_importance()

    Returns:
        Convergence report dict with overlapping regions and statistics.
    """
    fmri_networks = fmri_importance.get("network_scores", {})
    eeg_regions = eeg_importance.get("region_scores", {})

    # Map EEG scalp regions to approximate brain networks
    eeg_to_network = {
        "Frontal":   ["Default Mode Network (DMN)", "Frontoparietal Control",
                      "Salience / Ventral Attention"],
        "Central":   ["Somatomotor", "Dorsal Attention"],
        "Parietal":  ["Frontoparietal Control", "Dorsal Attention",
                      "Default Mode Network (DMN)"],
        "Temporal":  ["Limbic / Temporal", "Salience / Ventral Attention"],
        "Occipital": ["Visual"],
    }

    # Find convergent regions (high importance in both modalities)
    fmri_threshold = np.percentile(list(fmri_networks.values()), 50) if fmri_networks else 0
    eeg_threshold = np.percentile(list(eeg_regions.values()), 50) if eeg_regions else 0

    fmri_hot = {net for net, score in fmri_networks.items() if score >= fmri_threshold}
    eeg_hot_networks = set()
    for region, score in eeg_regions.items():
        if score >= eeg_threshold and region in eeg_to_network:
            eeg_hot_networks.update(eeg_to_network[region])

    convergent = fmri_hot & eeg_hot_networks
    fmri_only = fmri_hot - eeg_hot_networks
    eeg_only = eeg_hot_networks - fmri_hot

    convergence_score = len(convergent) / max(len(fmri_hot | eeg_hot_networks), 1)

    report = {
        "convergent_networks": sorted(convergent),
        "fmri_only_networks": sorted(fmri_only),
        "eeg_only_networks": sorted(eeg_only),
        "convergence_score": float(convergence_score),
        "fmri_network_scores": fmri_networks,
        "eeg_region_scores": eeg_regions,
    }

    log_info(f"Cross-modal convergence: {len(convergent)} shared networks "
             f"(score: {convergence_score:.2f})")
    log_info(f"  Convergent: {', '.join(sorted(convergent)) if convergent else 'None'}")
    log_info(f"  fMRI-only:  {', '.join(sorted(fmri_only)) if fmri_only else 'None'}")
    log_info(f"  EEG-only:   {', '.join(sorted(eeg_only)) if eeg_only else 'None'}")

    return report


@sanitize_errors("Failed to analyze attention weights.")
def compute_attention_weight_analysis(attn_weights_list: list) -> dict:
    """
    Analyze cross-attention weights from CrossAttentionFusion model.

    Args:
        attn_weights_list: List of attn_weight dicts from forward passes,
                           each containing 'fmri_to_eeg' and 'eeg_to_fmri' tensors.

    Returns:
        Analysis dict with per-head attention statistics.
    """
    if not attn_weights_list:
        return {"status": "no_attention_weights", "detail": "ConcatMLP fusion has no attention."}

    f2e_all = []
    e2f_all = []

    for aw in attn_weights_list:
        if aw is None:
            continue
        if "fmri_to_eeg" in aw:
            val = aw["fmri_to_eeg"]
            f2e_all.append(val.cpu().numpy() if hasattr(val, 'cpu') else np.array(val))
        if "eeg_to_fmri" in aw:
            val = aw["eeg_to_fmri"]
            e2f_all.append(val.cpu().numpy() if hasattr(val, 'cpu') else np.array(val))

    if not f2e_all:
        return {"status": "empty", "detail": "No valid attention weights found."}

    f2e = np.concatenate(f2e_all, axis=0)  # (N_total, num_heads)
    e2f = np.concatenate(e2f_all, axis=0)

    analysis = {
        "status": "computed",
        "fmri_to_eeg": {
            "mean_per_head": f2e.mean(axis=0).tolist(),
            "std_per_head": f2e.std(axis=0).tolist(),
            "overall_mean": float(f2e.mean()),
        },
        "eeg_to_fmri": {
            "mean_per_head": e2f.mean(axis=0).tolist(),
            "std_per_head": e2f.std(axis=0).tolist(),
            "overall_mean": float(e2f.mean()),
        },
        "dominant_direction": "fMRI->EEG" if f2e.mean() > e2f.mean() else "EEG->fMRI",
        "raw_f2e": f2e,
        "raw_e2f": e2f,
    }

    log_info(f"Attention analysis: dominant direction = {analysis['dominant_direction']} "
             f"(f2e={f2e.mean():.3f}, e2f={e2f.mean():.3f})")

    return analysis
