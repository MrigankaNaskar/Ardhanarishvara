"""
Ablation Study Runner for Ardhanarishvara Phase 6.
Systematically evaluates: fMRI-only, EEG-only, Fusion (Concat), Fusion (Attention)
across multiple seeds, computing mean ± std for accuracy, AUC, sensitivity, specificity.
"""

import os
import numpy as np
import pandas as pd
import torch

import config
from fusion.fusion_trainer import train_fusion_model, _train_unimodal_baseline
from fusion.unpaired_sampler import create_unpaired_splits
from fusion.embedding_extractor import get_device
from security.sanitized_logging import sanitize_errors, log_info


# Ablation configuration matrix
ABLATION_CONFIGS = [
    {"ablation_id": "fmri_only",     "fmri": True,  "eeg": False, "fusion_type": None},
    {"ablation_id": "eeg_only",      "fmri": False, "eeg": True,  "fusion_type": None},
    {"ablation_id": "fusion_concat", "fmri": True,  "eeg": True,  "fusion_type": "concat"},
    {"ablation_id": "fusion_attn",   "fmri": True,  "eeg": True,  "fusion_type": "attention"},
]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Compute accuracy, AUC, sensitivity, specificity from predictions."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    except (ValueError, IndexError):
        auc = 0.5

    # Sensitivity (recall for ASD=1) and Specificity (recall for TD=0)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


@sanitize_errors("Failed to run single ablation.")
def _run_single_ablation(ablation_config: dict,
                          fmri_embeddings: np.ndarray, fmri_labels: np.ndarray,
                          eeg_embeddings: np.ndarray, eeg_labels: np.ndarray,
                          seed: int) -> dict:
    """Run a single ablation configuration with a specific seed."""
    ablation_id = ablation_config["ablation_id"]
    fusion_type = ablation_config["fusion_type"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    if fusion_type is not None:
        # Fusion model — train via fusion_trainer
        _, history = train_fusion_model(
            fmri_embeddings, fmri_labels,
            eeg_embeddings, eeg_labels,
            fusion_type=fusion_type, seed=seed
        )

        targets = history.get("_val_targets")
        preds = history.get("_fusion_val_preds")

        if targets is not None and preds is not None:
            metrics = _compute_metrics(targets, preds)
            metrics["accuracy"] = history["best_fusion_acc"]
            metrics["auc"] = history["best_fusion_auc"]
        else:
            metrics = {"accuracy": history["best_fusion_acc"],
                       "auc": history["best_fusion_auc"],
                       "sensitivity": 0.0, "specificity": 0.0}

    else:
        # Unimodal baseline
        _, _, unimodal_splits = create_unpaired_splits(
            fmri_embeddings, fmri_labels, eeg_embeddings, eeg_labels,
            n_pairings=1, seed=seed
        )

        if ablation_config["fmri"]:
            splits = unimodal_splits["fmri"]
        else:
            splits = unimodal_splits["eeg"]

        _, best_acc, best_auc, best_preds = _train_unimodal_baseline(
            splits["train_embeds"], splits["train_labels"],
            splits["val_embeds"], splits["val_labels"],
            modality_name=ablation_id, epochs=config.FUSION_EPOCHS,
            lr=config.FUSION_LR, seed=seed
        )

        if best_preds is not None:
            metrics = _compute_metrics(splits["val_labels"], best_preds)
            metrics["accuracy"] = best_acc
            metrics["auc"] = best_auc
        else:
            metrics = {"accuracy": best_acc, "auc": best_auc,
                       "sensitivity": 0.0, "specificity": 0.0}

    metrics["ablation_id"] = ablation_id
    metrics["seed"] = seed

    return metrics


@sanitize_errors("Failed to run full ablation study.")
def run_full_ablation_study(fmri_embeddings: np.ndarray, fmri_labels: np.ndarray,
                            eeg_embeddings: np.ndarray, eeg_labels: np.ndarray,
                            seeds: list = None) -> pd.DataFrame:
    """
    Run the complete ablation study across all configurations and seeds.

    Args:
        fmri_embeddings: (N_fmri, 128) embeddings from frozen fMRI encoder
        fmri_labels:     (N_fmri,) labels
        eeg_embeddings:  (N_eeg, 128) embeddings from frozen EEG encoder
        eeg_labels:      (N_eeg,) labels
        seeds:           List of random seeds (defaults to config.ABLATION_SEEDS)

    Returns:
        DataFrame with all ablation results (per-seed and aggregated)
    """
    if seeds is None:
        seeds = config.ABLATION_SEEDS

    log_info(f"=== Phase 6: Running Full Ablation Study ===")
    log_info(f"Configurations: {len(ABLATION_CONFIGS)} | Seeds: {len(seeds)} | "
             f"Total runs: {len(ABLATION_CONFIGS) * len(seeds)}")

    all_results = []

    for cfg in ABLATION_CONFIGS:
        log_info(f"\n--- Ablation: {cfg['ablation_id']} ---")
        for seed in seeds:
            log_info(f"  Seed {seed}...")
            result = _run_single_ablation(
                cfg, fmri_embeddings, fmri_labels,
                eeg_embeddings, eeg_labels, seed
            )
            all_results.append(result)

    # Build DataFrame
    df = pd.DataFrame(all_results)

    # Aggregate: mean ± std per ablation config
    agg_rows = []
    for ablation_id in df["ablation_id"].unique():
        subset = df[df["ablation_id"] == ablation_id]
        agg_rows.append({
            "ablation_id": ablation_id,
            "accuracy_mean": subset["accuracy"].mean(),
            "accuracy_std": subset["accuracy"].std(),
            "auc_mean": subset["auc"].mean(),
            "auc_std": subset["auc"].std(),
            "sensitivity_mean": subset["sensitivity"].mean(),
            "sensitivity_std": subset["sensitivity"].std(),
            "specificity_mean": subset["specificity"].mean(),
            "specificity_std": subset["specificity"].std(),
            "n_seeds": len(subset),
        })

    agg_df = pd.DataFrame(agg_rows)

    # Log summary
    log_info("\n=== Ablation Study Summary ===")
    for _, row in agg_df.iterrows():
        log_info(f"  {row['ablation_id']:20s} | "
                 f"Acc: {row['accuracy_mean']*100:.2f}±{row['accuracy_std']*100:.2f}% | "
                 f"AUC: {row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
                 f"Sens: {row['sensitivity_mean']*100:.1f}% | "
                 f"Spec: {row['specificity_mean']*100:.1f}%")

    # Save results
    raw_path = os.path.join(config.TABLES_DIR, "ablation_raw_results.csv")
    agg_path = os.path.join(config.TABLES_DIR, "ablation_summary.csv")
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    log_info(f"Saved raw results to {os.path.basename(raw_path)}")
    log_info(f"Saved aggregated summary to {os.path.basename(agg_path)}")

    return agg_df
