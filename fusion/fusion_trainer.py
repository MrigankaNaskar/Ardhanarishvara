"""
Fusion Model Trainer for Ardhanarishvara Phase 4.
Trains the fusion model and simultaneously tracks fMRI-only, EEG-only, and fusion
accuracy side-by-side in every run. Generates the Phase 4 deliverable comparison table.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score

import config
from fusion.fusion_module import ConcatMLPFusion, CrossAttentionFusion, UnimodalClassifier
from fusion.unpaired_sampler import create_unpaired_splits
from fusion.embedding_extractor import get_device
from security.sanitized_logging import sanitize_errors, log_info


@sanitize_errors("Failed to train unimodal baseline classifier.")
def _train_unimodal_baseline(embeds_train: np.ndarray, labels_train: np.ndarray,
                              embeds_val: np.ndarray, labels_val: np.ndarray,
                              modality_name: str, epochs: int = 30,
                              lr: float = 5e-4, seed: int = 42):
    """
    Train a simple linear classifier on single-modality embeddings for baseline comparison.

    Returns:
        model: Trained UnimodalClassifier
        best_acc: Best validation accuracy
        best_auc: AUC at best accuracy epoch
        val_preds: Predictions on validation set at best epoch
    """
    device = get_device()
    torch.manual_seed(seed)

    model = UnimodalClassifier(embed_dim=config.EMBEDDING_DIM, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train = torch.tensor(embeds_train, dtype=torch.float32)
    y_train = torch.tensor(labels_train, dtype=torch.long)
    X_val = torch.tensor(embeds_val, dtype=torch.float32).to(device)
    y_val_np = labels_val.copy()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config.FUSION_BATCH_SIZE, shuffle=True
    )

    best_acc = 0.0
    best_auc = 0.5
    best_preds = None

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val, _ = model(X_val)
            probs_val = torch.softmax(logits_val, dim=1).cpu().numpy()
            preds_val = np.argmax(probs_val, axis=1)

            acc = accuracy_score(y_val_np, preds_val)
            try:
                auc = roc_auc_score(y_val_np, probs_val[:, 1])
            except (ValueError, IndexError):
                auc = 0.5

            if acc >= best_acc:
                best_acc = acc
                best_auc = auc
                best_preds = preds_val.copy()

    log_info(f"[{modality_name}-only Baseline] Best Val Acc: {best_acc*100:.2f}% | AUC: {best_auc:.4f}")
    return model, best_acc, best_auc, best_preds


@sanitize_errors("Failed to train fusion model.")
def train_fusion_model(fmri_embeddings: np.ndarray, fmri_labels: np.ndarray,
                       eeg_embeddings: np.ndarray, eeg_labels: np.ndarray,
                       fusion_type: str = "concat", seed: int = None):
    """
    Train fusion model with 3-way comparison tracking (fMRI-only vs EEG-only vs Fusion).

    Args:
        fmri_embeddings: (N_fmri, 128) embeddings from frozen fMRI encoder
        fmri_labels:     (N_fmri,) labels (0=TD, 1=ASD)
        eeg_embeddings:  (N_eeg, 128) embeddings from frozen EEG encoder
        eeg_labels:      (N_eeg,) labels
        fusion_type:     'concat' (ConcatMLPFusion) or 'attention' (CrossAttentionFusion)
        seed:            Random seed (defaults to config.RANDOM_SEED)

    Returns:
        fusion_model:       Trained fusion model
        comparison_history: Dict with per-epoch metrics and baseline comparisons
    """
    if seed is None:
        seed = config.RANDOM_SEED
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    log_info(f"=== Phase 4: Training Fusion Model (type={fusion_type}) on {device} ===")

    # ---- 1. Create unpaired data splits ----
    train_dataset, val_dataset, unimodal_splits = create_unpaired_splits(
        fmri_embeddings, fmri_labels, eeg_embeddings, eeg_labels,
        n_pairings=config.N_RANDOM_PAIRINGS, seed=seed
    )

    train_loader = DataLoader(train_dataset, batch_size=config.FUSION_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.FUSION_BATCH_SIZE, shuffle=False)

    # ---- 2. Train unimodal baselines for comparison ----
    log_info("Training fMRI-only baseline classifier...")
    fmri_baseline, fmri_acc, fmri_auc, fmri_preds = _train_unimodal_baseline(
        unimodal_splits["fmri"]["train_embeds"], unimodal_splits["fmri"]["train_labels"],
        unimodal_splits["fmri"]["val_embeds"], unimodal_splits["fmri"]["val_labels"],
        modality_name="fMRI", epochs=config.FUSION_EPOCHS, lr=config.FUSION_LR, seed=seed
    )

    log_info("Training EEG-only baseline classifier...")
    eeg_baseline, eeg_acc, eeg_auc, eeg_preds = _train_unimodal_baseline(
        unimodal_splits["eeg"]["train_embeds"], unimodal_splits["eeg"]["train_labels"],
        unimodal_splits["eeg"]["val_embeds"], unimodal_splits["eeg"]["val_labels"],
        modality_name="EEG", epochs=config.FUSION_EPOCHS, lr=config.FUSION_LR, seed=seed
    )

    # ---- 3. Build fusion model ----
    if fusion_type == "concat":
        fusion_model = ConcatMLPFusion(
            fmri_dim=config.EMBEDDING_DIM, eeg_dim=config.EMBEDDING_DIM,
            hidden_dim=config.FUSION_HIDDEN_DIM, num_classes=2,
            dropout=config.FUSION_DROPOUT
        ).to(device)
    elif fusion_type == "attention":
        fusion_model = CrossAttentionFusion(
            embed_dim=config.EMBEDDING_DIM, num_heads=config.ATTENTION_HEADS,
            hidden_dim=config.FUSION_HIDDEN_DIM, num_classes=2,
            dropout=config.FUSION_DROPOUT
        ).to(device)
    else:
        raise ValueError(f"Unknown fusion_type: '{fusion_type}'. Use 'concat' or 'attention'.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=config.FUSION_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.FUSION_EPOCHS)

    n_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    log_info(f"Fusion model ({fusion_type}): {n_params:,} trainable parameters")

    # ---- 4. Training loop with per-epoch tracking ----
    comparison_history = {
        "fusion_type": fusion_type,
        "fusion_train_loss": [],
        "fusion_val_acc": [],
        "fusion_val_auc": [],
        "fmri_only_acc": float(fmri_acc),
        "fmri_only_auc": float(fmri_auc),
        "eeg_only_acc": float(eeg_acc),
        "eeg_only_auc": float(eeg_auc),
        "best_fusion_acc": 0.0,
        "best_fusion_auc": 0.5,
        "n_fusion_params": n_params,
    }

    best_fusion_acc = 0.0
    best_fusion_preds = None

    for epoch in range(config.FUSION_EPOCHS):
        # -- Train --
        fusion_model.train()
        running_loss = 0.0
        n_train_samples = 0

        for fmri_b, eeg_b, y_b in train_loader:
            fmri_b = fmri_b.to(device)
            eeg_b = eeg_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits, _, _ = fusion_model(fmri_b, eeg_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * y_b.size(0)
            n_train_samples += y_b.size(0)

        scheduler.step()
        epoch_loss = running_loss / max(n_train_samples, 1)
        comparison_history["fusion_train_loss"].append(epoch_loss)

        # -- Validate --
        fusion_model.eval()
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for fmri_b, eeg_b, y_b in val_loader:
                fmri_b = fmri_b.to(device)
                eeg_b = eeg_b.to(device)
                logits, _, _ = fusion_model(fmri_b, eeg_b)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_probs.extend(probs[:, 1])
                all_targets.extend(y_b.numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        try:
            val_auc = roc_auc_score(all_targets, all_probs)
        except (ValueError, IndexError):
            val_auc = 0.5

        comparison_history["fusion_val_acc"].append(val_acc)
        comparison_history["fusion_val_auc"].append(val_auc)

        if val_acc >= best_fusion_acc:
            best_fusion_acc = val_acc
            comparison_history["best_fusion_acc"] = float(val_acc)
            comparison_history["best_fusion_auc"] = float(val_auc)
            best_fusion_preds = np.array(all_preds)
            # Save best checkpoint
            save_path = os.path.join(config.FUSION_DIR, "fusion_model.pt")
            torch.save(fusion_model.state_dict(), save_path)

        # Periodic logging with 3-way comparison
        if (epoch + 1) % 5 == 0 or epoch == config.FUSION_EPOCHS - 1:
            log_info(f"Epoch {epoch+1}/{config.FUSION_EPOCHS} | "
                     f"Loss: {epoch_loss:.4f} | "
                     f"Fusion: {val_acc*100:.1f}% | "
                     f"fMRI-only: {fmri_acc*100:.1f}% | "
                     f"EEG-only: {eeg_acc*100:.1f}%")

    # ---- 5. Save comparison history ----
    log_info(f"Best fusion ({fusion_type}) accuracy: {best_fusion_acc*100:.2f}% | "
             f"fMRI-only: {fmri_acc*100:.2f}% | EEG-only: {eeg_acc*100:.2f}%")

    history_path = os.path.join(config.RESULTS_DIR, f"fusion_comparison_{fusion_type}.json")
    _save_history(comparison_history, history_path)

    # Store predictions for downstream statistical tests
    comparison_history["_val_targets"] = np.array(all_targets)
    comparison_history["_fusion_val_preds"] = best_fusion_preds
    comparison_history["_fmri_val_preds"] = fmri_preds
    comparison_history["_eeg_val_preds"] = eeg_preds

    return fusion_model, comparison_history


def _save_history(history: dict, path: str):
    """Serialize comparison history to JSON (skip non-serializable entries)."""
    serializable = {}
    for k, v in history.items():
        if k.startswith("_"):
            continue  # Skip internal arrays
        if isinstance(v, (list, tuple)):
            serializable[k] = [float(x) if isinstance(x, (float, np.floating)) else x for x in v]
        elif isinstance(v, (float, np.floating)):
            serializable[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            serializable[k] = int(v)
        else:
            serializable[k] = v

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    log_info(f"Saved comparison history to {os.path.basename(path)}")


def generate_comparison_table(history: dict, save_path: str = None) -> str:
    """
    Generate formatted markdown comparison table from training history.
    This is the primary Phase 4 deliverable.
    """
    rows = [
        ["fMRI-only (Linear)", f"{history['fmri_only_acc']*100:.2f}%",
         f"{history['fmri_only_auc']:.4f}"],
        ["EEG-only (Linear)", f"{history['eeg_only_acc']*100:.2f}%",
         f"{history['eeg_only_auc']:.4f}"],
        [f"Fusion ({history['fusion_type'].title()})",
         f"{history['best_fusion_acc']*100:.2f}%",
         f"{history['best_fusion_auc']:.4f}"],
    ]

    # Build markdown table
    header = "| Model | Val Accuracy | Val AUC |"
    separator = "|:---|:---|:---|"
    table_lines = [header, separator]
    for row in rows:
        table_lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")

    table_str = "\n".join(table_lines)
    log_info(f"\n{'='*60}\n  Phase 4 Fusion Comparison ({history['fusion_type']})\n{'='*60}\n{table_str}")

    if save_path is None:
        save_path = os.path.join(config.TABLES_DIR,
                                 f"comparison_table_{history['fusion_type']}.md")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(f"# Fusion Model Comparison — {history['fusion_type'].title()}\n\n")
        f.write(table_str + "\n\n")
        f.write(f"*Fusion model parameters: {history.get('n_fusion_params', 'N/A'):,}*\n")

    log_info(f"Saved comparison table to {os.path.basename(save_path)}")
    return table_str
