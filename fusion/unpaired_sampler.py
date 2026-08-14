"""
Unpaired Multimodal Sampling Strategy for Ardhanarishvara Phase 4.
Handles the zero-overlap cohort pairing problem (ABIDE-I fMRI vs KAU EEG)
using label-conditioned random pairing with augmentation for fusion training.

Strategy: For each diagnosis class (ASD/TD), randomly pair fMRI embeddings
with EEG embeddings N_RANDOM_PAIRINGS times. This creates synthetic multimodal
pairs while respecting diagnostic labels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from security.sanitized_logging import sanitize_errors, log_info


class UnpairedMultimodalDataset(Dataset):
    """
    PyTorch Dataset for unpaired multimodal fusion training.
    Creates label-conditioned random pairings between fMRI and EEG embeddings.

    Each sample is a tuple: (fmri_embedding, eeg_embedding, shared_label)
    """

    def __init__(self, fmri_embeddings: np.ndarray, fmri_labels: np.ndarray,
                 eeg_embeddings: np.ndarray, eeg_labels: np.ndarray,
                 n_pairings: int = 5, seed: int = 42):
        """
        Args:
            fmri_embeddings: (N_fmri, 128) fMRI embeddings
            fmri_labels:     (N_fmri,) labels (0=TD, 1=ASD)
            eeg_embeddings:  (N_eeg, 128) EEG embeddings
            eeg_labels:      (N_eeg,) labels (0=TD, 1=ASD)
            n_pairings:      Number of random EEG partners per fMRI subject
            seed:            Random seed for reproducibility
        """
        super().__init__()
        self.fmri_tensors = []
        self.eeg_tensors = []
        self.label_tensors = []

        rng = np.random.RandomState(seed)

        for label in [0, 1]:
            fmri_class = fmri_embeddings[fmri_labels == label]
            eeg_class = eeg_embeddings[eeg_labels == label]

            if len(fmri_class) == 0 or len(eeg_class) == 0:
                label_name = "ASD" if label == 1 else "TD"
                log_info(f"WARNING: No subjects for class {label_name} in one modality. "
                         f"fMRI: {len(fmri_class)}, EEG: {len(eeg_class)}. Skipping.")
                continue

            # Create n_pairings random pairings per fMRI subject
            for _ in range(n_pairings):
                for fi in range(len(fmri_class)):
                    ei = rng.randint(0, len(eeg_class))
                    self.fmri_tensors.append(
                        torch.tensor(fmri_class[fi], dtype=torch.float32))
                    self.eeg_tensors.append(
                        torch.tensor(eeg_class[ei], dtype=torch.float32))
                    self.label_tensors.append(label)

        self.label_tensors = torch.tensor(self.label_tensors, dtype=torch.long)

        n_asd = int((self.label_tensors == 1).sum())
        n_td = int((self.label_tensors == 0).sum())
        log_info(f"UnpairedMultimodalDataset: {len(self)} samples "
                 f"(ASD: {n_asd}, TD: {n_td}, pairings: {n_pairings}x)")

    def __len__(self):
        return len(self.label_tensors)

    def __getitem__(self, idx):
        return self.fmri_tensors[idx], self.eeg_tensors[idx], self.label_tensors[idx]


@sanitize_errors("Failed to create unpaired data splits.")
def create_unpaired_splits(fmri_embeddings: np.ndarray, fmri_labels: np.ndarray,
                           eeg_embeddings: np.ndarray, eeg_labels: np.ndarray,
                           n_pairings: int = 5, val_ratio: float = 0.2,
                           seed: int = 42):
    """
    Create train/val splits for fusion training and unimodal baselines.

    Strategy:
      1. Randomly partition each modality's subjects into train/val
      2. Create UnpairedMultimodalDataset for fusion from the partitioned embeddings
      3. Return unimodal splits for baseline classifier comparison

    Args:
        fmri_embeddings: (N_fmri, 128) fMRI embeddings
        fmri_labels:     (N_fmri,) labels
        eeg_embeddings:  (N_eeg, 128) EEG embeddings
        eeg_labels:      (N_eeg,) labels
        n_pairings:      Augmentation factor for random pairing
        val_ratio:       Fraction held out for validation
        seed:            Random seed

    Returns:
        train_dataset:   UnpairedMultimodalDataset for training
        val_dataset:     UnpairedMultimodalDataset for validation
        unimodal_splits: Dict with train/val arrays for each modality
    """
    rng = np.random.RandomState(seed)

    # --- Stratified split for fMRI subjects ---
    fmri_train_idx, fmri_val_idx = [], []
    for cls in [0, 1]:
        cls_idx = np.where(fmri_labels == cls)[0]
        rng.shuffle(cls_idx)
        val_count = max(1, int(len(cls_idx) * val_ratio))
        fmri_val_idx.extend(cls_idx[:val_count])
        fmri_train_idx.extend(cls_idx[val_count:])
    fmri_train_idx = np.array(fmri_train_idx)
    fmri_val_idx = np.array(fmri_val_idx)

    # --- Stratified split for EEG subjects ---
    eeg_train_idx, eeg_val_idx = [], []
    for cls in [0, 1]:
        cls_idx = np.where(eeg_labels == cls)[0]
        rng.shuffle(cls_idx)
        val_count = max(1, int(len(cls_idx) * val_ratio))
        eeg_val_idx.extend(cls_idx[:val_count])
        eeg_train_idx.extend(cls_idx[val_count:])
    eeg_train_idx = np.array(eeg_train_idx)
    eeg_val_idx = np.array(eeg_val_idx)

    # --- Create multimodal fusion datasets ---
    train_dataset = UnpairedMultimodalDataset(
        fmri_embeddings[fmri_train_idx], fmri_labels[fmri_train_idx],
        eeg_embeddings[eeg_train_idx], eeg_labels[eeg_train_idx],
        n_pairings=n_pairings, seed=seed
    )
    val_dataset = UnpairedMultimodalDataset(
        fmri_embeddings[fmri_val_idx], fmri_labels[fmri_val_idx],
        eeg_embeddings[eeg_val_idx], eeg_labels[eeg_val_idx],
        n_pairings=n_pairings, seed=seed + 1
    )

    # --- Unimodal splits for baseline comparison ---
    unimodal_splits = {
        "fmri": {
            "train_embeds": fmri_embeddings[fmri_train_idx],
            "train_labels": fmri_labels[fmri_train_idx],
            "val_embeds": fmri_embeddings[fmri_val_idx],
            "val_labels": fmri_labels[fmri_val_idx],
        },
        "eeg": {
            "train_embeds": eeg_embeddings[eeg_train_idx],
            "train_labels": eeg_labels[eeg_train_idx],
            "val_embeds": eeg_embeddings[eeg_val_idx],
            "val_labels": eeg_labels[eeg_val_idx],
        }
    }

    log_info(f"Data splits — Fusion train: {len(train_dataset)}, val: {len(val_dataset)} | "
             f"fMRI train: {len(fmri_train_idx)}, val: {len(fmri_val_idx)} | "
             f"EEG train: {len(eeg_train_idx)}, val: {len(eeg_val_idx)}")

    return train_dataset, val_dataset, unimodal_splits
