"""
Scaffolding for Phase 4 Multimodal Fusion Model.
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    Placeholder Multimodal Fusion Module for combining 128-dim fMRI and 128-dim EEG embeddings.
    """
    def __init__(self, fmri_dim: int = 128, eeg_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.eeg_dim = eeg_dim
        self.fc = nn.Linear(fmri_dim + eeg_dim, num_classes)

    def forward(self, fmri_embed: torch.Tensor, eeg_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([fmri_embed, eeg_embed], dim=1)
        return self.fc(combined)
