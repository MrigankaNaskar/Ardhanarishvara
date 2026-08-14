"""
Global Project Configuration for Ardhanarishvara.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
FMRI_DIR = os.path.join(DATA_DIR, "fmri")
EEG_DIR = os.path.join(DATA_DIR, "eeg")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", os.path.join(DATA_DIR, "processed"))
PROCESSED_FMRI_DIR = os.path.join(PROCESSED_DIR, "fmri")
PROCESSED_EEG_DIR = os.path.join(PROCESSED_DIR, "eeg")
MANIFEST_DIR = os.getenv("MANIFEST_DIR", os.path.join(DATA_DIR, "manifests"))
LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "logs"))

MODELS_DIR = os.path.join(BASE_DIR, "models")
FMRI_MODEL_DIR = os.path.join(MODELS_DIR, "fmri")
EEG_MODEL_DIR = os.path.join(MODELS_DIR, "eeg")
FUSION_DIR = os.path.join(BASE_DIR, "fusion")
XAI_DIR = os.path.join(BASE_DIR, "xai")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# Phase 4-6 and output directories
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Ensure all directories exist
for path in [
    DATA_DIR, FMRI_DIR, EEG_DIR, PROCESSED_DIR, PROCESSED_FMRI_DIR, PROCESSED_EEG_DIR,
    MANIFEST_DIR, LOG_DIR, MODELS_DIR, FMRI_MODEL_DIR, EEG_MODEL_DIR, FUSION_DIR, XAI_DIR, NOTEBOOKS_DIR,
    EVALUATION_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR
]:
    os.makedirs(path, exist_ok=True)

# Dataset configurations
CC200_N_ROIS = 200
EEG_N_CHANNELS = 16            # King Abdulaziz University (KAU) 16-electrode 10-20 standard montage
EMBEDDING_DIM = 128
RANDOM_SEED = 42

# Phase 4 — Fusion Layer
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.3
FUSION_LR = 5e-4
FUSION_EPOCHS = 30
FUSION_BATCH_SIZE = 16
N_RANDOM_PAIRINGS = 5           # Augmentation factor for unpaired label-conditioned sampling
ATTENTION_HEADS = 4             # Cross-attention heads
ATTENTION_DIM = EMBEDDING_DIM   # Must match EMBEDDING_DIM

# Phase 5 — XAI / Explainability
GRADCAM_TARGET_LAYER = "conv3"  # Target layer for Grad-CAM in both encoders
SHAP_N_BACKGROUND = 10         # SHAP background samples (fast & representative)
SHAP_N_EXPLAIN = 10            # SHAP explanation samples

# Phase 6 — Evaluation & Ablations
BOOTSTRAP_N_ITERATIONS = 10000  # Paired bootstrap resamples
BOOTSTRAP_CI_LEVEL = 0.95       # 95% confidence interval
ABLATION_SEEDS = [42, 123, 456, 789, 1024]  # Multiple seeds for stability

