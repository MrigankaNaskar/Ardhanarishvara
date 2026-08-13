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

# Ensure all directories exist
for path in [
    DATA_DIR, FMRI_DIR, EEG_DIR, PROCESSED_DIR, PROCESSED_FMRI_DIR, PROCESSED_EEG_DIR,
    MANIFEST_DIR, LOG_DIR, MODELS_DIR, FMRI_MODEL_DIR, EEG_MODEL_DIR, FUSION_DIR, XAI_DIR, NOTEBOOKS_DIR
]:
    os.makedirs(path, exist_ok=True)

# Dataset configurations
CC200_N_ROIS = 200
EEG_N_CHANNELS = 64
EMBEDDING_DIM = 128
RANDOM_SEED = 42
