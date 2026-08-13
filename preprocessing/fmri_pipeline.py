"""
fMRI CPAC Preprocessing & CC200 Connectivity Matrix Pipeline for Ardhanarishvara.
Port CPAC preprocessing + CC200 parcellation + connectivity matrix generation.
Caches matrices as .npy/.h5 so CPAC loading never reruns.
Enforces strict security validation, rate-limiting, and error sanitization.
"""

import os
import numpy as np
import h5py
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
import config
from security.sanitized_logging import sanitize_errors, log_info
from security.validation import validate_connectivity_matrix, validate_file_path
from security.rate_limiter import rate_limit_downloads


@rate_limit_downloads()
def fetch_abide_subject_data(n_subjects: int = 1):
    """Fetch ABIDE-I preprocessed dataset (CPAC pipeline, CC200 atlas) with rate limiting."""
    log_info(f"Fetching ABIDE-I preprocessed dataset (CPAC pipeline, n_subjects={n_subjects})...")
    abide_data = datasets.fetch_abide_pcp(
        data_dir=config.FMRI_DIR,
        pipeline="cpac",
        bandpass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_cc200"],
        n_subjects=n_subjects
    )
    return abide_data


@sanitize_errors("Failed to compute fMRI CC200 connectivity matrix.")
def process_fmri_subject(timeseries_file_or_data, subject_id: str = "sub_001") -> np.ndarray:
    """
    Process CC200 ROI timeseries into a 200x200 Fisher z-transformed correlation connectivity matrix.
    Uses caching (.npy / .h5) to prevent duplicate preprocessing.
    """
    cache_npy_path = os.path.join(config.PROCESSED_FMRI_DIR, f"{subject_id}_cc200.npy")
    cache_h5_path = os.path.join(config.PROCESSED_FMRI_DIR, f"{subject_id}_cc200.h5")

    # Check cache first
    if os.path.exists(cache_npy_path):
        validate_file_path(cache_npy_path)
        log_info(f"Loading cached fMRI connectivity matrix for {subject_id} from {cache_npy_path}")
        matrix = np.load(cache_npy_path)
        return validate_connectivity_matrix(matrix, expected_dim=(200, 200), name=f"fMRI CC200 Matrix ({subject_id})")

    # Load timeseries data
    if isinstance(timeseries_file_or_data, str):
        validate_file_path(timeseries_file_or_data)
        ts_data = np.loadtxt(timeseries_file_or_data)
    elif isinstance(timeseries_file_or_data, np.ndarray):
        ts_data = timeseries_file_or_data
    else:
        # If passed from nilearn fetch_abide_pcp rois_cc200 list
        ts_data = np.asarray(timeseries_file_or_data)

    # Ensure shape is (T, 200)
    if ts_data.ndim == 2 and ts_data.shape[0] == 200 and ts_data.shape[1] != 200:
        ts_data = ts_data.T

    # Compute Pearson Correlation connectivity matrix (Vectorized)
    corr_matrix = np.corrcoef(ts_data.T)  # (200, 200)

    # Apply Fisher z-transform: z = arctanh(r), clipping r to (-0.9999, 0.9999) to avoid inf on diagonals
    clipped_corr = np.clip(corr_matrix, -0.9999, 0.9999)
    z_matrix = np.arctanh(clipped_corr)
    # Set self-correlation diagonal to 1.0 (or z-transformed equivalent)
    np.fill_diagonal(z_matrix, 1.0)

    # Security validation
    validated_matrix = validate_connectivity_matrix(z_matrix, expected_dim=(200, 200), name=f"fMRI CC200 Matrix ({subject_id})")

    # Cache to .npy and .h5
    np.save(cache_npy_path, validated_matrix)
    with h5py.File(cache_h5_path, "w") as f:
        f.create_dataset("connectivity", data=validated_matrix)

    log_info(f"Successfully generated and cached fMRI CC200 matrix for {subject_id} (Shape: {validated_matrix.shape})")
    return validated_matrix
