import numpy as np
from typing import Dict, List, Set, Tuple
from pathlib import Path

# Try to import project modules
try:
    from config import get_config
    from data import FMRIDataProcessor, SMRIDataProcessor, match_multimodal_subjects
    PROJECT_MODULES_AVAILABLE = True
except ImportError:
    PROJECT_MODULES_AVAILABLE = False


def get_matched_subject_ids(
    fmri_roi_dir: str = None,
    smri_data_path: str = None,
    phenotypic_file: str = None,
    verbose: bool = True
) -> Set[str]:
    """
    Get the set of subject IDs that have both fMRI and sMRI data.
    
    Args:
        fmri_roi_dir: Path to fMRI ROI directory
        smri_data_path: Path to sMRI data directory
        phenotypic_file: Path to phenotypic file
        verbose: Whether to print progress
        
    Returns:
        Set of subject IDs present in both modalities
    """
    if not PROJECT_MODULES_AVAILABLE:
        raise ImportError("Project modules not available for subject matching")
    
    # Use default paths if not provided
    if fmri_roi_dir is None or smri_data_path is None or phenotypic_file is None:
        try:
            config = get_config('cross_attention')
            fmri_roi_dir = fmri_roi_dir or config.fmri_roi_dir
            smri_data_path = smri_data_path or config.smri_data_path
            phenotypic_file = phenotypic_file or config.phenotypic_file
        except:
            # Fallback for Google Colab
            fmri_roi_dir = fmri_roi_dir or "/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200"
            smri_data_path = smri_data_path or "/content/drive/MyDrive/processed_smri_data"
            phenotypic_file = phenotypic_file or "/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
    
    if verbose:
        print("Matching subjects")
    
    # Load fMRI subjects
    fmri_processor = FMRIDataProcessor(
        roi_dir=Path(fmri_roi_dir),
        pheno_file=Path(phenotypic_file),
        n_rois=200
    )
    fmri_data = fmri_processor.load_all_subjects()
    fmri_subjects = set(fmri_data.keys())
    
    # Load sMRI subjects
    smri_processor = SMRIDataProcessor(
        data_path=Path(smri_data_path),
        feature_selection_k=None,  # Don't select features, just get subjects
        scaler_type='standard'
    )
    smri_data = smri_processor.load_all_subjects(Path(phenotypic_file))
    smri_subjects = set(smri_data.keys())
    
    # Find intersection
    matched_subjects = fmri_subjects & smri_subjects
    
    if verbose:
        print(f"Matched {len(matched_subjects)} subjects")
    
    return matched_subjects


def filter_data_by_subjects(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: List[str],
    target_subject_ids: Set[str],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Filter features, labels, and subject_ids to include only target subjects.
    
    Args:
        features: Feature array
        labels: Label array
        subject_ids: List of subject IDs corresponding to the data
        target_subject_ids: Set of subject IDs to keep
        verbose: Whether to print filtering results
        
    Returns:
        Tuple of (filtered_features, filtered_labels, filtered_subject_ids)
    """
    # Find indices of subjects to keep
    keep_indices = []
    kept_subject_ids = []
    
    for i, sub_id in enumerate(subject_ids):
        if sub_id in target_subject_ids:
            keep_indices.append(i)
            kept_subject_ids.append(sub_id)
    
    # Filter data
    filtered_features = features[keep_indices]
    filtered_labels = labels[keep_indices]
    
    if verbose:
        print(f"Filtered to {len(kept_subject_ids)} subjects")
    
    return filtered_features, filtered_labels, kept_subject_ids


def get_matched_datasets(
    fmri_roi_dir: str = None,
    smri_data_path: str = None,
    phenotypic_file: str = None,
    smri_feature_selection_k: int = 800,
    verbose: bool = True
) -> Dict:
    """
    Get matched datasets for all three experiments (fMRI, sMRI, cross-attention).
    
    Args:
        fmri_roi_dir: Path to fMRI ROI directory
        smri_data_path: Path to sMRI data directory
        phenotypic_file: Path to phenotypic file
        smri_feature_selection_k: Number of sMRI features to select
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing matched data for all experiments
    """
    if not PROJECT_MODULES_AVAILABLE:
        raise ImportError("Project modules not available for dataset creation")
    
    if verbose:
        print("Creating datasets")
    
    # Get matched subject IDs
    matched_subject_ids = get_matched_subject_ids(
        fmri_roi_dir, smri_data_path, phenotypic_file, verbose
    )
    
    # Use default paths if not provided
    if fmri_roi_dir is None or smri_data_path is None or phenotypic_file is None:
        try:
            config = get_config('cross_attention')
            fmri_roi_dir = fmri_roi_dir or config.fmri_roi_dir
            smri_data_path = smri_data_path or config.smri_data_path
            phenotypic_file = phenotypic_file or config.phenotypic_file
        except:
            # Fallback for Google Colab
            fmri_roi_dir = fmri_roi_dir or "/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200"
            smri_data_path = smri_data_path or "/content/drive/MyDrive/processed_smri_data"
            phenotypic_file = phenotypic_file or "/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
    
    # Load and filter fMRI data
    if verbose:
        print("Loading fMRI data")
    
    fmri_processor = FMRIDataProcessor(
        roi_dir=Path(fmri_roi_dir),
        pheno_file=Path(phenotypic_file),
        n_rois=200
    )
    fmri_features, fmri_labels, fmri_subject_ids, _ = fmri_processor.process_all_subjects(verbose=False)
    
    # Filter fMRI to matched subjects
    fmri_features_matched, fmri_labels_matched, fmri_subject_ids_matched = filter_data_by_subjects(
        fmri_features, fmri_labels, fmri_subject_ids, matched_subject_ids, verbose
    )
    
    # Load and filter sMRI data
    if verbose:
        print("Loading sMRI data")
    
    smri_processor = SMRIDataProcessor(
        data_path=Path(smri_data_path),
        feature_selection_k=None,  # Don't do feature selection yet
        scaler_type='standard'
    )
    smri_features, smri_labels, smri_subject_ids = smri_processor.process_all_subjects(
        phenotypic_file=Path(phenotypic_file), verbose=False
    )
    
    # Filter sMRI to matched subjects
    smri_features_matched, smri_labels_matched, smri_subject_ids_matched = filter_data_by_subjects(
        smri_features, smri_labels, smri_subject_ids, matched_subject_ids, verbose
    )
    
    # Verify subject order matches
    if fmri_subject_ids_matched != smri_subject_ids_matched:
        # Sort both to ensure same order
        combined_data = list(zip(fmri_subject_ids_matched, fmri_features_matched, fmri_labels_matched))
        combined_data.sort(key=lambda x: x[0])
        fmri_subject_ids_matched, fmri_features_matched, fmri_labels_matched = zip(*combined_data)
        fmri_features_matched = np.array(fmri_features_matched)
        fmri_labels_matched = np.array(fmri_labels_matched)
        fmri_subject_ids_matched = list(fmri_subject_ids_matched)
        
        combined_data = list(zip(smri_subject_ids_matched, smri_features_matched, smri_labels_matched))
        combined_data.sort(key=lambda x: x[0])
        smri_subject_ids_matched, smri_features_matched, smri_labels_matched = zip(*combined_data)
        smri_features_matched = np.array(smri_features_matched)
        smri_labels_matched = np.array(smri_labels_matched)
        smri_subject_ids_matched = list(smri_subject_ids_matched)
    
    # Verify labels match
    if not np.array_equal(fmri_labels_matched, smri_labels_matched):
        raise ValueError("Label mismatch between fMRI and sMRI matched data!")
    
    if verbose:
        print(f"Dataset ready: {len(fmri_subject_ids_matched)} subjects")
    
    return {
        'fmri_features': fmri_features_matched,
        'fmri_labels': fmri_labels_matched,
        'fmri_subject_ids': fmri_subject_ids_matched,
        'smri_features': smri_features_matched,
        'smri_labels': smri_labels_matched,
        'smri_subject_ids': smri_subject_ids_matched,
        'matched_subject_ids': list(matched_subject_ids),
        'num_matched_subjects': len(matched_subject_ids)
    } 