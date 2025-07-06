import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

from .base_dataset import DataAugmentation


class MultiModalPreprocessor:
    """Preprocess features for both fMRI and sMRI modalities."""

    def __init__(self, smri_feature_selection_k: int = 800):
        """
        Initialize multimodal preprocessor.
        
        Args:
            smri_feature_selection_k: Number of sMRI features to select
        """
        self.fmri_scaler = StandardScaler()
        self.smri_scaler = RobustScaler()
        self.smri_feature_selector = None
        self.smri_feature_selection_k = smri_feature_selection_k
        self.selected_smri_features = None

    def fit(self, fmri_features: np.ndarray, smri_features: np.ndarray, labels: np.ndarray):
        """
        Fit preprocessors on training data.
        
        Args:
            fmri_features: fMRI feature array
            smri_features: sMRI feature array
            labels: Label array
        """
        # Fit fMRI scaler
        self.fmri_scaler.fit(fmri_features)

        # Fit sMRI scaler and feature selector
        smri_scaled = self.smri_scaler.fit_transform(smri_features)

        if self.smri_feature_selection_k and self.smri_feature_selection_k < smri_features.shape[1]:
            self.smri_feature_selector = SelectKBest(
                score_func=f_classif,
                k=self.smri_feature_selection_k
            )
            self.smri_feature_selector.fit(smri_scaled, labels)
            self.selected_smri_features = self.smri_feature_selector.get_support()
            print(f"Selected {self.smri_feature_selection_k} best sMRI features")

    def transform(self, fmri_features: np.ndarray, smri_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features using fitted preprocessors.
        
        Args:
            fmri_features: fMRI feature array
            smri_features: sMRI feature array
            
        Returns:
            Tuple of transformed (fmri_features, smri_features)
        """
        # Transform fMRI
        fmri_transformed = self.fmri_scaler.transform(fmri_features)

        # Transform sMRI
        smri_transformed = self.smri_scaler.transform(smri_features)
        if self.smri_feature_selector is not None:
            smri_transformed = self.smri_feature_selector.transform(smri_transformed)

        return fmri_transformed, smri_transformed


class MultiModalDataset(Dataset):
    """Dataset for multimodal fMRI-sMRI data."""

    def __init__(
        self,
        fmri_features: np.ndarray,
        smri_features: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[List[str]] = None,
        augment: bool = False,
        noise_std: float = 0.01,
        augment_prob: float = 0.3
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            fmri_features: fMRI feature array
            smri_features: sMRI feature array  
            labels: Label array
            subject_ids: Optional list of subject IDs
            augment: Whether to apply data augmentation
            noise_std: Standard deviation for Gaussian noise
            augment_prob: Probability of applying augmentation
        """
        self.fmri_features = torch.FloatTensor(fmri_features)
        self.smri_features = torch.FloatTensor(smri_features)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids
        self.augment = augment
        self.noise_std = noise_std
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fmri = self.fmri_features[idx].clone()
        smri = self.smri_features[idx].clone()
        label = self.labels[idx]

        # Data augmentation
        if self.augment and torch.rand(1).item() < self.augment_prob:
            # Add Gaussian noise to both modalities
            fmri = DataAugmentation.add_gaussian_noise(fmri, self.noise_std, 1.0)
            smri = DataAugmentation.add_gaussian_noise(smri, self.noise_std, 1.0)

        return fmri, smri, label

    def get_subject_id(self, idx: int) -> Optional[str]:
        """Get subject ID for a given index."""
        return self.subject_ids[idx] if self.subject_ids else None
    
    def get_class_distribution(self) -> dict:
        """Get class distribution statistics."""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {
            'classes': unique.tolist(),
            'counts': counts.tolist(),
            'total': len(self.labels)
        }


def match_multimodal_subjects(
    fmri_data: Dict, 
    smri_data: Dict, 
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Match subjects between fMRI and sMRI modalities.
    
    Args:
        fmri_data: Dictionary of fMRI data by subject ID
        smri_data: Dictionary of sMRI data by subject ID
        verbose: Whether to print matching statistics
        
    Returns:
        Tuple of (fmri_features, smri_features, labels, subject_ids)
    """
    # Find common subjects
    common_subjects = list(set(fmri_data.keys()) & set(smri_data.keys()))
    
    if verbose:
        print(f"Found {len(common_subjects)} subjects with both modalities")
    
    # Create matched dataset
    matched_fmri_features = []
    matched_smri_features = []
    matched_labels = []
    matched_subject_ids = []
    label_mismatches = 0

    for sub_id in sorted(common_subjects):
        # Verify labels match between modalities
        fmri_label = fmri_data[sub_id]['label']
        smri_label = smri_data[sub_id]['label']

        if fmri_label != smri_label:
            if verbose:
                print(f"Oops! Label mismatch for subject {sub_id}: fMRI={fmri_label}, sMRI={smri_label}")
            label_mismatches += 1
            continue

        matched_fmri_features.append(fmri_data[sub_id]['features'])
        matched_smri_features.append(smri_data[sub_id]['features'])
        matched_labels.append(fmri_label)
        matched_subject_ids.append(sub_id)

    if verbose:
        print(f"Label mismatches: {label_mismatches}")

    # Convert to numpy arrays
    matched_fmri_features = np.array(matched_fmri_features)
    matched_smri_features = np.array(matched_smri_features)
    matched_labels = np.array(matched_labels)

    return matched_fmri_features, matched_smri_features, matched_labels, matched_subject_ids 