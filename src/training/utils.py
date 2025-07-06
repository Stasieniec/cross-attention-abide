import random
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import ABIDEDataset, SMRIDataset, MultiModalDataset


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping with optional model checkpointing."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-4, mode: str = 'min', save_checkpoints: bool = False):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            save_checkpoints: Whether to save model checkpoints (disabled by default to save Google Drive space)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_checkpoints = save_checkpoints
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float, model: torch.nn.Module, path: Path) -> None:
        """
        Check if training should stop and optionally save best model.
        
        Args:
            val_score: Current validation score
            model: Model to potentially save
            path: Path to save the model (only used if save_checkpoints=True)
        """
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
            if self.save_checkpoints:
                self.save_checkpoint(val_score, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_checkpoints:
                self.save_checkpoint(val_score, model, path)
            self.counter = 0

    def save_checkpoint(self, val_score: float, model: torch.nn.Module, path: Path) -> None:
        """Save model checkpoint (only called if save_checkpoints=True)."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_score': val_score
        }, path)


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    augment_train: bool = True,
    noise_std: float = 0.01,
    augment_prob: float = 0.3,
    dataset_type: str = 'base'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders with proper settings.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features  
        y_val: Validation labels
        batch_size: Batch size
        augment_train: Whether to augment training data
        noise_std: Standard deviation for noise augmentation
        augment_prob: Probability of applying augmentation
        dataset_type: Type of dataset ('base', 'smri')
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Choose dataset class and prepare parameters
    if dataset_type == 'smri':
        DatasetClass = SMRIDataset
        # SMRIDataset uses 'noise_factor' instead of 'noise_std' - use smaller noise for sMRI
        train_dataset = DatasetClass(
            X_train, y_train,
            augment=augment_train,
            noise_factor=0.005  # Smaller noise factor from working notebook
        )
        val_dataset = DatasetClass(X_val, y_val, augment=False)
    else:
        DatasetClass = ABIDEDataset
        # ABIDEDataset uses 'noise_std'
        train_dataset = DatasetClass(
            X_train, y_train,
            augment=augment_train,
            noise_std=noise_std,
            augment_prob=augment_prob
        )
        val_dataset = DatasetClass(X_val, y_val, augment=False)

    # Handle class imbalance with weighted sampling for training
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


def create_multimodal_data_loaders(
    fmri_train: np.ndarray,
    smri_train: np.ndarray,
    y_train: np.ndarray,
    fmri_val: np.ndarray,
    smri_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    augment_train: bool = True,
    noise_std: float = 0.01,
    augment_prob: float = 0.3
) -> Tuple[DataLoader, DataLoader]:
    """
    Create multimodal train and validation data loaders.
    
    Args:
        fmri_train: Training fMRI features
        smri_train: Training sMRI features
        y_train: Training labels
        fmri_val: Validation fMRI features
        smri_val: Validation sMRI features
        y_val: Validation labels
        batch_size: Batch size
        augment_train: Whether to augment training data
        noise_std: Standard deviation for noise
        augment_prob: Probability of applying augmentation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = MultiModalDataset(
        fmri_train, smri_train, y_train,
        augment=augment_train,
        noise_std=noise_std,
        augment_prob=augment_prob
    )

    val_dataset = MultiModalDataset(
        fmri_val, smri_val, y_val,
        augment=False
    )

    # Handle class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


def calculate_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        labels: Label array
        device: Device to place weights on
        
    Returns:
        Class weights tensor
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    class_weights = total_samples / (len(unique_labels) * counts)
    return torch.FloatTensor(class_weights).to(device) 