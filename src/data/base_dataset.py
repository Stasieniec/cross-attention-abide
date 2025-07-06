import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union


class DataAugmentation:
    """Data augmentation strategies for neuroimaging data."""

    @staticmethod
    def add_gaussian_noise(x: torch.Tensor, std: float = 0.01, prob: float = 0.3) -> torch.Tensor:
        """Add Gaussian noise to input with given probability."""
        if torch.rand(1).item() < prob:
            noise = torch.randn_like(x) * std
            return x + noise
        return x

    @staticmethod
    def dropout_connections(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        """Randomly drop connections (set to 0) with given probability."""
        if torch.rand(1).item() < 0.3:  # Apply dropout 30% of the time
            mask = torch.rand_like(x) > drop_prob
            return x * mask
        return x


class ABIDEDataset(Dataset):
    """
    Base ABIDE dataset class with augmentation support.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[List[str]] = None,
        augment: bool = False,
        noise_std: float = 0.01,
        augment_prob: float = 0.3
    ):
        """
        Initialize ABIDE dataset.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            subject_ids: Optional list of subject IDs
            augment: Whether to apply data augmentation
            noise_std: Standard deviation of Gaussian noise for augmentation
            augment_prob: Probability of applying augmentation
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.subject_ids = subject_ids
        self.augment = augment
        self.noise_std = noise_std
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx]

        # Apply augmentation during training
        if self.augment:
            x = DataAugmentation.add_gaussian_noise(x, self.noise_std, self.augment_prob)

        return x, y

    def get_subject_id(self, idx: int) -> Optional[str]:
        """Get subject ID for a given index."""
        return self.subject_ids[idx] if self.subject_ids else None
    
    def get_class_distribution(self) -> dict:
        """Get class distribution statistics."""
        unique, counts = torch.unique(self.y, return_counts=True)
        return {
            'classes': unique.tolist(),
            'counts': counts.tolist(),
            'total': len(self.y)
        } 