import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class BaseConfig:
    """Base configuration class."""
    
    # Data paths (Google Drive mounted)
    base_dir: Path = Path('/content/drive/MyDrive/b_data')
    fmri_roi_dir: Path = field(default_factory=lambda: Path('/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200'))
    phenotypic_file: Path = field(default_factory=lambda: Path('/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'))
    smri_data_path: Path = Path('/content/drive/MyDrive/processed_smri_data')
    
    # Output directory
    output_dir: Optional[Path] = None
    
    # Random seed
    seed: int = 42
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 300
    early_stop_patience: int = 30
    warmup_epochs: int = 10
    
    # Cross-validation
    num_folds: int = 5
    val_size: float = 0.2
    
    # Data augmentation
    augment_prob: float = 0.3
    noise_std: float = 0.01
    
    # Mixed precision training
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # Logging and checkpointing
    log_every: int = 10
    save_best_only: bool = True
    
    def __post_init__(self):
        """Set up output directory if not provided."""
        if self.output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path(f'/content/drive/MyDrive/abide_outputs_{timestamp}')
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class FMRIConfig(BaseConfig):
    """Configuration for fMRI-only experiments."""
    
    # fMRI specific parameters
    n_rois: int = 200
    feat_dim: int = 19_900  # CC200: 200*(200-1)/2
    
    # Model architecture (SAT)
    d_model: int = 256
    dim_feedforward: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Training parameters (METAFormer settings)
    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 750
    early_stop_patience: int = 40
    
    def __post_init__(self):
        super().__post_init__()
        if self.output_dir.name.startswith('abide_outputs'):
            self.output_dir = self.output_dir.parent / f"fmri_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass 
class SMRIConfig(BaseConfig):
    """Configuration for sMRI-only experiments."""
    
    # sMRI specific parameters
    feature_selection_k: int = 800
    scaler_type: str = 'standard'  # 'robust' or 'standard'
    
    # Model architecture
    d_model: int = 192            # Increased from 64 for better capacity
    num_heads: int = 8            # Increased from 4 for better attention
    num_layers: int = 3           # Increased from 2 for more depth
    dropout: float = 0.1          # Reduced from 0.3 for less regularization
    layer_dropout: float = 0.05   # Reduced from 0.1 for less regularization
    
    # Training parameters
    batch_size: int = 64          # Increased from 16 for stability
    learning_rate: float = 2e-3   # Increased from 1e-3 for faster learning
    weight_decay: float = 1e-4    # Kept same
    num_epochs: int = 200
    early_stop_patience: int = 25  # Increased patience
    warmup_epochs: int = 10
    
    # sMRI-specific improvements
    use_class_weights: bool = True
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        super().__post_init__()
        if self.output_dir.name.startswith('abide_outputs'):
            self.output_dir = self.output_dir.parent / f"smri_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CrossAttentionConfig(BaseConfig):
    """Configuration for cross-attention experiments."""
    
    # fMRI parameters
    n_rois: int = 200
    fmri_feat_dim: int = 19_900
    
    # sMRI parameters  
    smri_feat_selection: int = 800
    
    # Model architecture (reduced complexity to prevent overfitting)
    d_model: int = 128  # Reduced from 256
    d_cross: int = 64   # Reduced from 128
    num_heads: int = 4  # Reduced from 8
    num_layers: int = 2 # Reduced from 4
    num_cross_layers: int = 1  # Reduced from 2
    dropout: float = 0.3  # Increased dropout
    
    # Training parameters for stability
    batch_size: int = 32
    learning_rate: float = 1e-4  # Increased from 5e-5
    weight_decay: float = 1e-3   # Increased regularization
    num_epochs: int = 300
    early_stop_patience: int = 30
    warmup_epochs: int = 15
    
    def __post_init__(self):
        super().__post_init__()
        if self.output_dir.name.startswith('abide_outputs'):
            self.output_dir = self.output_dir.parent / f"cross_attention_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.output_dir.mkdir(parents=True, exist_ok=True)


def get_config(experiment_type: str, **kwargs) -> BaseConfig:
    """Factory function to get configuration based on experiment type.
    
    Args:
        experiment_type: One of 'fmri', 'smri', 'cross_attention'
        **kwargs: Additional configuration parameters to override
        
    Returns:
        Configuration object for the specified experiment type
    """
    config_map = {
        'fmri': FMRIConfig,
        'smri': SMRIConfig, 
        'cross_attention': CrossAttentionConfig
    }
    
    if experiment_type not in config_map:
        raise ValueError(f"Unknown experiment type: {experiment_type}. "
                        f"Available types: {list(config_map.keys())}")
    
    config_class = config_map[experiment_type]
    
    # Create config with any provided overrides
    config = config_class(**kwargs)
    
    return config 