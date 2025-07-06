import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMRITokenizer:
    """
    Tokenizer for sMRI structural data with multiple strategies.
    
    Supports various tokenization approaches:
    - Simple token creation
    - Hemisphere-based tokenization
    - Feature type-based tokenization
    - Brain network-based tokenization
    - Pre-computed tokenization loading
    """
    
    def __init__(self):
        logger.info(f"SMRITokenizer initialized")
        
    def tokenize_simple(self, 
                       smri_data: np.ndarray, 
                       n_tokens: int = 16, 
                       **kwargs) -> np.ndarray:
        """
        Creates simple tokens by grouping structural features.
        
        Args:
            smri_data: sMRI structural data [subjects, features]
            n_tokens: Number of tokens to create
            
        Returns:
            Tokenized data [subjects, n_tokens, features_per_token]
        """
        n_subjects, n_features = smri_data.shape
        features_per_token = n_features // n_tokens
        
        logger.info(f"Tokenizing {n_tokens} simple features")
        
        # Group features into tokens
        tokens = []
        for token_idx in range(n_tokens):
            start_idx = token_idx * features_per_token
            if token_idx == n_tokens - 1:
                # Last token gets remaining features
                end_idx = n_features
            else:
                end_idx = start_idx + features_per_token
            
            token_features = smri_data[:, start_idx:end_idx]
            tokens.append(token_features)
        
        # Stack tokens: [subjects, n_tokens, features_per_token]
        tokenized_data = np.stack(tokens, axis=1)
        
        return tokenized_data
    
    def tokenize_hemisphere(self, 
                          smri_data: np.ndarray, 
                          **kwargs) -> np.ndarray:
        """
        Creates hemisphere-based tokens (left/right brain).
        
        Args:
            smri_data: sMRI structural data
            
        Returns:
            Tokenized data with hemisphere-based organization
        """
        n_subjects, n_features = smri_data.shape
        
        logger.info("Tokenizing hemisphere features")
        
        # Split features into left and right hemisphere
        # Assuming features are organized as [left_features, right_features]
        mid_point = n_features // 2
        
        left_hemisphere = smri_data[:, :mid_point]
        right_hemisphere = smri_data[:, mid_point:]
        
        # Stack hemispheres as tokens: [subjects, 2, features_per_hemisphere]
        tokenized_data = np.stack([left_hemisphere, right_hemisphere], axis=1)
        
        return tokenized_data
    
    def tokenize_feature_type(self, 
                            smri_data: np.ndarray, 
                            feature_types: Optional[Dict[str, List[int]]] = None,
                            **kwargs) -> np.ndarray:
        """
        Creates feature type-based tokens (e.g., cortical thickness, surface area, volume).
        
        Args:
            smri_data: sMRI structural data
            feature_types: Dictionary mapping feature type names to feature indices
            
        Returns:
            Tokenized data organized by feature types
        """
        n_subjects, n_features = smri_data.shape
        
        logger.info("Tokenizing feature types")
        
        # Default feature type organization if not provided
        if feature_types is None:
            feature_types = self._get_default_feature_types(n_features)
        
        tokens = []
        n_feature_types = len(feature_types)
        
        for feature_type, feature_indices in feature_types.items():
            # Extract features for this type
            if isinstance(feature_indices, list):
                # Use specific indices
                type_features = smri_data[:, feature_indices]
            else:
                # Use range
                type_features = smri_data[:, feature_indices]
            
            logger.info(f"     {feature_type}: {type_features.shape[1]} features")
            tokens.append(type_features)
        
        # Stack tokens: [subjects, n_feature_types, features_per_type]
        tokenized_data = np.stack(tokens, axis=1)
        
        logger.info(f"Created {n_feature_types} feature type tokens: {tokenized_data.shape}")
        
        return tokenized_data
    
    def tokenize_brain_network(self, 
                             smri_data: np.ndarray, 
                             brain_networks: Optional[Dict[str, List[int]]] = None,
                             **kwargs) -> np.ndarray:
        """
        Creates brain network-based tokens using anatomical network definitions.
        
        Args:
            smri_data: sMRI structural data
            brain_networks: Dictionary mapping network names to ROI indices
            
        Returns:
            Tokenized data organized by brain networks
        """
        n_subjects, n_features = smri_data.shape
        
        logger.info("Tokenizing brain networks")
        
        # Default brain network organization if not provided
        if brain_networks is None:
            brain_networks = self._get_default_brain_networks(n_features)
        
        tokens = []
        n_brain_networks = len(brain_networks)
        
        for network_name, network_indices in brain_networks.items():
            # Extract features for this network
            if isinstance(network_indices, list):
                network_features = smri_data[:, network_indices]
            else:
                network_features = smri_data[:, network_indices]
            
            logger.info(f"     {network_name}: {network_features.shape[1]} features")
            tokens.append(network_features)
        
        # Stack tokens: [subjects, n_networks, features_per_network]
        tokenized_data = np.stack(tokens, axis=1)
        
        logger.info(f"Created {n_brain_networks} brain network tokens: {tokenized_data.shape}")
        
        return tokenized_data
    
    def load_precomputed_tokens(self, 
                              strategy_name: str,
                              data_path: Union[str, Path],
                              **kwargs) -> np.ndarray:
        """
        Loads pre-computed tokenized sMRI data.
        
        Args:
            strategy_name: Name of the tokenization strategy
            data_path: Path to the tokenized data directory
            
        Returns:
            Pre-computed tokenized data
        """
        data_path = Path(data_path)
        strategy_path = data_path / f"smri_{strategy_name}"
        
        logger.info(f"Loading {strategy_name} from {strategy_path}")
        
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy path not found: {strategy_path}")
        
        # Load token files
        token_files = list(strategy_path.glob("*.pkl"))
        if not token_files:
            raise FileNotFoundError(f"No token files found in {strategy_path}")
        
        # Load and combine tokens
        all_tokens = []
        for token_file in sorted(token_files):
            with open(token_file, 'rb') as f:
                token_data = pickle.load(f)
                all_tokens.append(token_data)
        
        # Combine tokens
        if len(all_tokens) == 1:
            combined_tokens = all_tokens[0]
        else:
            combined_tokens = np.concatenate(all_tokens, axis=1)
        
        # Log token information
        for i, token_file in enumerate(sorted(token_files)):
            token_name = token_file.stem
            token_features = all_tokens[i]
            logger.info(f"   {token_name}: {token_features.shape}")
        
        if len(all_tokens) > 1:
            logger.info(f"   Combined shape: {combined_tokens.shape}")
        
        return combined_tokens
    
    def _get_default_feature_types(self, n_features: int) -> Dict[str, slice]:
        """
        Provides default feature type organization.
        
        Args:
            n_features: Total number of features
            
        Returns:
            Dictionary mapping feature type names to feature slices
        """
        # Default assumption: features are organized as
        # [cortical_thickness, surface_area, volume, ...]
        features_per_type = n_features // 4
        
        feature_types = {
            'cortical_thickness': slice(0, features_per_type),
            'surface_area': slice(features_per_type, 2 * features_per_type),
            'volume': slice(2 * features_per_type, 3 * features_per_type),
            'other_metrics': slice(3 * features_per_type, n_features)
        }
        
        return feature_types
    
    def _get_default_brain_networks(self, n_features: int) -> Dict[str, slice]:
        """
        Provides default brain network organization.
        
        Args:
            n_features: Total number of features
            
        Returns:
            Dictionary mapping network names to feature slices
        """
        # Default assumption: features are organized by major brain networks
        networks_per_region = n_features // 8
        
        brain_networks = {
            'frontal': slice(0, networks_per_region),
            'parietal': slice(networks_per_region, 2 * networks_per_region),
            'temporal': slice(2 * networks_per_region, 3 * networks_per_region),
            'occipital': slice(3 * networks_per_region, 4 * networks_per_region),
            'limbic': slice(4 * networks_per_region, 5 * networks_per_region),
            'subcortical': slice(5 * networks_per_region, 6 * networks_per_region),
            'cerebellar': slice(6 * networks_per_region, 7 * networks_per_region),
            'other_regions': slice(7 * networks_per_region, n_features)
        }
        
        return brain_networks
    
    def tokenize(self, 
                smri_data: np.ndarray, 
                tokenization_type: str, 
                **kwargs) -> np.ndarray:
        """
        Main tokenization interface.
        
        Args:
            smri_data: sMRI structural data
            tokenization_type: Type of tokenization strategy
            **kwargs: Additional arguments for specific tokenization methods
            
        Returns:
            Tokenized sMRI data
        """
        if tokenization_type == 'simple':
            return self.tokenize_simple(smri_data, **kwargs)
        elif tokenization_type == 'hemisphere':
            return self.tokenize_hemisphere(smri_data, **kwargs)
        elif tokenization_type == 'feature_type':
            return self.tokenize_feature_type(smri_data, **kwargs)
        elif tokenization_type == 'brain_network':
            return self.tokenize_brain_network(smri_data, **kwargs)
        elif tokenization_type in ['precomputed', 'improved']:
            # For pre-computed strategies, load from data path
            data_path = kwargs.get('data_path', '/content/drive/MyDrive/thesis_data/tokenized_data')
            return self.load_precomputed_tokens(tokenization_type, data_path, **kwargs)
        else:
            raise ValueError(f"Unknown sMRI tokenization type: {tokenization_type}")
    
    def create_feature_tokens(self, 
                             smri_data: np.ndarray,
                             labels: np.ndarray,
                             n_tokens: int = 16,
                             feature_selection: bool = True,
                             **kwargs) -> np.ndarray:
        """
        Creates tokens using feature selection and dimensionality reduction.
        
        Args:
            smri_data: sMRI structural data
            labels: Classification labels for feature selection
            n_tokens: Number of tokens to create
            feature_selection: Whether to apply feature selection
            
        Returns:
            Tokenized data
        """
        n_subjects, n_features = smri_data.shape
        
        # Apply feature selection if requested
        if feature_selection:
            # Select top features based on univariate statistics
            selector = SelectKBest(f_classif, k=min(n_features, n_tokens * 50))
            selected_features = selector.fit_transform(smri_data, labels)
            logger.info(f"Selected {selected_features.shape[1]} features from {n_features}")
        else:
            selected_features = smri_data
        
        # Apply dimensionality reduction within each token
        n_selected_features = selected_features.shape[1]
        features_per_token = n_selected_features // n_tokens
        
        tokens = []
        for token_idx in range(n_tokens):
            start_idx = token_idx * features_per_token
            if token_idx == n_tokens - 1:
                end_idx = n_selected_features
            else:
                end_idx = start_idx + features_per_token
            
            token_features = selected_features[:, start_idx:end_idx]
            
            # Apply PCA if token has many features
            if token_features.shape[1] > 32:
                pca = PCA(n_components=min(32, token_features.shape[1]))
                token_features = pca.fit_transform(token_features)
            
            tokens.append(token_features)
        
        # Stack tokens: [subjects, n_tokens, features_per_token]
        tokenized_data = np.stack(tokens, axis=1)
        
        return tokenized_data 