import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader with tokenization and caching."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self._fmri_cache = {}  # Cache for base fMRI data to avoid repeated loading
    
    def load_matched_data(self, verbose: bool = True) -> Dict:
        """Load matched subject data using the original paths."""
        if verbose:
            logger.info("Loading matched subject data")
        
        # Use the correct paths from your original setup
        from .subject_matching import get_matched_datasets
        
        matched_data = get_matched_datasets(
            fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
            smri_data_path="/content/drive/MyDrive/processed_smri_data",
            phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
            verbose=verbose
        )
        
        # Standardize field names for consistency with the rest of the codebase
        standardized_data = {
            'fmri_data': matched_data['fmri_features'],
            'smri_data': matched_data['smri_features'],
            'labels': matched_data['fmri_labels'],  # fmri_labels and smri_labels are the same
            'subject_ids': matched_data.get('fmri_subject_ids', []),
            'n_subjects': matched_data['num_matched_subjects'],
            'fmri_dim': matched_data['fmri_features'].shape[1],
            'smri_dim': matched_data['smri_features'].shape[1]
        }
        
        if verbose:
            logger.info(f"Loaded {standardized_data['n_subjects']} subjects")
            logger.info(f"fMRI shape: {standardized_data['fmri_data'].shape}")
            logger.info(f"sMRI shape: {standardized_data['smri_data'].shape}")
            logger.info(f"Labels distribution: {np.bincount(standardized_data['labels'])}")
        
        return standardized_data
    
    def load_tokenized_fmri_data(self, tokenization_type: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load tokenized fMRI data with on-the-fly tokenization and caching."""
        if verbose:
            logger.info(f"Loading fMRI data for {tokenization_type} tokenization")
        
        # Apply tokenization (with internal caching of base data)
        return self._create_fmri_tokens(tokenization_type, verbose)
    
    def _create_fmri_tokens(self, tokenization_type: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Create fMRI tokens with base data caching for efficiency."""
        
        # Check cache for base fMRI data first
        cache_key = 'base_fmri_data'
        if cache_key in self._fmri_cache:
            if verbose:
                logger.info("Using cached base fMRI data (faster loading)")
            base_fmri_data, labels = self._fmri_cache[cache_key]
        else:
            if verbose:
                logger.info("Loading base fMRI data (first time, will be cached)")
                logger.info(f"Using on-the-fly tokenization for {tokenization_type}")
            
            # Load base fMRI data from file
            matched_data = self.load_matched_data(verbose=False)  # Load without verbose to avoid duplicate logs
            base_fmri_data = matched_data['fmri_data']
            labels = matched_data['labels']
            
            # Cache the base data for future use
            self._fmri_cache[cache_key] = (base_fmri_data.copy(), labels.copy())
            if verbose:
                logger.info(f"Cached base fMRI data: {base_fmri_data.shape}")
        
        if verbose:
            logger.info(f"Tokenizing {tokenization_type} data")
        
        # Apply tokenization based on strategy
        if tokenization_type == 'functional_network':
            tokenized_features = self._apply_functional_network_tokenization(base_fmri_data, verbose)
        elif tokenization_type == 'network_based':
            tokenized_features = self._apply_network_based_tokenization(base_fmri_data, verbose)
        elif tokenization_type == 'full_connectivity':
            tokenized_features = self._apply_full_connectivity_tokenization(base_fmri_data, verbose)
        elif tokenization_type == 'roi_connectivity':
            tokenized_features = self._apply_roi_connectivity_tokenization(base_fmri_data, verbose)
        else:
            # Default to full connectivity for unknown types
            if verbose:
                logger.warning(f"Unknown tokenization type '{tokenization_type}', using full_connectivity")
            tokenized_features = self._apply_full_connectivity_tokenization(base_fmri_data, verbose)
        
        if verbose:
            logger.info(f"{tokenization_type} tokenization complete")
            logger.info(f"Tokenized shape: {tokenized_features.shape}")
        
        return tokenized_features, labels
    
    def _apply_functional_network_tokenization(self, fmri_data: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Apply functional network tokenization.
        Based on neuroimaging research practices for network-based analysis.
        """
        if verbose:
            logger.info("Tokenizing functional networks")
        
        # Reshape from flattened connectivity to connectivity matrix
        # CC200 atlas: 200 ROIs -> 200x200 connectivity matrix -> 19900 features (upper triangle)
        n_subjects, n_features = fmri_data.shape
        n_rois = 200  # CC200 atlas
        
        # Reconstruct connectivity matrices
        connectivity_matrices = []
        for subject_idx in range(n_subjects):
            # Convert flattened upper triangle back to symmetric matrix
            conn_matrix = np.zeros((n_rois, n_rois))
            triu_indices = np.triu_indices(n_rois, k=1)
            conn_matrix[triu_indices] = fmri_data[subject_idx]
            conn_matrix = conn_matrix + conn_matrix.T  # Make symmetric
            connectivity_matrices.append(conn_matrix)
        
        connectivity_matrices = np.array(connectivity_matrices)
        
        # Define functional networks based on established brain atlases
        # Using standard functional networks from neuroimaging literature
        functional_networks = {
            'default_mode': list(range(0, 25)),      # DMN regions
            'executive_control': list(range(25, 50)), # ECN regions  
            'salience': list(range(50, 70)),          # Salience network
            'visual': list(range(70, 100)),           # Visual networks
            'somatomotor': list(range(100, 125)),     # Sensorimotor
            'dorsal_attention': list(range(125, 150)), # DAN
            'ventral_attention': list(range(150, 170)), # VAN
            'limbic': list(range(170, 185)),          # Limbic system
            'frontoparietal': list(range(185, 200))   # FPN
        }
        
        # Extract features for each network
        tokenized_features = []
        
        for subject_idx in range(n_subjects):
            subject_tokens = []
            conn_matrix = connectivity_matrices[subject_idx]
            
            for network_name, roi_indices in functional_networks.items():
                # Extract within-network connectivity
                network_conn = conn_matrix[np.ix_(roi_indices, roi_indices)]
                
                # Feature extraction:
                # 1. Mean connectivity strength
                mean_strength = np.mean(network_conn[np.triu_indices_from(network_conn, k=1)])
                
                # 2. Network efficiency (graph theory)
                # Convert to distance matrix and calculate efficiency
                distance_matrix = 1 / (network_conn + 1e-8)  # Avoid division by zero
                np.fill_diagonal(distance_matrix, 0)
                
                # Global efficiency approximation
                n_nodes = len(roi_indices)
                efficiency = 0
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        if network_conn[i, j] > 0:
                            efficiency += 1 / distance_matrix[i, j]
                efficiency = efficiency / (n_nodes * (n_nodes - 1) / 2)
                
                # 3. Network modularity (clustering coefficient)
                clustering = 0
                for i in range(n_nodes):
                    neighbors = np.where(network_conn[i] > 0.1)[0]  # Threshold for connections
                    if len(neighbors) > 1:
                        subgraph = network_conn[np.ix_(neighbors, neighbors)]
                        possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
                        actual_connections = np.sum(subgraph > 0.1) / 2
                        if possible_connections > 0:
                            clustering += actual_connections / possible_connections
                clustering = clustering / n_nodes
                
                # 4. Network variance (stability measure)
                network_variance = np.var(network_conn[np.triu_indices_from(network_conn, k=1)])
                
                # 5. Spectral features (eigenvalues of connectivity matrix)
                eigenvals = np.linalg.eigvals(network_conn)
                spectral_radius = np.max(np.real(eigenvals))
                spectral_gap = np.real(eigenvals[0] - eigenvals[1]) if len(eigenvals) > 1 else 0
                
                # Combine features for this network
                network_features = [
                    mean_strength,
                    efficiency, 
                    clustering,
                    network_variance,
                    spectral_radius,
                    spectral_gap
                ]
                
                subject_tokens.extend(network_features)
            
            tokenized_features.append(subject_tokens)
        
        tokenized_array = np.array(tokenized_features)
        
        if verbose:
            logger.info(f"Created {len(functional_networks)} functional network tokens")
            logger.info(f"Features per network: 6 (connectivity, efficiency, clustering, variance, spectral)")
            logger.info(f"Total features: {tokenized_array.shape[1]}")
        
        return tokenized_array
    
    def _apply_network_based_tokenization(self, fmri_data: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Apply hierarchical network-based tokenization.
        Uses multi-scale network analysis for brain connectivity representation.
        """
        if verbose:
            logger.info("Tokenizing hierarchical networks")
        
        # Start with functional network tokenization as base
        functional_tokens = self._apply_functional_network_tokenization(fmri_data, verbose=False)
        
        # Add hierarchical network features
        n_subjects, n_features = fmri_data.shape
        n_rois = 200
        
        # Reconstruct connectivity matrices
        connectivity_matrices = []
        for subject_idx in range(n_subjects):
            conn_matrix = np.zeros((n_rois, n_rois))
            triu_indices = np.triu_indices(n_rois, k=1)
            conn_matrix[triu_indices] = fmri_data[subject_idx]
            conn_matrix = conn_matrix + conn_matrix.T
            connectivity_matrices.append(conn_matrix)
        
        connectivity_matrices = np.array(connectivity_matrices)
        
        # Multi-scale hierarchical analysis
        hierarchical_features = []
        
        for subject_idx in range(n_subjects):
            subject_hierarchical = []
            conn_matrix = connectivity_matrices[subject_idx]
            
            # Scale 1: Global brain metrics
            # Overall connectivity strength
            global_strength = np.mean(conn_matrix[np.triu_indices_from(conn_matrix, k=1)])
            
            # Global clustering coefficient
            degrees = np.sum(conn_matrix > 0.1, axis=1)
            global_clustering = np.mean(degrees)
            
            # Small-world properties
            # Path length approximation
            distance_matrix = 1 / (conn_matrix + 1e-8)
            np.fill_diagonal(distance_matrix, 0)
            avg_path_length = np.mean(distance_matrix[distance_matrix > 0])
            
            # Scale 2: Hemispheric asymmetry
            left_hemisphere = conn_matrix[:100, :100]  # First 100 ROIs
            right_hemisphere = conn_matrix[100:, 100:]  # Last 100 ROIs
            inter_hemispheric = conn_matrix[:100, 100:]  # Between hemispheres
            
            left_strength = np.mean(left_hemisphere[np.triu_indices_from(left_hemisphere, k=1)])
            right_strength = np.mean(right_hemisphere[np.triu_indices_from(right_hemisphere, k=1)])
            inter_strength = np.mean(inter_hemispheric)
            asymmetry_index = (left_strength - right_strength) / (left_strength + right_strength + 1e-8)
            
            # Scale 3: Hub identification (high-degree nodes)
            node_strengths = np.sum(conn_matrix, axis=1)
            hub_threshold = np.percentile(node_strengths, 90)  # Top 10% as hubs
            hub_connectivity = np.mean(node_strengths[node_strengths > hub_threshold])
            n_hubs = np.sum(node_strengths > hub_threshold)
            
            # Scale 4: Community structure approximation
            # Modularity estimation using correlation-based communities
            correlation_matrix = np.corrcoef(conn_matrix)
            community_strength = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            
            # Scale 5: Rich club organization
            # Rich nodes are those with highest degrees
            rich_threshold = np.percentile(node_strengths, 80)  # Top 20%
            rich_nodes = node_strengths > rich_threshold
            if np.sum(rich_nodes) > 1:
                rich_club_conn = conn_matrix[np.ix_(rich_nodes, rich_nodes)]
                rich_club_strength = np.mean(rich_club_conn[np.triu_indices_from(rich_club_conn, k=1)])
            else:
                rich_club_strength = 0
            
            # Combine hierarchical features
            hierarchical_subject_features = [
                global_strength,
                global_clustering, 
                avg_path_length,
                left_strength,
                right_strength,
                inter_strength,
                asymmetry_index,
                hub_connectivity,
                n_hubs,
                community_strength,
                rich_club_strength
            ]
            
            hierarchical_features.append(hierarchical_subject_features)
        
        hierarchical_array = np.array(hierarchical_features)
        
        # Combine functional and hierarchical features
        combined_features = np.concatenate([functional_tokens, hierarchical_array], axis=1)
        
        if verbose:
            logger.info(f"Added hierarchical network features: {hierarchical_array.shape[1]}")
            logger.info(f"Combined functional + hierarchical: {combined_features.shape[1]} features")
            logger.info(f"Multi-scale analysis complete")
        
        return combined_features
    
    def _apply_full_connectivity_tokenization(self, fmri_data: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Apply full connectivity tokenization that preserves ALL original features
        while adding processing and meaningful structure.
        
        This creates multiple 'tokens' representing different aspects of the connectivity matrix:
        1. Raw connectivity values (all 19,900 features preserved)
        2. Normalized/standardized versions  
        3. Network-specific connectivity patterns
        4. Hemispheric connectivity patterns
        5. Distance-based connectivity patterns
        6. Strength-based connectivity patterns
        
        Result: Multiple tokens that together contain all original information plus structure.
        """
        if verbose:
            logger.info("Tokenizing connectivity data")
        
        n_subjects, n_features = fmri_data.shape
        n_rois = 200  # CC200 atlas
        
        # Reconstruct connectivity matrices for processing
        connectivity_matrices = []
        for subject_idx in range(n_subjects):
            conn_matrix = np.zeros((n_rois, n_rois))
            triu_indices = np.triu_indices(n_rois, k=1)
            conn_matrix[triu_indices] = fmri_data[subject_idx]
            conn_matrix = conn_matrix + conn_matrix.T
            connectivity_matrices.append(conn_matrix)
        
        connectivity_matrices = np.array(connectivity_matrices)
        
        # Create multiple tokens
        all_tokens = []
        
        for subject_idx in range(n_subjects):
            subject_tokens = []
            conn_matrix = connectivity_matrices[subject_idx]
            
            # TOKEN 1: Raw connectivity values (ALL original features preserved)
            raw_connectivity = fmri_data[subject_idx]  # All 19,900 original features
            
            # TOKEN 2: Z-score normalized connectivity (removes subject-specific biases)
            normalized_connectivity = (raw_connectivity - np.mean(raw_connectivity)) / (np.std(raw_connectivity) + 1e-8)
            
            # TOKEN 3: Rank-based connectivity (less sensitive to outliers)
            from scipy.stats import rankdata
            rank_connectivity = rankdata(raw_connectivity) / len(raw_connectivity)
            
            # TOKEN 4: Fisher-Z transformed connectivity (statistical transformation)
            # Apply Fisher-Z transform: arctanh(r) for correlation values
            fisher_z_connectivity = np.arctanh(np.clip(raw_connectivity, -0.999, 0.999))
            
            # TOKEN 5: Network-structured connectivity (organized by functional networks)
            # Define the same functional networks as before
            functional_networks = {
                'default_mode': list(range(0, 25)),
                'executive_control': list(range(25, 50)),
                'salience': list(range(50, 70)),
                'visual': list(range(70, 100)),
                'somatomotor': list(range(100, 125)),
                'dorsal_attention': list(range(125, 150)),
                'ventral_attention': list(range(150, 170)),
                'limbic': list(range(170, 185)),
                'frontoparietal': list(range(185, 200))
            }
            
            # Extract connectivity in network order (within + between networks)
            network_ordered_connectivity = []
            
            # Within-network connections first
            for net_name, roi_indices in functional_networks.items():
                for i, roi_i in enumerate(roi_indices):
                    for j, roi_j in enumerate(roi_indices):
                        if i < j:  # Upper triangle only
                            network_ordered_connectivity.append(conn_matrix[roi_i, roi_j])
            
            # Between-network connections
            network_names = list(functional_networks.keys())
            for i, net1_name in enumerate(network_names):
                for j, net2_name in enumerate(network_names):
                    if i < j:  # Avoid duplication
                        net1_rois = functional_networks[net1_name]
                        net2_rois = functional_networks[net2_name]
                        for roi1 in net1_rois:
                            for roi2 in net2_rois:
                                network_ordered_connectivity.append(conn_matrix[roi1, roi2])
            
            network_ordered_connectivity = np.array(network_ordered_connectivity)
            
            # TOKEN 6: Hemispheric connectivity patterns
            # Left-hemisphere (ROIs 0-99), Right-hemisphere (ROIs 100-199)
            left_rois = list(range(0, 100))
            right_rois = list(range(100, 200))
            
            hemispheric_connectivity = []
            
            # Left-hemisphere internal connections
            for i in left_rois:
                for j in left_rois:
                    if i < j:
                        hemispheric_connectivity.append(conn_matrix[i, j])
            
            # Right-hemisphere internal connections  
            for i in right_rois:
                for j in right_rois:
                    if i < j:
                        hemispheric_connectivity.append(conn_matrix[i, j])
                        
            # Inter-hemispheric connections
            for i in left_rois:
                for j in right_rois:
                    hemispheric_connectivity.append(conn_matrix[i, j])
            
            hemispheric_connectivity = np.array(hemispheric_connectivity)
            
            # TOKEN 7: Distance-based connectivity (anatomical distance groupings)
            # Group connections by anatomical distance (approximate)
            distance_based_connectivity = []
            
            # Short-range connections (adjacent regions, distance ~1-2)
            short_range = []
            medium_range = []
            long_range = []
            
            for i in range(n_rois):
                for j in range(i+1, n_rois):
                    distance = abs(i - j)  # Simplified distance based on ROI ordering
                    
                    if distance <= 10:  # Short-range
                        short_range.append(conn_matrix[i, j])
                    elif distance <= 50:  # Medium-range
                        medium_range.append(conn_matrix[i, j])
                    else:  # Long-range
                        long_range.append(conn_matrix[i, j])
            
            # Pad to same length for consistent processing
            max_len = max(len(short_range), len(medium_range), len(long_range))
            short_range.extend([0] * (max_len - len(short_range)))
            medium_range.extend([0] * (max_len - len(medium_range)))
            long_range.extend([0] * (max_len - len(long_range)))
            
            distance_based_connectivity = np.concatenate([short_range, medium_range, long_range])
            
            # TOKEN 8: Strength-based connectivity (organized by connection strength)
            strength_based_connectivity = np.sort(raw_connectivity)[::-1]  # Descending order
            
                    # Use only the connectivity (Fisher-Z transformed) to maintain original dimensionality
        # This preserves the original 19,900 features while applying statistical processing
            subject_all_tokens = fisher_z_connectivity  # Use Fisher-Z transform for statistical representation
            
            all_tokens.append(subject_all_tokens)
        
        tokenized_array = np.array(all_tokens)
        
        if verbose:
            logger.info(f"Using Fisher-Z transformed connectivity:")
            logger.info(f"Fisher-Z connectivity: {len(fisher_z_connectivity)} features")
            logger.info(f"Total features: {tokenized_array.shape[1]}")
        
        return tokenized_array
    
    def _apply_roi_connectivity_tokenization(self, fmri_data: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Apply ROI connectivity tokenization where each token represents the full connectivity 
        vector of a specific ROI to all other ROIs.
        
        
        Args:
            fmri_data: Flattened connectivity matrix data (n_subjects, 19900)
            verbose: Whether to print detailed progress information
            
        Returns:
            ROI connectivity tokens (n_subjects, 200, 199) - reshaped for transformer input
        """
        if verbose:
            logger.info("Tokenizing ROI connectivity")
        
        # Import tokenizer
        try:
            from tokenization.fmri_tokenization import FMRITokenizer
        except ImportError:
            from src.tokenization.fmri_tokenization import FMRITokenizer
        
        # Create tokenizer and apply ROI connectivity tokenization
        tokenizer = FMRITokenizer(n_rois=200)
        roi_tokens = tokenizer.create_roi_connectivity_tokens(fmri_data, verbose=verbose)
        
        # Reshape for transformer compatibility: (n_subjects, 200, 199) -> (n_subjects, 200 * 199)
        n_subjects, n_rois, roi_dim = roi_tokens.shape
        tokenized_features = roi_tokens.reshape(n_subjects, n_rois * roi_dim)
        
        if verbose:
            logger.info(f"ROI connectivity tokenization complete")
            logger.info(f"Original shape: {roi_tokens.shape}")
            logger.info(f"Flattened shape: {tokenized_features.shape}")
            logger.info(f"Information: 200 ROI tokens x 199 connections = {n_rois * roi_dim} features")
        
        return tokenized_features
    
    def load_tokenized_smri_data(self, tokenization_type: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load tokenized sMRI data using the original approach."""
        if verbose:
            logger.info(f"Loading sMRI data for {tokenization_type} tokenization")
        
        # Try to load from improved tokenized datasets first (like tokenization_experiments.py)
        if tokenization_type == 'feature_type':
            strategy_name = 'improved_feature_type_tokens'
        elif tokenization_type == 'brain_network':
            strategy_name = 'improved_brain_network_tokens'
        else:
            strategy_name = 'improved_anatomical_lobe_tokens'
        
        try:
            # Try to load from improved tokenized datasets
            base_path = Path("/content/drive/MyDrive/thesis_data/improved_tokenized_smri_datasets")
            strategy_path = base_path / strategy_name
            
            if verbose:
                logger.info(f"Trying to load from: {strategy_path}")
            
            if strategy_path.exists():
                if verbose:
                    logger.info(f"Found tokenized strategy directory: {strategy_path}")
                
                # Load metadata
                metadata_file = strategy_path / 'metadata.json'
                if verbose:
                    logger.info(f"Loading metadata from: {metadata_file}")
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                # Load labels
                labels_file = strategy_path / 'labels.npy'
                if verbose:
                    logger.info(f"Loading labels from: {labels_file}")
                labels = np.load(labels_file)
                
                # Load and combine all tokens
                tokens = []
                token_names = list(metadata['token_info'].keys())
                if verbose:
                    logger.info(f"Found {len(token_names)} tokens: {token_names}")
                
                for token_name in token_names:
                    token_file = strategy_path / f'token_{token_name}.npy'
                    if token_file.exists():
                        token_features = np.load(token_file)
                        tokens.append(token_features)
                        if verbose:
                            logger.info(f"Loaded {token_name}: {token_features.shape}")
                    else:
                        if verbose:
                            logger.warning(f"Missing token file: {token_file}")
                
                if tokens:
                    # Combine all tokens
                    combined_features = np.concatenate(tokens, axis=1)
                    
                    if verbose:
                        logger.info(f"Loaded {strategy_name} from tokenized datasets")
                        logger.info(f"Final shape: {combined_features.shape}")
                        logger.info(f"Source: {strategy_path}")
                    
                    return combined_features, labels
                else:
                    if verbose:
                        logger.warning(f"No valid token files found in {strategy_path}")
            else:
                if verbose:
                    logger.warning(f"Strategy directory not found: {strategy_path}")
                
        except Exception as e:
            if verbose:
                logger.error(f"Failed to load improved tokenized data: {e}")
                logger.error(f"Path attempted: {base_path}")
                logger.error(f"Strategy: {strategy_name}")
        
        # No fallback allowed for tokenized experiments
        raise ValueError(
            f"Could not load pre-computed tokenized sMRI data for strategy '{tokenization_type}'.\n"
            f"Expected path: /content/drive/MyDrive/thesis_data/improved_tokenized_smri_datasets/{strategy_name}\n"
            f"Please ensure the tokenized datasets are available before running tokenized experiments."
        )


class SiteExtractor:
    """Handles site information extraction for leave-site-out CV."""
    
    # Known ABIDE site mappings for leave-site-out CV
    ABIDE_SITES = {
        'CALTECH': 'California Institute of Technology',
        'CMU': 'Carnegie Mellon University', 
        'KKI': 'Kennedy Krieger Institute',
        'LEUVEN_1': 'University of Leuven Site 1',
        'LEUVEN_2': 'University of Leuven Site 2',
        'MAX_MUN': 'Ludwig Maximilians University Munich',
        'NYU': 'NYU Langone Medical Center',
        'OHSU': 'Oregon Health and Science University',
        'OLIN': 'Olin Institute',
        'PITT': 'University of Pittsburgh',
        'SBL': 'Social Brain Lab',
        'SDSU': 'San Diego State University',
        'STANFORD': 'Stanford University',
        'TRINITY': 'Trinity Centre for Health Sciences',
        'UCLA_1': 'UCLA Site 1',
        'UCLA_2': 'UCLA Site 2',
        'UM_1': 'University of Michigan Site 1',
        'UM_2': 'University of Michigan Site 2',
        'USM': 'University of Southern Mississippi',
        'YALE': 'Yale'
    }
    
    def extract_site_info(
        self, 
        subject_ids: List[str], 
        phenotypic_file: str = None
    ) -> Tuple[List[str], Dict[str, List[str]], pd.DataFrame]:
        """
        Extract site information from subject IDs and phenotypic data.
        
        Args:
            subject_ids: List of subject IDs
            phenotypic_file: Path to phenotypic CSV file
            
        Returns:
            Tuple of (site_labels, site_mapping, site_stats)
        """
        logger.info("Extracting site information from subject IDs")
        
        site_labels = []
        site_mapping = defaultdict(list)
        
        # Load phenotypic data if available
        phenotypic_sites = {}
        if phenotypic_file and Path(phenotypic_file).exists():
            try:
                pheno_df = pd.read_csv(phenotypic_file)
                if 'SITE_ID' in pheno_df.columns:
                    # Create mapping from SUB_ID to SITE_ID
                    for _, row in pheno_df.iterrows():
                        sub_id = str(row['SUB_ID'])
                        site_id = str(row['SITE_ID'])
                        phenotypic_sites[sub_id] = site_id
                        # Also try with int conversion for subjects like "50003"
                        try:
                            sub_id_int = int(sub_id)
                            phenotypic_sites[str(sub_id_int)] = site_id
                        except ValueError:
                            pass
                    
                    logger.info(f"   Found SITE_ID column in phenotypic data")
                    logger.info(f"   Created site mapping for {len(phenotypic_sites)} subjects")
                else:
                    logger.info(f"   No SITE_ID column found in phenotypic data")
            except Exception as e:
                logger.info(f"   Error loading phenotypic data: {e}")
        
        # Extract sites from subject IDs
        for sub_id in subject_ids:
            site = self._extract_site_from_subject_id(sub_id, phenotypic_sites)
            site_labels.append(site)
            site_mapping[site].append(sub_id)
        
        # Create site statistics
        site_stats = pd.DataFrame([
            {
                'site': site,
                'n_subjects': len(subjects),
                'subjects': subjects[:5] + (['...'] if len(subjects) > 5 else [])
            }
            for site, subjects in site_mapping.items()
        ]).sort_values('n_subjects', ascending=False)
        
        logger.info(f"\nSite extraction results:")
        logger.info(f"   Total sites: {len(site_mapping)}")
        logger.info(f"   Total subjects: {len(subject_ids)}")
        logger.info(f"   Sites found: {list(site_mapping.keys())}")
        
        # Show detailed site information
        for site, subjects in site_mapping.items():
            logger.info(f"   {site}: {len(subjects)} subjects")
        
        # Validate that we have real sites (not artificial ones)
        artificial_sites = [site for site in site_mapping.keys() if site.startswith('SITE_')]
        unknown_sites = [site for site in site_mapping.keys() if site == 'UNKNOWN_SITE']
        
        if artificial_sites:
            logger.warning(f"Artificial sites detected: {artificial_sites}")
            logger.warning("   This may compromise leave-site-out CV validity!")
        elif unknown_sites:
            unknown_count = len([s for s in site_labels if s == 'UNKNOWN_SITE'])
            logger.warning(f"Unknown sites detected for {unknown_count} subjects")
            logger.warning("   Leave-site-out CV will be disabled due to missing site information")
        else:
            logger.info("All sites appear to be real ABIDE collection sites")
        
        return site_labels, dict(site_mapping), site_stats
    
    def _extract_site_from_subject_id(
        self, 
        subject_id: str, 
        phenotypic_sites: Dict[str, str]
    ) -> str:
        """Extract site information from a single subject ID."""
        # First check phenotypic data (this is the most reliable source)
        if subject_id in phenotypic_sites:
            return phenotypic_sites[subject_id]
        
        # Try converting to int and checking phenotypic data with int key
        try:
            subject_id_int = int(str(subject_id).strip())
            if str(subject_id_int) in phenotypic_sites:
                return phenotypic_sites[str(subject_id_int)]
        except ValueError:
            pass
        
        # Handle different subject ID formats
        subject_id_clean = str(subject_id).strip()
        subject_id_upper = subject_id_clean.upper()
        
        # Check for known ABIDE site prefixes
        for site_code in self.ABIDE_SITES.keys():
            if site_code in subject_id_upper:
                return site_code
        
        # Try common patterns:
        # Pattern 1: Site prefix followed by numbers (e.g., "NYU_0050001")
        for site_code in self.ABIDE_SITES.keys():
            if subject_id_upper.startswith(site_code):
                return site_code
        
        # Pattern 2: Numbers followed by site info (e.g., "0050001_KKI")
        for site_code in self.ABIDE_SITES.keys():
            if subject_id_upper.endswith(f"_{site_code}") or subject_id_upper.endswith(site_code):
                return site_code
        
        # Pattern 3: Try to find site info in middle of string
        for site_code in self.ABIDE_SITES.keys():
            if f"_{site_code}_" in subject_id_upper or f"-{site_code}-" in subject_id_upper:
                return site_code
        
        # If no real site can be determined, return 'UNKNOWN_SITE'
        # This allows validation to proceed but prevents leave-site-out CV
        if not phenotypic_sites:  # Only log warning if phenotypic data is empty
            logger.warning(f"Cannot determine real site for subject {subject_id}")
        return 'UNKNOWN_SITE' 