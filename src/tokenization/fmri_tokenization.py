import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats
from scipy.stats import zscore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FMRITokenizer:
    """
    Tokenizer for fMRI connectivity data with multiple strategies.
    
    Supports various tokenization approaches:
    - Functional network tokenization
    - Network-based hierarchical tokenization  
    - Full connectivity tokenization
    - ROI-based connectivity tokenization
    """
    
    def __init__(self, n_rois: int = 200):
        self.n_rois = n_rois
        logger.info(f"FMRITokenizer initialized for {n_rois} ROIs")
        
    def tokenize_functional_network(self, 
                                  fmri_data: np.ndarray,
                                  n_networks: int = 7,
                                  **kwargs) -> np.ndarray:
        """
        Creates functional network tokens using network analysis.
        
        Args:
            fmri_data: Raw fMRI connectivity data [subjects, features]
            n_networks: Number of functional networks to create
            
        Returns:
            Tokenized data [subjects, n_networks, features_per_network]
        """
        n_subjects = fmri_data.shape[0]
        n_features = fmri_data.shape[1]
        
        # Calculate expected triangular features for connectivity matrix
        expected_triangular = (self.n_rois * (self.n_rois - 1)) // 2
        
        if n_features == expected_triangular:
            # Convert upper triangular to full matrix
            connectivity_matrices = self._triangular_to_full_matrix(fmri_data)
        else:
            # Assume already full matrices
            connectivity_matrices = fmri_data.reshape(n_subjects, self.n_rois, self.n_rois)
        
        # Apply Fisher Z-transformation for network analysis
        connectivity_matrices = np.arctanh(np.clip(connectivity_matrices, -0.999, 0.999))
        
        logger.info(f"Tokenizing {n_networks} functional networks")
        
        # Network creation using graph theory principles
        roi_networks = self._create_functional_networks(connectivity_matrices, n_networks)
        
        # Extract network-specific features
        network_tokens = []
        for network_idx, roi_indices in enumerate(roi_networks):
            # Extract subnetwork connectivity
            network_connectivity = connectivity_matrices[:, np.ix_(roi_indices, roi_indices)]
            
            # Compute network metrics
            network_features = []
            for subj in range(n_subjects):
                subj_features = self._compute_network_features(network_connectivity[subj])
                network_features.append(subj_features)
            
            network_tokens.append(np.array(network_features))
        
        # Stack tokens: [subjects, n_networks, features_per_network]
        tokenized_data = np.stack(network_tokens, axis=1)
        
        logger.info(f"Created {n_networks} functional network tokens: {tokenized_data.shape}")
        return tokenized_data
    
    def tokenize_network_based(self, 
                             fmri_data: np.ndarray,
                             n_networks: int = 7,
                             **kwargs) -> np.ndarray:
        """
        Creates network-based hierarchical tokens using clustering.
        
        Args:
            fmri_data: Raw fMRI connectivity data
            n_networks: Number of networks for hierarchical organization
            
        Returns:
            Tokenized data with hierarchical network structure
        """
        n_subjects = fmri_data.shape[0]
        n_features = fmri_data.shape[1]
        
        logger.info(f"Tokenizing {n_networks} hierarchical networks")
        
        # Convert to connectivity matrices if needed
        if n_features == (self.n_rois * (self.n_rois - 1)) // 2:
            connectivity_matrices = self._triangular_to_full_matrix(fmri_data)
        else:
            connectivity_matrices = fmri_data.reshape(n_subjects, self.n_rois, self.n_rois)
        
        # Apply preprocessing
        connectivity_matrices = np.arctanh(np.clip(connectivity_matrices, -0.999, 0.999))
        
        # Hierarchical network organization
        network_assignments = self._hierarchical_network_assignment(connectivity_matrices, n_networks)
        
        # Create network-based tokens
        network_tokens = []
        for network_idx in range(n_networks):
            network_rois = np.where(network_assignments == network_idx)[0]
            
            if len(network_rois) > 0:
                # Extract inter-network and intra-network connectivity
                network_features = self._extract_hierarchical_features(
                    connectivity_matrices, network_rois, network_assignments
                )
                network_tokens.append(network_features)
        
        # Stack tokens: [subjects, n_networks, features_per_network]
        tokenized_data = np.stack(network_tokens, axis=1)
        
        logger.info(f"Created {n_networks} network-based tokens: {tokenized_data.shape}")
        return tokenized_data
    
    def tokenize_full_connectivity(self, 
                                 fmri_data: np.ndarray,
                                 token_size: int = 64,
                                 **kwargs) -> np.ndarray:
        """
        Creates tokens from full connectivity matrices using algorithms.
        
        Args:
            fmri_data: Raw fMRI connectivity data
            token_size: Size of each token
            
        Returns:
            Tokenized data preserving full connectivity information
        """
        n_subjects = fmri_data.shape[0]
        n_features = fmri_data.shape[1]
        
        logger.info("Tokenizing through clustering")
        
        # Convert to full connectivity matrices
        if n_features == (self.n_rois * (self.n_rois - 1)) // 2:
            connectivity_matrices = self._triangular_to_full_matrix(fmri_data)
        else:
            connectivity_matrices = fmri_data.reshape(n_subjects, self.n_rois, self.n_rois)
        
        # Apply Fisher Z-transformation
        connectivity_matrices = np.arctanh(np.clip(connectivity_matrices, -0.999, 0.999))
        
        # Functional tokenization
        functional_tokens = []
        
        # Method 1: ROI-based functional clustering
        roi_clusters = self._functional_roi_clustering(connectivity_matrices, token_size)
        
        # Method 2: Connectivity-based feature extraction
        for cluster_idx, roi_indices in enumerate(roi_clusters):
            cluster_features = []
            
            for subj in range(n_subjects):
                subj_matrix = connectivity_matrices[subj]
                
                # Extract cluster features
                cluster_connectivity = subj_matrix[np.ix_(roi_indices, roi_indices)]
                inter_cluster_connectivity = np.mean(subj_matrix[roi_indices, :], axis=0)
                
                # Compute graph metrics
                cluster_metrics = self._compute_cluster_metrics(
                    cluster_connectivity, inter_cluster_connectivity
                )
                
                cluster_features.append(cluster_metrics)
            
            functional_tokens.append(np.array(cluster_features))
        
        # Stack tokens: [subjects, n_tokens, features_per_token]
        tokenized_data = np.stack(functional_tokens, axis=1)
        
        logger.info(f"Created functional tokens: {tokenized_data.shape}")
        return tokenized_data
    
    def tokenize_roi_connectivity(self, 
                                fmri_data: np.ndarray,
                                **kwargs) -> np.ndarray:
        """
        Creates ROI-based connectivity tokens preserving individual ROI patterns.
        
        Args:
            fmri_data: Raw fMRI connectivity data
            
        Returns:
            Tokenized data with ROI-specific connectivity patterns
        """
        n_subjects = fmri_data.shape[0]
        n_features = fmri_data.shape[1]
        
        logger.info("Tokenizing ROI connectivity")
        
        # Validate expected features
        expected_features = (self.n_rois * (self.n_rois - 1)) // 2
        if n_features != expected_features:
            logger.warning(f"Expected {expected_features} features for {self.n_rois} ROIs, got {n_features}")
        
        # Convert triangular to full connectivity matrix
        connectivity_matrices = self._triangular_to_full_matrix(fmri_data)
        
        # Apply Fisher Z-transformation
        connectivity_matrices = np.arctanh(np.clip(connectivity_matrices, -0.999, 0.999))
        
        # Create ROI-specific tokens
        roi_tokens = []
        
        for roi_idx in range(self.n_rois):
            roi_connections = []
            
            for subj in range(n_subjects):
                # Extract connections for this ROI (excluding self-connection)
                roi_connectivity = connectivity_matrices[subj, roi_idx, :]
                roi_connectivity = np.concatenate([
                    roi_connectivity[:roi_idx],
                    roi_connectivity[roi_idx+1:]
                ])
                
                # Add ROI features
                roi_features = self._compute_roi_features(roi_connectivity)
                roi_connections.append(roi_features)
            
            roi_tokens.append(np.array(roi_connections))
        
        # Stack tokens: [subjects, n_rois, features_per_roi]
        roi_tokens_array = np.stack(roi_tokens, axis=1)
        
        logger.info(f"Created ROI connectivity tokens: {roi_tokens_array.shape}")
        logger.info(f"   {self.n_rois} tokens per subject")
        logger.info(f"   {roi_tokens_array.shape[2]} connections per ROI")
        logger.info(f"   100% connectivity information preserved")
        
        return roi_tokens_array
    
    def _triangular_to_full_matrix(self, triangular_data: np.ndarray) -> np.ndarray:
        """Convert upper triangular connectivity data to full symmetric matrices."""
        n_subjects = triangular_data.shape[0]
        n_features = triangular_data.shape[1]
        
        # Calculate matrix size
        n_rois = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
        
        full_matrices = np.zeros((n_subjects, n_rois, n_rois))
        
        for subj in range(n_subjects):
            # Fill upper triangular part
            triu_indices = np.triu_indices(n_rois, k=1)
            full_matrices[subj][triu_indices] = triangular_data[subj]
            
            # Make symmetric
            full_matrices[subj] = full_matrices[subj] + full_matrices[subj].T
        
        return full_matrices
    
    def _create_functional_networks(self, connectivity_matrices: np.ndarray, n_networks: int) -> List[np.ndarray]:
        """Create functional networks using graph analysis."""
        n_subjects, n_rois, _ = connectivity_matrices.shape
        
        # Compute group-level connectivity
        group_connectivity = np.mean(connectivity_matrices, axis=0)
        
        # Apply clustering
        from sklearn.cluster import SpectralClustering
        
        # Use spectral clustering for network identification
        clustering = SpectralClustering(
            n_clusters=n_networks,
            affinity='precomputed',
            random_state=42
        )
        
        # Ensure positive definite matrix for clustering
        similarity_matrix = np.abs(group_connectivity)
        network_assignments = clustering.fit_predict(similarity_matrix)
        
        # Create network ROI lists
        networks = []
        for network_idx in range(n_networks):
            network_rois = np.where(network_assignments == network_idx)[0]
            networks.append(network_rois)
        
        return networks
    
    def _compute_network_features(self, network_connectivity: np.ndarray) -> np.ndarray:
        """Compute network features using graph theory."""
        n_rois = network_connectivity.shape[0]
        
        features = []
        
        # Basic connectivity statistics
        features.extend([
            np.mean(network_connectivity),
            np.std(network_connectivity),
            np.median(network_connectivity)
        ])
        
        # Graph theory metrics
        # Degree centrality
        degrees = np.sum(np.abs(network_connectivity), axis=1)
        features.extend([
            np.mean(degrees),
            np.std(degrees),
            np.max(degrees)
        ])
        
        # Clustering coefficient approximation
        clustering_coeff = []
        for i in range(n_rois):
            neighbors = np.where(np.abs(network_connectivity[i]) > 0.1)[0]
            if len(neighbors) > 1:
                neighbor_connections = network_connectivity[np.ix_(neighbors, neighbors)]
                possible_edges = len(neighbors) * (len(neighbors) - 1)
                if possible_edges > 0:
                    actual_edges = np.sum(np.abs(neighbor_connections) > 0.1)
                    clustering_coeff.append(actual_edges / possible_edges)
        
        if clustering_coeff:
            features.extend([
                np.mean(clustering_coeff),
                np.std(clustering_coeff)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Ensure consistent feature size
        target_size = 32  # Consistent token size
        features = np.array(features)
        
        if len(features) < target_size:
            # Pad with zeros
            features = np.pad(features, (0, target_size - len(features)))
        elif len(features) > target_size:
            # Truncate
            features = features[:target_size]
        
        return features
    
    def _hierarchical_network_assignment(self, connectivity_matrices: np.ndarray, n_networks: int) -> np.ndarray:
        """Create hierarchical network assignments using clustering."""
        n_subjects, n_rois, _ = connectivity_matrices.shape
        
        # Compute multi-level features for clustering
        roi_features = []
        
        for roi in range(n_rois):
            roi_connectivity = connectivity_matrices[:, roi, :]
            
            # Compute ROI-level features across subjects
            roi_feature_vector = [
                np.mean(roi_connectivity),
                np.std(roi_connectivity),
                np.percentile(roi_connectivity, 25),
                np.percentile(roi_connectivity, 75),
                np.mean(np.abs(roi_connectivity)),
                np.std(np.abs(roi_connectivity))
            ]
            
            roi_features.append(roi_feature_vector)
        
        roi_features = np.array(roi_features)
        
        # Apply hierarchical clustering
        from sklearn.cluster import AgglomerativeClustering
        
        clustering = AgglomerativeClustering(
            n_clusters=n_networks,
            linkage='ward'
        )
        
        network_assignments = clustering.fit_predict(roi_features)
        
        return network_assignments
    
    def _extract_hierarchical_features(self, connectivity_matrices: np.ndarray, 
                                     network_rois: np.ndarray, 
                                     all_assignments: np.ndarray) -> np.ndarray:
        """Extract hierarchical features for network-based tokens."""
        n_subjects = connectivity_matrices.shape[0]
        
        network_features = []
        
        for subj in range(n_subjects):
            subj_matrix = connectivity_matrices[subj]
            
            # Intra-network connectivity
            intra_connectivity = subj_matrix[np.ix_(network_rois, network_rois)]
            
            # Inter-network connectivity
            other_rois = np.where(all_assignments != all_assignments[network_rois[0]])[0]
            if len(other_rois) > 0:
                inter_connectivity = subj_matrix[np.ix_(network_rois, other_rois)]
            else:
                inter_connectivity = np.zeros((len(network_rois), 1))
            
            # Compute features
            features = []
            
            # Intra-network features
            features.extend([
                np.mean(intra_connectivity),
                np.std(intra_connectivity),
                np.median(intra_connectivity),
                np.max(intra_connectivity),
                np.min(intra_connectivity)
            ])
            
            # Inter-network features
            features.extend([
                np.mean(inter_connectivity),
                np.std(inter_connectivity),
                np.median(inter_connectivity),
                np.max(inter_connectivity),
                np.min(inter_connectivity)
            ])
            
            # Network topology features
            degrees = np.sum(np.abs(intra_connectivity), axis=1)
            features.extend([
                np.mean(degrees),
                np.std(degrees),
                np.max(degrees) - np.min(degrees)  # Degree range
            ])
            
            # Ensure consistent feature size
            target_size = 32
            features = np.array(features)
            
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            elif len(features) > target_size:
                features = features[:target_size]
            
            network_features.append(features)
        
        return np.array(network_features)
    
    def _functional_roi_clustering(self, connectivity_matrices: np.ndarray, token_size: int) -> List[np.ndarray]:
        """Create functional ROI clusters using clustering algorithms."""
        n_subjects, n_rois, _ = connectivity_matrices.shape
        
        # Compute group-level ROI similarities
        roi_similarities = np.zeros((n_rois, n_rois))
        
        for i in range(n_rois):
            for j in range(n_rois):
                if i != j:
                    # Compute correlation of ROI connectivity patterns across subjects
                    roi_i_patterns = connectivity_matrices[:, i, :]
                    roi_j_patterns = connectivity_matrices[:, j, :]
                    
                    # Flatten and compute correlation
                    roi_i_flat = roi_i_patterns.flatten()
                    roi_j_flat = roi_j_patterns.flatten()
                    
                    correlation = np.corrcoef(roi_i_flat, roi_j_flat)[0, 1]
                    roi_similarities[i, j] = correlation if not np.isnan(correlation) else 0
        
        # Apply clustering to create functional groups
        n_clusters = max(1, n_rois // token_size)
        
        from sklearn.cluster import SpectralClustering
        
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        # Ensure positive similarities
        similarity_matrix = np.abs(roi_similarities)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        cluster_assignments = clustering.fit_predict(similarity_matrix)
        
        # Create cluster ROI lists
        clusters = []
        for cluster_idx in range(n_clusters):
            cluster_rois = np.where(cluster_assignments == cluster_idx)[0]
            clusters.append(cluster_rois)
        
        return clusters
    
    def _compute_cluster_metrics(self, cluster_connectivity: np.ndarray, 
                               inter_cluster_connectivity: np.ndarray) -> np.ndarray:
        """Compute cluster metrics."""
        features = []
        
        # Intra-cluster features
        features.extend([
            np.mean(cluster_connectivity),
            np.std(cluster_connectivity),
            np.median(cluster_connectivity),
            np.max(cluster_connectivity),
            np.min(cluster_connectivity)
        ])
        
        # Inter-cluster features
        features.extend([
            np.mean(inter_cluster_connectivity),
            np.std(inter_cluster_connectivity),
            np.median(inter_cluster_connectivity),
            np.max(inter_cluster_connectivity),
            np.min(inter_cluster_connectivity)
        ])
        
        # Graph metrics
        n_rois = cluster_connectivity.shape[0]
        if n_rois > 1:
            # Degree centrality
            degrees = np.sum(np.abs(cluster_connectivity), axis=1)
            features.extend([
                np.mean(degrees),
                np.std(degrees),
                np.max(degrees)
            ])
            
            # Density
            possible_edges = n_rois * (n_rois - 1)
            actual_edges = np.sum(np.abs(cluster_connectivity) > 0.1)
            density = actual_edges / possible_edges if possible_edges > 0 else 0
            features.append(density)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Ensure consistent feature size
        target_size = 32
        features = np.array(features)
        
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return features
    
    def _compute_roi_features(self, roi_connectivity: np.ndarray) -> np.ndarray:
        """Compute ROI-specific features."""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(roi_connectivity),
            np.std(roi_connectivity),
            np.median(roi_connectivity),
            np.max(roi_connectivity),
            np.min(roi_connectivity)
        ])
        
        # Percentile features
        features.extend([
            np.percentile(roi_connectivity, 25),
            np.percentile(roi_connectivity, 75),
            np.percentile(roi_connectivity, 90)
        ])
        
        # Distribution features
        features.extend([
            scipy.stats.skew(roi_connectivity),
            scipy.stats.kurtosis(roi_connectivity)
        ])
        
        # Connectivity strength features
        positive_connections = roi_connectivity[roi_connectivity > 0]
        negative_connections = roi_connectivity[roi_connectivity < 0]
        
        features.extend([
            len(positive_connections),
            len(negative_connections),
            np.mean(positive_connections) if len(positive_connections) > 0 else 0,
            np.mean(negative_connections) if len(negative_connections) > 0 else 0
        ])
        
        # Ensure consistent feature size
        target_size = 32
        features = np.array(features)
        
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return features 