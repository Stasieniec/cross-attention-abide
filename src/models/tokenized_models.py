import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FMRIFunctionalNetworkTransformer(nn.Module):
    """fMRI Functional Network Transformer for tokenization"""
    
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 192,
        num_heads: int = 6,
        num_layers: int = 3,
        dropout: float = 0.25,
        layer_dropout: float = 0.15,
        num_classes: int = 2,
        n_functional_networks: int = 7
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.n_functional_networks = n_functional_networks
        
        # Functional network tokenization
        self.network_size = feat_dim // n_functional_networks
        
        # Network-specific processors with layer dropout
        self.network_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.network_size, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(layer_dropout)
            ) for _ in range(n_functional_networks)
        ])
        
        # Transformer encoder with layer dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_dropout = nn.Dropout(layer_dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Tokenize into functional networks
        network_tokens = []
        for i in range(self.n_functional_networks):
            start_idx = i * self.network_size
            end_idx = start_idx + self.network_size
            if i == self.n_functional_networks - 1:  # Last network gets remaining features
                end_idx = self.feat_dim
            
            network_features = x[:, start_idx:end_idx]
            network_token = self.network_processors[i](network_features)
            network_tokens.append(network_token)
        
        # Stack tokens and process with transformer
        tokens = torch.stack(network_tokens, dim=1)  # (batch, n_networks, d_model)
        
        # Apply transformer with layer dropout
        transformed = self.transformer(tokens)
        transformed = self.layer_dropout(transformed)
        
        # Global pooling and classification
        pooled = transformed.mean(dim=1)  # (batch, d_model)
        return self.classifier(pooled)
    
    def get_model_info(self) -> dict:
        return {
            'model_name': 'FMRIFunctionalNetworkTransformer',
            'strategy': 'Functional Network Tokenization',
            'total_params': sum(p.numel() for p in self.parameters()),
            'n_networks': self.n_functional_networks,
            'd_model': self.d_model
        }


class FMRINetworkBasedTransformer(nn.Module):
    """fMRI Network-Based Transformer with hierarchical tokenization"""
    
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
        layer_dropout: float = 0.2,
        num_classes: int = 2,
        n_rois: int = 200
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.n_rois = n_rois
        self.roi_dim = feat_dim // n_rois  # Calculate exact ROI dimension 
        self.hidden_dim = d_model
        
        # Define anatomical network structure (CC200 atlas networks)
        self.network_assignments = self._create_network_assignments()
        self.n_networks = len(set(self.network_assignments))
        
        # ROI-level processing
        self.roi_projection = nn.Linear(self.roi_dim, self.hidden_dim)
        self.roi_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, 
                nhead=num_heads, 
                dim_feedforward=self.hidden_dim*2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Network-level processing with layer dropout
        self.network_aggregation = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_dropout = nn.Dropout(layer_dropout)
        
        # Global classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_network_assignments(self):
        """Create network assignments for CC200 ROIs based on anatomical knowledge."""
        # Simplified network assignment (8 major networks)
        assignments = []
        rois_per_network = self.n_rois // 8
        
        for i in range(self.n_rois):
            network_id = i // rois_per_network
            assignments.append(min(network_id, 7))  # Ensure last network gets remaining ROIs
            
        return assignments
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, fmri_features: torch.Tensor) -> torch.Tensor:
        batch_size = fmri_features.shape[0]
        
        # Handle remainder features by truncating to exact divisible amount
        total_features_needed = self.n_rois * self.roi_dim
        if fmri_features.shape[1] > total_features_needed:
            fmri_features = fmri_features[:, :total_features_needed]
        
        # Reshape to ROI tokens: (batch_size, n_rois, roi_dim)
        roi_tokens = fmri_features.view(batch_size, self.n_rois, self.roi_dim)
        
        # ROI-level processing
        x = self.roi_projection(roi_tokens)  # (batch, n_rois, hidden_dim)
        x = self.roi_encoder(x)
        
        # Network-level aggregation
        network_representations = []
        for network_id in range(self.n_networks):
            # Get ROIs belonging to this network
            network_mask = torch.tensor([i == network_id for i in self.network_assignments], 
                                      dtype=torch.bool, device=x.device)
            network_rois = x[:, network_mask, :]  # (batch, network_size, hidden_dim)
            
            # Aggregate within network
            network_repr = network_rois.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
            network_representations.append(network_repr)
        
        # Stack network representations
        network_features = torch.cat(network_representations, dim=1)  # (batch, n_networks, hidden_dim)
        
        # Cross-network attention with layer dropout
        attended_networks, _ = self.network_aggregation(
            network_features, network_features, network_features
        )
        attended_networks = self.layer_dropout(attended_networks)
        
        # Global pooling and classification
        global_repr = attended_networks.mean(dim=1)  # (batch, hidden_dim)
        logits = self.classifier(global_repr)
        
        return logits
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'FMRINetworkBasedTransformer',
            'strategy': 'Network-based hierarchical tokenization',
            'total_params': total_params,
            'n_networks': self.n_networks,
            'n_rois': self.n_rois,
            'roi_dim': self.roi_dim
        }


class FMRIFullConnectivityTransformer(nn.Module):
    """fMRI Full Connectivity Transformer - For high-dimensional connectivity matrices"""
    
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.4,
        layer_dropout: float = 0.25,
        num_classes: int = 2,
        n_connectivity_blocks: int = 16
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.n_connectivity_blocks = n_connectivity_blocks
        self.layer_dropout = layer_dropout
        
        # DYNAMIC BLOCK SIZE: Calculate expected block size for initialization
        self.expected_block_size = feat_dim // n_connectivity_blocks
        
                # FLEXIBLE BLOCK PROCESSORS: Create a template processor that can be adapted
        # We'll create dynamic processors in forward() to handle variable input sizes
        self._create_block_processor = lambda block_size: nn.Sequential(
            nn.Linear(block_size, d_model),
                nn.LayerNorm(d_model),  # Use LayerNorm instead of BatchNorm1d to handle batch_size=1
                nn.GELU(),
                nn.Dropout(layer_dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(layer_dropout)
        )
        
        # Create default processors for expected block size
        self.block_processors = nn.ModuleList([
            self._create_block_processor(self.expected_block_size) 
            for _ in range(n_connectivity_blocks)
        ])
        
        # Lightweight transformer (avoid overfitting on high-dim data)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model,  # Keep FFN small
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_dropout = nn.Dropout(layer_dropout)
        
        # Heavily regularized classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Handle variable input dimensions dynamically
        actual_feat_dim = x.shape[1]
        actual_block_size = actual_feat_dim // self.n_connectivity_blocks
        
        # Tokenize into connectivity blocks
        block_tokens = []
        for i in range(self.n_connectivity_blocks):
            start_idx = i * actual_block_size
            end_idx = start_idx + actual_block_size
            if i == self.n_connectivity_blocks - 1:  # Last block gets remaining features
                end_idx = actual_feat_dim
            
            block_features = x[:, start_idx:end_idx]
            current_block_size = block_features.shape[1]
            
            # DYNAMIC BLOCK PROCESSOR: Adapt to actual block size
            if current_block_size != self.expected_block_size:
                # Create a temporary projection layer for this block size
                temp_processor = self._create_block_processor(current_block_size).to(x.device)
                
                # Initialize the temporary layer properly
                with torch.no_grad():
                    for module in temp_processor.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                nn.init.zeros_(module.bias)
                        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                            nn.init.ones_(module.weight)
                            nn.init.zeros_(module.bias)
                
                block_token = temp_processor(block_features)
            else:
                # Use the pre-initialized block processor
                block_token = self.block_processors[i](block_features)
            
            block_tokens.append(block_token)
        
        # Stack tokens and process with transformer
        tokens = torch.stack(block_tokens, dim=1)  # (batch, n_blocks, d_model)
        
        # Apply transformer with heavy regularization
        transformed = self.transformer(tokens)
        transformed = self.layer_dropout(transformed)
        
        # Global pooling and classification
        pooled = transformed.mean(dim=1)  # (batch, d_model)
        return self.classifier(pooled)
    
    def get_model_info(self) -> dict:
        return {
            'model_name': 'FMRIFullConnectivityTransformer',
            'strategy': 'Full Connectivity Tokenization',
            'total_params': sum(p.numel() for p in self.parameters()),
            'n_blocks': self.n_connectivity_blocks,
            'd_model': self.d_model
        }


class FMRIROIConnectivityTransformer(nn.Module):
    """ROI Connectivity Transformer - Individual ROI connectivity patterns as tokens
    
    This model treats each ROI as a separate token with its full connectivity vector.
    - 200 tokens (one per ROI in CC200 atlas)
    - Each token has 199 features (connections to all other ROIs) 
    - Preserves ALL connectivity information without loss
    - Allows attention between individual ROI connectivity patterns
    - Spatially-aware tokenization based on brain anatomy
    """
    
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 192,
        num_heads: int = 6,
        num_layers: int = 3,
        dropout: float = 0.2,
        layer_dropout: float = 0.15,
        num_classes: int = 2,
        n_rois: int = 200
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.n_rois = n_rois
        self.roi_dim = 199  # Each ROI connects to 199 other ROIs
        
        # Validate that input dimensions match expected ROI structure
        expected_dim = self.n_rois * self.roi_dim
        if feat_dim != expected_dim:
            raise ValueError(f"Expected feat_dim={expected_dim} for {n_rois} ROIs x {self.roi_dim} connections, got {feat_dim}")
        
        # ROI-specific projection layers - each ROI gets its own embedding
        self.roi_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.roi_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(layer_dropout)
            ) for _ in range(self.n_rois)
        ])
        
        # Positional encoding for ROI spatial relationships
        self.roi_position_embedding = nn.Parameter(torch.randn(self.n_rois, d_model) * 0.02)
        
        # Transformer encoder with attention between ROI tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 3,  # Slightly larger feedforward for ROI complexity
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_dropout = nn.Dropout(layer_dropout)
        
        # ROI-aware attention pooling
        self.roi_attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head with ROI-specific processing
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Reshape input: (batch, n_rois * roi_dim) -> (batch, n_rois, roi_dim)
        roi_features = x.view(batch_size, self.n_rois, self.roi_dim)
        
        # Project each ROI's connectivity vector to embedding space
        roi_tokens = []
        for roi_idx in range(self.n_rois):
            roi_connectivity = roi_features[:, roi_idx, :]  # (batch, roi_dim)
            roi_token = self.roi_projections[roi_idx](roi_connectivity)  # (batch, d_model)
            roi_tokens.append(roi_token)
        
        # Stack ROI tokens and add positional embeddings
        tokens = torch.stack(roi_tokens, dim=1)  # (batch, n_rois, d_model)
        tokens = tokens + self.roi_position_embedding.unsqueeze(0)  # Add positional info
        
        # Apply transformer to model ROI-ROI interactions
        transformed = self.transformer(tokens)  # (batch, n_rois, d_model)
        transformed = self.layer_dropout(transformed)
        
        # ROI-aware attention pooling to focus on most important ROIs
        pooled, attention_weights = self.roi_attention_pooling(
            query=transformed.mean(dim=1, keepdim=True),  # Global query
            key=transformed,
            value=transformed
        )
        pooled = pooled.squeeze(1)  # (batch, d_model)
        
        # Classification
        return self.classifier(pooled)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights to see which ROIs are most important."""
        with torch.no_grad():
            batch_size = x.shape[0]
            roi_features = x.view(batch_size, self.n_rois, self.roi_dim)
            
            # Project ROI features
            roi_tokens = []
            for roi_idx in range(self.n_rois):
                roi_connectivity = roi_features[:, roi_idx, :]
                roi_token = self.roi_projections[roi_idx](roi_connectivity)
                roi_tokens.append(roi_token)
            
            tokens = torch.stack(roi_tokens, dim=1)
            tokens = tokens + self.roi_position_embedding.unsqueeze(0)
            
            # Get attention weights from pooling
            transformed = self.transformer(tokens)
            _, attention_weights = self.roi_attention_pooling(
                query=transformed.mean(dim=1, keepdim=True),
                key=transformed,
                value=transformed
            )
            
            return attention_weights.squeeze(1)  # (batch, n_rois)
    
    def get_model_info(self) -> dict:
        return {
            'model_name': 'FMRIROIConnectivityTransformer',
            'strategy': 'ROI Connectivity Tokenization',
            'total_params': sum(p.numel() for p in self.parameters()),
            'n_rois': self.n_rois,
            'roi_dim': self.roi_dim,
            'd_model': self.d_model,
            'description': 'Each ROI as a token with full connectivity vector (199 connections)'
        }


class SMRIFeatureTypeTokenizedTransformer(nn.Module):
    """sMRI transformer using improved feature type tokenization."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.15,
        layer_dropout: float = 0.1,
        num_classes: int = 2,
        n_feature_types: int = 5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_feature_types = n_feature_types
        
        # Feature type tokenization
        features_per_type = input_dim // n_feature_types
        remaining_features = input_dim % n_feature_types
        
        self.feature_type_projections = nn.ModuleList()
        for i in range(n_feature_types):
            # Last token type gets extra features if division is not even
            actual_features = features_per_type + (1 if i == n_feature_types - 1 and remaining_features > 0 else 0)
            if i < remaining_features:
                actual_features += 1
                
            self.feature_type_projections.append(
                nn.Sequential(
                    nn.Linear(actual_features, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Feature type attention
        self.feature_type_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Global transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, smri_features: torch.Tensor) -> torch.Tensor:
        batch_size = smri_features.shape[0]
        actual_input_dim = smri_features.shape[1]
        
        # Use pre-computed tokenized features
        # Features come pre-tokenized from improved_tokenized_smri_datasets with proper feature engineering
        # Each "token" represents domain knowledge (cortical thickness, surface area, etc.)
        
        # Calculate tokens based on pre-computed features
        features_per_type = actual_input_dim // self.n_feature_types
        remaining_features = actual_input_dim % self.n_feature_types
        feature_type_tokens = []
        
        current_idx = 0
        for i, projection in enumerate(self.feature_type_projections):
            # Calculate actual features for this token type
            actual_features = features_per_type + (1 if i < remaining_features else 0)
            end_idx = current_idx + actual_features
            
            # Extract pre-computed token features
            # (These should be Fisher-Z transformed, normalized, domain-specific features)
            type_features = smri_features[:, current_idx:end_idx]
            
            # Adjust projection input size if needed for pre-computed data
            if type_features.shape[1] != projection[0].in_features:
                # Create a new projection layer with correct input size for features
                correct_projection = nn.Sequential(
                    nn.Linear(type_features.shape[1], self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.GELU(),
                    nn.Dropout(0.15)
                ).to(smri_features.device)
                type_token = correct_projection(type_features)
            else:
                type_token = projection(type_features)
            
            feature_type_tokens.append(type_token.unsqueeze(1))
            current_idx = end_idx
        
        # Combine feature type tokens
        feature_type_sequence = torch.cat(feature_type_tokens, dim=1)
        
        # Cross-feature-type attention
        attended_types, _ = self.feature_type_attention(
            feature_type_sequence, feature_type_sequence, feature_type_sequence
        )
        
        # Global processing
        global_features = self.transformer(attended_types)
        
        # Classification
        global_repr = global_features.mean(dim=1)
        logits = self.classifier(global_repr)
        
        return logits
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'SMRIFeatureTypeTokenizedTransformer',
            'strategy': 'Improved feature type tokenization',
            'total_params': total_params,
            'n_feature_types': self.n_feature_types
        }


class SMRIBrainNetworkTokenizedTransformer(nn.Module):
    """sMRI transformer using improved brain network tokenization."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.15,
        layer_dropout: float = 0.1,
        num_classes: int = 2,
        n_brain_networks: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_brain_networks = n_brain_networks
        
        # Brain network tokenization
        features_per_network = input_dim // n_brain_networks
        remaining_features = input_dim % n_brain_networks
        
        self.brain_network_projections = nn.ModuleList()
        for i in range(n_brain_networks):
            # Handle remaining features by distributing them across first tokens
            actual_features = features_per_network + (1 if i < remaining_features else 0)
                
            self.brain_network_projections.append(
                nn.Sequential(
                    nn.Linear(actual_features, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Cross-network attention
        self.cross_network_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Hierarchical processing
        self.hierarchical_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, smri_features: torch.Tensor) -> torch.Tensor:
        batch_size = smri_features.shape[0]
        actual_input_dim = smri_features.shape[1]
        
        # Use pre-computed brain network tokenized features
        # Features come pre-tokenized from improved_tokenized_smri_datasets with network analysis
        # Each "token" represents brain network measures (default mode, executive control, etc.)
        
        # Calculate tokens based on pre-computed network features
        features_per_network = actual_input_dim // self.n_brain_networks
        remaining_features = actual_input_dim % self.n_brain_networks
        brain_network_tokens = []
        
        current_idx = 0
        for i, projection in enumerate(self.brain_network_projections):
            # Calculate actual features for this network
            actual_features = features_per_network + (1 if i < remaining_features else 0)
            end_idx = current_idx + actual_features
            
            # Extract pre-computed brain network features
            # (These should be connectivity measures, spectral features, network topology metrics)
            network_features = smri_features[:, current_idx:end_idx]
            
            # Adjust projection input size if needed for pre-computed data
            if network_features.shape[1] != projection[0].in_features:
                # Create a new projection layer with correct input size for network features
                correct_projection = nn.Sequential(
                    nn.Linear(network_features.shape[1], self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.GELU(),
                    nn.Dropout(0.15)
                ).to(smri_features.device)
                network_token = correct_projection(network_features)
            else:
                network_token = projection(network_features)
            
            brain_network_tokens.append(network_token.unsqueeze(1))
            current_idx = end_idx
        
        # Combine brain network tokens
        brain_network_sequence = torch.cat(brain_network_tokens, dim=1)
        
        # Cross-network attention
        attended_networks, _ = self.cross_network_attention(
            brain_network_sequence, brain_network_sequence, brain_network_sequence
        )
        
        # Hierarchical processing
        hierarchical_features = self.hierarchical_transformer(attended_networks)
        
        # Classification
        global_repr = hierarchical_features.mean(dim=1)
        logits = self.classifier(global_repr)
        
        return logits
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'SMRIBrainNetworkTokenizedTransformer',
            'strategy': 'Improved brain network tokenization',
            'total_params': total_params,
            'n_brain_networks': self.n_brain_networks
        }


class TokenizedCrossAttentionTransformer(nn.Module):
    """Cross-attention between tokenized fMRI and sMRI modalities."""
    
    def __init__(
        self,
        fmri_dim: int,
        smri_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.15,
        num_classes: int = 2,
        fmri_tokenization_type: str = 'functional_network',
        smri_tokenization_type: str = 'feature_type'
    ):
        super().__init__()
        
        self.fmri_dim = fmri_dim
        self.smri_dim = smri_dim
        self.d_model = d_model
        self.fmri_tokenization_type = fmri_tokenization_type
        self.smri_tokenization_type = smri_tokenization_type
        
        # fMRI tokenization encoder
        if fmri_tokenization_type == 'functional_network':
            self.n_fmri_tokens = 50
        elif fmri_tokenization_type == 'network_based':
            self.n_fmri_tokens = 8
        else:
            self.n_fmri_tokens = 1
        
        # sMRI tokenization encoder
        if smri_tokenization_type == 'feature_type':
            self.n_smri_tokens = 5
        elif smri_tokenization_type == 'brain_network':
            self.n_smri_tokens = 8
        else:
            self.n_smri_tokens = 1
        
        # Token projections - calculate features per token properly
        self.fmri_features_per_token = fmri_dim // self.n_fmri_tokens
        self.smri_features_per_token = smri_dim // self.n_smri_tokens
        
        # Handle case where tokenization is disabled (single token)
        if self.n_fmri_tokens == 1:
            self.fmri_features_per_token = fmri_dim
        if self.n_smri_tokens == 1:
            self.smri_features_per_token = smri_dim
        
        self.fmri_token_projection = nn.Linear(self.fmri_features_per_token, d_model)
        self.smri_token_projection = nn.Linear(self.smri_features_per_token, d_model)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'fmri_to_smri': nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                ),
                'smri_to_fmri': nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(d_model * (self.n_fmri_tokens + self.n_smri_tokens), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _tokenize_fmri_features(self, fmri_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Tokenize fMRI features based on the specified strategy."""
        
        if self.fmri_tokenization_type == 'functional_network':
            # Functional network tokenization (50 networks)
            features_per_token = fmri_features.shape[1] // self.n_fmri_tokens
            remainder = fmri_features.shape[1] % self.n_fmri_tokens
            
            # Better handling of remainder to prevent dimension mismatch
            if remainder > 0:
                # Pad the features to be divisible by n_tokens
                padding_needed = self.n_fmri_tokens - remainder
                fmri_features = torch.nn.functional.pad(fmri_features, (0, padding_needed), mode='constant', value=0)
                features_per_token = fmri_features.shape[1] // self.n_fmri_tokens
            
            fmri_tokens = fmri_features.view(batch_size, self.n_fmri_tokens, features_per_token)
            
        elif self.fmri_tokenization_type == 'network_based':
            # Network-based tokenization (8 anatomical networks for CC200 atlas)
            features_per_token = fmri_features.shape[1] // self.n_fmri_tokens
            remainder = fmri_features.shape[1] % self.n_fmri_tokens
            
            # Better handling of remainder to prevent dimension mismatch
            if remainder > 0:
                # Pad the features to be divisible by n_tokens
                padding_needed = self.n_fmri_tokens - remainder
                fmri_features = torch.nn.functional.pad(fmri_features, (0, padding_needed), mode='constant', value=0)
                features_per_token = fmri_features.shape[1] // self.n_fmri_tokens
            
            fmri_tokens = fmri_features.view(batch_size, self.n_fmri_tokens, features_per_token)
            
        else:
            # Default: no tokenization, treat whole feature vector as single token
            fmri_tokens = fmri_features.unsqueeze(1)  # (batch_size, 1, fmri_dim)
        
        # Ensure projection layer matches the actual token dimension
        actual_token_dim = fmri_tokens.shape[2]
        if actual_token_dim != self.fmri_token_projection.in_features:
            # Create a new projection layer with correct input size
            self.fmri_token_projection = torch.nn.Linear(actual_token_dim, self.d_model).to(fmri_tokens.device)
        
        # Project to embedding space
        return self.fmri_token_projection(fmri_tokens)
    
    def _tokenize_smri_features(self, smri_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Tokenize sMRI features based on the specified strategy."""
        
        if self.smri_tokenization_type == 'feature_type':
            # Feature type tokenization (5 different anatomical feature types)
            features_per_token = smri_features.shape[1] // self.n_smri_tokens
            remainder = smri_features.shape[1] % self.n_smri_tokens
            
            # Better handling of remainder to prevent dimension mismatch
            if remainder > 0:
                # Pad the features to be divisible by n_tokens
                padding_needed = self.n_smri_tokens - remainder
                smri_features = torch.nn.functional.pad(smri_features, (0, padding_needed), mode='constant', value=0)
                features_per_token = smri_features.shape[1] // self.n_smri_tokens
            
            smri_tokens = smri_features.view(batch_size, self.n_smri_tokens, features_per_token)
            
        elif self.smri_tokenization_type == 'brain_network':
            # Brain network tokenization (8 major brain networks)
            features_per_token = smri_features.shape[1] // self.n_smri_tokens
            remainder = smri_features.shape[1] % self.n_smri_tokens
            
            # Better handling of remainder to prevent dimension mismatch
            if remainder > 0:
                # Pad the features to be divisible by n_tokens
                padding_needed = self.n_smri_tokens - remainder
                smri_features = torch.nn.functional.pad(smri_features, (0, padding_needed), mode='constant', value=0)
                features_per_token = smri_features.shape[1] // self.n_smri_tokens
            
            smri_tokens = smri_features.view(batch_size, self.n_smri_tokens, features_per_token)
            
        else:
            # Default: no tokenization, treat whole feature vector as single token
            smri_tokens = smri_features.unsqueeze(1)  # (batch_size, 1, smri_dim)
        
        # Ensure projection layer matches the actual token dimension
        actual_token_dim = smri_tokens.shape[2]
        if actual_token_dim != self.smri_token_projection.in_features:
            # Create a new projection layer with correct input size
            self.smri_token_projection = torch.nn.Linear(actual_token_dim, self.d_model).to(smri_tokens.device)
        
        # Project to embedding space
        return self.smri_token_projection(smri_tokens)

    def forward(self, fmri_features: torch.Tensor, smri_features: torch.Tensor) -> torch.Tensor:
        batch_size = fmri_features.shape[0]
        
        # Tokenize fMRI features using PROPER tokenization strategy
        fmri_embeddings = self._tokenize_fmri_features(fmri_features, batch_size)
        
        # Tokenize sMRI features using PROPER tokenization strategy
        smri_embeddings = self._tokenize_smri_features(smri_features, batch_size)
        
        # Cross-attention processing
        for layer in self.cross_attention_layers:
            # fMRI attends to sMRI
            fmri_attended, _ = layer['fmri_to_smri'](fmri_embeddings, smri_embeddings, smri_embeddings)
            fmri_embeddings = layer['norm1'](fmri_embeddings + fmri_attended)
            
            # sMRI attends to fMRI
            smri_attended, _ = layer['smri_to_fmri'](smri_embeddings, fmri_embeddings, fmri_embeddings)
            smri_embeddings = layer['norm2'](smri_embeddings + smri_attended)
        
        # Fusion
        fmri_flat = fmri_embeddings.flatten(start_dim=1)
        smri_flat = smri_embeddings.flatten(start_dim=1)
        combined = torch.cat([fmri_flat, smri_flat], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        return logits
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'TokenizedCrossAttentionTransformer',
            'strategy': f'Cross-attention with {self.fmri_tokenization_type} fMRI and {self.smri_tokenization_type} sMRI',
            'total_params': total_params,
            'fmri_tokens': self.n_fmri_tokens,
            'smri_tokens': self.n_smri_tokens
        } 