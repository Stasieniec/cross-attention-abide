import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MinimalCrossAttentionTransformer(nn.Module):
    """
    Minimal cross-attention transformer that truly replicates baseline processing.
    
    This is what basic cross-attention SHOULD be:
    - fMRI processed as single token (like baseline fMRI)
    - sMRI processed as single vector (like baseline sMRI)  
    - Simple cross-attention between these two representations
    """

    def __init__(
        self,
        fmri_dim: int,
        smri_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,  # SAME as tokenized version default
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()

        self.fmri_dim = fmri_dim
        self.smri_dim = smri_dim
        self.d_model = d_model

        # Baseline-equivalent processing
        
        # fMRI processing: EXACT same as SingleAtlasTransformer baseline
        self.fmri_projection = nn.Linear(fmri_dim, d_model, bias=True)
        self.fmri_scale = math.sqrt(d_model)
        self.fmri_norm = nn.LayerNorm(d_model)
        self.fmri_dropout = nn.Dropout(dropout)
        
        # sMRI processing: EXACT same as baseline sMRI transformer  
        self.smri_projection = nn.Sequential(
            nn.Linear(smri_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.smri_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        
        # Simple transformer layers for individual modality processing
        # (minimal processing, just like baselines)
        self.fmri_transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.smri_transformer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
            dim_feedforward=d_model * 2,
                dropout=dropout,
                activation='gelu',
            batch_first=True
        )
        
        # Cross-attention layers (same as tokenized version but single tokens)
        
        # BIDIRECTIONAL cross-attention layers with SAME structure as tokenized version
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
            for _ in range(n_layers)  # Same number of layers
        ])
        
        # Fusion and classifier (same as tokenized version)
        
        # EXACT SAME FUSION as tokenized version (but for single tokens)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # fMRI + sMRI single tokens  
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # EXACT SAME CLASSIFIER as tokenized version
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights like baseline models."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, 
        fmri_features: torch.Tensor, 
        smri_features: torch.Tensor, 
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with TRUE baseline-equivalent processing.
        
        Args:
            fmri_features: (batch_size, fmri_dim)
            smri_features: (batch_size, smri_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = fmri_features.shape[0]
        
        # Process fMRI exactly like baseline
        
        # Project and scale (exact same as SingleAtlasTransformer)
        fmri_projected = self.fmri_projection(fmri_features) / self.fmri_scale
        fmri_projected = self.fmri_norm(fmri_projected)
        fmri_projected = self.fmri_dropout(fmri_projected)
        
        # Add sequence dimension for transformer (single token)
        fmri_token = fmri_projected.unsqueeze(1)  # (batch, 1, d_model)
        
        # Apply minimal transformer processing (like baseline)
        fmri_processed = self.fmri_transformer(fmri_token)  # (batch, 1, d_model)
        
        # Process sMRI exactly like baseline
        
        # Project with batch norm (exact same as baseline sMRI)
        smri_projected = self.smri_projection(smri_features)  # (batch, d_model)
        
        # Add sequence dimension and positional embedding (like baseline)
        smri_token = smri_projected.unsqueeze(1)  # (batch, 1, d_model)
        smri_token = smri_token + self.smri_pos_embedding
        
        # Apply minimal transformer processing (like baseline)
        smri_processed = self.smri_transformer(smri_token)  # (batch, 1, d_model)
        
        # Cross-attention same as tokenized version (single tokens)
        
        # Initialize embeddings (single tokens for each modality)
        fmri_embeddings = fmri_processed  # (batch, 1, d_model)
        smri_embeddings = smri_processed  # (batch, 1, d_model)
        
        attention_weights = []
        
        # EXACT same bidirectional cross-attention pattern as tokenized version
        for layer in self.cross_attention_layers:
            # fMRI attends to sMRI (EXACT same as tokenized)
            fmri_attended, attn_weights = layer['fmri_to_smri'](
                fmri_embeddings, smri_embeddings, smri_embeddings
            )
            fmri_embeddings = layer['norm1'](fmri_embeddings + fmri_attended)
            
            # sMRI attends to fMRI (EXACT same as tokenized)  
            smri_attended, attn_weights = layer['smri_to_fmri'](
                smri_embeddings, fmri_embeddings, fmri_embeddings
            )
            smri_embeddings = layer['norm2'](smri_embeddings + smri_attended)
            
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Fusion same as tokenized version (single tokens)
        
        # Flatten single tokens (same as tokenized version but with 1 token each)
        fmri_flat = fmri_embeddings.flatten(start_dim=1)  # (batch, d_model)
        smri_flat = smri_embeddings.flatten(start_dim=1)  # (batch, d_model)
        combined = torch.cat([fmri_flat, smri_flat], dim=1)  # (batch, d_model*2)
        fused = self.fusion(combined)  # (batch, d_model)

        # Classification
        logits = self.classifier(fused)

        if return_attention:
            return logits, attention_weights
        return logits

    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'MinimalCrossAttentionTransformer',
            'description': 'TRUE baseline cross-attention with single-token processing',
            'fmri_dim': self.fmri_dim,
            'smri_dim': self.smri_dim,
            'd_model': self.d_model,
            'total_params': total_params,
            'processing': 'Single-token per modality (baseline-equivalent)'
        } 