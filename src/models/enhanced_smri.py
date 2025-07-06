import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class EnhancedSMRITransformer(nn.Module):
    """
    sMRI Transformer with feature preprocessing and residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        layer_dropout: float = 0.05,
        num_classes: int = 2,
        use_feature_engineering: bool = True,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_feature_engineering = use_feature_engineering
        self.use_positional_encoding = use_positional_encoding
        
        # Feature engineering layer
        if use_feature_engineering:
            self.feature_engineering = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.LayerNorm(input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(input_dim * 2, input_dim)
            )
        
        # Input projection with residual
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Residual connection for input
        if input_dim != d_model:
            self.input_residual = nn.Linear(input_dim, d_model)
        else:
            self.input_residual = nn.Identity()
        
        # Positional encoding (learned)
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Multi-scale transformer blocks
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                layer_dropout=layer_dropout
            )
            for _ in range(n_layers)
        ])
        
        # Multi-head classifier with different pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        classifier_input_dim = d_model * 2  # avg + max pooling
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Better weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Feature engineering
        if self.use_feature_engineering:
            x_eng = self.feature_engineering(x)
            x = x + x_eng  # Residual connection
        
        # Input projection with residual
        x_proj = self.input_projection(x)
        x_res = self.input_residual(x)
        x = x_proj + x_res
        
        # Add sequence dimension and positional encoding
        x = x.unsqueeze(1)  # [batch, 1, d_model]
        
        if self.use_positional_encoding:
            x = x + self.pos_encoding
        
        # Transform through blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Multi-strategy pooling
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [batch, d_model]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [batch, d_model]
        
        # Combine pooling strategies
        x = torch.cat([avg_pool, max_pool], dim=-1)  # [batch, d_model*2]
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class EnhancedTransformerBlock(nn.Module):
    """Transformer block with enhancements."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        layer_dropout: float = 0.05,
        ff_multiplier: int = 4
    ):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(dropout)
        )
        
        # Stochastic depth (layer dropout)
        self.layer_dropout = layer_dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm multi-head attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        
        # Stochastic depth for attention
        if self.training and self.layer_dropout > 0:
            if torch.rand(1).item() < self.layer_dropout:
                attn_out = attn_out * 0
        
        x = x + attn_out
        
        # Pre-norm feed forward
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        
        # Stochastic depth for FF
        if self.training and self.layer_dropout > 0:
            if torch.rand(1).item() < self.layer_dropout:
                ff_out = ff_out * 0
        
        x = x + ff_out
        
        return x


class SMRIEnsemble(nn.Module):
    """Ensemble of multiple sMRI models for better performance."""
    
    def __init__(
        self,
        input_dim: int,
        num_models: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.models = nn.ModuleList([
            EnhancedSMRITransformer(
                input_dim=input_dim,
                d_model=d_model + i * 32,  # Slightly different sizes
                n_heads=n_heads,
                n_layers=n_layers + (i % 2),  # Slightly different depths
                dropout=dropout + i * 0.02,  # Slightly different dropout
                use_feature_engineering=(i % 2 == 0),
                use_positional_encoding=(i != 1)
            )
            for i in range(num_models)
        ])
        
        # Ensemble combination
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted ensemble
        ensemble_output = torch.zeros_like(outputs[0])
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        for i, output in enumerate(outputs):
            ensemble_output += weights[i] * output
        
        return ensemble_output 