import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


class PositionalEncoding(nn.Module):
    """Optional positional encoding for sequence modeling."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class EnhancedEncoderBlock(nn.Module):
    """
    Transformer encoder block with pre-norm architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # Pre-normalization (shown to stabilize training)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm multi-head attention
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout(attn_output)

        # Pre-norm feedforward
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output

        return x


class SingleAtlasTransformer(nn.Module):
    """
    Single Atlas Transformer (SAT) for ABIDE classification.
    """
    
    def __init__(
        self,
        feat_dim: int = 19_900,
        d_model: int = 256,
        dim_feedforward: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        """
        Initialize Single Atlas Transformer.
        
        Args:
            feat_dim: Input feature dimension (fMRI connectivity features)
            d_model: Model dimension (embedding size)
            dim_feedforward: Feedforward network dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__()

        self.feat_dim = feat_dim
        self.d_model = d_model

        # Input projection with scaling (important for transformers)
        self.input_projection = nn.Linear(feat_dim, d_model, bias=True)
        self.scale = math.sqrt(d_model)

        # Layer normalization after projection
        self.input_norm = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.input_dropout = nn.Dropout(dropout)

        # Stack of transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            EnhancedEncoderBlock(
                d_model=d_model,
                n_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])

        # Classification head with additional regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Special initialization for attention
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                    nn.init.zeros_(module.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch_size, feat_dim)
            return_attention: Whether to return attention weights

        Returns:
            logits: Classification logits (batch_size, num_classes)
            attentions: Optional list of attention weights from each layer
        """
        # Input projection and scaling
        x = self.input_projection(x) / self.scale
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch, 1, d_model)

        # Pass through encoder layers
        attentions = []
        for layer in self.encoder_layers:
            if return_attention:
                # Modified to capture attention
                x_norm = layer.norm1(x)
                attn_output, attn_weights = layer.self_attn(
                    x_norm, x_norm, x_norm, need_weights=True
                )
                attentions.append(attn_weights)
                x = x + layer.dropout(attn_output)
                x_norm = layer.norm2(x)
                x = x + layer.ffn(x_norm)
            else:
                x = layer(x)

        # Remove sequence dimension and classify
        x = x.squeeze(1)  # (batch, d_model)
        logits = self.classifier(x)

        if return_attention:
            return logits, attentions
        return logits

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SingleAtlasTransformer',
            'feat_dim': self.feat_dim,
            'd_model': self.d_model,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        } 