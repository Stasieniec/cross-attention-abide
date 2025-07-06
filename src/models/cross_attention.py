import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


class CrossModalAttention(nn.Module):
    """Cross-attention module for fMRI-sMRI interaction."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize cross-modal attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-attention.
        
        Args:
            query: Features from one modality (batch, seq_len, d_model)
            key_value: Features from other modality (batch, seq_len, d_model)
            
        Returns:
            output: Updated query features
            attn_weights: Attention weights
        """
        # Cross-attention
        attn_output, attn_weights = self.cross_attn(
            query, key_value, key_value
        )
        query = self.norm1(query + attn_output)

        # FFN
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)

        return output, attn_weights


class ModalitySpecificEncoder(nn.Module):
    """Modality-specific transformer encoder."""

    def __init__(
        self, 
        input_dim: int, 
        d_model: int, 
        n_heads: int, 
        n_layers: int, 
        dropout: float = 0.1,
        use_cls_token: bool = True  # NEW: Control CLS token usage
    ):
        """
        Initialize modality-specific encoder.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
            use_cls_token: Whether to use CLS tokens (True for fMRI, False for sMRI)
        """
        super().__init__()
        
        self.use_cls_token = use_cls_token

        # Input projection
        if use_cls_token:
            # Original approach for fMRI (sequence-like data)
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            # Learnable [CLS] token for fMRI
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        else:
            # Working notebook approach for sMRI (tabular data)
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.BatchNorm1d(d_model),  # BatchNorm1d like working notebook
                nn.ReLU(),                # ReLU like working notebook
                nn.Dropout(dropout)
            )
            # Positional embedding for single position (working notebook approach)
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through modality encoder.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Encoded features (batch_size, seq_len, d_model) or (batch_size, d_model)
        """
        # Project input
        x = self.input_projection(x)

        if self.use_cls_token:
            # Original fMRI approach with CLS tokens
            # Add sequence dimension if needed
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            # Add [CLS] token
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Pass through transformer layers
            for layer in self.layers:
                x = layer(x)
                
            return x  # (batch_size, seq_len+1, d_model)
        else:
            # Working notebook sMRI approach - direct processing
            # Add sequence dimension and positional embedding (working notebook style)
            x = x.unsqueeze(1)  # (batch_size, 1, d_model)
            x = x + self.pos_embedding

            # Pass through transformer layers
            for layer in self.layers:
                x = layer(x)

            # Global pooling (working notebook style) - squeeze back to (batch, d_model)
            x = x.squeeze(1)  # (batch_size, d_model)
            return x


class CrossAttentionTransformer(nn.Module):
    """Main cross-attention transformer for multimodal autism classification."""

    def __init__(
        self,
        fmri_dim: int,
        smri_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_cross_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        """
        Initialize cross-attention transformer.
        
        Args:
            fmri_dim: fMRI input dimension
            smri_dim: sMRI input dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of layers per modality encoder
            n_cross_layers: Number of cross-attention layers
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__()

        self.fmri_dim = fmri_dim
        self.smri_dim = smri_dim
        self.d_model = d_model

        # Modality-specific encoders
        self.fmri_encoder = ModalitySpecificEncoder(
            fmri_dim, d_model, n_heads, n_layers // 2, dropout, True  # fMRI uses CLS tokens
        )
        self.smri_encoder = ModalitySpecificEncoder(
            smri_dim, d_model, n_heads, n_layers // 2, dropout, False  # sMRI uses direct processing
        )

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Final transformer layers after fusion
        self.final_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layers // 2
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, 
        fmri_features: torch.Tensor, 
        smri_features: torch.Tensor, 
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]]]]:
        """
        Forward pass through cross-attention transformer.
        
        Args:
            fmri_features: fMRI input features (batch_size, fmri_dim)
            smri_features: sMRI input features (batch_size, smri_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits (batch_size, num_classes)
            attentions: Optional list of attention weights
        """
        # Encode each modality
        fmri_encoded = self.fmri_encoder(fmri_features)  # (batch, seq_len+1, d_model)
        smri_encoded = self.smri_encoder(smri_features)  # (batch, d_model) - direct output

        # Convert sMRI to sequence format for cross-attention
        smri_sequence = smri_encoded.unsqueeze(1)  # (batch, 1, d_model)

        attention_weights = []

        # Apply cross-attention layers
        for i, cross_layer in enumerate(self.cross_attention_layers):
            if i % 2 == 0:
                # fMRI attends to sMRI
                fmri_encoded, attn_fmri_to_smri = cross_layer(fmri_encoded, smri_sequence)
                if return_attention:
                    attention_weights.append(('fmri_to_smri', attn_fmri_to_smri))
            else:
                # sMRI attends to fMRI (update sMRI sequence)
                smri_sequence, attn_smri_to_fmri = cross_layer(smri_sequence, fmri_encoded)
                if return_attention:
                    attention_weights.append(('smri_to_fmri', attn_smri_to_fmri))

        # Extract features
        fmri_cls = fmri_encoded[:, 0]  # Extract CLS token from fMRI (batch, d_model)
        smri_cls = smri_sequence.squeeze(1)  # Convert back to (batch, d_model)

        # Fuse modalities
        fused = torch.cat([fmri_cls, smri_cls], dim=-1)
        fused = self.fusion(fused)  # (batch, d_model)

        # Add sequence dimension for final transformer
        fused = fused.unsqueeze(1)  # (batch, 1, d_model)

        # Final processing
        output = self.final_transformer(fused)
        output = output.squeeze(1)  # (batch, d_model)

        # Classification
        logits = self.classifier(output)

        if return_attention:
            return logits, attention_weights
        return logits

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CrossAttentionTransformer',
            'fmri_dim': self.fmri_dim,
            'smri_dim': self.smri_dim,
            'd_model': self.d_model,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        } 