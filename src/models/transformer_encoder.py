import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal tokens.
    """
    def __init__(self, d_model: int = 256, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (BATCH, SEQ_LEN, D_MODEL)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LightCurveEncoder(nn.Module):
    """
    AstroFoundation v4.0 Light Curve Transformer Backbone.
    Ingests 1D astronomical flux timelines (e.g. 8192 steps),
    applies patch-based temporal embedding + learnable [CLS] token,
    multi-head self-attention transformer blocks, and outputs a strictly
    contracted (BATCH, 256) float32 L2 unit-norm embedding vector.
    """
    def __init__(
        self, 
        input_dim: int = 1, 
        d_model: int = 256, 
        nhead: int = 8, 
        num_layers: int = 4, 
        patch_size: int = 16,
        max_len: int = 8192, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        
        # 1. Patch-based temporal projection (maps 1D flux chunks -> d_model continuous tokens)
        self.patch_proj = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 2. Learnable [CLS] token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 3. Positional encoding for patched sequence + CLS token
        max_tokens = (max_len // patch_size) + 64
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_tokens)
        
        # 4. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Projection Head for strict 256-dim output contract
        self.output_proj = nn.Linear(d_model, 256)

    def forward(self, flux: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            flux: Raw resampled light curves of shape (BATCH, 8192) or (BATCH, 1, 8192) or (BATCH, 8192, 1).
            mask: Telemetry boolean mask of shape (BATCH, 8192).
                  True = Valid observation, False = Telemetry gap interpolation.
        Returns:
            normalized_embedding: Unit hypersphere vector of shape (BATCH, 256) with Euclidean norm of 1.0.
        """
        # Step 1: Ensure input is 3D -> (BATCH, 1, SEQ_LEN)
        if flux.dim() == 2:
            x_in = flux.unsqueeze(1)
        elif flux.dim() == 3:
            if flux.size(1) != 1 and flux.size(2) == 1:
                x_in = flux.transpose(1, 2)
            else:
                x_in = flux
        else:
            raise ValueError(f"Expected 2D or 3D flux tensor, got shape {flux.shape}")

        batch_size = x_in.size(0)

        # Step 2: Temporal Patch Embedding -> (BATCH, d_model, num_patches) -> (BATCH, num_patches, d_model)
        patches = self.patch_proj(x_in).transpose(1, 2)

        # Step 3: Prepend learnable [CLS] token -> (BATCH, num_patches + 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)

        # Step 4: Add positional encoding
        x = self.pos_encoder(x)

        # Step 5: Handle telemetry mask
        padding_mask = None
        if mask is not None:
            # Pool mask over patch windows: True if any observation in patch is valid
            mask_float = mask.float().unsqueeze(1)
            pooled_mask = F.max_pool1d(
                mask_float, 
                kernel_size=self.patch_size, 
                stride=self.patch_size
            ).squeeze(1).bool()
            # CLS token is always valid (prepend True)
            cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device)
            valid_mask = torch.cat([cls_valid, pooled_mask], dim=1)
            # Invert: PyTorch src_key_padding_mask requires True for tokens that should be ignored
            padding_mask = ~valid_mask

        # Step 6: Transformer forward pass
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Step 7: Extract [CLS] token representation -> (BATCH, d_model)
        cls_output = x[:, 0, :]

        # Step 8: Project to 256-dim and apply L2 unit-norm normalization
        emb = self.output_proj(cls_output)
        normalized_embedding = F.normalize(emb, p=2, dim=-1, eps=1e-8)

        return normalized_embedding