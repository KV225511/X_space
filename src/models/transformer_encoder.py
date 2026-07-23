import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LightCurveEncoder(nn.Module):
    """
    AstroFoundation v4.0 Light Curve Transformer Backbone.
    Ingests 1D astronomical flux timelines, applies FlashAttention-2 SDPA, 
    and outputs a strictly contracted (BATCH, 256) float32 L2 unit-norm embedding vector.
    """
    def __init__(
        self, 
        input_dim: int = 1, 
        d_model: int = 256, 
        nhead: int = 8, 
        num_layers: int = 6, 
        max_len: int = 8192, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Feature Expansion (1D flux -> 256D continuous feature space)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Positional Time Stamping (N=8192)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        # 3. Pre-LN Transformer Layer configured for FlashAttention-2 (SDPA)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-Layer Normalization for deep gradient stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Projection Head for strict 256-dim output contract
        self.output_proj = nn.Linear(d_model, 256)

    def forward(self, flux: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            flux: Raw resampled light curves of shape (BATCH, 8192) or (BATCH, 8192, 1).
            mask: Telemetry boolean mask of shape (BATCH, 8192).
                  True = Valid observation, False = Telemetry gap interpolation.
        Returns:
            normalized_embedding: Unit hypersphere vector of shape (BATCH, 256) with Euclidean norm of 1.0.
        """
        # Step 1: Ensure input is 3D -> (BATCH, SEQ_LEN, 1)
        if flux.dim() == 2:
            flux = flux.unsqueeze(-1)
            
        # Step 2: Project into feature space and stamp time coordinates
        x = self.input_proj(flux)
        x = self.pos_encoder(x)
        
        # Step 3: Automated Mask Inversion for PyTorch SDPA
        # Inverts valid observation flags (True) into attention ignore tokens (False -> True)
        padding_mask = None
        if mask is not None:
            padding_mask = ~mask.to(torch.bool)
            
        # Step 4: Parallel Attention Pass & Sequence Reduction
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = torch.mean(x, dim=1)  # Global Average Pooling across 8192 time steps -> (BATCH, 256)
        x = self.output_proj(x)
        
        # Step 5: Universal L2 Unit-Norm Contract (eps=1e-8 prevents zero-division NaN)
        normalized_embedding = F.normalize(x, p=2, dim=-1, eps=1e-8)
        
        return normalized_embedding
        
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int=256,max_len:int=8192):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x