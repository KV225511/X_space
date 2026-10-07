"""
AstroFoundation — Full Multi-Modal Architecture
===============================================
Assembles Encoders -> Projection Heads -> Cross-Attention Fusion -> Task Heads
into an end-to-end PyTorch module.
Supports:
- Full tri-modal forward passes (Light Curve + FITS + Spectra)
- Bi-modal and Uni-modal inference with graceful degradation
- Structured prediction outputs via TaskHeadOutput
"""

from typing import Optional
import torch
import torch.nn as nn

from .transformer_encoder import LightCurveEncoder
from .image_encoder import FITSEncoder
from .spectral_encoder import SpectralEncoder
from .projection_heads import MultiModalProjectionHead
from .fusion import CrossAttentionFusion
from .task_heads import TaskHeads, TaskHeadOutput


class AstroFoundationModel(nn.Module):
    """
    AstroFoundation v4.0 Unified Multi-Modal Foundation Model.
    Ingests any combination of:
      1. Light curve timelines: (BATCH, 8192) + optional mask: (BATCH, 8192)
      2. Multi-band FITS images: (BATCH, 3, 224, 224) or (BATCH, 5, 224, 224)
      3. 1D Stellar spectra: (BATCH, 4096)
    Fuses representations via pairwise cross-attention and predicts 6-class
    astronomical event labels + energy-based anomaly detection.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 6,
        fits_channels: int = 3,
        num_heads: int = 8,
        modality_dropout: float = 0.15,
        anomaly_threshold: float = 0.0,
    ):
        super().__init__()
        # 1. Modality Encoders
        self.lc_encoder = LightCurveEncoder(d_model=embed_dim)
        self.image_encoder = FITSEncoder(in_channels=fits_channels, d_model=embed_dim)
        self.spectral_encoder = SpectralEncoder(d_model=embed_dim)

        # 2. Metric Projection Heads
        self.projections = MultiModalProjectionHead(embed_dim=embed_dim, proj_dim=embed_dim)

        # 3. Cross-Attention Fusion Backbone
        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_p=modality_dropout
        )

        # 4. Multi-Task Heads
        self.task_heads = TaskHeads(
            in_features=embed_dim,
            num_classes=num_classes,
            anomaly_threshold=anomaly_threshold
        )

    def extract_features(
        self,
        flux: Optional[torch.Tensor] = None,
        fits: Optional[torch.Tensor] = None,
        spec: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extracts the 256-dim fused unit embedding without task head classification.
        Useful for zero-shot retrieval, vector indexing, and clustering.
        """
        # Step 1: Modality Encoders
        e_lc = self.lc_encoder(flux, mask=mask) if flux is not None else None
        e_img = self.image_encoder(fits) if fits is not None else None
        e_spec = self.spectral_encoder(spec) if spec is not None else None

        # Step 2: Hypersphere Projections
        p_lc, p_img, p_spec = self.projections(e_lc, e_img, e_spec)

        # Step 3: Multi-Modal Fusion
        fused = self.fusion(p_lc, p_img, p_spec)
        return fused

    def forward(
        self,
        flux: Optional[torch.Tensor] = None,
        fits: Optional[torch.Tensor] = None,
        spec: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> TaskHeadOutput:
        """
        End-to-end forward pass.
        At least one modality (flux, fits, or spec) must be non-None.
        """
        if flux is None and fits is None and spec is None:
            raise ValueError("AstroFoundationModel received all None inputs. Provide at least one modality.")

        fused_embedding = self.extract_features(flux=flux, fits=fits, spec=spec, mask=mask)
        output = self.task_heads(fused_embedding)
        return output
