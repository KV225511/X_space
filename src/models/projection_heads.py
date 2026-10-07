"""
AstroFoundation — Modality Projection Heads
============================================
Projects 256-dimensional unit embeddings from each modality encoder into
a shared metric hypersphere representation space before cross-attention fusion.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjectionHead(nn.Module):
    """
    Non-linear projection head for a single modality.
    Applies: Linear(256 -> 256) -> LayerNorm -> GELU -> Linear(256 -> 256) -> L2 Normalization.
    """
    def __init__(self, in_features: int = 256, out_features: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw encoder embedding of shape (BATCH, in_features).
        Returns:
            Projected unit-norm embedding of shape (BATCH, out_features).
        """
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1, eps=1e-8)


class MultiModalProjectionHead(nn.Module):
    """
    Tri-modal projection container coordinating independent projection heads
    for Light Curve, FITS Image, and Spectral representations.
    """
    def __init__(self, embed_dim: int = 256, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lc_proj = ModalityProjectionHead(embed_dim, proj_dim, dropout)
        self.img_proj = ModalityProjectionHead(embed_dim, proj_dim, dropout)
        self.spec_proj = ModalityProjectionHead(embed_dim, proj_dim, dropout)

    def forward(
        self,
        lc_emb: Optional[torch.Tensor] = None,
        img_emb: Optional[torch.Tensor] = None,
        spec_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Projects provided embeddings into the common space.
        None inputs are passed through as None to support missing modalities.
        """
        p_lc = self.lc_proj(lc_emb) if lc_emb is not None else None
        p_img = self.img_proj(img_emb) if img_emb is not None else None
        p_spec = self.spec_proj(spec_emb) if spec_emb is not None else None
        return p_lc, p_img, p_spec
