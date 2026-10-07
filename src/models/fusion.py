"""
AstroFoundation — Multi-Modal Cross-Attention Fusion
===================================================
Fuses representations across Light Curves, FITS Images, and Spectra via:
1. Pairwise multi-head cross-attention (LC <-> Img, LC <-> Spec, Img <-> Spec)
2. Learnable gating network (sigma / softmax confidence weighting)
3. Modality dropout (p=0.15 during training, keeping at least one modality active)
Outputs a strictly contracted (BATCH, 256) L2 unit-norm fused embedding.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention block between two modality representations.
    Treats modality A and modality B as sequence tokens [A, B] and applies
    multi-head attention followed by Pre-LN feed-forward refinement.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha_a2b = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_b2a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm_a1 = nn.LayerNorm(embed_dim)
        self.norm_b1 = nn.LayerNorm(embed_dim)
        
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        self.norm_a2 = nn.LayerNorm(embed_dim)
        self.norm_b2 = nn.LayerNorm(embed_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            a, b: Modality tensors of shape (BATCH, embed_dim)
        Returns:
            a_out, b_out: Cross-attended tensors of shape (BATCH, embed_dim)
        """
        a_seq = a.unsqueeze(1)
        b_seq = b.unsqueeze(1)

        # Cross attention: a attends to b, b attends to a
        attn_a, _ = self.mha_a2b(a_seq, b_seq, b_seq)
        attn_b, _ = self.mha_b2a(b_seq, a_seq, a_seq)

        a_res = self.norm_a1(a + attn_a.squeeze(1))
        b_res = self.norm_b1(b + attn_b.squeeze(1))

        a_out = self.norm_a2(a_res + self.ffn_a(a_res))
        b_out = self.norm_b2(b_res + self.ffn_b(b_res))

        return a_out, b_out


class ModalityDropout(nn.Module):
    """
    Stochastically drops entire modalities during training with probability p.
    Guarantees that at least one modality remains present per sample.
    """
    def __init__(self, p: float = 0.15):
        super().__init__()
        self.p = p

    def forward(
        self,
        lc: Optional[torch.Tensor],
        img: Optional[torch.Tensor],
        spec: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.training or self.p <= 0.0:
            return lc, img, spec

        available = [m for m in [lc, img, spec] if m is not None]
        if len(available) <= 1:
            return lc, img, spec

        batch_size = available[0].shape[0]
        device = available[0].device

        # Random mask: shape (3,)
        keep_mask = torch.rand(3, device=device) >= self.p
        if not keep_mask.any():
            # Keep at least one randomly
            keep_mask[torch.randint(0, 3, (1,))] = True

        out_lc = lc if (lc is not None and keep_mask[0]) else None
        out_img = img if (img is not None and keep_mask[1]) else None
        out_spec = spec if (spec is not None and keep_mask[2]) else None

        # Fallback if all non-None modalities got masked
        if out_lc is None and out_img is None and out_spec is None:
            # Revert to whatever was provided
            return lc, img, spec

        return out_lc, out_img, out_spec


class GatingNetwork(nn.Module):
    """
    Computes learned scalar gating weights across modalities/pairs
    using a sigmoid-activated linear projection.
    """
    def __init__(self, embed_dim: int = 256, num_branches: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, branch_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Weights each available branch by its gating score and pools them.
        """
        weights = [self.gate(b) for b in branch_tensors]  # List of (B, 1)
        total_w = sum(weights) + 1e-8
        norm_weights = [w / total_w for w in weights]
        
        fused = sum(w * b for w, b in zip(norm_weights, branch_tensors))
        return fused


class CrossAttentionFusion(nn.Module):
    """
    Full Tri-Modal Cross-Attention Fusion module.
    Ingests projected (B, 256) embeddings for LC, Image, and Spectra,
    applies pairwise cross-attention for all active pairs, and gates
    the multi-modal representations into a final (B, 256) unit-norm vector.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout_p: float = 0.15):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_dropout = ModalityDropout(p=dropout_p)
        
        # 3 Pairwise cross-attention blocks
        self.cross_lc_img = CrossAttentionBlock(embed_dim, num_heads)
        self.cross_lc_spec = CrossAttentionBlock(embed_dim, num_heads)
        self.cross_img_spec = CrossAttentionBlock(embed_dim, num_heads)
        
        # Gating network
        self.gating = GatingNetwork(embed_dim)
        
        # Final output projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        lc_emb: Optional[torch.Tensor] = None,
        img_emb: Optional[torch.Tensor] = None,
        spec_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            lc_emb: Projected Light Curve embedding (BATCH, 256) or None
            img_emb: Projected FITS Image embedding (BATCH, 256) or None
            spec_emb: Projected Spectral embedding (BATCH, 256) or None
        Returns:
            fused_embedding: Fused unit-norm hypersphere vector (BATCH, 256)
        """
        # Step 1: Modality dropout during training
        lc, img, spec = self.modality_dropout(lc_emb, img_emb, spec_emb)

        # Collect available representations
        active_branches = []

        # Step 2: Pairwise cross-attentions
        if lc is not None and img is not None:
            lc_img_a, lc_img_b = self.cross_lc_img(lc, img)
            active_branches.append((lc_img_a + lc_img_b) * 0.5)

        if lc is not None and spec is not None:
            lc_sp_a, lc_sp_b = self.cross_lc_spec(lc, spec)
            active_branches.append((lc_sp_a + lc_sp_b) * 0.5)

        if img is not None and spec is not None:
            im_sp_a, im_sp_b = self.cross_img_spec(img, spec)
            active_branches.append((im_sp_a + im_sp_b) * 0.5)

        # Include unimodal fallbacks if no pairs are formed (e.g. single modality present)
        if not active_branches:
            for m in [lc, img, spec]:
                if m is not None:
                    active_branches.append(m)

        if not active_branches:
            raise ValueError("CrossAttentionFusion received all None modalities.")

        # Step 3: Gated combination
        fused = self.gating(active_branches)

        # Step 4: Final projection and L2 normalization
        out = self.fusion_proj(fused)
        normalized = F.normalize(out, p=2, dim=-1, eps=1e-8)

        return normalized
