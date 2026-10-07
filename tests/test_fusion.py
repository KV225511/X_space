"""Unit tests for AstroFoundation Multi-Modal Cross-Attention Fusion & Gating Network."""

import pytest
import torch
import torch.nn as nn

from src.models.fusion import CrossAttentionFusion, ModalityDropout, GatingNetwork
from src.models.projection_heads import MultiModalProjectionHead, ModalityProjectionHead
from src.models.embedding_validator import validate_embedding


@pytest.fixture
def dummy_embeddings():
    batch_size = 4
    dim = 256
    e_lc = torch.randn(batch_size, dim)
    e_img = torch.randn(batch_size, dim)
    e_spec = torch.randn(batch_size, dim)
    return e_lc, e_img, e_spec


def test_modality_projection_head_normalization():
    """Verify projection heads project to 256-dim unit hypersphere."""
    proj = ModalityProjectionHead(in_features=256, out_features=256)
    x = torch.randn(8, 256)
    out = proj(x)
    assert out.shape == (8, 256)
    validate_embedding(out, "ModalityProjection")
    norm = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_cross_attention_fusion_tri_modal(dummy_embeddings):
    """Verify tri-modal fusion produces valid unit-normalized 256-dim fused vector."""
    e_lc, e_img, e_spec = dummy_embeddings
    fusion = CrossAttentionFusion(embed_dim=256, num_heads=8, dropout_p=0.0)
    fused = fusion(e_lc, e_img, e_spec)
    assert fused.shape == (4, 256)
    validate_embedding(fused, "TriModal-Fused")


def test_cross_attention_fusion_partial_modalities(dummy_embeddings):
    """Verify graceful degraded operation with missing modalities."""
    e_lc, e_img, e_spec = dummy_embeddings
    fusion = CrossAttentionFusion(embed_dim=256, num_heads=8, dropout_p=0.0)

    # Bi-modal: LC + Spectra (missing image)
    fused_bi = fusion(e_lc, None, e_spec)
    assert fused_bi.shape == (4, 256)
    validate_embedding(fused_bi, "BiModal-LC-Spec")

    # Uni-modal: Light curve only
    fused_uni = fusion(e_lc, None, None)
    assert fused_uni.shape == (4, 256)
    validate_embedding(fused_uni, "UniModal-LC")


def test_modality_dropout_guarantees_active_modality(dummy_embeddings):
    """Verify ModalityDropout guarantees at least one modality remains active."""
    e_lc, e_img, e_spec = dummy_embeddings
    drop = ModalityDropout(p=0.5)
    drop.train()

    for _ in range(20):
        m_lc, m_img, m_spec = drop(e_lc, e_img, e_spec)
        active_count = sum(m is not None for m in [m_lc, m_img, m_spec])
        assert active_count >= 1, "Modality dropout must guarantee at least one modality remains!"


def test_gating_network_weights(dummy_embeddings):
    """Verify gating network pools branch tensors into weighted combination."""
    e_lc, e_img, e_spec = dummy_embeddings
    gating = GatingNetwork(embed_dim=256)
    fused = gating([e_lc, e_img, e_spec])

    assert fused.shape == (4, 256)
    validate_embedding(fused, "Gating-Pooled")
