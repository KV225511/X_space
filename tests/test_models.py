import torch
import torch.nn as nn
import pytest

from src.models.embedding_validator import validate_embedding
from src.models.image_encoder import FITSEncoder
from src.models.spectral_encoder import SpectralEncoder
from src.models.transformer_encoder import LightCurveEncoder
from src.models.projection_heads import ModalityProjectionHead, MultiModalProjectionHead
from src.models.fusion import CrossAttentionFusion
from src.models.task_heads import TaskHeads, TaskHeadOutput
from src.models.full_model import AstroFoundationModel

BATCH_SIZE = 4


def test_light_curve_encoder_stub():
    global BATCH_SIZE
    tensor = torch.randn((BATCH_SIZE, 8192))
    inst = LightCurveEncoder()
    output = inst(tensor)
    validate_embedding(output, "LightCurveEncoder")
    assert output.shape == (BATCH_SIZE, 256)


def test_spectral_encoder_stub():
    global BATCH_SIZE
    tensor = torch.randn((BATCH_SIZE, 4096))
    inst = SpectralEncoder()
    output = inst(tensor)
    validate_embedding(output, "SpectralEncoder")
    assert output.shape == (BATCH_SIZE, 256)


def test_image_encoder_stub():
    global BATCH_SIZE
    tensor = torch.randn((BATCH_SIZE, 3, 224, 224))
    inst = FITSEncoder(in_channels=3)
    output = inst(tensor)
    validate_embedding(output, "FITSEncoder-3ch")
    assert output.shape == (BATCH_SIZE, 256)


def test_image_encoder_5channel():
    global BATCH_SIZE
    tensor = torch.randn((BATCH_SIZE, 5, 224, 224))
    inst = FITSEncoder(in_channels=5)
    output = inst(tensor)
    validate_embedding(output, "FITSEncoder-5ch")
    assert output.shape == (BATCH_SIZE, 256)


def test_projection_heads():
    global BATCH_SIZE
    proj = MultiModalProjectionHead(embed_dim=256, proj_dim=256)
    e_lc = torch.randn(BATCH_SIZE, 256)
    e_img = torch.randn(BATCH_SIZE, 256)
    e_spec = torch.randn(BATCH_SIZE, 256)

    p_lc, p_img, p_spec = proj(e_lc, e_img, e_spec)
    assert p_lc.shape == (BATCH_SIZE, 256)
    assert p_img.shape == (BATCH_SIZE, 256)
    assert p_spec.shape == (BATCH_SIZE, 256)
    validate_embedding(p_lc, "Projected-LC")
    validate_embedding(p_img, "Projected-Img")
    validate_embedding(p_spec, "Projected-Spec")


def test_cross_attention_fusion():
    global BATCH_SIZE
    fusion = CrossAttentionFusion(embed_dim=256, num_heads=8, dropout_p=0.0)
    e_lc = torch.randn(BATCH_SIZE, 256)
    e_img = torch.randn(BATCH_SIZE, 256)
    e_spec = torch.randn(BATCH_SIZE, 256)

    # 1. Tri-modal fusion
    fused_3 = fusion(e_lc, e_img, e_spec)
    assert fused_3.shape == (BATCH_SIZE, 256)
    validate_embedding(fused_3, "Fusion-TriModal")

    # 2. Bi-modal fusion (missing image)
    fused_2 = fusion(e_lc, None, e_spec)
    assert fused_2.shape == (BATCH_SIZE, 256)
    validate_embedding(fused_2, "Fusion-BiModal")

    # 3. Uni-modal fusion (only light curve)
    fused_1 = fusion(e_lc, None, None)
    assert fused_1.shape == (BATCH_SIZE, 256)
    validate_embedding(fused_1, "Fusion-UniModal")


def test_task_heads():
    global BATCH_SIZE
    heads = TaskHeads(in_features=256, num_classes=6, anomaly_threshold=0.0)
    fused = torch.randn(BATCH_SIZE, 256)
    fused = torch.nn.functional.normalize(fused, p=2, dim=-1)

    output = heads(fused)
    assert isinstance(output, TaskHeadOutput)
    assert output.logits.shape == (BATCH_SIZE, 6)
    assert output.probabilities.shape == (BATCH_SIZE, 6)
    assert output.predicted_class.shape == (BATCH_SIZE,)
    assert output.energy_score.shape == (BATCH_SIZE,)
    assert output.is_anomaly.shape == (BATCH_SIZE,)
    assert output.fused_embedding.shape == (BATCH_SIZE, 256)
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)


def test_full_model_end_to_end():
    global BATCH_SIZE
    model = AstroFoundationModel(embed_dim=256, num_classes=6, fits_channels=3)
    model.eval()

    flux = torch.randn(BATCH_SIZE, 8192)
    fits = torch.randn(BATCH_SIZE, 3, 224, 224)
    spec = torch.randn(BATCH_SIZE, 4096)

    # Tri-modal forward pass
    with torch.no_grad():
        out_full = model(flux=flux, fits=fits, spec=spec)
        assert isinstance(out_full, TaskHeadOutput)
        assert out_full.logits.shape == (BATCH_SIZE, 6)
        validate_embedding(out_full.fused_embedding, "AstroFoundation-FullModel")

        # Feature extraction
        emb = model.extract_features(flux=flux, fits=fits, spec=spec)
        assert emb.shape == (BATCH_SIZE, 256)
        validate_embedding(emb, "AstroFoundation-Features")

        # Degraded mode: Only Light Curve provided
        out_lc_only = model(flux=flux)
        assert out_lc_only.logits.shape == (BATCH_SIZE, 6)