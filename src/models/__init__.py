"""
AstroFoundation — Models Package
"""

from .transformer_encoder import LightCurveEncoder, PositionalEncoding
from .image_encoder import FITSEncoder
from .spectral_encoder import SpectralEncoder, SpectralBlock
from .embedding_validator import validate_embedding
from .projection_heads import ModalityProjectionHead, MultiModalProjectionHead
from .fusion import CrossAttentionBlock, ModalityDropout, GatingNetwork, CrossAttentionFusion
from .task_heads import ClassificationHead, AnomalyHead, TaskHeads, TaskHeadOutput
from .full_model import AstroFoundationModel

__all__ = [
    "LightCurveEncoder",
    "PositionalEncoding",
    "FITSEncoder",
    "SpectralEncoder",
    "SpectralBlock",
    "validate_embedding",
    "ModalityProjectionHead",
    "MultiModalProjectionHead",
    "CrossAttentionBlock",
    "ModalityDropout",
    "GatingNetwork",
    "CrossAttentionFusion",
    "ClassificationHead",
    "AnomalyHead",
    "TaskHeads",
    "TaskHeadOutput",
    "AstroFoundationModel",
]
