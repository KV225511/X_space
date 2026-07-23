import torch
import torch.nn as nn
import pytest
from src.models.embedding_validator import validate_embedding
from src.models.image_encoder import FITSEncoder
from src.models.spectral_encoder import SpectralEncoder
from src.models.transformer_encoder import LightCurveEncoder

BATCH_SIZE = 8

def test_light_curve_encoder_stub():
    global BATCH_SIZE
    # Light curves expect an 8,192-step temporal sequence
    tensor = torch.randn((BATCH_SIZE, 8192))
    inst = LightCurveEncoder()
    output = inst(tensor)
    validate_embedding(output, "LightCurveEncoder")
    assert output.shape == (BATCH_SIZE, 256)

def test_spectral_encoder_stub():
    global BATCH_SIZE
    # Spectra expect a 4,096-point continuum-normalized wavelength grid
    tensor = torch.randn((BATCH_SIZE, 4096))
    inst = SpectralEncoder()
    output = inst(tensor)
    validate_embedding(output, "SpectralEncoder")
    assert output.shape == (BATCH_SIZE, 256)

def test_image_encoder_stub():
    global BATCH_SIZE
    # FITS images expect 5 photometric channels (u, g, r, i, z), not 3 RGB channels
    tensor = torch.randn((BATCH_SIZE, 5, 224, 224))
    inst = FITSEncoder()
    output = inst(tensor)
    validate_embedding(output, "FITSEncoder")
    assert output.shape == (BATCH_SIZE, 256)