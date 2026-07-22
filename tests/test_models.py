import torch
import torch.nn as nn
import pytest
from src.models.embedding_validator import validate_embedding
from src.models.image_encoder import ImageEncoder
from src.models.spectral_encoder import SpectralEncoder
from src.models.transformer_encoder import LightCurveEncoder

BATCH_SIZE=8

def test_light_curve_encoder_stub():
    global BATCH_SIZE
    tensor=torch.randn((BATCH_SIZE,1024))
    inst=LightCurveEncoder()
    output=inst(tensor)
    validate_embedding(output,"LightCurveEncoder")
    assert output.shape==(8,256)
def test_spectral_encoder_stub():
    global BATCH_SIZE
    tensor=torch.randn((BATCH_SIZE,128))
    inst=SpectralEncoder()
    output=inst(tensor)
    validate_embedding(output,"SpectralEncoder")
    assert output.shape==(8,256)
def test_image_encoder_stub():
    tensor=torch.randn((BATCH_SIZE,3,224,224))
    inst=ImageEncoder()
    output=inst(tensor)
    validate_embedding(output,"ImageEncoder")
    assert output.shape==(8,256)
     


