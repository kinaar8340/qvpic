# /tests/test_encoder_decoder.py
# Unit tests for RubikEncoder / RubikDecoder and topological layers.

import torch
import pytest

from src.encoder import RubikEncoder
from src.decoder import RubikDecoder


@pytest.fixture
def sample_batch():
    """Small batch for encoder/decoder tests."""
    return torch.randn(4, 128)  # adjust dimensions to match your model


def test_rubik_encoder_instantiation():
    encoder = RubikEncoder()
    assert encoder is not None


def test_encoder_forward(sample_batch):
    encoder = RubikEncoder()
    output = encoder(sample_batch)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == sample_batch.shape[0]


def test_encoder_decoder_roundtrip(sample_batch):
    """Smoke test: encode → decode should not crash."""
    encoder = RubikEncoder()
    decoder = RubikDecoder()

    encoded = encoder(sample_batch)
    decoded = decoder(encoded)

    assert decoded.shape[0] == sample_batch.shape[0]

# Add more tests as you implement Minimal Copresheaf TNN, etc.