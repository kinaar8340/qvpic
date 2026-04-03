"""Unit tests for the real conduit.py classes (v10.2)."""

import pytest
import torch

from src.conduit import (
    TwistedHelicalConduit,
    RubikConeConduit,
    RingConeChain,
    qmul,
    qnormalize,
    safe_cosine,
)

def test_quaternion_helpers():
    """Test quaternion math utilities."""
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    result = qmul(q1, q2)
    normalized = qnormalize(result)
    assert result.shape == (4,)
    assert torch.allclose(torch.norm(normalized), torch.tensor(1.0), atol=1e-6)

def test_safe_cosine():
    """Test safe cosine similarity."""
    a = torch.randn(10, 128)
    b = torch.randn(10, 128)
    sim = safe_cosine(a, b)
    assert sim.shape == (10,)
    assert torch.all(sim >= -1.0) and torch.all(sim <= 1.0)

@pytest.mark.parametrize("cls", [TwistedHelicalConduit, RubikConeConduit])
def test_conduit_instantiation(cls):
    """Smoke test: can instantiate the main conduits."""
    conduit = cls(embed_dim=384)
    assert conduit is not None
    assert isinstance(conduit, torch.nn.Module)

def test_rubik_cone_conduit_forward():
    """Fixed: RubikConeConduit now uses correct [B, 54, embed_dim] sticker shape."""
    device = torch.device("cpu")
    conduit = RubikConeConduit(embed_dim=384).to(device)
    
    batch_size = 2
    face_grids = torch.randn(batch_size, 54, 384, device=device)      # ← 54 stickers
    orientations = torch.randint(0, 24, (batch_size, 54), device=device)
    vortex_digits = torch.randint(0, 10, (batch_size, 54), device=device)
    
    output = conduit(face_grids, orientations, vortex_digits)
    assert output.shape[0] == batch_size

def test_ring_cone_chain():
    """Fixed: RingConeChain now fully respects device for all buffers."""
    device = torch.device("cpu")
    chain = RingConeChain(embed_dim=384).to(device)
    
    batch_size = 4
    inner = torch.randn(batch_size, 384, device=device)
    outer = torch.randn(batch_size, 384, device=device)
    
    out = chain(inner, outer)
    assert out.shape[0] == batch_size
