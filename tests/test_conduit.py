# /tests/test_conduit.py
# Unit tests for core conduit modules.

import pytest
import torch

from src.conduit import (
    TwistedHelicalConduit,
    RubikConeConduit,
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


@pytest.mark.skip(reason="TODO: match exact RubikEncoder input shape for forward pass")
def test_rubik_cone_conduit_forward():
    pass


@pytest.mark.skip(reason="TODO: fix device mismatch in RingConeChain")
def test_ring_cone_chain():
    pass