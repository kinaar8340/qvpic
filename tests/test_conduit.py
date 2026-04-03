# /tests/test_conduit.py
# Unit tests for core conduit modules.

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
 """Correct input shapes for RubikConeConduit (now matches the encoder/face_embed)."""
 conduit = RubikConeConduit(embed_dim=384)

 batch_size = 2
 # Correct shape that produces the expected 54-feature input to face_embed Linear
 face_grids = torch.randn(batch_size, 6, 9, 9, 384) # ← this was already correct
 orientations = torch.randint(0, 24, (batch_size, 54))
 vortex_digits = torch.randint(0, 10, (batch_size, 54))

 # Force CPU to avoid device mismatch
 device = torch.device("cpu")
 conduit = conduit.to(device)
 face_grids = face_grids.to(device)
 orientations = orientations.to(device)
 vortex_digits = vortex_digits.to(device)

 output = conduit(face_grids, orientations, vortex_digits)
 assert isinstance(output, torch.Tensor)
 assert output.shape[0] == batch_size


def test_ring_cone_chain():
 """Fixed device handling for RingConeChain."""
 device = torch.device("cpu")
 chain = RingConeChain(embed_dim=384).to(device)

 inner = torch.randn(4, 384, device=device)
 outer = torch.randn(4, 384, device=device)

 out = chain(inner, outer)
 assert out.shape[0] == 4