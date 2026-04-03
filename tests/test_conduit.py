# /tests/test_conduit.py
# Unit tests for core conduit modules.

import pytest
import torch
from omegaconf import OmegaConf

from src.conduit import (
    TwistedHelicalConduit,
    RubikConeConduit,
    VQCEnhancedHelicalConduit,
)
from src.config import load_default_config  # assuming this exists


@pytest.fixture
def default_config():
    """Load the default config for tests."""
    return load_default_config()  # or OmegaConf.create({ ... minimal config ... })


def test_twisted_helical_conduit_instantiation(default_config):
    """Basic smoke test: can we create the continuous conduit?"""
    conduit = TwistedHelicalConduit(config=default_config)
    assert conduit is not None
    assert hasattr(conduit, "forward")  # or whatever your main method is


def test_rubik_cone_conduit_instantiation(default_config):
    """Test the discrete 216-cube hierarchical conduit."""
    conduit = RubikConeConduit(config=default_config)
    assert conduit is not None


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_conduit_forward_pass(default_config, batch_size):
    """Ensure forward pass runs without crashing and returns correct shape."""
    conduit = TwistedHelicalConduit(config=default_config)

    # Example dummy input – adjust to match your actual input shape
    x = torch.randn(batch_size, conduit.input_dim)  # update input_dim if needed

    output = conduit(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == batch_size
    # Add more shape assertions based on your design


def test_quaternion_operations():
    """Quick test of any quaternion helpers you have in conduit.py."""
    # Example – replace with your actual qmul, qnormalize, etc.
    from src.conduit import qmul, qnormalize  # if these are exposed

    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    result = qmul(q1, q2)
    normalized = qnormalize(result)

    assert result.shape == (4,)
    assert torch.allclose(torch.norm(normalized), torch.tensor(1.0), atol=1e-6)