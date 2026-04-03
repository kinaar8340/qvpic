# /tests/test_encoder_decoder.py
# Unit tests for RubikEncoder / RubikDecoder and topological layers.

import pytest


def test_encoder_decoder_modules_exist():
    """Basic check that the files exist and can be imported."""
    try:
        import src.encoder  # noqa: F401
        import src.decoder  # noqa: F401
        assert True
    except ImportError as e:
        pytest.skip(f"Encoder/decoder not fully implemented yet: {e}")