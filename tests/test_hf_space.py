"""Smoke tests for the Hugging Face Space bundle."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space" / "qvpic"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_hf_space_bundle_imports():
    sync = ROOT / "scripts" / "sync_hf_space.sh"
    assert sync.is_file()
    subprocess.run(["bash", str(sync)], check=True, cwd=ROOT)

    assert (SPACE / "app.py").is_file()
    assert (SPACE / "demo_core.py").is_file()
    assert (SPACE / "src").is_dir()
    assert (SPACE / "facts" / "demo_public_facts.json").is_file()

    sys.path.insert(0, str(SPACE))
    try:
        demo_core = _load_module("qvp_demo_core", SPACE / "demo_core.py")
        params = demo_core.default_run_params()
        assert "bake_steps" in params
        assert demo_core.is_hf_space() is False

        pytest.importorskip("gradio")
        app_mod = _load_module("qvp_app", SPACE / "app.py")
        blocks = app_mod.build_app()
        assert blocks is not None
    finally:
        sys.path.remove(str(SPACE))
        for key in ("qvp_demo_core", "qvp_app", "demo_core", "build_info", "src"):
            sys.modules.pop(key, None)