#!/usr/bin/env python3
"""Unit tests for self-improver CLI parsing and JSON extraction robustness."""

import json
import re
from scripts.agent import parse_cli_command
from scripts.self_improver import extract_proposal_json


def test_parse_cli_command_long_stem():
    """Test /self-apply with long stem (no spaces) goes fully to value."""
    verb, key, value = parse_cli_command("/self-apply 20240613_123456_improve-fidelity-with-very-long-stem-and-more")
    assert verb == "self-apply"
    assert value == "20240613_123456_improve-fidelity-with-very-long-stem-and-more"
    assert "long-stem" in value


def test_parse_cli_command_long_goal():
    """Test /self-propose with long multi-word goal captures full text in value."""
    verb, key, value = parse_cli_command("/self-propose Improve benchmark by adding better metrics in run_benchmark_lite and make JSON extraction more robust with long description here")
    assert verb == "self-propose"
    assert value.startswith("Improve benchmark")
    assert "long description here" in value
    assert len(value) > 50


def test_parse_cli_command_legacy_add():
    """Legacy /add key "value with spaces" still works."""
    verb, key, value = parse_cli_command('/add core/identity/name "Aaron with long name here"')
    assert verb == "add"
    assert key == "core/identity/name"
    assert value == "Aaron with long name here"


def test_parse_cli_command_self_proposals():
    """ /self-proposals with arg."""
    verb, key, value = parse_cli_command("/self-proposals accepted")
    assert verb == "self-proposals"
    assert value == "accepted"


def test_parse_cli_command_self_cycle():
    """ /self-cycle with long goal."""
    verb, key, value = parse_cli_command("/self-cycle make the UI more robust for long goals and proposals")
    assert verb == "self-cycle"
    assert "make the UI more robust" in value


def test_extract_proposal_json_direct():
    """Direct valid JSON."""
    raw = '{"goal": "test", "risk_level": "low", "files_to_change": ["scripts/foo.py"]}'
    prop = extract_proposal_json(raw, "forced-goal")
    assert prop["goal"] == "forced-goal"  # always forced
    assert prop["risk_level"] == "low"


def test_extract_proposal_json_with_fences():
    """Strips markdown fences."""
    raw = """```json
{"goal": "foo", "risk_level": "low", "unified_diff": "diff --git ..."}
```"""
    prop = extract_proposal_json(raw, "bar")
    assert prop["goal"] == "bar"
    assert "diff --git" in prop["unified_diff"]


def test_extract_proposal_json_fallback():
    """Falls back gracefully on bad output, still sets goal."""
    raw = "Here is some explanation and then {bad json without closing"
    prop = extract_proposal_json(raw, "my-goal")
    assert prop["goal"] == "my-goal"
    assert "raw_output" in prop


def test_extract_proposal_json_with_text_before_after():
    """Handles LLM adding text."""
    raw = "Sure, here is the proposal:\n{\"goal\": \"x\", \"risk_level\": \"medium\"}\nHope that helps!"
    prop = extract_proposal_json(raw, "y")
    assert prop["goal"] == "y"
    assert prop["risk_level"] == "medium"
