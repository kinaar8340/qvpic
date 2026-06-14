#!/usr/bin/env python3
"""
scripts/bud_squad_bridge.py — Example integration between Bud DevSquad (~/Bud/Team) and QVPIC memory.

The DevSquad (@Yuhan, @Sansa, @Arya) can treat a running QVPIC conduit + agent journals/facts
as the shared, drift-proof long-term memory for the permanent coding team.

Usage sketch (inside the squad or a meta process):
  - Yuhan designs an improvement to qvpic.
  - Sansa implements (edits files inside the qvpic tree).
  - Arya reviews + runs /self-eval or full qvpic_test.
  - All three append "team memory" via the agent_self_improve path (or call the conduit bake APIs).
  - Future squad tasks start by recalling past work with perfect fidelity from the helix.

This file is a thin example / reference. Real usage happens by @-mentioning the squad
with the qvpic worktree as the active project and using the /self-* commands or the
self_improver API directly.
"""

import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Bud DevSquad <-> QVPIC bridge (reference)")
print("Point your squad agents at this tree and use the new self-improvement CLI or API.")
print("See docs/self_improvement.md and scripts/self_improver.py")
print("Example recall of past squad work is automatic via the conduit once facts are baked.")
print("To bake a squad decision:")
print('  python -c "from agent import append_fact; append_fact(\"Squad decided X\", fact_type=\"journal\", source=\"agent_self_improve\")"')
print("\n(Requires the full qvpic agent environment + LLM + conduit loaded.)")
