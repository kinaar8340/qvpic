#!/usr/bin/env python3
"""Minimal QVPIC memory integration — bake, recall, drift benchmark.

Run from repo root:
    python examples/agent_memory_integration.py

Use this as a template for wiring QVPIC into LangChain, CrewAI, or a custom agent loop.
See docs/INTEGRATIONS.md for framework-specific patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from demo_core import (  # noqa: E402
    default_run_params,
    run_benchmark_demo,
    run_query_recall,
)


def main() -> None:
    params = default_run_params()
    print("QVPIC agent memory integration demo\n")

    print("=== 1. Bake → recall → drift benchmark ===\n")
    result = run_benchmark_demo(
        bake_steps=params["bake_steps"],
        bandwidth=params["bandwidth"],
        use_vqc=False,
        drift_samples=min(12, params["drift_samples"]),
        max_facts=params["max_facts"],
        include_lattice=False,
    )
    print(result.metrics_text)

    print("\n=== 2. Query recall (top-k cubes) ===\n")
    recall_text = run_query_recall(
        params["query_text"],
        bake_steps=params["bake_steps"],
        bandwidth=params["bandwidth"],
        use_vqc=False,
        max_facts=params["max_facts"],
        top_k=3,
    )
    print(recall_text)

    print("\n=== 3. Agent loop hook (pseudocode) ===\n")
    print(
        "  for user_msg in agent_session:\n"
        "      hits = run_query_recall(user_msg, ...)\n"
        "      context = format_cube_hits(hits)\n"
        "      reply = llm.chat(system + context + user_msg)\n"
        "      maybe_bake_new_fact(conduit, reply)  # scripts/main.py pattern\n"
    )


if __name__ == "__main__":
    main()