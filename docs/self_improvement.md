# Self-Improving AI with QVPIC

**Goal**: Build a recursively self-improving AI whose "self" (knowledge, capabilities, goals, improvement autobiography, and lessons) is protected by the drift-resistant topological memory of the Quaternion Vortex Persistent Identity Conduit.

QVPIC's core value for self-improvement is **anti-drift persistence**: global invariants (winding number, quaternion braiding/linking phase, ShellCube radial differential) + copresheaf TNN higher-order reasoning on the RingConeChain act as a mathematical constitution and checksum. The agent ("Bud") can evolve aggressively while its memory of "who I was and what I learned" stays extremely high-fidelity (benchmarked 0.98–1.000 cosine recall).

## Architecture

- **Substrate**: RubikConeConduit (or VQC mode) + RingConeChain + Minimal Copresheaf TNN.
  - Facts, journals, chat summaries, benchmark results, and improvement meta-data are embedded geometrically and "baked" via `training_step` + `bake_ring`.
  - `monitor_topological_winding()` provides the live "health + signature" used as a gate for all changes.

- **Guardrails (hard-coded + config-driven)**:
  - Agent may only append to its own identity files (`agent_*`) and the new `agent_self_improve` source (writes to ajournal + facts JSON).
  - Core geometry (`src/conduit.py`, agent runtime, etc.) is protected: proposals may describe changes but live patches to CORE_AREAS require explicit review (current implementation is conservative — no auto core patches).
  - Every cycle must produce before/after benchmark + topo signature.
  - All outcomes (proposals, evals, accepts/rejects) are recorded in `proposals/`, `improvements/{accepted,rejected}/` **and** baked back into the conduit.

- **Loop** (`scripts/self_improver.py` + CLI integration):
  1. **Introspect**: `get_self_source_summary()`, `run_benchmark_lite()`, `get_topological_signature()`, past cycles via `load_past_improvements()`, helix health.
  2. **Propose**: LLM (grounded on current helix facts + source + past lessons) emits structured proposal (goal, files, expected impact, risk, test plan, optional diff).
  3. **Evaluate**: Short benchmark run + topo capture. Compute deltas. Conservative heuristic.
  4. **Decide & Record**: Accept/reject logged + **baked into conduit journal/facts** with full topo stamp. This is the permanent "growth memory".
  5. (Future) Very-low-risk auto-apply in isolated context + re-validate.

## CLI (via chat or / commands in UI)

```
/self-eval
/self-propose Raise fidelity to 0.999 while keeping braiding stable
/self-cycle Improve training_step speed and add new safe meta-reasoning primitive
/self-history
```

These are also available programmatically:
```python
from scripts import self_improver as si
from agent import ...
si.wire({...})
si.run_improvement_cycle("My goal here")
```

## Directories (git-tracked)

- `proposals/` — pending ideas (timestamped JSON + topo_before)
- `improvements/accepted/` + `improvements/rejected/` — full records with before/after metrics + final topo
- `logs/improvements/` — append-only structured log

## How to Trigger Autonomous Cycles

- Manual: Use the `/self-*` commands in the Gradio chat.
- Heartbeat extension: The existing Swiss-watch heartbeat can be extended (see `main.py` + `heartbeat.py`) to occasionally call a light self-reflection or full cycle when idle.
- External driver: `python -m scripts.self_improver --goal "..."` (standalone mode has limited LLM; prefer wiring through agent).
- Multi-agent: The Bud DevSquad (`~/Bud/Team/`) can be pointed at qvpic proposals. @Yuhan designs, @Sansa implements candidates, @Arya runs the eval+review using the baked history for context.

## Safety & Constitution (from config)

```
self_improvement:
  constitution: "Topological invariants (winding, braiding_phase, ShellCube differential) are the supreme law. Any change that measurably degrades them or recall fidelity is rejected and the lesson baked."
```

Rollback is trivial: reload previous checkpoint + the conduit state encodes the "good" prior identity. Git provides source history.

## Extending

- Add more metrics to `run_benchmark_lite` / full `qvpic_test.py` (e.g. specific recall suites, training throughput).
- Teach the proposer to emit real small unified diffs for safe areas and add a `apply_proposal_safely()` that does `patch`, runs tests, then bakes.
- Use `vortex_swarm.py` / Ray for parallel proposal generation + diverse candidate evals.
- Feed the copresheaf TNN layer improvement ideas (new sheaf constructions, better message passing on the combinatorial complex) — the TNN itself is designed for higher-order reasoning that can be turned inward.

## Relation to the VQC Patent & Vision

The patent and QVPIC emphasize persistent identity via topological protection. Self-improvement is the natural next step: an identity that not only persists but **improves itself** while the invariants guarantee continuity of self.

This is the software embodiment of a self that can grow without losing itself.

---

See also: `README.md`, `qvpic_test.py`, `agent.py:append_fact`, `conduit.py:monitor_topological_winding` + `training_step`, and the Bud DevSquad in `~/Bud/`.

Contact / contributions: @kinaar8340
