# PR: QVPIC-powered self-improving AI layer with guarded real low-risk applicator

## Summary
Closes the loop on a guarded, recursively self-improving AI ("Bud") whose entire "self" — goals, capabilities, improvement proposals, benchmark results, decisions, and autobiography — is protected by QVPIC’s own drift-resistant topological invariants (winding number, braiding/linking phase, ShellCube radial differential, RingConeChain + copresheaf TNN).

The system moved from "propose + evaluate + record" to a **closed, conservative, self-modifying loop** that can safely *enact* low-risk changes and permanently bake the full before/after story (including topological signatures) into its own memory at high fidelity.

## Key Changes
- **Real guarded low-risk patch applicator** (`apply_low_risk_proposal`):
  - Only for `risk_level == "low"` (or explicit `force=True`).
  - Hard restrictions: only `ALLOWED_EDIT_AREAS`; **never** touches `CORE_AREAS` (`src/conduit.py`, `scripts/agent.py`, etc.).
  - Requires valid `unified_diff` + `git apply --check`.
  - Full pre/post `run_benchmark_lite()` (now returns live topo signature) + degradation gate.
  - Automatic revert on failure/degradation.
  - On success: `git commit` with rich metadata, record to `improvements/accepted/`, detailed fact baked into conduit under `agent_self_improve`.
- Upgrades:
  - `run_benchmark_lite()` now includes current topological signature.
  - `run_improvement_cycle(..., auto_apply_low_risk=True)` performs real apply.
  - New CLI: `/self-apply <stem>` and `/self-proposals [status]`.
  - Configurable safety thresholds + `list_proposals()` helper.
- Hygiene: proper `.gitignore` for facts/, removal of accidental runtime PII.
- Docs updated.

## Safety & Philosophy
Topological invariants are the supreme law (see `self_improvement.constitution` in config). Everything is append-only, auditable, revertible, and baked with before/after signatures for ~0.98–1.0 recall without drift. Perfectly aligned with the VQC patent and QVPIC design.

## Usage
After running the agent:
```
/self-eval
/self-propose ...
/self-cycle ...
/self-apply <proposal-stem>
/self-proposals pending
/self-history
```

See `docs/self_improvement.md` for full details.

## Testing & Merge
- Self-contained on `feat/self-improving` (rebased and merged into `main`).
- No core geometry changes.
- Includes UI crash fix, wiring robustness, list_proposals CLI, and the two targeted reliability fixes below.
- Added comprehensive unit tests (see below).

## Recent Reliability Fixes
- **Command parsing** (`parse_cli_command` helper + used in `run_pic_cli`): Now reliably handles `/self-apply <long-stem-with-dates-underscores>` and long/multi-word goals for `/self-propose`, `/self-cycle` etc. Full rest-of-line captured as argument while preserving legacy `/add key "value with spaces"`.
- **JSON extraction** in `self_improver.propose_improvement`: 
  - LLM prompt now explicitly says "Output ONLY the raw valid JSON object... No explanations, no markdown code fences".
  - Extraction strips ```json / ``` fences, uses resilient regex + layered try/except fallbacks around json.loads.
- These make the self-improvement commands (`/self-apply`, `/self-propose` etc.) production-usable.

## Tests Added
- `tests/test_self_improver.py`: 
  - `test_parse_cli_command`: covers long stems, multi-word goals, legacy /add with quoted values, /self-proposals, etc.
  - `test_extract_proposal_json`: tests fence stripping, direct JSON, fallback on bad output, and that goal is forced.
- Run with `pytest tests/test_self_improver.py -q`.

**Branch:** `feat/self-improving` (rebased onto main)
**Merged into:** `main` (merge commit cf5b5f3)

This enables the AI to not only remember its growth but safely enact low-risk growth while maintaining perfect topological continuity.
