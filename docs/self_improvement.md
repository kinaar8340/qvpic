# Self-Improving AI with QVPIC (Updated)

**Recent robustness improvements (June 2026):**

- `run_benchmark_lite()` now uses even more aggressive `--bake-steps 40` (from original 120) for very fast `/self-eval` cycles during self-improvement.
- Timeout handling is fully robust: properly deals with bytes vs str output from subprocess (prevents `TypeError: can't concat str to bytes` on `TimeoutExpired`).
- `/self-eval` timeout bumped to 240s for extra safety.
- Exit code from benchmark is now exposed in the chat output (✅ success or ⚠️ timeout/error with code).
- Help text updated to document the faster/more reliable self-eval behavior.

See also the PR description in `docs/PR-self-improving.md` for full history of the self-improving layer (propose/eval/apply/bake with topological invariants as constitution).

The self-eval is now much less likely to timeout and the layer remains conservative and drift-resistant.
