# Self-Improving AI with QVPIC (Updated)

**Recent robustness improvements (June 2026):**

- `run_benchmark_lite()` now uses even more aggressive `--bake-steps 40` (from original 120) for very fast `/self-eval` cycles during self-improvement.
- Timeout handling is fully robust: properly deals with bytes vs str output from subprocess (prevents `TypeError: can't concat str to bytes` on `TimeoutExpired`).
- `/self-eval` timeout bumped to 240s for extra safety.
- Exit code from benchmark is now exposed in the chat output (✅ success or ⚠️ timeout/error with code).
- Help text updated to document the faster/more reliable self-eval behavior.

See also the PR description in `docs/PR-self-improving.md` for full history of the self-improving layer (propose/eval/apply/bake with topological invariants as constitution).

The self-eval is now much less likely to timeout and the layer remains conservative and drift-resistant.

### Background Execution for /self-eval (Added June 2026)
To keep the chat responsive, `/self-eval` now runs the benchmark in a background thread (using ThreadPoolExecutor).

- `/self-eval` immediately returns "🚀 Self-evaluation started in background..." (does not block).
- Use `/self-eval-status` to poll for completion and view results (including duration, metrics, and exit code).
- If another eval is already running, it will tell you to wait.
- Once checked via status, the future is cleared.

This is a simple, non-blocking addition that doesn't change the core propose/eval/apply architecture. The topological invariants and guardrails remain fully enforced.

**Further improvements (latest):**
- Ultra-lightweight mode for /self-eval: only 20 bake-steps (dedicated fast path in run_benchmark_lite via `lightweight=True`).
- Timeout increased to 300s for higher completion chance on slower hardware.
- **Auto-notification**: After background task completes, the next *any* user message (in chat_fn) will automatically append a clean result summary to the assistant reply (and clear the pending task).
- **Cleaner status**: /self-eval-status (and auto-notify) now shows only key metrics (status, duration, fidelity, drift protection) in readable text instead of raw JSON dump. Full details still available via the result object if needed.
