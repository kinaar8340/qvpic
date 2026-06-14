# Self-Improving AI with QVPIC (Updated)

**Recent robustness improvements (June 2026):**

- `run_benchmark_lite()` now uses even more aggressive `--bake-steps 40` (from original 120) for very fast `/self-eval` cycles during self-improvement.
- Timeout handling is fully robust: properly deals with bytes vs str output from subprocess (prevents `TypeError: can't concat str to bytes` on `TimeoutExpired`).
- `/self-eval` timeout bumped to 240s for extra safety.
- Exit code from benchmark is now exposed in the chat output (✅ success or ⚠️ timeout/error with code).
- Help text updated to document the faster/more reliable self-eval behavior.

See also the PR description in `docs/PR-self-improving.md` for full history of the self-improving layer (propose/eval/apply/bake with topological invariants as constitution).

The self-eval is now much less likely to timeout and the layer remains conservative and drift-resistant.

### Background Execution for /self-eval (Updated June 2026)
To keep the chat responsive, `/self-eval` runs the (ultra-light 20-step) benchmark in a background thread using `ThreadPoolExecutor`.

**New behavior:**
- `/self-eval` immediately returns a "started in background" message (non-blocking). Any previous unviewed result is cleared.
- The result is **persisted** in `self_eval_result` even after the `Future` is cleared.
- **Auto-notification**: On the *next* user message (any non-empty input in `chat_fn`), if a stored result exists, a clean summary is automatically appended to the assistant reply, and the result is cleared.
- `/self-eval-status` now has three clear cases:
  - Still running → "⏳ Benchmark is still running..."
  - Just completed → Shows status + key metrics, stores result, clears future.
  - No active task but previous result pending → Shows "(previous result)" + key metrics, then clears the stored result.
- If an error occurs retrieving the background task, it is reported and the future is cleared.

Key metrics shown (in both status and auto-notify): status (with exit code), duration, fidelity, drift protection. No raw JSON dumps for readability.

**Benchmark speed:** The lightweight path now uses only 5 bake-steps (via `lightweight=True` in run_benchmark_lite) to ensure /self-eval completes quickly (typically under 30s, well under the 300s timeout) and produces useful fidelity/drift numbers on the next message.

This keeps the original propose/eval/apply + topological guardrails intact while making long-running self-evals practical in an interactive chat.

See also `docs/PR-self-improving.md` for the full feature history.
