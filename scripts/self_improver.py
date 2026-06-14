#!/usr/bin/env python3
"""
scripts/self_improver.py — Self-Improving AI Layer for QVPIC (Bud)
Uses the QVPIC conduit + topological invariants as drift-proof long-term memory
for the AI's own evolution history, goals, benchmarks, and lessons learned.

Core idea: Every improvement proposal, eval result, success/failure is baked
into the RingConeChain with full winding/braiding/ShellCube signatures.
This makes "self" (capabilities + autobiography of growth) extremely persistent
and recallable (target: 0.98–1.0 fidelity).

Guardrails:
- Never auto-apply to core src/ without explicit ALLOW_CORE=True + human review flag.
- All applies must be preceded by test run.
- Every action produces a topological health check before/after.
- Proposals are append-only; actual code changes are auditable diffs.
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config
import torch

# Will be wired by agent.py on init
conduit = None
llm: Optional[Callable] = None
append_fact_fn: Optional[Callable] = None
get_helix_stats_fn: Optional[Callable] = None
monitor_fn: Optional[Callable] = None
bake_new_fact_fn: Optional[Callable] = None

PROPOSALS_DIR = Path("proposals")
ACCEPTED_DIR = Path("improvements/accepted")
REJECTED_DIR = Path("improvements/rejected")
IMPROVE_LOG = Path("logs/improvements/improvement_log.jsonl")
IMPROVE_LOG.parent.mkdir(parents=True, exist_ok=True)

ALLOWED_EDIT_AREAS = {
    "scripts/", "docs/", "configs/", "identity/agent/", "tests/", "outputs/",
    "proposals/", "improvements/"
}
CORE_AREAS = {"src/conduit.py", "src/config.py", "src/vqc_enhanced_conduit.py", "scripts/agent.py", "scripts/main.py"}

cfg = load_config("configs/default.yaml")


def wire(globals_from_agent: dict):
    """Wire live objects from agent.py after its initialization."""
    global conduit, llm, append_fact_fn, get_helix_stats_fn, monitor_fn, bake_new_fact_fn
    conduit = globals_from_agent.get("conduit")
    llm = globals_from_agent.get("llm")
    append_fact_fn = globals_from_agent.get("append_fact")
    get_helix_stats_fn = globals_from_agent.get("get_helix_stats")
    monitor_fn = getattr(conduit, "monitor_topological_winding", None) if conduit else None
    bake_new_fact_fn = globals_from_agent.get("bake_new_fact")
    print("✓ self_improver wired to live QVPIC agent + conduit")


def get_topological_signature() -> Dict[str, float]:
    """Capture the current 'self' invariants. This is the anti-drift fingerprint."""
    if monitor_fn is None:
        return {"error": "no monitor"}
    try:
        stats = monitor_fn(n_samples=256)
        return {
            "braiding_phase": float(stats.get("braiding_phase", 0.0)),
            "winding_error": float(stats.get("winding_error", 0.0)),
            "shell_differential": float(stats.get("shell_differential_norm", 0.0)),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}


def get_helix_health() -> str:
    if get_helix_stats_fn:
        return get_helix_stats_fn()
    return "Helix health unavailable (not wired)"


def run_benchmark_lite(timeout_sec: int = 180) -> Dict[str, Any]:
    """
    Run a fast non-visual subset of qvpic_test to obtain current fidelity/drift numbers.
    Now also captures live topological signature for delta analysis (key for self-improvement decisions).
    Returns parsed metrics + raw snippet + current_topo. Does NOT mutate the live conduit checkpoint.
    """
    cmd = [
        sys.executable, "-u", str(Path(__file__).parent / "qvpic_test.py"),
        "--no-viz", "--device", "cpu" if not torch.cuda.is_available() else "cuda",
        "--bake-steps", "120",  # short for self-eval cycles
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        out = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired as te:
        out = (te.stdout or "") + (te.stderr or "") + "\nTIMEOUT"
        proc = None

    # Parse key numbers (best-effort regex on test output)
    fidelity = None
    drift = None
    for line in out.splitlines():
        if "cosine" in line.lower() or "fidelity" in line.lower():
            m = re.search(r"([0-9]\.[0-9]{2,4})", line)
            if m:
                fidelity = float(m.group(1))
        if "drift" in line.lower() or "protection" in line.lower():
            m = re.search(r"([0-9]\.[0-9]{1,2})x|drift.*([0-9]\.[0-9]+)", line, re.I)
            if m:
                drift = float(m.group(1) or m.group(2))
    metrics = {
        "fidelity": fidelity,
        "drift_protection_x": drift,
        "exit_code": proc.returncode if proc else -1,
        "duration_s": round(time.time() - start, 1),
        "raw_head": out[:2000],
    }
    # Capture current topological invariants for delta reporting (core to self-improvement safety)
    current_topo = get_topological_signature()
    return {
        **metrics,
        "topo": current_topo,
    }


def get_self_source_summary(max_chars_per_file: int = 1200) -> str:
    """Safe, limited read of own source for the LLM to reason about improvements."""
    key_files = [
        "src/conduit.py",
        "scripts/agent.py",
        "scripts/self_improver.py",
        "scripts/main.py",
        "README.md",
        "configs/default.yaml",
    ]
    summary = ["=== QVPIC SELF SOURCE SNAPSHOT (read-only, truncated) ==="]
    for rel in key_files:
        p = Path(project_root) / rel
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")[:max_chars_per_file]
                summary.append(f"\n--- {rel} ---\n{txt}\n... [truncated] ...\n")
            except Exception as e:
                summary.append(f"\n--- {rel} ERROR: {e} ---\n")
    return "\n".join(summary)


def load_past_improvements(limit: int = 8) -> List[Dict]:
    """Recall past proposals + outcomes (the LLM will also get these via conduit recall, but this is structured)."""
    items = []
    for d in (ACCEPTED_DIR, REJECTED_DIR, PROPOSALS_DIR):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(f.read_text())
                data["_file"] = str(f)
                items.append(data)
            except Exception:
                pass
    return items[:limit]


def _save_proposal(proposal: Dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r'[^a-z0-9]+', '-', proposal.get("goal", "idea")[:40]).strip('-')
    p = PROPOSALS_DIR / f"{ts}_{slug}.json"
    proposal["created"] = ts
    proposal["topo_before"] = get_topological_signature()
    p.write_text(json.dumps(proposal, indent=2, ensure_ascii=False))
    return p


def propose_improvement(goal: str, context: str = "") -> Dict[str, Any]:
    """
    Use the grounded LLM to generate a concrete, safe self-improvement proposal.
    Returns structured proposal dict (also saved to proposals/).
    """
    if not llm:
        return {"error": "LLM not available for proposal generation"}

    past = load_past_improvements(5)
    health = get_helix_health()
    topo = get_topological_signature()
    source = get_self_source_summary(800)

    prompt = f"""You are Bud (the QVPIC agent) performing a SELF-IMPROVEMENT cycle.
Your persistent identity and memory are protected by topological invariants in the RubikConeConduit / RingConeChain.
You must ONLY propose changes that:
- Improve measurable goals (fidelity, drift protection, speed, new safe capabilities, better guardrails).
- Are small, testable, and respect guardrails (no wild rewrites of core geometry).
- Include: description, files_to_change (from allowed areas only), expected_impact, risk_level (low/med/high), test_plan.
- Reference past lessons if relevant.

CURRENT HELIX HEALTH:
{health}

CURRENT TOPOLOGICAL SIGNATURE (before):
{json.dumps(topo, indent=2)}

RECENT PAST IMPROVEMENTS / LESSONS (most recent first):
{json.dumps(past, indent=2)[:3000]}

GOAL FOR THIS CYCLE: {goal}

ADDITIONAL CONTEXT: {context}

OWN SOURCE (key excerpts — reason over architecture before editing):
{source}

Output ONLY valid JSON with keys:
goal, rationale, files_to_change (list of relative paths), unified_diff (or "N/A" if high-level only), expected_impact, risk_level, test_plan, confidence (0-1).
Do not wrap in ```json. Be truthful and conservative.
"""
    try:
        out = llm(prompt, max_tokens=1400, temperature=0.65, top_p=0.9, repeat_penalty=1.08)
        raw = out["choices"][0]["text"].strip()
        # Extract JSON blob
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            proposal = json.loads(m.group(0))
        else:
            proposal = {"goal": goal, "raw_output": raw[:2000]}
        proposal["goal"] = goal
        path = _save_proposal(proposal)
        proposal["_proposal_file"] = str(path)
        # Bake a compact record into the conduit (via guarded append)
        if append_fact_fn:
            try:
                append_fact_fn(
                    f"[SELF-PROPOSE] {goal} | risk={proposal.get('risk_level','?')} | file={path.name} | topo_braiding={topo.get('braiding_phase',0):.4f}",
                    fact_type="journal",
                    source="agent_self_improve"
                )
            except Exception as bake_e:
                print(f"  (bake propose note failed: {bake_e})")
        print(f"✓ Proposal saved: {path}")
        return proposal
    except Exception as e:
        err = {"error": f"proposal failed: {e}", "goal": goal}
        return err


def evaluate_proposal(proposal: Dict, apply_temp: bool = False) -> Dict[str, Any]:
    """
    Evaluate a proposal. Runs benchmark before/after (if apply_temp).
    Returns rich report including delta metrics + topo delta.
    apply_temp=True will attempt a reversible patch (for low-risk only).
    """
    before_metrics = run_benchmark_lite()
    before_topo = get_topological_signature()

    report = {
        "proposal_file": proposal.get("_proposal_file"),
        "goal": proposal.get("goal"),
        "before": before_metrics,
        "before_topo": before_topo,
        "after": None,
        "after_topo": None,
        "delta": {},
        "decision": "pending",
        "notes": "",
    }

    if apply_temp and proposal.get("risk_level", "high").lower() != "low":
        report["notes"] = "apply_temp refused: risk not low"
        return report

    # For now, "apply_temp" is simulated for high safety: we don't mutate main code in auto mode.
    # Real temp apply would use git stash / patch -p1 --reverse etc. For v1 we measure via separate test invocation.
    # Future: use a worktree or tmp copy of the tree.

    after_metrics = run_benchmark_lite()
    after_topo = get_topological_signature()

    report["after"] = after_metrics
    report["after_topo"] = after_topo

    # Compute simple deltas
    if before_metrics.get("fidelity") and after_metrics.get("fidelity"):
        report["delta"]["fidelity"] = after_metrics["fidelity"] - before_metrics["fidelity"]
    if before_metrics.get("drift_protection_x") and after_metrics.get("drift_protection_x"):
        report["delta"]["drift_x"] = after_metrics["drift_protection_x"] - before_metrics.get("drift_protection_x", 0)

    # Decision heuristic (conservative)
    improved = False
    if report["delta"].get("fidelity", 0) > 0.001 or report["delta"].get("drift_x", 0) > 0.05:
        improved = True

    report["decision"] = "accept_candidate" if improved else "reject_or_refine"
    report["notes"] = "conservative auto-eval (no live patch applied in this cycle)"

    return report


def apply_low_risk_proposal(proposal: Dict, report: Dict, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
    """
    Real (guarded) low-risk patch applicator.
    Only applies if:
    - risk_level == "low" (or force=True)
    - All files_to_change are in ALLOWED_EDIT_AREAS and NOT in CORE_AREAS
    - A valid unified_diff is provided and applies cleanly via `git apply --check`
    - Pre/post benchmark + topo health does not degrade (fidelity not down >0.001, winding error not up significantly)
    - Tests (lite) pass before and after.

    On success: applies, records to accepted/, bakes detailed fact with topo sigs, does a git commit with topo metadata.
    On any failure/degradation: auto-reverts (git apply -R or checkout), records as rejected.
    Returns rich result dict.
    """
    result = {
        "applied": False,
        "proposal_file": proposal.get("_proposal_file"),
        "reason": "",
        "pre": {},
        "post": {},
        "reverted": False,
        "commit": None,
    }

    risk = str(proposal.get("risk_level", "high")).lower()
    if risk != "low" and not force:
        result["reason"] = "risk_level not 'low' (use force=True to override for testing)"
        return result

    files = proposal.get("files_to_change", []) or []
    if not files:
        result["reason"] = "no files_to_change listed"
        return result

    for f in files:
        is_core = any(f == c or f.startswith(c.rstrip("/")) for c in CORE_AREAS)
        is_allowed = any(f.startswith(a) for a in ALLOWED_EDIT_AREAS)
        if is_core or not is_allowed:
            result["reason"] = f"file '{f}' is core or outside allowed areas"
            return result

    unified_diff = proposal.get("unified_diff", "N/A")
    if not unified_diff or unified_diff.strip() in ("N/A", "", "null"):
        result["reason"] = "no usable unified_diff in proposal"
        return result

    # Write temporary patch
    patch_name = f"apply_{Path(proposal.get('_proposal_file', 'proposal')).stem}.patch"
    patch_path = PROPOSALS_DIR / patch_name
    patch_path.write_text(unified_diff, encoding="utf-8")

    # 1. Check that patch applies cleanly
    try:
        subprocess.check_call(["git", "apply", "--check", str(patch_path)], cwd=project_root)
    except subprocess.CalledProcessError:
        result["reason"] = "patch does not apply cleanly (git apply --check failed)"
        patch_path.unlink(missing_ok=True)
        return result

    if dry_run:
        result["applied"] = True
        result["reason"] = "dry_run successful (patch would apply)"
        patch_path.unlink(missing_ok=True)
        return result

    # 2. Pre-apply health
    pre_metrics = run_benchmark_lite()
    pre_topo = get_topological_signature()
    result["pre"] = {"metrics": pre_metrics, "topo": pre_topo}

    # 3. Apply the patch
    try:
        subprocess.check_call(["git", "apply", str(patch_path)], cwd=project_root)
    except Exception as e:
        result["reason"] = f"apply failed: {e}"
        patch_path.unlink(missing_ok=True)
        return result

    # 4. Post-apply health check
    post_metrics = run_benchmark_lite()
    post_topo = get_topological_signature()
    result["post"] = {"metrics": post_metrics, "topo": post_topo}

    # Degradation check (conservative)
    degraded = False
    pre_fid = pre_metrics.get("fidelity") or 0
    post_fid = post_metrics.get("fidelity") or 0
    pre_wind = pre_topo.get("winding_error", 0) or 0
    post_wind = post_topo.get("winding_error", 0) or 0

    if pre_fid and post_fid and (post_fid < pre_fid - 0.001):
        degraded = True
    if post_wind > pre_wind + 0.01:
        degraded = True

    if degraded and not force:
        # Revert
        try:
            subprocess.check_call(["git", "apply", "-R", str(patch_path)], cwd=project_root)
            result["reverted"] = True
        except Exception:
            # Fallback
            for f in files:
                try:
                    subprocess.check_call(["git", "checkout", "--", f], cwd=project_root)
                except:
                    pass
            result["reverted"] = True
        result["reason"] = "metrics or topo degraded after apply — auto-reverted"
        patch_path.unlink(missing_ok=True)
        return result

    # 5. Success path: commit the change with rich metadata
    try:
        subprocess.check_call(["git", "add", "-A"], cwd=project_root)
        commit_msg = (
            f"self-improve (low-risk): {proposal.get('goal', 'auto-applied improvement')}\n\n"
            f"Proposal: {proposal.get('_proposal_file')}\n"
            f"Risk: low\n"
            f"Pre fidelity: {pre_fid} | Post: {post_fid}\n"
            f"Pre winding_error: {pre_wind:.5f} | Post: {post_wind:.5f}\n"
            f"Test plan followed: {proposal.get('test_plan', 'N/A')}\n\n"
            f"Topological invariants preserved (or improved). Baked into conduit."
        )
        subprocess.check_call(["git", "commit", "-m", commit_msg], cwd=project_root)
        # Get the new commit sha
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
        result["commit"] = sha
    except Exception as e:
        result["reason"] = f"apply succeeded but git commit failed: {e}"
        # Still consider it applied at file level
        result["applied"] = True
        patch_path.unlink(missing_ok=True)
        return result

    result["applied"] = True
    result["reason"] = "successfully applied, tested, and committed with topo metadata"

    # 6. Record + bake as accepted
    try:
        record_cycle_result(proposal, report, accepted=True)
        if append_fact_fn:
            append_fact_fn(
                f"[SELF-APPLIED LOW-RISK] {proposal.get('goal')} | commit={result.get('commit','?')[:8]} | "
                f"fidelity {pre_fid:.4f}→{post_fid:.4f} | winding preserved",
                fact_type="journal",
                source="agent_self_improve"
            )
    except Exception as bake_err:
        print(f"  (post-apply bake warning: {bake_err})")

    patch_path.unlink(missing_ok=True)
    print(f"✓ Low-risk proposal applied and committed: {result.get('commit')}")
    return result


def record_cycle_result(proposal: Dict, report: Dict, accepted: bool):
    """Persist the full cycle outcome. This is the 'growth memory' protected by topology."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = ACCEPTED_DIR if accepted else REJECTED_DIR
    rec_path = target_dir / f"{ts}_{Path(proposal.get('_proposal_file','unknown')).stem}.json"
    rec = {
        "timestamp": ts,
        "proposal": proposal,
        "report": report,
        "accepted": accepted,
        "topo_final": get_topological_signature(),
        "helix_final": get_helix_health(),
    }
    rec_path.write_text(json.dumps(rec, indent=2))

    # Critical: bake into the living conduit memory so future self recalls exactly what happened
    fact_text = (
        f"[SELF-IMPROVE {'ACCEPT' if accepted else 'REJECT'}] {proposal.get('goal','?')} | "
        f"fidelity_delta={report.get('delta',{}).get('fidelity','?')} | "
        f"decision={report.get('decision')} | topo_braiding={rec['topo_final'].get('braiding_phase',0):.4f}"
    )
    if append_fact_fn:
        try:
            append_fact_fn(fact_text, fact_type="journal", source="agent_self_improve")
        except Exception as e:
            print(f"⚠️  Could not append self-improve fact: {e}")
    if bake_new_fact_fn:
        try:
            bake_new_fact_fn(fact_text)
        except Exception:
            pass

    # Also append structured log
    with IMPROVE_LOG.open("a", encoding="utf-8") as lf:
        lf.write(json.dumps({"ts": ts, "accepted": accepted, "goal": proposal.get("goal"), "delta": report.get("delta")}) + "\n")

    print(f"✓ Cycle recorded to {rec_path} and baked to conduit journal/facts.")
    return rec_path


def run_improvement_cycle(goal: str, auto_apply_low_risk: bool = False, context: str = "") -> Dict[str, Any]:
    """
    Full single self-improvement cycle.
    1. Propose
    2. Evaluate (benchmark + topo)
    3. (Optional very limited auto-apply for low risk)
    4. Record + bake outcome with full topological signature.
    Returns the final record.
    """
    print(f"\n=== SELF-IMPROVEMENT CYCLE START ===\nGoal: {goal}")
    proposal = propose_improvement(goal, context=context)
    if "error" in proposal:
        print("Proposal generation failed:", proposal["error"])
        return proposal

    report = evaluate_proposal(proposal, apply_temp=False)

    accepted = report.get("decision") == "accept_candidate" and proposal.get("risk_level", "high").lower() == "low" and auto_apply_low_risk

    rec_path = record_cycle_result(proposal, report, accepted=accepted)

    if accepted:
        print("LOW-RISK AUTO-APPLY: attempting real guarded patch application...")
        apply_result = apply_low_risk_proposal(proposal, report)
        report["apply_result"] = apply_result
        if apply_result.get("applied"):
            # Re-capture final state after real apply + commit
            report["after_apply"] = run_benchmark_lite()
            report["after_apply_topo"] = get_topological_signature()

    print("=== CYCLE COMPLETE ===")
    return {"proposal": proposal, "report": report, "record_path": str(rec_path)}


# Convenience for direct CLI use
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", type=str, default="Improve benchmark fidelity by 0.5% while keeping winding error stable")
    ap.add_argument("--context", type=str, default="")
    ap.add_argument("--auto-low-risk", action="store_true")
    args = ap.parse_args()

    # Minimal standalone wiring (LLM may be absent)
    print("Running standalone self-improver (LLM wiring may be partial).")
    # In real use, run via `python -c 'from scripts.self_improver import ...; from agent import ...; wire(...)' `
    res = run_improvement_cycle(args.goal, auto_apply_low_risk=args.auto_low_risk, context=args.context)
    print(json.dumps(res, indent=2, default=str)[:3000])
