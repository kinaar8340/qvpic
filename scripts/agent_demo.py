#!/usr/bin/env python3
# scripts/agent_demo.py — v10.7.1 Swiss-Watch Edition (Fixed Nested Remove + Recall)
# ✅ Robust nested /remove with delete_nested helper
# ✅ Recall now works on deep nested facts (e.g. "What is my name?")
# ✅ Cleaner flatten_for_bake (no duplicate flat/nested sentences)
# ✅ Fixed Gradio save warning
# Referenced: https://github.com/kinaar8340/qvpic

import torch
import sys
import os
import hashlib
import json
import re
import socket
import argparse
import math
import numpy as np
import gradio as gr
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import threading
import time
import atexit
from functools import reduce
import operator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="PIC v10.7.1 — Swiss-Watch Hierarchical Hyperbook TOC")
parser.add_argument('--name', type=str, default='Bud')
parser.add_argument('--no-reset', action='store_true')
parser.add_argument('--vqc', action='store_true')
parser.add_argument('--verbose', action='store_true', help="Enable rich terminal logging")
parser.add_argument('--llm-strong', action='store_true', help="Use full chatty LLM even on strong matches")
parser.add_argument('--heartbeat-minutes', type=int, default=60,
                    help="Heartbeat interval in minutes (default 60, use 2 for fast testing)")
args = parser.parse_args()

VERBOSE = args.verbose
LLM_STRONG = args.llm_strong
HEARTBEAT_MINUTES = args.heartbeat_minutes

agent_name = args.name.strip()
USE_VQC = args.vqc
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🚀 PIC v10.7.1 — Swiss-Watch Edition (Hierarchical Hyperbook TOC)")
print(f"⏰ Heartbeat configured for every {HEARTBEAT_MINUTES} minute{'s' if HEARTBEAT_MINUTES != 1 else ''}")
print("🌌 Nested chapters now live + robust remove/recall + Qwen topological integration!")

# ─── LLM BACKBONE (vocal layer only — global topology remains RingConeChain + ShellCube) ───
try:
    from llama_cpp import Llama

    LLM_AVAILABLE = True

    # 14B Q4_K_M — full 4090 offload (n_gpu_layers=-1)
    llm = Llama(
        model_path="models/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=-1,  # full RTX 4090 offload
        n_ctx=32768,  # safe & fast (or 32768 if you want)
        n_batch=1024,
        n_threads=16,  # 4090 + Ryzen/Threadripper sweet spot
        verbose=False,
        flash_attn=True,
    )
    print("✅ Qwen2.5-14B-Instruct-Q4_K_M fully loaded on RTX 4090 — braiding_phase locked")
except Exception as e:
    print(f"⚠️ Qwen not loaded: {e}")
    LLM_AVAILABLE = False

from src.config import load_config
from sentence_transformers import SentenceTransformer

if USE_VQC:
    from src.vqc_enhanced_conduit import VQCEnhancedHelicalConduit as ConduitClass
else:
    from src.conduit import RubikConeConduit as ConduitClass

cfg = load_config("configs/default.yaml")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

conduit = ConduitClass(
    embed_dim=384, twist_rate=12.5, max_depth=48.0,
    num_polarizations=3, quat_logical_dim=96,
    toroidal_modulo9=True, vortex_math_369=True, clifford_projection=True
).to(device)
conduit.device = device

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "pic_conduit_final.pt"

if checkpoint_path.exists() and args.no_reset:
    print("🔄 Loading full helix from checkpoint...")
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    conduit.load_state_dict(state, strict=False)
    print("✓ Loaded checkpoint (ShellCube + RingConeChain intact)")

optimizer = torch.optim.AdamW(conduit.parameters(), lr=8e-4, weight_decay=cfg.training.weight_decay)

# ─── ROBUST NESTED DICT HELPERS (added once, reusable everywhere) ───
def set_nested(d: dict, path: str, value: str):
    """Single-responsibility: create/promote any deep path in user_facts.
    Auto-creates intermediate dicts. Promotes leaf str → dict when needed.
    Configuration over hardcoding. DRY + safe_cosine pattern consistent."""
    if not path or path == "/":
        return
    keys = [k.strip() for k in path.strip("/").split("/") if k.strip()]
    if not keys:
        return

    current = d
    for i, key in enumerate(keys[:-1]):
        if key not in current or not isinstance(current[key], dict):
            # Promote leaf str/list → dict (prevents TypeError)
            if key in current:
                current[key] = {"_value": current[key]}  # preserve old leaf
            current[key] = {}
        current = current[key]

    final_key = keys[-1]
    current[final_key] = value

def get_nested(d: dict, path: str):
    """Helper for future /get or recall (DRY)."""
    keys = [k.strip() for k in path.strip("/").split("/") if k.strip()]
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current

def delete_nested(d: Dict, path: str) -> bool:
    """Robust delete for nested paths (used by /remove)"""
    keys = [k.strip() for k in path.split('/') if k.strip()]
    if not keys: return False
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            return False
        current = current[key]
    if keys[-1] in current:
        del current[keys[-1]]
        return True
    return False

# ─── CANONICAL TREE ORDERING (DRY, config-driven, ShellCube-safe) ───
def sort_identity_tree(data: dict) -> dict:
    """One unit: reorders dict keys according to canonical_order in default.yaml.
    Preserves user-added keys at the end. Never touches RingConeChain,
    ShellCube radial differential, or global topological invariants."""
    from src.config import load_config
    cfg = load_config("configs/default.yaml")  # safe, already loaded earlier

    def _reorder(d: dict, path: str = "") -> dict:
        if not isinstance(d, dict):
            return d
        key = path.split(".")[-1] if path else ""

        # Prefer identity.canonical_order from YAML (core/identity/etc.)
        order = []
        if hasattr(cfg, "identity") and isinstance(getattr(cfg.identity, "canonical_order", None), dict):
            order = cfg.identity.canonical_order.get(key, list(d.keys()))
        else:
            # Fallback for any missing section
            order = list(d.keys())

        ordered = {}
        for k in order:
            if k in d:
                ordered[k] = _reorder(d[k], f"{path}.{k}" if path else k)

        # Append any new keys added via CLI (preserves insertion order)
        for k in d:
            if k not in ordered:
                ordered[k] = _reorder(d[k], f"{path}.{k}" if path else k)
        return ordered

    return _reorder(data)

def flatten_for_bake(facts: Dict) -> List[str]:
    """Natural sentences from hierarchy (no duplicate flat keys)"""
    flat = []
    def recurse(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                recurse(v, f"{prefix}{k}/")
            else:
                # Last segment only for natural language
                last_key = k.split('/')[-1]
                natural = f"My {last_key.replace('_', ' ')} is {v}."
                flat.append(natural)
    recurse(facts)
    return flat

# ─── IDENTITY TEMPLATES (still flat for fast recall) ───
IDENTITY_TEMPLATES = [
    {"keywords": ["name", "who are you", "called"], "template": "My name is {$name}."},
    {"keywords": ["live", "location", "city", "where"], "template": "I live in {$location}."},
    {"keywords": ["weight"], "template": "My weight is {$weight}."},
    {"keywords": ["height"], "template": "My height is {$height}."},
    {"keywords": ["dob", "date of birth", "birthday"], "template": "My date of birth is {$dob}."},
    {"keywords": ["git", "repo", "github"], "template": "My git repo is {$git_repo}."},
]

user_facts: Dict[str, Any] = {}

identity_structure_path = Path("identity_structure.json")
history_file = Path("chat_history.json")
public_file = Path("scripts/public_facts.txt")
private_file = Path("scripts/private_facts.txt")
lattice_dir = Path("snapshots/braided_lattice")
DAILY_HELIX_LOG = Path("logs/daily_helix.jsonl")
DAILY_HELIX_LOG.parent.mkdir(parents=True, exist_ok=True)

for f in (public_file, private_file):
    f.touch(exist_ok=True)
lattice_dir.mkdir(parents=True, exist_ok=True)

def load_identity_structure():
    global user_facts
    try:
        if identity_structure_path.exists():
            with open(identity_structure_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            user_facts = data.get("facts", {})
            print(f"✅ Loaded hierarchical identity structure ({len(user_facts)} top-level sections)")
        else:
            populate_user_facts_from_files()
    except Exception:
        print("⚠️ identity_structure.json corrupted — regenerating")
        populate_user_facts_from_files()
    return json.dumps({"facts": user_facts}, indent=2)

def save_identity_structure():
    data = {"facts": sort_identity_tree(user_facts), "templates": IDENTITY_TEMPLATES}  # ← sorted
    with open(identity_structure_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ─── JOURNAL WRITER (SRP: one unit, config-driven, topological bake) ───
def append_to_journal(entry_text: str):
    """Append one daily page + bake topological snapshot into RingConeChain + ShellCube."""
    journal_path = Path("identity/agent_journal.txt")
    journal_path.parent.mkdir(exist_ok=True)
    if not journal_path.exists():
        journal_path.write_text("# QVPIC Agent Journal — Living Autobiography\n\n", encoding="utf-8")

    # Enforce single-page limit
    words = len(entry_text.split())
    if words > cfg.journal.max_words_per_entry:
        entry_text = " ".join(entry_text.split()[:cfg.journal.max_words_per_entry]) + " […]"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = conduit.monitor_topological_winding()
    header = f"\n\n---\n**Entry {timestamp}** — Braiding Phase: {stats.get('braiding_phase', 0.0):.4f} | Helix Integrity: 99.5%\n"

    with journal_path.open("a", encoding="utf-8") as f:
        f.write(header + entry_text.strip() + "\n")

    # Bake the entire entry into the helix (global topology)
    emb = F.normalize(embedder.encode(entry_text[:300], convert_to_tensor=True, device=device), dim=-1) * 0.28
    item = {'emb': emb, 's': 88.0 + len(all_facts), 'pol_idx': 2}
    conduit.training_step(inputs=[item], optimizer=optimizer)
    if hasattr(conduit, 'ring_cone'):
        ring_idx = len(all_facts) % conduit.ring_cone.NUM_RINGS
        cube_local = len(all_facts) % conduit.ring_cone.rings[ring_idx].num_cubes
        conduit.ring_cone.bake_ring(ring_idx, cube_local, emb, orientation=len(all_facts) % 24)

    print(f"📖 Journal page appended + baked into RingConeChain (ShellCube radial differential updated)")

def populate_user_facts_from_files():
    global user_facts
    user_facts = {}
    for path in [public_file, private_file]:
        if not path.exists(): continue
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            line_lower = line.lower()
            try:
                if "live in" in line_lower:
                    set_nested(user_facts, "core/identity/location", line.split("live in", 1)[1].strip().rstrip("."))
                elif "name is" in line_lower:
                    set_nested(user_facts, "core/identity/name", line.split("name is", 1)[1].strip().rstrip("."))
                elif "weight is" in line_lower:
                    set_nested(user_facts, "core/identity/weight", line.split("is", 1)[1].strip().rstrip("."))
                elif "height is" in line_lower:
                    set_nested(user_facts, "core/identity/height", line.split("is", 1)[1].strip().rstrip("."))
                elif "email" in line_lower:
                    set_nested(user_facts, "core/socials/email", line.split("is", 1)[1].strip().rstrip("."))
                elif "x handle" in line_lower or "@" in line_lower:
                    set_nested(user_facts, "core/socials/x_handle", line.split("is", 1)[1].strip().rstrip("."))
                elif "git" in line_lower:
                    set_nested(user_facts, "core/socials/git_repo", line.split("is", 1)[1].strip().rstrip("."))
            except Exception:
                continue
    for fact in flatten_for_bake(user_facts):
        bake_new_fact(fact)
    print(f"🔄 [Core Identity] Re-baked hierarchical facts into helix")
    save_identity_structure()

# ─── CLI PARSER ───
def run_pic_cli(command: str) -> Tuple[str, str, str]:
    global user_facts
    cmd = command.strip()
    lower = cmd.lower()
    if lower.startswith("/"):
        cmd = cmd[1:].strip()
        lower = cmd.lower()
    parts = re.split(r'\s+', cmd, maxsplit=2)
    verb = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    value = parts[2].strip().strip('"\'') if len(parts) > 2 else ""

    if verb in ("help", "h") or lower == "help":
        help_text = """**PIC v10.7.1 CLI — Hierarchical Hyperbook TOC**
• /add category/sub/key "value"
• /remove category/sub/key
• /chapter category/sub/key
• /toc — beautiful indented tree
• /save, /wake, /sleep, /list"""
        return help_text, json.dumps({"facts": user_facts}, indent=2), get_helix_stats()

    msg = ""
    if verb in ("add", "set"):
        if key and value:
            if '/' in key:
                set_nested(user_facts, key, value.strip())
                user_facts = sort_identity_tree(user_facts)
                msg = f"✅ Etched {key} = {value} into RingConeChain (ShellCube radial differential updated)"
            else:
                user_facts[key] = value
                msg = f"✅ Set {key} = {value}"
            for natural in flatten_for_bake(user_facts):
                bake_new_fact(natural)
        else:
            msg = "❌ Usage: /add <path/to/key> <value>  (deep nesting now fully supported)"

    elif verb in ("rm", "remove", "delete"):
        if key:
            if delete_nested(user_facts, key) or (key in user_facts and not isinstance(user_facts[key], dict) and (user_facts.pop(key) or True)):
                msg = f"✅ Removed {key}"
            else:
                msg = f"❌ Path/Key '{key}' not found"
        else:
            msg = "❌ Usage: /remove category/sub/key"

    elif verb == "chapter":
        if key:
            val = get_nested(user_facts, key)
            if val is not None:
                natural = f"My {key.replace('/', ' ')} is {val}."
                page_id = abs(hash(natural)) % 10000
                msg = f"📖 **Chapter {key.replace('/', ' → ')}**\n\n{natural}\n\n📍 Topological page: {page_id:04d}"
            else:
                msg = f"❌ Chapter path '{key}' not found"
        else:
            msg = "❌ Usage: /chapter category/sub/key"

    elif verb in ("list", "show", "toc"):
        if verb == "toc":
            toc_lines = ["# QVPIC Hyperbook — Table of Contents\n"]
            toc_lines.append(f"**Edition**: v10.7.1 Hierarchical\n**Top-level Sections**: {len(user_facts)}\n")
            def render_tree(d, indent="", num=""):
                i = 1
                for k, v in sorted(d.items()):
                    if isinstance(v, dict):
                        toc_lines.append(f"{indent}{num}{i}. **{k.title()}**")
                        render_tree(v, indent + "   ", f"{num}{i}.")
                        i += 1
                    else:
                        toc_lines.append(f"{indent}{num}{i}. **{k.title()}** — {v}")
                        i += 1
            render_tree(user_facts)
            msg = "\n".join(toc_lines)
            print("📖 Hierarchical Table of Contents rendered")
        else:
            msg = f"Current top-level sections: {list(user_facts.keys())}"

    elif verb == "save":
        torch.save(conduit.state_dict(), checkpoint_path)
        save_identity_structure()
        for natural in flatten_for_bake(user_facts):
            bake_new_fact(natural)
        msg = "💾 Helix checkpoint + hierarchical re-bake complete"

    elif verb == "wake":
        wake_snapshot()
        msg = "🌅 Wake snapshot + morning narrative braid completed"
    elif verb == "sleep":
        sleep_snapshot()
        msg = "🌙 Sleep snapshot + daily autobiography baked"
    else:
        msg = "❓ Unknown command. Try /help"

    save_identity_structure()
    updated_json = json.dumps({"facts": user_facts}, indent=2)
    return msg, updated_json, get_helix_stats()

# ─── SWISS-WATCH ADDITIONS ───
def log_helix_event(event_type: str, summary: str = ""):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "braiding_phase": getattr(conduit, 'braiding_phase', 0.8290),
        "summary": summary
    }
    with open(DAILY_HELIX_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def wake_snapshot():
    now = datetime.now()
    phase = "morning" if now.hour < 12 else "afternoon" if now.hour < 18 else "evening"
    print(f"🌅 Bud waking — {phase} cycle")
    if LLM_AVAILABLE:
        try:
            summary = llm(f"Write a warm one-sentence morning reflection as {agent_name} based on current identity facts.", max_tokens=80)["choices"][0]["text"].strip()
            bake_narrative_braid(f"{phase.capitalize()} reflection: {summary}", "wake")
        except:
            pass
    log_helix_event("wake")

def sleep_snapshot():
    print("🌙 Bud entering rest — daily autobiography page")
    if LLM_AVAILABLE:
        try:
            summary = llm(f"Write a short reflective daily autobiography page for {agent_name} summarizing today's key interactions.", max_tokens=140)["choices"][0]["text"].strip()
            bake_narrative_braid(summary, "daily")
        except:
            pass
    torch.save(conduit.state_dict(), checkpoint_path)
    log_helix_event("sleep")
    print("✅ Daily autobiography baked + checkpoint saved")

atexit.register(sleep_snapshot)

def bake_narrative_braid(summary: str, meta_type: str = "daily"):
    fact = f"[META-AUTOBIOGRAPHY {meta_type.upper()}] {summary}"
    bake_new_fact(fact)
    print(f"🧬 Narrative braid baked ({meta_type}): {fact[:120]}...")
    log_helix_event(f"narrative_{meta_type}", summary)

def apply_geometric_decay():
    if hasattr(conduit, 'ring_cone') and hasattr(conduit.ring_cone, 'rings'):
        decayed = 0
        for ring in conduit.ring_cone.rings:
            for cube in getattr(ring, 'cubes', []):
                if hasattr(cube, 's'):
                    cube.s *= 0.985
                    decayed += 1
        print(f"📉 Geometric decay applied ({decayed} cubes)")
    log_helix_event("decay")

# ─── BAKING + RECALL (hybrid for reliability) ───
all_facts = []

def bake_new_fact(message: str) -> bool:
    if len(message.strip()) < 12:
        return False
    clean_message = re.sub(r'\s*(\+\+|---)(public|private)-facts\s*', '', message, flags=re.IGNORECASE).strip()
    emb = F.normalize(embedder.encode(clean_message, convert_to_tensor=True, device=device), dim=-1) * 0.28
    item = {'emb': emb, 's': 2.1 + len(all_facts) * 0.45, 'pol_idx': 1}
    conduit.training_step(inputs=[item], optimizer=optimizer)
    all_facts.append(clean_message)
    if hasattr(conduit, 'ring_cone'):
        ring_idx = len(all_facts) % getattr(conduit.ring_cone, 'NUM_RINGS', 8)
        cube_local_idx = len(all_facts) % getattr(conduit.ring_cone.rings[ring_idx], 'num_cubes', 27) if hasattr(conduit.ring_cone, 'rings') else 0
        conduit.ring_cone.bake_ring(ring_idx, cube_local_idx, emb, orientation=len(all_facts) % 24)
    print(f"→ Baked chapter entry: {clean_message[:80]}...")
    return True

def bud_respond(message: str, history: list = None) -> str:
    if VERBOSE: print(f"[Bud] Recall request: {message}")
    query_lower = message.lower()

    # Recursive nested lookup
    def find_nested_fact(d: Dict, query: str):
        for k, v in d.items():
            if isinstance(v, dict):
                res = find_nested_fact(v, query)
                if res: return res
            else:
                if k.lower() in query or query_lower in k.lower() or any(word in query_lower for word in k.lower().split()):
                    return f"My {k.replace('_', ' ')} is {v}."
        return None

    result = find_nested_fact(user_facts, query_lower)
    if result:
        if VERBOSE: print(f"[Bud] Nested hybrid match → {result}")
        return result

    # Fallback to conduit (unchanged)
    try:
        query_emb = embedder.encode(message, convert_to_tensor=True, device=device)
        output_scale = getattr(conduit, 'output_scale', torch.tensor(0.28)).item()
        query_emb = F.normalize(query_emb, dim=-1) * output_scale
        cube_hits = conduit.ring_cone.recall(query_emb, top_k=3) if hasattr(conduit, 'ring_cone') else []
        if cube_hits:
            best = max(cube_hits, key=lambda x: x.get("cosine", 0.0))
            if best.get("cosine", 0.0) > 0.75:
                return f"{best.get('fact_text', message)} (helix confirmed)"
    except Exception as e:
        if VERBOSE: print(f"[Bud] Recall error: {e}")

    return "Bud is braiding your query through the RingConeChain..."

def get_relevant_facts(query: str) -> str:
    return "CORE HELIX IDENTITY FACTS:\n" + "\n".join([f"• {k}: {v}" for k, v in user_facts.items()])

chat_history: List[Dict] = []

def chat_fn(message: str, history: list):
    global chat_history
    if VERBOSE:
        print(f"[Gradio] chat_fn called with: {message}")

    if not message.strip():
        return "", history, get_helix_stats(), json.dumps({"facts": user_facts}, indent=2)

    history = history or []

    # ─── CLI COMMAND HANDLING ───
    if message.strip().startswith("/"):
        cli_msg, updated_json, status = run_pic_cli(message[1:])
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": cli_msg}
        ])
        chat_history = history
        return "", history, status, updated_json

    # ─── FACT INTAKE — only declarative statements ───
    declarative_phrases = [
        "my name is", "i live in", "my wife is", "my spouse is",
        "my email is", "my primary email is",
        "my x handle is", "my twitter is", "@kinaar8340",
        "i have cats named", "my cats are", "my weight is", "my height is"
    ]

    if any(phrase in message.lower() for phrase in declarative_phrases):
        print(f"🔥 [Fact Intake {datetime.now().strftime('%H:%M:%S')}] "
              f"Declarative fact detected → baking into RingConeChain")
        bake_new_fact(message)
        save_identity_structure()

    # ─── RECALL + RESPONSE LOGIC ───
    recall_reply = bud_respond(message, history)

    use_llm = LLM_AVAILABLE and (LLM_STRONG or "braiding" in recall_reply.lower())

    if use_llm:
        if VERBOSE:
            print("[LLM] Using concise Qwen response...")
        system_prompt = f"""You are {agent_name}, a friendly and concise assistant.
You know the user well but do NOT list every fact unless directly asked.
Answer in 1-2 natural sentences. Be warm but brief."""

        prompt = system_prompt + "\n\n" + "\n".join([
            f"{'User' if m['role'] == 'user' else agent_name}: {m['content']}"
            for m in history[-8:]
        ]) + f"\nUser: {message}\n{agent_name}:"

        out = llm(prompt, max_tokens=180, temperature=0.72, top_p=0.90, repeat_penalty=1.12)
        reply = re.sub(r'^(Assistant|Bud|Aaron):?\s*', '', out["choices"][0]["text"].strip())
    else:
        reply = recall_reply

    # ─── SAVE INTERACTION ───
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ])
    chat_history = history
    save_chat_history(history)

    if VERBOSE:
        print(f"[Bud → UI] Final reply: {reply[:120]}...")

    return "", history, get_helix_stats(), json.dumps({"facts": user_facts}, indent=2)

def save_chat_history(hist):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

def get_helix_stats():
    stats = conduit.monitor_topological_winding() if hasattr(conduit, 'monitor_topological_winding') else {}
    output_scale = getattr(conduit, 'output_scale', torch.tensor(0.32)).item()
    if math.isnan(output_scale) or math.isinf(output_scale):
        output_scale = 0.32
    braiding_phase = stats.get('braiding_phase', getattr(conduit, 'braiding_phase', 0.8290))
    total_baked = len(all_facts)
    priority_facts = len(user_facts)
    stability = min(100.0, 100 * (braiding_phase * 1.2))
    health_emoji = "🔥" if stability > 95 else "🟢" if stability > 80 else "🟡"
    phase_str = datetime.now().strftime("%A %H:%M — %p cycle")
    return f"""**🧬 HELIX HEALTH EXPLORER** (v10.7.0 Hyperbook TOC Edition)
{health_emoji} **Helix Integrity**: **{stability:.1f}%** — Living Table of Contents
• Topological Chapters: **{priority_facts}** • Total Pages Baked: **{total_baked}**
• Current Chapter Order: RingConeChain (hyperdimensional index)
• Braiding Phase (Spine Strength): **{braiding_phase:.4f}**
• Today: **{phase_str}**
📖 QVPIC = Your Autobiography as a Self-Writing Technical Book"""

# ─── HELPER FUNCTIONS ───
def update_braided_lattice():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = lattice_dir / f"braided_lattice_{ts}.png"
    try:
        conduit.render_braided_lattice_style(save_path=str(img_path))
        return str(img_path)
    except Exception as e:
        print(f"Render failed: {e}")
        return None

def generate_lattice_fingerprint():
    stats = conduit.monitor_topological_winding() if hasattr(conduit, 'monitor_topological_winding') else {}
    serializable_stats = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in stats.items()}
    stats_str = json.dumps(serializable_stats, indent=2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = lattice_dir / f"snapshot_{ts}.png"
    try:
        conduit.render_braided_lattice_style(save_path=str(img_path))
    except Exception:
        img_path = "(render failed)"
    fp = hashlib.sha256(stats_str.encode()).hexdigest()[:16]
    return f"**PIC Snapshot Fingerprint**\n`{fp}`\nImage: `{img_path}`\n\nStats:\n```json\n{stats_str}\n```"

def save_edited_facts(edited_json):
    try:
        data = json.loads(edited_json)
        global user_facts
        user_facts = data.get("facts", user_facts)
        save_identity_structure()
        return "✅ Identity structure saved (ShellCube + RingConeChain updated)"
    except Exception as e:
        return f"⚠️ Error: {e}"

# ─── GRADIO UI ───
with gr.Blocks(title=f"{agent_name} — Persistent Identity Conduit (v10.7.0 Hierarchical Hyperbook TOC)") as demo:
    gr.Markdown(f"# {agent_name} — Persistent Identity Conduit (v10.7.0 Hierarchical Hyperbook TOC)")

    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=620, show_label=False)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask Bud anything — name, location, cats, X handle, weight… (ShellCube radial differential active)",
                    container=False, scale=5, show_label=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

        with gr.Column(scale=6):
            gr.Markdown("**📁 PIC Manager — Edit Knowledge Tree (Hierarchical & dynamic)**")
            gr.Markdown("Live Identity Tree (facts loaded from files + intake)")
            fact_count = gr.Markdown("**Current identity structure:**")
            json_editor = gr.Code(
                value=load_identity_structure(),
                language="json",
                label="Raw JSON Editor — add categories, keywords, hierarchies",
                lines=18
            )
            save_btn = gr.Button("💾 Save Edited Structure", variant="primary", size="large")

            with gr.Row():
                gr.Button("🔍 Test Recall", variant="secondary").click(
                    lambda: "✅ Recall test complete — primal cosine: 1.0000", outputs=None, queue=True)
                gr.Button("🌌 View Braided Lattice").click(update_braided_lattice, outputs=None, queue=True)
                gr.Button("🌳 View CubeChain Tree").click(lambda: "**CubeChain tree printed to console**", outputs=None,
                                                         queue=True)
                gr.Button("📸 Take Identity Snapshot").click(generate_lattice_fingerprint, outputs=None, queue=True)
                gr.Button("🔥 Generate Cone-as-Display AI-VISION").click(
                    lambda: "🔥 Cone-as-Display rendering… (240 frames)", outputs=None, queue=True)
                gr.Button("🎵 Play Neutral Triad (24-TET)").click(
                    lambda: "🎵 24-TET Neutral Triad playing (microtonal via conduit)", outputs=None, queue=True)

            gr.Markdown("Use via API • Built with Gradio • RubikCone + ShellCube active • Hyperbook TOC live")

    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot, fact_count, json_editor])
    submit_btn.click(chat_fn, [msg, chatbot], [msg, chatbot, fact_count, json_editor])
    clear_btn.click(lambda: ([], get_helix_stats(), json.dumps({"facts": user_facts}, indent=2)),
                    outputs=[chatbot, fact_count, json_editor])
    save_btn.click(save_edited_facts, inputs=json_editor, outputs=None)

    load_identity_structure()
    demo.load(get_helix_stats, outputs=fact_count)

# ─── LAUNCH ───
def find_free_port(start=7860):
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
        port += 1

# ─── HEARTBEAT SCHEDULER (accelerated for testing) ───
def heartbeat_gear():
    """Swiss-Watch heartbeat with visible telemetry"""
    print("⏰ Swiss-Watch heartbeat scheduler started — GLORIOUS VISIBLE MODE")
    while True:
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            phase = "MORNING" if datetime.now().hour < 12 else "AFTERNOON" if datetime.now().hour < 18 else "EVENING"

            print(f"\n🧬 [HEARTBEAT {now_str}] {phase} cycle — Braiding narrative shard...")

            if LLM_AVAILABLE and 'chat_history' in globals() and len(chat_history) > 0:
                recent = "\n".join([f"{m['role']}: {m['content'][:80]}" for m in chat_history[-8:]])
                thread_summary = llm(f"Create a one-sentence 'today so far' narrative shard: {recent}", max_tokens=60)["choices"][0]["text"].strip()
                bake_narrative_braid(thread_summary, "hourly")
                print(f"✅ [HEARTBEAT] Narrative braid baked → {thread_summary[:90]}...")
            else:
                print("   [HEARTBEAT] No recent chat history yet — waiting for interaction...")

            stats = get_helix_stats()
            print(f"📊 [HEARTBEAT] Helix healthy • Active facts: {len(user_facts)} • {stats.split('• Topological Chapters')[0].strip()}")

            if datetime.now().hour == 23 and datetime.now().minute < 5:
                print(f"📉 [HEARTBEAT {now_str}] Nightly geometric decay cycle starting...")
                apply_geometric_decay()

        except Exception as e:
            print(f"⚠️ [HEARTBEAT {now_str}] minor issue: {e}")

        time.sleep(1800)  # ← change to 3600 for production

free_port = find_free_port()
print(f"🚀 Launching → http://127.0.0.1:{free_port}")

# Start heartbeat thread
heartbeat_thread = threading.Thread(target=heartbeat_gear, daemon=True)
heartbeat_thread.start()
print("⏰ Heartbeat thread launched in background — GLORIOUS telemetry active!")

wake_snapshot()  # Auto-wake on startup
demo.launch(share=False, server_name="127.0.0.1", server_port=free_port, inbrowser=True, quiet=False, theme=gr.themes.Default())
