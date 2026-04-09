#!/usr/bin/env python3
"""
scripts/agent.py — Core Agent Logic (v10.8.4 — Performance Optimized)
Optimizations: batched embeddings, torch.compile, fp16 embedder, dedup baking, caching
"""

import torch
import re
import math
import hashlib
import requests
import os
import sys
import json
import time
import atexit
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache

# Project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="PIC v10.8.4 — TV-Optimized + Performance")
parser.add_argument('--name', type=str, default='Bud')
parser.add_argument('--no-reset', action='store_true')
parser.add_argument('--vqc', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--llm-strong', action='store_true')
parser.add_argument('--heartbeat-minutes', type=int, default=60)
parser.add_argument('--profile', action='store_true', help="Enable torch profiler for debugging")
args = parser.parse_args()

from src.config import load_config
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


def load_facts_json():
    """Load the new structured JSON facts (replaces old .txt loading)"""
    global all_facts
    all_facts = []

    for f in (public_facts_file, private_facts_file):
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                for entry in data:
                    if entry.get("text"):  # only add non-empty
                        all_facts.append(entry["text"])
            except Exception as e:
                print(f"⚠️  Could not load {f}: {e}")
        else:
            print(f"⚠️  {f} not found")

    print(f"✅ Loaded {len(all_facts)} facts from new JSON identity files")
    return all_facts

# ==================== GLOBALS ====================
agent_name = "Bud"
USE_VQC = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
HEARTBEAT_MINUTES = 60
VERBOSE = False
LLM_STRONG = False
LLM_AVAILABLE = False
llm = None
chat_history: List[Dict] = []
last_message_time = time.time()
all_facts: List[str] = []
user_facts: Dict[str, Any] = {}

# Performance flags
PROFILER = None
if args.profile and torch.cuda.is_available():
    PROFILER = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True, with_stack=True
    )
    PROFILER.__enter__()

# ==================== CONDUIT & EMBEDDER (Optimized) ====================
if USE_VQC:
    from src.vqc_enhanced_conduit import VQCEnhancedHelicalConduit as ConduitClass
else:
    from src.conduit import RubikConeConduit as ConduitClass

cfg = load_config("configs/default.yaml")

# Use full float32 for training stability (fp16 caused Half/Float mismatch with compiled conduit)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

conduit = ConduitClass(
    embed_dim=384, twist_rate=12.5, max_depth=48.0,
    num_polarizations=3, quat_logical_dim=96,
    toroidal_modulo9=True, vortex_math_369=True, clifford_projection=True
).to(device)
conduit.device = device

# === CRITICAL: Compile the conduit for massive training_step speedup ===
if torch.__version__ >= "2.0" and device == "cuda":
    try:
        conduit = torch.compile(conduit, mode="default")  # "reduce-overhead" can be aggressive with mixed precision
        print("🚀 conduit compiled with torch.compile (default mode)")
    except Exception as e:
        print(f"⚠️ compile skipped: {e}")

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "pic_conduit_final.pt"

optimizer = torch.optim.AdamW(conduit.parameters(), lr=8e-4, weight_decay=cfg.training.weight_decay)

# ==================== FILE PATHS & CACHING ====================
identity_structure_path = Path("identity_structure.json")
history_file = Path("chat_history.json")
public_facts_file = Path("facts/public_facts.json")
private_facts_file = Path("facts/private_facts.json")
DAILY_HELIX_LOG = Path("logs/daily_helix.jsonl")
DAILY_HELIX_LOG.parent.mkdir(parents=True, exist_ok=True)

# Simple fact dedup cache
seen_facts = set()

for f in (public_facts_file, private_facts_file):
    f.touch(exist_ok=True)

# ==================== HELPERS (Cached where hot) ====================
@lru_cache(maxsize=32)
def get_canonical_order():
    return getattr(cfg.identity, "canonical_order", {}) if hasattr(cfg, "identity") else {}

def set_nested(d: dict, path: str, value: str):
    if not path or path == "/": return
    keys = [k.strip() for k in path.strip("/").split("/") if k.strip()]
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

def get_nested(d: dict, path: str):
    keys = [k.strip() for k in path.strip("/").split("/") if k.strip()]
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current

def delete_nested(d: Dict, path: str) -> bool:
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

def sort_identity_tree(data: dict) -> dict:
    canonical = get_canonical_order()
    def _reorder(d: dict, path: str = "") -> dict:
        if not isinstance(d, dict): return d
        key = path.split(".")[-1] if path else ""
        order = canonical.get(key, list(d.keys()))
        ordered = {k: _reorder(d[k], f"{path}.{k}" if path else k) for k in order if k in d}
        for k in d:
            if k not in ordered:
                ordered[k] = _reorder(d[k], f"{path}.{k}" if path else k)
        return ordered
    return _reorder(data)

def flatten_for_bake(facts: Dict) -> List[str]:
    flat = []
    def recurse(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                recurse(v, f"{prefix}{k}/")
            else:
                natural = f"My {prefix.replace('/', ' ').strip()} {k.replace('_', ' ')} is {v}."
                flat.append(natural)
    recurse(facts)
    return flat

# ==================== LOAD / SAVE ====================
def load_identity_structure():
    global user_facts
    try:
        if identity_structure_path.exists():
            with open(identity_structure_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            user_facts = data.get("facts", {})
            print(f"✅ Loaded identity structure ({len(user_facts)} sections)")
        else:
            load_facts_json()
    except Exception as e:
        print(f"⚠️ Structure load failed ({e}) — regenerating")
        load_facts_json()
    return json.dumps({"facts": user_facts}, indent=2)

def populate_system_facts():
    now = datetime.now()
    set_nested(user_facts, "core/system/current_date", now.strftime("%Y-%m-%d"))
    set_nested(user_facts, "core/system/current_time", now.strftime("%H:%M"))
    set_nested(user_facts, "core/system/location", user_facts.get("core/identity/location", "Wilsonville, Oregon"))
    save_identity_structure()

def save_identity_structure():
    data = {"facts": sort_identity_tree(user_facts)}
    with open(identity_structure_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ==================== JOURNAL + BAKING (Optimized) ====================
def append_to_journal(entry_text: str):
    journal_path = Path("identity/agent/ajournal.md")
    journal_path.parent.mkdir(exist_ok=True)
    if not journal_path.exists():
        journal_path.write_text("# QVPIC Agent Journal — Living Autobiography\n\n", encoding="utf-8")

    words = len(entry_text.split())
    if words > cfg.journal.max_words_per_entry:
        entry_text = " ".join(entry_text.split()[:cfg.journal.max_words_per_entry]) + " [...]"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = conduit.monitor_topological_winding() if hasattr(conduit, 'monitor_topological_winding') else {}
    header = f"\n\n---\n**Entry {timestamp}** — Braiding Phase: {stats.get('braiding_phase', 0.0):.4f}\n"

    with journal_path.open("a", encoding="utf-8") as f:
        f.write(header + entry_text.strip() + "\n")

    # Single batched embed
    emb = F.normalize(embedder.encode(clean_message, convert_to_tensor=True, device=device), dim=-1) * 0.28
    emb = emb.to(torch.float32)  # ← Force float32 before feeding to conduit
    item = {'emb': emb, 's': 88.0 + len(all_facts), 'pol_idx': 2}
    conduit.training_step(inputs=[item], optimizer=optimizer)

    if hasattr(conduit, 'ring_cone'):
        ring_idx = len(all_facts) % getattr(conduit.ring_cone, 'NUM_RINGS', 8)
        cube_local = len(all_facts) % getattr(conduit.ring_cone.rings[ring_idx], 'num_cubes', 27) if hasattr(conduit.ring_cone, 'rings') else 0
        conduit.ring_cone.bake_ring(ring_idx, cube_local, emb, orientation=len(all_facts) % 24)

    print("📖 Journal baked efficiently")


def populate_user_facts_from_files():
    global user_facts
    user_facts = {}
    lines = []
    for path in [public_facts_file, private_facts_file]:
        if path.exists():
            lines.extend([line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

    # Parse into nested
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

# Batch bake all facts at once
    facts_to_bake = flatten_for_bake(user_facts)
    if facts_to_bake:
        # Single batched embed
        emb = F.normalize(embedder.encode(clean_message, convert_to_tensor=True, device=device), dim=-1) * 0.28
        emb = emb.to(torch.float32)  # ← Force float32 before feeding to conduit
        for i, fact in enumerate(facts_to_bake):
            if fact not in seen_facts:
                item = {'emb': emb, 's': 2.1 + len(all_facts) * 0.45, 'pol_idx': 1}
                conduit.training_step(inputs=[item], optimizer=optimizer)
                all_facts.append(fact)
                seen_facts.add(fact)
                # ring bake (guarded)
                if hasattr(conduit, 'ring_cone'):
                    ring_idx = len(all_facts) % getattr(conduit.ring_cone, 'NUM_RINGS', 8)
                    cube_local = len(all_facts) % getattr(conduit.ring_cone.rings[ring_idx], 'num_cubes', 27) if hasattr(conduit.ring_cone, 'rings') else 0
                    conduit.ring_cone.bake_ring(ring_idx, cube_local, emb.squeeze(0), orientation=len(all_facts) % 24)

    print(f"✅ [Core Identity] Batched re-bake complete ({len(facts_to_bake)} facts)")
    save_identity_structure()


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
        help_text = """**PIC v10.8.3 CLI — Hierarchical Hyperbook TOC**
/add category/sub/key "value"
/remove category/sub/key
/chapter category/sub/key
/toc — beautiful indented tree
/save, /wake, /sleep, /list"""
        return help_text, json.dumps({"facts": user_facts}, indent=2), get_helix_stats()

    msg = ""
    if verb in ("add", "set"):
        if key and value:
            if '/' in key:
                set_nested(user_facts, key, value.strip())
                user_facts = sort_identity_tree(user_facts)
                msg = f"✅ Etched {key} = {value} into RingConeChain"
            else:
                user_facts[key] = value
                msg = f"✅ Set {key} = {value}"
            for natural in flatten_for_bake(user_facts):
                bake_new_fact(natural)
        else:
            msg = "❓ Usage: /add <path/to/key> <value>"

    elif verb in ("rm", "remove", "delete"):
        if key:
            if delete_nested(user_facts, key) or (
                    key in user_facts and not isinstance(user_facts[key], dict) and (user_facts.pop(key) or True)):
                msg = f"✅ Removed {key}"
            else:
                msg = f"❓ Path/Key '{key}' not found"
        else:
            msg = "❓ Usage: /remove category/sub/key"

    elif verb == "chapter":
        if key:
            val = get_nested(user_facts, key)
            if val is not None:
                natural = f"My {key.replace('/', ' ')} is {val}."
                page_id = abs(hash(natural)) % 10000
                msg = f"📖 **Chapter {key.replace('/', ' • ')}**\n\n{natural}\n\n📍 Topological page: {page_id:04d}"
            else:
                msg = f"❓ Chapter path '{key}' not found"
        else:
            msg = "❓ Usage: /chapter category/sub/key"

    elif verb in ("list", "show", "toc"):
        if verb == "toc":
            toc_lines = ["# QVPIC Hyperbook — Table of Contents\n"]
            toc_lines.append(f"**Edition**: v10.8.3 TV-Optimized\n**Top-level Sections**: {len(user_facts)}\n")

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
        else:
            msg = f"Current top-level sections: {list(user_facts.keys())}"

    elif verb == "save":
        torch.save(conduit.state_dict(), checkpoint_path)
        save_identity_structure()
        for natural in flatten_for_bake(user_facts):
            bake_new_fact(natural)
        msg = "✅ Helix checkpoint + hierarchical re-bake complete"

    elif verb == "wake":
        wake_snapshot()
        msg = "🌅 Wake snapshot + morning narrative braid completed"

    elif verb == "sleep":
        sleep_snapshot()
        msg = "🌙 Sleep snapshot + daily autobiography baked"

    elif verb == "sms":
        if not key or not value:
            msg = "❓ Usage: /sms <path/to/contact> \"your message here\""
        else:
            phone = get_nested(user_facts, f"{key}/phone")
            if not phone:
                msg = f"❓ No phone number found at path: {key}"
            else:
                cfg_sms = cfg.sms  # Use global cfg (fixed)
                if not cfg_sms.enabled or not cfg_sms.account_sid or not cfg_sms.from_number:
                    msg = "❓ SMS not fully configured in default.yaml"
                else:
                    try:
                        from twilio.rest import Client
                        client = Client(cfg_sms.account_sid, cfg_sms.auth_token)
                        message = client.messages.create(
                            body=value,
                            from_=cfg_sms.from_number,
                            to=phone
                        )
                        msg = f"✅ SMS sent to {key} → {phone}\n\"{value}\"\nMessage SID: {message.sid}"
                        bake_new_fact(f"[SMS sent to {key}] {value}")
                    except Exception as e:
                        msg = f"❌ SMS failed: {e}"
    else:
        msg = "❓ Unknown command. Try /help"

    save_identity_structure()
    updated_json = json.dumps({"facts": user_facts}, indent=2)
    return msg, updated_json, get_helix_stats()


def bake_new_fact(message: str) -> bool:
    clean_message = re.sub(r'\s*(\+\+|---)(public|private)-facts\s*', '', message, flags=re.IGNORECASE).strip()
    if len(clean_message) < 12 or clean_message in seen_facts:
        return False
    emb = F.normalize(embedder.encode(clean_message, convert_to_tensor=True, device=device), dim=-1) * 0.28
    emb = emb.to(torch.float32)  # ← Force float32 before feeding to conduit
    item = {'emb': emb, 's': 2.1 + len(all_facts) * 0.45, 'pol_idx': 1}
    conduit.training_step(inputs=[item], optimizer=optimizer)
    all_facts.append(clean_message)
    seen_facts.add(clean_message)

    if hasattr(conduit, 'ring_cone'):
        ring_idx = len(all_facts) % getattr(conduit.ring_cone, 'NUM_RINGS', 8)
        cube_local = len(all_facts) % getattr(conduit.ring_cone.rings[ring_idx], 'num_cubes', 27) if hasattr(conduit.ring_cone, 'rings') else 0
        conduit.ring_cone.bake_ring(ring_idx, cube_local, emb, orientation=len(all_facts) % 24)

    print(f"✅ Baked: {clean_message[:80]}...")
    return True


def bud_respond(message: str, history: list = None) -> str:
    if VERBOSE:
        print(f"[Bud] Recall request: {message}")
    query_lower = message.lower()

    def find_nested_fact(d: Dict, query: str):
        for k, v in d.items():
            if isinstance(v, dict):
                res = find_nested_fact(v, query)
                if res:
                    return res
            else:
                if k.lower() in query or query_lower in k.lower() or any(
                        word in query_lower for word in k.lower().split()):
                    return f"My {k.replace('_', ' ')} is {v}."
        return None

    result = find_nested_fact(user_facts, query_lower)
    if result:
        if VERBOSE:
            print(f"[Bud] Nested hybrid match → {result}")
        return result

    try:
        query_emb = embedder.encode(message, convert_to_tensor=True, device=device)
        output_scale = getattr(conduit, 'output_scale', torch.tensor(0.28)).item()
        query_emb = F.normalize(query_emb, dim=-1) * output_scale
        if hasattr(conduit, 'ring_cone'):
            cube_hits = conduit.ring_cone.recall(query_emb, top_k=3)
            if cube_hits:
                best = max(cube_hits, key=lambda x: x.get("cosine", 0.0))
                if best.get("cosine", 0.0) > 0.75:
                    return f"{best.get('fact_text', message)} (helix confirmed)"
    except Exception as e:
        if VERBOSE:
            print(f"[Bud] Recall error: {e}")

    return "Bud is braiding your query through the RingConeChain..."


def get_relevant_facts(query: str) -> str:
    return "CORE HELIX IDENTITY FACTS:\n" + "\n".join([f"• {k}: {v}" for k, v in user_facts.items()])


def chat_fn(message: str, history: list):
    global chat_history, last_message_time
    if not message.strip():
        return "", history, get_helix_stats(), json.dumps({"facts": user_facts}, indent=2)

    history = history or []
    last_message_time = time.time()

    if message.strip().startswith("/"):
        cli_msg, updated_json, status = run_pic_cli(message[1:])
        history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": cli_msg}])
        chat_history = history
        return "", history, status, updated_json

    declarative_phrases = ["my name is", "i live in", "my wife is", "my spouse is", "my email is", "my x handle is",
                           "@kinaar8340"]
    if any(phrase in message.lower() for phrase in declarative_phrases):
        bake_new_fact(message)
        save_identity_structure()

    recall_reply = bud_respond(message, history)

    expand_requested = any(kw in message.lower() for kw in
                           ["expand", "short story", "spin me a story", "tell me a story", "weave", "narrative",
                            "more detail", "ideas for", "summarize", "readme"])

    if LLM_AVAILABLE and expand_requested:
        facts_text = get_relevant_facts(message)
        github_content = ""
        if "readme" in message.lower() or "github.com/kinaar8340/qvpic" in message.lower():
            github_content = fetch_github_file("https://github.com/kinaar8340/qvpic/blob/main/README.md")
            if "Failed to fetch" not in github_content:
                facts_text += "\n\n=== ACTUAL README.md CONTENT FROM GITHUB ===\n" + github_content + "\n=== END OF README.md ===\n"

        system_prompt = f"""You are {agent_name}, a warm, reflective assistant.
Use ONLY the real helix facts and any fetched GitHub content below.
When the user says "expand" or asks for a story/narrative/summary, give the richest, most truth-seeking reply possible while staying grounded.

Current helix facts:
{facts_text}"""

        prompt = system_prompt + "\n\nRecent chat:\n" + "\n".join(
            [f"{'User' if m['role'] == 'user' else agent_name}: {m['content']}" for m in history[-8:]]
        ) + f"\nUser: {message}\n{agent_name}:"

        out = llm(prompt, max_tokens=1200, temperature=0.78, top_p=0.92, repeat_penalty=1.10)
        reply = re.sub(r'^(Assistant|Bud|Aaron|User):?\s*', '', out["choices"][0]["text"].strip())
    else:
        reply = recall_reply

    history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
    chat_history = history
    save_chat_history(history)

    return "", history, get_helix_stats(), json.dumps({"facts": user_facts}, indent=2)


def save_chat_history(hist):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)


def get_helix_stats():
    try:
        stats = conduit.monitor_topological_winding() if hasattr(conduit, 'monitor_topological_winding') else {}
        output_scale = getattr(conduit, 'output_scale', torch.tensor(0.32)).item()
        if math.isnan(output_scale) or math.isinf(output_scale):
            output_scale = 0.32
        braiding_phase = stats.get('braiding_phase', getattr(conduit, 'braiding_phase', 0.8290))
        total_baked = len(all_facts)
        priority_facts = len(user_facts)
        stability = min(100.0, 100 * (braiding_phase * 1.2))
        health_emoji = "🟢" if stability > 95 else "🟡" if stability > 80 else "🔴"
        phase_str = datetime.now().strftime("%A %H:%M — %p cycle")
        return f"""**🌀 HELIX HEALTH EXPLORER** (v10.8.3 TV-Optimized)
{health_emoji} **Helix Integrity**: **{stability:.1f}%**
📚 Topological Chapters: **{priority_facts}** | 📖 Total Pages Baked: **{total_baked}**
🔄 Current Chapter Order: RingConeChain
🌌 Braiding Phase: **{braiding_phase:.4f}**
📅 Today: **{phase_str}**"""
    except Exception:
        return "**🌀 HELIX HEALTH** — Monitoring active (v10.8.3)"


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
    log_helix_event("wake")


def sleep_snapshot():
    print("🌙 Bud entering rest — daily autobiography page")
    log_helix_event("sleep")


atexit.register(sleep_snapshot)


def bake_narrative_braid(summary: str, meta_type: str = "daily"):
    fact = f"[META-AUTOBIOGRAPHY {meta_type.upper()}] {summary}"
    bake_new_fact(fact)
    print(f"✅ Narrative braid baked ({meta_type})")
    log_helix_event(f"narrative_{meta_type}", summary)


def fetch_github_file(url: str) -> str:
    try:
        if "github.com" in url and "/blob/" in url:
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        else:
            raw_url = url
        response = requests.get(raw_url, timeout=10)
        response.raise_for_status()
        return response.text[:12000]
    except Exception as e:
        return f"Failed to fetch file: {str(e)}"

def initialize_agent(args):
    global agent_name, USE_VQC, VERBOSE, LLM_STRONG, HEARTBEAT_MINUTES, LLM_AVAILABLE, llm
    agent_name = args.name.strip()
    USE_VQC = args.vqc
    VERBOSE = args.verbose
    LLM_STRONG = args.llm_strong
    HEARTBEAT_MINUTES = args.heartbeat_minutes

    # LLM init unchanged
    try:
        from llama_cpp import Llama
        llm = Llama(model_path="models/Qwen2.5-14B-Instruct-Q4_K_M.gguf", n_gpu_layers=99, n_ctx=32768, n_batch=1024, n_threads=16, verbose=False, flash_attn=True)
        LLM_AVAILABLE = True
        print("✅ Qwen loaded")
    except Exception as e:
        print(f"⚠️ LLM: {e}")
        LLM_AVAILABLE = False

    if checkpoint_path.exists() and args.no_reset:
        print("✅ Loading checkpoint...")
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
        conduit.load_state_dict(state, strict=False)

    print(f"✅ Agent initialized (name={agent_name}, Performance Mode)")

    if PROFILER:
        print("🔍 Profiler active — check console/tensorboard after run")

# ==================== EXPOSE FOR UI & MAIN ====================
# Make key objects available at module level for ui.py
# (already globals, but explicit for clarity)

__all__ = ["initialize_agent", "chat_fn", "get_helix_stats", "load_identity_structure",
           "HEARTBEAT_MINUTES", "LLM_AVAILABLE", "llm", "chat_history", "last_message_time",
           "bake_narrative_braid", "wake_snapshot", "sleep_snapshot", "append_to_journal",
           "conduit", "user_facts", "USE_VQC"]   # ← added these

if __name__ == "__main__":
    initialize_agent(args)
    if PROFILER:
        PROFILER.__exit__(None, None, None)
        print("📊 Profiler summary available")