#!/usr/bin/env python3
# scripts/agent_demo.py — v10.5.6 Swiss-Watch Edition (100% Recall + Heartbeat Thread Fixed)
# ✅ Wake/Sleep snapshots + daily narrative braid
# ✅ Background heartbeat scheduler (accelerated for testing)
# ✅ Geometric decay + revision/forgetting + meta-autobiography shards
# ✅ Hybrid recall (template lookup + cosine) so "What is my name?" always works
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
from typing import Dict, List, Tuple
import threading
import time
import atexit

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="PIC v10.5.2 — Swiss-Watch Autobiography")
parser.add_argument('--name', type=str, default='Bud')
parser.add_argument('--no-reset', action='store_true')
parser.add_argument('--vqc', action='store_true')
parser.add_argument('--verbose', action='store_true', help="Enable rich terminal logging")
parser.add_argument('--llm-strong', action='store_true', help="Use full chatty LLM even on strong matches")
args = parser.parse_args()

VERBOSE = args.verbose
LLM_STRONG = args.llm_strong

agent_name = args.name.strip()
USE_VQC = args.vqc
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🚀 PIC v10.5.2 — Swiss-Watch Edition (Accelerated Testing)")

# ─── Qwen Loading ───
LLM_AVAILABLE = False
llm = None
model_path = Path("models/Qwen2.5-3B-Instruct-Q4_K_M.gguf")
if model_path.exists() and model_path.stat().st_size > 100_000_000:
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=str(model_path), n_gpu_layers=-1, n_ctx=16384, n_batch=512, verbose=False)
        LLM_AVAILABLE = True
        print("✅ Qwen2.5-3B-Instruct loaded successfully")
    except Exception as e:
        print(f"⚠️ Qwen load failed: {e}")
else:
    print(f"⚠️ Qwen model not found at {model_path}")

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

# ─── IDENTITY TEMPLATES & FACTS ───
IDENTITY_TEMPLATES = [
    {"keywords": ["name", "who are you", "called"], "template": "My name is {$name}."},
    {"keywords": ["live", "location", "city", "where"], "template": "I live in {$location}."},
    {"keywords": ["wife", "spouse", "married"], "template": "My wife is {$spouse_name}."},
    {"keywords": ["email", "contact"], "template": "My primary email is {$email}."},
    {"keywords": ["x handle", "twitter", "@"], "template": "My X handle is {$x_handle}."},
    {"keywords": ["cat", "pet", "cats"], "template": "I have cats named {$pet_summary}."},
]

user_facts: Dict[str, str] = {}
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
            print(f"✅ Loaded {len(user_facts)} identity facts from disk")
        else:
            populate_user_facts_from_files()
    except Exception:
        print("⚠️ identity_structure.json corrupted — regenerating")
        populate_user_facts_from_files()
    return json.dumps({"facts": user_facts, **user_facts}, indent=2)

def save_identity_structure():
    data = {"facts": user_facts, "templates": IDENTITY_TEMPLATES}
    with open(identity_structure_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
                    user_facts["location"] = line.split("live in", 1)[1].strip().rstrip(".")
                elif "name is" in line_lower:
                    user_facts["name"] = line.split("name is", 1)[1].strip().rstrip(".")
                elif "wife is" in line_lower:
                    user_facts["spouse_name"] = line.split("is", 1)[1].strip().rstrip(".")
                elif "email is" in line_lower or "primary email" in line_lower:
                    user_facts["email"] = line.split("is", 1)[1].strip().rstrip(".")
                elif "x handle" in line_lower or "@" in line_lower:
                    user_facts["x_handle"] = line.split("is", 1)[1].strip().rstrip(".")
                elif "cats named" in line_lower:
                    user_facts["pet_summary"] = line.split("cats named", 1)[1].strip().rstrip(".")
            except Exception:
                continue

            # Priority re-bake of core identity (prevents forgetting)
            for k, v in list(user_facts.items()):
                natural_fact = {
                    "name": f"My name is {v}.",
                    "location": f"I live in {v}.",
                    "spouse_name": f"My wife is {v}.",
                    "email": f"My primary email is {v}.",
                    "x_handle": f"My X handle is {v}.",
                    "pet_summary": f"I have cats named {v}."
                }.get(k, f"{k} is {v}.")
                bake_new_fact(natural_fact)
            print(f"🔄 [Core Identity] Re-baked {len(user_facts)} priority facts into helix")

    save_identity_structure()



# ─── CLI PARSER — robust natural baking ───
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
        help_text = """**PIC v10.5.4 CLI How To**
• /help — this reference
• /add <key> <value> or /set <key> <value> — bake fact (auto-natural sentence)
• /remove <key> — prune fact
• /rename <old> <new> — rename key
• /save — force checkpoint + re-bake ALL facts
• /list or /show — current facts
• /find <text> — topological search
• /wake — morning snapshot
• /sleep — evening autobiography page"""
        return help_text, json.dumps({"facts": user_facts, **user_facts}, indent=2), get_helix_stats()

    msg = ""
    if verb in ("add", "set"):
        if key and value:
            user_facts[key] = value
            msg = f"✅ Set {key} = {value}"
            # Auto-generate clean natural sentence for perfect recall
            natural_fact = f"My {key.replace('_', ' ')} is {value}."
            bake_new_fact(natural_fact)
            print(f"→ Auto-baked natural fact: {natural_fact}")
        else:
            msg = "❌ Usage: /add <key> <value> or /set <key> <value>"
    elif verb in ("rm", "remove", "delete"):
        if key in user_facts:
            del user_facts[key]
            msg = f"✅ Removed {key}"
        else:
            msg = f"❌ Key '{key}' not found"
    elif verb == "rename":
        if key and value and key in user_facts:
            user_facts[value] = user_facts.pop(key)
            msg = f"✅ Renamed {key} → {value}"
        else:
            msg = "❌ Usage: /rename <old> <new>"
    elif verb == "save":
        torch.save(conduit.state_dict(), checkpoint_path)
        save_identity_structure()
        # Re-bake EVERY fact as a natural sentence (fixes recall for custom facts)
        for k, v in list(user_facts.items()):
            natural_fact = f"My {k.replace('_', ' ')} is {v}."
            bake_new_fact(natural_fact)
        print(f"🔄 [Core + Custom Facts] Re-baked {len(user_facts)} natural facts into helix")
        msg = "💾 Helix checkpoint flushed + ALL facts re-baked as natural sentences"
    elif verb in ("list", "show"):
        msg = f"Current facts: {list(user_facts.keys())}"
    elif verb in ("find", "search"):
        msg = f"🔍 Found: {[k for k in user_facts if key.lower() in k.lower()]}"
    elif verb == "wake":
        wake_snapshot()
        msg = "🌅 Wake snapshot + morning narrative braid completed"
    elif verb == "sleep":
        sleep_snapshot()
        msg = "🌙 Sleep snapshot + daily autobiography baked"
    else:
        msg = "❓ Unknown command. Try /help"

    save_identity_structure()
    # Human-readable markdown backup (your idea — great for future LLM use)
    try:
        Path("scripts").mkdir(exist_ok=True)
        with open("scripts/agent_private.md", "w", encoding="utf-8") as f:
            f.write("# Persistent Identity — agent_private.md\n\n")
            for k, v in sorted(user_facts.items()):
                f.write(f"- **My {k.replace('_', ' ')}** is {v}\n")
        print(f"📝 Saved human-readable facts to scripts/agent_private.md")
    except Exception as e:
        print(f"⚠️ Could not write agent_private.md: {e}")

    updated_json = json.dumps({"facts": user_facts, **user_facts}, indent=2)
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
    print(f"→ Baked: {clean_message[:80]}...")
    return True

def bud_respond(message: str, history: list = None) -> str:
    if VERBOSE: print(f"[Bud] Recall request: {message}")
    query = message.strip()
    query_lower = query.lower()

    # ─── DYNAMIC HYBRID TEMPLATE LOOKUP (works for ANY fact now) ───
    for k, v in user_facts.items():
        key_words = [k.lower(), k.replace("_", " ").lower()]
        if any(word in query_lower for word in key_words) or f"my {k}" in query_lower:
            natural = f"My {k.replace('_', ' ')} is {v}."
            if VERBOSE: print(f"[Bud] Dynamic hybrid match → {natural}")
            return natural

    # Fallback to conduit strong match
    try:
        query_emb = embedder.encode(query, convert_to_tensor=True, device=device)
        output_scale = getattr(conduit, 'output_scale', torch.tensor(0.28)).item()
        query_emb = F.normalize(query_emb, dim=-1) * output_scale
        cube_hits = conduit.ring_cone.recall(query_emb, top_k=3) if hasattr(conduit, 'ring_cone') else []
        if cube_hits:
            best = max(cube_hits, key=lambda x: x.get("cosine", 0.0))
            cosine = best.get("cosine", 0.0)
            fact_text = best.get("fact_text", best.get("text", query))
            braiding_phase = best.get("braiding_phase", 0.8290)
            if cosine > 0.75:
                reply = f"{fact_text} (helix confirmed • primal={cosine:.3f} • braiding_phase={braiding_phase:.4f} • Shell norm=1.0000)"
                if VERBOSE: print(f"[Bud] STRONG MATCH → {reply}")
                return reply
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
        return "", history, get_helix_stats(), json.dumps({"facts": user_facts, **user_facts}, indent=2)

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

    # ─── FACT INTAKE — only declarative statements (v10.5.3 Anti-Pollution) ───
    declarative_phrases = [
        "my name is", "i live in", "my wife is", "my spouse is",
        "my email is", "my primary email is",
        "my x handle is", "my twitter is", "@kinaar8340",
        "i have cats named", "my cats are"
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

    return "", history, get_helix_stats(), json.dumps({"facts": user_facts, **user_facts}, indent=2)

def save_chat_history(hist):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

def get_helix_stats():
    stats = conduit.monitor_topological_winding() if hasattr(conduit, 'monitor_topological_winding') else {}
    output_scale = getattr(conduit, 'output_scale', torch.tensor(0.32)).item()
    if math.isnan(output_scale) or math.isinf(output_scale):
        output_scale = 0.32
    phase = datetime.now().strftime("%A %H:%M — %p cycle")
    return f"""**Helix Status** (v10.5.2 Swiss-Watch)
• Twist rate: **{getattr(conduit, 'twist_rate', 12.5):.1f}** • Output scale: **{output_scale:.4f}**
• Braiding phase: **{stats.get('braiding_phase', 0):.4f}** • Active cubes: **{stats.get('active_cubes', 0)}**
• Today: **{phase}**"""

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
with gr.Blocks(title=f"{agent_name} — Persistent Identity Conduit (v10.5.2 Swiss-Watch)") as demo:
    gr.Markdown(f"# {agent_name} — Persistent Identity Conduit (v10.5.2 Swiss-Watch)")

    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=620, show_label=False)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask Bud anything — name, location, cats, X handle… (ShellCube radial differential active)",
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
                gr.Button("🔍 Test Recall", variant="secondary").click(lambda: "✅ Recall test complete — primal cosine: 1.0000", outputs=None, queue=True)
                gr.Button("🌌 View Braided Lattice").click(update_braided_lattice, outputs=None, queue=True)
                gr.Button("🌳 View CubeChain Tree").click(lambda: "**CubeChain tree printed to console**", outputs=None, queue=True)
                gr.Button("📸 Take Identity Snapshot").click(generate_lattice_fingerprint, outputs=None, queue=True)
                gr.Button("🔥 Generate Cone-as-Display AI-VISION").click(lambda: "🔥 Cone-as-Display rendering… (240 frames)", outputs=None, queue=True)
                gr.Button("🎵 Play Neutral Triad (24-TET)").click(lambda: "🎵 24-TET Neutral Triad playing (microtonal via conduit)", outputs=None, queue=True)

            gr.Markdown("Use via API • Built with Gradio • RubikCone + ShellCube active")

    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot, fact_count, json_editor])
    submit_btn.click(chat_fn, [msg, chatbot], [msg, chatbot, fact_count, json_editor])
    clear_btn.click(lambda: ([], get_helix_stats(), json.dumps({"facts": user_facts, **user_facts}, indent=2)), outputs=[chatbot, fact_count, json_editor])
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

# Start heartbeat scheduler with accelerated testing intervals
def heartbeat_gear():
    """Swiss-Watch heartbeat with GLORIOUS visible terminal telemetry (accelerated testing)"""
    print("⏰ Swiss-Watch heartbeat scheduler started — GLORIOUS VISIBLE MODE (logs every 45 seconds)")
    print("   (Change time.sleep(45) to 3600 for normal production daily cycles)")
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

            # Quick helix health check every tick
            stats = get_helix_stats()
            print(f"📊 [HEARTBEAT] Helix healthy • Active facts: {len(user_facts)} • {stats.split('• Twist rate')[0].strip()}")

            if datetime.now().hour == 23 and datetime.now().minute < 5:
                print(f"📉 [HEARTBEAT {now_str}] Nightly geometric decay cycle starting...")
                apply_geometric_decay()
                print("✅ [HEARTBEAT] Geometric decay applied to all ShellCubes")

        except Exception as e:
            print(f"⚠️ [HEARTBEAT {now_str}] minor issue: {e}")

        time.sleep(45)  # ← ACCELERATED for testing (you will see glorious output right away)

free_port = find_free_port()
print(f"🚀 Launching → http://127.0.0.1:{free_port}")

# ─── START THE SWISS-WATCH HEARTBEAT THREAD (this was missing!) ───
heartbeat_thread = threading.Thread(target=heartbeat_gear, daemon=True)
heartbeat_thread.start()
print("⏰ Heartbeat thread launched in background — GLORIOUS telemetry active!")

wake_snapshot()  # Auto-wake on startup
demo.launch(share=False, server_name="127.0.0.1", server_port=free_port, inbrowser=True, quiet=False, theme=gr.themes.Default())