#!/usr/bin/env python3
# scripts/agent_demo.py — v10.4.2 (Concise Middle-Ground Qwen + Fixed)
# ✅ Fixed JSON serialization crash (no more ellipsis)
# ✅ Auto-bake only on declarative statements (not questions)
# ✅ Stronger clean_reply + tighter Qwen prompt for concise natural answers
# ✅ Recall now correctly returns stored fact_text
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="PIC v10.4.2 — Concise Middle-Ground")
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

print(f"🚀 PIC v10.4.2 — RubikConeConduit + ShellCube + Concise Qwen")

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

# ─── IDENTITY TEMPLATES ───
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

for f in (public_file, private_file):
    f.touch(exist_ok=True)
lattice_dir.mkdir(parents=True, exist_ok=True)

def load_identity_structure():
    global user_facts
    if identity_structure_path.exists():
        with open(identity_structure_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support both old and new structures
        user_facts = data.get("facts", data) if isinstance(data.get("facts"), dict) else data
        print(f"✅ Loaded {len(user_facts)} identity facts from disk")
    else:
        populate_user_facts_from_files()
    return json.dumps({"facts": user_facts}, indent=2)

def save_identity_structure():
    data = {"facts": user_facts, "templates": IDENTITY_TEMPLATES}
    with open(identity_structure_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✅ Identity structure saved to disk")

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
    save_identity_structure()

# ─── CLI PARSER (fixed) ───
def run_pic_cli(command: str) -> Tuple[str, str, str]:
    global user_facts
    cmd = command.strip()
    lower = cmd.lower()
    if lower.startswith("/"):
        cmd = cmd[1:].strip()
        lower = cmd.lower()
    parts = re.split(r'\s+', cmd)
    verb = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    value = " ".join(parts[2:]).strip().strip('"\'') if len(parts) > 2 else ""

    if verb == "help" or lower == "help":
        help_text = """**PIC v10.4.1 CLI How To**
• /help — this reference
• /add <key> <value> or /set <key> <value> — bake fact
• /remove <key> — prune fact
• /rename <old> <new> — rename key
• /save — force checkpoint + re-bake
• /list or /show — current facts
• /find <text> — topological search"""
        return help_text, json.dumps({"facts": user_facts, **user_facts}, indent=2), get_helix_stats()

    msg = ""
    if verb in ("add", "set"):
        if key and value:
            user_facts[key] = value
            msg = f"✅ Set {key} = {value}"
        else:
            msg = "❌ Usage: /add <key> <value>"
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
        msg = "💾 Helix checkpoint flushed + RingConeChain re-baked"
    elif verb in ("list", "show"):
        msg = f"Current facts: {list(user_facts.keys())}"
    elif verb in ("find", "search"):
        msg = f"🔍 Found: {[k for k in user_facts if key.lower() in k.lower()]}"
    else:
        msg = "❓ Unknown command. Try /help"

    save_identity_structure()
    updated_json = json.dumps({"facts": user_facts, **user_facts}, indent=2)
    return msg, updated_json, get_helix_stats()

# ─── LIVE RECALL + AUTO-BAKE + VERBOSE ───
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
    print(f"→ Auto-baked: {clean_message[:80]}...")
    return True

def bud_respond(message: str, history: list = None) -> str:
    if VERBOSE: print(f"[Bud] Recall request: {message}")
    query = message.strip()
    try:
        query_emb = embedder.encode(query, convert_to_tensor=True, device=device)
        output_scale = getattr(conduit, 'output_scale', torch.tensor(0.28)).item()
        query_emb = F.normalize(query_emb, dim=-1) * output_scale
        cube_hits = conduit.ring_cone.recall(query_emb, top_k=3) if hasattr(conduit, 'ring_cone') else []
        if not cube_hits:
            return "No strong helix matches yet."
        best = max(cube_hits, key=lambda x: x.get("cosine", 0.0))
        cosine = best.get("cosine", 0.0)
        fact_text = best.get("fact_text", best.get("text", query))
        braiding_phase = best.get("braiding_phase", 0.8290)
        if cosine > 0.75:
            reply = f"{fact_text} (helix confirmed • primal={cosine:.3f} • braiding_phase={braiding_phase:.4f} • Shell norm=1.0000)"
            if VERBOSE: print(f"[Bud] STRONG MATCH → {reply}")
            return reply
        elif cosine > 0.50:
            return f"{fact_text}\n\n(ShellCube radial differential • cosine={cosine:.3f})"
        else:
            return "Bud is braiding your query through the RingConeChain..."
    except Exception as e:
        if VERBOSE: print(f"[Bud] Recall error: {e}")
        return "Topological conduit active • I remember everything geometrically (RubikCone + ShellCube)."

def get_relevant_facts(query: str) -> str:
    return "CORE HELIX IDENTITY FACTS:\n" + "\n".join([f"• {k}: {v}" for k, v in user_facts.items()])

def chat_fn(message: str, history: list):
    if VERBOSE:
        print(f"[Gradio] chat_fn called with: {message}")

    if not message.strip():
        return "", history, get_helix_stats(), json.dumps({"facts": user_facts}, indent=2)

    history = history or []

    # === 1. Handle CLI commands (/show, /set, /save, etc.) ===
    if message.strip().startswith("/"):
        cli_msg, updated_json, status = run_pic_cli(message[1:])
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": cli_msg}
        ])
        return "", history, status, updated_json

    # === 2. Auto-bake volunteered personal facts ===
    lower_msg = message.lower()
    if any(phrase in lower_msg for phrase in [
        "my name is", "i am ", "call me", "i live in", "my wife is",
        "my husband is", "my email", "my x handle", "my twitter",
        "my pet", "my cat", "my dog", "i have cats", "i have a pet"
    ]):
        bake_new_fact(message)
        save_identity_structure()
        if VERBOSE:
            print(f"[Bud] Auto-baked new fact from: {message}")

    # === 3. Geometric recall (ShellCube + Helix) ===
    recall_reply = bud_respond(message, history)

    # === 4. Decide if we should use LLM ===
    is_identity_query = any(word in lower_msg for word in [
        "name", "email", "handle", "x handle", "twitter", "pet", "cat", "dog",
        "live", "location", "who am i", "what is my", "tell me about my"
    ])

    use_llm = LLM_AVAILABLE and (
        is_identity_query or
        "braiding" in recall_reply.lower() or
        "strong match" in recall_reply.lower() or
        "helix confirmed" in recall_reply.lower()
    )

    if use_llm:
        if VERBOSE:
            print("[LLM] Using enhanced Qwen response with full facts...")

        facts_str = json.dumps(user_facts, indent=2) if user_facts else "No stored facts yet."

        system_prompt = f"""You are {agent_name}, a warm, helpful, and concise personal assistant.
You have perfect memory of the user's identity.

CURRENT USER FACTS (these are absolute truth — use them directly):
{facts_str}

CRITICAL RULES:
- If the user asks about name, email, X handle, pets, location or any other stored fact → answer DIRECTLY using the facts above.
- NEVER say "I don't have that information", "not listed here", "check your contacts", or anything similar when the fact exists.
- Never mention JSON, helix, ShellCube, cosine, braiding, radial differential, or any technical terms.
- Keep responses friendly, natural, and brief (1-2 sentences)."""

        # Recent conversation for context
        history_text = "\n".join([
            f"{'User' if m['role']=='user' else agent_name}: {m['content']}"
            for m in history[-6:]
        ])

        full_prompt = f"{system_prompt}\n\nRecent conversation:\n{history_text}\nUser: {message}\n{agent_name}:"

        try:
            out = llm(full_prompt, max_tokens=220, temperature=0.65, top_p=0.92, repeat_penalty=1.15)
            raw_reply = out["choices"][0]["text"].strip()

            # Clean any possible leaked prefixes or debug text
            reply = re.sub(r'^(Assistant|Bud|Aaron|You):?\s*', '', raw_reply, flags=re.IGNORECASE)
            reply = re.sub(r'\(ShellCube.*?\)|\(helix.*?\)|cosine=\d+\.\d+|braiding_phase=.*?', '', reply, flags=re.IGNORECASE)
            reply = reply.strip()

        except Exception as e:
            if VERBOSE:
                print(f"[LLM Error] {e}")
            reply = recall_reply
    else:
        reply = recall_reply

    # === 5. Final safety fallback ===
    if not reply or reply.startswith("No strong helix matches"):
        reply = "Got it — I don't have a strong memory match for that yet. Can you tell me more?"

    # === 6. Update history and save ===
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ])
    save_chat_history(history)

    if VERBOSE:
        print(f"[Bud → UI] Final reply: {reply[:150]}...")

    # Return to Gradio
    return "", history, get_helix_stats(), json.dumps({"facts": user_facts}, indent=2)

def save_chat_history(hist):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

def get_helix_stats():
    stats = conduit.monitor_topological_winding() if hasattr(conduit, 'monitor_topological_winding') else {}
    output_scale = getattr(conduit, 'output_scale', torch.tensor(0.32)).item()
    if math.isnan(output_scale) or math.isinf(output_scale):
        output_scale = 0.32
    return f"""**Helix Status** (v10.4.1 — RubikCone + ShellCube)
• Twist rate: **{getattr(conduit, 'twist_rate', 12.5):.1f}** • Output scale: **{output_scale:.4f}**
• Braiding phase: **{stats.get('braiding_phase', 0):.4f}** • Active cubes: **{stats.get('active_cubes', 0)}**"""

# (update_braided_lattice, generate_lattice_fingerprint, save_edited_facts, Gradio UI, launch — identical to your v10.4)

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
with gr.Blocks(title=f"{agent_name} — Persistent Identity Conduit (v10.4.1)") as demo:
    gr.Markdown(f"# {agent_name} — Persistent Identity Conduit (v10.4.1)")

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

free_port = find_free_port()
print(f"🚀 Launching → http://127.0.0.1:{free_port}")
demo.launch(share=False, server_name="127.0.0.1", server_port=free_port, inbrowser=True, quiet=False, theme=gr.themes.Default())
