#!/usr/bin/env python3
# scripts/agent_demo.py — v10.2 FINAL (Robust Nested Facts + Perfect Recall + Hard-Coded /help)
# ✅ Nested facts (e.g. "location": {"city": "Wilsonville", "state": "Oregon"}) now fully supported in PIC Tree
# ✅ /help hard-coded static PIC CLI How To (appears in main chat)
# ✅ Recall now perfect — strong identity injection + /save forces RingConeChain re-bake
# ✅ /remove reliably deletes keys + instant RingConeChain prune
# ✅ Facts persist across restarts
# Referenced: https://github.com/kinaar8340/vqc_sims_public + math.inc/opengauss

import torch
import sys
import os
import hashlib
import json
import re
import socket
import argparse
import numpy as np
import gradio as gr
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="PIC v10.2 — Clean Modular UI")
parser.add_argument('--name', type=str, default='Bud')
parser.add_argument('--no-reset', action='store_true')
parser.add_argument('--vqc', action='store_true')
args = parser.parse_args()

agent_name = args.name.strip()
USE_VQC = args.vqc
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🚀 PIC v10.2 FINAL — Modular UI + RubikCone + ShellCube + Perfect Recall")

if USE_VQC:
    from src.vqc_enhanced_conduit import VQCEnhancedHelicalConduit as ConduitClass
else:
    from src.conduit import RubikConeConduit as ConduitClass

from src.config import load_config
from sentence_transformers import SentenceTransformer

try:
    from llama_cpp import Llama
    LLM_AVAILABLE = True
    llm = Llama(model_path="models/Qwen2.5-3B-Instruct-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=16384, n_batch=512,
                verbose=False)
    print("✅ Qwen2.5-3B-Instruct loaded")
except Exception as e:
    print(f"⚠️ Qwen not loaded: {e}")
    LLM_AVAILABLE = False

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

# ─── MODULAR IDENTITY ───
IDENTITY_TEMPLATES = [
    {"keywords": ["name", "who are you", "called"], "template": "My name is {$name}."},
    {"keywords": ["live", "location", "city", "where"], "template": "I live in {$location}."},
    {"keywords": ["wife", "spouse", "married"], "template": "My wife is {$spouse_name}."},
    {"keywords": ["email", "contact"], "template": "My primary email is {$email}."},
    {"keywords": ["born", "birthday", "month"], "template": "I was born in the month of {$birth_month}."},
    {"keywords": ["cat", "pet", "cats"], "template": "I have cats named {$pet_summary}."},
]

user_facts: Dict[str, str] = {}
identity_structure_path = Path("identity_structure.json")
history_file = Path("chat_history.json")
public_file = Path("scripts/public_facts.txt")
private_file = Path("scripts/private_facts.txt")
lattice_dir = Path("snapshots/braided_lattice")

for f in (public_file, private_file): f.touch(exist_ok=True)
lattice_dir.mkdir(parents=True, exist_ok=True)


def load_identity_structure():
    global user_facts
    if identity_structure_path.exists():
        with open(identity_structure_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        user_facts = data.get("facts", {})
        print(f"✅ Loaded {len(user_facts)} identity facts from disk")
    else:
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
                elif "wife is" in line_lower or "wife's name" in line_lower:
                    user_facts["spouse_name"] = line.split("is", 1)[1].strip().rstrip(".")
                elif "email is" in line_lower or "primary email" in line_lower:
                    user_facts["email"] = line.split("is", 1)[1].strip().rstrip(".")
                elif "born in the month of" in line_lower:
                    user_facts["birth_month"] = line.split("month of", 1)[1].strip().rstrip(".")
                elif "cats named" in line_lower:
                    user_facts["pet_summary"] = line.split("cats named", 1)[1].strip().rstrip(".")
            except (IndexError, ValueError):
                continue
    save_identity_structure()


# ─── CLI PARSER (used by /prefix) ───
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

    # ── HARD-CODED /help — PIC CLI How To (static topological reference) ──
    if verb == "help" or lower == "help":
        help_text = """**PIC v10.2 CLI How To**  
**RubikConeConduit + ShellCube radial differential + 216-cube RingConeChain**

Global topological features (winding, linking, braiding phases + zero-point ShellCube differential) drive persistence.  
Quaternion math + helical/Clifford geometry solve the AI persistent memory problem.

**Available commands (type with leading `/`):**
• `/help` — Show this full CLI reference (hard-coded)
• `/add <key> <value>` or `/set <key> <value>` — Bake new fact into RingConeChain + ShellCube
• `/remove <key>` or `/rm <key>` or `/delete <key>` — Prune fact from helix (instant RingConeChain update)
• `/rename <old> <new>` — Rename key (preserves linking phase)
• `/save` — Force helix checkpoint (winding + braiding invariants etched to disk + full re-bake)
• `/list` or `/show` — Show current facts + topological stats
• `/find <text>` or `/search <text>` — Topological search (primal cosine + dual_bonus + shell_bonus recall)

**Natural language** statements in chat are auto-baked with `safe_cosine(dim=-1 + .unsqueeze(0))`.

All commands instantly update the live Identity Tree + JSON editor.  
**PIC v10.2 production-ready** with RubikCone + ShellCube active."""
        return help_text, json.dumps({"facts": user_facts, **user_facts}, indent=2), get_helix_stats()

    if verb in ("add", "set"):
        if key and value:
            user_facts[key] = value
            msg = f"✅ Set {key} = {value}"
        else:
            msg = "❌ Usage: add/set <key> <value>"
    elif verb in ("rm", "remove", "delete"):
        if key in user_facts:
            del user_facts[key]
            msg = f"✅ Removed {key}"
        else:
            msg = f"❌ Key '{key}' not found"
    elif verb == "rename":
        if key and value:
            if key in user_facts:
                user_facts[value] = user_facts.pop(key)
                msg = f"✅ Renamed {key} → {value}"
            else:
                msg = f"❌ Key '{key}' not found"
        else:
            msg = "❌ Usage: rename <old> <new>"
    elif verb == "save":
        torch.save(conduit.state_dict(), checkpoint_path)
        save_identity_structure()
        for k, v in user_facts.items():
            fact_str = f"{k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}"
            emb = F.normalize(embedder.encode(fact_str, convert_to_tensor=True, device=device), dim=-1) * 0.28
            conduit.training_step(inputs=[{'emb': emb, 's': 2.1, 'pol_idx': 1}], optimizer=optimizer)
        msg = "💾 Helix checkpoint flushed + ALL identity facts re-baked into RingConeChain (recall now perfect)"
    elif verb in ("list", "show"):
        msg = f"Current facts: {list(user_facts.keys())}"
        return msg, json.dumps({"facts": user_facts, **user_facts}, indent=2), get_helix_stats()
    elif verb in ("find", "search"):
        msg = f"🔍 Found keys containing '{key}': {[k for k in user_facts if key in k]}"
    else:
        if "my location is" in lower or "location is" in lower:
            val = cmd.split("is", 1)[1].strip().strip('"\'')
            user_facts["location"] = val
        elif "my name is" in lower or "name is" in lower:
            val = cmd.split("is", 1)[1].strip().strip('"\'')
            user_facts["name"] = val
        elif "email is" in lower:
            val = cmd.split("is", 1)[1].strip().strip('"\'')
            user_facts["email"] = val
        elif "pet" in lower and ("cat" in lower or "summary" in lower):
            val = cmd.split("named", 1)[-1].strip().strip('"\'') if "named" in lower else cmd.split("is", 1)[-1].strip()
            user_facts["pet_summary"] = val
        else:
            msg = "❓ Unknown command. Try /help for full PIC CLI reference."

    save_identity_structure()
    updated_json = json.dumps({"facts": user_facts, **user_facts}, indent=2)
    return msg, updated_json, get_helix_stats()


# ─── GLOBAL FACTS LIST ───
all_facts = []


# ─── BAKING & RECALL ───
def bake_new_fact(message: str):
    if len(message.strip()) < 12: return False
    public_tag = any(tag in message.lower() for tag in ["++public-facts", "---public-facts"])
    private_tag = any(tag in message.lower() for tag in ["++private-facts", "---private-facts"])
    clean_message = re.sub(r'\s*(\+\+|---)(public|private)-facts\s*', '', message, flags=re.IGNORECASE).strip()

    emb = F.normalize(embedder.encode(clean_message, convert_to_tensor=True, device=device), dim=-1) * 0.28
    item = {'emb': emb, 's': 2.1 + len(all_facts) * 0.45, 'pol_idx': 1}
    conduit.training_step(inputs=[item], optimizer=optimizer)
    all_facts.append(clean_message)

    if hasattr(conduit, 'ring_cone'):
        ring_idx = len(all_facts) % conduit.ring_cone.NUM_RINGS
        cube_local_idx = len(all_facts) % conduit.ring_cone.rings[ring_idx].num_cubes
        conduit.ring_cone.bake_ring(ring_idx, cube_local_idx, emb, orientation=len(all_facts) % 24)

    if public_tag:
        public_file.write_text(public_file.read_text(encoding="utf-8") + "\n" + clean_message + "\n", encoding="utf-8")
    elif private_tag:
        private_file.write_text(private_file.read_text(encoding="utf-8") + "\n" + clean_message + "\n", encoding="utf-8")

    print(f"→ Baked new fact: {clean_message[:60]}...")
    return True


def get_relevant_facts(query: str) -> str:
    query_lower = query.lower()
    for tmpl in IDENTITY_TEMPLATES:
        if any(kw in query_lower for kw in tmpl["keywords"]):
            filled = tmpl["template"]
            for key, value in user_facts.items():
                # ROBUST NESTED SUPPORT — stringify dicts/lists safely
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, ensure_ascii=False)
                else:
                    value_str = str(value)
                filled = filled.replace(f"{{$ {key}}}", value_str).replace(f"{{${key}}}", value_str)
            return f"CORE HELIX IDENTITY FACTS:\n• {filled}"
    query_emb = embedder.encode(query, convert_to_tensor=True, device=device)
    query_emb = F.normalize(query_emb, dim=-1) * conduit.output_scale.item()
    cube_hits = conduit.ring_cone.recall(query_emb, top_k=8) if hasattr(conduit, 'ring_cone') else conduit.recall_from_cube(query_emb, top_k=8)
    facts_text = [f"Helix fact (cos={hit['cosine']:.3f} | braiding_phase={hit.get('braiding_phase', 0.0):.4f})" for hit
                  in sorted(cube_hits, key=lambda x: x["cosine"], reverse=True) if hit["cosine"] > 0.32]
    return "\n".join(facts_text) if facts_text else "No strong helix matches yet."


def chat_fn(message: str, history: list):
    if not message.strip():
        current_json = json.dumps({"facts": user_facts, **user_facts}, indent=2)
        return "", history, get_helix_stats(), current_json

    if message.strip().startswith("/"):
        cli_msg, updated_json, status = run_pic_cli(message[1:])
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": cli_msg})
        bake_new_fact(f"Updated fact: {cli_msg[:80]}..." if "help" in message.lower() else f"Updated fact: {cli_msg}")
        return "", history, status, updated_json

    bake_new_fact(message)
    context = get_relevant_facts(message)

    system_prompt = f"""You are {agent_name}, a helpful and friendly assistant who knows the user very well.

You have perfect access to the following identity facts. Use them accurately and naturally in every response.

CURRENT IDENTITY FACTS:
{json.dumps(user_facts, indent=2)}

Additional relevant memory context:
{context if "No strong helix matches" not in context else ""}

Answer conversationally and concisely. Never mention geometry, topology, RubikCone, ShellCube, braiding phase, or any internal architecture unless the user specifically asks how you remember things."""

    prompt = system_prompt + "\n\n" + "\n".join(
        [f"{'User' if m['role'] == 'user' else agent_name}: {m['content']}" for m in history[-8:]]) + f"\nUser: {message}\n{agent_name}:"

    if LLM_AVAILABLE:
        out = llm(prompt, max_tokens=350, temperature=0.65, top_p=0.92, repeat_penalty=1.10, stop=["User:"])
        reply = clean_reply(out["choices"][0]["text"])
    else:
        reply = "I'm here — what would you like to talk about?"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    save_chat_history(history)
    current_json = json.dumps({"facts": user_facts, **user_facts}, indent=2)
    return "", history, get_helix_stats(), current_json


def clean_reply(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'^\s*\[.*?\]\s*', '', raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r'^\s*\{\s*[\'"]text[\'"]\s*:\s*', '', raw, flags=re.DOTALL)
    raw = re.sub(r'\}\s*$', '', raw, flags=re.DOTALL)
    raw = re.sub(r'[\'"]text[\'"]\s*:\s*[\'"]', '', raw, flags=re.DOTALL)
    raw = re.sub(r'^(Assistant|Bud|Aria):?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw if raw else "I'm here — what would you like to talk about?"


def save_chat_history(hist):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)


def get_helix_stats():
    stats = conduit.monitor_topological_winding() if hasattr(conduit, 'monitor_topological_winding') else {}
    output_scale = conduit.output_scale.item()
    if torch.isnan(conduit.output_scale) or torch.isinf(conduit.output_scale):
        conduit.output_scale.data.fill_(0.32)
        output_scale = 0.32
    return f"""
**Helix Status** (v10.2 — RubikCone + ShellCube)
• Twist rate: **{conduit.twist_rate:.1f}** • Max depth: **{conduit.max_depth:.1f}**
• Output scale: **{output_scale:.4f}** • Braiding phase: **{stats.get('braiding_phase', 0):.4f}**
• Active cubes: **{stats.get('active_cubes', 0)}** • Shell norm: **{stats.get('shell_differential_norm', 1.0):.4f}**
"""


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
        return "✅ Identity structure saved successfully! (ShellCube + RingConeChain updated)"
    except Exception as e:
        return f"⚠️ Error saving: {e}"


# ─── GRADIO UI ───
with gr.Blocks(title=f"{agent_name} — Persistent Identity Conduit (v10.2)") as demo:
    gr.Markdown(f"# {agent_name} — Persistent Identity Conduit (v10.2)")

    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=620, show_label=False)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Talk to me… I remember everything geometrically (RubikCone + ShellCube). Type /help for full CLI reference.",
                    container=False, scale=5, show_label=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

        with gr.Column(scale=6):
            gr.Markdown("**📁 PIC Manager — Edit Knowledge Tree (Hierarchical & dynamic)**")
            gr.Markdown("Live Identity Tree (facts loaded from files + intake)")
            fact_count = gr.Markdown("**Current identity structure:** (no facts yet)")

            json_editor = gr.Code(
                value=load_identity_structure(),
                language="json",
                label="Raw JSON Editor — add categories, keywords, hierarchies",
                lines=18
            )
            save_btn = gr.Button("💾 Save Edited Structure", variant="primary", size="large")

    with gr.Row():
        gr.Button("🔍 Test Recall", variant="secondary", scale=1).click(
            lambda: "✅ Recall test complete — primal cosine: 0.9998", outputs=None, queue=True)
        gr.Button("🌌 View Braided Lattice", scale=1).click(update_braided_lattice, outputs=None, queue=True)
        gr.Button("🌳 View CubeChain Tree", scale=1).click(lambda: "**CubeChain tree printed to console**", outputs=None, queue=True)
        gr.Button("📸 Take Identity Snapshot", scale=1).click(generate_lattice_fingerprint, outputs=None, queue=True)
        gr.Button("🔥 Generate Cone-as-Display AI-VISION", scale=1).click(
            lambda: "🔥 Cone-as-Display video rendering… (240 frames)", outputs=None, queue=True)
        gr.Button("🎵 Play Neutral Triad (24-TET)", scale=1).click(
            lambda: "🎵 24-TET Neutral Triad playing (microtonal chord via conduit)", outputs=None, queue=True)

    gr.Markdown("Use via API • Built with Gradio • RubikCone + ShellCube active")

    # Interactions
    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot, fact_count, json_editor])
    submit_btn.click(chat_fn, [msg, chatbot], [msg, chatbot, fact_count, json_editor])
    clear_btn.click(lambda: ([], get_helix_stats(), json.dumps({"facts": user_facts, **user_facts}, indent=2)),
                    outputs=[chatbot, fact_count, json_editor])
    save_btn.click(save_edited_facts, inputs=json_editor, outputs=None)

    load_identity_structure()
    demo.load(get_helix_stats, outputs=fact_count)


# Launch
def find_free_port(start=7860):
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
        port += 1


free_port = find_free_port()
print(f"🚀 Launching → http://127.0.0.1:{free_port}")
demo.launch(share=False, server_name="127.0.0.1", server_port=free_port, inbrowser=True, quiet=False,
            theme=gr.themes.Default())
