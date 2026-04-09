#!/usr/bin/env python3
"""
~/qvpic/scripts/ui.py — Pure Gradio interface (v10.8.4 — Fixed for Gradio 6 + modular)
"""


import json
import time
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import socket
import threading

from pathlib import Path
from datetime import datetime


# Import ONLY what agent.py actually exports
from agent import (
    chat_fn, get_helix_stats, load_identity_structure,
    HEARTBEAT_MINUTES, LLM_AVAILABLE, llm, chat_history,
    last_message_time, bake_narrative_braid, wake_snapshot, sleep_snapshot,
    conduit, user_facts, USE_VQC, populate_system_facts   # ← added this
)

# For direct module access
import agent

custom_theme = gr.themes.Base(
    primary_hue="blue", secondary_hue="cyan", neutral_hue="slate",
    radius_size="md", text_size="lg", spacing_size="md"
).set(
    body_background_fill="#f8f9fb",
    body_background_fill_dark="#1c1c20"
)


def render_hyperbook_3d(path: str = "core/identity"):
    """Enhanced RingConeChain + high-quality 3D Braided Lattice with custom views, colors, and labels"""
    cleanup_snapshots()
    clean_path = (path or "root").strip("/")

    ring_cone = getattr(conduit, 'ring_cone', None)
    if ring_cone is None:
        fig = go.Figure()
        fig.add_annotation(text="RingConeChain not initialized", showarrow=False)
        return fig, None, None, None

    # RingConeChain 2D plot (unchanged - already looks great)
    path_hash = abs(hash(clean_path)) % getattr(ring_cone, 'TOTAL_CUBES', 216)
    ring_idx = path_hash % getattr(ring_cone, 'NUM_RINGS', 8)
    cube_local = path_hash % getattr(ring_cone.rings[ring_idx], 'num_cubes', 27) if hasattr(ring_cone, 'rings') else 0
    global_cube_idx = sum(r.num_cubes for r in getattr(ring_cone, 'rings', [])) + cube_local

    fig_ring = go.Figure()
    colors = ['#00ffff', '#39ff14', '#ff00ff', '#ffff00', '#ffaa00', '#00ffaa', '#aa00ff', '#ff5500']
    max_radius = 2.95

    for r_idx in range(min(8, getattr(ring_cone, 'NUM_RINGS', 8))):
        base_dot_size = 2.25 + r_idx * 1.05
        dot_size = base_dot_size * 0.68 if r_idx >= 5 else base_dot_size
        radius = 0.42 + r_idx * 0.22
        theta = np.linspace(0, 2 * np.pi, 200)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        color = colors[r_idx % len(colors)]
        opacity = 0.98 if r_idx == ring_idx else 0.78
        fig_ring.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=dot_size, color=color, opacity=opacity),
            name=f"Ring {r_idx}"
        ))

    fig_ring.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=38, color='#00ffff', line=dict(width=5, color='white')),
        text=[str(global_cube_idx)], textposition="middle center",
        textfont=dict(size=19, color="#0a0a0a")
    ))

    fig_ring.update_layout(
        title=f"Hyperbook Render — {clean_path} — Cube {global_cube_idx}",
        xaxis=dict(visible=False, range=[-max_radius, max_radius]),
        yaxis=dict(visible=False, range=[-max_radius, max_radius], scaleanchor="x", scaleratio=1),
        height=820,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#00ffcc',
        margin=dict(l=45, r=45, t=80, b=45)
    )

    # === IMPROVED BRAIDED LATTICE 3D VIEWS ===
    try:
        # Current View — angled perspective (best overall view)
        current_view = conduit.render_braided_lattice_style(
            save_path="snapshots/lattice_current.png",
            elev=25, azim=-45, title="Current View — Helical Braiding"
        )

        # Side / Front View — straight side-on
        side_view = conduit.render_braided_lattice_style(
            save_path="snapshots/lattice_side.png",
            elev=0, azim=0, title="Side / Front View"
        )

        # Top View — looking straight down
        top_view = conduit.render_braided_lattice_style(
            save_path="snapshots/lattice_top.png",
            elev=90, azim=0, title="Top View — Toroidal Projection"
        )
    except Exception as e:
        print(f"⚠️ Lattice render failed: {e}")
        current_view = side_view = top_view = None

    return fig_ring, current_view, side_view, top_view

def cleanup_snapshots():
    snapshot_dir = Path("snapshots")
    if not snapshot_dir.exists():
        return
    now = time.time()
    for f in snapshot_dir.glob("*.png"):
        if f.stat().st_mtime < now - 3600:
            try:
                f.unlink()
            except Exception:
                pass

def get_fresh_identity_json():
    """Always returns the latest identity structure as JSON for the UI"""
    return json.dumps(load_identity_structure(), indent=2, ensure_ascii=False)

def create_ui():
    with gr.Blocks(title=f"{agent.agent_name} — Quaternion Vortex Persistent Identity Conduit") as demo:
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(height=720, show_label=False, container=True, elem_classes="chat-window")
                with gr.Row(elem_classes="input-bar"):
                    msg = gr.Textbox(placeholder="Ask Bud anything — name, location, cats, X handle, weight?",
                                     container=False, scale=8, show_label=False, max_lines=8)
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", scale=1)

            with gr.Tab("Live Identity Tree"):
                fact_count = gr.Markdown("**Current identity structure:**")
                json_editor = gr.Code(
                    value=json.dumps(load_identity_structure(), indent=2),
                    language="json",
                    label="Raw JSON Editor",
                    lines=18
                )

            with gr.Tab("Hyperbook Render (3D RingConeChain)"):
                gr.Markdown("**Type any chapter path**")
                path_input = gr.Textbox(value="core/identity", label="Chapter Path")
                render_btn = gr.Button("Render in RingConeChain", variant="primary")
                hyperbook_3d = gr.Plot(label="Live RingConeChain")

            with gr.Tab("Focused Braided Lattice"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Current View**")
                        lattice_current = gr.Image(label="", height=380, show_label=False)
                    with gr.Column(scale=1):
                        gr.Markdown("**Side / Front View**")
                        lattice_side = gr.Image(label="", height=380, show_label=False)
                    with gr.Column(scale=1):
                        gr.Markdown("**Top View**")
                        lattice_top = gr.Image(label="", height=380, show_label=False)

            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    theme_selector = gr.Radio(
                        choices=["Dark", "Light", "TV Mode", "System"],
                        value="TV Mode",
                        label="UI Theme",
                        interactive=True
                    )
                    heartbeat_slider = gr.Slider(
                        minimum=5, maximum=120, value=HEARTBEAT_MINUTES,
                        step=5, label="Heartbeat Interval (minutes)"
                    )
                    vqc_toggle = gr.Checkbox(
                        value=USE_VQC,
                        label="Enable VQC-Enhanced Mode (OAM flux + Clifford Torus)"
                    )

                reset_btn = gr.Button("⚠️ Full Identity Reset (clear helix + facts)", variant="stop")
                status_md = gr.Markdown("")


        # Event bindings with forced refresh of Live Identity Tree after every action
        render_btn.click(render_hyperbook_3d, inputs=path_input,
                         outputs=[hyperbook_3d, lattice_current, lattice_side, lattice_top])

        # Chat events + immediate refresh of identity tree
        msg.submit(chat_fn, [msg, chatbot], [msg, chatbot, fact_count, json_editor]) \
            .then(get_fresh_identity_json, outputs=[json_editor])

        submit_btn.click(chat_fn, [msg, chatbot], [msg, chatbot, fact_count, json_editor]) \
            .then(get_fresh_identity_json, outputs=[json_editor])

        # Clear button also refreshes the identity tree
        clear_btn.click(lambda: ([], get_helix_stats(), get_fresh_identity_json()),
                        outputs=[chatbot, fact_count, json_editor])

    return demo
