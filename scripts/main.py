#!/usr/bin/env python3
# scripts/main.py v10.8.2 Multi-File TV-Optimized Edition
# Thin launcher - keeps everything modular

import argparse
import socket
import threading
import time
import sys
import os
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent import (
    initialize_agent, HEARTBEAT_MINUTES, LLM_AVAILABLE, llm,
    chat_history, last_message_time, bake_narrative_braid,
    wake_snapshot, sleep_snapshot
)
from ui import create_ui, custom_theme

parser = argparse.ArgumentParser(description="QVPIC v10.8.2 ? TV-Optimized")
parser.add_argument('--name', type=str, default='Bud')
parser.add_argument('--no-reset', action='store_true')
parser.add_argument('--vqc', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--llm-strong', action='store_true')
parser.add_argument('--heartbeat-minutes', type=int, default=60)
args = parser.parse_args()

initialize_agent(args)   # passes args into agent module

def find_free_port(start=7860):
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
        port += 1


def heartbeat_gear():
    """Swiss-Watch heartbeat — now prints visibly every interval"""
    print(f"? Swiss-Watch heartbeat scheduler started (TV-optimized, interval = {HEARTBEAT_MINUTES} minutes)")

    global last_message_time

    while True:
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Always print a clear heartbeat line first
            print(f"\n[HEARTBEAT {now_str}] Running (interval: {HEARTBEAT_MINUTES} min)")

            # Your original logic (recent chat check + narrative bake)
            if (time.time() - last_message_time) < 1800:  # 30 minutes
                if LLM_AVAILABLE and len(chat_history) > 0:
                    recent = "\n".join([f"{m['role']}: {m['content'][:120]}" for m in chat_history[-6:]])
                    thread_summary = llm(
                        f"Create a short neutral summary of the recent conversation. "
                        f"Use ONLY facts from the chat. Keep it under 30 words.\n\nRecent chat:\n{recent}",
                        max_tokens=80, temperature=0.3, top_p=0.9
                    )["choices"][0]["text"].strip()
                    bake_narrative_braid(thread_summary, "hourly")
                    print(f"? [HEARTBEAT {now_str}] Grounded narrative baked")
                else:
                    print(f"   [HEARTBEAT {now_str}] No recent chat ? skipping")
            else:
                print(f"   [HEARTBEAT {now_str}] Idle ? skipping narrative bake")

        except Exception as e:
            print(f"?? [HEARTBEAT {now_str}] minor issue: {e}")

        time.sleep(HEARTBEAT_MINUTES * 60)

free_port = find_free_port()
print(f"? Launching QVPIC ? http://127.0.0.1:{free_port}")

heartbeat_thread = threading.Thread(target=heartbeat_gear, daemon=True)
heartbeat_thread.start()
print("? Heartbeat thread launched")

wake_snapshot()

demo = create_ui()
demo.launch(
    share=False,
    server_name="127.0.0.1",
    server_port=free_port,
    inbrowser=True,
    quiet=False,
    theme=custom_theme,
    css="""
    /* TV Mode - bright & high contrast (stops Samsung dimming) */
    .tv-mode, :root.tv-mode {
        --background-fill-primary: #f8f9fb;
        --background-fill-secondary: #ffffff;
        --border-color-primary: #2a2a2a;
        --text-color: #111111;
        --accent-color: #0066ff;
    }
    .tv-mode .chat-window, .tv-mode .gr-panel { background: #ffffff !important; color: #111111 !important; }
    .light-mode, :root.light-mode {
        --background-fill-primary: #f8f9fb;
        --background-fill-secondary: #f0f1f5;
        --border-color-primary: #c4c8d0;
        --text-color: #1f2128;
        --accent-color: #0066cc;
    }
    .dark-mode, :root.dark-mode {
        --background-fill-primary: #1c1c20;
        --background-fill-secondary: #27272b;
        --border-color-primary: #3f3f46;
        --text-color: #e2e2e8;
        --accent-color: #00d4ff;
    }
    .gradio-container { background: var(--background-fill-primary) !important; color: var(--text-color) !important; }
    .tab-nav button { color: var(--text-color) !important; background: transparent !important; }
    .tab-nav button.selected { background: var(--accent-color) !important; color: white !important; }
    .chat-window, .gr-panel, .gr-box { background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary) !important; color: var(--text-color) !important; }
    .input-bar { background: var(--background-fill-secondary) !important; }
    .gr-button { background: #333336 !important; color: var(--text-color) !important; }
    .light-mode .gr-button, .tv-mode .gr-button { background: #e5e7eb !important; color: #1f2128 !important; }
    """
)