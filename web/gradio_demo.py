#!/usr/bin/env python3
"""QVPIC Hugging Face Space — QUARTZ AI SYNTHESIZER vintage control panel."""

from __future__ import annotations

import logging
import os
import traceback

import gradio as gr

from demo_core import (
    DEFAULT_QUERY_TEXT,
    GITHUB_URL,
    HF_SPACE_URL,
    VQC_URL,
    default_run_params,
    get_build_label,
    is_hf_space,
    run_benchmark_demo,
    run_query_recall,
    terminal_conduit_analogy,
)

logger = logging.getLogger(__name__)

_DEFAULTS = default_run_params()
_PHOSPHOR = "#00FF41"
_PANEL_BG = "#b8bcc4"

TOP_TABS: tuple[str, ...] = ("chat", "settings", "history", "memory", "tools")
TAB_LABELS: dict[str, str] = {
    "chat": "CHAT",
    "settings": "SETTINGS",
    "history": "HISTORY",
    "memory": "MEMORY",
    "tools": "TOOLS",
}

PROG_PROMPTS: dict[str, str] = {
    "prog1": DEFAULT_QUERY_TEXT,
    "prog2": "Explain the RubikCone identity conduit analogy",
    "prog3": "Run benchmark on demo facts",
    "prog4": "Show topology braiding and drift protection",
}

INITIAL_TERMINAL = "\n".join(
    [
        "> SYSTEM READY — QVPIC v10.2 CONDUIT",
        "> QUATERNION VORTEX · PERSISTENT IDENTITY · VQC",
        "> USER: Hello",
        "> QVPIC: How can I assist you with bake → recall → topology today?",
        "> _",
    ]
)

HELP_TEXT = "\n".join(
    [
        "> HELP — QUARTZ AI SYNTHESIZER / QVPIC",
        "> CHAT: terminal + command input",
        "> SETTINGS: tune bake steps, bandwidth, drift samples",
        "> MEMORY: run full benchmark (bake → recall → drift)",
        "> TOOLS: repo links and CLI pointers",
        "> NAV ENTER / SEND: submit command · EXEC: run recall/benchmark",
        "> PROG 1–4: load sample prompts into input",
        "> CLEAR: wipe terminal · MODE: toggle VQCEnhanced flag",
        f"> Repo: {GITHUB_URL}",
    ]
)

MODE_TEXT = "\n".join(
    [
        "> MODE — VQCEnhanced toggles experimental helical conduit.",
        "> Default: RubikConeConduit + RingConeChain (HF-safe).",
        "> Enable in SETTINGS panel, then EXEC to re-run.",
    ]
)


def _patch_gradio_client_bool_schema() -> None:
    try:
        from gradio_client import utils as client_utils

        if getattr(client_utils, "_qvp_bool_patch", False):
            return

        orig_get_type = client_utils.get_type

        def get_type(schema):  # noqa: ANN001
            if isinstance(schema, bool):
                return "boolean"
            return orig_get_type(schema)

        client_utils.get_type = get_type
        client_utils._qvp_bool_patch = True
    except Exception:
        logger.warning("Could not patch gradio_client", exc_info=True)


_patch_gradio_client_bool_schema()


def _default_ui_state() -> dict:
    return {
        "active_tab": "chat",
        "history": [],
        "cmd_index": -1,
        "last_cmd": "",
    }


def _append_terminal(terminal: str, *lines: str) -> str:
    base = terminal.rstrip()
    chunk = "\n".join(lines)
    return f"{base}\n{chunk}" if base else chunk


def _tab_btn_classes(active: str, tab_id: str) -> list[str]:
    classes = ["quartz-tab", f"quartz-tab-{tab_id}"]
    if tab_id == active:
        classes.append("quartz-tab-active")
    return classes


def _tab_updates(active: str) -> tuple:
    return tuple(
        gr.update(elem_classes=_tab_btn_classes(active, tab_id)) for tab_id in TOP_TABS
    )


def _metallic_btn(*extra: str) -> list[str]:
    return ["quartz-btn", *extra]


def _simulate_chat_response(message: str) -> str:
    lower = message.lower().strip()
    if any(word in lower for word in ("hello", "hi", "hey")):
        return (
            "> QVPIC: Conduit online. Try PROG 1 for a recall query, "
            "MEMORY for benchmark, or ask about quaternion identity."
        )
    if "benchmark" in lower:
        return "> QVPIC: Tap MEMORY or EXEC with 'benchmark' to run bake → recall metrics."
    if any(word in lower for word in ("conduit", "quaternion", "vortex", "rubik")):
        return "> QVPIC: " + terminal_conduit_analogy().replace("\n", "\n> ")
    if any(word in lower for word in ("help", "keypad", "prog")):
        return HELP_TEXT
    if "topology" in lower or "braid" in lower:
        return (
            "> QVPIC: ShellCube braiding_phase + geometric_winding shield "
            "persistent identity from flat-vector drift."
        )
    if "github" in lower or "repo" in lower:
        return f"> QVPIC: Source {GITHUB_URL} · Space {HF_SPACE_URL}"
    return (
        f"> QVPIC: Received '{message[:80]}'. "
        "Use EXEC to run recall on this text, or PROG keys for presets."
    )


def _make_tab_switch(tab_id: str):
    def handler(terminal: str, state: dict) -> tuple:
        return _switch_tab(tab_id, terminal, state)

    return handler


def _switch_tab(tab_id: str, terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state["active_tab"] = tab_id
    label = TAB_LABELS[tab_id]
    if tab_id == "history":
        hist = state.get("history") or []
        if hist:
            lines = ["> HISTORY — recent commands:"] + [f">   · {c}" for c in hist[-12:]]
            terminal = _append_terminal(terminal, *lines)
        else:
            terminal = _append_terminal(terminal, "> HISTORY: (empty)")
    elif tab_id == "memory":
        terminal = _append_terminal(
            terminal,
            "> MEMORY: press EXEC or wait — launching benchmark…",
        )
    elif tab_id == "tools":
        terminal = _append_terminal(
            terminal,
            "> TOOLS:",
            f">   GitHub  {GITHUB_URL}",
            f">   VQC     {VQC_URL}",
            f">   Space   {HF_SPACE_URL}",
            ">   Local agent: python scripts/main.py",
        )
    elif tab_id == "settings":
        terminal = _append_terminal(
            terminal,
            "> SETTINGS: tune dials below the terminal strip.",
        )
    else:
        terminal = _append_terminal(terminal, f"> TAB: {label}")
    show_settings = tab_id == "settings"
    return (
        terminal,
        state,
        gr.update(visible=show_settings),
        *_tab_updates(tab_id),
    )


def _push_history(state: dict, cmd: str) -> dict:
    state = dict(state) if state else _default_ui_state()
    cmd = cmd.strip()
    if not cmd:
        return state
    hist = list(state.get("history") or [])
    if not hist or hist[-1] != cmd:
        hist.append(cmd)
    state["history"] = hist[-50:]
    state["cmd_index"] = len(state["history"]) - 1
    state["last_cmd"] = cmd
    return state


def _handle_send(
    cmd: str,
    terminal: str,
    state: dict,
) -> tuple:
    state = dict(state) if state else _default_ui_state()
    cmd = (cmd or "").strip()
    if not cmd:
        return terminal, "", state
    state = _push_history(state, cmd)
    terminal = _append_terminal(terminal, f"> USER: {cmd}", _simulate_chat_response(cmd), "> _")
    return terminal, "", state


def _make_nav_handler(action: str):
    def handler(terminal: str, state: dict, cmd: str) -> tuple:
        return _handle_nav(action, terminal, state, cmd)

    return handler


def _handle_nav(
    action: str,
    terminal: str,
    state: dict,
    cmd: str,
) -> tuple:
    state = dict(state) if state else _default_ui_state()
    if action == "enter":
        return _handle_send(cmd or state.get("last_cmd", ""), terminal, state)
    if action == "up":
        terminal = _append_terminal(terminal, "> NAV: ▲ scroll up")
        return terminal, cmd, state
    if action == "down":
        terminal = _append_terminal(terminal, "> NAV: ▼ scroll down")
        return terminal, cmd, state
    hist: list[str] = list(state.get("history") or [])
    idx = int(state.get("cmd_index", -1))
    if action == "left" and hist:
        idx = max(0, idx - 1)
        state["cmd_index"] = idx
        cmd = hist[idx]
        terminal = _append_terminal(terminal, f"> NAV: ◀ recall command [{idx + 1}/{len(hist)}]")
        return terminal, cmd, state
    if action == "right" and hist:
        idx = min(len(hist) - 1, idx + 1)
        state["cmd_index"] = idx
        cmd = hist[idx]
        terminal = _append_terminal(terminal, f"> NAV: ▶ recall command [{idx + 1}/{len(hist)}]")
        return terminal, cmd, state
    terminal = _append_terminal(terminal, f"> NAV: {action}")
    return terminal, cmd, state


def _make_prog_handler(prog_id: str):
    def handler(terminal: str, state: dict) -> tuple:
        return _handle_prog(prog_id, terminal, state)

    return handler


def _handle_prog(
    prog_id: str,
    terminal: str,
    state: dict,
) -> tuple:
    prompt = PROG_PROMPTS.get(prog_id, "")
    terminal = _append_terminal(terminal, f"> PROG: loaded preset into input")
    return terminal, prompt, state


def _handle_clear(terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    return INITIAL_TERMINAL, state


def _handle_help(terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    return _append_terminal(terminal, HELP_TEXT, "> _"), state


def _handle_mode(terminal: str, state: dict, use_vqc: bool) -> tuple:
    state = dict(state) if state else _default_ui_state()
    terminal = _append_terminal(terminal, MODE_TEXT, f"> MODE: VQCEnhanced = {use_vqc}", "> _")
    return terminal, state


def _handle_exec(
    cmd: str,
    terminal: str,
    state: dict,
    bake_steps: float,
    bandwidth: float,
    use_vqc: bool,
    drift_samples: float,
    max_facts: float,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple:
    state = dict(state) if state else _default_ui_state()
    text = (cmd or state.get("last_cmd") or DEFAULT_QUERY_TEXT).strip()
    state = _push_history(state, text)
    run_benchmark = text.lower() in {"benchmark", "run benchmark", "exec benchmark"} or (
        "benchmark" in text.lower() and "recall" not in text.lower()
    )
    try:
        if run_benchmark or state.get("active_tab") == "memory":
            terminal = _append_terminal(
                terminal,
                "> EXEC: running benchmark (bake → recall → drift)…",
            )
            result = run_benchmark_demo(
                bake_steps=int(bake_steps),
                bandwidth=float(bandwidth),
                use_vqc=bool(use_vqc),
                drift_samples=int(drift_samples),
                max_facts=int(max_facts),
                include_lattice=False,
                progress_cb=progress,
            )
            body = result.metrics_text
            title = "BENCHMARK RESULTS"
        else:
            terminal = _append_terminal(terminal, f"> EXEC: query recall @ {text[:60]}…")
            body = run_query_recall(
                text,
                bake_steps=int(bake_steps),
                bandwidth=float(bandwidth),
                use_vqc=bool(use_vqc),
                max_facts=int(max_facts),
                progress_cb=progress,
            )
            title = "QUERY RECALL"
        wrapped = "\n".join(
            f"> {line}" if line.strip() else ">"
            for line in f"{title}\n{'─' * 40}\n{body}".splitlines()
        )
        terminal = _append_terminal(terminal, wrapped, "> _")
    except Exception as exc:
        logger.exception("exec failed")
        terminal = _append_terminal(
            terminal,
            f"> EXEC ERROR: {exc}",
            traceback.format_exc(),
            "> _",
        )
    return terminal, "", state


QUARTZ_HEAD = """
<script>
(function() {
    function fitPanel() {
        var h = window.innerHeight || document.documentElement.clientHeight || 0;
        if (window.visualViewport && window.visualViewport.height > 0) {
            h = window.visualViewport.height;
        }
        if (h < 1) return;
        document.documentElement.style.setProperty('--quartz-vh', h + 'px');
        document.body.style.overflow = 'hidden';
        var gc = document.querySelector('.gradio-container');
        if (gc) { gc.style.height = h + 'px'; gc.style.overflow = 'hidden'; }
        var chrome = 0;
        ['.quartz-top-tabs', '.quartz-prog-row', '.quartz-footer-wrap', '.quartz-settings'].forEach(function(sel) {
            var el = document.querySelector(sel);
            if (el) chrome += el.offsetHeight;
        });
        chrome += 36;
        document.documentElement.style.setProperty('--quartz-chrome', chrome + 'px');
        var ta = document.querySelector('.quartz-terminal textarea');
        if (ta) {
            var th = Math.max(120, h - chrome - 52);
            ta.style.height = th + 'px';
            ta.style.maxHeight = th + 'px';
        }
    }
    function scrollTerminal(delta) {
        var ta = document.querySelector('.quartz-terminal textarea');
        if (ta) ta.scrollTop += delta;
    }
    window.quartzScrollUp = function() { scrollTerminal(-48); };
    window.quartzScrollDown = function() { scrollTerminal(48); };
    function clickPulse(btn) {
        if (!btn) return;
        btn.classList.add('quartz-btn-pulse');
        setTimeout(function() { btn.classList.remove('quartz-btn-pulse'); }, 140);
    }
    document.addEventListener('click', function(e) {
        var b = e.target && e.target.closest('button.quartz-btn, button.quartz-tab');
        if (b) clickPulse(b);
    });
    fitPanel();
    window.addEventListener('resize', fitPanel);
    document.addEventListener('DOMContentLoaded', fitPanel);
    setTimeout(fitPanel, 200);
    setTimeout(fitPanel, 800);
})();
</script>
"""

QUARTZ_CSS = f"""
:root {{
    --quartz-phosphor: {_PHOSPHOR};
    --quartz-panel: {_PANEL_BG};
    --quartz-display: #0d0d0d;
    --quartz-btn-top: #e8eaee;
    --quartz-btn-mid: #a8adb8;
    --quartz-btn-bot: #6e7380;
    --quartz-border: #4a4f5c;
}}
html, body {{
    background: #1a1c22 !important;
    height: var(--quartz-vh, 100dvh) !important;
    overflow: hidden !important;
    margin: 0 !important;
}}
.gradio-container {{
    max-width: 100% !important;
    height: var(--quartz-vh, 100dvh) !important;
    padding: 0.35rem 0.5rem !important;
    background: #1a1c22 !important;
    overflow: hidden !important;
}}
.gradio-container .block, .gradio-container .form {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
.gradio-container .quartz-panel {{
    background: linear-gradient(180deg, #d4d8e0 0%, #b0b5c0 8%, #9aa0ac 92%, #7a808c 100%) !important;
    border: 2px solid #5c616d !important;
    border-radius: 6px !important;
    padding: 0.45rem 0.55rem 0.35rem !important;
    box-shadow:
        inset 0 2px 0 rgba(255,255,255,0.55),
        inset 0 -3px 8px rgba(0,0,0,0.25),
        0 8px 24px rgba(0,0,0,0.45) !important;
    height: calc(var(--quartz-vh, 100dvh) - 0.7rem) !important;
    max-height: calc(var(--quartz-vh, 100dvh) - 0.7rem) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}}
.gradio-container .quartz-top-tabs {{
    gap: 0.28rem !important;
    margin: 0 0 0.35rem 0 !important;
    flex-shrink: 0 !important;
}}
.gradio-container button.quartz-tab {{
    flex: 1 1 0 !important;
    min-height: clamp(1.65rem, 3.5vh, 2.1rem) !important;
    padding: 0.25rem 0.35rem !important;
    border: 1px solid var(--quartz-border) !important;
    border-radius: 4px 4px 0 0 !important;
    background: linear-gradient(180deg, var(--quartz-btn-top) 0%, var(--quartz-btn-mid) 55%, var(--quartz-btn-bot) 100%) !important;
    color: #2a2e36 !important;
    font-family: "Segoe UI", system-ui, sans-serif !important;
    font-size: clamp(0.52rem, 1.1vh, 0.68rem) !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.7),
        0 2px 4px rgba(0,0,0,0.35) !important;
    cursor: pointer !important;
}}
.gradio-container button.quartz-tab-active {{
    background: linear-gradient(180deg, #ffffff 0%, #d8dce6 45%, #b8bdc8 100%) !important;
    color: #101218 !important;
    box-shadow:
        inset 0 1px 0 #fff,
        0 0 12px rgba(255,255,255,0.45),
        0 2px 6px rgba(0,0,0,0.3) !important;
    border-bottom-color: #c8ccd6 !important;
}}
.gradio-container .quartz-main {{
    flex: 1 1 auto !important;
    min-height: 0 !important;
    gap: 0.35rem !important;
    align-items: stretch !important;
}}
.gradio-container .quartz-terminal-col {{
    position: relative !important;
    min-width: 0 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    background: #1a1a1a !important;
    border: 3px solid #080808 !important;
    border-radius: 4px !important;
    padding: 0.28rem 0.32rem 0.22rem !important;
    box-shadow: inset 0 0 18px rgba(0,0,0,0.85) !important;
}}
.gradio-container .quartz-terminal {{
    flex: 1 1 auto !important;
    min-height: 0 !important;
    margin: 0 !important;
}}
.gradio-container .quartz-terminal textarea {{
    background: var(--quartz-display) !important;
    color: var(--quartz-phosphor) !important;
    -webkit-text-fill-color: var(--quartz-phosphor) !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: clamp(0.58rem, 1.2vh, 0.72rem) !important;
    line-height: 1.38 !important;
    border: none !important;
    border-radius: 2px !important;
    box-shadow: inset 0 0 24px rgba(0,255,65,0.06) !important;
    text-shadow: 0 0 6px rgba(0,255,65,0.35) !important;
    resize: none !important;
    overflow-y: auto !important;
    height: calc(var(--quartz-vh, 100dvh) - var(--quartz-chrome, 220px) - 52px) !important;
    max-height: calc(var(--quartz-vh, 100dvh) - var(--quartz-chrome, 220px) - 52px) !important;
}}
.gradio-container .quartz-terminal-col::after {{
    content: "" !important;
    pointer-events: none !important;
    position: absolute !important;
    inset: 0 !important;
    background: repeating-linear-gradient(
        0deg,
        transparent 0px,
        transparent 2px,
        rgba(0,0,0,0.12) 2px,
        rgba(0,0,0,0.12) 3px
    ) !important;
    opacity: 0.35 !important;
    border-radius: 2px !important;
}}
.gradio-container .quartz-input-row {{
    gap: 0.28rem !important;
    margin: 0.22rem 0 0 0 !important;
    flex-shrink: 0 !important;
    align-items: stretch !important;
}}
.gradio-container .quartz-cmd-input input {{
    background: #0a0a0a !important;
    border: 1px solid #1a3a1a !important;
    border-radius: 3px !important;
    color: var(--quartz-phosphor) !important;
    -webkit-text-fill-color: var(--quartz-phosphor) !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: clamp(0.54rem, 1.1vh, 0.66rem) !important;
    min-height: clamp(1.45rem, 3vh, 1.85rem) !important;
    box-shadow: inset 0 0 10px rgba(0,255,65,0.08) !important;
}}
.gradio-container button.quartz-send {{
    flex: 0 0 clamp(3.2rem, 7vw, 4.2rem) !important;
    min-width: clamp(3.2rem, 7vw, 4.2rem) !important;
    color: var(--quartz-phosphor) !important;
    -webkit-text-fill-color: var(--quartz-phosphor) !important;
    font-size: clamp(0.48rem, 1vh, 0.58rem) !important;
    letter-spacing: 0.1em !important;
}}
.gradio-container .quartz-nav-col {{
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.22rem !important;
    padding: 0.15rem 0.1rem !important;
    min-width: clamp(5.5rem, 14vw, 8rem) !important;
}}
.gradio-container .quartz-nav-label {{
    color: #3a3e48 !important;
    font-size: clamp(0.48rem, 1vh, 0.58rem) !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    margin-bottom: 0.15rem !important;
    text-align: center !important;
    width: 100% !important;
}}
.gradio-container button.quartz-btn {{
    border: 1px solid var(--quartz-border) !important;
    border-radius: 4px !important;
    background: linear-gradient(180deg, var(--quartz-btn-top) 0%, var(--quartz-btn-mid) 50%, var(--quartz-btn-bot) 100%) !important;
    color: #2a2e36 !important;
    -webkit-text-fill-color: #2a2e36 !important;
    font-family: "Segoe UI", system-ui, sans-serif !important;
    font-weight: 700 !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.65),
        0 2px 4px rgba(0,0,0,0.35) !important;
    cursor: pointer !important;
    transition: filter 0.1s ease !important;
}}
.gradio-container button.quartz-btn:hover {{
    filter: brightness(1.06) !important;
}}
.gradio-container button.quartz-btn-pulse {{
    filter: brightness(0.92) !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.35) !important;
}}
.gradio-container button.quartz-nav {{
    width: clamp(3.2rem, 7.5vw, 4.5rem) !important;
    min-height: clamp(1.35rem, 2.8vh, 1.75rem) !important;
    font-size: clamp(0.42rem, 0.88vh, 0.52rem) !important;
    letter-spacing: 0.06em !important;
    white-space: pre-line !important;
    line-height: 1.15 !important;
    padding: 0.15rem 0.1rem !important;
}}
.gradio-container button.quartz-nav-mid {{
    width: clamp(2.8rem, 6.5vw, 3.8rem) !important;
}}
.gradio-container button.quartz-enter {{
    width: clamp(3.2rem, 7.5vw, 4.5rem) !important;
    min-height: clamp(1.55rem, 3.2vh, 2rem) !important;
    margin-top: 0.12rem !important;
    font-size: clamp(0.5rem, 1.05vh, 0.62rem) !important;
    letter-spacing: 0.14em !important;
}}
.gradio-container .quartz-nav-mid-row {{
    gap: 0.22rem !important;
    justify-content: center !important;
}}
.gradio-container .quartz-prog-row {{
    gap: 0.22rem !important;
    margin: 0.32rem 0 0.22rem 0 !important;
    flex-shrink: 0 !important;
}}
.gradio-container button.quartz-prog {{
    flex: 1 1 0 !important;
    min-width: 0 !important;
    min-height: clamp(1.35rem, 2.8vh, 1.7rem) !important;
    font-size: clamp(0.4rem, 0.82vh, 0.5rem) !important;
    letter-spacing: 0.05em !important;
    padding: 0.12rem 0.08rem !important;
    position: relative !important;
}}
.gradio-container button.quartz-prog-led::after {{
    content: "" !important;
    position: absolute !important;
    top: 0.18rem !important;
    right: 0.18rem !important;
    width: 5px !important;
    height: 5px !important;
    border-radius: 50% !important;
    background: #ff2222 !important;
    box-shadow: 0 0 6px rgba(255,40,40,0.8) !important;
}}
.gradio-container .quartz-footer-wrap {{
    flex-shrink: 0 !important;
    text-align: right !important;
    padding: 0.18rem 0.35rem 0.05rem 0 !important;
    border-top: 2px solid #5a5f6a !important;
    margin-top: 0.1rem !important;
}}
.gradio-container .quartz-footer {{
    color: #e8eaef !important;
    font-family: "Segoe UI", system-ui, sans-serif !important;
    font-size: clamp(0.44rem, 0.9vh, 0.54rem) !important;
    font-weight: 700 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    text-shadow: 0 1px 0 rgba(0,0,0,0.5) !important;
}}
.gradio-container .quartz-settings {{
    margin: 0.2rem 0 !important;
    padding: 0.28rem 0.35rem !important;
    background: rgba(0,0,0,0.12) !important;
    border: 1px solid #6a6f7a !important;
    border-radius: 4px !important;
}}
.gradio-container .quartz-settings input[type="range"] {{
    accent-color: var(--quartz-phosphor) !important;
}}
footer {{ visibility: hidden !important; }}
"""


def _build_theme() -> gr.themes.Base:
    return gr.themes.Base(primary_hue="neutral", neutral_hue="gray").set(
        body_background_fill="transparent",
        block_background_fill="transparent",
        button_primary_background_fill="#a8adb8",
        button_primary_text_color="#101218",
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="QVPIC — QUARTZ AI SYNTHESIZER",
        analytics_enabled=False,
        theme=_build_theme(),
        head=QUARTZ_HEAD,
        css=QUARTZ_CSS,
        fill_width=True,
    ) as demo:
        ui_state = gr.State(_default_ui_state())
        tab_btns: dict[str, gr.Button] = {}

        with gr.Column(elem_classes=["quartz-panel"]):
            with gr.Row(elem_classes=["quartz-top-tabs"]):
                for tab_id in TOP_TABS:
                    tab_btns[tab_id] = gr.Button(
                        TAB_LABELS[tab_id],
                        elem_classes=_tab_btn_classes("chat", tab_id),
                        variant="secondary",
                    )

            with gr.Row(elem_classes=["quartz-main"]):
                with gr.Column(scale=7, elem_classes=["quartz-terminal-col"]):
                    terminal = gr.Textbox(
                        value=INITIAL_TERMINAL,
                        label="Terminal",
                        show_label=False,
                        interactive=False,
                        lines=12,
                        max_lines=80,
                        elem_classes=["quartz-terminal"],
                    )
                    with gr.Row(elem_classes=["quartz-input-row"]):
                        cmd_input = gr.Textbox(
                            placeholder="Type command or message...",
                            show_label=False,
                            max_lines=1,
                            scale=5,
                            elem_classes=["quartz-cmd-input"],
                        )
                        send_btn = gr.Button(
                            "SEND",
                            scale=0,
                            elem_classes=_metallic_btn("quartz-send"),
                        )

                with gr.Column(scale=3, elem_classes=["quartz-nav-col"]):
                    gr.HTML('<div class="quartz-nav-label">NAV</div>')
                    nav_up = gr.Button(
                        "▲\nUP",
                        elem_classes=_metallic_btn("quartz-nav"),
                    )
                    with gr.Row(elem_classes=["quartz-nav-mid-row"]):
                        nav_left = gr.Button(
                            "◀\nLEFT",
                            elem_classes=_metallic_btn("quartz-nav", "quartz-nav-mid"),
                        )
                        nav_right = gr.Button(
                            "▶\nRIGHT",
                            elem_classes=_metallic_btn("quartz-nav", "quartz-nav-mid"),
                        )
                    nav_down = gr.Button(
                        "▼\nDOWN",
                        elem_classes=_metallic_btn("quartz-nav"),
                    )
                    nav_enter = gr.Button(
                        "ENTER",
                        elem_classes=_metallic_btn("quartz-nav", "quartz-enter"),
                    )

            with gr.Column(elem_classes=["quartz-settings"], visible=False) as settings_panel:
                with gr.Row():
                    bake_steps = gr.Slider(
                        10, 150, value=_DEFAULTS["bake_steps"], step=5, label="Bake steps"
                    )
                    bandwidth = gr.Slider(
                        0.1, 1.0, value=_DEFAULTS["bandwidth"], step=0.05, label="Bandwidth"
                    )
                    drift_samples = gr.Slider(
                        10, 80, value=_DEFAULTS["drift_samples"], step=5, label="Drift samples"
                    )
                    max_facts = gr.Slider(
                        3, 12, value=_DEFAULTS["max_facts"], step=1, label="Max facts"
                    )
                    use_vqc = gr.Checkbox(label="VQCEnhanced", value=_DEFAULTS["use_vqc"])

            with gr.Row(elem_classes=["quartz-prog-row"]):
                prog_btns = {}
                for index in range(1, 5):
                    prog_btns[f"prog{index}"] = gr.Button(
                        f"PROG {index}",
                        elem_classes=_metallic_btn("quartz-prog"),
                    )
                exec_btn = gr.Button(
                    "EXEC",
                    elem_classes=_metallic_btn("quartz-prog", "quartz-prog-led"),
                )
                clear_btn = gr.Button(
                    "CLEAR",
                    elem_classes=_metallic_btn("quartz-prog", "quartz-prog-led"),
                )
                mode_btn = gr.Button(
                    "MODE",
                    elem_classes=_metallic_btn("quartz-prog", "quartz-prog-led"),
                )
                help_btn = gr.Button(
                    "HELP",
                    elem_classes=_metallic_btn("quartz-prog", "quartz-prog-led"),
                )

            gr.HTML(
                '<div class="quartz-footer-wrap"><span class="quartz-footer">'
                "QUARTZ AI SYNTHESIZER</span></div>"
            )

        core_outputs = [terminal, cmd_input, ui_state]
        tune_inputs = [bake_steps, bandwidth, use_vqc, drift_samples, max_facts]

        tab_outputs = [terminal, ui_state, settings_panel, *tab_btns.values()]

        for tab_id, btn in tab_btns.items():
            evt = btn.click(
                _make_tab_switch(tab_id),
                inputs=[terminal, ui_state],
                outputs=tab_outputs,
            )
            if tab_id == "memory":
                evt.then(
                    _handle_exec,
                    inputs=[cmd_input, terminal, ui_state, *tune_inputs],
                    outputs=[terminal, cmd_input, ui_state],
                )

        send_btn.click(_handle_send, inputs=[cmd_input, terminal, ui_state], outputs=core_outputs)
        cmd_input.submit(_handle_send, inputs=[cmd_input, terminal, ui_state], outputs=core_outputs)

        nav_enter.click(
            _make_nav_handler("enter"),
            inputs=[terminal, ui_state, cmd_input],
            outputs=core_outputs,
        )
        nav_up.click(
            _make_nav_handler("up"),
            inputs=[terminal, ui_state, cmd_input],
            outputs=core_outputs,
            js="() => { window.quartzScrollUp && window.quartzScrollUp(); }",
        )
        nav_down.click(
            _make_nav_handler("down"),
            inputs=[terminal, ui_state, cmd_input],
            outputs=core_outputs,
            js="() => { window.quartzScrollDown && window.quartzScrollDown(); }",
        )
        nav_left.click(
            _make_nav_handler("left"),
            inputs=[terminal, ui_state, cmd_input],
            outputs=core_outputs,
        )
        nav_right.click(
            _make_nav_handler("right"),
            inputs=[terminal, ui_state, cmd_input],
            outputs=core_outputs,
        )

        for prog_id, btn in prog_btns.items():
            btn.click(
                _make_prog_handler(prog_id),
                inputs=[terminal, ui_state],
                outputs=[terminal, cmd_input, ui_state],
            )

        clear_btn.click(_handle_clear, inputs=[terminal, ui_state], outputs=[terminal, ui_state])
        help_btn.click(_handle_help, inputs=[terminal, ui_state], outputs=[terminal, ui_state])
        mode_btn.click(
            _handle_mode,
            inputs=[terminal, ui_state, use_vqc],
            outputs=[terminal, ui_state],
        )
        exec_btn.click(
            _handle_exec,
            inputs=[cmd_input, terminal, ui_state, *tune_inputs],
            outputs=[terminal, cmd_input, ui_state],
        )

    return demo


demo = build_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    on_hf = bool(os.environ.get("SPACE_ID"))
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        show_api=False,
        inbrowser=False,
        share=not on_hf,
    )


if __name__ == "__main__":
    main()