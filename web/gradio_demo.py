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
_PANEL_BG = "#1a1d22"

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

ACTIVE_SKIN = "quartz-default"

GRID_LAYER_HTML = """
<div class="quartz-grid-layer" aria-hidden="true"></div>
"""

DISPLAY_BACKING_HTML = """
<div class="quartz-display-backing" aria-hidden="true"></div>
"""

TORUS_STATE_BRIDGE_HTML = (
    '<div class="quartz-torus-bridge" data-echo="0" data-nudge="0" hidden></div>'
)

PANEL_SKIN_HTML = """
<div class="quartz-skin quartz-skin--quartz-default" data-skin="quartz-default" aria-hidden="true">
  <div class="quartz-skin-case"></div>
  <div class="quartz-skin-tab-rail"></div>
  <div class="quartz-skin-display-frame"></div>
  <div class="quartz-skin-prog-tray"></div>
  <div class="quartz-skin-footer-strip"></div>
</div>
"""

PANEL_SKINS: dict[str, str] = {
    "quartz-default": PANEL_SKIN_HTML,
}


def _panel_skin_html() -> str:
    return PANEL_SKINS.get(ACTIVE_SKIN, PANEL_SKIN_HTML)

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
        "> CHAT: toggle four-panel grid view on/off",
        "> SETTINGS: tune bake steps, bandwidth, drift samples",
        "> MEMORY: run full benchmark (bake → recall → drift)",
        "> TOOLS: repo links and CLI pointers",
        "> SEND / Enter key: submit command · EXEC: run recall/benchmark",
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
        "grid_view": True,
        "history": [],
        "cmd_index": -1,
        "last_cmd": "",
        "torus_echo": 0.0,
        "torus_nudge": 0,
    }


def _touch_torus(state: dict, strength: float = 1.0) -> dict:
    state = dict(state) if state else _default_ui_state()
    state["torus_echo"] = round(float(state.get("torus_echo", 0.0)) + strength * 0.618034, 6)
    state["torus_nudge"] = int(state.get("torus_nudge", 0)) + 1
    return state


def _torus_bridge_html(state: dict) -> str:
    echo = float(state.get("torus_echo", 0.0))
    nudge = int(state.get("torus_nudge", 0))
    return (
        f'<div class="quartz-torus-bridge" data-echo="{echo}" '
        f'data-nudge="{nudge}" hidden></div>'
    )


def _root_classes(grid_on: bool) -> list[str]:
    return ["quartz-root", "quartz-grid-on" if grid_on else "quartz-grid-off"]


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


def _handle_grid_toggle(terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state = _touch_torus(state, 0.35)
    grid_on = not bool(state.get("grid_view", True))
    state["grid_view"] = grid_on
    state["active_tab"] = "chat"
    label = "ON — four-panel grid visible" if grid_on else "OFF — solid display"
    terminal = _append_terminal(terminal, f"> GRID VIEW: {label}")
    return (
        terminal,
        state,
        gr.update(visible=False),
        *_tab_updates("chat"),
        gr.update(elem_classes=_root_classes(grid_on)),
        gr.update(value=_torus_bridge_html(state)),
    )


def _switch_tab(tab_id: str, terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state = _touch_torus(state, 0.25)
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
    grid_on = bool(state.get("grid_view", True))
    return (
        terminal,
        state,
        gr.update(visible=show_settings),
        *_tab_updates(tab_id),
        gr.update(elem_classes=_root_classes(grid_on)),
        gr.update(value=_torus_bridge_html(state)),
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
        return terminal, "", state, gr.update(value=_torus_bridge_html(state))
    state = _push_history(state, cmd)
    state = _touch_torus(state, 1.0)
    terminal = _append_terminal(terminal, f"> USER: {cmd}", _simulate_chat_response(cmd), "> _")
    return terminal, "", state, gr.update(value=_torus_bridge_html(state))


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
    state = _touch_torus(state, 0.45)
    terminal = _append_terminal(terminal, f"> PROG: loaded preset into input")
    return terminal, prompt, state, gr.update(value=_torus_bridge_html(state))


def _handle_clear(terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state = _touch_torus(state, 0.3)
    return INITIAL_TERMINAL, state, gr.update(value=_torus_bridge_html(state))


def _handle_help(terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state = _touch_torus(state, 0.2)
    return _append_terminal(terminal, HELP_TEXT, "> _"), state, gr.update(value=_torus_bridge_html(state))


def _handle_mode(terminal: str, state: dict, use_vqc: bool) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state = _touch_torus(state, 0.55)
    terminal = _append_terminal(terminal, MODE_TEXT, f"> MODE: VQCEnhanced = {use_vqc}", "> _")
    return terminal, state, gr.update(value=_torus_bridge_html(state))


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
    state = _touch_torus(state, 1.35)
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
    return terminal, "", state, gr.update(value=_torus_bridge_html(state))


QUARTZ_HEAD = """
<script>
(function() {
    function whenBodyReady(fn) {
        if (document.body) {
            fn();
            return;
        }
        document.addEventListener('DOMContentLoaded', fn, { once: true });
    }
    function torusMountRoot() {
        return document.body || null;
    }
    var quartzResizeTimer = null;
    function debouncedFitPanel() {
        if (quartzResizeTimer) clearTimeout(quartzResizeTimer);
        quartzResizeTimer = setTimeout(fitPanel, 80);
    }
    function clampViewport() {
        var vw = window.innerWidth || document.documentElement.clientWidth || 0;
        var vh = window.innerHeight || document.documentElement.clientHeight || 0;
        if (vw < 1) return;
        document.documentElement.style.setProperty('--quartz-vw', vw + 'px');
        if (document.body) {
            document.body.style.maxWidth = vw + 'px';
            document.body.style.overflowX = 'hidden';
            document.body.style.overflowY = 'hidden';
        }
        var gc = document.querySelector('.gradio-container');
        if (gc) {
            gc.style.maxWidth = vw + 'px';
            gc.style.width = '100%';
            gc.style.overflowX = 'hidden';
            if (vh > 0) gc.style.height = vh + 'px';
            gc.style.overflowY = 'hidden';
        }
        ['.quartz-root', '.quartz-panel', '.quartz-content'].forEach(function(sel) {
            var el = document.querySelector(sel);
            if (!el) return;
            el.style.maxWidth = '100%';
            el.style.overflowX = 'hidden';
        });
    }
    function terminalOutputRect() {
        var ta = document.querySelector('.quartz-terminal textarea');
        return ta ? ta.getBoundingClientRect() : null;
    }
    function fitPanel() {
        var h = window.innerHeight || document.documentElement.clientHeight || 0;
        if (window.visualViewport && window.visualViewport.height > 0) {
            h = window.visualViewport.height;
        }
        if (h < 1) return;
        clampViewport();
        document.documentElement.style.setProperty('--quartz-vh', h + 'px');
        if (document.body) document.body.style.overflow = 'hidden';
        var gc = document.querySelector('.gradio-container');
        if (gc) { gc.style.height = h + 'px'; gc.style.overflow = 'hidden'; }
        var chrome = 0;
        ['.quartz-top-tabs', '.quartz-prog-row', '.quartz-footer-wrap', '.quartz-settings'].forEach(function(sel) {
            var el = document.querySelector(sel);
            if (el) chrome += el.offsetHeight;
        });
        chrome += 36;
        var bay = document.querySelector('.quartz-display-bay');
        if (bay) {
            var rs = getComputedStyle(bay);
            chrome += parseFloat(rs.marginTop) + parseFloat(rs.marginBottom);
        }
        document.documentElement.style.setProperty('--quartz-chrome', chrome + 'px');
        window.quartzChromePx = chrome;
        syncTerminalOverflow(h);
        alignGridWindow();
        fitSkinMetrics();
        fitDisplayBacking();
        ensureTorusStage();
        fitTorusMount();
    }
    function ensureTorusStage() {
        var root = torusMountRoot();
        if (!root) return null;
        var stage = document.querySelector('.quartz-torus-stage');
        if (!stage) {
            stage = document.createElement('div');
            stage.className = 'quartz-torus-stage';
            stage.setAttribute('aria-hidden', 'true');
            stage.innerHTML = ''
                + '<div class="quartz-torus-frame">'
                + '<svg class="quartz-torus-svg" viewBox="0 0 220 220" role="img" aria-label="Geodesic torus conduit">'
                + '<circle class="quartz-torus-orbit" cx="110" cy="110" r="102"></circle>'
                + '<g class="quartz-torus-mesh"></g>'
                + '<rect class="quartz-torus-core" x="106" y="106" width="8" height="8"></rect>'
                + '</svg></div>';
            root.appendChild(stage);
        }
        return stage.querySelector('.quartz-torus-mesh');
    }
    function lockTerminalOverflowX() {
        [
            '.quartz-display-bay',
            '.quartz-terminal-col',
            '.quartz-terminal',
            '.quartz-terminal .wrap',
            '.quartz-terminal textarea'
        ].forEach(function(sel) {
            document.querySelectorAll(sel).forEach(function(el) {
                el.style.overflowX = 'hidden';
                el.style.maxWidth = '100%';
            });
        });
    }
    function syncTerminalOverflow(viewportH) {
        var ta = document.querySelector('.quartz-terminal textarea');
        if (!ta) return;
        var inputRow = document.querySelector('.quartz-input-row');
        var inputH = inputRow ? inputRow.offsetHeight : 52;
        var th = Math.max(120, viewportH - (window.quartzChromePx || 220) - inputH - 8);
        ta.style.height = th + 'px';
        ta.style.maxHeight = th + 'px';
        ta.style.overflowX = 'hidden';
        ta.style.overflowY = ta.scrollHeight > ta.clientHeight + 2 ? 'auto' : 'hidden';
        lockTerminalOverflowX();
    }
    function fitTorusFrame() {
        var mount = document.querySelector('.quartz-torus-stage');
        var frame = mount && mount.querySelector('.quartz-torus-frame');
        if (!mount || !frame) return;
        var w = mount.clientWidth;
        var h = mount.clientHeight;
        if (w < 40 || h < 40) return;
        var pad = Math.max(6, Math.round(Math.min(w, h) * 0.045));
        var rightX0 = w * 0.5 + pad * 0.5;
        var rightW = Math.max(20, w * 0.5 - pad * 1.5);
        var availH = Math.max(20, h - pad * 2);
        var size = Math.floor(Math.min(rightW, availH) * 0.82);
        var originX = rightX0 + rightW * 0.5;
        var originY = h * 0.5;
        frame.style.position = 'absolute';
        frame.style.left = Math.round(originX - size * 0.5) + 'px';
        frame.style.top = Math.round(originY - size * 0.5) + 'px';
        frame.style.width = size + 'px';
        frame.style.height = size + 'px';
        frame.style.right = 'auto';
        frame.style.transform = 'none';
        frame.style.display = 'flex';
        frame.style.alignItems = 'center';
        frame.style.justifyContent = 'center';
    }
    function torusMeshScale() {
        var frame = document.querySelector('.quartz-torus-stage .quartz-torus-frame');
        if (!frame) return 58;
        var side = Math.min(frame.clientWidth, frame.clientHeight);
        if (side < 1) return 58;
        return Math.max(38, Math.min(76, side * 0.28));
    }
    function fitTorusMount() {
        var mount = document.querySelector('.quartz-torus-stage');
        var tr = terminalOutputRect();
        if (!mount || !tr) {
            if (mount) mount.style.display = 'none';
            return;
        }
        if (tr.width < 40 || tr.height < 40) {
            mount.style.display = 'none';
            return;
        }
        mount.style.display = 'block';
        mount.style.position = 'fixed';
        mount.style.top = Math.round(tr.top) + 'px';
        mount.style.left = Math.round(tr.left) + 'px';
        mount.style.width = Math.round(tr.width) + 'px';
        mount.style.height = Math.round(tr.height) + 'px';
        mount.style.zIndex = '99999';
        mount.style.overflow = 'hidden';
        fitTorusFrame();
    }
    function fitDisplayBacking() {
        var backing = document.querySelector('.quartz-display-backing');
        var ta = document.querySelector('.quartz-terminal textarea');
        var col = document.querySelector('.quartz-terminal-col');
        if (!backing || !ta || !col) return;
        var cr = col.getBoundingClientRect();
        var tr = ta.getBoundingClientRect();
        backing.style.top = (tr.top - cr.top) + 'px';
        backing.style.left = (tr.left - cr.left) + 'px';
        backing.style.width = tr.width + 'px';
        backing.style.height = tr.height + 'px';
    }
    function fitSkinMetrics() {
        var tabs = document.querySelector('.quartz-top-tabs');
        var prog = document.querySelector('.quartz-prog-row');
        var footer = document.querySelector('.quartz-footer-wrap');
        var settings = document.querySelector('.quartz-settings');
        if (tabs) {
            document.documentElement.style.setProperty(
                '--quartz-skin-tab-h',
                tabs.offsetHeight + 'px'
            );
        }
        var progH = prog ? prog.offsetHeight : 0;
        var footerH = footer ? footer.offsetHeight : 0;
        var settingsH = settings && settings.offsetParent ? settings.offsetHeight : 0;
        document.documentElement.style.setProperty('--quartz-skin-prog-h', progH + 'px');
        document.documentElement.style.setProperty('--quartz-skin-footer-h', footerH + 'px');
        document.documentElement.style.setProperty('--quartz-skin-settings-h', settingsH + 'px');
    }
    function alignGridWindow() {
        var root = document.querySelector('.quartz-grid-on');
        var aperture = document.querySelector('.quartz-terminal-col');
        if (!root || !aperture) {
            document.documentElement.style.removeProperty('--quartz-grid-x');
            document.documentElement.style.removeProperty('--quartz-grid-y');
            return;
        }
        var rect = aperture.getBoundingClientRect();
        document.documentElement.style.setProperty('--quartz-grid-x', Math.round(-rect.left) + 'px');
        document.documentElement.style.setProperty('--quartz-grid-y', Math.round(-rect.top) + 'px');
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
    function bootQuartzTorus() {
        var PHI = 1.61803398875;
        var mesh = ensureTorusStage();
        if (!mesh) {
            setTimeout(bootQuartzTorus, 220);
            return;
        }
        if (window.quartzTorus && window.quartzTorus.ready) return;
        var state = {
            rotX: 0.62,
            rotY: 0.0,
            rotZ: 0.18,
            echo: 0.0,
            nudge: 0,
            spinY: 0.00115,
            pulseUntil: 0,
            targetRotX: 0.62,
            targetRotZ: 0.18
        };
        var major = 1.0;
        var minor = 0.36;
        var uSeg = 36;
        var vSeg = 18;
        var cx = 110;
        var cy = 110;

        function rotatePoint(x, y, z) {
            var cx1 = Math.cos(state.rotX), sx1 = Math.sin(state.rotX);
            var y1 = y * cx1 - z * sx1;
            var z1 = y * sx1 + z * cx1;
            var cy1 = Math.cos(state.rotY), sy1 = Math.sin(state.rotY);
            var x2 = x * cy1 + z1 * sy1;
            var z2 = -x * sy1 + z1 * cy1;
            var cz1 = Math.cos(state.rotZ), sz1 = Math.sin(state.rotZ);
            var x3 = x2 * cz1 - y1 * sz1;
            var y3 = x2 * sz1 + y1 * cz1;
            return { x: x3, y: y3, z: z2 };
        }

        function torusPoint(u, v) {
            var cu = Math.cos(u), su = Math.sin(u);
            var cv = Math.cos(v), sv = Math.sin(v);
            var ring = major + minor * cv;
            return rotatePoint(ring * cu, ring * su, minor * sv);
        }

        function echoIntensity(u, v) {
            var helix = Math.sin(v * 3.0 + u * PHI + state.echo);
            var coil = Math.cos(u * 6.0 - state.echo * 1.1 + Math.sin(v * 2.0));
            var triad = ((Math.floor(u * 4.77 + state.echo * 2.0) % 3) === 0) ? 0.22 : 0.0;
            return Math.min(1.0, Math.max(0.12, 0.28 + 0.42 * Math.max(0, helix * coil) + triad));
        }

        function lineSvg(x1, y1, x2, y2, u, v) {
            var glow = echoIntensity(u, v);
            var width = 0.45 + glow * 0.85;
            var opacity = 0.55 + glow * 0.4;
            var hue = 118 + glow * 18;
            return '<line x1="' + x1.toFixed(2) + '" y1="' + y1.toFixed(2)
                + '" x2="' + x2.toFixed(2) + '" y2="' + y2.toFixed(2)
                + '" stroke="#00FF41" stroke-width="' + width.toFixed(2)
                + '" stroke-opacity="' + opacity.toFixed(3) + '" />';
        }

        function render() {
            var scale = torusMeshScale();
            var parts = [];
            var uStep = (Math.PI * 2) / uSeg;
            var vStep = (Math.PI * 2) / vSeg;
            var ui, vi, u, v, p1, p2;
            for (ui = 0; ui <= uSeg; ui++) {
                u = ui * uStep;
                for (vi = 0; vi < vSeg; vi++) {
                    v = vi * vStep;
                    p1 = torusPoint(u, v);
                    p2 = torusPoint(u, v + vStep);
                    parts.push(lineSvg(
                        cx + p1.x * scale, cy - p1.y * scale,
                        cx + p2.x * scale, cy - p2.y * scale,
                        u, v
                    ));
                }
            }
            for (vi = 0; vi <= vSeg; vi++) {
                v = vi * vStep;
                for (ui = 0; ui < uSeg; ui++) {
                    u = ui * uStep;
                    p1 = torusPoint(u, v);
                    p2 = torusPoint(u + uStep, v);
                    parts.push(lineSvg(
                        cx + p1.x * scale, cy - p1.y * scale,
                        cx + p2.x * scale, cy - p2.y * scale,
                        u, v
                    ));
                }
            }
            mesh.innerHTML = parts.join('');
        }

        function tick() {
            var now = Date.now();
            state.rotY += state.spinY;
            if (now < state.pulseUntil) {
                state.rotZ += 0.0065;
                state.rotX += (state.targetRotX - state.rotX) * 0.04;
            } else {
                state.rotX += (0.62 + Math.sin(now * 0.00009) * 0.07 - state.rotX) * 0.02;
                state.rotZ += (state.targetRotZ - state.rotZ) * 0.015;
            }
            render();
            requestAnimationFrame(tick);
        }

        function absorbBridge(node) {
            if (!node) return;
            var echo = parseFloat(node.getAttribute('data-echo') || '0');
            var nudge = parseInt(node.getAttribute('data-nudge') || '0', 10);
            if (nudge !== state.nudge) {
                state.nudge = nudge;
                state.echo = echo;
                state.pulseUntil = Date.now() + 1400;
                state.targetRotX = 0.45 + (nudge % 5) * 0.08;
                state.targetRotZ = state.rotZ + 0.35 + (echo % 1.0) * 0.5;
                state.spinY = 0.00115 + (echo % 1.0) * 0.0008;
            }
        }

        function watchBridge() {
            var node = document.querySelector('.quartz-torus-bridge');
            if (!node) return;
            absorbBridge(node);
        }

        watchBridge();
        setInterval(watchBridge, 350);
        tick();
        window.quartzTorus = {
            ready: true,
            pulse: function() { state.pulseUntil = Date.now() + 1200; }
        };
    }
    whenBodyReady(function() {
        bootQuartzTorus();
        fitPanel();
    });
    window.addEventListener('resize', debouncedFitPanel);
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', debouncedFitPanel);
    }
    document.addEventListener('DOMContentLoaded', fitPanel);
    setTimeout(function() { whenBodyReady(fitPanel); }, 200);
    setTimeout(function() { whenBodyReady(fitPanel); }, 800);
    setTimeout(function() { whenBodyReady(bootQuartzTorus); }, 300);
    setTimeout(function() { whenBodyReady(bootQuartzTorus); }, 1200);
    setInterval(function() {
        var h = window.innerHeight || document.documentElement.clientHeight || 0;
        if (h > 0) syncTerminalOverflow(h);
        if (!torusMountRoot()) return;
        ensureTorusStage();
        fitTorusMount();
        if (!window.quartzTorus || !window.quartzTorus.ready) bootQuartzTorus();
    }, 1500);
    whenBodyReady(function() {
        var ta = document.querySelector('.quartz-terminal textarea');
        if (!ta || ta._quartzOverflowObs) return;
        ta._quartzOverflowObs = true;
        new MutationObserver(function() {
            var vh = window.innerHeight || document.documentElement.clientHeight || 0;
            if (vh > 0) syncTerminalOverflow(vh);
        }).observe(ta, { childList: true, characterData: true, subtree: true });
    });
})();
</script>
"""

QUARTZ_CSS = f"""
:root {{
    --quartz-phosphor: {_PHOSPHOR};
    --quartz-phosphor-dim: #00cc34;
    --quartz-phosphor-bright: #33ff66;
    --quartz-panel: {_PANEL_BG};
    --quartz-display: #0a0a0a;
    --quartz-btn-top: #3d424a;
    --quartz-btn-mid: #2a2e35;
    --quartz-btn-bot: #181b20;
    --quartz-border: #121418;
    --quartz-border-hi: #3a3f47;
    --quartz-inset: clamp(0.25rem, 2vw, 0.2in);
    --quartz-grid-line: rgba(90, 96, 104, 0.55);
    --quartz-skin-case-top: #2e333b;
    --quartz-skin-case-mid: #1e2228;
    --quartz-skin-case-bot: #0a0c0e;
    --quartz-skin-tab-h: 2.5rem;
    --quartz-skin-prog-h: 3rem;
    --quartz-skin-footer-h: 1.4rem;
    --quartz-skin-settings-h: 0px;
    --torus-line: {_PHOSPHOR};
    --torus-line-dim: rgba(0, 204, 52, 0.45);
    --torus-orbit: rgba(0, 255, 65, 0.22);
    --torus-core: {_PHOSPHOR};
}}
html, body {{
    background: #000000 !important;
    width: 100% !important;
    max-width: 100vw !important;
    height: var(--quartz-vh, 100dvh) !important;
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}}
.gradio-container {{
    width: 100% !important;
    max-width: 100vw !important;
    height: var(--quartz-vh, 100dvh) !important;
    padding: 0.35rem 0.5rem !important;
    background: #000000 !important;
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    position: relative !important;
    box-sizing: border-box !important;
}}
.gradio-container .quartz-root {{
    position: relative !important;
    z-index: 1 !important;
    width: 100% !important;
    max-width: 100% !important;
    height: calc(var(--quartz-vh, 100dvh) - 0.7rem) !important;
    max-height: calc(var(--quartz-vh, 100dvh) - 0.7rem) !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}}
.gradio-container .quartz-grid-layer {{
    position: fixed !important;
    inset: 0 !important;
    z-index: 0 !important;
    pointer-events: none !important;
    background-color: #000000 !important;
    background-image:
        linear-gradient(var(--quartz-grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--quartz-grid-line) 1px, transparent 1px) !important;
    background-size: 14px 14px !important;
    background-position: var(--quartz-grid-x, 0px) var(--quartz-grid-y, 0px) !important;
}}
.gradio-container .quartz-grid-off .quartz-grid-layer {{
    opacity: 0 !important;
    visibility: hidden !important;
}}


.gradio-container .quartz-display-backing {{
    background: var(--quartz-display) !important;
    opacity: 1 !important;
    visibility: visible !important;
}}
.gradio-container .block, .gradio-container .form {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
.gradio-container .quartz-panel {{
    position: relative !important;
    z-index: 1 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.45rem 0.55rem 0.35rem !important;
    box-shadow: none !important;
    height: calc(var(--quartz-vh, 100dvh) - 0.7rem) !important;
    max-height: calc(var(--quartz-vh, 100dvh) - 0.7rem) !important;
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    box-sizing: border-box !important;
}}
.gradio-container .quartz-panel > .block.quartz-content-wrap,
.gradio-container .quartz-panel > .block:not(.quartz-skin-mount),
.gradio-container .quartz-panel > .form:not(.quartz-skin-mount) {{
    position: relative !important;
    z-index: 5 !important;
}}
.gradio-container .quartz-skin-mount {{
    position: absolute !important;
    inset: 0 !important;
    z-index: 1 !important;
    pointer-events: none !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    overflow: hidden !important;
}}
.gradio-container .quartz-skin-mount > .block,
.gradio-container .quartz-skin-mount > .form {{
    position: absolute !important;
    inset: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    overflow: visible !important;
}}
.gradio-container .quartz-skin {{
    position: absolute !important;
    inset: 0 !important;
    pointer-events: none !important;
    z-index: 1 !important;
}}
.gradio-container .quartz-skin-case {{
    position: absolute !important;
    inset: 0 !important;
    z-index: 1 !important;
    border-radius: 6px !important;
    background: linear-gradient(
        165deg,
        var(--quartz-skin-case-top) 0%,
        #252930 12%,
        var(--quartz-skin-case-mid) 55%,
        #14171c 92%,
        var(--quartz-skin-case-bot) 100%
    ) !important;
    border: 2px solid #0a0c0e !important;
    box-shadow:
        inset 0 1px 0 rgba(120,128,140,0.35),
        inset 0 -4px 12px rgba(0,0,0,0.55),
        0 8px 28px rgba(0,0,0,0.65) !important;
}}
.gradio-container .quartz-skin-tab-rail,
.gradio-container .quartz-skin-display-frame,
.gradio-container .quartz-skin-prog-tray,
.gradio-container .quartz-skin-footer-strip {{
    z-index: 2 !important;
}}
.gradio-container .quartz-skin-tab-rail {{
    position: absolute !important;
    top: 0.45rem !important;
    left: 0.55rem !important;
    right: 0.55rem !important;
    height: var(--quartz-skin-tab-h) !important;
    border-radius: 4px 4px 0 0 !important;
    background: linear-gradient(180deg, #252930 0%, #1a1d22 100%) !important;
    border: 1px solid #121418 !important;
    box-shadow: inset 0 1px 0 rgba(90,98,110,0.22) !important;
}}
.gradio-container .quartz-skin-display-frame {{
    position: absolute !important;
    top: calc(0.45rem + var(--quartz-skin-tab-h) + 0.35rem) !important;
    left: calc(0.55rem + var(--quartz-inset)) !important;
    right: calc(0.55rem + var(--quartz-inset)) !important;
    bottom: calc(
        0.35rem + var(--quartz-skin-footer-h) + var(--quartz-skin-prog-h)
        + var(--quartz-skin-settings-h) + 0.54rem
    ) !important;
    border-radius: 4px !important;
    background: transparent !important;
    border: 2px solid #3a3f47 !important;
    box-shadow:
        inset 0 0 0 1px rgba(0,0,0,0.65),
        0 2px 10px rgba(0,0,0,0.4) !important;
    pointer-events: none !important;
}}
.gradio-container .quartz-skin-prog-tray {{
    position: absolute !important;
    left: 0.55rem !important;
    right: 0.55rem !important;
    bottom: calc(0.35rem + var(--quartz-skin-footer-h) + 0.1rem) !important;
    height: var(--quartz-skin-prog-h) !important;
    border-radius: 4px !important;
    background: linear-gradient(180deg, #252930 0%, #1a1d22 100%) !important;
    border: 1px solid #1e2228 !important;
    box-shadow: inset 0 1px 0 rgba(100,108,120,0.2) !important;
}}
.gradio-container .quartz-skin-footer-strip {{
    position: absolute !important;
    left: 0.55rem !important;
    right: 0.55rem !important;
    bottom: 0.35rem !important;
    height: var(--quartz-skin-footer-h) !important;
    border-top: 2px solid #1e2228 !important;
    background: linear-gradient(180deg, #252930 0%, #14171c 100%) !important;
}}
.gradio-container .quartz-content {{
    position: relative !important;
    z-index: 5 !important;
    background: transparent !important;
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    min-width: 0 !important;
    max-width: 100% !important;
    height: 100% !important;
    overflow-x: hidden !important;
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
    color: var(--quartz-phosphor-dim) !important;
    -webkit-text-fill-color: var(--quartz-phosphor-dim) !important;
    font-family: "Segoe UI", system-ui, sans-serif !important;
    font-size: clamp(0.62rem, 1.32vh, 0.82rem) !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    text-shadow:
        0 1px 2px rgba(0,0,0,0.85),
        0 0 6px rgba(0,255,65,0.25) !important;
    box-shadow:
        inset 0 1px 0 rgba(140,148,160,0.35),
        0 2px 5px rgba(0,0,0,0.45) !important;
    cursor: pointer !important;
}}
.gradio-container button.quartz-tab span {{
    color: var(--quartz-phosphor-dim) !important;
    -webkit-text-fill-color: var(--quartz-phosphor-dim) !important;
    text-shadow:
        0 1px 2px rgba(0,0,0,0.85),
        0 0 6px rgba(0,255,65,0.25) !important;
}}
.gradio-container button.quartz-tab-active {{
    background: linear-gradient(180deg, #4a515c 0%, #353a42 42%, #252930 100%) !important;
    color: var(--quartz-phosphor-bright) !important;
    -webkit-text-fill-color: var(--quartz-phosphor-bright) !important;
    text-shadow:
        0 0 10px rgba(0,255,65,0.55),
        0 1px 2px rgba(0,0,0,0.9),
        0 0 1px rgba(0,255,65,0.2) !important;
    box-shadow:
        inset 0 1px 0 rgba(100,108,120,0.35),
        0 0 14px rgba(0,255,65,0.18),
        0 2px 6px rgba(0,0,0,0.5) !important;
    border-color: var(--quartz-border-hi) !important;
}}
.gradio-container button.quartz-tab-active span {{
    color: var(--quartz-phosphor-bright) !important;
    -webkit-text-fill-color: var(--quartz-phosphor-bright) !important;
    text-shadow:
        0 0 10px rgba(0,255,65,0.55),
        0 1px 2px rgba(0,0,0,0.9) !important;
}}
.gradio-container .quartz-display-bay {{
    flex: 1 1 auto !important;
    min-height: 0 !important;
    min-width: 0 !important;
    width: calc(100% - 2 * var(--quartz-inset)) !important;
    max-width: calc(100% - 2 * var(--quartz-inset)) !important;
    margin: var(--quartz-inset) !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    position: relative !important;
    z-index: 5 !important;
    overflow-x: hidden !important;
    overflow-y: visible !important;
}}
.gradio-container .quartz-display-bay > .block,
.gradio-container .quartz-display-bay > .form {{
    overflow-x: hidden !important;
    max-width: 100% !important;
    min-width: 0 !important;
}}
.gradio-container .quartz-display-backing {{
    position: absolute !important;
    z-index: 0 !important;
    pointer-events: none !important;
    border-radius: 2px !important;
    border: 1px solid #1a3a1a !important;
    box-shadow: inset 0 0 10px rgba(0,255,65,0.08) !important;
}}
.gradio-container .quartz-terminal-col {{
    position: relative !important;
    z-index: 2 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    background: transparent !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.22rem 0.26rem 0.18rem !important;
    box-shadow: none !important;
    margin: 0.2rem !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
    overflow-y: visible !important;
}}
.gradio-container .quartz-terminal-col > .block,
.gradio-container .quartz-terminal-col > .form {{
    overflow-x: hidden !important;
    max-width: 100% !important;
    min-width: 0 !important;
}}
.gradio-container .quartz-grid-off .quartz-terminal-col {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
.gradio-container .quartz-grid-on .quartz-terminal-col {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
.gradio-container .quartz-terminal {{
    flex: 1 1 auto !important;
    min-height: 0 !important;
    min-width: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
}}
.gradio-container .quartz-terminal > .block,
.gradio-container .quartz-terminal > div,
.gradio-container .quartz-terminal .wrap,
.gradio-container .quartz-terminal label {{
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}}
.quartz-torus-stage {{
    position: fixed !important;
    z-index: 99999 !important;
    pointer-events: none !important;
    overflow: hidden !important;
    background: transparent !important;
    box-sizing: border-box !important;
}}
.quartz-torus-stage .quartz-torus-frame {{
    position: absolute !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: none !important;
}}
.quartz-torus-stage .quartz-torus-svg {{
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
    opacity: 1 !important;
    overflow: visible !important;
    filter: drop-shadow(0 0 10px rgba(0, 255, 65, 0.45)) !important;
}}
.quartz-torus-stage .quartz-torus-mesh line {{
    vector-effect: non-scaling-stroke !important;
}}
.quartz-torus-stage .quartz-torus-orbit {{
    fill: none !important;
    stroke: var(--torus-orbit) !important;
    stroke-width: 0.65 !important;
}}
.quartz-torus-stage .quartz-torus-core {{
    fill: var(--torus-core) !important;
    stroke: none !important;
}}
.gradio-container .quartz-terminal textarea {{
    position: relative !important;
    z-index: 1 !important;
    width: 100% !important;
    padding-right: 50% !important;
    background: var(--quartz-display) !important;
    background-color: var(--quartz-display) !important;
    color: var(--quartz-phosphor) !important;
    -webkit-text-fill-color: var(--quartz-phosphor) !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: clamp(0.7rem, 1.44vh, 0.86rem) !important;
    line-height: 1.38 !important;
    border: 1px solid #1a3a1a !important;
    border-radius: 2px !important;
    box-shadow: inset 0 0 10px rgba(0,255,65,0.08) !important;
    text-shadow: 0 0 6px rgba(0,255,65,0.35) !important;
    resize: none !important;
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    box-sizing: border-box !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    max-width: 100% !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(0, 255, 65, 0.22) transparent !important;
}}
.gradio-container .quartz-terminal textarea::-webkit-scrollbar {{
    width: 5px !important;
    height: 0 !important;
}}
.gradio-container .quartz-terminal textarea::-webkit-scrollbar:horizontal {{
    display: none !important;
    height: 0 !important;
}}
.gradio-container .quartz-terminal textarea::-webkit-scrollbar-track {{
    background: transparent !important;
}}
.gradio-container .quartz-terminal textarea::-webkit-scrollbar-thumb {{
    background: rgba(0, 255, 65, 0.22) !important;
    border-radius: 3px !important;
}}
.gradio-container .quartz-terminal-col::after {{
    content: none !important;
    display: none !important;
}}
.gradio-container .quartz-grid-off .quartz-terminal-col::after {{
    content: "" !important;
    display: block !important;
    pointer-events: none !important;
    position: absolute !important;
    inset: 0 !important;
    z-index: 4 !important;
    background: repeating-linear-gradient(
        0deg,
        transparent 0px,
        transparent 2px,
        rgba(0,0,0,0.12) 2px,
        rgba(0,0,0,0.12) 3px
    ) !important;
    opacity: 0.3 !important;
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
    font-size: clamp(0.65rem, 1.32vh, 0.8rem) !important;
    min-height: clamp(1.45rem, 3vh, 1.85rem) !important;
    box-shadow: inset 0 0 10px rgba(0,255,65,0.08) !important;
}}
.gradio-container button.quartz-btn {{
    border: 1px solid var(--quartz-border) !important;
    border-radius: 4px !important;
    background: linear-gradient(
        180deg,
        var(--quartz-btn-top) 0%,
        var(--quartz-btn-mid) 48%,
        var(--quartz-btn-bot) 100%
    ) !important;
    color: var(--quartz-phosphor) !important;
    -webkit-text-fill-color: var(--quartz-phosphor) !important;
    font-family: "Segoe UI", system-ui, sans-serif !important;
    font-weight: 700 !important;
    font-size: clamp(0.58rem, 1.2vh, 0.74rem) !important;
    text-shadow:
        0 0 8px rgba(0,255,65,0.35),
        0 1px 2px rgba(0,0,0,0.9),
        0 -1px 0 rgba(0,255,65,0.08) !important;
    box-shadow:
        inset 0 1px 0 rgba(140,148,160,0.32),
        inset 0 -2px 4px rgba(0,0,0,0.45),
        0 2px 5px rgba(0,0,0,0.45) !important;
    cursor: pointer !important;
    transition: filter 0.12s ease, box-shadow 0.12s ease !important;
}}
.gradio-container button.quartz-btn span {{
    color: var(--quartz-phosphor) !important;
    -webkit-text-fill-color: var(--quartz-phosphor) !important;
    text-shadow:
        0 0 8px rgba(0,255,65,0.35),
        0 1px 2px rgba(0,0,0,0.9) !important;
}}
.gradio-container button.quartz-btn:hover {{
    filter: brightness(1.12) !important;
    box-shadow:
        inset 0 1px 0 rgba(100,108,120,0.32),
        0 0 12px rgba(0,255,65,0.22),
        0 2px 6px rgba(0,0,0,0.5) !important;
}}
.gradio-container button.quartz-btn:hover span {{
    color: var(--quartz-phosphor-bright) !important;
    -webkit-text-fill-color: var(--quartz-phosphor-bright) !important;
    text-shadow:
        0 0 12px rgba(0,255,65,0.5),
        0 1px 2px rgba(0,0,0,0.9) !important;
}}
.gradio-container button.quartz-send {{
    flex: 0 0 clamp(3.2rem, 7vw, 4.2rem) !important;
    min-width: clamp(3.2rem, 7vw, 4.2rem) !important;
    font-size: clamp(0.58rem, 1.2vh, 0.7rem) !important;
    letter-spacing: 0.1em !important;
}}
.gradio-container button.quartz-btn-pulse {{
    filter: brightness(0.92) !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.35) !important;
}}
.gradio-container .quartz-prog-row {{
    gap: 0.22rem !important;
    margin: 0.32rem 0 0.22rem 0 !important;
    flex-shrink: 0 !important;
    padding: 0.28rem 0.22rem !important;
    background: transparent !important;
    border: none !important;
    border-radius: 4px !important;
    box-shadow: none !important;
}}
.gradio-container button.quartz-prog {{
    flex: 1 1 0 !important;
    min-width: 0 !important;
    min-height: clamp(1.35rem, 2.8vh, 1.7rem) !important;
    font-size: clamp(0.48rem, 0.98vh, 0.6rem) !important;
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
    border-top: none !important;
    margin-top: 0.1rem !important;
    background: transparent !important;
}}
.gradio-container .quartz-footer {{
    color: var(--quartz-phosphor-dim) !important;
    font-family: "Segoe UI", system-ui, sans-serif !important;
    font-size: clamp(0.53rem, 1.08vh, 0.65rem) !important;
    font-weight: 700 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    text-shadow:
        0 0 6px rgba(0,255,65,0.25),
        0 1px 2px rgba(0,0,0,0.8) !important;
}}
.gradio-container .quartz-settings {{
    margin: 0.2rem 0 !important;
    padding: 0.28rem 0.35rem !important;
    background: rgba(0,0,0,0.28) !important;
    border: 1px solid #3d434d !important;
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
        block_background_fill_dark="transparent",
        input_background_fill="transparent",
        button_primary_background_fill="#4a515c",
        button_primary_text_color=_PHOSPHOR,
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

        with gr.Column(elem_classes=_root_classes(True)) as root_col:
            torus_bridge = gr.HTML(TORUS_STATE_BRIDGE_HTML, visible=False)
            gr.HTML(GRID_LAYER_HTML)
            with gr.Column(elem_classes=["quartz-panel"]):
                gr.HTML(_panel_skin_html(), elem_classes=["quartz-skin-mount"])
                with gr.Column(elem_classes=["quartz-content", "quartz-content-wrap"]):
                    with gr.Row(elem_classes=["quartz-top-tabs"]):
                        for tab_id in TOP_TABS:
                            tab_btns[tab_id] = gr.Button(
                                TAB_LABELS[tab_id],
                                elem_classes=_tab_btn_classes("chat", tab_id),
                                variant="secondary",
                            )

                    with gr.Column(elem_classes=["quartz-display-bay"]):
                        with gr.Column(elem_classes=["quartz-terminal-col"]):
                            gr.HTML(DISPLAY_BACKING_HTML)
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

        core_outputs = [terminal, cmd_input, ui_state, torus_bridge]
        tune_inputs = [bake_steps, bandwidth, use_vqc, drift_samples, max_facts]

        tab_outputs = [
            terminal,
            ui_state,
            settings_panel,
            *tab_btns.values(),
            root_col,
            torus_bridge,
        ]

        tab_btns["chat"].click(
            _handle_grid_toggle,
            inputs=[terminal, ui_state],
            outputs=tab_outputs,
        )
        for tab_id, btn in tab_btns.items():
            if tab_id == "chat":
                continue
            evt = btn.click(
                _make_tab_switch(tab_id),
                inputs=[terminal, ui_state],
                outputs=tab_outputs,
            )
            if tab_id == "memory":
                evt.then(
                    _handle_exec,
                    inputs=[cmd_input, terminal, ui_state, *tune_inputs],
                    outputs=[terminal, cmd_input, ui_state, torus_bridge],
                )

        send_btn.click(_handle_send, inputs=[cmd_input, terminal, ui_state], outputs=core_outputs)
        cmd_input.submit(_handle_send, inputs=[cmd_input, terminal, ui_state], outputs=core_outputs)

        for prog_id, btn in prog_btns.items():
            btn.click(
                _make_prog_handler(prog_id),
                inputs=[terminal, ui_state],
                outputs=[terminal, cmd_input, ui_state, torus_bridge],
            )

        clear_btn.click(
            _handle_clear,
            inputs=[terminal, ui_state],
            outputs=[terminal, ui_state, torus_bridge],
        )
        help_btn.click(
            _handle_help,
            inputs=[terminal, ui_state],
            outputs=[terminal, ui_state, torus_bridge],
        )
        mode_btn.click(
            _handle_mode,
            inputs=[terminal, ui_state, use_vqc],
            outputs=[terminal, ui_state, torus_bridge],
        )
        exec_btn.click(
            _handle_exec,
            inputs=[cmd_input, terminal, ui_state, *tune_inputs],
            outputs=[terminal, cmd_input, ui_state, torus_bridge],
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