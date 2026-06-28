#!/usr/bin/env python3
"""QVPIC Hugging Face Space — QUARTZ AI SYNTHESIZER vintage control panel."""

from __future__ import annotations

import json
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
    terminal_guided_onboarding,
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

STARTUP_STRING = "TEST EVERYTHING, HOLD FAST WHAT IS GOOD AND KNOW YOUR GOD"
STARTUP_DISPLAY_STRING = STARTUP_STRING + "....."
STARTUP_CHAR_DELAY_MS = 200
STARTUP_POST_DELAY_MS = 2000
# (label, action_key) — max 9 entries per page; overflow uses next/prev page slots.
HOME_MENU_PAGES: tuple[tuple[tuple[str, str], ...], ...] = (
    (
        ("Quick Diagnostic", "quick_diagnostic"),
        ("Games", "games"),
        ("Quaternion Vortex Controls", "vortex"),
        ("Persistent Identity Settings", "identity"),
        ("VQC Tuning Panel", "vqc"),
        ("Bake → Recall Benchmark", "benchmark"),
        ("System Diagnostics", "diagnostics"),
        ("Tools & Repositories", "tools"),
        ("Next page", "next_page"),
    ),
    (
        ("Guided Tour (Onboarding)", "guided_tour"),
        ("Command History", "history"),
        ("Grid View Toggle", "grid_toggle"),
        ("About / Credits", "about"),
        ("Help & Keypad Guide", "help"),
        ("Previous page", "prev_page"),
    ),
)

PROG_BANK_SIZE = 16
def _default_prog_states() -> dict[str, bool]:
    states = {f"prog{i}": False for i in range(1, PROG_BANK_SIZE + 1)}
    states["prog1"] = True
    return states


DEFAULT_PROG_STATES = _default_prog_states()
PROG_ROW_1 = tuple(range(1, 9))
PROG_ROW_2 = tuple(range(9, PROG_BANK_SIZE + 1))

ACTIVE_SKIN = "quartz-default"

GRID_LAYER_HTML = """
<div class="quartz-grid-layer" aria-hidden="true"></div>
"""

DISPLAY_BACKING_HTML = """
<div class="quartz-display-backing" aria-hidden="true"></div>
<div class="quartz-invaders-stage" aria-hidden="true">
  <canvas class="quartz-invaders-canvas"></canvas>
  <div class="quartz-invaders-crt" aria-hidden="true"></div>
</div>
"""

TORUS_STATE_BRIDGE_HTML = (
    '<div id="quartz-torus-bridge" class="quartz-torus-bridge" data-echo="0" data-nudge="0" '
    'data-visible="1" hidden></div>'
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
        "> SEND / Enter key: submit command",
        "> HOME: toggle selection menu (red LED = on) · enter index to select",
        "> PROG 1–16: maintained toggles (red LED = on) · Prog 1 = torus · Prog 9 = move",
        "> MEMORY tab: run benchmark · SETTINGS: tune conduit dials",
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
        "active_tab": "home",
        "grid_view": True,
        "history": [],
        "cmd_index": -1,
        "last_cmd": "",
        "torus_echo": 0.0,
        "torus_nudge": 0,
        "torus_visible": True,
        "home_active": True,
        "menu_page": 0,
        "pending_startup_replay": False,
        "game_active": False,
        "pending_game_start": False,
    }


def _touch_torus(state: dict, strength: float = 1.0) -> dict:
    state = dict(state) if state else _default_ui_state()
    state["torus_echo"] = round(float(state.get("torus_echo", 0.0)) + strength * 0.618034, 6)
    state["torus_nudge"] = int(state.get("torus_nudge", 0)) + 1
    return state


def _torus_bridge_html(state: dict) -> str:
    echo = float(state.get("torus_echo", 0.0))
    nudge = int(state.get("torus_nudge", 0))
    visible = 1 if state.get("torus_visible", True) else 0
    replay = 1 if state.get("pending_startup_replay") else 0
    game_start = 1 if state.get("pending_game_start") else 0
    return (
        f'<div id="quartz-torus-bridge" class="quartz-torus-bridge" data-echo="{echo}" '
        f'data-nudge="{nudge}" data-visible="{visible}" '
        f'data-startup-replay="{replay}" data-game-start="{game_start}" hidden></div>'
    )


def _root_classes(grid_on: bool) -> list[str]:
    return ["quartz-root", "quartz-grid-on" if grid_on else "quartz-grid-off"]


def _append_terminal(terminal: str, *lines: str) -> str:
    base = terminal.rstrip()
    chunk = "\n".join(lines)
    return f"{base}\n{chunk}" if base else chunk


def _tab_btn_classes(active: str, tab_id: str) -> list[str]:
    classes = ["quartz-tab", f"quartz-tab-{tab_id}"]
    if active != "home" and tab_id == active:
        classes.append("quartz-tab-active")
    return classes


def _tab_updates(active: str) -> tuple:
    return tuple(
        gr.update(elem_classes=_tab_btn_classes(active, tab_id)) for tab_id in TOP_TABS
    )


def _home_btn_classes(active: bool = False) -> list[str]:
    classes = ["quartz-btn", "quartz-tab", "quartz-home-btn"]
    if active:
        classes.append("quartz-home-active")
    return classes


def _home_menu_text(page: int = 0) -> str:
    page = max(0, min(page, len(HOME_MENU_PAGES) - 1))
    items = HOME_MENU_PAGES[page]
    lines = [
        "> ═══════════════════════════════════════════════════════",
        ">  QVPIC //@ SELECTION MENU",
        f">  PAGE {page + 1}/{len(HOME_MENU_PAGES)}",
        "> ═══════════════════════════════════════════════════════",
        ">  Enter index number and press SEND:",
        ">  Tip: page 2 → Guided Tour for first-time onboarding",
        "> ",
    ]
    for index, (title, _action) in enumerate(items, start=1):
        lines.append(f">  {index}. {title}")
    lines.extend(["> ", "> _"])
    return "\n".join(lines)


HOME_MENU_TEXT = _home_menu_text(0)


def _effective_terminal(terminal: str, state: dict) -> str:
    """Use server menu text when the TUI was painted client-side only."""
    text = (terminal or "").rstrip()
    if state.get("home_active"):
        page = int(state.get("menu_page", 0))
        menu = _home_menu_text(page)
        if not text or "SELECTION MENU" in text:
            return menu
    return text


def _handle_game_quit(state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state["game_active"] = False
    state["pending_game_start"] = False
    state["home_active"] = True
    state["active_tab"] = "home"
    state["menu_page"] = 0
    return (
        _home_menu_text(0),
        state,
        gr.update(elem_classes=_home_btn_classes(True)),
        gr.update(value=_torus_bridge_html(state)),
    )


def _handle_game_start_done(state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state["pending_game_start"] = False
    state["game_active"] = True
    state["home_active"] = False
    state["active_tab"] = "games"
    return state, gr.update(value=_torus_bridge_html(state))


def _handle_startup_replay_done(state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state["pending_startup_replay"] = False
    state["home_active"] = True
    state["active_tab"] = "home"
    state["menu_page"] = 0
    state["torus_visible"] = False
    return (
        _home_menu_text(0),
        state,
        gr.update(elem_classes=_home_btn_classes(True)),
        gr.update(value=_torus_bridge_html(state)),
    )


def _sync_boot_state(state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state["home_active"] = True
    state["active_tab"] = "home"
    state["menu_page"] = 0
    return (
        _home_menu_text(0),
        state,
        gr.update(elem_classes=_home_btn_classes(True)),
    )


def _build_game_menu_ack_js() -> str:
    return """
function() {
    setTimeout(function() {
        var node = document.querySelector('#quartz-torus-bridge, .quartz-torus-bridge');
        var bridgeStart = node ? parseInt(node.getAttribute('data-game-start') || '0', 10) === 1 : false;
        var ta = document.querySelector('.quartz-terminal textarea');
        var textStart = !!(ta && ta.value && ta.value.indexOf('SPACE INVADERS') >= 0);
        if (!bridgeStart && !textStart) return;
        if (typeof window.quartzStartPlayerGame === 'function') {
            window.quartzStartPlayerGame();
        }
    }, 320);
}
"""


def _build_startup_load_js() -> str:
    return """
async () => {
    const BOOT_KEY = 'qvpic-boot-complete';
    const ta = await window.quartzWaitForTerminal(0);
    if (sessionStorage.getItem(BOOT_KEY)) {
        window.quartzPaintTerminal(ta, window.QUARTZ_HOME_MENU_TEXT);
        window.quartzSetHomeLed(true);
        window.quartzBootDone = true;
        return window.QUARTZ_HOME_MENU_TEXT;
    }
    await window.quartzRunStartupSequence(ta, { postDelay: 0, persistBoot: true });
    window.quartzBootDone = true;
    return window.QUARTZ_HOME_MENU_TEXT;
}
"""


def _metallic_btn(*extra: str) -> list[str]:
    return ["quartz-btn", *extra]


def _prog_btn_classes(index: int) -> list[str]:
    return _metallic_btn("quartz-prog", "quartz-prog-toggle", f"quartz-prog-id-{index}")


def _simulate_chat_response(message: str) -> str:
    lower = message.lower().strip()
    if any(word in lower for word in ("hello", "hi", "hey")):
        return (
            "> QVPIC: Conduit online. Toggle Prog keys below, "
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


def _deactivate_home(state: dict) -> dict:
    state = dict(state)
    state["home_active"] = False
    return state


def _handle_home_toggle(terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state["game_active"] = False
    state["pending_game_start"] = False
    state = _touch_torus(state, 0.25)
    active = not bool(state.get("home_active", False))
    state["home_active"] = active
    grid_on = bool(state.get("grid_view", True))
    if active:
        state["active_tab"] = "home"
        state["menu_page"] = 0
        terminal = _home_menu_text(0)
        tab_key = "home"
    else:
        if state.get("active_tab") == "home":
            state["active_tab"] = "chat"
        terminal = _append_terminal(terminal, "> HOME: selection menu closed", "> _")
        tab_key = state.get("active_tab", "chat")
    return (
        terminal,
        state,
        gr.update(visible=False),
        *_tab_updates(tab_key),
        gr.update(elem_classes=_home_btn_classes(active)),
        gr.update(elem_classes=_root_classes(grid_on)),
        gr.update(value=_torus_bridge_html(state)),
    )


def _route_menu_action(action: str, terminal: str, state: dict) -> tuple[str, dict, str, bool]:
    """Return (terminal, state, active_tab, show_settings)."""
    state = _deactivate_home(state)
    if action == "next_page":
        state["home_active"] = True
        state["menu_page"] = min(int(state.get("menu_page", 0)) + 1, len(HOME_MENU_PAGES) - 1)
        return _home_menu_text(state["menu_page"]), state, "home", False
    if action == "prev_page":
        state["home_active"] = True
        state["menu_page"] = max(int(state.get("menu_page", 0)) - 1, 0)
        return _home_menu_text(state["menu_page"]), state, "home", False
    if action == "quick_diagnostic":
        state["pending_startup_replay"] = True
        state["home_active"] = False
        state["menu_page"] = 0
        terminal = _append_terminal(terminal, "> QUICK DIAGNOSTIC: centered startup + LED sync…")
        return terminal, state, "home", False
    if action == "games":
        state["game_active"] = True
        state["pending_game_start"] = True
        state["home_active"] = False
        terminal = _append_terminal(
            terminal,
            "> GAMES: SPACE INVADERS",
            "> Arrows move · Space shoot · Esc quit",
        )
        return terminal, state, "games", False
    if action == "vortex":
        terminal = _append_terminal(terminal, "> MENU: Quaternion Vortex Controls", "> SETTINGS: tune conduit dials below.")
        return terminal, state, "settings", True
    if action == "identity":
        terminal = _append_terminal(
            terminal,
            "> MENU: Persistent Identity Settings",
            "> SETTINGS: bandwidth and drift samples affect identity retention.",
        )
        return terminal, state, "settings", True
    if action == "vqc":
        terminal = _append_terminal(terminal, "> MENU: VQC Tuning Panel", "> SETTINGS: enable VQCEnhanced for helical conduit.")
        return terminal, state, "settings", True
    if action == "benchmark":
        terminal = _append_terminal(terminal, "> MENU: Bake → Recall Benchmark", "> MEMORY: press SEND with 'benchmark' or tap MEMORY.")
        return terminal, state, "memory", False
    if action == "diagnostics":
        terminal = _append_terminal(terminal, "> MENU: System Diagnostics", "> MEMORY: launching benchmark diagnostics…")
        return terminal, state, "memory", False
    if action == "tools":
        terminal = _append_terminal(
            terminal,
            "> MENU: Tools & Repositories",
            f">   GitHub  {GITHUB_URL}",
            f">   VQC     {VQC_URL}",
            f">   Space   {HF_SPACE_URL}",
            "> _",
        )
        return terminal, state, "tools", False
    if action == "guided_tour":
        tour = terminal_guided_onboarding().replace("\n", "\n> ")
        terminal = _append_terminal(
            terminal,
            "> MENU: Guided Tour (Onboarding)",
            "> " + tour,
            "> ",
            "> NEXT: MEMORY tab → type benchmark → SEND",
            "> _",
        )
        return terminal, state, "memory", False
    if action == "history":
        hist = state.get("history") or []
        if hist:
            lines = ["> MENU: Command History", "> HISTORY — recent commands:"] + [
                f">   · {c}" for c in hist[-12:]
            ]
            terminal = _append_terminal(terminal, *lines, "> _")
        else:
            terminal = _append_terminal(terminal, "> MENU: Command History", "> HISTORY: (empty)", "> _")
        return terminal, state, "history", False
    if action == "grid_toggle":
        grid_on = not bool(state.get("grid_view", True))
        state["grid_view"] = grid_on
        label = "ON — four-panel grid visible" if grid_on else "OFF — solid display"
        terminal = _append_terminal(terminal, "> MENU: Grid View Toggle", f"> GRID VIEW: {label}", "> _")
        return terminal, state, "chat", False
    if action == "about":
        terminal = _append_terminal(
            terminal,
            "> MENU: About / Credits",
            "> QVPIC — Quaternion Vortex Persistent Identity Conduit",
            "> QUARTZ AI SYNTHESIZER control panel",
            f"> Repo: {GITHUB_URL}",
            "> _",
        )
        return terminal, state, "tools", False
    if action == "help":
        terminal = _append_terminal(terminal, "> MENU: Help & Keypad Guide", HELP_TEXT, "> _")
        return terminal, state, "chat", False
    terminal = _append_terminal(terminal, f"> MENU: unknown action '{action}'", "> _")
    return terminal, state, "chat", False


def _handle_menu_selection(cmd: str, terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    terminal = _effective_terminal(terminal, state)
    page = int(state.get("menu_page", 0))
    page = max(0, min(page, len(HOME_MENU_PAGES) - 1))
    items = HOME_MENU_PAGES[page]
    raw = (cmd or "").strip()
    try:
        index = int(raw)
    except ValueError:
        terminal = _append_terminal(
            terminal,
            f"> SELECT: enter a number 1–{len(items)} and press SEND",
            "> _",
        )
        grid_on = bool(state.get("grid_view", True))
        return (
            terminal,
            "",
            state,
            gr.update(value=_torus_bridge_html(state)),
            gr.update(elem_classes=_home_btn_classes(True)),
            gr.update(visible=False),
            *_tab_updates("home"),
            gr.update(elem_classes=_root_classes(grid_on)),
        )
    if index < 1 or index > len(items):
        terminal = _append_terminal(
            terminal,
            f"> SELECT: invalid index — choose 1–{len(items)}",
            "> _",
        )
        grid_on = bool(state.get("grid_view", True))
        return (
            terminal,
            "",
            state,
            gr.update(value=_torus_bridge_html(state)),
            gr.update(elem_classes=_home_btn_classes(True)),
            gr.update(visible=False),
            *_tab_updates("home"),
            gr.update(elem_classes=_root_classes(grid_on)),
        )
    title, action = items[index - 1]
    state = _touch_torus(state, 0.4)
    if action == "quick_diagnostic":
        state["pending_startup_replay"] = True
        state["home_active"] = False
        state["menu_page"] = 0
        state["active_tab"] = "home"
        terminal = _append_terminal(
            terminal,
            f"> SELECT: {index} — {title}",
            "> QUICK DIAGNOSTIC: centered startup + Prog LED sync…",
        )
        grid_on = bool(state.get("grid_view", True))
        return (
            terminal,
            "",
            state,
            gr.update(value=_torus_bridge_html(state)),
            gr.update(elem_classes=_home_btn_classes(False)),
            gr.update(visible=False),
            *_tab_updates("home"),
            gr.update(elem_classes=_root_classes(grid_on)),
        )
    if action == "games":
        state["game_active"] = True
        state["pending_game_start"] = True
        state["home_active"] = False
        state["active_tab"] = "games"
        terminal = _append_terminal(
            terminal,
            f"> SELECT: {index} — {title}",
            "> SPACE INVADERS loading…",
            "> Arrows move · Space shoot · Esc quit",
        )
        grid_on = bool(state.get("grid_view", True))
        return (
            terminal,
            "",
            state,
            gr.update(value=_torus_bridge_html(state)),
            gr.update(elem_classes=_home_btn_classes(False)),
            gr.update(visible=False),
            *_tab_updates("games"),
            gr.update(elem_classes=_root_classes(grid_on)),
        )
    if action in {"next_page", "prev_page"}:
        terminal, state, active_tab, show_settings = _route_menu_action(action, terminal, state)
    else:
        terminal = _append_terminal(terminal, f"> SELECT: {index} — {title}")
        terminal, state, active_tab, show_settings = _route_menu_action(action, terminal, state)
    state["active_tab"] = active_tab
    grid_on = bool(state.get("grid_view", True))
    home_on = bool(state.get("home_active", False))
    return (
        terminal,
        "",
        state,
        gr.update(value=_torus_bridge_html(state)),
        gr.update(elem_classes=_home_btn_classes(home_on)),
        gr.update(visible=show_settings),
        *_tab_updates("home" if home_on else active_tab),
        gr.update(elem_classes=_root_classes(grid_on)),
    )


def _handle_grid_toggle(terminal: str, state: dict) -> tuple:
    state = dict(state) if state else _default_ui_state()
    state = _touch_torus(state, 0.35)
    grid_on = not bool(state.get("grid_view", True))
    state["grid_view"] = grid_on
    state = _deactivate_home(state)
    state["active_tab"] = "chat"
    label = "ON — four-panel grid visible" if grid_on else "OFF — solid display"
    terminal = _append_terminal(terminal, f"> GRID VIEW: {label}")
    return (
        terminal,
        state,
        gr.update(visible=False),
        *_tab_updates("chat"),
        gr.update(elem_classes=_home_btn_classes(False)),
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
    state = _deactivate_home(state)
    return (
        terminal,
        state,
        gr.update(visible=show_settings),
        *_tab_updates(tab_id),
        gr.update(elem_classes=_home_btn_classes(False)),
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
        return (
            terminal,
            "",
            state,
            gr.update(value=_torus_bridge_html(state)),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    if state.get("game_active"):
        if cmd == "2":
            state["pending_game_start"] = True
            state = _touch_torus(state, 0.2)
            terminal = _append_terminal(
                terminal,
                "> SPACE INVADERS loading…",
                "> Arrows move · Space shoot · Esc quit",
            )
        return (
            terminal,
            "",
            state,
            gr.update(value=_torus_bridge_html(state)),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    if state.get("home_active"):
        return _handle_menu_selection(cmd, _effective_terminal(terminal, state), state)
    state = _push_history(state, cmd)
    state = _touch_torus(state, 1.0)
    terminal = _append_terminal(terminal, f"> USER: {cmd}", _simulate_chat_response(cmd), "> _")
    grid_on = bool(state.get("grid_view", True))
    return (
        terminal,
        "",
        state,
        gr.update(value=_torus_bridge_html(state)),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(elem_classes=_root_classes(grid_on)),
    )


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


def _quartz_startup_js_block() -> str:
    return f"""
<script>
window.QUARTZ_STARTUP_STRING = {json.dumps(STARTUP_STRING)};
window.QUARTZ_HOME_MENU_TEXT = {json.dumps(HOME_MENU_TEXT)};
window.QUARTZ_CHAR_DELAY = {STARTUP_CHAR_DELAY_MS};
window.QUARTZ_MENU_POST_DELAY = {STARTUP_POST_DELAY_MS};
window.QUARTZ_PROG_BANK_SIZE = {PROG_BANK_SIZE};
window.QUARTZ_DEFAULT_PROG_STATES = {json.dumps(DEFAULT_PROG_STATES)};
window.quartzWaitForTerminal = async function(attempts) {{
    attempts = attempts || 0;
    var ta = document.querySelector('.quartz-terminal textarea');
    if (ta) return ta;
    if (attempts > 120) return null;
    await new Promise(function(resolve) {{ setTimeout(resolve, 100); }});
    return window.quartzWaitForTerminal(attempts + 1);
}};
window.quartzPaintTerminal = function(ta, text, quiet) {{
    if (!ta) return;
    ta.value = text;
    if (quiet || window.quartzTypewriterActive) return;
    ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
    var vh = window.innerHeight || document.documentElement.clientHeight || 0;
    if (vh > 0 && typeof syncTerminalOverflow === 'function') syncTerminalOverflow(vh);
}};
window.quartzBeginTypewriter = function(ta) {{
    if (!ta) return;
    var vh = window.innerHeight || document.documentElement.clientHeight || 0;
    if (vh > 0 && typeof syncTerminalOverflow === 'function') syncTerminalOverflow(vh);
    window.quartzTypewriterActive = true;
    ta.classList.add('quartz-typewriter-active');
    ta.style.overflowY = 'hidden';
}};
window.quartzEndTypewriter = function(ta) {{
    window.quartzTypewriterActive = false;
    if (!ta) ta = document.querySelector('.quartz-terminal textarea');
    if (ta) {{
        ta.classList.remove('quartz-typewriter-active');
        ta.classList.remove('quartz-startup-centered');
        ta.style.overflowY = '';
        ta.style.textAlign = '';
    }}
    var vh = window.innerHeight || document.documentElement.clientHeight || 0;
    if (vh > 0 && typeof syncTerminalOverflow === 'function') syncTerminalOverflow(vh);
    if (typeof fitDisplayBacking === 'function') fitDisplayBacking();
}};
window.quartzSetHomeLed = function(on) {{
    document.querySelectorAll('button.quartz-home-btn').forEach(function(btn) {{
        btn.classList.toggle('quartz-home-active', !!on);
        btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    }});
}};
window.quartzSetProgLed = function(index, on) {{
    document.querySelectorAll('button.quartz-prog-id-' + index).forEach(function(btn) {{
        btn.classList.toggle('quartz-prog-active', !!on);
        btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    }});
}};
window.quartzClearProgLeds = function() {{
    var i;
    for (i = 1; i <= window.QUARTZ_PROG_BANK_SIZE; i++) window.quartzSetProgLed(i, false);
}};
window.quartzSetAllProgLeds = function(on) {{
    var i;
    for (i = 1; i <= window.QUARTZ_PROG_BANK_SIZE; i++) window.quartzSetProgLed(i, !!on);
}};
window.quartzSyncProgLedsForCharIndex = function(index, strLen) {{
    if (index === strLen - 1) {{
        window.quartzSetAllProgLeds(false);
        return;
    }}
    window.quartzSetAllProgLeds(index % 2 === 0);
}};
window.quartzResetAllProgStates = function(active) {{
    var bankSize = window.QUARTZ_PROG_BANK_SIZE || 16;
    var i;
    if (!window.quartzProgStates) window.quartzProgStates = {{}};
    for (i = 1; i <= bankSize; i++) window.quartzProgStates['prog' + i] = !!active;
    if (typeof window.quartzApplyProgStates === 'function') {{
        window.quartzApplyProgStates(window.quartzProgStates);
    }}
    if (typeof window.quartzSaveProgStates === 'function') {{
        window.quartzSaveProgStates(window.quartzProgStates);
    }}
}};
window.quartzApplyDefaultProgStates = function() {{
    var bankSize = window.QUARTZ_PROG_BANK_SIZE || 16;
    var i;
    if (!window.quartzProgStates) window.quartzProgStates = {{}};
    if (window.QUARTZ_DEFAULT_PROG_STATES) {{
        Object.assign(window.quartzProgStates, window.QUARTZ_DEFAULT_PROG_STATES);
    }} else {{
        for (i = 1; i <= bankSize; i++) window.quartzProgStates['prog' + i] = false;
        window.quartzProgStates.prog1 = true;
    }}
    if (typeof window.quartzApplyProgStates === 'function') {{
        window.quartzApplyProgStates(window.quartzProgStates);
    }}
    if (typeof window.quartzSaveProgStates === 'function') {{
        window.quartzSaveProgStates(window.quartzProgStates);
    }}
}};
window.quartzRunStartupSequence = async function(ta, options) {{
    options = options || {{}};
    var charDelay = options.charDelay != null ? options.charDelay : window.QUARTZ_CHAR_DELAY;
    var postDelay = options.postDelay != null ? options.postDelay : 0;
    var persistBoot = !!options.persistBoot;
    var runProgDiagnostic = !!options.runProgDiagnostic;
    if (!ta) ta = await window.quartzWaitForTerminal(0);
    var strLen = window.QUARTZ_STARTUP_STRING.length;
    if (runProgDiagnostic) {{
        window.quartzClearProgLeds();
        ta.classList.add('quartz-startup-centered');
    }}
    window.quartzBeginTypewriter(ta);
    var text = runProgDiagnostic ? '' : '> ';
    window.quartzPaintTerminal(ta, text, true);
    var i;
    for (i = 0; i < strLen; i++) {{
        await new Promise(function(resolve) {{ setTimeout(resolve, charDelay); }});
        text += window.QUARTZ_STARTUP_STRING.charAt(i);
        window.quartzPaintTerminal(ta, text, true);
        if (runProgDiagnostic) window.quartzSyncProgLedsForCharIndex(i, strLen);
    }}
    for (i = 0; i < 5; i++) {{
        await new Promise(function(resolve) {{ setTimeout(resolve, charDelay); }});
        text += '.';
        window.quartzPaintTerminal(ta, text, true);
    }}
    if (postDelay > 0) {{
        await new Promise(function(resolve) {{ setTimeout(resolve, postDelay); }});
    }}
    if (persistBoot) {{
        sessionStorage.setItem('qvpic-boot-complete', '1');
    }}
    window.quartzEndTypewriter(ta);
    if (runProgDiagnostic) {{
        window.quartzSetAllProgLeds(false);
        if (typeof window.quartzResetAllProgStates === 'function') {{
            window.quartzResetAllProgStates(false);
        }}
    }}
    window.quartzPaintTerminal(ta, window.QUARTZ_HOME_MENU_TEXT);
    window.quartzSetHomeLed(true);
    return window.QUARTZ_HOME_MENU_TEXT;
}};
window.quartzClickStartupDone = function() {{
    var btn = document.querySelector('button.quartz-startup-done-btn');
    if (btn) btn.click();
}};
</script>
"""


def _quartz_games_js_block() -> str:
    return """
<script>
(function() {
    var PHOSPHOR = '#00D4FF';
    var PHOSPHOR_BRIGHT = '#80EEFF';
    var PHOSPHOR_DIM = '#00A8CC';
    var SPRITES = {
        squid: [
            ['00100000100','00010001000','00111111100','01101110110','11111111111','10111111101','10100000101','01101111010'],
            ['00100000100','00010001000','10111111101','11101110111','11111111111','00111111100','10100000101','01011111010']
        ],
        crab: [
            ['00010001000','00100000100','00011111000','01101111110','11111111111','10111111101','10100000101','00110110010'],
            ['00010001000','00100000100','10011111001','11101111111','11111111111','00111111100','10010001001','01100110100']
        ],
        octopus: [
            ['00001110000','00011111000','01111111110','11101110111','11111111111','00100100100','10100000101','01000000010'],
            ['00001110000','00011111000','01111111110','11101110111','11111111111','00100100100','00100000100','10000000001']
        ],
        player: ['00011000000','00111100100','01111111110','11111111111','11111111111','11111111111','11111111111','00000000000'],
        life: ['00011000000','00111100100','01111111110','11111111111']
    };
    var ROW_TYPES = ['squid', 'squid', 'crab', 'crab', 'octopus', 'octopus', 'octopus', 'octopus', 'octopus'];
    function spriteRows(bits) {
        return bits.map(function(row) {
            return row.split('').map(function(ch) { return ch === '1' ? 1 : 0; });
        });
    }
    function spriteFrames(type) {
        return SPRITES[type].map(spriteRows);
    }
    function displayApertureRect() {
        var backing = document.querySelector('.quartz-display-backing');
        if (backing) {
            var br = backing.getBoundingClientRect();
            if (br.width >= 40 && br.height >= 40) return br;
        }
        var ta = document.querySelector('.quartz-terminal textarea');
        if (ta) {
            var tr = ta.getBoundingClientRect();
            if (tr.width >= 40 && tr.height >= 40) return tr;
        }
        var bay = document.querySelector('.quartz-display-bay');
        var input = document.querySelector('.quartz-input-row');
        if (bay) {
            var bayRect = bay.getBoundingClientRect();
            var inputH = input ? input.getBoundingClientRect().height : 0;
            if (bayRect.width >= 40 && bayRect.height - inputH >= 40) {
                return {
                    top: bayRect.top,
                    left: bayRect.left,
                    width: bayRect.width,
                    height: Math.max(40, bayRect.height - inputH - 6),
                    right: bayRect.right,
                    bottom: bayRect.bottom - inputH
                };
            }
        }
        return null;
    }
    function terminalDisplayRect() {
        return displayApertureRect();
    }
    function ensureInvadersStage() {
        var stage = document.querySelector('.quartz-invaders-stage');
        if (!stage) {
            stage = document.createElement('div');
            stage.className = 'quartz-invaders-stage';
            stage.setAttribute('aria-hidden', 'true');
            stage.innerHTML =
                '<canvas class="quartz-invaders-canvas"></canvas>' +
                '<div class="quartz-invaders-crt" aria-hidden="true"></div>';
        }
        if (stage.parentElement !== document.body) {
            document.body.appendChild(stage);
        }
        return stage;
    }
    function fitGameStage() {
        var stage = ensureInvadersStage();
        var tr = displayApertureRect();
        if (!stage || !tr) return null;
        stage.style.position = 'fixed';
        stage.style.top = Math.round(tr.top) + 'px';
        stage.style.left = Math.round(tr.left) + 'px';
        stage.style.width = Math.round(tr.width) + 'px';
        stage.style.height = Math.round(tr.height) + 'px';
        stage.style.zIndex = '100001';
        return stage;
    }
    function resizeCanvas(g) {
        var stage = fitGameStage();
        if (!stage) return false;
        var canvas = stage.querySelector('.quartz-invaders-canvas');
        if (!canvas) return false;
        var rect = stage.getBoundingClientRect();
        var w = Math.max(1, Math.floor(rect.width));
        var h = Math.max(1, Math.floor(rect.height));
        if (w < 48 || h < 48) return false;
        canvas.width = w;
        canvas.height = h;
        g.width = w;
        g.height = h;
        g.scale = Math.max(2, Math.min(Math.floor(w / 72), Math.floor(h / 56)));
        g.gridW = Math.max(56, Math.floor(w / g.scale));
        g.gridH = Math.max(48, Math.floor(h / g.scale));
        g.offsetX = Math.max(0, Math.floor((w - g.gridW * g.scale) / 2));
        g.offsetY = Math.max(0, Math.floor((h - g.gridH * g.scale) / 2));
        g.player.x = Math.floor(g.gridW / 2);
        g.player.y = g.gridH - 6;
        g.ctx = canvas.getContext('2d');
        if (!g.ctx) return false;
        g.ctx.imageSmoothingEnabled = false;
        if (!g.layoutReady || g._lastGridW !== g.gridW || g._lastGridH !== g.gridH) {
            g._lastGridW = g.gridW;
            g._lastGridH = g.gridH;
            g.layoutReady = true;
            resetWave(g);
        }
        return true;
    }
    function activateGameOverlay(g) {
        var stage = ensureInvadersStage();
        if (stage) stage.style.display = 'block';
        document.body.classList.add('quartz-game-on');
        var torus = document.querySelector('.quartz-torus-stage');
        if (torus) torus.style.display = 'none';
        var ta = document.querySelector('.quartz-terminal textarea');
        if (ta) ta.classList.add('quartz-game-active');
        g.overlayReady = true;
    }
    function deactivateGameOverlay(g) {
        if (g && g.overlayReady) {
            document.body.classList.remove('quartz-game-on');
            var ta = document.querySelector('.quartz-terminal textarea');
            if (ta) ta.classList.remove('quartz-game-active');
        }
        var stage = document.querySelector('.quartz-invaders-stage');
        if (stage) {
            stage.style.display = 'none';
            stage.style.position = '';
            stage.style.top = '';
            stage.style.left = '';
            stage.style.width = '';
            stage.style.height = '';
            stage.style.zIndex = '';
        }
    }
    function bootGameSurface(g, attempt, done) {
        attempt = attempt || 0;
        if (!g.running) return;
        if (typeof fitPanel === 'function') fitPanel();
        var sized = resizeCanvas(g);
        if (sized) {
            renderGame(g);
            activateGameOverlay(g);
            if (!g.loopId) {
                g.loopId = setInterval(function() { tickGame(g); }, 33);
            }
            if (!g.attractMode && typeof window.quartzClickGameStartDone === 'function') {
                window.quartzClickGameStartDone();
            }
            if (done) done(true);
            return;
        }
        if (attempt >= 48) {
            if (done) done(false);
            return;
        }
        setTimeout(function() {
            bootGameSurface(g, attempt + 1, done);
        }, attempt < 12 ? 60 : 120);
    }
    function makeBunker(x, y, scale) {
        var pattern = [
            '000111111110000',
            '011111111111110',
            '111111111111111',
            '111111111111111',
            '111111111111111',
            '111111111111111',
            '111111111111111',
            '111111111111111',
            '111111111111111',
            '011111111111110',
            '001111111111100'
        ];
        var cells = [];
        var r, c, row;
        for (r = 0; r < pattern.length; r++) {
            row = pattern[r];
            for (c = 0; c < row.length; c++) {
                if (row.charAt(c) === '1') {
                    cells.push({ x: x + c, y: y + r, alive: true });
                }
            }
        }
        return { x: x, y: y, scale: scale, cells: cells };
    }
    function initBunkers(g) {
        var bw = 15;
        var gap = Math.max(4, Math.floor((g.gridW - bw * 4) / 5));
        var y = Math.floor(g.gridH * 0.74);
        var bunkers = [];
        var i, x;
        for (i = 0; i < 4; i++) {
            x = gap + i * (bw + gap);
            bunkers.push(makeBunker(x, y, 1));
        }
        return bunkers;
    }
    function initInvaders(g) {
        var invaders = [];
        var cols = 11;
        var rows = ROW_TYPES.length;
        var spacingX = 14;
        var spacingY = 10;
        var totalW = cols * spacingX;
        var startX = Math.floor((g.gridW - totalW) / 2);
        var startY = Math.floor(g.gridH * 0.22);
        var r, c, type;
        for (r = 0; r < rows; r++) {
            type = ROW_TYPES[r];
            for (c = 0; c < cols; c++) {
                invaders.push({
                    type: type,
                    x: startX + c * spacingX,
                    y: startY + r * spacingY,
                    alive: true,
                    row: r
                });
            }
        }
        return invaders;
    }
    function resetWave(g) {
        g.invaders = initInvaders(g);
        g.direction = 1;
        g.stepDown = false;
        g.invaderTick = 0;
        g.enemyBullets = [];
        g.playerBullet = null;
        g.bunkers = initBunkers(g);
        g.invaderSpeed = g.attractMode
            ? Math.max(26, 34 - g.wave * 2)
            : Math.max(4, 22 - g.wave * 2);
        g.player.x = Math.floor(g.gridW / 2);
        g.player.y = g.gridH - 6;
    }
    function initGameState(opts) {
        opts = opts || {};
        var g = {
            running: true,
            attractMode: !!opts.attract,
            width: 0,
            height: 0,
            scale: 3,
            gridW: 56,
            gridH: 64,
            offsetX: 0,
            offsetY: 0,
            layoutReady: false,
            _lastGridW: 0,
            _lastGridH: 0,
            ctx: null,
            wave: 1,
            score: 0,
            hiScore: parseInt(localStorage.getItem('qvpic-invaders-hi') || '0', 10) || 0,
            lives: 3,
            player: { x: 28, y: 58 },
            invaders: [],
            direction: 1,
            stepDown: false,
            invaderTick: 0,
            invaderSpeed: 20,
            animFrame: 0,
            playerBullet: null,
            enemyBullets: [],
            bunkers: [],
            keys: {},
            loopId: null,
            enemyShootCooldown: 0,
            gameOver: false,
            attractTimer: 0,
            attractShootTimer: 0,
            attractResetTimer: 0,
            warmupTicks: opts.attract ? 120 : 30
        };
        return g;
    }
    function tickAttract(g) {
        if (!g.attractMode || g.gameOver) return;
        g.attractTimer += 1;
        g.player.x = (g.gridW * 0.5) + Math.sin(g.attractTimer * 0.045) * 14;
        g.player.x = Math.max(6, Math.min(g.gridW - 6, g.player.x));
        g.attractShootTimer += 1;
        if (g.attractShootTimer > 42 && !g.playerBullet) {
            firePlayerBullet(g);
            g.attractShootTimer = 0;
        }
    }
    function leaveAttractMode(g) {
        if (!g.attractMode) return;
        g.attractMode = false;
        g.attractTimer = 0;
        g.attractShootTimer = 0;
    }
    function drawPixel(ctx, x, y, scale, bright) {
        ctx.fillStyle = bright ? PHOSPHOR_BRIGHT : PHOSPHOR;
        ctx.fillRect(Math.floor(x * scale), Math.floor(y * scale), scale, scale);
    }
    function drawSprite(ctx, sprite, x, y, scale, bright) {
        var r, c;
        for (r = 0; r < sprite.length; r++) {
            for (c = 0; c < sprite[r].length; c++) {
                if (sprite[r][c]) drawPixel(ctx, x + c, y + r, scale, bright);
            }
        }
    }
    function drawText(ctx, text, x, y, scale, bright) {
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.fillStyle = bright ? PHOSPHOR_BRIGHT : PHOSPHOR;
        ctx.font = 'bold ' + Math.max(8, Math.floor(scale * 5.2)) + 'px "Courier New", Courier, monospace';
        ctx.shadowColor = PHOSPHOR;
        ctx.shadowBlur = bright ? 3 : 1;
        ctx.fillText(text, Math.floor(x * scale), Math.floor((y + 4) * scale));
        ctx.restore();
    }
    function drawBunker(ctx, bunker, scale) {
        bunker.cells.forEach(function(cell) {
            if (!cell.alive) return;
            drawPixel(ctx, cell.x, cell.y, scale, false);
        });
    }
    function renderGame(g) {
        if (!g.ctx) return;
        var ctx = g.ctx;
        var scale = g.scale;
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, g.width, g.height);
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, g.width, g.height);
        ctx.translate(g.offsetX || 0, g.offsetY || 0);
        drawText(ctx, 'SPACE INVADERS', Math.max(2, Math.floor(g.gridW / 2) - 8), 2, scale, true);
        drawText(ctx, 'SCORE ' + String(g.score).padStart(4, '0'), 2, 9, scale, false);
        drawText(ctx, 'HI ' + String(Math.max(g.score, g.hiScore)).padStart(4, '0'),
            Math.max(24, g.gridW - 16), 9, scale, false);
        var frame = g.animFrame % 2;
        g.invaders.forEach(function(inv) {
            if (!inv.alive) return;
            var frames = spriteFrames(inv.type);
            drawSprite(ctx, frames[frame], inv.x, inv.y, scale, inv.type === 'squid');
        });
        g.bunkers.forEach(function(b) { drawBunker(ctx, b, scale); });
        if (g.playerBullet) {
            drawPixel(ctx, g.playerBullet.x, g.playerBullet.y, scale, true);
            if (g.playerBullet.y > 0) drawPixel(ctx, g.playerBullet.x, g.playerBullet.y - 1, scale, true);
        }
        g.enemyBullets.forEach(function(b) {
            drawPixel(ctx, b.x, b.y, scale, false);
            drawPixel(ctx, b.x, b.y + 0.5, scale, false);
        });
        if (!g.gameOver) {
            drawSprite(ctx, spriteRows(SPRITES.player), g.player.x - 5, g.player.y, scale, true);
        }
        if (g.attractMode && !g.gameOver) {
            var blink = Math.floor(g.animFrame / 18) % 2 === 0;
            drawText(ctx, 'LIVES', 2, g.gridH - 4, scale, false);
            var lifeSprite = spriteRows(SPRITES.life);
            var i;
            for (i = 0; i < g.lives; i++) {
                drawSprite(ctx, lifeSprite, 14 + i * 8, g.gridH - 6, scale, false);
            }
            if (blink) {
                drawText(ctx, 'ARROWS TO PLAY', Math.max(18, g.gridW - 28), g.gridH - 4, scale, true);
            }
        } else {
            drawText(ctx, 'LIVES', 2, g.gridH - 4, scale, false);
            var lifeSprite = spriteRows(SPRITES.life);
            var j;
            for (j = 0; j < g.lives; j++) {
                drawSprite(ctx, lifeSprite, 14 + j * 8, g.gridH - 6, scale, false);
            }
        }
        if (g.gameOver && !g.attractMode) {
            drawText(ctx, 'GAME OVER', Math.floor(g.gridW / 2) - 5, Math.floor(g.gridH / 2), scale, true);
            drawText(ctx, 'ESC TO QUIT', Math.floor(g.gridW / 2) - 5, Math.floor(g.gridH / 2) + 8, scale, false);
        }
        ctx.restore();
    }
    function aliveInvaders(g) {
        return g.invaders.filter(function(inv) { return inv.alive; });
    }
    function invaderBounds(g) {
        var alive = aliveInvaders(g);
        if (!alive.length) return null;
        var minX = 999, maxX = -999, maxY = -999;
        alive.forEach(function(inv) {
            minX = Math.min(minX, inv.x);
            maxX = Math.max(maxX, inv.x + 11);
            maxY = Math.max(maxY, inv.y + 8);
        });
        return { minX: minX, maxX: maxX, maxY: maxY };
    }
    function hitBunker(g, x, y) {
        var hit = false;
        g.bunkers.forEach(function(bunker) {
            bunker.cells.forEach(function(cell) {
                if (!cell.alive || hit) return;
                if (Math.abs(cell.x - x) < 1.2 && Math.abs(cell.y - y) < 1.2) {
                    cell.alive = false;
                    hit = true;
                }
            });
        });
        return hit;
    }
    function firePlayerBullet(g) {
        if (g.playerBullet || g.gameOver) return;
        g.playerBullet = { x: g.player.x, y: g.player.y - 1, vy: -1.4 };
    }
    function fireEnemyBullet(g) {
        var alive = aliveInvaders(g);
        if (!alive.length) return;
        var shooter = alive[Math.floor(Math.random() * alive.length)];
        g.enemyBullets.push({
            x: shooter.x + 5,
            y: shooter.y + 8,
            vy: 0.55 + Math.random() * 0.35
        });
    }
    function tickGame(g) {
        if (!g.running) return;
        g.animFrame += 1;
        if (!g.gameOver) {
            if (g.warmupTicks > 0) g.warmupTicks -= 1;
            if (g.attractMode) {
                tickAttract(g);
            } else {
                var speed = g.keys.ArrowLeft || g.keys.ArrowRight ? 0.85 : 0.55;
                if (g.keys.ArrowLeft) g.player.x = Math.max(6, g.player.x - speed);
                if (g.keys.ArrowRight) g.player.x = Math.min(g.gridW - 6, g.player.x + speed);
                if (g.keys[' ']) firePlayerBullet(g);
            }
            g.invaderTick += 1;
            if (g.invaderTick >= g.invaderSpeed) {
                g.invaderTick = 0;
                var bounds = invaderBounds(g);
                if (bounds) {
                    var step = g.direction * 1.2;
                    if ((g.direction > 0 && bounds.maxX + step >= g.gridW - 2) ||
                        (g.direction < 0 && bounds.minX + step <= 2)) {
                        g.direction *= -1;
                        aliveInvaders(g).forEach(function(inv) { inv.y += 1.4; });
                    } else {
                        aliveInvaders(g).forEach(function(inv) { inv.x += step; });
                    }
                }
            }
            if (g.warmupTicks <= 0 && g.enemyShootCooldown <= 0) {
                fireEnemyBullet(g);
                g.enemyShootCooldown = g.attractMode
                    ? Math.max(52, 68 - g.wave * 2)
                    : Math.max(18, 48 - g.wave * 3);
            } else if (g.enemyShootCooldown > 0) {
                g.enemyShootCooldown -= 1;
            }
        }
        if (g.playerBullet) {
            g.playerBullet.y += g.playerBullet.vy;
            var bx = g.playerBullet.x;
            var by = g.playerBullet.y;
            var hit = false;
            if (by < 12) {
                g.playerBullet = null;
            } else {
                g.invaders.forEach(function(inv) {
                    if (!inv.alive || hit) return;
                    if (bx >= inv.x && bx <= inv.x + 11 && by >= inv.y && by <= inv.y + 8) {
                        inv.alive = false;
                        g.score += (3 - Math.min(2, Math.floor(inv.row / 3))) * 10 + 10;
                        if (g.score > g.hiScore) {
                            g.hiScore = g.score;
                            localStorage.setItem('qvpic-invaders-hi', String(g.hiScore));
                        }
                        hit = true;
                    }
                });
                if (!hit && hitBunker(g, bx, by)) hit = true;
                if (hit) g.playerBullet = null;
                else if (by < 0) g.playerBullet = null;
            }
        }
        g.enemyBullets = g.enemyBullets.filter(function(b) {
            b.y += b.vy;
            if (b.y > g.gridH) return false;
            if (!g.gameOver && g.warmupTicks <= 0 &&
                Math.abs(b.x - g.player.x) < 3 && b.y >= g.player.y && b.y <= g.player.y + 7) {
                if (g.attractMode) {
                    return false;
                }
                g.lives -= 1;
                if (g.lives <= 0) g.gameOver = true;
                return false;
            }
            if (hitBunker(g, b.x, b.y)) return false;
            return true;
        });
        if (!g.gameOver) {
            var bounds = invaderBounds(g);
            if (bounds && bounds.maxY >= g.player.y - 2) {
                if (g.attractMode) {
                    resetWave(g);
                } else {
                    g.gameOver = true;
                }
            }
            if (!aliveInvaders(g).length) {
                g.wave += 1;
                resetWave(g);
            }
        }
        renderGame(g);
    }
    window.quartzFitGameStage = fitGameStage;
    window.quartzTerminalReady = function() {
        return !!terminalDisplayRect();
    };
    window.quartzStartPlayerGame = function() {
        if (window.quartzGame && window.quartzGame.running) return;
        window.quartzStartSpaceInvaders({ attract: false });
    };
    window.quartzBeginPlayerMode = window.quartzStartPlayerGame;
    window.quartzStartSpaceInvaders = function(opts) {
        opts = opts || {};
        if (window.quartzGame && window.quartzGame.running) return;
        if (typeof fitPanel === 'function') fitPanel();
        var cmd = document.querySelector('.quartz-cmd-input textarea');
        if (cmd) cmd.blur();
        var g = initGameState(opts);
        g.overlayReady = false;
        window.quartzGame = g;
        bootGameSurface(g, 0, function(ok) {
            if (ok || !g.running) return;
            g.running = false;
            window.quartzGame = null;
            deactivateGameOverlay(g);
        });
        g.keyDown = function(e) {
            if (!g.running) return;
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
            }
            if (g.attractMode && (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === ' ')) {
                leaveAttractMode(g);
            }
            if (e.key === ' ') {
                g.keys[' '] = true;
                firePlayerBullet(g);
            } else if (e.key === 'Escape') {
                window.quartzStopSpaceInvaders();
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                g.keys[e.key] = true;
            }
        };
        g.keyUp = function(e) {
            if (e.key === ' ') g.keys[' '] = false;
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') g.keys[e.key] = false;
        };
        document.addEventListener('keydown', g.keyDown, true);
        document.addEventListener('keyup', g.keyUp, true);
        g.resizeHandler = function() {
            if (!g.running || !g.overlayReady) return;
            if (typeof fitPanel === 'function') fitPanel();
            if (resizeCanvas(g)) renderGame(g);
        };
        window.addEventListener('resize', g.resizeHandler);
    };
    window.quartzStopSpaceInvaders = function() {
        var g = window.quartzGame;
        if (!g) return;
        g.running = false;
        if (g.loopId) clearInterval(g.loopId);
        document.removeEventListener('keydown', g.keyDown, true);
        document.removeEventListener('keyup', g.keyUp, true);
        if (g.resizeHandler) window.removeEventListener('resize', g.resizeHandler);
        deactivateGameOverlay(g);
        window.quartzGame = null;
        document.body.classList.remove('quartz-game-on');
        var torus = document.querySelector('.quartz-torus-stage');
        if (torus && window.quartzTorusPrefs && window.quartzTorusPrefs.visible) {
            torus.style.display = 'block';
        }
        var ta = document.querySelector('.quartz-terminal textarea');
        if (ta) {
            ta.classList.remove('quartz-game-active');
            if (window.QUARTZ_HOME_MENU_TEXT) {
                window.quartzPaintTerminal(ta, window.QUARTZ_HOME_MENU_TEXT);
            }
            window.quartzSetHomeLed(true);
        }
        if (typeof fitPanel === 'function') fitPanel();
        if (typeof window.quartzClickGameQuit === 'function') {
            window.quartzClickGameQuit();
        }
    };
    window.quartzClickGameQuit = function() {
        var btn = document.querySelector('button.quartz-game-quit-btn');
        if (btn) btn.click();
    };
    window.quartzClickGameStartDone = function() {
        var btn = document.querySelector('button.quartz-game-start-done-btn');
        if (btn) btn.click();
    };
})();
</script>
"""


_QUARTZ_HEAD_TEMPLATE = """
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
        ['.quartz-top-tabs', '.quartz-prog-bank', '.quartz-footer-wrap', '.quartz-settings'].forEach(function(sel) {
            var el = document.querySelector(sel);
            if (el) chrome += el.offsetHeight;
        });
        chrome += 24;
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
        layoutTorusStage();
    }
    var TORUS_STORE_KEY = 'qvpic-torus-prefs';
    function loadTorusPrefs() {
        try {
            var raw = localStorage.getItem(TORUS_STORE_KEY);
            if (raw) return JSON.parse(raw);
        } catch (e) {}
        return {
            visible: true, custom: false, left: null, top: null, width: null, height: null,
            rotX: null, rotY: null, rotZ: null
        };
    }
    function saveTorusPrefs(prefs) {
        try { localStorage.setItem(TORUS_STORE_KEY, JSON.stringify(prefs)); } catch (e) {}
    }
    window.quartzTorusPrefs = loadTorusPrefs();
    var PROG_STORE_KEY = 'qvpic-prog-states';
    var PROG_BANK_SIZE = 16;
    function defaultProgStates() {
        var states = {};
        var i;
        for (i = 1; i <= PROG_BANK_SIZE; i++) states['prog' + i] = false;
        states.prog1 = true;
        return states;
    }
    function loadProgStates() {
        try {
            var raw = localStorage.getItem(PROG_STORE_KEY);
            if (raw) {
                return Object.assign(defaultProgStates(), JSON.parse(raw));
            }
        } catch (e) {}
        return defaultProgStates();
    }
    function saveProgStates(states) {
        try { localStorage.setItem(PROG_STORE_KEY, JSON.stringify(states)); } catch (e) {}
    }
    window.quartzProgStates = loadProgStates();
    function applyProgStates(states) {
        var i, pid, on;
        for (i = 1; i <= PROG_BANK_SIZE; i++) {
            pid = 'prog' + i;
            on = !!states[pid];
            document.querySelectorAll('button.quartz-prog-id-' + i).forEach(function(btn) {
                btn.classList.toggle('quartz-prog-active', on);
                btn.setAttribute('aria-pressed', on ? 'true' : 'false');
            });
        }
        if (states.prog1 !== undefined) {
            applyTorusVisibility(!!states.prog1, false);
        }
        syncTorusInteractionMode();
    }
    function torusMoveMode() {
        return !!(window.quartzProgStates && window.quartzProgStates.prog9);
    }
    function syncTorusInteractionMode() {
        var stage = document.querySelector('.quartz-torus-stage');
        var moveMode = torusMoveMode();
        window.quartzTorusSpinPaused = moveMode;
        if (stage) {
            stage.classList.toggle('quartz-torus-move-mode', moveMode);
        }
    }
    function isNearTorusEdge(stage, clientX, clientY, margin) {
        var r = stage.getBoundingClientRect();
        var m = margin || 14;
        return (clientX - r.left < m) || (r.right - clientX < m)
            || (clientY - r.top < m) || (r.bottom - clientY < m);
    }
    function toggleProg(index) {
        var pid = 'prog' + index;
        window.quartzProgStates[pid] = !window.quartzProgStates[pid];
        saveProgStates(window.quartzProgStates);
        applyProgStates(window.quartzProgStates);
    }
    window.quartzApplyProgStates = applyProgStates;
    window.quartzSaveProgStates = saveProgStates;
    function initProgToggles() {
        applyProgStates(window.quartzProgStates);
        document.querySelectorAll('.quartz-prog-bank').forEach(function(bank) {
            if (bank._quartzProgInit) return;
            bank._quartzProgInit = true;
            bank.addEventListener('click', function(e) {
                var btn = e.target && e.target.closest('button.quartz-prog-toggle');
                if (!btn) return;
                var m = btn.className.match(/quartz-prog-id-(\\d+)/);
                if (!m) return;
                toggleProg(parseInt(m[1], 10));
                clickPulse(btn);
            });
        });
    }
    function applyTorusVisibility(visible, savePrefs) {
        if (savePrefs === undefined) savePrefs = true;
        var stage = document.querySelector('.quartz-torus-stage');
        window.quartzTorusPrefs.visible = !!visible;
        if (stage) {
            stage.style.display = visible ? 'block' : 'none';
            stage.classList.toggle('quartz-torus-hidden', !visible);
        }
        if (savePrefs) saveTorusPrefs(window.quartzTorusPrefs);
    }
    function defaultTorusBox() {
        var tr = terminalOutputRect();
        if (!tr || tr.width < 40 || tr.height < 40) return null;
        var w = tr.width;
        var h = tr.height;
        var pad = Math.max(6, Math.round(Math.min(w, h) * 0.045));
        var rightX0 = w * 0.5 + pad * 0.5;
        var rightW = Math.max(20, w * 0.5 - pad * 1.5);
        var availH = Math.max(20, h - pad * 2);
        var size = Math.floor(Math.min(rightW, availH) * 0.82);
        var originX = tr.left + rightX0 + rightW * 0.5;
        var originY = tr.top + h * 0.5;
        return {
            left: Math.round(originX - size * 0.5),
            top: Math.round(originY - size * 0.5),
            width: size,
            height: size
        };
    }
    function clampTorusOnScreen(stage, left, top) {
        var vw = window.innerWidth || document.documentElement.clientWidth || 0;
        var vh = window.innerHeight || document.documentElement.clientHeight || 0;
        var w = stage.offsetWidth || 120;
        var h = stage.offsetHeight || 120;
        var minVis = 40;
        return {
            left: Math.max(-w + minVis, Math.min(vw - minVis, left)),
            top: Math.max(0, Math.min(vh - minVis, top))
        };
    }
    function applyTorusBox(box) {
        var stage = document.querySelector('.quartz-torus-stage');
        var frame = stage && stage.querySelector('.quartz-torus-frame');
        if (!stage || !frame || !box) return;
        stage.style.position = 'fixed';
        stage.style.left = box.left + 'px';
        stage.style.top = box.top + 'px';
        stage.style.width = box.width + 'px';
        stage.style.height = box.height + 'px';
        stage.style.zIndex = '99999';
        stage.style.overflow = 'visible';
        frame.style.position = 'absolute';
        frame.style.inset = '0';
        frame.style.left = '0';
        frame.style.top = '0';
        frame.style.right = '0';
        frame.style.bottom = '0';
        frame.style.width = '100%';
        frame.style.height = '100%';
        frame.style.transform = 'none';
        frame.style.display = 'flex';
        frame.style.alignItems = 'center';
        frame.style.justifyContent = 'center';
        stage.style.display = window.quartzTorusPrefs.visible ? 'block' : 'none';
    }
    function layoutTorusStage() {
        if (window.quartzTorusDragging || window.quartzTorusRotating || window.quartzTorusResizing) {
            return;
        }
        var stage = document.querySelector('.quartz-torus-stage');
        if (!stage) return;
        if (!window.quartzTorusPrefs.visible) {
            stage.style.display = 'none';
            return;
        }
        var box;
        if (window.quartzTorusPrefs.custom
            && window.quartzTorusPrefs.left != null
            && window.quartzTorusPrefs.top != null) {
            var clamped = clampTorusOnScreen(
                stage,
                window.quartzTorusPrefs.left,
                window.quartzTorusPrefs.top
            );
            box = {
                left: clamped.left,
                top: clamped.top,
                width: window.quartzTorusPrefs.width || 160,
                height: window.quartzTorusPrefs.height || 160
            };
        } else {
            box = defaultTorusBox();
        }
        if (!box) {
            stage.style.display = 'none';
            return;
        }
        applyTorusBox(box);
    }
    function saveTorusOrientation() {
        if (!window.quartzTorus || !window.quartzTorus.state) return;
        var s = window.quartzTorus.state;
        window.quartzTorusPrefs.rotX = s.rotX;
        window.quartzTorusPrefs.rotY = s.rotY;
        window.quartzTorusPrefs.rotZ = s.rotZ;
        saveTorusPrefs(window.quartzTorusPrefs);
    }
    function initTorusInteraction() {
        var stage = document.querySelector('.quartz-torus-stage');
        if (!stage || stage._quartzInteractInit) return;
        stage._quartzInteractInit = true;
        var offsetX = 0;
        var offsetY = 0;
        var lastRotX = 0;
        var lastRotY = 0;
        var resizeStartX = 0;
        var resizeStartSize = 0;
        stage.addEventListener('mousedown', function(e) {
            if (e.button !== 0 || !window.quartzTorusPrefs.visible) return;
            if (torusMoveMode()) {
                if (isNearTorusEdge(stage, e.clientX, e.clientY, 14)) {
                    window.quartzTorusResizing = true;
                    resizeStartX = e.clientX;
                    resizeStartSize = stage.offsetWidth;
                    stage.classList.add('quartz-torus-resizing');
                } else {
                    window.quartzTorusDragging = true;
                    stage.classList.add('quartz-torus-dragging');
                    var rect = stage.getBoundingClientRect();
                    offsetX = e.clientX - rect.left;
                    offsetY = e.clientY - rect.top;
                }
            } else {
                window.quartzTorusRotating = true;
                window.quartzTorusSpinPaused = true;
                stage.classList.add('quartz-torus-rotating');
                lastRotX = e.clientX;
                lastRotY = e.clientY;
            }
            e.preventDefault();
        });
        stage.addEventListener('mousemove', function(e) {
            if (window.quartzTorusDragging || window.quartzTorusRotating || window.quartzTorusResizing) {
                return;
            }
            if (!torusMoveMode()) return;
            stage.style.cursor = isNearTorusEdge(stage, e.clientX, e.clientY, 14)
                ? 'nwse-resize' : 'move';
        });
        document.addEventListener('mousemove', function(e) {
            if (window.quartzTorusResizing) {
                var dx = e.clientX - resizeStartX;
                var size = Math.max(56, Math.min(420, Math.round(resizeStartSize + dx)));
                stage.style.width = size + 'px';
                stage.style.height = size + 'px';
                return;
            }
            if (window.quartzTorusRotating) {
                var rdx = e.clientX - lastRotX;
                var rdy = e.clientY - lastRotY;
                lastRotX = e.clientX;
                lastRotY = e.clientY;
                if (window.quartzTorus && window.quartzTorus.spinBy) {
                    window.quartzTorus.spinBy(rdx, rdy);
                }
                return;
            }
            if (!window.quartzTorusDragging) return;
            var clamped = clampTorusOnScreen(
                stage,
                e.clientX - offsetX,
                e.clientY - offsetY
            );
            stage.style.left = clamped.left + 'px';
            stage.style.top = clamped.top + 'px';
        });
        document.addEventListener('mouseup', function() {
            if (window.quartzTorusRotating) {
                window.quartzTorusRotating = false;
                stage.classList.remove('quartz-torus-rotating');
                saveTorusOrientation();
                syncTorusInteractionMode();
            }
            if (window.quartzTorusResizing) {
                window.quartzTorusResizing = false;
                stage.classList.remove('quartz-torus-resizing');
                window.quartzTorusPrefs.custom = true;
                window.quartzTorusPrefs.left = parseFloat(stage.style.left) || 0;
                window.quartzTorusPrefs.top = parseFloat(stage.style.top) || 0;
                window.quartzTorusPrefs.width = stage.offsetWidth;
                window.quartzTorusPrefs.height = stage.offsetHeight;
                saveTorusPrefs(window.quartzTorusPrefs);
                syncTorusInteractionMode();
                return;
            }
            if (!window.quartzTorusDragging) return;
            window.quartzTorusDragging = false;
            stage.classList.remove('quartz-torus-dragging');
            window.quartzTorusPrefs.custom = true;
            window.quartzTorusPrefs.left = parseFloat(stage.style.left) || 0;
            window.quartzTorusPrefs.top = parseFloat(stage.style.top) || 0;
            window.quartzTorusPrefs.width = stage.offsetWidth;
            window.quartzTorusPrefs.height = stage.offsetHeight;
            saveTorusPrefs(window.quartzTorusPrefs);
            syncTorusInteractionMode();
        });
        syncTorusInteractionMode();
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
            initTorusInteraction();
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
        if (window.quartzTypewriterActive) return;
        var ta = document.querySelector('.quartz-terminal textarea');
        if (!ta) return;
        var inputRow = document.querySelector('.quartz-input-row');
        var inputH = inputRow ? inputRow.offsetHeight : 52;
        var th = Math.max(120, viewportH - (window.quartzChromePx || 220) - inputH - 8);
        ta.style.height = th + 'px';
        ta.style.maxHeight = th + 'px';
        ta.style.minHeight = th + 'px';
        ta.style.overflowX = 'hidden';
        ta.style.overflowY = 'auto';
        lockTerminalOverflowX();
    }
    function torusMeshScale() {
        var stage = document.querySelector('.quartz-torus-stage');
        if (!stage) return 58;
        var side = Math.min(stage.offsetWidth, stage.offsetHeight);
        if (side < 1) return 58;
        return Math.max(38, Math.min(76, side * 0.28));
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
        if (typeof window.quartzFitGameStage === 'function') {
            window.quartzFitGameStage();
        }
    }
    function fitSkinMetrics() {
        var tabs = document.querySelector('.quartz-top-tabs');
        var prog = document.querySelector('.quartz-prog-bank');
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
        var prefs = window.quartzTorusPrefs || {};
        var state = {
            rotX: prefs.rotX != null ? prefs.rotX : 0.62,
            rotY: prefs.rotY != null ? prefs.rotY : 0.0,
            rotZ: prefs.rotZ != null ? prefs.rotZ : 0.18,
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
            if (!window.quartzTorusSpinPaused) {
                state.rotY += state.spinY;
                if (now < state.pulseUntil) {
                    state.rotZ += 0.0065;
                    state.rotX += (state.targetRotX - state.rotX) * 0.04;
                } else {
                    state.rotX += (0.62 + Math.sin(now * 0.00009) * 0.07 - state.rotX) * 0.02;
                    state.rotZ += (state.targetRotZ - state.rotZ) * 0.015;
                }
            }
            render();
            requestAnimationFrame(tick);
        }

        function absorbBridge(node) {
            if (!node) return;
            var echo = parseFloat(node.getAttribute('data-echo') || '0');
            var nudge = parseInt(node.getAttribute('data-nudge') || '0', 10);
            if (nudge !== state.nudge) {
                var visAttr = node.getAttribute('data-visible');
                if (visAttr !== null) {
                    var torusOn = visAttr !== '0';
                    applyTorusVisibility(torusOn, true);
                    window.quartzProgStates.prog1 = torusOn;
                    saveProgStates(window.quartzProgStates);
                    document.querySelectorAll('button.quartz-prog-id-1').forEach(function(btn) {
                        btn.classList.toggle('quartz-prog-active', torusOn);
                        btn.setAttribute('aria-pressed', torusOn ? 'true' : 'false');
                    });
                    if (!window.quartzTorusPrefs.custom) layoutTorusStage();
                }
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
        initTorusInteraction();
        window.quartzTorus = {
            ready: true,
            state: state,
            spinBy: function(dx, dy) {
                state.rotY += dx * 0.012;
                state.rotX += dy * 0.012;
                state.rotX = Math.max(-1.35, Math.min(1.35, state.rotX));
                render();
            },
            pulse: function() { state.pulseUntil = Date.now() + 1200; }
        };
    }
    var quartzLastStartupNudge = -1;
    function watchStartupReplay() {
        if (window.quartzReplayRunning) return;
        var node = document.querySelector('.quartz-torus-bridge');
        if (!node) return;
        var replay = parseInt(node.getAttribute('data-startup-replay') || '0', 10);
        var nudge = parseInt(node.getAttribute('data-nudge') || '0', 10);
        if (!replay || nudge === quartzLastStartupNudge) return;
        quartzLastStartupNudge = nudge;
        window.quartzReplayRunning = true;
        window.quartzRunStartupSequence(null, {
            postDelay: window.QUARTZ_MENU_POST_DELAY || 2000,
            persistBoot: false,
            runProgDiagnostic: true
        }).then(function() {
            window.quartzReplayRunning = false;
            if (typeof window.quartzClickStartupDone === 'function') {
                window.quartzClickStartupDone();
            }
        });
    }
    setInterval(watchStartupReplay, 200);
    var quartzLastGameNudge = -1;
    function watchGameStart() {
        var node = document.querySelector('#quartz-torus-bridge, .quartz-torus-bridge');
        if (!node) return;
        var start = parseInt(node.getAttribute('data-game-start') || '0', 10);
        if (!start) return;
        var nudge = parseInt(node.getAttribute('data-nudge') || '0', 10);
        if (nudge === quartzLastGameNudge) return;
        quartzLastGameNudge = nudge;
        setTimeout(function() {
            if (typeof window.quartzStartPlayerGame === 'function') {
                window.quartzStartPlayerGame();
            }
        }, 100);
    }
    setInterval(watchGameStart, 200);
    whenBodyReady(function() {
        initProgToggles();
        syncTorusInteractionMode();
        bootQuartzTorus();
        fitPanel();
        initTorusInteraction();
        watchStartupReplay();
        watchGameStart();
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
        layoutTorusStage();
        if (!window.quartzTorus || !window.quartzTorus.ready) bootQuartzTorus();
        initProgToggles();
    }, 1500);
    whenBodyReady(function() {
        var ta = document.querySelector('.quartz-terminal textarea');
        if (!ta || ta._quartzOverflowObs) return;
        ta._quartzOverflowObs = true;
        new MutationObserver(function() {
            if (window.quartzTypewriterActive) return;
            var vh = window.innerHeight || document.documentElement.clientHeight || 0;
            if (vh > 0) syncTerminalOverflow(vh);
        }).observe(ta, { childList: true, characterData: true, subtree: true });
    });
})();
</script>
"""

QUARTZ_HEAD = _quartz_startup_js_block() + _quartz_games_js_block() + _QUARTZ_HEAD_TEMPLATE

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
    --quartz-tab-display-gap: 0.12rem;
    --quartz-display-inset-top: clamp(0.1rem, 0.8vw, 0.12rem);
    --quartz-prog-btn-width: calc((100% - 1.12rem) / 8 * 0.95);
    --quartz-grid-line: rgba(90, 96, 104, 0.55);
    --quartz-skin-case-top: #2e333b;
    --quartz-skin-case-mid: #1e2228;
    --quartz-skin-case-bot: #0a0c0e;
    --quartz-skin-tab-h: 2.5rem;
    --quartz-skin-prog-h: 3rem;
    --quartz-skin-footer-h: 1.4rem;
    --quartz-skin-settings-h: 0px;
    --quartz-btn-label-size: clamp(calc(0.42rem + 2px), calc(0.9vh + 2px), calc(0.56rem + 2px));
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
    top: calc(0.45rem + var(--quartz-skin-tab-h) + var(--quartz-tab-display-gap)) !important;
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
    margin: 0 0 var(--quartz-tab-display-gap) 0 !important;
    flex-shrink: 0 !important;
    align-items: stretch !important;
}}
.gradio-container .quartz-chat-home-stack {{
    display: flex !important;
    flex-direction: column !important;
    gap: 0.1rem !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}}
.gradio-container .quartz-chat-home-stack > .block,
.gradio-container .quartz-chat-home-stack > .form {{
    width: 100% !important;
    min-width: 0 !important;
}}
.gradio-container .quartz-chat-home-stack button.quartz-tab,
.gradio-container .quartz-chat-home-stack button.quartz-home-btn {{
    flex: 0 0 auto !important;
    width: 100% !important;
    min-height: clamp(1.65rem, 3.5vh, 2.1rem) !important;
}}
.gradio-container button.quartz-home-btn {{
    border-radius: 0 0 4px 4px !important;
    position: relative !important;
}}
.gradio-container button.quartz-home-btn.quartz-home-active::after {{
    content: "" !important;
    position: absolute !important;
    top: 0.32rem !important;
    right: 0.14rem !important;
    width: 5px !important;
    height: 5px !important;
    border-radius: 50% !important;
    background: #ff2222 !important;
    box-shadow: 0 0 6px rgba(255, 40, 40, 0.85) !important;
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
    font-size: var(--quartz-btn-label-size) !important;
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
    font-size: inherit !important;
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
    margin: var(--quartz-display-inset-top) var(--quartz-inset) var(--quartz-inset) !important;
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
    pointer-events: auto !important;
    overflow: visible !important;
    background: transparent !important;
    box-sizing: border-box !important;
    cursor: grab !important;
    touch-action: none !important;
}}
.quartz-torus-stage.quartz-torus-dragging {{
    cursor: grabbing !important;
    opacity: 0.88 !important;
}}
.quartz-torus-stage.quartz-torus-rotating {{
    cursor: grabbing !important;
    opacity: 0.94 !important;
}}
.quartz-torus-stage.quartz-torus-move-mode {{
    cursor: move !important;
    outline: 1px solid rgba(0, 255, 65, 0.38) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 14px rgba(0, 255, 65, 0.16) !important;
}}
.quartz-torus-stage.quartz-torus-move-mode .quartz-torus-orbit {{
    stroke: rgba(0, 255, 65, 0.42) !important;
}}
.quartz-torus-stage.quartz-torus-resizing {{
    cursor: nwse-resize !important;
    opacity: 0.9 !important;
}}
.quartz-torus-stage.quartz-torus-hidden {{
    display: none !important;
    pointer-events: none !important;
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
    overflow-y: auto !important;
    box-sizing: border-box !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    max-width: 100% !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(0, 255, 65, 0.22) transparent !important;
    scrollbar-gutter: stable !important;
}}
.gradio-container .quartz-terminal textarea.quartz-typewriter-active {{
    overflow-y: hidden !important;
    scrollbar-gutter: auto !important;
}}
.gradio-container .quartz-terminal textarea.quartz-startup-centered {{
    text-align: center !important;
}}
.gradio-container .quartz-terminal textarea.quartz-game-active {{
    opacity: 0 !important;
    pointer-events: none !important;
    overflow: hidden !important;
}}
.gradio-container .quartz-invaders-stage {{
    position: fixed !important;
    z-index: 100001 !important;
    display: none !important;
    pointer-events: none !important;
    overflow: hidden !important;
    border-radius: 3px / 5px !important;
    background: #000000 !important;
    box-shadow: inset 0 0 28px rgba(0, 212, 255, 0.08) !important;
}}
body.quartz-game-on .quartz-invaders-stage {{
    display: block !important;
    pointer-events: auto !important;
}}
body.quartz-game-on .quartz-terminal,
body.quartz-game-on .quartz-terminal > .block,
body.quartz-game-on .quartz-terminal > div,
body.quartz-game-on .quartz-terminal .wrap {{
    visibility: hidden !important;
    pointer-events: none !important;
    z-index: 0 !important;
}}
.gradio-container .quartz-invaders-canvas {{
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    display: block !important;
    background: #000000 !important;
    image-rendering: pixelated !important;
    image-rendering: crisp-edges !important;
    filter: none !important;
}}
.gradio-container .quartz-invaders-crt {{
    position: absolute !important;
    inset: 0 !important;
    pointer-events: none !important;
    border-radius: inherit !important;
    background:
        repeating-linear-gradient(
            0deg,
            rgba(0, 0, 0, 0) 0px,
            rgba(0, 0, 0, 0) 2px,
            rgba(0, 0, 0, 0.18) 2px,
            rgba(0, 0, 0, 0.18) 3px
        ),
        radial-gradient(
            ellipse 92% 88% at 50% 48%,
            rgba(0, 212, 255, 0.05) 0%,
            rgba(0, 0, 0, 0) 52%,
            rgba(0, 0, 0, 0.42) 100%
        ) !important;
    box-shadow:
        inset 0 0 42px rgba(0, 212, 255, 0.09),
        inset 0 0 80px rgba(0, 0, 0, 0.55) !important;
    mix-blend-mode: screen !important;
    opacity: 0.92 !important;
}}
body.quartz-game-on .quartz-torus-stage {{
    display: none !important;
    pointer-events: none !important;
}}
body.quartz-game-on .quartz-display-backing {{
    z-index: 100000 !important;
}}
body.quartz-game-on .quartz-terminal-col {{
    flex: 1 1 auto !important;
    min-height: 0 !important;
    padding: 0.05rem 0.08rem 0.04rem !important;
    margin: 0.06rem !important;
}}
body.quartz-game-on .quartz-display-bay {{
    flex: 1 1 auto !important;
    min-height: 0 !important;
}}
body.quartz-game-on .quartz-terminal,
body.quartz-game-on .quartz-terminal .wrap {{
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
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
    font-size: var(--quartz-btn-label-size) !important;
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
    font-size: inherit !important;
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
    font-size: var(--quartz-btn-label-size) !important;
    letter-spacing: 0.1em !important;
}}
.gradio-container button.quartz-btn-pulse {{
    filter: brightness(0.92) !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.35) !important;
}}
.gradio-container .quartz-prog-bank {{
    display: flex !important;
    flex-direction: column !important;
    gap: 0.2rem !important;
    margin: 0.3rem 0 0.2rem 0 !important;
    flex-shrink: 0 !important;
    padding: 0.24rem 0.2rem !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    background: transparent !important;
    border: none !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    overflow-x: hidden !important;
}}
.gradio-container .quartz-prog-row {{
    gap: 0.16rem !important;
    margin: 0 !important;
    flex-shrink: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    justify-content: center !important;
}}
.gradio-container button.quartz-prog {{
    flex: 0 1 var(--quartz-prog-btn-width) !important;
    max-width: var(--quartz-prog-btn-width) !important;
    min-width: 0 !important;
    min-height: clamp(2.56rem, 5.2vh, 3.24rem) !important;
    font-size: var(--quartz-btn-label-size) !important;
    letter-spacing: 0.03em !important;
    padding: 0.2rem 0.28rem 0.2rem 0.06rem !important;
    position: relative !important;
    text-transform: none !important;
    text-align: center !important;
}}
.gradio-container button.quartz-prog-active::after {{
    content: "" !important;
    position: absolute !important;
    top: 0.32rem !important;
    right: 0.14rem !important;
    width: 5px !important;
    height: 5px !important;
    border-radius: 50% !important;
    background: #ff2222 !important;
    box-shadow: 0 0 6px rgba(255, 40, 40, 0.85) !important;
}}
.gradio-container button.quartz-prog-id-1,
.gradio-container button.quartz-prog-id-9 {{
    padding-left: 1.45em !important;
}}
.gradio-container button.quartz-prog-id-1::before,
.gradio-container button.quartz-prog-id-9::before {{
    content: "" !important;
    position: absolute !important;
    left: 0.32rem !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-size: var(--quartz-btn-label-size) !important;
    line-height: 1 !important;
    font-weight: 900 !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    z-index: 1 !important;
    pointer-events: none !important;
}}
.gradio-container button.quartz-prog-id-1::before {{
    content: "↵" !important;
}}
.gradio-container button.quartz-prog-id-9::before {{
    content: "▶" !important;
}}
.gradio-container button.quartz-prog-id-9.quartz-prog-active::before {{
    content: "||" !important;
    letter-spacing: -0.07em !important;
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
.gradio-container button.quartz-startup-done-btn,
.gradio-container button.quartz-game-quit-btn,
.gradio-container button.quartz-game-start-done-btn {{
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}}
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
        startup_done_btn = gr.Button(
            "StartupDone",
            elem_classes=["quartz-startup-done-btn"],
        )
        game_quit_btn = gr.Button(
            "GameQuit",
            elem_classes=["quartz-game-quit-btn"],
        )
        game_start_done_btn = gr.Button(
            "GameStartDone",
            elem_classes=["quartz-game-start-done-btn"],
        )

        with gr.Column(elem_classes=_root_classes(True)) as root_col:
            torus_bridge = gr.HTML(TORUS_STATE_BRIDGE_HTML, visible=False)
            gr.HTML(GRID_LAYER_HTML)
            with gr.Column(elem_classes=["quartz-panel"]):
                gr.HTML(_panel_skin_html(), elem_classes=["quartz-skin-mount"])
                with gr.Column(elem_classes=["quartz-content", "quartz-content-wrap"]):
                    with gr.Row(elem_classes=["quartz-top-tabs"]):
                        with gr.Column(elem_classes=["quartz-chat-home-stack"]):
                            tab_btns["chat"] = gr.Button(
                                TAB_LABELS["chat"],
                                elem_classes=_tab_btn_classes("home", "chat"),
                                variant="secondary",
                            )
                            home_btn = gr.Button(
                                "HOME",
                                elem_classes=_home_btn_classes(True),
                                variant="secondary",
                            )
                        for tab_id in TOP_TABS:
                            if tab_id == "chat":
                                continue
                            tab_btns[tab_id] = gr.Button(
                                TAB_LABELS[tab_id],
                                elem_classes=_tab_btn_classes("home", tab_id),
                                variant="secondary",
                            )

                    with gr.Column(elem_classes=["quartz-display-bay"]):
                        with gr.Column(elem_classes=["quartz-terminal-col"]):
                            gr.HTML(DISPLAY_BACKING_HTML)
                            terminal = gr.Textbox(
                                value="",
                                label="Terminal",
                                show_label=False,
                                interactive=False,
                                lines=12,
                                max_lines=80,
                                elem_classes=["quartz-terminal"],
                            )
                            with gr.Row(elem_classes=["quartz-input-row"]):
                                cmd_input = gr.Textbox(
                                    placeholder="Enter menu index or command...",
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

                    with gr.Column(elem_classes=["quartz-prog-bank"]):
                        with gr.Row(elem_classes=["quartz-prog-row", "quartz-prog-row-1"]):
                            for index in PROG_ROW_1:
                                gr.Button(
                                    f"Prog {index}",
                                    elem_classes=_prog_btn_classes(index),
                                )
                        with gr.Row(elem_classes=["quartz-prog-row", "quartz-prog-row-2"]):
                            for index in PROG_ROW_2:
                                gr.Button(
                                    f"Prog {index}",
                                    elem_classes=_prog_btn_classes(index),
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
            tab_btns["chat"],
            tab_btns["settings"],
            tab_btns["history"],
            tab_btns["memory"],
            tab_btns["tools"],
            home_btn,
            root_col,
            torus_bridge,
        ]
        send_outputs = [
            terminal,
            cmd_input,
            ui_state,
            torus_bridge,
            home_btn,
            settings_panel,
            tab_btns["chat"],
            tab_btns["settings"],
            tab_btns["history"],
            tab_btns["memory"],
            tab_btns["tools"],
            root_col,
        ]

        home_btn.click(
            _handle_home_toggle,
            inputs=[terminal, ui_state],
            outputs=tab_outputs,
        )

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

        send_evt = send_btn.click(
            _handle_send, inputs=[cmd_input, terminal, ui_state], outputs=send_outputs
        )
        send_evt.then(js=_build_game_menu_ack_js())
        cmd_submit_evt = cmd_input.submit(
            _handle_send, inputs=[cmd_input, terminal, ui_state], outputs=send_outputs
        )
        cmd_submit_evt.then(js=_build_game_menu_ack_js())

        startup_done_btn.click(
            _handle_startup_replay_done,
            inputs=[ui_state],
            outputs=[terminal, ui_state, home_btn, torus_bridge],
        )
        game_quit_btn.click(
            _handle_game_quit,
            inputs=[ui_state],
            outputs=[terminal, ui_state, home_btn, torus_bridge],
        )
        game_start_done_btn.click(
            _handle_game_start_done,
            inputs=[ui_state],
            outputs=[ui_state, torus_bridge],
        )

        boot_evt = demo.load(None, None, terminal, js=_build_startup_load_js())
        boot_evt.then(
            _sync_boot_state,
            inputs=[ui_state],
            outputs=[ui_state, home_btn],
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