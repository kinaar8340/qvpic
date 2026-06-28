#!/usr/bin/env python3
"""Gradio web demo for QVPIC — identity conduit control panel (VQC design)."""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
from collections.abc import Callable, Iterator

import gradio as gr

from demo_core import (
    BOOT_QUOTE_STRING,
    DEFAULT_QUERY_TEXT,
    GITHUB_URL,
    HFB_URL,
    HF_SPACE_URL,
    TERM_KEY_ACTIONS,
    VQC_URL,
    default_run_params,
    get_build_label,
    is_hf_space,
    run_benchmark_demo,
    run_query_recall,
    terminal_claims_snapshot,
    terminal_conduit_analogy,
    terminal_keypad_map,
    terminal_metrics_baseline,
    terminal_pipeline_scope,
    terminal_presets_catalog,
    terminal_recall_export,
    terminal_topology_shards,
)

logger = logging.getLogger(__name__)


def _patch_gradio_client_bool_schema() -> None:
    """Avoid gradio_client crash when JSON schema contains bare bool nodes."""
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
        logger.info("Patched gradio_client bool JSON-schema handling")
    except Exception:
        logger.warning("Could not patch gradio_client", exc_info=True)


_patch_gradio_client_bool_schema()
_DEFAULTS = default_run_params()

_VQC_ACCENT = "#ea580c"
_VQC_FIELD_FILL = "rgba(10, 8, 24, 0.50)"
_VQC_TAB_GREEN_BG = "#14532d"
_VQC_TAB_GREEN_BORDER = "#1ed760"
_VQC_TAB_GREEN_TEXT = "#86efac"
_VQC_TAB_ORANGE_BG = "#7c2d12"
_VQC_TAB_ORANGE_BORDER = "#ea580c"
_VQC_TAB_ORANGE_TEXT = "#fdba74"
_VQC_MATRIX_GREEN = "#33ff66"
_VQC_LOGO_GOLD = "#c9a227"
_VQC_HOME_KEY_BG = "#000000"

WEBGL_3D_NOTE = ""  # unused — kept for CSS compat

SLM_PACKAGE_IDLE = ""  # unused

OPTICS_LOGO_HTML = """
<div class="vqc-optics-logo" role="img" aria-label="QVPIC Identity Conduit Control Panel">
  <span class="vqc-optics-brand">QVPIC</span>
  <span class="vqc-optics-panel-title">Identity Conduit Control Panel</span>
  <span class="vqc-optics-subtitle">INTERACTIVE DEMO — KEYPAD TEACHES · REPO HAS DEPTH</span>
</div>
"""

DEMO_LINKS_HTML = f"""
<div class="vqc-demo-links" role="navigation" aria-label="Repository links">
  <span class="vqc-source-label">Fine details →</span>
  <a href="{GITHUB_URL}" class="vqc-source-tab" target="_blank" rel="noopener noreferrer">GitHub</a>
  <a href="{VQC_URL}" class="vqc-source-tab" target="_blank" rel="noopener noreferrer">vqc_proto</a>
  <a href="{HFB_URL}" class="vqc-source-tab" target="_blank" rel="noopener noreferrer">hfb</a>
  <a href="{GITHUB_URL}/blob/main/scripts/ui.py" class="vqc-source-tab" target="_blank" rel="noopener noreferrer">local UI</a>
</div>
"""

# Client-side CSS VQC conduit helix drift — distinct from Mystery phosphor scan (HF-safe).
CONDUIT_HELIX_SCANNER_HTML = f"""
<div class="vqc-oam-helix-scan" role="img" aria-label="VQC conduit helix drift">
  <div class="vqc-oam-rings" aria-hidden="true">
    <div class="vqc-oam-ring vqc-oam-ring-1"></div>
    <div class="vqc-oam-ring vqc-oam-ring-2"></div>
    <div class="vqc-oam-ring vqc-oam-ring-3"></div>
  </div>
  <div class="vqc-oam-helix-beam" aria-hidden="true"></div>
  <pre class="vqc-oam-helix-body">VQC CONDUIT HELIX — QVPIC
{'─' * 72}
quaternion vortex   RingConeChain persistent identity
helical depth s → primal recall → drift protection
topology shield   ShellCube braiding · winding invariants
query ref.        "{DEFAULT_QUERY_TEXT}"
RubikCone bake · VQCEnhanced optional · FastICA-free recall
{'─' * 72}
▸ facts baked as oriented cubes on braided lattice
▸ coordinate + vector drift recovery vs naive cosine
▸ shared VQC vision with orbital-braille optical carrier
▸ {GITHUB_URL}
{'─' * 72}
press any keypad to exit</pre>
</div>
"""

_OPTICS_TERM_BAR = "─" * 48
_OPTICS_TERM_CHAR_DELAY_S = 0.014
_OPTICS_TERM_NEWLINE_DELAY_S = 0.048
_OPTICS_TERM_UPLINK_DELAY_S = 0.22
_OPTICS_TERM_CURSOR = "▌"
_OPTICS_TERM_RELEASE_DELAY_S = 0.25
_BOOT_QUOTE_CHAR_DELAY_S = 0.1
_BOOT_POST_QUOTE_DELAY_S = 3.0
_BOOT_DOT_INTERVAL_S = 0.5
_BOOT_DOT_COUNT = 6
_BOOT_TERM_LINES = 14
_BOOT_TERM_COLS = 56


def _strip_md_plain(text: str) -> str:
    """Flatten markdown blockquotes and emphasis for terminal readout."""
    plain = re.sub(r"^>\s*", "", text.strip(), flags=re.MULTILINE)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"`([^`]+)`", r"\1", plain)
    return plain.strip()


def _optics_terminal_frame(title: str, body: str) -> str:
    return f"{title}\n{_OPTICS_TERM_BAR}\n{body}"


TERM_KEYPAD_PROG_COLS = 12
TERM_KEYPAD_PROG_ROWS = 2
TERM_KEYPAD_COUNT = TERM_KEYPAD_PROG_COLS * TERM_KEYPAD_PROG_ROWS
TERM_KEYPAD_DEFINED: dict[int, str] = {
    index: action for index, (action, _desc) in TERM_KEY_ACTIONS.items()
}
TERM_KEYPAD_HOME_KEY = "key01"
TERM_KEYPAD_DESCRIPTIONS: dict[int, str] = {
    index: desc for index, (_action, desc) in TERM_KEY_ACTIONS.items()
}
TERM_MENU_ACTIONS: tuple[str, ...] = (
    "home",
    "status",
    "conduit",
    "pipeline",
    "metrics",
    "build",
    "help",
    "helix",
)
TERM_UI_MENU = "menu"
TERM_UI_PAGE = "page"
TERM_NAV_KEYS: tuple[str, ...] = (
    "dpad_select",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "clear",
)
TERM_DPAD_HOLD_KEYS: tuple[str, ...] = (
    "dpad_select",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
)
TERM_NAV_DEFINED: dict[str, str] = {
    "dpad_select": "Enter — confirm menu item",
    "dpad_up": "Up — previous menu item",
    "dpad_down": "Down — next menu item",
    "dpad_left": "Left — previous menu item",
    "dpad_right": "Right — next menu item",
    "clear": "Clear — blank display",
}
TERM_KEYPAD_CONTROL_ORDER: tuple[str, ...] = (
    *TERM_NAV_KEYS,
    *(f"key{i:02d}" for i in range(1, TERM_KEYPAD_COUNT + 1)),
)


def _optics_terminal_home() -> str:
    return _optics_terminal_frame("PROGRAMMABLE KEYPAD", terminal_keypad_map())


def _default_term_ui_state() -> dict:
    return {"mode": TERM_UI_MENU, "index": 0, "scan": False}


def _optics_terminal_menu(menu_index: int) -> str:
    lines = [
        "Interactive demo — keypad pages teach bake → recall → topology",
        "▲▼ ◀▶ move · enter opens · 03–11 tour · Run benchmark below",
        "",
    ]
    for index, (_action, keypad_key, label, _stream) in enumerate(_term_menu_items()):
        mark = "▶" if index == menu_index else " "
        lines.append(f"{keypad_key:02d} --- [{mark}] {label}")
    return _optics_terminal_frame("SELECTION MENU", "\n".join(lines))


def _term_menu_label(action: str) -> str:
    labels = {
        "home": "Home — Keypad Map",
        "status": "Status — Live Pipeline",
        "conduit": "Conduit — RubikCone Analogy",
        "pipeline": "Pipeline — Bake → Drift",
        "metrics": "Metrics — Benchmark Baseline",
        "build": "Build — Deploy Stamp",
        "help": "Help — D-pad Navigation",
        "helix": "Helix — VQC Conduit Display",
    }
    return labels.get(action, action)


def _term_menu_keypad_index(action: str) -> int:
    for index, (key, _desc) in TERM_KEY_ACTIONS.items():
        if key == action:
            return index
    return 1


def _term_menu_items() -> tuple[tuple[str, int, str, Callable[[], Iterator[str]]], ...]:
    items = []
    for action in TERM_MENU_ACTIONS:
        stream_fn = TERM_KEYPAD_STREAMERS.get(action)
        if stream_fn is None:
            continue
        items.append(
            (
                action,
                _term_menu_keypad_index(action),
                _term_menu_label(action),
                stream_fn,
            )
        )
    return tuple(items)


def _term_menu_index_for_action(action: str) -> int:
    for index, (key, _keypad, _label, _stream) in enumerate(_term_menu_items()):
        if key == action:
            return index
    return 0


def _term_menu_step(menu_index: int, delta: int) -> int:
    count = len(_term_menu_items())
    return (menu_index + delta) % count


def _optics_terminal_status() -> str:
    on_hf = is_hf_space()
    env = "Hugging Face Space" if on_hf else "Local Gradio"
    llm_note = (
        "LLM disabled on HF (local agent only)"
        if on_hf
        else "LLM available via scripts/main.py"
    )
    return _optics_terminal_frame(
        "SYSTEM STATUS",
        "\n".join(
            [
                f"Environment : {env}",
                "Package     : QVPIC v10.2 (RubikCone default)",
                "Embedder    : all-MiniLM-L6-v2",
                f"LLM         : {llm_note}",
                "Pipeline    : embed → bake → recall → drift test",
                "Modules     : src/conduit.py · RingConeChain · ShellCube",
                "",
                get_build_label().replace("`", ""),
                "",
                "Keypad 03–11 teach the demo · Run benchmark when ready.",
            ]
        ),
    )


def _optics_terminal_conduit() -> str:
    return _optics_terminal_frame("CONDUIT ANALOGY", terminal_conduit_analogy())


def _optics_terminal_pipeline() -> str:
    return _optics_terminal_frame("PIPELINE SCOPE", terminal_pipeline_scope())


def _optics_terminal_metrics() -> str:
    return _optics_terminal_frame("METRICS BASELINE", terminal_metrics_baseline())


def _optics_terminal_claims() -> str:
    return _optics_terminal_frame("VQC CLAIMS MAP", terminal_claims_snapshot())


def _optics_terminal_topology() -> str:
    return _optics_terminal_frame("TOPOLOGY & BRAIDING", terminal_topology_shards())


def _optics_terminal_recall() -> str:
    return _optics_terminal_frame("QUERY RECALL", terminal_recall_export())


def _optics_terminal_presets() -> str:
    return _optics_terminal_frame("PRESET CATALOG", terminal_presets_catalog())


def _optics_terminal_build() -> str:
    build = get_build_label().replace("`", "")
    return _optics_terminal_frame(
        "BUILD / LAST UPDATED",
        "\n".join(
            [
                build,
                "",
                "Synced from vqc_proto via scripts/sync_hf_space.sh on deploy.",
                f"Repo: {GITHUB_URL}",
                f"Space: {HF_SPACE_URL}",
            ]
        ),
    )


def _optics_terminal_help() -> str:
    return _optics_terminal_frame(
        "KEYPAD REFERENCE",
        "\n".join(
            [
                "D-pad — ▲▼ ◀▶ move highlight · enter opens item",
                "01 Home → selection menu (momentary)",
                "02–08 mirror menu · 09–12 direct shortcuts",
                "08 / menu 08 → VQC conduit helix screensaver (any key stops)",
                "clear → blank display",
                "",
                "Press 01 Home for full keypad map.",
            ]
        ),
    )


def _stream_optics_terminal_text(full_text: str) -> Iterator[str]:
    """Reveal terminal text one character at a time — typewriter / uplink effect."""
    shown = ""
    for ch in full_text:
        shown += ch
        yield shown + _OPTICS_TERM_CURSOR
        time.sleep(_OPTICS_TERM_NEWLINE_DELAY_S if ch == "\n" else _OPTICS_TERM_CHAR_DELAY_S)
    yield shown


def _optics_terminal_uplink_banner(mode: str) -> str:
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    return f"> UPLINK {mode.upper()} @ {stamp} UTC…\n"


def _optics_terminal_stream(builder: Callable[[], str], *, mode: str) -> Iterator[str]:
    """Stream a keyed readout: uplink banner, then body character-by-character."""
    banner = _optics_terminal_uplink_banner(mode)
    yield banner + _OPTICS_TERM_CURSOR
    time.sleep(_OPTICS_TERM_UPLINK_DELAY_S)
    yield from _stream_optics_terminal_text(banner + builder())


def _stream_optics_terminal_home() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_home, mode="home")


def _stream_optics_terminal_status() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_status, mode="status")


def _stream_optics_terminal_conduit() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_conduit, mode="conduit")


def _stream_optics_terminal_pipeline() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_pipeline, mode="pipeline")


def _stream_optics_terminal_metrics() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_metrics, mode="metrics")


def _stream_optics_terminal_claims() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_claims, mode="claims")


def _stream_optics_terminal_topology() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_topology, mode="topology")


def _stream_optics_terminal_recall() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_recall, mode="recall")


def _stream_optics_terminal_presets() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_presets, mode="presets")


def _stream_helix_stub() -> Iterator[str]:
    """Menu placeholder — helix display uses CSS toggle via key 08 / d-pad."""
    yield ""


def _stream_optics_terminal_build() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_build, mode="build")


def _stream_optics_terminal_help() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_help, mode="help")


def _stream_optics_terminal_menu(menu_index: int = 0) -> Iterator[str]:
    yield from _optics_terminal_stream(
        lambda: _optics_terminal_menu(menu_index),
        mode="menu",
    )


def _stream_optics_terminal_clear(current: str) -> Iterator[str]:
    """Erase display in paced chunks — inverse of the typewriter feed."""
    text = current or ""
    if not text:
        yield ""
        return
    chunk = max(1, len(text) // 36)
    for end in range(len(text), -1, -chunk):
        yield text[:end] + (_OPTICS_TERM_CURSOR if end else "")
        time.sleep(0.01)
    yield ""


TERM_KEYPAD_STREAMERS: dict[str, Callable[[], Iterator[str]]] = {}


def _term_key_id(index: int) -> str:
    return f"key{index:02d}"


def _term_keypad_label(index: int) -> str:
    """Home key is '01 Home'; other prog keys are zero-padded."""
    if index == 1:
        return "01 Home"
    return f"{index:02d}"


def _term_key_is_defined_prog(key: str) -> bool:
    """True for assigned prog keys (02–24) that have a real function."""
    for index in TERM_KEYPAD_DEFINED:
        if index == 1:
            continue
        if _term_key_id(index) == key:
            return True
    return False


def _term_key_btn_classes(key: str, active: str) -> list[str]:
    """Black/white idle caps; matrix-green latch on active keys (never home)."""
    classes = ["vqc-optics-key"]
    if key in TERM_NAV_KEYS:
        classes.append("vqc-optics-dpad-key")
    if key == TERM_KEYPAD_HOME_KEY:
        classes.append("vqc-optics-key-home")
    elif key.startswith("dpad_"):
        classes.append("vqc-optics-key-dpad")
    if key == "clear":
        classes.append("vqc-optics-key-clear")
    if _term_key_is_defined_prog(key):
        classes.append("vqc-optics-key-defined")
    if key == active and key != TERM_KEYPAD_HOME_KEY:
        classes.append("active")
    return classes


def _term_keypad_btn_updates(active: str) -> tuple:
    return tuple(
        gr.update(elem_classes=_term_key_btn_classes(key_id, active))
        for key_id in TERM_KEYPAD_CONTROL_ORDER
    )


def _term_keypad_outputs(terminal_text: str, active: str, ui_state: dict | None = None) -> tuple:
    state = _default_term_ui_state() if ui_state is None else ui_state
    scanning = bool(state.get("scan"))
    return (
        gr.update(
            value=terminal_text,
            visible=not scanning,
            elem_classes=["vqc-optics-terminal-wrap", "vqc-optics-terminal"],
        ),
        gr.update(visible=scanning),
        *_term_keypad_btn_updates(active),
        active,
        state,
    )


def _term_yield_stream_then_release(
    stream: Iterator[str],
    *,
    active: str,
    ui_state: dict,
    release_delay: float | None = None,
) -> Iterator[tuple]:
    """Stream terminal text, latch while typing, release latch after a short pause."""
    delay = _OPTICS_TERM_RELEASE_DELAY_S if release_delay is None else release_delay
    last_partial = ""
    for partial in stream:
        last_partial = partial
        yield _term_keypad_outputs(partial, active, ui_state)
    time.sleep(delay)
    yield _term_keypad_outputs(last_partial, "", ui_state)


def _term_stream_with_latch(
    stream_fn: Callable[[], Iterator[str]],
    *,
    active: str,
    ui_state: dict,
) -> Iterator[tuple]:
    """Stream terminal text and latch matrix-green active state on the pressed key."""
    yield from _term_yield_stream_then_release(stream_fn(), active=active, ui_state=ui_state)


def _make_term_stream_click(
    active_key: str,
    stream_fn: Callable[[], Iterator[str]],
    *,
    menu_action: str | None = None,
):
    def handler(ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        if menu_action is not None:
            state.update(
                {
                    "mode": TERM_UI_PAGE,
                    "index": _term_menu_index_for_action(menu_action),
                    "scan": menu_action == "helix",
                }
            )
        else:
            state["scan"] = False
        yield from _term_stream_with_latch(stream_fn, active=active_key, ui_state=state)

    return handler


def _make_term_clear_click(active_key: str):
    def handler(current: str, ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        state["scan"] = False
        yield from _term_yield_stream_then_release(
            _stream_optics_terminal_clear(current),
            active=active_key,
            ui_state=state,
        )

    return handler


def _make_term_momentary_click(active_key: str, *, release_delay: float):
    """Brief latch flash — momentary, no maintained state."""

    def handler(current: str, ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        yield _term_keypad_outputs(current, active_key, state)
        time.sleep(release_delay)
        yield _term_keypad_outputs(current, "", state)

    return handler


def _make_term_dpad_click(active_key: str):
    """D-pad click — navigate menu, confirm with SEL, brief matrix-green latch."""

    def handler(_current: str, ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        mode = state.get("mode", TERM_UI_MENU)
        menu_index = int(state.get("index", 0))
        nav_delta = {
            "dpad_up": -1,
            "dpad_left": -1,
            "dpad_down": 1,
            "dpad_right": 1,
        }

        if active_key in nav_delta:
            if mode == TERM_UI_PAGE:
                menu_state = {"mode": TERM_UI_MENU, "index": menu_index, "scan": False}
                text = _optics_terminal_menu(menu_index)
            else:
                new_index = _term_menu_step(menu_index, nav_delta[active_key])
                menu_state = {"mode": TERM_UI_MENU, "index": new_index, "scan": False}
                text = _optics_terminal_menu(new_index)
            yield _term_keypad_outputs(text, active_key, menu_state)
            time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
            yield _term_keypad_outputs(text, "", menu_state)
            return

        if active_key == "dpad_select":
            if mode == TERM_UI_MENU:
                action, _keypad, _label, stream_fn = _term_menu_items()[menu_index]
                if action == "helix":
                    page_state = {
                        "mode": TERM_UI_PAGE,
                        "index": menu_index,
                        "scan": True,
                    }
                    yield _term_keypad_outputs("", "dpad_select", page_state)
                    time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
                    yield _term_keypad_outputs("", "", page_state)
                    return
                page_state = {
                    "mode": TERM_UI_PAGE,
                    "index": menu_index,
                    "scan": False,
                }
                yield from _term_yield_stream_then_release(
                    stream_fn(),
                    active="dpad_select",
                    ui_state=page_state,
                )
                return
            menu_state = {"mode": TERM_UI_MENU, "index": menu_index, "scan": False}
            text = _optics_terminal_menu(menu_index)
            yield _term_keypad_outputs(text, active_key, menu_state)
            time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
            yield _term_keypad_outputs(text, "", menu_state)

    return handler


def _make_term_latch_click(active_key: str):
    """Undefined keypad slots — latch matrix-green only, terminal unchanged."""

    def handler(current: str, ui_state: dict) -> tuple:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        state["scan"] = False
        return _term_keypad_outputs(current, active_key, state)

    return handler


def _make_activate_oam_helix_scan(active_key: str, *, menu_action: str = "helix"):
    """Toggle CSS VQC conduit helix drift — one yield, no server animation loop."""

    def handler(ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        state.update(
            {
                "mode": TERM_UI_PAGE,
                "index": _term_menu_index_for_action(menu_action),
                "scan": True,
            }
        )
        yield _term_keypad_outputs("", active_key, state)
        time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
        yield _term_keypad_outputs("", "", state)

    return handler


def _make_term_home_momentary():
    """Home — momentary return to the selection menu."""

    def handler(current_active: str, ui_state: dict) -> Iterator[tuple]:
        menu_state = {"mode": TERM_UI_MENU, "index": 0, "scan": False}
        menu_text = _optics_terminal_menu(0)
        yield _term_keypad_outputs(menu_text, current_active, menu_state)
        time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
        yield _term_keypad_outputs(menu_text, "", menu_state)

    return handler


def _boot_quote_prefix() -> str:
    """Pad so the quote types out near the middle of the terminal panel."""
    v_pad = max(0, (_BOOT_TERM_LINES - 1) // 2)
    h_pad = max(0, (_BOOT_TERM_COLS - len(BOOT_QUOTE_STRING)) // 2)
    return "\n" * v_pad + (" " * h_pad)


def _stream_term_boot() -> Iterator[tuple]:
    """One-shot startup: centered quote, dot countdown, then selection menu."""
    boot_state = _default_term_ui_state()
    shown = _boot_quote_prefix()
    yield _term_keypad_outputs(shown, "", boot_state)

    for ch in BOOT_QUOTE_STRING:
        shown += ch
        yield _term_keypad_outputs(shown + _OPTICS_TERM_CURSOR, "", boot_state)
        time.sleep(_BOOT_QUOTE_CHAR_DELAY_S)

    yield _term_keypad_outputs(shown, "", boot_state)
    time.sleep(_BOOT_POST_QUOTE_DELAY_S)

    for _ in range(_BOOT_DOT_COUNT):
        shown += "."
        yield _term_keypad_outputs(shown, "", boot_state)
        time.sleep(_BOOT_DOT_INTERVAL_S)

    menu_text = _optics_terminal_menu(0)
    yield _term_keypad_outputs(menu_text, "", boot_state)


def _register_term_keypad_streamers() -> None:
    TERM_KEYPAD_STREAMERS.update(
        {
            "home": _stream_optics_terminal_home,
            "status": _stream_optics_terminal_status,
            "conduit": _stream_optics_terminal_conduit,
            "pipeline": _stream_optics_terminal_pipeline,
            "metrics": _stream_optics_terminal_metrics,
            "build": _stream_optics_terminal_build,
            "help": _stream_optics_terminal_help,
            "helix": _stream_helix_stub,
            "claims": _stream_optics_terminal_claims,
            "topology": _stream_optics_terminal_topology,
            "recall": _stream_optics_terminal_recall,
            "presets": _stream_optics_terminal_presets,
        }
    )


_register_term_keypad_streamers()


def _build_vqc_theme() -> gr.themes.Base:
    """Dark transparent theme — works in light and dark OS modes (HF-safe)."""
    return (
        gr.themes.Base(
            primary_hue=gr.themes.colors.orange,
            secondary_hue=gr.themes.colors.zinc,
            neutral_hue=gr.themes.colors.zinc,
        )
        .set(
            body_background_fill="transparent",
            body_background_fill_dark="transparent",
            background_fill_primary="transparent",
            background_fill_primary_dark="transparent",
            background_fill_secondary="transparent",
            background_fill_secondary_dark="transparent",
            block_background_fill=_VQC_FIELD_FILL,
            block_background_fill_dark=_VQC_FIELD_FILL,
            panel_background_fill=_VQC_FIELD_FILL,
            panel_background_fill_dark=_VQC_FIELD_FILL,
            input_background_fill=_VQC_FIELD_FILL,
            input_background_fill_dark=_VQC_FIELD_FILL,
            body_text_color="#e8e0f8",
            body_text_color_dark="#e8e0f8",
            block_label_text_color="#c9b8ff",
            block_label_text_color_dark="#c9b8ff",
            block_title_text_color="#f0e6ff",
            block_title_text_color_dark="#f0e6ff",
            border_color_primary="rgba(255, 255, 255, 0.12)",
            border_color_primary_dark="rgba(255, 255, 255, 0.12)",
            button_primary_background_fill="#ea580c",
            button_primary_background_fill_dark="#ea580c",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="rgba(28, 22, 48, 0.92)",
            button_secondary_background_fill_dark="rgba(28, 22, 48, 0.92)",
            button_secondary_text_color="#e8e0f8",
            button_secondary_text_color_dark="#e8e0f8",
            checkbox_label_background_fill="transparent",
            checkbox_label_background_fill_dark="transparent",
            checkbox_label_background_fill_hover="transparent",
            checkbox_label_background_fill_hover_dark="transparent",
            slider_color=_VQC_ACCENT,
            slider_color_dark=_VQC_ACCENT,
            link_text_color=_VQC_ACCENT,
            link_text_color_dark=_VQC_ACCENT,
            link_text_color_hover="#f97316",
            link_text_color_hover_dark="#f97316",
            link_text_color_active=_VQC_ACCENT,
            link_text_color_active_dark=_VQC_ACCENT,
            link_text_color_visited=_VQC_ACCENT,
            link_text_color_visited_dark=_VQC_ACCENT,
        )
    )


# Full-viewport shell — solid backdrop so the control panel owns the page.
WALLPAPER_HEAD = """
<style id="vqc-fullscreen-style">
html, body { background-color: #0a0818 !important; }
</style>
"""

HFB_CSS = f"""
:root, :root .dark {{
    --body-background-fill: transparent !important;
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --block-background-fill: {_VQC_FIELD_FILL} !important;
    --panel-background-fill: {_VQC_FIELD_FILL} !important;
    --input-background-fill: {_VQC_FIELD_FILL} !important;
    --body-text-color: #e8e0f8 !important;
    --block-label-text-color: #c9b8ff !important;
    --block-title-text-color: #f0e6ff !important;
    --border-color-primary: rgba(255, 255, 255, 0.12) !important;
    --link-text-color: {_VQC_ACCENT} !important;
    --link-text-color-hover: #f97316 !important;
    --link-text-color-active: {_VQC_ACCENT} !important;
    --link-text-color-visited: {_VQC_ACCENT} !important;
    color-scheme: dark;
}}
html {{
    background-color: #0a0818 !important;
    min-height: 100% !important;
}}
body {{
    background: transparent !important;
    background-color: transparent !important;
    color: #e8e0f8 !important;
    min-height: 100vh !important;
    width: 100% !important;
    overflow-x: hidden !important;
    position: relative !important;
}}
#root, .app {{
    background: #0a0818 !important;
    background-color: #0a0818 !important;
    min-height: 100vh !important;
    height: 100% !important;
    width: 100% !important;
}}
.gradio-container {{
    position: relative !important;
    width: 100% !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #0a0818 !important;
    background-color: #0a0818 !important;
}}
.gradio-container .vqc-fullscreen-shell {{
    width: 100% !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
}}
.gradio-container .vqc-demo-links {{
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.35rem 0.75rem !important;
    margin: 0 !important;
    padding: 0 !important;
}}
.gradio-container .vqc-terminal-stage {{
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    width: 100% !important;
}}
.gradio-container .vqc-terminal-stage > .block,
.gradio-container .vqc-terminal-stage > .form {{
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
}}
.gradio-container .vqc-lattice-compact {{
    flex: 0 0 auto !important;
    max-height: min(22vh, 220px) !important;
    margin: 0.35rem 0 0.45rem 0 !important;
}}
.gradio-container .vqc-lattice-compact .image-container,
.gradio-container .vqc-lattice-compact img {{
    max-height: min(22vh, 220px) !important;
    object-fit: contain !important;
}}
.gradio-container .vqc-action-row {{
    gap: 0.55rem !important;
    margin: 0.35rem 0 0.45rem 0 !important;
}}
footer {{
    background: transparent !important;
}}
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .form,
.gradio-container .column,
.gradio-container .row,
.gradio-container .gr-group,
.gradio-container label.wrap,
.gradio-container .label-wrap {{
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}
.gradio-container .block {{
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    background-color: {_VQC_FIELD_FILL} !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
}}
.gradio-container .markdown,
.gradio-container .prose,
.gradio-container .markdown p,
.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown li {{
    color: #e8e0f8 !important;
}}
.gradio-container .vqc-source-tabs-row {{
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.45rem 0.65rem !important;
    margin: 0.35rem 0 0.1rem 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
.gradio-container .vqc-source-nav-row {{
    margin: 0 0 0.1rem 0 !important;
}}
.gradio-container .vqc-links-panel {{
    margin: 0 0 0.35rem 0 !important;
    padding: 0.65rem 0.85rem !important;
}}
.gradio-container .vqc-links-panel .markdown h3 {{
    margin: 0 !important;
    font-size: 1rem !important;
    color: #f0e6ff !important;
}}
.gradio-container .vqc-panel-header-row {{
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 0.5rem !important;
    width: 100% !important;
    margin: 0 0 0.35rem 0 !important;
}}
.gradio-container .vqc-panel-header-row > .block,
.gradio-container .vqc-panel-header-row > .form {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    flex: 1 1 auto !important;
    width: auto !important;
}}
.gradio-container button.vqc-panel-minimize {{
    flex: 0 0 auto !important;
    min-width: 2.1rem !important;
    padding: 0.2rem 0.6rem !important;
    border-radius: 999px !important;
    border: 1px solid {_VQC_TAB_GREEN_BORDER} !important;
    background: {_VQC_TAB_GREEN_BG} !important;
    color: {_VQC_TAB_GREEN_TEXT} !important;
    -webkit-text-fill-color: {_VQC_TAB_GREEN_TEXT} !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    line-height: 1 !important;
    box-shadow: none !important;
    opacity: 0.8 !important;
    cursor: pointer !important;
}}
.gradio-container button.vqc-panel-minimize:hover {{
    border-color: {_VQC_TAB_ORANGE_BORDER} !important;
    background: {_VQC_TAB_ORANGE_BG} !important;
    color: {_VQC_TAB_ORANGE_TEXT} !important;
    -webkit-text-fill-color: {_VQC_TAB_ORANGE_TEXT} !important;
}}
.gradio-container .vqc-source-tabs-row > .block,
.gradio-container .vqc-source-tabs-row > .form,
.gradio-container .vqc-source-tabs-row .block,
.gradio-container .vqc-source-tabs-row .form {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    width: auto !important;
    flex: 0 0 auto !important;
}}
.gradio-container .vqc-source-tabs-row .html-container {{
    padding: 0 !important;
    margin: 0 !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.secondary,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.secondary:hover,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.secondary:focus {{
    border: none !important;
    outline: none !important;
    background: transparent !important;
    box-shadow: none !important;
}}
.gradio-container .vqc-source-label {{
    color: #e8e0f8 !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    margin-right: 0.15rem !important;
    line-height: 1.2 !important;
}}
.gradio-container .vqc-source-tab,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab span,
.gradio-container .vqc-nav-cell a.vqc-source-tab {{
    display: inline !important;
    padding: 0 !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    color: {_VQC_MATRIX_GREEN} !important;
    -webkit-text-fill-color: {_VQC_MATRIX_GREEN} !important;
    text-decoration: underline !important;
    text-decoration-color: {_VQC_MATRIX_GREEN} !important;
    text-underline-offset: 0.18em !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    line-height: 1.35 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    min-height: unset !important;
    height: auto !important;
    width: auto !important;
    margin: 0 !important;
    opacity: 1 !important;
    text-shadow: 0 0 6px rgba(51, 255, 102, 0.25) !important;
    transition: color 0.15s ease, text-decoration-color 0.15s ease, opacity 0.15s ease;
}}
.gradio-container a.vqc-source-tab:hover,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab:not(.active):hover,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab:not(.active):hover span {{
    color: #7dff9a !important;
    -webkit-text-fill-color: #7dff9a !important;
    text-decoration-color: #7dff9a !important;
    background: transparent !important;
    text-decoration: underline !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab {{
    cursor: pointer !important;
    font-family: inherit !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab:disabled:not(.active),
.gradio-container .vqc-source-tabs-row button.vqc-source-tab[disabled]:not(.active),
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.secondary:disabled:not(.active),
.gradio-container .vqc-source-tabs-row button.vqc-source-tab:disabled:not(.active) span,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab[disabled]:not(.active) span {{
    cursor: pointer !important;
    color: {_VQC_MATRIX_GREEN} !important;
    -webkit-text-fill-color: {_VQC_MATRIX_GREEN} !important;
    text-decoration-color: {_VQC_MATRIX_GREEN} !important;
    background: transparent !important;
    text-decoration: underline !important;
}}
.gradio-container .vqc-source-tab.active,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active span,
.gradio-container .vqc-source-tab.active:hover,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active:hover,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active:hover span,
.gradio-container a.vqc-source-tab.active {{
    color: {_VQC_LOGO_GOLD} !important;
    -webkit-text-fill-color: {_VQC_LOGO_GOLD} !important;
    text-decoration-color: {_VQC_LOGO_GOLD} !important;
    background: transparent !important;
    text-decoration: underline !important;
    text-decoration-thickness: 2px !important;
    cursor: default !important;
    opacity: 1 !important;
    text-shadow: 0 0 8px rgba(201, 162, 39, 0.45) !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active:disabled,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active[disabled],
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active:disabled span,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active[disabled] span {{
    color: {_VQC_LOGO_GOLD} !important;
    -webkit-text-fill-color: {_VQC_LOGO_GOLD} !important;
    text-decoration-color: {_VQC_LOGO_GOLD} !important;
    background: transparent !important;
    text-decoration: underline !important;
    text-decoration-thickness: 2px !important;
    cursor: default !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active::before {{
    content: none !important;
    display: none !important;
}}
.gradio-container a:hover:not(.vqc-source-tab),
.gradio-container .markdown a:hover:not(.vqc-source-tab),
.gradio-container .prose a:hover:not(.vqc-source-tab) {{
    color: #f97316 !important;
    -webkit-text-fill-color: #f97316 !important;
}}
.gradio-container .vqc-build-label {{
    color: #a89ec8 !important;
    font-size: 0.9rem;
    margin: 0 0 0.5rem 0;
}}
.gradio-container .vqc-gallery-page .markdown h2 {{
    font-size: 1.35rem !important;
    margin: 0.15rem 0 0.35rem 0 !important;
}}
.gradio-container .vqc-gallery-page .markdown p {{
    font-size: 0.92rem !important;
    margin: 0.15rem 0 0.35rem 0 !important;
    line-height: 1.45 !important;
}}
.gradio-container .vqc-gallery-page,
.gradio-container .vqc-gallery-page > .block,
.gradio-container .vqc-gallery-page .html-container {{
    width: 100% !important;
    max-width: 100% !important;
}}
.gradio-container .vqc-screencast-wrap {{
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 0.75rem !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0.25rem 0 0.5rem 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}}
.gradio-container .vqc-screencast-video {{
    width: 100% !important;
    min-width: 0 !important;
    height: auto !important;
    aspect-ratio: 16 / 9 !important;
    max-height: min(36vh, 360px) !important;
    object-fit: contain !important;
    border-radius: 8px !important;
    display: block !important;
    background: rgba(10, 8, 24, 0.35) !important;
}}
.gradio-container .vqc-optics-panel {{
    background: linear-gradient(165deg, #2a1810 0%, #1a1008 38%, #120c06 100%) !important;
    border: none !important;
    border-top: 3px solid #6b4f1d !important;
    border-radius: 0 !important;
    box-shadow:
        inset 0 2px 8px rgba(255, 220, 150, 0.08),
        inset 0 -4px 14px rgba(0, 0, 0, 0.55) !important;
    padding: 0 0.85rem 0.75rem !important;
    margin: 0 !important;
    gap: 0 !important;
    flex: 1 1 auto !important;
    min-height: 100vh !important;
    height: 100% !important;
    width: 100% !important;
    max-width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
}}
.gradio-container .vqc-optics-panel > .gap {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}
.gradio-container .vqc-optics-panel > .block,
.gradio-container .vqc-optics-panel .block {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
.gradio-container .vqc-optics-panel-header {{
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.75rem 1.1rem !important;
    margin: 0 0 0 0 !important;
    padding: 0.7rem 0.85rem 1.35rem !important;
    border: none !important;
    border-bottom: 1px solid rgba(74, 56, 24, 0.65) !important;
    border-radius: 10px 10px 0 0 !important;
    background: linear-gradient(180deg, #1f140a 0%, #0f0a06 100%) !important;
    box-shadow: inset 0 0 18px rgba(0, 0, 0, 0.65) !important;
    width: 100% !important;
    min-height: 5.25rem !important;
}}
.gradio-container .vqc-optics-panel-header > .block,
.gradio-container .vqc-optics-panel-header > .form,
.gradio-container .vqc-optics-panel-header .block,
.gradio-container .vqc-optics-panel-header .form {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
}}
.gradio-container .vqc-optics-panel-header > .block:first-child,
.gradio-container .vqc-optics-panel-header > .form:first-child {{
    flex: 0 0 auto !important;
    width: auto !important;
}}
.gradio-container .vqc-optics-panel-nav {{
    flex: 1 1 18rem !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0.28rem !important;
    justify-content: center !important;
    min-width: 0 !important;
    width: auto !important;
}}
.gradio-container .vqc-optics-panel-nav > .block,
.gradio-container .vqc-optics-panel-nav > .form {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
}}
.gradio-container .vqc-optics-panel-nav .vqc-source-tabs-row {{
    margin: 0 !important;
}}
.gradio-container .vqc-nav-spreadsheet-row {{
    display: grid !important;
    grid-template-columns: 4.75rem repeat(5, minmax(4.5rem, 1fr)) !important;
    gap: 0.2rem 0.45rem !important;
    align-items: center !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}}
.gradio-container .vqc-nav-spreadsheet-row > .block,
.gradio-container .vqc-nav-spreadsheet-row > .form,
.gradio-container .vqc-nav-spreadsheet-row > .column {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    width: 100% !important;
}}
.gradio-container .vqc-nav-row-label {{
    justify-self: end !important;
    align-self: center !important;
    text-align: right !important;
    padding-right: 0.15rem !important;
}}
.gradio-container .vqc-nav-cell,
.gradio-container .vqc-nav-cell > .block,
.gradio-container .vqc-nav-cell > .form {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    min-height: 1.55rem !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0.1rem 0.2rem !important;
}}
.gradio-container .vqc-nav-cell .html-container,
.gradio-container .vqc-nav-cell .html-container p {{
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    width: 100% !important;
}}
.gradio-container .vqc-nav-cell-empty {{
    visibility: hidden !important;
}}
.gradio-container .vqc-optics-logo {{
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 0.1rem !important;
    min-width: 10.5rem !important;
    padding-right: 0.65rem !important;
    border-right: 1px solid rgba(107, 79, 29, 0.45) !important;
}}
.gradio-container .vqc-optics-brand {{
    font-size: 0.62rem !important;
    letter-spacing: 0.28em !important;
    color: {_VQC_LOGO_GOLD} !important;
    font-weight: 700 !important;
}}
.gradio-container .vqc-optics-panel-title {{
    font-size: 1.15rem !important;
    letter-spacing: 0.12em !important;
    color: #f5e6c8 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    text-shadow: 0 0 10px rgba(255, 180, 80, 0.35) !important;
}}
.gradio-container .vqc-optics-subtitle {{
    font-size: 0.68rem !important;
    letter-spacing: 0.22em !important;
    color: #9a8458 !important;
}}
.gradio-container .vqc-optics-terminal-caption {{
    font-size: 0.58rem !important;
    letter-spacing: 0.18em !important;
    color: #3dff7a !important;
    text-shadow: 0 0 8px rgba(61, 255, 122, 0.45) !important;
    margin-top: 0.1rem !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-terminal textarea,
.gradio-container .vqc-optics-panel .vqc-optics-terminal input {{
    background: rgba(2, 10, 4, 0.1) !important;
    border: 2px inset #1a4d2a !important;
    color: #33ff66 !important;
    -webkit-text-fill-color: #33ff66 !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.45 !important;
    text-shadow: 0 0 6px rgba(51, 255, 102, 0.35) !important;
    box-shadow:
        inset 0 0 18px rgba(0, 40, 12, 0.65),
        0 0 12px rgba(51, 255, 102, 0.08) !important;
    border-radius: 6px !important;
    caret-color: #33ff66 !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-terminal .label-wrap span {{
    color: #3dff7a !important;
    letter-spacing: 0.14em !important;
    text-shadow: 0 0 6px rgba(61, 255, 122, 0.35) !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-terminal-wrap {{
    background: rgba(2, 10, 4, 0.1) !important;
    border: 1px solid #1a4d2a !important;
    border-radius: 10px !important;
    padding: 0.5rem 0.6rem 0.45rem !important;
    margin: 0.55rem 0 0.55rem 0 !important;
}}
.gradio-container .vqc-animations-nav-row {{
    margin: 0.35rem 0 0.65rem 0 !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-terminal textarea {{
    min-height: calc(100vh - 24rem) !important;
    height: 100% !important;
    flex: 1 1 auto !important;
    white-space: pre !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-terminal-wrap {{
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: calc(100vh - 24rem) !important;
}}
.gradio-container .vqc-oam-helix-scan {{
    position: relative !important;
    width: 100% !important;
    min-height: 14rem !important;
    margin: 0.55rem 0 !important;
    padding: 0.65rem 0.75rem !important;
    background: rgba(10, 8, 24, 0.55) !important;
    border: 2px inset #5c4212 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
}}
.gradio-container .vqc-oam-rings {{
    position: absolute !important;
    inset: 0 !important;
    pointer-events: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
.gradio-container .vqc-oam-ring {{
    position: absolute !important;
    border-radius: 50% !important;
    border: 1px solid rgba(234, 88, 12, 0.22) !important;
    box-shadow: 0 0 12px rgba(234, 88, 12, 0.08) !important;
}}
.gradio-container .vqc-oam-ring-1 {{
    width: 88% !important;
    height: 42% !important;
    animation: vqc-oam-spin 18s linear infinite !important;
}}
.gradio-container .vqc-oam-ring-2 {{
    width: 62% !important;
    height: 30% !important;
    animation: vqc-oam-spin 12s linear infinite reverse !important;
}}
.gradio-container .vqc-oam-ring-3 {{
    width: 38% !important;
    height: 18% !important;
    animation: vqc-oam-spin 8s linear infinite !important;
}}
.gradio-container .vqc-oam-helix-beam {{
    position: absolute !important;
    left: 0 !important;
    right: 0 !important;
    height: 22% !important;
    pointer-events: none !important;
    background: linear-gradient(
        180deg,
        transparent 0%,
        rgba(234, 88, 12, 0.10) 42%,
        rgba(255, 180, 80, 0.28) 50%,
        rgba(234, 88, 12, 0.10) 58%,
        transparent 100%
    ) !important;
    animation: vqc-oam-beam 5s ease-in-out infinite !important;
}}
.gradio-container .vqc-oam-helix-body {{
    position: relative !important;
    z-index: 1 !important;
    margin: 0 !important;
    color: #ffb347 !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.45 !important;
    text-shadow: 0 0 8px rgba(234, 88, 12, 0.35) !important;
    white-space: pre-wrap !important;
    animation: vqc-oam-flicker 4s ease-in-out infinite !important;
}}
@keyframes vqc-oam-spin {{
    0% {{ transform: rotate(0deg) scaleX(1.15); }}
    100% {{ transform: rotate(360deg) scaleX(1.15); }}
}}
@keyframes vqc-oam-beam {{
    0%, 100% {{ top: -25%; }}
    50% {{ top: 80%; }}
}}
@keyframes vqc-oam-flicker {{
    0%, 100% {{ opacity: 1; }}
    47% {{ opacity: 0.9; }}
    50% {{ opacity: 0.75; }}
    53% {{ opacity: 0.92; }}
}}
.gradio-container .vqc-optics-keypad {{
    background: linear-gradient(180deg, #16120c 0%, #0a0806 100%) !important;
    border: 2px inset #3d3020 !important;
    border-radius: 10px !important;
    padding: 0.42rem 0.38rem 0.48rem !important;
    margin: 0 0 0.65rem 0 !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.55) !important;
}}
.gradio-container .vqc-optics-keypad > .block,
.gradio-container .vqc-optics-keypad .block {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}}
.gradio-container .vqc-optics-dpad-group {{
    margin: 0 0 0.28rem 0 !important;
    padding: 0 0 0.12rem 0 !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-dpad-row,
.gradio-container .vqc-optics-panel .vqc-optics-prog-row {{
    gap: 0.2rem !important;
    margin: 0 0 0.2rem 0 !important;
    justify-content: stretch !important;
    width: 100% !important;
}}
.gradio-container .vqc-optics-keypad button.vqc-optics-key,
.gradio-container .vqc-optics-keypad button.vqc-optics-key span {{
    font-family: "Courier New", Courier, monospace !important;
    font-size: 1.44rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
}}
.gradio-container .vqc-optics-keypad button.vqc-optics-key {{
    flex: 1 1 0 !important;
    min-width: 0 !important;
    max-width: none !important;
    min-height: 3rem !important;
    height: 3rem !important;
    max-height: 3rem !important;
    aspect-ratio: auto !important;
    background: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    letter-spacing: 0.03em !important;
    padding: 0.28rem 0.1rem !important;
    box-shadow: none !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-dpad-row button.vqc-optics-key-dpad,
.gradio-container .vqc-optics-panel .vqc-optics-dpad-row button.vqc-optics-key-dpad span {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    font-size: 1.44rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-dpad-row button.vqc-optics-key-dpad:active,
.gradio-container .vqc-optics-panel .vqc-optics-dpad-row button.vqc-optics-key-dpad:active span {{
    background: {_VQC_MATRIX_GREEN} !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    box-shadow: 0 0 12px rgba(51, 255, 102, 0.45) !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key-clear {{
    text-transform: lowercase !important;
    letter-spacing: 0.06em !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key-home,
.gradio-container .vqc-optics-panel button.vqc-optics-key-home:hover {{
    background: {_VQC_HOME_KEY_BG} !important;
    box-shadow: none !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key-home,
.gradio-container .vqc-optics-panel button.vqc-optics-key-home:hover,
.gradio-container .vqc-optics-panel button.vqc-optics-key-home span {{
    color: {_VQC_MATRIX_GREEN} !important;
    -webkit-text-fill-color: {_VQC_MATRIX_GREEN} !important;
    font-size: 1.44rem !important;
    font-weight: 700 !important;
    text-shadow: 0 0 6px rgba(51, 255, 102, 0.35) !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key-home:hover {{
    background: #141414 !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key-defined:not(.active),
.gradio-container .vqc-optics-panel button.vqc-optics-key-defined:not(.active) span {{
    color: {_VQC_MATRIX_GREEN} !important;
    -webkit-text-fill-color: {_VQC_MATRIX_GREEN} !important;
    text-shadow: 0 0 6px rgba(51, 255, 102, 0.35) !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key-defined:not(.active):hover,
.gradio-container .vqc-optics-panel button.vqc-optics-key-defined:not(.active):hover span {{
    color: #7dff9a !important;
    -webkit-text-fill-color: #7dff9a !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key:not(.active):not(.vqc-optics-key-home):not(.vqc-optics-key-defined):hover {{
    background: #141414 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key-defined:not(.active):hover {{
    background: #141414 !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key.active,
.gradio-container .vqc-optics-panel button.vqc-optics-key.active:hover {{
    background: {_VQC_MATRIX_GREEN} !important;
    box-shadow: 0 0 12px rgba(51, 255, 102, 0.45) !important;
}}
.gradio-container .vqc-optics-panel button.vqc-optics-key.active,
.gradio-container .vqc-optics-panel button.vqc-optics-key.active:hover,
.gradio-container .vqc-optics-panel button.vqc-optics-key.active span {{
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    text-shadow: none !important;
    -webkit-text-stroke: none !important;
}}
.gradio-container .vqc-optics-panel .label-wrap span,
.gradio-container .vqc-optics-panel label span {{
    color: #e8d4a8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}}
.gradio-container .vqc-optics-panel .info {{
    color: #9a8458 !important;
    font-size: 0.72rem !important;
    font-style: italic !important;
}}
.gradio-container .vqc-optics-panel input[type="text"],
.gradio-container .vqc-optics-panel textarea {{
    background: #120c06 !important;
    border: 2px inset #5c4a1f !important;
    color: #ffb347 !important;
    font-family: "Courier New", Courier, monospace !important;
    border-radius: 6px !important;
    box-shadow: inset 0 0 10px rgba(255, 140, 40, 0.12) !important;
}}
.gradio-container .vqc-optics-panel input[type="number"] {{
    background: #120c06 !important;
    border: 2px inset #5c4a1f !important;
    color: #ffb347 !important;
    font-family: "Courier New", Courier, monospace !important;
    font-weight: 700 !important;
    text-align: center !important;
    border-radius: 4px !important;
    box-shadow: inset 0 0 12px rgba(255, 140, 40, 0.18) !important;
    min-width: 4.2rem !important;
}}
.gradio-container .vqc-optics-panel input[type="range"] {{
    height: 6px !important;
    background: linear-gradient(90deg, #1a1208, #3d2e14, #1a1208) !important;
    border: 1px solid #5c4a1f !important;
    border-radius: 999px !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.6) !important;
}}
.gradio-container .vqc-optics-panel input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none !important;
    width: 24px !important;
    height: 24px !important;
    border-radius: 50% !important;
    background: radial-gradient(circle at 32% 28%, #fff2cc 0%, #c9a227 38%, #5c4212 72%, #2a1f08 100%) !important;
    border: 2px solid #1a1208 !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.55), inset 0 -2px 4px rgba(0, 0, 0, 0.35) !important;
    cursor: pointer !important;
}}
.gradio-container .vqc-optics-panel input[type="range"]::-moz-range-thumb {{
    width: 24px !important;
    height: 24px !important;
    border-radius: 50% !important;
    background: radial-gradient(circle at 32% 28%, #fff2cc 0%, #c9a227 38%, #5c4212 72%, #2a1f08 100%) !important;
    border: 2px solid #1a1208 !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.55) !important;
    cursor: pointer !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-dial-wrap {{
    background: rgba(0, 0, 0, 0.22) !important;
    border: 1px solid #4a3818 !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.65rem 0.45rem !important;
    margin: 0 !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-tune-row {{
    gap: 0.65rem !important;
    margin-bottom: 0.55rem !important;
}}
.gradio-container .vqc-optics-panel .vqc-optics-dial-row {{
    gap: 0.65rem !important;
    align-items: stretch !important;
}}
.gradio-container .vqc-optics-panel fieldset {{
    background: rgba(0, 0, 0, 0.18) !important;
    border: 1px solid #4a3818 !important;
    border-radius: 10px !important;
    padding: 0.45rem 0.55rem !important;
}}
.gradio-container .vqc-optics-panel .vqc-band-switch button {{
    border: 1px solid #6b4f1d !important;
    background: #1a1208 !important;
    color: #c9a227 !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.45) !important;
}}
.gradio-container .vqc-optics-panel .vqc-band-switch button.selected,
.gradio-container .vqc-optics-panel .vqc-band-switch button[aria-checked="true"] {{
    background: linear-gradient(180deg, #8b6914 0%, #4a3818 100%) !important;
    color: #fff2cc !important;
    box-shadow: 0 0 10px rgba(255, 160, 60, 0.35) !important;
}}
.gradio-container .vqc-optics-presets-label {{
    color: #c9a227 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 0.45rem 0 0.35rem 0 !important;
    text-align: center !important;
}}
.gradio-container .vqc-optics-panel button.vqc-receiver-preset {{
    background: linear-gradient(180deg, #3d2e14 0%, #1f1608 100%) !important;
    border: 2px solid #6b4f1d !important;
    color: #f5e6c8 !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: inset 0 1px 0 rgba(255, 220, 150, 0.15), 0 2px 4px rgba(0, 0, 0, 0.4) !important;
}}
.gradio-container .vqc-optics-panel button.vqc-receiver-preset:hover {{
    background: linear-gradient(180deg, #6b4f1d 0%, #3d2e14 100%) !important;
    color: #fff8e8 !important;
}}
.gradio-container .vqc-optics-panel .vqc-slm-toggle label {{
    color: #c9a227 !important;
    font-size: 0.76rem !important;
}}
.gradio-container .markdown blockquote {{
    border-left-color: rgba(255, 180, 80, 0.5) !important;
    background: transparent !important;
}}
.gradio-container .accordion,
.gradio-container details,
.gradio-container summary {{
    background-color: {_VQC_FIELD_FILL} !important;
    color: #e8e0f8 !important;
    border-radius: 10px;
}}
.gradio-container .image-container img,
.gradio-container .gr-image img,
.gradio-container video,
.gradio-container .plot-container {{
    background-color: transparent !important;
    opacity: 1 !important;
}}
.gradio-container button,
.gradio-container .gr-button {{
    opacity: 1 !important;
}}
.gradio-container input[type="range"] {{
    opacity: 1 !important;
}}
.gradio-container .vqc-full-width {{
    width: 100% !important;
}}
.gradio-container .main,
.gradio-container .wrap {{
    width: 100% !important;
    max-width: 100% !important;
    min-height: 0 !important;
    height: auto !important;
}}
.gradio-container .contain {{
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto !important;
    min-height: 0 !important;
    height: auto !important;
}}
.gradio-container .vqc-animation-panel,
.gradio-container .vqc-figure-panel,
.gradio-container .vqc-plot3d-panel {{
    width: 100% !important;
    max-width: 100% !important;
    min-height: 0 !important;
}}
.gradio-container .vqc-animation-panel video,
.gradio-container .vqc-animation-panel .image-container,
.gradio-container .vqc-animation-panel img,
.gradio-container .vqc-figure-panel .image-container,
.gradio-container .vqc-figure-panel img {{
    width: 100% !important;
    max-width: 100% !important;
    object-fit: contain;
}}
.gradio-container .vqc-plot3d-panel .plot-container {{
    width: 100% !important;
    min-height: 360px;
}}
.gradio-container .gr-video .empty,
.gradio-container .gr-image .empty,
.gradio-container .gr-image .icon-wrap {{
    min-height: 80px !important;
    background: transparent !important;
}}
.gradio-container .vqc-gallery-wrap {{
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)) !important;
    gap: 0.75rem !important;
    width: 100% !important;
}}
.gradio-container .vqc-gallery-figure img {{
    width: 100% !important;
    border-radius: 8px !important;
    background: rgba(10, 8, 24, 0.35) !important;
}}
.gradio-container .vqc-gallery-figure figcaption {{
    color: #c9b8ff !important;
    font-size: 0.82rem !important;
    text-align: center !important;
    margin-top: 0.35rem !important;
}}
footer {{ visibility: hidden; }}
"""


def run_benchmark(
    bake_steps: float,
    bandwidth: float,
    use_vqc: bool,
    drift_samples: float,
    max_facts: float,
    include_lattice: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str | None]:
    try:
        result = run_benchmark_demo(
            bake_steps=int(bake_steps),
            bandwidth=float(bandwidth),
            use_vqc=bool(use_vqc),
            drift_samples=int(drift_samples),
            max_facts=int(max_facts),
            include_lattice=bool(include_lattice),
            progress_cb=progress,
        )
        return result.metrics_text, result.lattice_path
    except Exception as exc:
        logger.exception("run_benchmark failed")
        return f"Error: {exc}\n\n{traceback.format_exc()}", None


def run_query(
    query_text: str,
    bake_steps: float,
    bandwidth: float,
    use_vqc: bool,
    max_facts: float,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> str:
    try:
        return run_query_recall(
            query_text,
            bake_steps=int(bake_steps),
            bandwidth=float(bandwidth),
            use_vqc=bool(use_vqc),
            max_facts=int(max_facts),
            progress_cb=progress,
        )
    except Exception as exc:
        logger.exception("run_query failed")
        return f"Error: {exc}\n\n{traceback.format_exc()}"


def _frame_demo_output(title: str, body: str) -> str:
    """Route benchmark/recall results into the matrix terminal."""
    return _optics_terminal_frame(title, body)


def run_benchmark_terminal(
    bake_steps: float,
    bandwidth: float,
    use_vqc: bool,
    drift_samples: float,
    max_facts: float,
    include_lattice: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, gr.Update]:
    metrics, lattice = run_benchmark(
        bake_steps,
        bandwidth,
        use_vqc,
        drift_samples,
        max_facts,
        include_lattice,
        progress,
    )
    lattice_update = (
        gr.update(value=lattice, visible=True)
        if lattice
        else gr.update(value=None, visible=False)
    )
    return _frame_demo_output("BENCHMARK RESULTS", metrics), lattice_update


def run_query_terminal(
    query_text: str,
    bake_steps: float,
    bandwidth: float,
    use_vqc: bool,
    max_facts: float,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, gr.Update]:
    result = run_query(
        query_text,
        bake_steps,
        bandwidth,
        use_vqc,
        max_facts,
        progress,
    )
    return _frame_demo_output("QUERY RECALL", result), gr.update(visible=False)


def build_app() -> gr.Blocks:
    on_hf = is_hf_space()
    lattice_info = (
        "Lattice render on HF uses reduced samples — first run downloads embedder weights"
        if on_hf
        else "Renders braided lattice PNG after benchmark completes"
    )

    with gr.Blocks(
        title="QVPIC — Identity Conduit Demo",
        analytics_enabled=False,
        theme=_build_vqc_theme(),
        head=WALLPAPER_HEAD,
        css=HFB_CSS,
        fill_width=True,
    ) as demo:
        with gr.Column(elem_classes=["vqc-fullscreen-shell"]):
            with gr.Group(elem_classes=["vqc-optics-panel"]):
                with gr.Row(elem_classes=["vqc-optics-panel-header"]):
                    gr.HTML(OPTICS_LOGO_HTML)
                    with gr.Column(elem_classes=["vqc-optics-panel-nav"], scale=1):
                        gr.HTML(DEMO_LINKS_HTML)
                with gr.Column(elem_classes=["vqc-terminal-stage"]):
                    optics_terminal = gr.Textbox(
                        label="Demo terminal — keypad teaches · results appear here",
                        value="",
                        lines=1,
                        max_lines=80,
                        interactive=False,
                        elem_classes=["vqc-optics-terminal-wrap", "vqc-optics-terminal"],
                    )
                    term_conduit_helix_scan = gr.HTML(
                        CONDUIT_HELIX_SCANNER_HTML,
                        visible=False,
                        elem_classes=["vqc-oam-helix-host"],
                    )
                    lattice_figure = gr.Image(
                        label="Braided lattice (after benchmark)",
                        type="filepath",
                        visible=False,
                        show_label=False,
                        elem_classes=["vqc-lattice-compact"],
                    )
                term_active_key = gr.State("")
                term_ui_state = gr.State(_default_term_ui_state())
                term_all_btns: dict[str, gr.Button] = {}
                _dpad_row_labels = {
                    "dpad_select": "enter",
                    "dpad_up": "▲",
                    "dpad_down": "▼",
                    "dpad_left": "◀",
                    "dpad_right": "▶",
                    "clear": "clear",
                }

                with gr.Row(elem_classes=["vqc-action-row"]):
                    benchmark_btn = gr.Button("Run benchmark", variant="primary", scale=1)
                    query_btn = gr.Button("Run query recall", variant="secondary", scale=1)

                with gr.Row(elem_classes=["vqc-optics-tune-row"]):
                    bake_steps = gr.Slider(
                        10,
                        150,
                        value=_DEFAULTS["bake_steps"],
                        step=5,
                        label="Bake steps / fact",
                        elem_classes=["vqc-optics-dial-wrap"],
                    )
                    bandwidth = gr.Slider(
                        0.1,
                        1.0,
                        value=_DEFAULTS["bandwidth"],
                        step=0.05,
                        label="Read bandwidth",
                        elem_classes=["vqc-optics-dial-wrap"],
                    )
                    drift_samples = gr.Slider(
                        10,
                        80,
                        value=_DEFAULTS["drift_samples"],
                        step=5,
                        label="Drift samples",
                        elem_classes=["vqc-optics-dial-wrap"],
                    )
                    max_facts = gr.Slider(
                        3,
                        12,
                        value=_DEFAULTS["max_facts"],
                        step=1,
                        label="Max demo facts",
                        elem_classes=["vqc-optics-dial-wrap"],
                    )
                with gr.Row(elem_classes=["vqc-optics-tune-row"]):
                    query_text = gr.Textbox(
                        label="Query recall text",
                        value=_DEFAULTS["query_text"],
                        elem_classes=["vqc-optics-dial-wrap"],
                    )
                    use_vqc = gr.Checkbox(
                        label="VQCEnhanced conduit (experimental)",
                        value=_DEFAULTS["use_vqc"],
                        elem_classes=["vqc-optics-dial-wrap"],
                    )
                    include_lattice = gr.Checkbox(
                        label="Include braided lattice PNG",
                        value=True,
                        info=lattice_info,
                        elem_classes=["vqc-optics-dial-wrap"],
                    )

                with gr.Column(elem_classes=["vqc-optics-keypad"]):
                    with gr.Row(elem_classes=["vqc-optics-dpad-row"], equal_height=True):
                        for nav_key in TERM_NAV_KEYS:
                            term_all_btns[nav_key] = gr.Button(
                                _dpad_row_labels[nav_key],
                                elem_classes=_term_key_btn_classes(nav_key, ""),
                                scale=1,
                                variant="secondary",
                            )
                    with gr.Row(elem_classes=["vqc-optics-prog-row"], equal_height=True):
                        for index in range(1, 13):
                            key_id = _term_key_id(index)
                            term_all_btns[key_id] = gr.Button(
                                _term_keypad_label(index),
                                elem_classes=_term_key_btn_classes(key_id, ""),
                                scale=1,
                                variant="secondary",
                            )
                    with gr.Row(elem_classes=["vqc-optics-prog-row"], equal_height=True):
                        for index in range(13, 25):
                            key_id = _term_key_id(index)
                            term_all_btns[key_id] = gr.Button(
                                _term_keypad_label(index),
                                elem_classes=_term_key_btn_classes(key_id, ""),
                                scale=1,
                                variant="secondary",
                            )
                term_keypad_outputs = [
                    optics_terminal,
                    term_conduit_helix_scan,
                    *[term_all_btns[key_id] for key_id in TERM_KEYPAD_CONTROL_ORDER],
                    term_active_key,
                    term_ui_state,
                ]
                term_cancels: list = []

                def _bind_term_event(btn: gr.Button, fn, *, inputs: list) -> None:
                    term_cancels.append(
                        btn.click(
                            fn,
                            inputs=inputs,
                            outputs=term_keypad_outputs,
                            cancels=term_cancels,
                        )
                    )

            helix_key = _term_key_id(8)
            _bind_term_event(
                term_all_btns[helix_key],
                _make_activate_oam_helix_scan(helix_key),
                inputs=[term_ui_state],
            )
            _bind_term_event(
                term_all_btns["clear"],
                _make_term_clear_click("clear"),
                inputs=[optics_terminal, term_ui_state],
            )
            for hold_key in TERM_DPAD_HOLD_KEYS:
                _bind_term_event(
                    term_all_btns[hold_key],
                    _make_term_dpad_click(hold_key),
                    inputs=[optics_terminal, term_ui_state],
                )
            _bind_term_event(
                term_all_btns[TERM_KEYPAD_HOME_KEY],
                _make_term_home_momentary(),
                inputs=[term_active_key, term_ui_state],
            )
            for index in range(1, TERM_KEYPAD_COUNT + 1):
                key_id = _term_key_id(index)
                if index == 1 or index == 8:
                    continue
                if index in TERM_KEYPAD_DEFINED:
                    action = TERM_KEYPAD_DEFINED[index]
                    _bind_term_event(
                        term_all_btns[key_id],
                        _make_term_stream_click(
                            key_id,
                            TERM_KEYPAD_STREAMERS[action],
                            menu_action=action,
                        ),
                        inputs=[term_ui_state],
                    )
                else:
                    _bind_term_event(
                        term_all_btns[key_id],
                        _make_term_latch_click(key_id),
                        inputs=[optics_terminal, term_ui_state],
                    )

            tune_inputs = [bake_steps, bandwidth, use_vqc, drift_samples, max_facts]
            benchmark_btn.click(
                run_benchmark_terminal,
                inputs=[*tune_inputs, include_lattice],
                outputs=[optics_terminal, lattice_figure],
            )
            query_btn.click(
                run_query_terminal,
                inputs=[query_text, bake_steps, bandwidth, use_vqc, max_facts],
                outputs=[optics_terminal, lattice_figure],
            )
            demo.load(_stream_term_boot, outputs=term_keypad_outputs)
    return demo


demo = build_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    try:
        demo.get_api_info()
        logger.info("Gradio API info check passed")
    except Exception:
        logger.exception("Gradio API info check failed")

    on_hf = bool(os.environ.get("SPACE_ID"))
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    launch_kwargs: dict = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "show_error": True,
        "show_api": False,
        "inbrowser": False,
        "share": False if on_hf else True,
    }

    demo.queue(default_concurrency_limit=2).launch(**launch_kwargs)


if __name__ == "__main__":
    main()