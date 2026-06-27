#!/usr/bin/env python3
"""Gradio web demo for QVPIC — identity conduit control panel."""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
from collections.abc import Callable, Iterator

import gradio as gr

from demo_core import (
    GITHUB_URL,
    HF_SPACE_URL,
    HFB_URL,
    QVPIC_WALLPAPER_URL,
    VQC_URL,
    default_run_params,
    get_build_label,
    is_hf_space,
    run_benchmark_demo,
    run_query_recall,
)

logger = logging.getLogger(__name__)


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
        logger.info("Patched gradio_client bool JSON-schema handling")
    except Exception:
        logger.warning("Could not patch gradio_client", exc_info=True)


_patch_gradio_client_bool_schema()

_DEFAULTS = default_run_params()
_GALLERY_BANNER_URL = "https://raw.githubusercontent.com/kinaar8340/qvpic/main/images/qvpic.png"
_GALLERY_BANNER2_URL = "https://raw.githubusercontent.com/kinaar8340/qvpic/main/images/qvpic_banner.png"

_QVP_ACCENT = "#a855f7"
_QVP_FIELD_FILL = "rgba(12, 8, 24, 0.50)"
_QVP_TAB_GREEN_BG = "#14532d"
_QVP_TAB_GREEN_BORDER = "#1ed760"
_QVP_TAB_GREEN_TEXT = "#86efac"
_QVP_TAB_ORANGE_BG = "#581c87"
_QVP_TAB_ORANGE_BORDER = "#a855f7"
_QVP_TAB_ORANGE_TEXT = "#e9d5ff"
_QVP_MATRIX_GREEN = "#33ff66"
_QVP_LOGO_GOLD = "#c9a227"
_QVP_HOME_KEY_BG = "#000000"

SCOPE_MD = """
> **Simulation demo** — browser-based bake → recall → drift benchmark on demo facts.
> Uses RubikConeConduit / RingConeChain (no local LLM on HF). Full agent chat runs locally
> via `python scripts/main.py`.
"""

BAKE_BANNER_MD = """
> **Bake & recall** — embed demo facts with SentenceTransformer, bake into RingConeChain,
> measure primal cosine recall and ShellCube / braiding topology invariants.
"""

TOPOLOGY_BANNER_MD = """
> **Drift & topology** — noisy vector + coordinate drift recovery benchmark reports
> protection factor vs naive cosine. Winding and braiding_phase come from
> `monitor_topological_winding()`.
"""

OPTICS_LOGO_HTML = """
<div class="qvp-optics-logo" role="img" aria-label="QVPIC Identity Conduit Control Panel">
  <span class="qvp-optics-brand">QVPIC</span>
  <span class="qvp-optics-panel-title">Identity Conduit Control Panel</span>
  <span class="qvp-optics-subtitle">QUATERNION · VORTEX · TOPOLOGY · RECALL</span>
</div>
"""

_OPTICS_TERM_BAR = "─" * 48
_OPTICS_TERM_CHAR_DELAY_S = 0.014
_OPTICS_TERM_NEWLINE_DELAY_S = 0.048
_OPTICS_TERM_UPLINK_DELAY_S = 0.22
_OPTICS_TERM_CURSOR = "▌"
_OPTICS_TERM_RELEASE_DELAY_S = 0.25

TERM_KEYPAD_PROG_COLS = 12
TERM_KEYPAD_PROG_ROWS = 2
TERM_KEYPAD_COUNT = TERM_KEYPAD_PROG_COLS * TERM_KEYPAD_PROG_ROWS
TERM_KEYPAD_DEFINED: dict[int, str] = {
    1: "home",
    2: "status",
    3: "bake",
    4: "topology",
    5: "help",
}
TERM_KEYPAD_HOME_KEY = "key01"
TERM_KEYPAD_DESCRIPTIONS: dict[int, str] = {
    1: "Return to selection menu — momentary",
    2: "Status — pipeline & environment",
    3: "Bake & recall — RingConeChain demo",
    4: "Drift & topology — protection factor",
    5: "Help — keypad & scope notes",
}
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


def _strip_md_plain(text: str) -> str:
    plain = re.sub(r"^>\s*", "", text.strip(), flags=re.MULTILINE)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"`([^`]+)`", r"\1", plain)
    return plain.strip()


def _optics_terminal_frame(title: str, body: str) -> str:
    return f"{title}\n{_OPTICS_TERM_BAR}\n{body}"


def _optics_assigned_keypad_lines() -> str:
    lines = []
    for index in sorted(TERM_KEYPAD_DEFINED):
        tag = "01 Home" if index == 1 else f"{index:02d}"
        lines.append(f"  [{tag}]  {TERM_KEYPAD_DESCRIPTIONS[index]}")
    for nav_key in TERM_NAV_KEYS:
        if nav_key in TERM_NAV_DEFINED:
            tag = "CLR" if nav_key == "clear" else nav_key.removeprefix("dpad_").upper()
            lines.append(f"  [{tag}]  {TERM_NAV_DEFINED[nav_key]}")
    return "\n".join(lines)


def _optics_terminal_home() -> str:
    return _optics_terminal_frame("PROGRAMMABLE KEYPAD", _optics_assigned_keypad_lines())


def _default_term_ui_state() -> dict:
    return {"mode": TERM_UI_MENU, "index": 0}


def _optics_terminal_menu(menu_index: int) -> str:
    lines = [
        "▲▼ ◀▶ move highlight · enter confirm · 01 Home",
        "",
    ]
    for index, (_action, keypad_key, label, _stream) in enumerate(_term_menu_items()):
        mark = "▶" if index == menu_index else " "
        lines.append(f"{keypad_key:02d} --- [{mark}] {label}")
    return _optics_terminal_frame("SELECTION MENU", "\n".join(lines))


def _term_menu_items() -> tuple[tuple[str, int, str, Callable[[], Iterator[str]]], ...]:
    return (
        ("home", 1, "Home Keypad Legend", _stream_optics_terminal_home),
        ("status", 2, "Status Pipeline & Environment", _stream_optics_terminal_status),
        ("bake", 3, "Bake & Recall Benchmark", _stream_optics_terminal_bake),
        ("topology", 4, "Drift & Topological Invariants", _stream_optics_terminal_topology),
        ("help", 5, "Help Keypad & Scope Notes", _stream_optics_terminal_help),
    )


def _term_menu_index_for_action(action: str) -> int:
    for index, (key, _keypad, _label, _stream) in enumerate(_term_menu_items()):
        if key == action:
            return index
    return 0


def _term_menu_step(menu_index: int, delta: int) -> int:
    return (menu_index + delta) % len(_term_menu_items())


def _optics_terminal_status() -> str:
    on_hf = is_hf_space()
    env = "Hugging Face Space" if on_hf else "Local Gradio"
    llm_note = "LLM disabled on HF (local agent only)" if on_hf else "LLM available via scripts/main.py"
    return _optics_terminal_frame(
        "SYSTEM STATUS",
        "\n".join(
            [
                f"Environment : {env}",
                f"Package     : QVPIC v10.2 (RubikCone default)",
                f"Embedder    : all-MiniLM-L6-v2",
                f"LLM         : {llm_note}",
                "Pipeline    : embed → bake → recall → drift test",
                "Modules     : src/conduit.py · RingConeChain · ShellCube",
                "",
                get_build_label().replace("`", ""),
                "",
                "Tune dials below, then RUN BENCHMARK or QUERY RECALL.",
            ]
        ),
    )


def _optics_terminal_bake() -> str:
    return _optics_terminal_frame("BAKE & RECALL", _strip_md_plain(BAKE_BANNER_MD))


def _optics_terminal_topology() -> str:
    return _optics_terminal_frame("DRIFT & TOPOLOGY", _strip_md_plain(TOPOLOGY_BANNER_MD))


def _optics_terminal_help() -> str:
    return _optics_terminal_frame(
        "KEYPAD REFERENCE",
        "\n".join(
            [
                "D-pad TUI — ▲▼ ◀▶ move · enter opens highlighted item",
                "Prog keys 02–05 mirror menu items · 01 Home → menu",
                "",
                _optics_assigned_keypad_lines(),
                "",
                _strip_md_plain(SCOPE_MD),
            ]
        ),
    )


def _stream_optics_terminal_text(full_text: str) -> Iterator[str]:
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
    banner = _optics_terminal_uplink_banner(mode)
    yield banner + _OPTICS_TERM_CURSOR
    time.sleep(_OPTICS_TERM_UPLINK_DELAY_S)
    yield from _stream_optics_terminal_text(banner + builder())


def _stream_optics_terminal_home() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_home, mode="home")


def _stream_optics_terminal_status() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_status, mode="status")


def _stream_optics_terminal_bake() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_bake, mode="bake")


def _stream_optics_terminal_topology() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_topology, mode="topology")


def _stream_optics_terminal_help() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_help, mode="help")


def _stream_optics_terminal_clear(current: str) -> Iterator[str]:
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
    if index == 1:
        return "01 Home"
    return f"{index:02d}"


def _term_key_is_defined_prog(key: str) -> bool:
    for index in TERM_KEYPAD_DEFINED:
        if index == 1:
            continue
        if _term_key_id(index) == key:
            return True
    return False


def _term_key_btn_classes(key: str, active: str) -> list[str]:
    classes = ["qvp-optics-key"]
    if key in TERM_NAV_KEYS:
        classes.append("qvp-optics-dpad-key")
    if key == TERM_KEYPAD_HOME_KEY:
        classes.append("qvp-optics-key-home")
    elif key.startswith("dpad_"):
        classes.append("qvp-optics-key-dpad")
    if key == "clear":
        classes.append("qvp-optics-key-clear")
    if _term_key_is_defined_prog(key):
        classes.append("qvp-optics-key-defined")
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
    return (terminal_text, *_term_keypad_btn_updates(active), active, state)


def _term_yield_stream_then_release(
    stream: Iterator[str],
    *,
    active: str,
    ui_state: dict,
    release_delay: float | None = None,
) -> Iterator[tuple]:
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
            state = {
                "mode": TERM_UI_PAGE,
                "index": _term_menu_index_for_action(menu_action),
            }
        yield from _term_stream_with_latch(stream_fn, active=active_key, ui_state=state)

    return handler


def _make_term_clear_click(active_key: str):
    def handler(current: str, ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        yield from _term_yield_stream_then_release(
            _stream_optics_terminal_clear(current),
            active=active_key,
            ui_state=state,
        )

    return handler


def _make_term_dpad_click(active_key: str):
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
                menu_state = {"mode": TERM_UI_MENU, "index": menu_index}
                text = _optics_terminal_menu(menu_index)
            else:
                new_index = _term_menu_step(menu_index, nav_delta[active_key])
                menu_state = {"mode": TERM_UI_MENU, "index": new_index}
                text = _optics_terminal_menu(new_index)
            yield _term_keypad_outputs(text, active_key, menu_state)
            time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
            yield _term_keypad_outputs(text, "", menu_state)
            return

        if active_key == "dpad_select":
            if mode == TERM_UI_MENU:
                _action, _keypad, _label, stream_fn = _term_menu_items()[menu_index]
                page_state = {"mode": TERM_UI_PAGE, "index": menu_index}
                yield from _term_yield_stream_then_release(
                    stream_fn(),
                    active="dpad_select",
                    ui_state=page_state,
                )
                return
            menu_state = {"mode": TERM_UI_MENU, "index": menu_index}
            text = _optics_terminal_menu(menu_index)
            yield _term_keypad_outputs(text, active_key, menu_state)
            time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
            yield _term_keypad_outputs(text, "", menu_state)

    return handler


def _make_term_latch_click(active_key: str):
    def handler(current: str, ui_state: dict) -> tuple:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        return _term_keypad_outputs(current, active_key, state)

    return handler


def _make_term_home_momentary():
    def handler(current_active: str, ui_state: dict) -> Iterator[tuple]:
        menu_state = {"mode": TERM_UI_MENU, "index": 0}
        menu_text = _optics_terminal_menu(0)
        yield _term_keypad_outputs(menu_text, current_active, menu_state)
        time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
        yield _term_keypad_outputs(menu_text, "", menu_state)

    return handler


def _term_boot_home() -> tuple:
    boot_state = _default_term_ui_state()
    return _term_keypad_outputs(_optics_terminal_menu(0), "", boot_state)


def _register_term_keypad_streamers() -> None:
    TERM_KEYPAD_STREAMERS.update(
        {
            "home": _stream_optics_terminal_home,
            "status": _stream_optics_terminal_status,
            "bake": _stream_optics_terminal_bake,
            "topology": _stream_optics_terminal_topology,
            "help": _stream_optics_terminal_help,
        }
    )


_register_term_keypad_streamers()


def _external_tab_html(label: str, url: str, tab_id: str) -> str:
    return (
        f'<a href="{url}" class="qvp-source-tab" data-tab="{tab_id}" '
        f'target="_blank" rel="noopener noreferrer">{label}</a>'
    )


def _source_tab_btn_update(*, active: bool) -> gr.Update:
    if active:
        return gr.update(interactive=False, elem_classes=["qvp-source-tab", "active"])
    return gr.update(interactive=True, elem_classes=["qvp-source-tab"], variant="secondary")


def _home_tab_update(*, on_demo_page: bool) -> gr.Update:
    if on_demo_page:
        return gr.update(interactive=False, elem_classes=["qvp-source-tab", "active"], variant="secondary")
    return gr.update(interactive=True, elem_classes=["qvp-source-tab"], variant="secondary")


def _close_links_panels() -> tuple:
    return (
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
    )


def _nav_to_page(page: str) -> tuple:
    on_demo = page == "demo"
    closed = _close_links_panels()
    return (
        gr.update(visible=on_demo),
        gr.update(visible=not on_demo),
        _home_tab_update(on_demo_page=on_demo),
        _source_tab_btn_update(active=not on_demo),
        *closed,
        _home_tab_update(on_demo_page=on_demo),
        _source_tab_btn_update(active=not on_demo),
        page,
    )


def _toggle_scope(is_open: bool) -> tuple:
    show = not is_open
    return (
        gr.update(visible=show),
        _source_tab_btn_update(active=show),
        show,
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
    )


def _minimize_scope() -> tuple:
    return (
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
    )


def _gallery_grid_html() -> str:
    panels = (
        ("QVPIC banner", _GALLERY_BANNER_URL),
        ("Repository banner", _GALLERY_BANNER2_URL),
    )
    imgs = "".join(
        f'<figure class="qvp-gallery-figure">'
        f'<img src="{url}" alt="{title}" loading="lazy" />'
        f'<figcaption>{title}</figcaption></figure>'
        for title, url in panels
    )
    return f'<div class="qvp-gallery-wrap">{imgs}</div>'


def _build_qvp_theme() -> gr.themes.Base:
    return (
        gr.themes.Base(
            primary_hue=gr.themes.colors.purple,
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
            block_background_fill=_QVP_FIELD_FILL,
            block_background_fill_dark=_QVP_FIELD_FILL,
            panel_background_fill=_QVP_FIELD_FILL,
            panel_background_fill_dark=_QVP_FIELD_FILL,
            input_background_fill=_QVP_FIELD_FILL,
            input_background_fill_dark=_QVP_FIELD_FILL,
            body_text_color="#ede9fe",
            body_text_color_dark="#ede9fe",
            block_label_text_color="#c4b5fd",
            block_label_text_color_dark="#c4b5fd",
            block_title_text_color="#f5f3ff",
            block_title_text_color_dark="#f5f3ff",
            border_color_primary="rgba(255, 255, 255, 0.12)",
            border_color_primary_dark="rgba(255, 255, 255, 0.12)",
            button_primary_background_fill="#7c3aed",
            button_primary_background_fill_dark="#7c3aed",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="rgba(12, 28, 24, 0.92)",
            button_secondary_background_fill_dark="rgba(12, 28, 24, 0.92)",
            button_secondary_text_color="#e0f2f1",
            button_secondary_text_color_dark="#e0f2f1",
            checkbox_label_background_fill="transparent",
            checkbox_label_background_fill_dark="transparent",
            slider_color=_QVP_ACCENT,
            slider_color_dark=_QVP_ACCENT,
            link_text_color=_QVP_ACCENT,
            link_text_color_dark=_QVP_ACCENT,
            link_text_color_hover="#5eead4",
            link_text_color_hover_dark="#5eead4",
            link_text_color_active=_QVP_ACCENT,
            link_text_color_active_dark=_QVP_ACCENT,
            link_text_color_visited=_QVP_ACCENT,
            link_text_color_visited_dark=_QVP_ACCENT,
        )
    )


WALLPAPER_HEAD = f"""
<style id="qvp-wallpaper-style">
#qvp-wallpaper {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -9999 !important;
    pointer-events: none !important;
    background-color: #061210 !important;
    background-image: url('{QVPIC_WALLPAPER_URL}') !important;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
}}
</style>
<script>
(function() {{
    function mountWallpaper() {{
        if (document.getElementById('qvp-wallpaper')) return;
        var wp = document.createElement('div');
        wp.id = 'qvp-wallpaper';
        wp.setAttribute('aria-hidden', 'true');
        document.body.insertBefore(wp, document.body.firstChild);
    }}
    if (document.body) mountWallpaper();
    document.addEventListener('DOMContentLoaded', mountWallpaper);
    window.addEventListener('load', mountWallpaper);
}})();
</script>
"""

QVP_CSS = f"""
:root, :root .dark {{
    --body-background-fill: transparent !important;
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --block-background-fill: {_QVP_FIELD_FILL} !important;
    --panel-background-fill: {_QVP_FIELD_FILL} !important;
    --input-background-fill: {_QVP_FIELD_FILL} !important;
    --body-text-color: #e0f2f1 !important;
    --block-label-text-color: #99f6e4 !important;
    --block-title-text-color: #ccfbf1 !important;
    --border-color-primary: rgba(255, 255, 255, 0.12) !important;
    --link-text-color: {_QVP_ACCENT} !important;
    color-scheme: dark;
}}
html {{ background-color: #061210 !important; min-height: 100% !important; }}
body {{
    background: transparent !important;
    color: #e0f2f1 !important;
    min-height: 100vh !important;
    width: 100% !important;
    overflow-x: hidden !important;
}}
body::before {{
    content: "" !important;
    position: fixed !important;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: -9998 !important;
    pointer-events: none !important;
    background-color: #061210 !important;
    background-image: url('{QVPIC_WALLPAPER_URL}') !important;
    background-size: cover !important;
    background-position: center center !important;
}}
.gradio-container {{
    position: relative !important;
    width: 100% !important;
    max-width: 100% !important;
    background: transparent !important;
}}
.gradio-container .block {{
    background-color: {_QVP_FIELD_FILL} !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(4px);
}}
.gradio-container .markdown, .gradio-container .prose, .gradio-container .markdown p {{
    color: #e0f2f1 !important;
}}
.gradio-container .qvp-source-tab,
.gradio-container .qvp-source-tabs-row button.qvp-source-tab,
.gradio-container .qvp-nav-cell a.qvp-source-tab {{
    color: {_QVP_MATRIX_GREEN} !important;
    -webkit-text-fill-color: {_QVP_MATRIX_GREEN} !important;
    text-decoration: underline !important;
    text-decoration-color: {_QVP_MATRIX_GREEN} !important;
    background: transparent !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    box-shadow: none !important;
    padding: 0 !important;
}}
.gradio-container .qvp-source-tab.active,
.gradio-container .qvp-source-tabs-row button.qvp-source-tab.active {{
    color: {_QVP_LOGO_GOLD} !important;
    -webkit-text-fill-color: {_QVP_LOGO_GOLD} !important;
    text-decoration-color: {_QVP_LOGO_GOLD} !important;
    text-decoration-thickness: 2px !important;
}}
.gradio-container .qvp-optics-panel {{
    background: linear-gradient(165deg, #102820 0%, #0a1a14 38%, #061210 100%) !important;
    border: 3px solid #1d6b5c !important;
    border-radius: 14px !important;
    padding: 0 1rem 1rem !important;
    margin: 0.5rem 0 0.75rem 0 !important;
}}
.gradio-container .qvp-optics-panel-header {{
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.75rem 1.1rem !important;
    padding: 0.7rem 0.85rem 1.35rem !important;
    border-bottom: 1px solid rgba(29, 107, 92, 0.65) !important;
    background: linear-gradient(180deg, #0f2018 0%, #061210 100%) !important;
    min-height: 5.25rem !important;
}}
.gradio-container .qvp-optics-panel-nav {{
    flex: 1 1 18rem !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0.28rem !important;
}}
.gradio-container .qvp-nav-spreadsheet-row {{
    display: grid !important;
    grid-template-columns: 4.75rem repeat(5, minmax(4.5rem, 1fr)) !important;
    gap: 0.2rem 0.45rem !important;
    align-items: center !important;
    width: 100% !important;
}}
.gradio-container .qvp-nav-row-label {{
    justify-self: end !important;
    text-align: right !important;
    color: #e0f2f1 !important;
    font-weight: 600 !important;
}}
.gradio-container .qvp-optics-logo {{
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 0.1rem !important;
    min-width: 10.5rem !important;
    padding-right: 0.65rem !important;
    border-right: 1px solid rgba(29, 107, 92, 0.45) !important;
}}
.gradio-container .qvp-optics-brand {{
    font-size: 0.62rem !important;
    letter-spacing: 0.28em !important;
    color: {_QVP_LOGO_GOLD} !important;
    font-weight: 700 !important;
}}
.gradio-container .qvp-optics-panel-title {{
    font-size: 1.15rem !important;
    letter-spacing: 0.12em !important;
    color: #ccfbf1 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
}}
.gradio-container .qvp-optics-subtitle {{
    font-size: 0.68rem !important;
    letter-spacing: 0.22em !important;
    color: #5eead4 !important;
}}
.gradio-container .qvp-optics-panel .qvp-optics-terminal textarea {{
    background: rgba(2, 10, 4, 0.1) !important;
    border: 2px inset #1a4d2a !important;
    color: #33ff66 !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: 0.78rem !important;
    min-height: 13.5rem !important;
}}
.gradio-container .qvp-optics-keypad {{
    background: linear-gradient(180deg, #0c1814 0%, #061210 100%) !important;
    border: 2px inset #1d4d3f !important;
    border-radius: 10px !important;
    padding: 0.42rem 0.38rem 0.48rem !important;
}}
.gradio-container .qvp-optics-keypad button.qvp-optics-key {{
    flex: 1 1 0 !important;
    min-height: 3rem !important;
    background: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: 1.44rem !important;
    font-weight: 700 !important;
}}
.gradio-container button.qvp-optics-key-home,
.gradio-container button.qvp-optics-key-home span {{
    color: {_QVP_MATRIX_GREEN} !important;
    background: {_QVP_HOME_KEY_BG} !important;
}}
.gradio-container button.qvp-optics-key-defined:not(.active),
.gradio-container button.qvp-optics-key-defined:not(.active) span {{
    color: {_QVP_MATRIX_GREEN} !important;
}}
.gradio-container button.qvp-optics-key.active {{
    background: {_QVP_MATRIX_GREEN} !important;
    color: #000000 !important;
}}
.gradio-container .qvp-optics-dial-wrap {{
    background: rgba(0, 0, 0, 0.22) !important;
    border: 1px solid #1d4d3f !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.65rem 0.45rem !important;
}}
.gradio-container .qvp-gallery-wrap {{
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)) !important;
    gap: 0.75rem !important;
    width: 100% !important;
}}
.gradio-container .qvp-gallery-figure img {{
    width: 100% !important;
    border-radius: 8px !important;
    background: rgba(6, 18, 16, 0.35) !important;
}}
.gradio-container .qvp-gallery-figure figcaption {{
    color: #99f6e4 !important;
    font-size: 0.82rem !important;
    text-align: center !important;
    margin-top: 0.35rem !important;
}}
.gradio-container button.qvp-panel-minimize {{
    border: 1px solid {_QVP_TAB_GREEN_BORDER} !important;
    background: {_QVP_TAB_GREEN_BG} !important;
    color: {_QVP_TAB_GREEN_TEXT} !important;
}}
.gradio-container .qvp-links-panel {{
    margin: 0 0 0.35rem 0 !important;
    padding: 0.65rem 0.85rem !important;
}}
.gradio-container .qvp-figure-panel img {{
    width: 100% !important;
    object-fit: contain !important;
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
        theme=_build_qvp_theme(),
        head=WALLPAPER_HEAD,
        css=QVP_CSS,
        fill_width=True,
    ) as demo:
        current_page = gr.State("demo")
        scope_open = gr.State(False)

        with gr.Column(visible=False, elem_classes=["qvp-links-panel"]) as panel_scope:
            with gr.Row():
                gr.Markdown("### Scope — simulation demo")
                scope_minimize_btn = gr.Button("▲", elem_classes=["qvp-panel-minimize"], scale=0)
            gr.Markdown(SCOPE_MD)

        with gr.Column(visible=True) as page_demo:
            with gr.Group(elem_classes=["qvp-optics-panel"]):
                with gr.Row(elem_classes=["qvp-optics-panel-header"]):
                    gr.HTML(OPTICS_LOGO_HTML)
                    with gr.Column(elem_classes=["qvp-optics-panel-nav"], scale=1):
                        with gr.Row(elem_classes=["qvp-nav-spreadsheet-row"]):
                            gr.HTML('<span class="qvp-nav-row-label">Source:</span>')
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                tab_demo_btn = gr.Button(
                                    "Live Demo",
                                    elem_classes=["qvp-source-tab", "active"],
                                    interactive=False,
                                    scale=0,
                                    variant="secondary",
                                )
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                tab_gallery_btn = gr.Button(
                                    "Gallery",
                                    elem_classes=["qvp-source-tab"],
                                    scale=0,
                                    variant="secondary",
                                )
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                tab_scope_btn = gr.Button(
                                    "Scope",
                                    elem_classes=["qvp-source-tab"],
                                    scale=0,
                                    variant="secondary",
                                )
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                gr.HTML('<span class="qvp-nav-cell-empty" aria-hidden="true">&nbsp;</span>')
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                gr.HTML('<span class="qvp-nav-cell-empty" aria-hidden="true">&nbsp;</span>')
                        with gr.Row(elem_classes=["qvp-nav-spreadsheet-row"]):
                            gr.HTML('<span class="qvp-nav-row-label">Links:</span>')
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                gr.HTML(_external_tab_html("GitHub", GITHUB_URL, "github"))
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                gr.HTML(_external_tab_html("vqc_proto", VQC_URL, "vqc"))
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                gr.HTML(_external_tab_html("hfb", HFB_URL, "hfb"))
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                gr.HTML('<span class="qvp-nav-cell-empty" aria-hidden="true">&nbsp;</span>')
                            with gr.Column(elem_classes=["qvp-nav-cell"], scale=1, min_width=72):
                                gr.HTML('<span class="qvp-nav-cell-empty" aria-hidden="true">&nbsp;</span>')

                optics_terminal = gr.Textbox(
                    label="Matrix status display — selection menu · d-pad nav",
                    value=_optics_terminal_menu(0),
                    lines=12,
                    max_lines=16,
                    interactive=False,
                    elem_classes=["qvp-optics-terminal-wrap", "qvp-optics-terminal"],
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

                with gr.Column(elem_classes=["qvp-optics-keypad"]):
                    with gr.Row(elem_classes=["qvp-optics-dpad-row"], equal_height=True):
                        for nav_key in TERM_NAV_KEYS:
                            term_all_btns[nav_key] = gr.Button(
                                _dpad_row_labels[nav_key],
                                elem_classes=_term_key_btn_classes(nav_key, ""),
                                scale=1,
                                variant="secondary",
                            )
                    with gr.Row(elem_classes=["qvp-optics-prog-row"], equal_height=True):
                        for index in range(1, 13):
                            key_id = _term_key_id(index)
                            term_all_btns[key_id] = gr.Button(
                                _term_keypad_label(index),
                                elem_classes=_term_key_btn_classes(key_id, ""),
                                scale=1,
                                variant="secondary",
                            )
                    with gr.Row(elem_classes=["qvp-optics-prog-row"], equal_height=True):
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
                    *[term_all_btns[key_id] for key_id in TERM_KEYPAD_CONTROL_ORDER],
                    term_active_key,
                    term_ui_state,
                ]

                with gr.Row(elem_classes=["qvp-optics-tune-row"]):
                    bake_steps = gr.Slider(
                        10,
                        150,
                        value=_DEFAULTS["bake_steps"],
                        step=5,
                        label="Bake steps / fact",
                        elem_classes=["qvp-optics-dial-wrap"],
                    )
                    bandwidth = gr.Slider(
                        0.1,
                        1.0,
                        value=_DEFAULTS["bandwidth"],
                        step=0.05,
                        label="Read bandwidth",
                        elem_classes=["qvp-optics-dial-wrap"],
                    )
                    drift_samples = gr.Slider(
                        10,
                        80,
                        value=_DEFAULTS["drift_samples"],
                        step=5,
                        label="Drift samples",
                        elem_classes=["qvp-optics-dial-wrap"],
                    )
                    max_facts = gr.Slider(
                        3,
                        12,
                        value=_DEFAULTS["max_facts"],
                        step=1,
                        label="Max demo facts",
                        elem_classes=["qvp-optics-dial-wrap"],
                    )

                with gr.Row(elem_classes=["qvp-optics-tune-row"]):
                    query_text = gr.Textbox(
                        label="Query recall text",
                        value=_DEFAULTS["query_text"],
                        elem_classes=["qvp-optics-dial-wrap"],
                    )
                    use_vqc = gr.Checkbox(
                        label="VQCEnhanced conduit (experimental)",
                        value=_DEFAULTS["use_vqc"],
                        elem_classes=["qvp-optics-dial-wrap"],
                    )
                    include_lattice = gr.Checkbox(
                        label="Include braided lattice PNG",
                        value=True,
                        info=lattice_info,
                        elem_classes=["qvp-optics-dial-wrap"],
                    )

            term_all_btns["clear"].click(
                _make_term_clear_click("clear"),
                inputs=[optics_terminal, term_ui_state],
                outputs=term_keypad_outputs,
            )
            for hold_key in TERM_DPAD_HOLD_KEYS:
                term_all_btns[hold_key].click(
                    _make_term_dpad_click(hold_key),
                    inputs=[optics_terminal, term_ui_state],
                    outputs=term_keypad_outputs,
                )
            term_all_btns[TERM_KEYPAD_HOME_KEY].click(
                _make_term_home_momentary(),
                inputs=[term_active_key, term_ui_state],
                outputs=term_keypad_outputs,
            )
            for index in range(1, TERM_KEYPAD_COUNT + 1):
                key_id = _term_key_id(index)
                if index == 1:
                    continue
                if index in TERM_KEYPAD_DEFINED:
                    action = TERM_KEYPAD_DEFINED[index]
                    term_all_btns[key_id].click(
                        _make_term_stream_click(
                            key_id,
                            TERM_KEYPAD_STREAMERS[action],
                            menu_action=action,
                        ),
                        inputs=[term_ui_state],
                        outputs=term_keypad_outputs,
                    )
                else:
                    term_all_btns[key_id].click(
                        _make_term_latch_click(key_id),
                        inputs=[optics_terminal, term_ui_state],
                        outputs=term_keypad_outputs,
                    )

            tune_inputs = [bake_steps, bandwidth, use_vqc, drift_samples, max_facts]
            benchmark_btn = gr.Button("Run benchmark", variant="primary")
            query_btn = gr.Button("Run query recall", variant="secondary")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    metrics_out = gr.Textbox(label="Benchmark / recall metrics", lines=16)
                with gr.Column(scale=2):
                    lattice_figure = gr.Image(
                        label="Braided lattice",
                        type="filepath",
                        elem_classes=["qvp-figure-panel"],
                    )

            benchmark_btn.click(
                run_benchmark,
                inputs=[*tune_inputs, include_lattice],
                outputs=[metrics_out, lattice_figure],
            )
            query_btn.click(
                run_query,
                inputs=[query_text, bake_steps, bandwidth, use_vqc, max_facts],
                outputs=[metrics_out],
            )

        with gr.Column(visible=False, elem_classes=["qvp-gallery-page"]) as page_gallery:
            with gr.Row(elem_classes=["qvp-source-tabs-row"]):
                gr.HTML('<span class="qvp-nav-row-label">Source:</span>')
                gal_tab_demo_btn = gr.Button("Live Demo", elem_classes=["qvp-source-tab"], scale=0, variant="secondary")
                gal_tab_gallery_btn = gr.Button(
                    "Gallery",
                    elem_classes=["qvp-source-tab", "active"],
                    interactive=False,
                    scale=0,
                    variant="secondary",
                )
            gr.Markdown("## Gallery — QVPIC visuals from the repository")
            gr.HTML(_gallery_grid_html())
            gr.Markdown(
                f"[qvpic.png]({_GALLERY_BANNER_URL}) · "
                f"[qvpic_banner.png]({_GALLERY_BANNER2_URL})"
            )

        scope_outputs = [panel_scope, tab_scope_btn, scope_open]
        nav_outputs = [
            page_demo,
            page_gallery,
            tab_demo_btn,
            tab_gallery_btn,
            panel_scope,
            tab_scope_btn,
            scope_open,
            gal_tab_demo_btn,
            gal_tab_gallery_btn,
            current_page,
        ]
        tab_demo_btn.click(lambda: _nav_to_page("demo"), outputs=nav_outputs)
        tab_gallery_btn.click(lambda: _nav_to_page("gallery"), outputs=nav_outputs)
        gal_tab_demo_btn.click(lambda: _nav_to_page("demo"), outputs=nav_outputs)
        gal_tab_gallery_btn.click(lambda: _nav_to_page("gallery"), outputs=nav_outputs)
        tab_scope_btn.click(_toggle_scope, inputs=[scope_open], outputs=scope_outputs)
        scope_minimize_btn.click(_minimize_scope, outputs=scope_outputs[:3])
        demo.load(_term_boot_home, outputs=term_keypad_outputs)

        gr.Markdown(
            f"MIT license · VQC patent embodiment · "
            f"[QVPIC on GitHub]({GITHUB_URL}) · [HF Space]({HF_SPACE_URL})"
        )
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