---
title: QVPIC Identity Conduit
emoji: 🌀
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.12.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
short_description: Quaternion vortex persistent identity — browser demo
---

# QVPIC — QUARTZ AI SYNTHESIZER

<p align="center">
  <img src="https://raw.githubusercontent.com/kinaar8340/qvpic/main/images/qvpic.png" alt="QVPIC banner" width="100%" style="max-width: 720px; border-radius: 12px;" />
</p>

**Vintage control-panel demo** for the [Quaternion Vortex Persistent Identity Conduit (QVPIC)](https://github.com/kinaar8340/qvpic) — geometric memory for drift-resistant persistent identity, styled after a 1970s/80s hardware synthesizer.

Open the **App** tab — no install required.

---

## Panel layout

| Zone | Purpose |
|------|---------|
| **Top tabs** | CHAT · SETTINGS · HISTORY · MEMORY · TOOLS |
| **Terminal** | Matrix-green phosphor readout (#00FF41 on #0d0d0d) |
| **Command line** | Type a message + **SEND** (or NAV **ENTER**) |
| **NAV D-pad** | Scroll, recall prior commands, submit |
| **PROG 1–4** | Load sample QVPIC prompts |
| **EXEC** | Run query recall or benchmark on current command |
| **CLEAR / HELP / MODE** | Reset terminal, help text, VQC mode info |

---

## Deploy / customize

**Hugging Face Space** — push via repo scripts:

```bash
bash scripts/sync_hf_space.sh
bash scripts/deploy_hf_space.sh "your commit message"
```

**Local:**

```bash
cd web && python gradio_demo.py
```

**Customize Grok-style chat responses** — edit `_simulate_chat_response()` in `web/gradio_demo.py`.

**Wire PROG / tab buttons** — handlers live in `_handle_prog`, `_switch_tab`, `_handle_exec`.

**Tune bake/recall** — open **SETTINGS** tab for sliders; results flow through **EXEC** / **MEMORY**.

---

## Links

- **Source:** [github.com/kinaar8340/qvpic](https://github.com/kinaar8340/qvpic)
- **VQC prototype:** [vqc_proto](https://github.com/kinaar8340/vqc_proto)
- **Local agent:** `python scripts/main.py`

MIT — see the [qvpic repository](https://github.com/kinaar8340/qvpic).