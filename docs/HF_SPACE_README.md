# QVPIC — QUARTZ AI SYNTHESIZER

<p align="center">
  <img src="https://raw.githubusercontent.com/kinaar8340/qvpic/main/images/qvpic.png" alt="QVPIC banner" width="100%" style="max-width: 720px; border-radius: 12px;" />
</p>

**Vintage control-panel demo** for the [Quaternion Vortex Persistent Identity Conduit (QVPIC)](https://github.com/kinaar8340/qvpic) — geometric memory for drift-resistant persistent identity, styled after a 1970s/80s hardware synthesizer.

Open the **App** tab — no install required.

---

## First visit? Start here (Guided Onboarding)

1. Click **HOME** (red LED under CHAT).
2. Type **9** → **SEND** (next page).
3. Type **1** → **SEND** → **Guided Tour (Onboarding)**.
4. Follow the tour, then open **MEMORY** and type `benchmark` → **SEND**.

The tour covers topology basics, agent use-cases, benchmark comparison vs flat memory, and where to share feedback.

---

## Panel layout

| Zone | Purpose |
|------|---------|
| **Top tabs** | CHAT · SETTINGS · HISTORY · MEMORY · TOOLS |
| **HOME** | Selection menu — numbered entries, type index + SEND |
| **Terminal** | Matrix-green phosphor readout (#00FF41 on #0d0d0d) |
| **Command line** | Type a message + **SEND** (or NAV **ENTER**) |
| **NAV D-pad** | Scroll, recall prior commands, submit |
| **PROG 1–16** | Torus visibility, move mode, latch toggles |
| **CLEAR / HELP / MODE** | Reset terminal, help text, VQC mode info |

---

## Three-step quickstart (60 seconds)

| Step | Action | What you learn |
|------|--------|----------------|
| 1 | MEMORY → `benchmark` → SEND | Recall cosine, protection factor, topology invariants |
| 2 | MEMORY → Run query recall | Top-k cube hits + braiding phase |
| 3 | SETTINGS → tune dials → repeat | How bake_steps / bandwidth affect fidelity |

---

## Selection menu map

**Page 1**

| # | Item |
|---|------|
| 1 | Quick Diagnostic |
| 2 | Games (Space Invaders) |
| 3–5 | Vortex / Identity / VQC settings |
| 6 | Bake → Recall Benchmark |
| 7–8 | Diagnostics / Tools |
| 9 | Next page |

**Page 2**

| # | Item |
|---|------|
| 1 | **Guided Tour (Onboarding)** |
| 2–5 | History / Grid / About / Help |
| 6 | Previous page |

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

**Integration docs:** [`docs/INTEGRATIONS.md`](https://github.com/kinaar8340/qvpic/blob/main/docs/INTEGRATIONS.md)

---

## Links

- **Source:** [github.com/kinaar8340/qvpic](https://github.com/kinaar8340/qvpic)
- **Integrations:** [docs/INTEGRATIONS.md](https://github.com/kinaar8340/qvpic/blob/main/docs/INTEGRATIONS.md)
- **VQC prototype:** [vqc_proto](https://github.com/kinaar8340/vqc_proto)
- **Local agent:** `python scripts/main.py`

MIT — see the [qvpic repository](https://github.com/kinaar8340/qvpic).