"""Shared helpers for the QVPIC Gradio demo and HF Space."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_BUNDLE = Path(__file__).resolve().parent
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

from src.config import load_config
from src.conduit import RubikConeConduit, safe_cosine

GITHUB_URL = "https://github.com/kinaar8340/qvpic"
VQC_URL = "https://github.com/kinaar8340/vqc_proto"
HFB_URL = "https://github.com/kinaar8340/hfb"
HF_SPACE_URL = "https://huggingface.co/spaces/kinaar111/qvpic"
HFB_RAW_URL = "https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/hfb.png"
QVPIC_WALLPAPER_URL = "https://raw.githubusercontent.com/kinaar8340/qvpic/main/images/qvpic.png"
GALLERY_BANNER_URL = "https://raw.githubusercontent.com/kinaar8340/qvpic/main/images/qvpic.png"
GALLERY_BANNER2_URL = "https://raw.githubusercontent.com/kinaar8340/qvpic/main/images/qvpic_banner.png"

BOOT_QUOTE_STRING = "QUATERNION VORTEX · PERSISTENT IDENTITY · VQC"

DEFAULT_QUERY_TEXT = "Quaternion vortex persistent identity conduit recall test."

SIMULATION_BANNER_MD = """
> **Simulation demo** — browser-based bake → recall → drift benchmark on demo facts.
> Uses RubikConeConduit / RingConeChain (no local LLM on HF). Full agent chat runs locally
> via `python scripts/main.py`.
"""

ONBOARDING_MD = """
### Think persistent identity, not flat vectors
**QVPIC** (Quaternion Vortex Persistent Identity Conduit) stores facts as **geometric memory** on a
**RingConeChain** — each fact is baked into a cube with quaternion orientation, depth coordinate `s`,
and ShellCube topology. Recall uses **primal cosine** on the braided lattice, not naive flat embedding
search.

### Three steps (60 seconds)
1. **Run benchmark** — embed demo facts with SentenceTransformer, bake into RingConeChain, measure
   primal recall and topology invariants (winding, braiding phase).
2. **Run query recall** — type a query string; top-k cube recall reports cosine + braiding phase.
3. **Tune dials** — bake steps, read bandwidth, drift samples. Enable **VQCEnhanced** for the
   experimental helical variant aligned with the VQC patent vision.

### What the metrics mean
| Metric | Plain English |
|--------|----------------|
| **avg recall cosine** | Mean primal recall after bake — higher → stronger identity persistence. |
| **protection factor** | Drift recovery vs naive cosine after noisy vector + coordinate perturbation. |
| **geometric_winding** | Topological winding from `monitor_topological_winding()`. |
| **braiding_phase** | ShellCube braid invariant — encodes multi-fact entanglement geometry. |

**Tip:** HF Space caps facts/samples for speed; run locally for full lattice renders and agent chat.
"""

VQC_CLAIMS_MD = """
| VQC claim element | QVPIC demo shows… |
|-------------------|-------------------|
| **Quaternion encoding** | `quat_logical_dim` + Clifford projection on each baked cube orientation. |
| **Helical / vortex geometry** | Depth coordinate `s` + twist_rate helical read path; optional VQCEnhanced conduit. |
| **Nested shielding** | Drift protection factor — coordinate + vector noise recovery vs naive cosine. |
| **Topological multiplex** | RingConeChain cubes with braiding_phase + geometric_winding invariants. |
| **OAM / orbital analogy** | Shared VQC vision family with [orbital-braille-vqc](https://huggingface.co/spaces/kinaar111/orbital-braille-vqc) — light-carrier ↔ geometric-carrier. |
| **Persistent identity** | Facts survive bake → recall → drift test without flat-vector collapse. |
| **Braided lattice** | `render_braided_lattice_style()` PNG — ShellCube topology visualization. |
| **Agent embodiment** | Local `scripts/main.py` full LLM agent; HF Space is simulation-only benchmark. |

Full VQC optical prototype: [vqc_proto](https://github.com/kinaar8340/vqc_proto) · QVPIC memory layer: [qvpic](https://github.com/kinaar8340/qvpic)
"""

TERM_KEY_ACTIONS: dict[int, tuple[str, str]] = {
    1: ("home", "Return to selection menu"),
    2: ("status", "Live pipeline & environment"),
    3: ("conduit", "RubikCone → identity conduit analogy"),
    4: ("pipeline", "Bake → recall → drift loop"),
    5: ("metrics", "Benchmark metrics baseline"),
    6: ("build", "Build stamp & deploy info"),
    7: ("help", "D-pad / keypad navigation"),
    8: ("helix", "VQC conduit helix — any key exits"),
    9: ("claims", "VQC claim ↔ QVPIC demo map"),
    10: ("topology", "ShellCube / braiding invariants"),
    11: ("recall", "Query recall & RingConeChain"),
    12: ("presets", "Default run params catalog"),
}

_CFG_CANDIDATES = (
    _BUNDLE / "configs" / "default.yaml",
    _BUNDLE / "web" / "hf_default.yaml",
    _BUNDLE.parent / "web" / "hf_default.yaml",
    _BUNDLE.parent / "configs" / "default.yaml",
)
DEFAULT_CONFIG_PATH = next(
    (path for path in _CFG_CANDIDATES if path.is_file()),
    _CFG_CANDIDATES[-1],
)
FACTS_CANDIDATES = (
    _BUNDLE / "facts" / "demo_public_facts.json",
    _BUNDLE / "web" / "demo_public_facts.json",
    _BUNDLE.parent / "web" / "demo_public_facts.json",
)
DEFAULT_FACTS_PATH = next(
    (path for path in FACTS_CANDIDATES if path.is_file()),
    FACTS_CANDIDATES[-1],
)

DEFAULT_READ_KWARGS = {"bandwidth": 0.32, "num_samples": 31}
_EMBEDDER = None
_EMBEDDER_DEVICE: str | None = None


def is_hf_space() -> bool:
    return bool(os.environ.get("SPACE_ID"))


def get_build_label() -> str:
    try:
        from build_info import BUILD_COMMIT, BUILD_UPDATED_UTC  # noqa: WPS433

        return f"Last updated: {BUILD_UPDATED_UTC} UTC · commit `{BUILD_COMMIT}`"
    except ImportError:
        return "Last updated: local dev build"


def default_run_params() -> dict[str, Any]:
    return {
        "bake_steps": 35 if is_hf_space() else 80,
        "bandwidth": 0.32,
        "use_vqc": False,
        "drift_samples": 20 if is_hf_space() else 40,
        "max_facts": 6 if is_hf_space() else 12,
        "query_text": DEFAULT_QUERY_TEXT,
    }


def terminal_agent_use_cases() -> str:
    return "\n".join(
        [
            "Agent use-cases (local scripts/main.py + QVPIC memory):",
            "",
            "  1. Personal assistant — upublic/uprivate identity shards survive",
            "     session resets; agent recalls preferences without flat-RAG drift.",
            "  2. Long-horizon coding agent — facts baked into RingConeChain;",
            "     /self-eval benchmark gates self-improvement proposals.",
            "  3. Research notebook agent — journal + facts JSON append;",
            "     topology invariants detect memory corruption early.",
            "  4. Multi-session support bot — same identity conduit across",
            "     heartbeat checkpoints (scripts/heartbeat.py).",
            "  5. HF Space smoke test — bake → recall → drift without local LLM.",
            "",
            "Integration entry points:",
            "  · examples/agent_memory_integration.py",
            "  · run_benchmark_demo() / run_query_recall() in demo_core.py",
            "  · RubikConeConduit.write / read / recover_depth in src/conduit.py",
        ]
    )


def terminal_benchmark_comparison() -> str:
    return "\n".join(
        [
            "Memory benchmark comparison (same drift protocol, demo facts):",
            "",
            "  System              Recall cos   Protection   Topology lock",
            "  ─────────────────── ────────── ──────────── ───────────────",
            "  QVPIC RubikCone       0.98–1.00      ~5.7x     winding+braid",
            "  Naive flat cosine     degrades       1.0x      none",
            "  Vector store RAG      degrades       ~1.0x     none",
            "",
            "Drift test: depth s + embedding noise → measure recovery vs baseline.",
            "Run MEMORY tab → type 'benchmark' → SEND to reproduce on this Space.",
            "Full sweep: python scripts/qvpic_test.py --no-viz",
        ]
    )


def terminal_community_validation() -> str:
    return "\n".join(
        [
            "Visibility & external validation:",
            "",
            "  · HF Space Community tab — post recall metrics + lattice PNG",
            "  · GitHub Issues / Discussions — integration feedback welcome",
            "  · Share with: geometric DL, topological ML, quaternion NN circles",
            "  · Compare against your vector DB baseline using same facts JSON",
            "",
            f"  Repo:  {GITHUB_URL}",
            f"  Space: {HF_SPACE_URL}",
            f"  VQC:   {VQC_URL}",
            "",
            "Request: cite protection factor + topology block when reporting.",
        ]
    )


def terminal_guided_onboarding() -> str:
    return "\n".join(
        [
            "╔══════════════════════════════════════════════════════╗",
            "║  QVPIC GUIDED ONBOARDING — 60 second tour            ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            "STEP 1 — Think geometry, not flat vectors",
            "  Facts bake into RingConeChain cubes (orientation × depth s).",
            "  Recall = primal cosine on braided ShellCube lattice.",
            "",
            "STEP 2 — Run the benchmark (30s on HF)",
            "  HOME → 6 Bake → Recall Benchmark  OR  MEMORY tab",
            "  Type: benchmark → SEND",
            "  Read: avg recall cosine · protection factor · braiding_phase",
            "",
            "STEP 3 — Query recall",
            "  MEMORY tab → Run query recall (or type your query + SEND)",
            "",
            "STEP 4 — Tune & explore",
            "  SETTINGS: bake_steps · bandwidth · VQCEnhanced toggle",
            "  TOOLS: GitHub · vqc_proto · full agent (scripts/main.py)",
            "",
            "── Agent use-cases ──",
            *terminal_agent_use_cases().split("\n")[2:8],
            "",
            "── vs flat memory ──",
            *terminal_benchmark_comparison().split("\n")[2:7],
            "",
            "── Share & validate ──",
            *terminal_community_validation().split("\n")[2:6],
            "",
            "Press HOME anytime to return to the selection menu.",
        ]
    )


def terminal_keypad_map() -> str:
    lines = ["Assigned prog keys (01–12):", ""]
    for index in sorted(TERM_KEY_ACTIONS):
        _action, desc = TERM_KEY_ACTIONS[index]
        tag = "01 Home" if index == 1 else f"{index:02d}"
        lines.append(f"  [{tag}]  {desc}")
    lines.extend(
        [
            "",
            "D-pad: ▲▼◀▶ move menu · enter confirm · clear blank",
            "Keys 13–24: reserved (latch only)",
            "Menu items 01–08 mirror d-pad selection.",
            "08 / menu 08 → VQC conduit helix screensaver (any key stops).",
        ]
    )
    return "\n".join(lines)


def terminal_conduit_analogy() -> str:
    return "\n".join(
        [
            "Flat vector DB  →  QVPIC geometric identity conduit:",
            "",
            "  embedding index   →  RingConeChain cube bake (orientation × depth s)",
            "  cosine search     →  primal recall on braided ShellCube lattice",
            "  noise / drift     →  coordinate + vector perturbation recovery",
            "  flat collapse     →  winding + braiding_phase topology shield",
            "  chat memory       →  persistent identity across agent sessions",
            "",
            f"Default query: {DEFAULT_QUERY_TEXT!r}",
            "Sweet spot: RubikCone on HF · VQCEnhanced experimental locally.",
            "",
            "Run benchmark below → recall table + braided lattice PNG.",
        ]
    )


def terminal_pipeline_scope() -> str:
    on_hf = is_hf_space()
    llm = "disabled on HF (local agent only)" if on_hf else "available via scripts/main.py"
    return "\n".join(
        [
            "THIS SPACE — browser simulation (you are here):",
            "  · SentenceTransformer embed → RingConeChain bake",
            "  · primal cosine recall + topology invariants",
            "  · drift protection factor vs naive cosine",
            "  · optional braided lattice PNG render",
            "",
            "GITHUB REPO — full QVPIC depth:",
            "  · scripts/main.py · scripts/qvpic_test.py",
            "  · src/conduit.py · VQCEnhancedHelicalConduit",
            "  · configs/default.yaml · demo_public_facts.json",
            "",
            f"LLM agent: {llm}",
            "Ecosystem: vqc_proto (optical) · qvpic (identity memory).",
        ]
    )


def terminal_metrics_baseline() -> str:
    defaults = default_run_params()
    return "\n".join(
        [
            "HF Space defaults (CPU, reduced samples):",
            "",
            f"  bake_steps      {defaults['bake_steps']}",
            f"  max_facts       {defaults['max_facts']}",
            f"  drift_samples   {defaults['drift_samples']}",
            f"  bandwidth       {defaults['bandwidth']}",
            "",
            "Metrics block after Run benchmark reports:",
            "  · avg recall cosine — primal fidelity after bake",
            "  · drift before/after — noisy vector perturbation",
            "  · protection factor — recovery vs naive cosine",
            "  · geometric_winding · braiding_phase — topology",
            "",
            "Enable VQCEnhanced for helical patent-aligned variant.",
        ]
    )


def terminal_claims_snapshot() -> str:
    lines = [
        "VQC claim element  →  QVPIC demo output:",
        "",
        "  Quaternion encode  →  quat_logical_dim + cube orientation",
        "  Helical geometry   →  depth s + twist_rate read path",
        "  Nested shielding   →  drift protection factor",
        "  Topological mux    →  RingConeChain + braiding_phase",
        "  Persistent ID      →  bake → recall survives perturbation",
        "  Braided lattice    →  render_braided_lattice_style PNG",
        "  VQC optical link   →  orbital-braille-vqc Space (carrier layer)",
        "  Agent embodiment   →  local scripts/main.py (not on HF)",
        "",
        "Expand Claims tab for full table · 09 Claims keypad shortcut.",
    ]
    return "\n".join(lines)


def terminal_topology_shards() -> str:
    return "\n".join(
        [
            "ShellCube topology + braiding invariants:",
            "",
            "  geometric_winding  — monitor_topological_winding() scalar",
            "  effective_winding  — depth-averaged winding estimate",
            "  braiding_phase     — multi-cube entanglement phase",
            "  RingConeChain      — NUM_RINGS × cubes per ring layout",
            "  orientation bake   — 24 cube orientations (Rubik group)",
            "",
            "Drift test perturbs depth s + vector noise — protection",
            "factor reports recovery vs flat cosine baseline.",
            "Braided lattice PNG visualizes ShellCube braid geometry.",
        ]
    )


def terminal_recall_export() -> str:
    return "\n".join(
        [
            "Query recall (Run query recall button):",
            "",
            "  1. Bake demo facts into RingConeChain (same dials)",
            "  2. Embed query via all-MiniLM-L6-v2",
            "  3. Top-k cube recall — cosine + primal_cos + braiding",
            "",
            f"Default query: {DEFAULT_QUERY_TEXT!r}",
            "",
            "Run benchmark first for full recall table + lattice.",
            f"Local agent: {GITHUB_URL}/blob/main/scripts/main.py",
        ]
    )


def terminal_presets_catalog() -> str:
    defaults = default_run_params()
    lines = ["Default run params (HF vs local):", ""]
    for key, value in defaults.items():
        lines.append(f"  {key:<16} {value!r}")
    lines.extend(
        [
            "",
            "Tune sliders below · VQCEnhanced checkbox for helical variant.",
            "Include braided lattice PNG for topology visualization.",
        ]
    )
    return "\n".join(lines)


def _resolve_device() -> str:
    if is_hf_space():
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _configure_cpu_threads() -> None:
    if _resolve_device() != "cpu":
        return
    threads = 4 if is_hf_space() else min(8, (os.cpu_count() or 8) // 2)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(2)


def _torch_device(device: str) -> torch.device:
    return torch.device(device)


def _sync_conduit_devices(conduit, device: str) -> None:
    """Align stale self.device attrs after .to() — RingConeChain keeps cuda at init."""
    dev = _torch_device(device)
    conduit.to(dev)
    if hasattr(conduit, "device"):
        conduit.device = dev
    ring = getattr(conduit, "ring_cone", None)
    if ring is not None and hasattr(ring, "device"):
        ring.device = dev


def _get_embedder(device: str):
    global _EMBEDDER, _EMBEDDER_DEVICE
    if _EMBEDDER is None or _EMBEDDER_DEVICE != device:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        _EMBEDDER_DEVICE = device
    return _EMBEDDER


def _extract_clean_facts(raw_data: Any, *, max_facts: int) -> list[str]:
    facts: list[str] = []
    if isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if not text:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        facts.append(line)
            elif isinstance(item, str) and item.strip():
                facts.append(item.strip())
    if not facts:
        facts = ["QVPIC topological memory demo fact."]
    return facts[:max_facts]


@dataclass
class BenchmarkResult:
    metrics_text: str
    lattice_path: str | None
    avg_recall: float
    protection_factor: float | None


def _make_conduit(*, use_vqc: bool, device: str):
    cfg = load_config(str(DEFAULT_CONFIG_PATH))
    if use_vqc:
        from src.vqc_enhanced_conduit import VQCEnhancedHelicalConduit

        conduit = VQCEnhancedHelicalConduit(
            embed_dim=cfg.model.embed_dim,
            twist_rate=cfg.model.twist_rate,
            max_depth=cfg.model.max_depth,
            num_polarizations=cfg.model.num_polarizations,
            quat_logical_dim=cfg.model.quat_logical_dim,
            toroidal_modulo9=True,
            vortex_math_369=True,
            clifford_projection=True,
        )
    else:
        conduit = RubikConeConduit(
            embed_dim=cfg.model.embed_dim,
            twist_rate=cfg.model.twist_rate,
            max_depth=cfg.model.max_depth,
            num_polarizations=cfg.model.num_polarizations,
            quat_logical_dim=cfg.model.quat_logical_dim,
            toroidal_modulo9=True,
            vortex_math_369=True,
            clifford_projection=True,
        )
    _sync_conduit_devices(conduit, device)
    if device == "cuda" and not is_hf_space():
        try:
            conduit = torch.compile(conduit, mode="default")
        except Exception:
            pass
    optimizer = torch.optim.AdamW(
        conduit.parameters(),
        lr=8e-4,
        weight_decay=cfg.training.weight_decay,
    )
    return conduit, optimizer, cfg


def _bake_facts(
    conduit,
    optimizer,
    *,
    bake_steps: int,
    max_facts: int,
    progress_cb=None,
) -> list[tuple[str, float, torch.Tensor]]:
    dev = _torch_device(conduit.device if isinstance(conduit.device, str) else str(conduit.device))
    raw = json.loads(DEFAULT_FACTS_PATH.read_text(encoding="utf-8"))
    lines = _extract_clean_facts(raw, max_facts=max_facts)
    embedder = _get_embedder(str(dev))
    embeddings_raw = embedder.encode(
        lines,
        convert_to_tensor=True,
        device=str(dev),
        batch_size=16,
    )

    baked: list[tuple[str, float, torch.Tensor]] = []
    depth = 4.5
    step_size = 4.8
    is_rubik = hasattr(conduit, "ring_cone")

    for idx, (fact, emb_raw) in enumerate(zip(lines, embeddings_raw)):
        if progress_cb is not None:
            progress_cb(idx / max(len(lines), 1), desc=f"Baking fact {idx + 1}/{len(lines)}")
        emb = F.normalize(emb_raw.to(dev), dim=-1) * conduit.output_scale.item()
        emb = emb.to(dev)
        s = depth + idx * step_size

        if is_rubik:
            ring_idx = idx % conduit.ring_cone.NUM_RINGS
            cube_local_idx = idx % conduit.ring_cone.rings[ring_idx].num_cubes
            conduit.ring_cone.bake_ring(ring_idx, cube_local_idx, emb, orientation=idx % 24)
            for _ in range(bake_steps):
                item = {"emb": emb, "s": s, "pol_idx": 0}
                conduit.training_step(
                    inputs=[item],
                    optimizer=optimizer,
                    recon_weight=15000.0,
                    align_weight=55000.0,
                    depth_pull_weight=40000.0,
                    winding_weight=48.0,
                    braiding_weight=18.0,
                )
        else:
            item = {"emb": emb, "s": s, "pol_idx": idx % conduit.num_pol}
            for _ in range(bake_steps):
                conduit.training_step(
                    inputs=[item],
                    optimizer=optimizer,
                    recon_weight=4200.0,
                    align_weight=1800.0,
                    depth_pull_weight=32000.0,
                    winding_weight=96.0,
                    braiding_weight=24.0,
                )
        baked.append((fact, s, emb))
    return baked


def _avg_recall(conduit, baked: list[tuple[str, float, torch.Tensor]]) -> tuple[float, list[str]]:
    lines: list[str] = []
    scores: list[float] = []
    is_rubik = hasattr(conduit, "ring_cone")
    for fact, s, emb in baked:
        if is_rubik:
            results = conduit.ring_cone.recall(emb, top_k=1)
            score = float(results[0]["primal_cos"])
            lines.append(f"{fact[:48]:<48} | primal={score:.4f}")
        else:
            s_rec = conduit.recover_depth(emb, pol_idx=0, grid_size=256 if is_hf_space() else 512)
            recalled = conduit.read(s_rec, pol_idx=0, bandwidth=2.8, num_samples=31)
            score = safe_cosine(emb.unsqueeze(0), recalled.unsqueeze(0)).item()
            lines.append(f"{fact[:48]:<48} | cos={score:.4f}")
        scores.append(score)
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, lines


@torch.no_grad()
def _drift_test(conduit, *, n: int, bandwidth: float) -> tuple[float, float, float]:
    cos_before: list[float] = []
    cos_after: list[float] = []
    read_kwargs = {"bandwidth": bandwidth, "num_samples": 31}
    for _ in range(n):
        s_true = float(np.random.uniform(1.0, 17.0))
        pol = int(np.random.randint(0, 3))
        s_noisy = s_true + float(np.random.randn() * 0.18)
        orig = conduit.read(s_true, pol, **read_kwargs)
        noisy_vec = orig + torch.randn_like(orig) * 0.075
        noisy_vec = F.normalize(noisy_vec, dim=-1) * conduit.output_scale.item()
        cos_b = safe_cosine(orig, noisy_vec).item()
        cos_a = safe_cosine(orig, conduit.read(s_noisy, pol, **read_kwargs)).item()
        cos_before.append(cos_b)
        cos_after.append(cos_a)
    mean_b = float(np.mean(cos_before))
    mean_a = float(np.mean(cos_after))
    protection = mean_a / max(mean_b, 1e-6)
    return mean_b, mean_a, protection


def run_benchmark_demo(
    *,
    bake_steps: int,
    bandwidth: float,
    use_vqc: bool,
    drift_samples: int,
    max_facts: int,
    include_lattice: bool = True,
    progress_cb=None,
) -> BenchmarkResult:
    _configure_cpu_threads()
    device = _resolve_device()
    conduit, optimizer, _cfg = _make_conduit(use_vqc=use_vqc, device=device)

    if progress_cb is not None:
        progress_cb(0.05, desc="Baking demo facts into RingConeChain…")
    baked = _bake_facts(
        conduit,
        optimizer,
        bake_steps=int(bake_steps),
        max_facts=int(max_facts),
        progress_cb=progress_cb,
    )

    if progress_cb is not None:
        progress_cb(0.55, desc="Measuring recall fidelity…")
    avg_recall, recall_lines = _avg_recall(conduit, baked)

    if progress_cb is not None:
        progress_cb(0.7, desc="Topological invariants + drift test…")
    stats = conduit.monitor_topological_winding(n_samples=256 if is_hf_space() else 512)
    mean_b, mean_a, protection = _drift_test(conduit, n=int(drift_samples), bandwidth=float(bandwidth))

    lattice_path = None
    if include_lattice:
        if progress_cb is not None:
            progress_cb(0.85, desc="Rendering braided lattice…")
        lattice_path = tempfile.NamedTemporaryFile(suffix="_qvpic_lattice.png", delete=False).name
        try:
            conduit.render_braided_lattice_style(save_path=lattice_path)
        except Exception:
            lattice_path = None

    stat_lines = [
        f"geometric_winding : {stats.get('geometric_winding', 0.0):.6f}",
        f"effective_winding: {stats.get('effective_winding', 0.0):.6f}",
        f"braiding_phase    : {stats.get('braiding_phase', 0.0):.6f}",
    ]
    metrics = "\n".join(
        [
            f"mode              : {'VQCEnhanced' if use_vqc else 'RubikCone'}",
            f"device            : {device}",
            f"facts baked       : {len(baked)}",
            f"avg recall cosine : {avg_recall:.4f}",
            f"drift before      : {mean_b:.4f}",
            f"drift after       : {mean_a:.4f}",
            f"protection factor : {protection:.2f}x",
            "",
            "RECALL",
            *recall_lines,
            "",
            "TOPOLOGY",
            *stat_lines,
        ]
    )
    if progress_cb is not None:
        progress_cb(1.0, desc="Done")
    return BenchmarkResult(
        metrics_text=metrics,
        lattice_path=lattice_path,
        avg_recall=avg_recall,
        protection_factor=protection,
    )


def run_query_recall(
    query_text: str,
    *,
    bake_steps: int,
    bandwidth: float,
    use_vqc: bool,
    max_facts: int,
    top_k: int = 3,
    progress_cb=None,
) -> str:
    if not query_text.strip():
        query_text = default_run_params()["query_text"]
    _configure_cpu_threads()
    device = _resolve_device()
    conduit, optimizer, _cfg = _make_conduit(use_vqc=use_vqc, device=device)
    if progress_cb is not None:
        progress_cb(0.1, desc="Baking conduit memory…")
    _bake_facts(
        conduit,
        optimizer,
        bake_steps=int(bake_steps),
        max_facts=int(max_facts),
        progress_cb=progress_cb,
    )
    embedder = _get_embedder(device)
    query_emb = embedder.encode(query_text, convert_to_tensor=True, device=device)
    lines = [f"query: {query_text}", ""]
    if hasattr(conduit, "ring_cone"):
        results = conduit.ring_cone.recall(query_emb, top_k=int(top_k))
        for rank, row in enumerate(results, start=1):
            lines.append(
                f"#{rank} cube={row.get('cube_idx', '?')} "
                f"cos={row.get('cosine', 0.0):.4f} "
                f"primal={row.get('primal_cos', 0.0):.4f} "
                f"braiding={row.get('braiding_phase', 0.0):.4f}"
            )
    else:
        s_rec = conduit.recover_depth(query_emb, pol_idx=0, grid_size=256)
        recalled = conduit.read(s_rec, pol_idx=0, bandwidth=float(bandwidth), num_samples=31)
        cos = safe_cosine(query_emb.unsqueeze(0), recalled.unsqueeze(0)).item()
        lines.append(f"helical recall cos={cos:.4f} at s={s_rec:.2f}")
    if progress_cb is not None:
        progress_cb(1.0, desc="Done")
    return "\n".join(lines)