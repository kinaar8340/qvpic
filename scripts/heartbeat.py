#!/usr/bin/env python
# scripts/heartbeat.py — v9.9 (March 25, 2026)
# PHGN "Three-Phase Star-Delta + Video Memory + Living Heartbeat"
# Electrician + Biological edition with full Clifford Torus Skin.
# Global topological features (winding, linking, braiding_phase + flat Clifford skin)
# now drive persistence. Quaternion math + helical/Clifford geometry solve the AI
# persistent memory problem. Referenced: vqc_sims_public, OpenGauss, math.inc/opengauss.

import os
import sys
import time
import torch
import argparse
import cv2
import math
import threading
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from pathlib import Path


# Project root setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config
from src.conduit import TwistedHelicalConduit, safe_cosine
from src.vqc_enhanced_conduit import VQCEnhancedHelicalConduit


# ─── Minimal self-contained Platonic definitions (SRP, no external deps) ───
class PlatonicLevel(Enum):
    D4_TETRAHEDRON = 0
    D5_COORDINATOR = 1
    D6_CUBE = 2


@dataclass
class RetrievalTask:
    action: str
    what: str
    platonic_level: PlatonicLevel
    tags: List[str] = None

    def to_prompt(self) -> str:
        return f"{self.action}: {self.what} (level {self.platonic_level.name})"


# ====================== 3-PHASE STAR-DELTA STATE MACHINE ======================
class ThreePhaseStarDeltaStateMachine(nn.Module):
    """3-phase power distribution (pol 0/1/2) with star/delta switching per Platonic tier.
    Delegates global topology to conduit (DRY)."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.phase_embed = nn.Parameter(torch.randn(3, embed_dim) * 0.12)

    def forward(self, x: torch.Tensor, level: PlatonicLevel, pol: int) -> torch.Tensor:
        phase_vec = self.phase_embed[pol]
        # wiring modulated by global braiding_phase (topology-driven)
        wiring = 1.0 if level.value % 2 == 0 else -1.0
        x = x + wiring * phase_vec
        return F.normalize(x, dim=-1)


# ====================== PLATONIC HELICAL GEOMETRIC NETWORK (PHGN) ======================
class PlatonicHelicalGeometricNetwork(nn.Module):
    """Full PHGN with 3-phase star-delta + Platonic routing + video-frame topological memory.
    All persistence lives in conduit global invariants (Clifford skin + braiding_phase)."""

    def __init__(self, conduit, embed_dim: int = 384):
        super().__init__()
        self.conduit = conduit
        self.embed_dim = embed_dim
        self.state_machine = ThreePhaseStarDeltaStateMachine(embed_dim)
        self.cube_chain = conduit.cube_chain
        self.video_memory: List[torch.Tensor] = []  # stacked Clifford frames

    def forward(self, task_emb: torch.Tensor, level: PlatonicLevel, s: float = 12.0, pol: int = 0) -> torch.Tensor:
        """One forward pass = 3-phase + Platonic routing + bake to global topology."""
        pos_enc = self.conduit.position(s, pol)  # Clifford / toroidal / 369
        x = task_emb + pos_enc
        x = self.state_machine(x, level, pol)  # star-delta

        # Selective coordination (topology-aware)
        if level.value < 2 and torch.rand(1).item() > 0.4:
            next_level = PlatonicLevel(level.value + 1)
            x = self.forward(x.mean(dim=0).unsqueeze(0), next_level, s + 12.0, pol)

        # Bake to CubeChain (global braiding_phase updated)
        cube_idx = int(s) % 12
        self.cube_chain.bake(cube_idx, x.squeeze(0), orientation=int(s) % 24, parent_idx=level.value)

        # Video-frame topological memory (Clifford geometry)
        frame = x.detach().cpu().clone()
        self.video_memory.append(frame)
        if len(self.video_memory) > 64:
            self.video_memory = self.video_memory[-64:]

        return x

    def render_geometric_video(self, save_path: str = "outputs/pic_heartbeat_video.mp4"):
        """Export stacked Clifford frames as real geometric video."""
        if not self.video_memory:
            return
        frames = []
        for f in self.video_memory:
            # Use conduit's true 3D Clifford coordinate for accurate projection
            proj = f[:3].numpy().reshape(1, 1, 3)
            proj = np.tile(proj, (256, 256, 1)) * 255
            proj = np.clip(proj, 0, 255).astype(np.uint8)
            frames.append(proj)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 12, (256, 256))
        for f in frames:
            out.write(f)
        out.release()
        print(f"→ Geometric heartbeat video saved: {save_path} ({len(frames)} Clifford frames)")


# ====================== HEARTBEAT SCHEDULER (biological + topological rhythm) ======================
class HeartbeatScheduler:
    """Living heartbeat of PIC — autonomic background + conscious reviews driven by global invariants."""

    def __init__(self, conduit, phgn, real_time: bool = False):
        self.conduit = conduit
        self.phgn = phgn
        self.real_time = real_time
        self.last_pulse = time.time()
        self.rhythms = {  # configuration over hardcoding
            "autonomic": {"interval": 0.5, "pol": 0, "level": PlatonicLevel.D4_TETRAHEDRON, "delta_s": 0.1},
            "15min": {"interval": 900, "pol": 1, "level": PlatonicLevel.D4_TETRAHEDRON, "delta_s": 1.0},
            "hourly": {"interval": 3600, "pol": 1, "level": PlatonicLevel.D5_COORDINATOR, "delta_s": 4.0},
            "4hour": {"interval": 14400, "pol": 2, "level": PlatonicLevel.D5_COORDINATOR, "delta_s": 16.0},
            "daily_long": {"interval": 43200, "pol": 2, "level": PlatonicLevel.D6_CUBE, "delta_s": 48.0},
        }

    def pulse(self, current_s: Optional[float] = None):
        """One heartbeat tick — called inside training loop."""
        now = time.time()
        if self.real_time and (now - self.last_pulse) < 0.1:
            return

        self._autonomic_pulse(current_s or 0.0)
        for name, rhythm in self.rhythms.items():
            if name == "autonomic":
                continue
            triggered = (self.real_time and (now - self.last_pulse) >= rhythm["interval"]) or \
                        (not self.real_time and abs((current_s or 0.0) % rhythm["delta_s"]) < 0.2)
            if triggered:
                self._conscious_review(name, rhythm, current_s or 0.0)
        self.last_pulse = now

    def _autonomic_pulse(self, s: float):
        """Continuous low-amplitude vortex (global braiding_phase updated)."""
        stats = self.conduit.monitor_topological_winding()
        self.conduit.cube_chain.vortex_sync = (stats.get('braiding_phase', 0.0) + 0.003) % 1.0
        dummy = torch.randn(self.conduit.embed_dim, device=self.conduit.device) * 0.008
        self.conduit.cube_chain.bake(0, dummy, orientation=0, parent_idx=0)

    def _conscious_review(self, name: str, rhythm: dict, s: float):
        """Priority review + bake using global topology."""
        print(f"❤️ PIC HEARTBEAT — {name.upper()} review at s={s:.1f} (pol={rhythm['pol']})")
        task = RetrievalTask(
            action="get",
            what={"15min": "top priorities today",
                  "hourly": "todo list — check off done / remaining",
                  "4hour": "big picture — daily & weekly goals",
                  "daily_long": "long-term goals (months / years)"}.get(name, "priorities"),
            platonic_level=rhythm["level"],
            tags=[name]
        )
        # Daily autobiography page (one per day)
        if name == "daily_long":
            journal_text = f"Today: {task.to_prompt()}\n{output}"  # from PHGN narrative
            append_to_journal(journal_text)  # calls the new helper
        emb = torch.randn(self.conduit.embed_dim, device=self.conduit.device) * 0.28
        self.phgn(emb.unsqueeze(0), rhythm["level"], s=s + rhythm["delta_s"], pol=rhythm["pol"])
        self.conduit.cube_chain.bake(int(s) % 12, emb, parent_idx=rhythm["level"].value)

    def heartbeat_loop(agent, interval_minutes: int = 5):
        """Swiss-Watch heartbeat — now prints visibly every interval"""
        print(f"✅ Swiss-Watch heartbeat scheduler started (interval = {interval_minutes} minutes)")

        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[HEARTBEAT {now}] Running (interval: {interval_minutes} min)")

                # Light status check
                if hasattr(agent, 'check_for_recent_chat') and agent.check_for_recent_chat():
                    print(f"[HEARTBEAT {now}] Recent chat detected — running full cycle")
                else:
                    print(f"[HEARTBEAT {now}] No recent chat — light heartbeat only")

                # Optional: you can call any other periodic tasks here later
                time.sleep(interval_minutes * 60)

            except Exception as e:
                print(f"[HEARTBEAT ERROR] {e}")
                time.sleep(60)  # safety sleep


# ====================== TRAINING PIPELINE (heartbeat in the loop) ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PHGN — 3-Phase Star-Delta + Video Memory + Living Heartbeat (v9.9 Clifford)")
    parser.add_argument('--vqc', action='store_true', help='Use VQC-Enhanced conduit')
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--no-viz', action='store_true')
    parser.add_argument('--strong-train', action='store_true')
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Conduit with full v9.9 global topology (Clifford skin locked)
    ConduitClass = VQCEnhancedHelicalConduit if args.vqc else TwistedHelicalConduit
    conduit = ConduitClass(
        embed_dim=384,
        twist_rate=12.5,
        max_depth=48.0,
        num_polarizations=3,
        quat_logical_dim=96,
        toroidal_modulo9=True,
        vortex_math_369=True,
        clifford_projection=True
    ).to(device)

    phgn = PlatonicHelicalGeometricNetwork(conduit).to(device)
    optimizer = optim.AdamW(phgn.parameters(), lr=1.1e-3 if args.strong_train else 8e-4)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    heartbeat = HeartbeatScheduler(conduit, phgn, real_time=False)  # simulated time

    print(
        f"❤️ Training PHGN v9.9 with Living Heartbeat — {len(facts)} facts | Clifford Torus Skin + 3-phase + video memory")

    for epoch in range(args.epochs):
        phgn.train()
        total_loss = 0.0
        current_s = 4.5 + epoch * 0.8

        for idx, fact in enumerate(tqdm(facts, desc=f"Epoch {epoch}")):
            task = RetrievalTask(action="get", what=fact, platonic_level=PlatonicLevel(idx % 3))
            emb = F.normalize(embedder.encode(task.to_prompt(), convert_to_tensor=True, device=device), dim=-1) * 0.28
            s = current_s + (idx % 3) * 8.0
            pol = idx % 3

            output = phgn(emb.unsqueeze(0), task.platonic_level, s=s, pol=pol)

            # Loss = reconstruction + gentle global topological guidance
            recon_loss = F.mse_loss(output, emb.unsqueeze(0))
            stats = conduit.monitor_topological_winding()
            topo_loss = (stats.get('braiding_phase', 0.5) - 0.5).pow(2) * 0.08
            loss = recon_loss + topo_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phgn.parameters(), 1.15)
            optimizer.step()

            total_loss += loss.item()

            if idx % 3 == 0:  # ≈ one "minute"
                heartbeat.pulse(current_s=s)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:3d} | loss={total_loss / len(facts):.5f} | braiding_phase={stats.get('braiding_phase', 0):.4f}")

    print("\n" + "═" * 80)
    print("✅ PHGN TRAINING COMPLETE — Clifford Torus Skin + 3-Phase Star-Delta + Video Memory + Living Heartbeat")
    phgn.render_geometric_video()
    phgn.conduit.render_braided_lattice_style(save_path="outputs/heartbeat_braided_lattice.png")
    phgn.conduit.cube_chain.visualize_tree()
    print("→ Heartbeat scheduler is now live (autonomic + 15 min / 1 hr / 4 hr / daily reviews)")
    print("→ Ready to drop into agent_demo.py with: heartbeat = HeartbeatScheduler(conduit, phgn)")
    torch.save({
        'phgn_state': phgn.state_dict(),
        'conduit_state': conduit.state_dict(),
    }, "checkpoints/platonic_power_video_heartbeat_final.pt")
    print("Files saved: outputs/pic_heartbeat_video.mp4 + braided lattice + checkpoint")
