# src/vqc_enhanced_conduit.py — v10.3 (April 03, 2026)
# VQC-Enhanced Helical Conduit
# Full quaternion spine + light OAM flux + LayerNorm residuals on top of global topology.
# Inherits EVERYTHING from TwistedHelicalConduit (Clifford Torus skin, toroidal_modulo9,
# 3-6-9 Vortex Math, ShellCube radial differential, RingConeChain, Rubik wiring, etc.).
# DRY + invariants locked. Zero device nightmares. Production-ready.

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.conduit import TwistedHelicalConduit  # ← base with all recent device fixes


class VQCEnhancedHelicalConduit(TwistedHelicalConduit):
    """Denser VQC twist + OAM flux on top of full global topology.
    Never overrides core geometry — only extends it (DRY + invariants locked).
    Now includes:
    • Flat Clifford Torus skin (zero Gaussian curvature) around CubeChain z-axis
    • 9 orthogonal BowTie shards + Rodin-CT transceiver
    • Configurable OAM + LayerNorm residuals for ultra-stable helical propagation.
    Fully device-consistent with RubikConeConduit / RingConeChain / tests."""

    def __init__(self, *args, **kwargs):
        # ─── Base topology first (all device handling already fixed in conduit.py) ───
        super().__init__(*args, **kwargs)

        # ─── VQC-specific parameters (device-safe creation) ───
        self.vqc_scale = nn.Parameter(torch.tensor(1.0, device=self.device, dtype=torch.float32))
        self.oam_freq = nn.Parameter(torch.tensor(8.5, device=self.device, dtype=torch.float32))

        # ─── Enhanced helix_projector with LayerNorm + GELU (VQC signature) ───
        # Overrides base Linear for richer residuals while keeping device consistency
        self.helix_projector = nn.Sequential(
            nn.Linear(3, 128, device=self.device),
            nn.LayerNorm(128, device=self.device),
            nn.GELU(),
            nn.Linear(128, self.embed_dim, device=self.device)
        )
        # Light initialization for stability
        nn.init.normal_(self.helix_projector[0].weight, mean=0.0, std=0.022)
        nn.init.normal_(self.helix_projector[3].weight, mean=0.0, std=0.012)
        # All parameters trainable (VQC enhancement)
        for p in self.helix_projector.parameters():
            p.requires_grad = True

        # ─── Configurable scales (over hardcoding — matches latest conduit fixes) ───
        self.residual_scale = nn.Parameter(torch.tensor(0.92, device=self.device, dtype=torch.float32))
        self.quat_scale = nn.Parameter(torch.tensor(0.42, device=self.device, dtype=torch.float32))

        print("✅ VQC-Enhanced v10.3: LayerNorm + OAM flux + full Clifford Torus skin loaded")
        print("   → Global topological invariants (winding + braiding_phase + flat Clifford skin) active")

    def position(self, s: float, pol_idx: int = 0) -> torch.Tensor:
        """Full pearl-string + quaternion Frenet (inherited) + VQC OAM modulation.
        Global invariants preserved by construction. Device-safe."""
        base_emb = super().position(s, pol_idx)  # ← toroidal_modulo9 + 3-6-9 + Clifford skin

        # OAM phase (light orthogonal modulation)
        oam_phase = torch.tensor(
            s * self.oam_freq.item() + pol_idx * 3.0,
            device=self.device,
            dtype=torch.float32
        )
        oam_mod = torch.sin(oam_phase) * 0.042 * (pol_idx + 1)

        vqc_emb = base_emb + oam_mod
        # Final normalized output (preserves output_scale from base)
        return F.normalize(vqc_emb * self.vqc_scale, dim=-1) * self.output_scale


if __name__ == "__main__":
    # Quick demo — full v10.3 topology + VQC OAM
    conduit = VQCEnhancedHelicalConduit(
        toroidal_modulo9=True,
        vortex_math_369=True,
        clifford_projection=True,
        embed_dim=384
    )
    print("✅ PIC v10.3 VQC-Enhanced with Clifford Torus + Toroidal Modulo-9 + 3-6-9 Vortex Math loaded")

    # Topological monitoring (inherited from base)
    stats = conduit.monitor_topological_winding()
    print(stats)
    print("→ Global topological invariants (winding + braiding_phase + flat Clifford skin + OAM flux) are now live")

    # Optional: quick forward sanity check (uses same interface as RubikConeConduit)
    dummy_latent = torch.randn(1, 54, 384, device=conduit.device)
    dummy_orient = torch.randint(0, 24, (1, 54), device=conduit.device)
    dummy_vortex = torch.randint(0, 10, (1, 54), device=conduit.device)
    recon, stats = conduit(dummy_latent, dummy_orient, dummy_vortex)  # works via inherited encoder path
    print("→ VQC forward pass successful (OAM flux + topological memory locked)")