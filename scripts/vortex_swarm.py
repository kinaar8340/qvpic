# ~/qvpic/scripts/vortex_swarm.py — v10.9 "Bud's Crew" Vortex Identity Swarm
"""
Expanded grid + full hyperparameter sweep (lr + recon_weight + more layers/pols/facts)
Fully distributed across 9-node R630 cluster.
"""
import ray
import torch
import torch.nn.functional as F
import pandas as pd
import json
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ray.init(address="auto", ignore_reinit_error=True)

print("🌟 Magic Island Sweep v1.1 — Flux Flywheel Resonance Hunter")
print(f"→ Connected nodes: {len(ray.nodes())}")
print(f"→ Total CPUs available: {ray.cluster_resources().get('CPU', 0)}")
if len(ray.nodes()) == 1:
    print("⚠️ Only 1 node detected → single-node mode on bud (RTX 4090)")

from src.config import load_config
from src.conduit import RubikConeConduit, CopresheafDiffusionStack

cfg = load_config("configs/default.yaml")
public_facts_file = Path("facts/public_facts.json")

# ==================== QUATERNION HELPERS ====================
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else q

def small_rotor(theta, axis=np.array([0., 0., 1.])):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    half = theta / 2
    return np.array([np.cos(half), *(np.sin(half) * axis)])


@ray.remote(num_cpus=12, num_gpus=0, max_retries=2, scheduling_strategy="SPREAD")
def run_qvpic_trial(trial_id: int, params: Dict):
    print(f"→ Trial {trial_id} running | layers={params['num_layers']} | pol={params['num_polarities']} | "
          f"coop={params['cooperative_sheaf']} | facts={params['max_facts']} | lr={params['lr']:.2e} | recon_w={params['recon_weight']}")

    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" → Using device: {device}")

    conduit = RubikConeConduit(
        embed_dim=cfg.model.embed_dim,
        twist_rate=cfg.model.twist_rate,
        max_depth=cfg.model.max_depth,
        num_polarizations=cfg.model.num_polarizations,
        quat_logical_dim=cfg.model.quat_logical_dim,
        toroidal_modulo9=True,
        vortex_math_369=True,
        clifford_projection=True
    ).to(device)

    # === GAUGED HOPF / TWO-GYRO UPGRADE ===
    use_gauged = params.get("use_gauged_hopf", False)
    print(f" → Gauged Hopf mode: {use_gauged}")

    # Rebuild stack with trial params
    ring_cone = conduit.ring_cone
    new_stack = CopresheafDiffusionStack(
        in_channels=ring_cone.embed_dim,
        hidden_channels=ring_cone.embed_dim,
        out_channels=ring_cone.embed_dim,
        num_layers=params["num_layers"],
        num_polarities=params["num_polarities"],
        dropout=0.05,
        sheaf_mode=False,
        use_cooperative_sheaf=params["cooperative_sheaf"],
        device=device
    )
    new_stack.prepare(ring_cone.edge_index, ring_cone.ring_polarities)
    new_stack = new_stack.to(device)
    ring_cone.tnn_stack = new_stack

    # === DIAGNOSTICS (unchanged) ===
    print(f" → Rebuilt CopresheafDiffusionStack (layers={params['num_layers']}, pol={params['num_polarities']}, coop={params['cooperative_sheaf']})")

    # === LOAD FACTS ===
    raw_data = json.loads(public_facts_file.read_text(encoding="utf-8"))
    lines = [line.strip() for item in raw_data if isinstance(item, dict)
             for line in (item.get("text") or str(item)).splitlines()
             if line.strip() and not line.startswith(("#", "/identity/"))]

    # === OPTIMIZER — NOW USES TRIAL-SPECIFIC LR ===
    optimizer = torch.optim.AdamW(
        conduit.parameters(),
        lr=params["lr"],
        weight_decay=1e-4
    )

    print(f" → Strong bake: {params['max_facts']} facts × 100 steps | lr={params['lr']:.2e} | recon_w={params['recon_weight']:.0f} on {device}...")

    for idx, fact in enumerate(lines[:params["max_facts"]]):
        emb = F.normalize(torch.randn(384, device=device), dim=-1) * 0.28
        ring_idx = idx % ring_cone.NUM_RINGS
        cube_local_idx = idx % ring_cone.rings[ring_idx].num_cubes
        ring_cone.bake_ring(ring_idx, cube_local_idx, emb, orientation=idx % 24)

        for step in range(100):
            item = {'emb': emb.unsqueeze(0), 's': torch.tensor([4.5 + idx * 4.8], device=device), 'pol_idx': 0}
            try:
                loss_dict = conduit.training_step(
                    inputs=[item],
                    optimizer=optimizer,
                    recon_weight=params["recon_weight"],
                    align_weight=55000.0,
                    depth_pull_weight=40000.0,
                    winding_weight=48.0,
                    braiding_weight=18.0
                )
            except Exception as e:
                print(f"   ⚠️ training_step skipped (step {step})")
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss.backward()
                optimizer.step()

        if (idx + 1) % 3 == 0:
            print(f"   → Fact {idx+1}/{params['max_facts']} baked")

        # === TWO-GYRO GAUGED HOPF UPGRADE (analytical-scale pointer) ===
        if use_gauged:
            # Create missing attributes on first use (safe fallback)
            if not hasattr(ring_cone, 'current_quaternion'):
                ring_cone.current_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # identity
            if not hasattr(ring_cone, 'twist_history'):
                ring_cone.twist_history = np.zeros(1)

            delta_L = small_rotor(0.025)
            delta_R = small_rotor(0.023)

            q_temp = q_mult(delta_L, ring_cone.current_quaternion)
            ring_cone.current_quaternion = q_mult(q_temp, q_conj(delta_R))
            ring_cone.current_quaternion = q_normalize(ring_cone.current_quaternion)

            # Gauge connection (the pointer on the analytical scale)
            avg_imbalance = np.mean(ring_cone.twist_history) % (2 * np.pi)
            gauge_alpha = -0.85 * avg_imbalance
            gauge_rot = np.array([np.cos(gauge_alpha), 0., 0., np.sin(gauge_alpha)])

            ring_cone.current_quaternion = q_mult(ring_cone.current_quaternion, gauge_rot)
            ring_cone.current_quaternion = q_normalize(ring_cone.current_quaternion)

            # Store for future steps and monitoring
            ring_cone.twist_history = np.append(ring_cone.twist_history,
                                                2 * np.arccos(np.clip(ring_cone.current_quaternion[0], -1.0, 1.0)))
            if not hasattr(ring_cone, 'gauge_alpha_history'):
                ring_cone.gauge_alpha_history = []
            ring_cone.gauge_alpha_history.append(gauge_alpha)

    print(f" → Trial {trial_id} bake complete")

    stats = conduit.monitor_topological_winding(n_samples=512)
    print(f" → Monitor → braiding_phase={stats.get('braiding_phase', 0):.5f} | active_cubes={stats.get('active_cubes', 0)}")

    return {
        "trial_id": trial_id,
        "num_layers": params["num_layers"],
        "num_polarities": params["num_polarities"],
        "cooperative_sheaf": params["cooperative_sheaf"],
        "max_facts": params["max_facts"],
        "lr": params["lr"],
        "recon_weight": params["recon_weight"],
        "braiding_phase": float(stats.get("braiding_phase", 0.0)),
        "geometric_winding": float(stats.get("geometric_winding", 0.0)),
        "active_cubes": int(stats.get("active_cubes", 0)),
        "use_gauged_hopf": use_gauged,
        "timestamp": datetime.now().isoformat()
    }

# ==================== LAUNCH ====================
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=12)
parser.add_argument("--max-facts", type=int, default=9)
args = parser.parse_args()

# ==================== EXPANDED GRID WITH GAUGE ====================
param_grid = [
    {"num_layers": nl, "num_polarities": np, "cooperative_sheaf": cs,
     "max_facts": mf, "lr": lr_val, "recon_weight": rw, "use_gauged_hopf": gh}
    for nl in [2, 3, 4, 5, 6]
    for np in [9, 12, 18, 24, 36]
    for cs in [True, False]
    for mf in [9, 18, 27, 36]
    for lr_val in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    for rw in [8000.0, 10000.0, 15000.0, 20000.0, 25000.0]
    for gh in [True, False]                     # ← new two-gyro flag
][:args.trials]

print(f"→ Launching {len(param_grid)} parallel trials (full grid = 8,640 combos)...")

futures = [run_qvpic_trial.remote(i, p) for i, p in enumerate(param_grid)]
results = ray.get(futures)

df = pd.DataFrame(results)
report_path = Path(f"outputs/swarm_report_{datetime.now():%Y%m%d_%H%M%S}.md")
df.to_markdown(report_path, index=False)

print(f"\n→ Swarm complete! Report saved → {report_path}")
print(df[["trial_id", "num_layers", "num_polarities", "cooperative_sheaf", "max_facts", "lr", "recon_weight", "braiding_phase", "active_cubes"]].head(20))

ray.shutdown()
print("→ Done.")