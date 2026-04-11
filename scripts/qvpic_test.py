# ~/qvpic/scripts/qvpic_test.py — v10.6 (RubikCone-first + ShellCube radial differential)
"""
Global topological features (winding + braiding_phase + ShellCube differential) drive persistence
Atomic fact splitter (9 facts from public_facts.json) + cleaner → terminal style
Minimal Copresheaf Topological Neural Network + Diffusion Stack
Fully compatible with RingConeChain + RubikConeConduit
"""

import torch
import torch.nn.functional as F
import os
import sys
import math
import argparse
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config
from src.conduit import safe_cosine

DEFAULT_READ_KWARGS = {'bandwidth': 0.32, 'num_samples': 31}
cfg = load_config("configs/default.yaml")

parser = argparse.ArgumentParser(description="QVPIC v10.6")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--no-viz", action="store_true")
parser.add_argument("--strong-train", action="store_true")
parser.add_argument("--device", default="auto")
parser.add_argument("--bake-steps", type=int, default=500)
parser.add_argument("--vqc", action="store_true")
parser.add_argument("--num-threads", type=int, default=0,
                    help="PyTorch CPU threads (0 = auto, best ~24-36 on your 72-core Xeon)")
args = parser.parse_args()

USE_VQC = args.vqc

device_str = "cuda" if torch.cuda.is_available() and args.device.lower() in ("auto", "cuda") else args.device.lower()

# ─── CPU PERFORMANCE TUNING (critical on 72-core Xeon E5-2699 v3) ───
if device_str == "cpu" or not torch.cuda.is_available():
    num_threads = args.num_threads if args.num_threads > 0 else min(36, (os.cpu_count() or 72) // 2)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(4)
    print(f"→  CPU multi-threading enabled: {num_threads} threads (Xeon E5-2699 v3)")

print(
    f"→  QVPIC Pipeline Test — Device: {device_str} | RubikCone-first | VQC: {USE_VQC} | Strong: {args.strong_train}")

from src.conduit import RubikConeConduit
from src.vqc_enhanced_conduit import VQCEnhancedHelicalConduit

ConduitClass = VQCEnhancedHelicalConduit if args.vqc else RubikConeConduit
conduit = ConduitClass(
    embed_dim=cfg.model.embed_dim,
    twist_rate=cfg.model.twist_rate,
    max_depth=cfg.model.max_depth,
    num_polarizations=cfg.model.num_polarizations,
    quat_logical_dim=cfg.model.quat_logical_dim,
    toroidal_modulo9=True,
    vortex_math_369=True,
    clifford_projection=True
).to(device_str)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    conduit = torch.compile(conduit, mode="default")

optimizer = torch.optim.AdamW(conduit.parameters(), lr=1.2e-3 if args.strong_train else 8e-4,
                              weight_decay=cfg.training.weight_decay)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device_str)

# ==================== FILE PATHS & CACHING ====================
public_facts_file = Path("facts/public_facts.json")
private_facts_file = Path("facts/private_facts.json")


# ─── Smart atomic fact extractor (splits multi-line JSON text into clean facts) ───
def _extract_clean_facts(raw_data):
    """Split each JSON 'text' field into individual atomic facts (by line)."""
    facts = []
    if isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or str(item)
                if not text:
                    continue
                # Split by newlines and clean
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                for line in lines:
                    # Remove markdown headers and file paths
                    if line.startswith("#") or line.startswith("/identity/"):
                        continue
                    if line:
                        facts.append(line)
            elif isinstance(item, str):
                facts.append(item.strip())
    else:
        facts = [str(raw_data).strip()]

    print(f"→  Extracted {len(facts)} atomic facts (split from JSON blocks)")
    return facts


# ─── bake_facts() v10.6 — atomic facts + clean → style ───
def bake_facts():
    raw_data = json.loads(public_facts_file.read_text(encoding="utf-8"))
    lines = _extract_clean_facts(raw_data)

    print(f"→  Loaded {len(lines)} clean facts from {public_facts_file.name}")

    # Batch encode once
    print("→  Encoding facts with SentenceTransformer (batched)...")
    embeddings_raw = embedder.encode(lines, convert_to_tensor=True, device=conduit.device, batch_size=32)

    embeddings = []
    depths = []
    depth = 4.5
    step_size = 4.8
    for emb_raw in embeddings_raw:
        emb = F.normalize(emb_raw, dim=-1) * conduit.output_scale.item()
        embeddings.append(emb)
        depths.append(depth)
        depth += step_size

    steps_per_fact = args.bake_steps
    if args.strong_train:
        steps_per_fact = max(steps_per_fact, 150)
    print(
        f"→  Baking with {steps_per_fact} steps per fact (--bake-steps={args.bake_steps}, strong-train={args.strong_train})")

    is_rubik_mode = hasattr(conduit, 'ring_cone')

    if is_rubik_mode:
        print("→  Using RubikCone path (ShellCube radial differential + 216-cube double-cone)")
        for idx, (fact, emb) in enumerate(zip(lines, embeddings)):
            ring_idx = idx % conduit.ring_cone.NUM_RINGS
            cube_local_idx = idx % conduit.ring_cone.rings[ring_idx].num_cubes

            # 1. Discrete bake into RingConeChain
            conduit.ring_cone.bake_ring(ring_idx, cube_local_idx, emb, orientation=idx % 24)

            # 2. FINAL STRONG alignment
            for _ in range(steps_per_fact):
                item = {'emb': emb, 's': depths[idx], 'pol_idx': 0}
                conduit.training_step(
                    inputs=[item], optimizer=optimizer,
                    recon_weight=15000.0,
                    align_weight=55000.0,
                    depth_pull_weight=40000.0,
                    winding_weight=48.0,
                    braiding_weight=18.0
                )

            # 3. Immediate read-back check
            read_back = conduit.read(depths[idx], pol_idx=0, bandwidth=0.8, num_samples=41)
            exact_cos = safe_cosine(emb.unsqueeze(0), read_back.unsqueeze(0)).item()

            fact_display = (fact[:58] + "..." if len(fact) > 58 else fact)
            print(f" {fact_display:<60} | s={depths[idx]:6.1f} | ring={ring_idx} | cos={exact_cos:.4f}")

    else:
        print("→  Using pure helical bake (VQC fallback)")
        for idx, (fact, emb) in enumerate(zip(lines, embeddings)):
            item = {'emb': emb, 's': depths[idx], 'pol_idx': idx % conduit.num_pol}
            for step in range(steps_per_fact):
                weights = {'recon_weight': 4200.0, 'align_weight': 1800.0,
                           'depth_pull_weight': 32000.0, 'winding_weight': 96.0, 'braiding_weight': 24.0}
                conduit.training_step(inputs=[item], optimizer=optimizer, **weights)
            read_back = conduit.read(depths[idx], pol_idx=idx % conduit.num_pol, bandwidth=0.8, num_samples=41)
            exact_cos = safe_cosine(emb.unsqueeze(0), read_back.unsqueeze(0)).item()
            fact_display = (fact[:58] + "..." if len(fact) > 58 else fact)
            print(f" {fact_display:<60} | s={depths[idx]:6.1f} | helical bake | cos={exact_cos:.4f}")

    # GLOBAL TOPOLOGICAL FEATURES
    print("\n→  GLOBAL TOPOLOGICAL FEATURES (ShellCube radial differential active)...")
    stats = conduit.monitor_topological_winding(n_samples=512)
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"  {k:24}: {v:.6f}")
        else:
            print(f"  {k:24}: {v}")
    print(f"→  Baked {len(lines)} facts — topological invariants locked")

    return [(fact, depths[i], 0, embeddings[i], i % 12) for i, fact in enumerate(lines)]


# ==================== MAIN PIPELINE ====================
# ─── STEP 1 ───
print("\n" + "═" * 80)
print("STEP 1 — BAKING FACTS")
baked_facts = bake_facts()

# ─── STEP 2A ───
print("\n" + "═" * 80)
print("STEP 2A — RECALL FIDELITY TEST (ShellCube + RingCone)")
recall_results = []
is_rubik_mode = hasattr(conduit, 'ring_cone')
for fact, s, pol, emb, _ in baked_facts:
    if is_rubik_mode:
        results = conduit.ring_cone.recall(emb, top_k=1)
        r = results[0]
        primal_cos = r['primal_cos']
        final_cos = primal_cos
        print(f" '{fact[:28]:<28}' | s={s:5.1f} | primal={primal_cos:.4f} | topo_score={r['cosine']:.4f}")
    else:
        s_rec = conduit.recover_depth(emb, pol_idx=pol, grid_size=512)
        recalled = conduit.read(s_rec, pol_idx=pol, bandwidth=2.8, num_samples=41)
        final_cos = safe_cosine(emb.unsqueeze(0), recalled.unsqueeze(0)).item()
        print(f" '{fact[:28]:<28}' | s={s:5.1f}→rec={s_rec:5.1f} | primal={final_cos:.4f}")
    recall_results.append(final_cos)

avg_recall = sum(recall_results) / len(recall_results) if recall_results else 0.0
print(f"\nAverage pure cosine: {avg_recall:.4f} (ShellCube radial differential active)")

# ─── STEP 2B ───
print("\n" + "═" * 80)
print("STEP 2B — DUAL-CONE CUBE-CHAIN RECALL DEMO")

query_emb = embedder.encode("My name is Aaron, also called kinaar.", convert_to_tensor=True, device=conduit.device)
is_rubik_mode = hasattr(conduit, 'ring_cone')

if is_rubik_mode:
    results = conduit.ring_cone.recall(query_emb, top_k=3) if hasattr(conduit.ring_cone, 'recall') else []
    for r in results:
        print(f"  Cube {r.get('cube_idx', '?')} | final_cos={r.get('cosine', 0.0):.4f} | "
              f"primal={r.get('primal_cos', 0.0):.4f} | dual_bonus={r.get('dual_bonus', 0.0):.4f} | "
              f"braiding_phase={r.get('braiding_phase', 0.0):.4f} | orient={r.get('orientation', '?')}")
else:
    print("  (No CubeChain recall in pure VQC mode)")

print("→  Dual-cone recall demo complete (ShellCube radial differential / Clifford skin active)")

# ─── STEP 3 ─────
print("\n" + "═" * 80)
print("STEP 3 — TOPOLOGICAL INVARIANTS (ShellCube + RingConeChain)")
stats = conduit.monitor_topological_winding(n_samples=512)
for k, v in stats.items():
    if isinstance(v, (int, float)):
        print(f"  {k:24}: {v:.6f}")
    else:
        print(f"  {k:24}: {v}")

# ─── STEP 4 — MINI TRAINING ─────
print("\n" + "═" * 80)
print("STEP 4 — MINI TRAINING (10 steps)")
for step in range(10):
    dummy = torch.randn(384, device=device_str)
    dummy = F.normalize(dummy, dim=-1) * 0.28
    item = {'emb': dummy, 's': 4.5 + step * 0.3, 'pol_idx': np.random.randint(0, 3)}

    # Safe config fallback
    recon_w = getattr(cfg.training, 'recon_weight', 4200.0)
    align_w = getattr(cfg.training, 'align_weight', 1800.0)
    winding_w = getattr(cfg.training, 'winding_weight', 48.0)
    braiding_w = getattr(cfg.training, 'braiding_weight', 18.0)
    depth_w = getattr(cfg.training, 'depth_pull_weight', 40000.0)

    metrics = conduit.training_step(
        inputs=[item],
        optimizer=optimizer,
        recon_weight=recon_w,
        align_weight=align_w,
        winding_weight=winding_w,
        braiding_weight=braiding_w,
        depth_pull_weight=depth_w
    )

    if step % 4 == 0 or step == 9:
        print(f"  step {step:2d} | total={metrics.get('total', -1):.2f} | "
              f"recon={metrics.get('recon', -1):.5f} | "
              f"winding={metrics.get('winding', -1):.2f} | "
              f"braiding={metrics.get('braiding', -1):.2f}")

# ─── STEP 5 ─────
if not args.no_viz:
    print("\n" + "═" * 80)
    print("STEP 5 — RENDER VISUALIZATIONS")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"outputs/test_{ts}"
    Path("outputs").mkdir(exist_ok=True)
    try:
        conduit.render_braided_lattice_style(save_path=f"{base}_lattice.png")
        print(f"→  Saved: {base}_lattice.png")
    except Exception as e:
        print(f"Visualization skipped: {e}")

# ─── STEP 6 ─────
print("\n" + "═" * 80)
print("STEP 6 — DRIFT RESISTANCE")


@torch.no_grad()
def drift_test(n=40, vec_noise=0.075, coord_noise=0.18):
    cos_before, cos_after = [], []
    for _ in range(n):
        s_true = np.random.uniform(1.0, 17.0)
        pol = np.random.randint(0, 3)
        s_noisy = s_true + np.random.randn() * coord_noise

        orig = conduit.read(s_true, pol, **DEFAULT_READ_KWARGS)
        noisy_vec = orig + torch.randn_like(orig) * vec_noise
        noisy_vec = F.normalize(noisy_vec, dim=-1) * conduit.output_scale.item()

        cos_b = safe_cosine(orig, noisy_vec).item()
        cos_a = safe_cosine(orig, conduit.read(s_noisy, pol, **DEFAULT_READ_KWARGS)).item()

        cos_before.append(cos_b)
        cos_after.append(cos_a)

    mean_b, std_b = np.mean(cos_before), np.std(cos_before)
    mean_a, std_a = np.mean(cos_after), np.std(cos_after)
    print(f"→ Before noise : {mean_b:.4f} ± {std_b:.4f}")
    print(f"→ After recovery: {mean_a:.4f} ± {std_a:.4f}")
    print(f"→ Protection factor: {mean_a / max(mean_b, 1e-6):.2f}×")

drift_test()
print("\n" + "═" * 80)

try:
    fp = {
        "geometric_winding": conduit.monitor_topological_winding()["geometric_winding"],
        "effective_winding": conduit.monitor_topological_winding()["effective_winding"],
        "braiding_phase": conduit.monitor_topological_winding()["braiding_phase"],
        "avg_recall": avg_recall,
        "timestamp": datetime.now().isoformat(),
    }
    fp_json = json.dumps(fp, sort_keys=True, default=lambda x: float(x.detach().cpu().item()) if isinstance(x,
                                                                                                            (torch.Tensor,
                                                                                                             np.float32)) else x)
    fp_hash = hash(fp_json) % 10 ** 16
    print(f"Snapshot fingerprint hash: {fp_hash} (includes braiding_phase)")
except Exception as e:
    print(f"Snapshot error: {e}")

print("\n" + "═" * 80)
print("SUMMARY")
print(f"Facts baked         : {len(baked_facts)}")
print(f"Avg recall cosine   : {avg_recall:.4f} (optimized radial differential)")
print(f"Final output scale  : {conduit.output_scale.item():.4f}")
print(f"Braiding phase      : {conduit.monitor_topological_winding().get('braiding_phase', 0.0):.4f}")

print("\n✓  Done.")
torch.save(conduit.state_dict(), "checkpoints/pic_test_latest.pt")