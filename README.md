# Quaternion Vortex Persistent Identity Conduit (QVPIC) v10.2

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Coverage](https://codecov.io/gh/kinaar8340/qvpic/branch/main/graph/badge.svg)

![Banner](images/qvpic.png)

**Geometric deep-learning memory architecture for drift-resistant persistent identity in AI agents**  
**Software embodiment of the Vortex Quaternion Conduit (VQC) patent**

**Current date context: April 2026**

## Abstract

The Quaternion Vortex Persistent Identity Conduit (QVPIC v10.2) is the software realization of the **Vortex Quaternion Conduit (VQC)** system described in U.S. Provisional Patent Application No. 63/913,110 (filed November 6, 2025) and the corresponding non-provisional application.

QVPIC encodes data as quaternion-compressed shards embedded in a helical fiber bundle over a Clifford-torus base. Global topological invariants — winding number, quaternion linking/braiding phase, and zero-point ShellCube radial differential — serve as the single source of truth for memory. Optional OAM modulation mirrors the patent’s nested helical shielding.

**New in v10.2**: A **Minimal Copresheaf Topological Neural Network (TNN)** layer performs higher-order sheaf diffusion reasoning directly on the RingConeChain combinatorial complex while keeping the underlying geometric identity lock completely frozen.

On standard benchmarks, QVPIC achieves **0.98–1.000 cosine recall fidelity** with **5.68× drift protection** relative to conventional vector-store baselines.

## Key Features (v10.2)

- **Minimal Copresheaf Topological Neural Network (TNN)**: Sheaf diffusion layer providing higher-order topological reasoning on the RingConeChain geometry (added April 2026).
- **Topological persistence layer**: Winding number, quaternion linking phase, and ShellCube radial differential (inscribed r=1 + circumscribed R=√3) enforce global consistency.
- **Dual-mode architecture**:
  - Default: `RubikConeConduit` + RingConeChain (216-cube hierarchical double-cone with message passing).
  - Experimental: `VQCEnhancedHelicalConduit` (continuous helical + configurable OAM flux).
- **Quaternion mathematics** throughout (qmul, qnormalize, Frenet–Serret spine).
- **Drift-resistant recall**: Hybrid dual-cone + ShellCube bonus using `safe_cosine(dim=-1)`.
- **Benchmarked fidelity**: 1.0000 average pure recall cosine, 5.68× protection factor.
- **Modular & production-ready**: SRP/DRY, configuration-driven, Torch 2.0 compiled.

## Relation to VQC Patent

QVPIC implements the patent’s core claims:
- Quaternion-compressed payload shards.
- Orthogonal OAM-mode encoding (ℓ modulation in VQCEnhanced mode).
- Nested helical phase shielding (Clifford-torus skin + braiding phase).
- Topological knot protections (winding / linking invariants).

The patent abstract and full specification are included in the repository as `docs/United_States_Non-Provisional_Patent_Application.pdf`.

## Quick Start: Quaternion Vortex Persistent Identity Conduit

1. Usage Options:

   i. Install from PyPI with:
   ```bash
   pip install qvpic
   ```

   ii. Clone the Repository:
   ```bash
   git clone https://github.com/kinaar8340/qvpic.git
   ```

   iii. Nightly Development Build updated every day at 2:00 AM UTC:
   ```bash
   pip install -i https://test.pypi.org/simple/ qvpic
   ```


2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```


3. Edit Identity Files in ~/qvpic/scripts/:
    
   --  agent_public.md 

   --  agent_private.md

   --  user_public.md

   --  user_private.md


4. Run the agent:
   ```bash
    # First run creates initial checkpoints.
    python scripts/agent_demo.py
    ```
   ```bash
    # Use --no-reset to save sessions.
    python scripts/agent_demo.py --no-reset
    ```


5. Experimental:
    ```bash
    # See docs/paper.md for details.
    python scripts/qvpic_test.py --strong-train --no-viz --vqc
    ```
   ```bash
   # Runs demo with vqc enhancement.
   python scripts/agent_demo.py --no-reset --verbose --vqc
   ```


6. Troubleshooting:
    ```bash
    # Runs full pipeline test.
    python scripts/qvpic_test.py --strong-train --no-viz
    ```
    ```bash
    # Runs all diagnostic scripts in tests/test_*.py.
    pytest -q --cov
    ```
   ```bash
   # Run demo with verbose output.
   python scripts/agent_demo.py --no-reset --verbose --heartbeat-minutes 15
   ```
      

7. Full agent reset, fresh start:
   ```bash
   # Removes old checkpoint + chat history
   rm -f checkpoints/pic_conduit_final.pt
   rm -f chat_history.json
   rm -rf snapshots/braided_lattice/*
   echo "✅ Agent's identity has been cleared."
   ```

## Benchmarks (RubikCone + ShellCube path)

| Metric                        | Value          | Notes |
|-------------------------------|----------------|-------|
| Average pure recall cosine    | 1.0000         | Immediate read-back after bake |
| Drift protection factor       | 5.68×          | vs. noisy vector baseline |
| Shell differential norm       | 1.0000         | Closed-system topological lock |
| Braiding phase (quaternion)   | ~0.82 (stable) | Toroidal window |
| Active cubes                  | 8+             | Discrete persistence layer |

## Architecture Overview

- **Continuous backbone**: `TwistedHelicalConduit` (Clifford-torus projection + quaternion Frenet spine).
- **Discrete layer**: `RingConeChain` (24→3 ring double-cone) + `ShellCube` radial differential.
- **Higher-order reasoning**: Minimal Copresheaf `TnnLayer` (sheaf diffusion on edge_index and ring polarities — the new 2026 cutting-edge addition).
- **Encoder/Decoder**: `RubikEncoder` / `RubikDecoder` with vortex-polarized message passing.
- **Read/Write**: `recover_depth` + `read` (soft-weighted, safe_cosine enforced) or RingCone recall.
- **Topological monitoring**: `monitor_topological_winding()` reports invariants at every step.

All cosine operations use the enforced pattern `safe_cosine(dim=-1 + .unsqueeze(0))`.

## Project Structure

```
qvpic/
├── models/                         # "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
├── src/
│   ├── conduit.py                  # Core TwistedHelicalConduit + RubikConeConduit
│   ├── vqc_enhanced_conduit.py     # OAM-modulated VQC subclass
│   ├── config.py
│   ├── encoder.py
│   └── decoder.py
├── scripts/
│   ├── agent_demo.py               # User Interface via Gradio
│   ├── qvpic_test.py               # Full Benchmark & Diagnostics
│   └── heartbeat.py                # Task Scheduler
├── identity/
│   ├── agent_public.md             # Agent's Public Identity
│   ├── agent_private.md            # Agent's Private Identity
│   ├── user_public.md              # User's Public Identity
│   └── user_private.md             # User's Private Identity
├── docs/
│   └── United_States_Non-Provisional_Patent_Application.pdf
├── configs/default.yaml
└── README.md
```

## License
MIT

## Acknowledgements

**Contact:** 
- kinaar0@protonmail.com
- X: @kinaar8340

**Repository:**
- https://github.com/kinaar8340/pic 
- https://github.com/kinaar8340/qvpic
- https://github.com/kinaar8340/vqc_sims_public


Built as the reference software implementation of the VQC patent.
Inspired by advances in topological photonics, geometric deep learning, 
quaternion neural networks, and sheaf/copresheaf topological neural networks.
