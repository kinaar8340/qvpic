# Quaternion Vortex Persistent Identity Conduit (QVPIC) v1.0

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

**Geometric deep-learning memory architecture for drift-resistant persistent identity in AI agents**  
**Software embodiment of the Vortex Quaternion Conduit (VQC) patent**

**Current date context: April 2026**

## Abstract

The Quaternion Vortex Persistent Identity Conduit (QVPIC v1.0) is the software realization of the **Vortex Quaternion Conduit (VQC)** system described in U.S. Provisional Patent Application No. 63/913,110 (filed November 6, 2025) and the corresponding non-provisional application titled *“Vortex Quaternion Conduit (VQC): Ultra-High-Density Quantum-Secure Data Transmission via Orbital Angular Momentum Mode Division Multiplexing with Quaternion Compression and Nested Helical Shielding”*.

QVPIC encodes data as quaternion-compressed shards embedded in a **helical fiber bundle over a Clifford-torus base**. Global topological invariants — winding number, quaternion linking/braiding phase, and zero-point ShellCube radial differential — serve as the single source of truth for memory. Optional OAM modulation (via the `VQCEnhancedHelicalConduit`) directly mirrors the patent’s nested helical shielding and mode-division multiplexing.  

On standard benchmarks, QVPIC achieves **0.98–1.000 cosine recall fidelity** with **5.68× drift protection** relative to conventional vector-store baselines, while preserving exact topological invariants under coordinate and vector noise. This provides a practical, reproducible software prototype of VQC’s ultra-high-density, quantum-secure encoding principles for AI agents.

## Key Features (v1.0)

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

## Quick Start

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Place facts in `scripts/public_facts.txt` and `scripts/private_facts.txt`.
4. Run the agent:
   ```bash
   python scripts/agent_demo.py --no-reset
   ```
5. Experimental VQC OAM mode:
   ```bash
   python scripts/pic_test.py --vqc
   ```

Full diagnostics and baking:
```bash
python scripts/pic_test.py --strong-train --no-viz
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
- **Encoder/Decoder**: `RubikEncoder` / `RubikDecoder` with vortex-polarized message passing.
- **Read/Write**: `recover_depth` + `read` (soft-weighted, safe_cosine enforced) or RingCone recall.
- **Topological monitoring**: `monitor_topological_winding()` reports invariants at every step.

All cosine operations use the enforced pattern `safe_cosine(dim=-1 + .unsqueeze(0))`.

## Project Structure

```
qvpic/
├── models/
├── src/
│   ├── conduit.py                  # Core TwistedHelicalConduit + RubikConeConduit
│   ├── vqc_enhanced_conduit.py     # OAM-modulated VQC subclass
│   ├── config.py
│   ├── encoder.py
│   └── decoder.py
├── scripts/
│   ├── agent_demo.py               # User Interface via Gradio
│   ├── qvpic_test.py               # Full benchmark & diagnostics
│   └── heartbeat.py
├── identity/
│   ├── agent_bud.md                # Agent's Identity
│   ├── user_public.md              # User's Public Identity
│   └── user_private.md             # User's Private Identity
├── docs/
│   └── United_States_Non-Provisional_Patent_Application.pdf
├── configs/default.yaml
└── README.md
```

## License

MIT

## Acknowledgments

Built as the reference software implementation of the VQC patent by Aaron Michael Kinder.  
Inspired by advances in topological photonics, geometric deep learning, and quaternion neural networks.  

**Global topological invariants (winding, linking, braiding + ShellCube differential) are the single source of truth.**

