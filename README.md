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

1. Install & Setup:
    ```bash
    # clone the Repo
    sudo apt update
    sudo apt install git -y
    mkdir -p ~/Projects
    cd ~/Projects
    git clone https://github.com/kinaar8340/qvpic.git
    ```
    ```bash
    # Setup a Virtual Environment
    sudo apt update
    cd ~/Projects/qvpic
    python3 -m venv venv
    cd ~/Projects/qvpic
    source venv/bin/activate
    ```


2. Install Dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```


3. Set up your identity, do this before your first run.
    Save & Exit: Ctrl+O → Enter → Ctrl+X
    ```bash
    nano identity/user/upublic.md      # Public data about you
    nano identity/user/uprivate.md     # Private / sensitive data
    nano identity/user/ujournal.md     # Your personal journal (optional)
    ```
    Then compile, uploads to Your Agent and done.
    ```bash
    python scripts/setup_identity.py
    ```


4. Run the agent:
    ```bash
    # first run creates initial checkpoints
    python scripts/main.py
    ```
    ```bash
    # all future runs Persistent Sessions
    python scripts/main.py --no-reset
    ```
    Additional Options:
    ```
    --vqc                   # experimental
    --verbose               # expanded terminal readout
    --heartbeat-minutes 5   # sets automatic checkpoint (default=60)
    ```

    ```bash
    # experimental vqc
    python scripts/main.py --no-reset --vqc --verbose --no-viz --heartbeat-minutes 5
    ```


5. Troubleshooting:
    ```bash
    # runs full pipeline test.
    python scripts/qvpic_test.py --strong-train 
    ```
    ```bash
    # runs all diagnostic scripts in tests/test_*.py.
    pytest -q --cov
    ```
      

6. Full Agent Reset (if needed):
    ```bash
    # deletes agent's memory
    rm -f checkpoints/pic_conduit_final.pt
    rm -f chat_history.json
    rm -rf snapshots/braided_lattice/*
    ```

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
│
├── identity/
│   ├── user/                       # HUMAN
│   │   ├── upublic.md              # User edits this with "PUBLIC" data.
│   │   ├── uprivate.md             # User edits this with "PRIVATE" data.
│   │   └── ujournal.md             # User's journal as long-term record.         
│   └── agent/                      # AI
│       ├── apublic.md              # Agent can modify these (with guardrails)
│       ├── aprivate.md             # Agent can modify these (with guardrails)
│       └── ajournal.md             # Agent's journal as long-term memory.
│
├── facts/                          # JSON – structured & appendable
│   ├── public_facts.json           # "PUBLIC" runtime facts 
│   └── private_facts.json          # "PRIVATE" runtime facts 
│
├── scripts/                        
│   ├── setup_identity.py           # One-time compiler: .md → JSON
│   ├── main.py                     # Executable
│   ├── agent.py                    # Agent's Guardrails
│   ├── ui.py                       # User Interface via Gradio
│   ├── heartbeat.py                # Task Scheduler
│   └── qvpic_test.py               # Full Benchmark & Diagnostics
│
├── src/                            
│   ├── conduit.py                  # Core TwistedHelicalConduit + RubikConeConduit
│   ├── vqc_enhanced_conduit.py     # OAM-modulated VQC subclass
│   └── config.py                   
│
├── tests/                          # Runs all diagnostic scripts
│   └── test_conduit.py             
│
├── pyproject.toml                  
├── configs/                        
│   └── default.yaml                
│
├── checkpoints/                    
├── logs/                           
├── outputs/                        
├── images/                         
├── requirements.txt                
├── README.md                       
└── docs/
    ├── non_technical_QVPIC_Whitepaper.md
    ├── QVPIC_Whitepaper.md
    └── VQC_NonProvisional_Patent_Application.md

```

## License
MIT

## Acknowledgements

**Contact:** 
- kinaar0@protonmail.com
- X: @kinaar8340


Built as the reference software implementation of the VQC patent.
Inspired by advances in topological photonics, geometric deep learning, 
quaternion neural networks, and sheaf/copresheaf topological neural networks.
