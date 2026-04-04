# Changelog

All notable changes to **qvpic** will be documented in this file.

## [10.2.0] - 2026-04-04

**Production Release — RubikCone + ShellCube Topology System**

### ✨ Highlights
- **RubikConeConduit** is now the default production path
- **ShellCube radial differential** (inscribed r=1 + circumscribed R=√3) fully integrated
- **RingConeChain** + **CopresheafDiffusionStack** (3-layer polarity-aware TNN) wired and stable
- Persistent topological memory with global invariants:
  - Geometric winding, effective winding, toroidal winding
  - Braiding phase, knot phase (3-6-9), Clifford torus skin
  - Vortex sync and ShellCube zero-point differential

### 🚀 New Features
- Full Rubik’s cube state encoding (54-sticker face grids + 24-orientation + vortex digits)
- Automatic topological monitoring (winding numbers, braiding_phase, drift resistance)
- Production-ready CLI: `qvpic-test --strong-train`

### 🔧 Improvements
- Complete device safety (CPU + CUDA) — tests now pass reliably in GitHub CI
- Ruff linting and code style fully cleaned (`--fix` applied)
- `pyproject.toml` polished for proper package distribution
- pytest collection fixed (scripts/ no longer scanned)

### 🛠️ Infrastructure
- Automated GitHub Releases workflow (`release.yml`)
- Trusted Publishing to PyPI (passwordless, secure)
- Full CI matrix with CPU-only Torch for reliable testing

### 📦 Package
```bash
pip install qvpic
qvpic-test --strong-train
