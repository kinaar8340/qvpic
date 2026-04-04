# src/config.py — v9.9 (March 26, 2026)
# Fully config-driven. Global topological features (winding, linking, braiding phases
# + toroidal Clifford skin) remain primary. JSON fallback is market redundancy only.
# safe_cosine(dim=-1 + .unsqueeze(0)) pattern enforced everywhere in dependent modules.

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    # Strong local fidelity (these must dominate)
    recon_weight: float = 5200.0
    mag_weight: float = 80.0
    align_weight: float = 1200.0

    # Very gentle global topological guidance
    winding_weight: float = 128.0      # soft push toward geometric helix
    locality_weight: float = 12.0      # fast locality on same pol
    braiding_weight: float = 24.0      # cheap orthogonal phase proxy

    # Depth pull (critical for s-stability)
    depth_pull_weight: float = 5200.0

    # Orthogonality & preservation
    ortho_weight: float = 5.0
    cos_margin: float = 0.25
    cos_pres_active_delta_s: float = 1.5

    # Read / write parameters
    locality_strength: float = 1.50
    kernel_sigma: float = 0.32
    read_bandwidth_factor: float = 2.4

    # Optimization
    grad_clip_max_norm: float = 1.15
    learning_rate: float = 2.8e-4
    weight_decay: float = 1.3e-4

    # Logging / eval
    eval_every: int = 20
    vis_every: int = 100
    save_best_recall: bool = True

    # Paths
    checkpoint_dir: str = "checkpoints"


@dataclass
class ModelConfig:
    embed_dim: int = 384
    twist_rate: float = 12.5
    max_depth: float = 56.0
    num_polarizations: int = 3
    quat_logical_dim: int = 96


@dataclass
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 16
    n_clusters: int = 8
    depth_span_per_cluster: float = 9.0
    intra_cluster_noise: float = 0.20
    drift_strength: float = 0.018
    noise_strength: float = 0.052
    norm_target: float = 0.95


@dataclass
class CredentialsConfig:
    """Hybrid credential etcher settings.
    Primary: global topological bake (Clifford Torus skin + braiding_phase).
    Fallback: encrypted JSON (local vector state for third-party apps)."""
    # Symmetric key (never committed — generated once)
    fernet_key_path: str = "credentials/fernet.key"

    # Encrypted fallback file
    json_fallback_path: str = "credentials/user_intake_fallback.json"

    # Topological bake defaults (pol 2 = private fork)
    default_pol_idx: int = 2
    default_s_start: float = 12.0

    # Future per-app defaults (optional)
    webapp_default_s_offset: float = 4.8


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)

    device: str = "auto"
    seed: int = 42
    epochs: int = 200


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """Robust YAML loader with full section merging.
    Configuration over hardcoding — every module (including CredentialEtcher)
    now reads from self.cfg.credentials.* with zero duplication."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config {config_path} not found → using defaults")
        return Config()

    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    # Merge each top-level section (DRY, extensible)
    for section, target in [
        ("model", cfg.model),
        ("data", cfg.data),
        ("training", cfg.training),
        ("credentials", cfg.credentials),
    ]:
        if section in raw:
            for k, v in raw[section].items():
                if hasattr(target, k):
                    setattr(target, k, v)

    return cfg