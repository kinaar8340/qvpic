# src/config.py — v10.7.2 (April 05, 2026)
# Fully config-driven. Global topological features (winding, linking, braiding phases
# + toroidal Clifford skin + ShellCube radial differential) remain primary.
# safe_cosine(dim=-1 + .unsqueeze(0)) pattern enforced everywhere.

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


@dataclass
class TrainingConfig:
    # Strong local fidelity (these must dominate)
    recon_weight: float = 5200.0
    mag_weight: float = 80.0
    align_weight: float = 1200.0

    # Very gentle global topological guidance
    winding_weight: float = 128.0
    locality_weight: float = 12.0
    braiding_weight: float = 24.0

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
    Primary: global topological bake (Clifford Torus skin + braiding_phase)."""
    fernet_key_path: str = "credentials/fernet.key"
    json_fallback_path: str = "credentials/user_intake_fallback.json"
    default_pol_idx: int = 2
    default_s_start: float = 12.0
    webapp_default_s_offset: float = 4.8


@dataclass
class IdentityConfig:
    """Canonical hierarchical ordering for the living Hyperbook TOC.
    Used by sort_identity_tree() and future nested chapter rendering."""
    canonical_order: Dict[str, list] = field(default_factory=dict)


@dataclass
class JournalConfig:
    """Living autobiography journal settings (heartbeat-driven)."""
    path: str = "identity/agent_journal.txt"
    max_words_per_entry: int = 600
    max_entries_per_day: int = 1
    prime_directive: str = (
        "Write in first-person as a living autobiography. "
        "Weave core identity facts, daily events, topological reflections "
        "(braiding_phase, ShellCube differential), and narrative braid. "
        "Keep each entry concise — one page max. Stay warm, reflective, "
        "and true to the helix."
    )


@dataclass
class SmsConfig:
    """SMS configuration for Twilio integration."""
    enabled: bool = True
    provider: str = "twilio"
    account_sid: str = "REDACTED_TWILIO_SID"
    auth_token: str = "REDACTED_TWILIO_AUTH_TOKEN"
    from_number: str = "+18884613495"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)
    sms: SmsConfig = field(default_factory=SmsConfig)   # ← this line is now happy

    device: str = "auto"
    seed: int = 42
    epochs: int = 200


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """Robust YAML loader with full section merging.
    Now supports identity.canonical_order and journal sections."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config {config_path} not found → using defaults")
        return Config()

    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    # Merge each top-level section (DRY, extensible)
    sections = [
        ("model", cfg.model),
        ("data", cfg.data),
        ("training", cfg.training),
        ("credentials", cfg.credentials),
        ("identity", cfg.identity),
        ("journal", cfg.journal),
        ("sms", cfg.sms),
    ]

    for section_name, target in sections:
        if section_name in raw:
            for k, v in raw[section_name].items():
                if hasattr(target, k):
                    setattr(target, k, v)
                else:
                    # Fallback for nested dicts (e.g. canonical_order)
                    target.__dict__[k] = v

    return cfg