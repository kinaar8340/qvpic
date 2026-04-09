#!/usr/bin/env python3
"""
scripts/setup_identity.py
QVPIC Identity Compiler v1.0 — u*/a* naming convention
"""

import json
from pathlib import Path
from datetime import datetime


def load_md_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    print(f"⚠️  {path} not found – using empty content")
    return ""


def main():
    root = Path(__file__).parent.parent
    identity = root / "identity"
    facts_dir = root / "facts"

    identity.mkdir(parents=True, exist_ok=True)
    (identity / "user").mkdir(exist_ok=True)
    (identity / "agent").mkdir(exist_ok=True)
    facts_dir.mkdir(exist_ok=True)

    print("🔧 QVPIC Identity Compiler v1.0 (u*/a* naming)")

    # Load all six human-readable files
    user_public = load_md_file(identity / "user" / "upublic.md")
    user_private = load_md_file(identity / "user" / "uprivate.md")
    user_journal = load_md_file(identity / "user" / "ujournal.md")
    agent_public = load_md_file(identity / "agent" / "apublic.md")
    agent_private = load_md_file(identity / "agent" / "aprivate.md")
    agent_journal = load_md_file(identity / "agent" / "ajournal.md")

    timestamp = datetime.now().isoformat()

    public_facts = [
        {"text": user_public, "source": "user_upublic", "timestamp": timestamp, "type": "public"},
        {"text": agent_public, "source": "agent_apublic", "timestamp": timestamp, "type": "public"},
        {"text": user_journal, "source": "user_ujournal", "timestamp": timestamp, "type": "journal"},
    ]
    private_facts = [
        {"text": user_private, "source": "user_uprivate", "timestamp": timestamp, "type": "private"},
        {"text": agent_private, "source": "agent_aprivate", "timestamp": timestamp, "type": "private"},
        {"text": agent_journal, "source": "agent_ajournal", "timestamp": timestamp, "type": "journal"},
    ]

    (facts_dir / "public_facts.json").write_text(
        json.dumps(public_facts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (facts_dir / "private_facts.json").write_text(
        json.dumps(private_facts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("✅ Identity compiled successfully!")
    print(f"   → facts/public_facts.json  ({len(public_facts)} entries)")
    print(f"   → facts/private_facts.json ({len(private_facts)} entries)")
    print("\nYou can now run: python scripts/main.py")


if __name__ == "__main__":
    main()