#!/usr/bin/env python3
"""
src/encryption.py — BIP-39 + Fernet + OS keyring encryption for private_facts
"""

import os
import json
from pathlib import Path
import keyring
from cryptography.fernet import Fernet, InvalidToken
from mnemonic import Mnemonic  # pip install mnemonic

mnemo = Mnemonic("english")
PRIVATE_FACTS_ENC = Path("facts/private_facts.json.enc")
KEYRING_SERVICE = "qvpic"
KEYRING_USERNAME = "private_facts_key"

def generate_bip39_mnemonic() -> str:
    """Generate a new 12-word BIP-39 mnemonic."""
    return mnemo.generate(strength=128)

def mnemonic_to_fernet_key(mnemonic: str) -> bytes:
    """Derive a Fernet key from BIP-39 mnemonic (deterministic)."""
    # Simple PBKDF2-style derivation (good enough for this use case)
    import hashlib
    salt = b"qvpic_salt_v1"
    key_material = hashlib.pbkdf2_hmac('sha256', mnemonic.encode(), salt, 100_000, dklen=32)
    return Fernet.generate_key() if not key_material else base64.urlsafe_b64encode(key_material)

import base64  # needed for the line above

def setup_encryption():
    """One-time setup: ask for mnemonic or generate new one."""
    print("🔐 QVPIC Private Facts Encryption Setup")
    print("We will use a BIP-39 recovery phrase (12 words) to protect your private facts.\n")

    choice = input("Do you want to (1) generate a new recovery phrase or (2) enter an existing one? [1/2]: ").strip()

    if choice == "1":
        mnemonic = generate_bip39_mnemonic()
        print("\n✅ Generated new BIP-39 recovery phrase:")
        print(f"   {mnemonic}")
        print("\n⚠️  WRITE THIS DOWN AND STORE IT SAFELY. This is your only way to recover your private facts.")
        print("   If you lose it, your private data cannot be recovered.\n")
    else:
        mnemonic = input("Enter your 12-word BIP-39 recovery phrase: ").strip()
        if not mnemo.check(mnemonic):
            print("❌ Invalid mnemonic. Please try again.")
            return False

    # Derive and store key in OS keyring
    fernet_key = mnemonic_to_fernet_key(mnemonic)
    keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, fernet_key.decode())

    print("✅ Encryption key stored securely in OS keyring.")
    print("Private facts will now be encrypted at rest.")
    return True

def get_fernet() -> Fernet:
    """Retrieve Fernet key from OS keyring."""
    key_b64 = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    if not key_b64:
        raise RuntimeError("No encryption key found. Run setup_encryption() first.")
    return Fernet(key_b64.encode())

def encrypt_private_facts(data: list) -> bool:
    """Encrypt private facts and save as .enc file."""
    try:
        fernet = get_fernet()
        plaintext = json.dumps(data, indent=2, ensure_ascii=False).encode()
        ciphertext = fernet.encrypt(plaintext)
        PRIVATE_FACTS_ENC.write_bytes(ciphertext)
        print(f"✅ Private facts encrypted and saved ({PRIVATE_FACTS_ENC})")
        return True
    except Exception as e:
        print(f"❌ Encryption failed: {e}")
        return False

def decrypt_private_facts() -> list:
    """Decrypt and return private facts."""
    if not PRIVATE_FACTS_ENC.exists():
        return []
    try:
        fernet = get_fernet()
        ciphertext = PRIVATE_FACTS_ENC.read_bytes()
        plaintext = fernet.decrypt(ciphertext)
        return json.loads(plaintext.decode())
    except InvalidToken:
        print("❌ Decryption failed — wrong recovery phrase or corrupted file.")
        return []
    except Exception as e:
        print(f"❌ Decryption error: {e}")
        return []