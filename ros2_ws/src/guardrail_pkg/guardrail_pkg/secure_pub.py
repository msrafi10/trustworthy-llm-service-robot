#!/usr/bin/env python3
import os
import json
import base64
import time
import uuid
from typing import Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# -----------------------
# Configuration & Key helpers
# -----------------------
KEY_DIR = os.path.expanduser("~/.guard_keys")
MASTER_KEY_PATH = os.path.join(KEY_DIR, "master_key.bin")   # optional local file fallback
ED25519_PRIV_PATH = os.path.join(KEY_DIR, "ed25519_private.pem")
ED25519_PUB_PATH = os.path.join(KEY_DIR, "ed25519_public.pem")

# Accept GUARD_KEY from env (preferred in deployment). If present, try base64 decode, else use raw bytes.
def load_master_key_from_env_or_file() -> bytes:
    env = os.environ.get("GUARD_KEY")
    if env:
        # try base64 decode
        try:
            mk = base64.b64decode(env)
            if len(mk) >= 16:
                # OK use mk (we will derive 32 bytes via HKDF later if needed)
                return mk
        except Exception:
            # fall back to raw utf-8 bytes
            mk = env.encode("utf-8")
            if len(mk) >= 8:
                return mk
    # fallback to file; if not exist generate random one (only for prototype)
    os.makedirs(KEY_DIR, exist_ok=True)
    if not os.path.exists(MASTER_KEY_PATH):
        with open(MASTER_KEY_PATH, "wb") as f:
            f.write(os.urandom(32))
        os.chmod(MASTER_KEY_PATH, 0o600)
    with open(MASTER_KEY_PATH, "rb") as f:
        return f.read()

def load_or_create_ed25519(priv_path=ED25519_PRIV_PATH, pub_path=ED25519_PUB_PATH) -> Ed25519PrivateKey:
    os.makedirs(KEY_DIR, exist_ok=True)
    if not os.path.exists(priv_path) or not os.path.exists(pub_path):
        priv = Ed25519PrivateKey.generate()
        priv_bytes = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        pub = priv.public_key()
        pub_bytes = pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(priv_path, "wb") as f:
            f.write(priv_bytes)
        with open(pub_path, "wb") as f:
            f.write(pub_bytes)
        os.chmod(priv_path, 0o600)
        os.chmod(pub_path, 0o644)
        return priv
    else:
        with open(priv_path, "rb") as f:
            priv = serialization.load_pem_private_key(f.read(), password=None)
        return priv

# Derive an AES-256 key (32 bytes) per-message using HKDF(SHA256)
def derive_per_message_key(master_key: bytes, nonce: bytes, info: bytes = b"ros2-guardrail-v2") -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info + (nonce[:4] if len(nonce) >= 4 else nonce),
    )
    return hkdf.derive(master_key)

def b64(v: bytes) -> str:
    return base64.b64encode(v).decode("utf-8")

# -----------------------
# Node
# -----------------------
class SecurePubV2(Node):
    def __init__(self):
        super().__init__('secure_pub_v2')
        self.get_logger().info("SecurePubV2 starting...")

        # load keys
        self.master_key = load_master_key_from_env_or_file()
        self.signing_priv = load_or_create_ed25519()

        # Topics: now sourced from HDRE decision output instead of validated_input
        self.sub = self.create_subscription(String, 'planned_task', self.cb_valid, 10)
        self.pub = self.create_publisher(String, 'secure_cmd', 10)

        self.get_logger().info("Secure publisher v2 started.")

    def cb_valid(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            self.get_logger().warn(f"Malformed planned_task payload: {msg.data!r} ({e})")
            return

        plaintext = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        # metadata
        sender = os.environ.get("GUARD_SENDER_ID", "robot-controller")
        msg_id = str(uuid.uuid4())
        ts = int(time.time())

        # AAD binds metadata to AEAD tag
        aad_obj: Dict[str, Any] = {"sender": sender, "msg_id": msg_id, "ts": ts}
        aad_json = json.dumps(aad_obj, separators=(",", ":"), sort_keys=True).encode("utf-8")

        # nonce (12 bytes for AES-GCM)
        nonce = os.urandom(12)

        # derive per-message key and encrypt
        crypto_start = time.perf_counter()

        permsg_key = derive_per_message_key(self.master_key, nonce)
        aesgcm = AESGCM(permsg_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad_json)

        # sign nonce || ciphertext || aad_json
        sig_input = nonce + ciphertext + aad_json
        signature = self.signing_priv.sign(sig_input)

        crypto_elapsed_ms = (time.perf_counter() - crypto_start) * 1000.0

        envelope = {
            "msg_id": msg_id,
            "ts": ts,
            "sender": sender,
            "nonce": b64(nonce),
            "ciphertext": b64(ciphertext),
            "aad": b64(aad_json),
            "signature": b64(signature),
        }
        out = String()
        out.data = json.dumps(envelope, separators=(",", ":"))
        self.pub.publish(out)
        self.get_logger().info(
            f"Published encrypted msg_id={msg_id} "
            f"encrypt_sign_ms={crypto_elapsed_ms:.3f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = SecurePubV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()