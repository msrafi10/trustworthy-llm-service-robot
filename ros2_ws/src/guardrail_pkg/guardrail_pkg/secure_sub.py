#!/usr/bin/env python3
import os
import json
import base64
import time
from typing import Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# -----------------------
# Config / key paths (must match publisher)
KEY_DIR = os.path.expanduser("~/.guard_keys")
MASTER_KEY_PATH = os.path.join(KEY_DIR, "master_key.bin")
ED25519_PUB_PATH = os.path.join(KEY_DIR, "ed25519_public.pem")

# load master key (same logic as publisher: env GUARD_KEY preferred)
def load_master_key_from_env_or_file() -> bytes:
    env = os.environ.get("GUARD_KEY")
    if env:
        try:
            mk = base64.b64decode(env)
            if len(mk) >= 16:
                return mk
        except Exception:
            mk = env.encode("utf-8")
            if len(mk) >= 8:
                return mk
    if not os.path.exists(MASTER_KEY_PATH):
        raise RuntimeError("Master key not found; set GUARD_KEY env or place file at " + MASTER_KEY_PATH)
    with open(MASTER_KEY_PATH, "rb") as f:
        return f.read()

def load_pub(pub_path=ED25519_PUB_PATH) -> Ed25519PublicKey:
    if not os.path.exists(pub_path):
        raise RuntimeError("Public signing key not found at " + pub_path)
    with open(pub_path, "rb") as f:
        pub = serialization.load_pem_public_key(f.read())
    return pub

def derive_per_message_key(master_key: bytes, nonce: bytes, info: bytes = b"ros2-guardrail-v2") -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info + (nonce[:4] if len(nonce) >= 4 else nonce),
    )
    return hkdf.derive(master_key)

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

# -----------------------
# Simple in-memory replay cache (msg_id -> ts). Keep entries for window seconds.
class ReplayCache:
    def __init__(self, window_seconds: int = 300):
        self.window = window_seconds
        self.store: Dict[str, int] = {}

    def is_replay(self, msg_id: str, ts: int) -> bool:
        # cleanup old entries
        now = int(time.time())
        to_delete = [k for k, v in self.store.items() if now - v > self.window]
        for k in to_delete:
            del self.store[k]
        # check if seen recently
        if msg_id in self.store:
            return True
        # not seen: accept and store
        self.store[msg_id] = ts
        return False

# -----------------------
# Node
class SecureSubV2(Node):
    def __init__(self):
        super().__init__('secure_sub_v2')
        self.get_logger().info("SecureSubV2 starting...")

        self.master_key = load_master_key_from_env_or_file()
        self.signing_pub = load_pub()
        self.replay_cache = ReplayCache(window_seconds=int(os.environ.get("GUARD_REPLAY_WINDOW", "300")))

        self.sub = self.create_subscription(String, 'secure_cmd', self.cb_secure, 10)
        self.pub_exec = self.create_publisher(String, 'executor_cmd', 10)

        self.get_logger().info("Secure subscriber v2 started.")

    def cb_secure(self, msg: String):
        try:
            envelope = json.loads(msg.data)
            nonce = b64d(envelope["nonce"])
            ciphertext = b64d(envelope["ciphertext"])
            aad_json = b64d(envelope["aad"])
            signature = b64d(envelope["signature"])
            sender = envelope.get("sender")
            ts = int(envelope.get("ts", 0))
            msg_id = envelope.get("msg_id")

            # 1) timestamp freshness
            now = int(time.time())
            allowed_skew = int(os.environ.get("GUARD_ALLOW_SKEW", "300"))
            if abs(now - ts) > allowed_skew:
                raise ValueError("Timestamp outside allowed skew window")

            # 2) replay detection
            if self.replay_cache.is_replay(msg_id, ts):
                raise ValueError("Replay detected for msg_id=" + str(msg_id))

            # 3) verify signature first: nonce || ciphertext || aad_json
            crypto_start = time.perf_counter()

            sig_input = nonce + ciphertext + aad_json
            try:
                self.signing_pub.verify(signature, sig_input)
            except Exception as e:
                raise ValueError("Signature verification failed") from e

            # 4) derive per-message key and decrypt
            permsg_key = derive_per_message_key(self.master_key, nonce)
            aesgcm = AESGCM(permsg_key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, aad_json)

            verify_decrypt_elapsed_ms = (
                time.perf_counter() - crypto_start
            ) * 1000.0

            plan = json.loads(plaintext.decode("utf-8"))
            out = String()
            out.data = json.dumps(plan)
            self.pub_exec.publish(out)
            self.get_logger().info(
                f"Decrypted msg_id={msg_id} "
                f"decision={plan.get('decision')} "
                f"command={plan.get('command')} "
                f"verify_decrypt_ms={verify_decrypt_elapsed_ms:.3f}"
            )


        except Exception as e:
            # On any failure: log and issue safe-stop
            self.get_logger().error(f"Decryption/validation failed: {e}")
            stop = String()
            stop.data = 'EMERGENCY_STOP'
            self.pub_exec.publish(stop)

def main(args=None):
    rclpy.init(args=args)
    node = SecureSubV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()