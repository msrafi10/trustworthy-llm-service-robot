#!/usr/bin/env python3
"""
hdre_node.py

Hierarchical Decision & Risk Evaluation (HDRE) node (CHANGED — now the
LAST decision point before encryption, and deliberately much simpler).

Pipeline position:

    Context Verification -> context_verified_input -> HDRE
        -> planned_task -> secure_pub_v2 -> secure_sub_v2 -> robot_controller

Why this got simpler:
    Previously HDRE combined trust (from auth), safety (from the ML
    guardrail's intent_confidence), and context validity into one scored
    decision, and ran BEFORE the LLM planner. Now the guardrail's safety
    check has already gated everything reaching this pipeline (nothing
    unsafe ever gets a task plan generated for it), and Context
    Verification has already checked the current step against real
    perception data. So HDRE's only remaining job is the one thing
    nothing else in the pipeline checks: is this session actually
    authenticated and trusted right now.

    HDRE's decision is therefore just two conditions, both required:
        1. context_valid — Context Verification's per-step check passed.
        2. trust_score >= HDRE_CONFIG["trust_allow"] — the auth session
           is authenticated and not expired.
    DENY if either fails, ALLOW if both pass. There is no REVIEW tier
    and no safety_score/mission_risk scoring here anymore — the ML
    guardrail already made the safety call upstream, before the LLM
    planner ever ran.

    IMPORTANT: the trust check is NOT optional and is NOT dropped, even
    though a simplified sketch of this node ("just check context_valid")
    was floated. Dropping the trust check here would mean an
    unauthenticated or expired session could still reach ALLOW as long
    as the current step's object happened to verify — that would undo
    the entire OTP authentication layer. Both checks are kept.

Subscribes:
    context_verified_input  (std_msgs/String, JSON)
        Published by context_verification_node.py. Relevant fields:
        {
            "command": str,
            "task_plan": [...],
            "current_action": str,
            "context_valid": bool,
            "context_reason": str,
            ... (everything else from upstream, carried forward)
        }

    auth_state       (std_msgs/String, JSON)
        Published by the authentication pipeline (OTP layer). Schema:
        {
            "authenticated": bool,
            "auth_method": str,
            "trust_level": float,
            "timestamp": int,
            "session_expires": int
        }

Publishes:
    planned_task      (std_msgs/String, JSON)
        The full incoming payload (task_plan, current_action, etc.),
        plus:
        {
            ... (everything from context_verified_input, unchanged) ...
            "decision": "ALLOW" | "DENY",
            "trust_score": float,
            "reason": str,
            "hdre_version": str
        }
        This is now the FINAL topic before encryption — secure_pub_v2.py
        subscribes to "planned_task" directly, unchanged.

Design notes:
    - Session lifetime is owned entirely by the auth pipeline: HDRE just
      compares time.time() against auth_state's own "session_expires"
      field, not a separately-maintained timeout.
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# =====================================================
# CONFIG
# =====================================================
HDRE_VERSION = "2.0"  # bumped: this is the simplified, step-aware version

HDRE_CONFIG = {
    "trust_allow": 0.70,  # minimum trust_score to ever ALLOW
}


class HDRENode(Node):
    def __init__(self):
        super().__init__("hdre_node")

        self.get_logger().info("🧭 HDRE node started (v2 — trust + context gate)")

        # Latest known auth state. Defaults to "unauthenticated" until we
        # actually hear from the auth pipeline (fail closed).
        self.latest_auth = {
            "authenticated": False,
            "auth_method": None,
            "trust_level": 0.0,
            "timestamp": 0,
            "session_expires": 0,
        }

        # --------------------------
        # ROS Interfaces
        # --------------------------
        # NOTE: subscription topic name is unchanged — Context Verification
        # already published to "context_verified_input" before this
        # refactor too, so no rename was needed here.
        self.sub_context = self.create_subscription(
            String, "context_verified_input", self.cb_context_verified, 10
        )

        self.sub_auth = self.create_subscription(
            String, "auth_state", self.cb_auth_state, 10
        )

        # NOTE: publishes to planned_task now (was "hdre_output"). This
        # is the topic secure_pub_v2.py already subscribes to, so no
        # change is needed in secure_pub_v2.py / secure_sub_v2.py /
        # robot_controller.py for this step.
        self.pub_hdre = self.create_publisher(
            String, "planned_task", 10
        )

    # =====================================================
    # AUTH STATE CALLBACK
    # =====================================================
    def cb_auth_state(self, msg: String):
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(f"Malformed auth_state payload: {msg.data!r}")
            return

        self.latest_auth = {
            "authenticated": bool(data.get("authenticated", False)),
            "auth_method": data.get("auth_method"),
            "trust_level": float(data.get("trust_level", 1.0 if data.get("authenticated") else 0.0)),
            "timestamp": int(data.get("timestamp", time.time())),
            "session_expires": int(data.get("session_expires", 0)),
        }

        self.get_logger().info(
            f"🔑 auth_state updated: authenticated={self.latest_auth['authenticated']} "
            f"method={self.latest_auth['auth_method']} trust={self.latest_auth['trust_level']:.2f}"
        )

    # =====================================================
    # HELPERS
    # =====================================================
    def compute_trust_score(self) -> float:
        if not self.latest_auth["authenticated"]:
            return 0.0

        if time.time() > self.latest_auth["session_expires"]:
            self.get_logger().warn(
                "⏱️ auth session has expired (session_expires passed) — treating trust as 0"
            )
            return 0.0

        return max(0.0, min(1.0, self.latest_auth["trust_level"]))

    def decide(self, context_valid: bool, context_reason: str, trust_score: float):
        if not context_valid:
            return "DENY", context_reason or "Context verification failed"

        if trust_score < HDRE_CONFIG["trust_allow"]:
            return "DENY", "User is not authenticated or authentication has expired"

        return "ALLOW", "Authenticated user and verified step"

    # =====================================================
    # CONTEXT VERIFIED CALLBACK
    # =====================================================
    def cb_context_verified(self, msg: String):
        hdre_start = time.perf_counter()

        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(f"Malformed context_verified_input payload: {msg.data!r}")
            return

        command = payload.get("command", "")
        current_action = payload.get("current_action")
        context_valid = bool(payload.get("context_valid", False))
        context_reason = payload.get("context_reason", "")

        trust_score = self.compute_trust_score()
        decision, reason = self.decide(context_valid, context_reason, trust_score)

        out_payload = dict(payload)
        out_payload.update({
            "decision": decision,
            "trust_score": round(trust_score, 4),
            "reason": reason,
            "hdre_version": HDRE_VERSION,
        })

        out = String()
        out.data = json.dumps(out_payload)
        self.pub_hdre.publish(out)

        hdre_elapsed_ms = (time.perf_counter() - hdre_start) * 1000.0
        self.get_logger().info(
            f"⏱️ HDRE latency = {hdre_elapsed_ms:.3f} ms"
        )

        self.get_logger().info(
            f"🧭 HDRE decision={decision} action={current_action} "
            f"trust={trust_score:.2f} "
            f"context_valid={context_valid} | '{command}'"
        )


# =====================================================
# MAIN
# =====================================================
def main(args=None):
    rclpy.init(args=args)
    node = HDRENode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()