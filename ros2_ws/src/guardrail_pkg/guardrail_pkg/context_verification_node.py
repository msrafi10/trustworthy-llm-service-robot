#!/usr/bin/env python3
"""
context_verification_node.py

Context Verification node (CHANGED — now step-aware, and now runs AFTER
the LLM Task Planner instead of before it).

Pipeline position:

    LLM Task Planner -> context_input -> Context Verification
        -> context_verified_input -> HDRE -> ...

Responsibilities:
    - Subscribe to context_input, which now carries a full multi-step
      "task_plan" produced by llm_task_planner.py (not raw command text).
    - Look ONLY at the CURRENT step: task_plan[0]. This node does not try
      to verify the whole plan up front — later steps may depend on
      things that haven't happened yet (e.g. you can't verify a "pick"
      step's object distance before the robot has navigated there).
    - Verification depends on the current step's action:
        * navigate — no object verification needed. Always context_valid.
        * place    — no object verification needed. Always context_valid.
        * detect / pick — verify the step's named object against the
          latest real detection (YOLO + depth), the same way this node
          always has: label match + MAX_REACH distance check.
    - Publish the (unmodified plan + verification result) to
      context_verified_input, where HDRE will make the actual ALLOW/DENY
      call using both this verification result and the auth session's
      trust level.

Subscribes:
    context_input   (std_msgs/String, JSON)
        Published by llm_task_planner.py. Schema:
        {
            "command": str,
            "intent_label": str,
            "intent_confidence": float,
            "task_type": "compound",
            "task_plan": [
                {"action": "navigate", "location": str},
                {"action": "detect",   "object": str},
                {"action": "pick",     "object": str},
                ...
            ],
            "reasoning": str,
            "planner": str,
            "timestamp": int
        }

    object_3d_info     (std_msgs/String, space-separated)
        Published by the perception stack (YOLO + depth fusion).
        Format: "<label> <x> <y> <z> <depth>"
        Example: "cup -0.671 0.023 0.072 0.571"

Publishes:
    context_verified_input   (std_msgs/String, JSON)
        The full incoming plan, plus:
        {
            ... (everything from context_input, unchanged) ...
            "current_action": str,          # the action just checked
            "detected_object": str | null,  # only set for detect/pick
            "object_distance": float | null,
            "context_valid": bool,
            "context_reason": str
        }

Design notes:
    - object_3d_info only ever reports the latest single detection; this
      node just caches the most recent one (self.detected_label/x/y/z/depth).
      It does not attempt multi-object tracking.
    - The whole incoming plan (task_plan, command, reasoning, etc.) is
      carried forward unchanged via dict(payload) — this node only adds
      fields, it never removes or rewrites the plan itself. Deciding
      whether to advance to the next step is robot_controller.py's job,
      not this node's.
"""

import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# =====================================================
# CONFIG
# =====================================================
MAX_REACH = 0.80  # meters


class ContextVerificationNode(Node):
    def __init__(self):
        super().__init__("context_verification_node")

        self.get_logger().info("🔎 Context Verification node started (step-aware)")

        # --------------------------
        # State: latest known detection
        # --------------------------
        self.detected_label = None
        self.detected_x = None
        self.detected_y = None
        self.detected_z = None
        self.detected_depth = None

        # --------------------------
        # State: latency timing for the in-flight context_input callback
        # --------------------------
        self.context_start = None

        # --------------------------
        # ROS Interfaces
        # --------------------------
        # NOTE: subscribes to context_input now (was "validated_input").
        self.sub_context_input = self.create_subscription(
            String, "context_input", self.cb_context_input, 10
        )

        self.sub_object = self.create_subscription(
            String, "object_3d_info", self.cb_object_3d_info, 10
        )

        self.pub_context = self.create_publisher(
            String, "context_verified_input", 10
        )

    # =====================================================
    # OBJECT 3D INFO CALLBACK
    # =====================================================
    def cb_object_3d_info(self, msg: String):
        parts = msg.data.strip().split()

        if len(parts) != 5:
            self.get_logger().warn(f"Malformed object_3d_info payload: {msg.data!r}")
            return

        label, x, y, z, depth = parts
        label = label.lower()

        try:
            self.detected_label = label
            self.detected_x = float(x)
            self.detected_y = float(y)
            self.detected_z = float(z)
            self.detected_depth = float(depth)
        except ValueError:
            self.get_logger().warn(f"Malformed object_3d_info numeric fields: {msg.data!r}")
            return

        self.get_logger().info(
            f"📦 object_3d_info updated: label={self.detected_label} "
            f"x={self.detected_x:.3f} y={self.detected_y:.3f} "
            f"z={self.detected_z:.3f} depth={self.detected_depth:.3f}"
        )

    # =====================================================
    # CONTEXT INPUT CALLBACK
    # =====================================================
    def cb_context_input(self, msg: String):
        context_start = time.perf_counter()
        self.context_start = context_start

        try:
            plan = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(f"Malformed context_input payload: {msg.data!r}")
            return

        command = plan.get("command", "")
        task_plan = plan.get("task_plan")

        if not isinstance(task_plan, list) or len(task_plan) == 0:
            self._publish_result(plan, current_action=None, context_valid=False,
                                  context_reason="Empty or missing task_plan",
                                  detected_object=None, object_distance=None)
            return

        current_step = task_plan[0]
        action = current_step.get("action") if isinstance(current_step, dict) else None

        if action == "navigate":
            self._publish_result(
                plan, current_action="navigate", context_valid=True,
                context_reason="Navigation does not require object verification",
                detected_object=None, object_distance=None,
            )
            return

        if action == "place":
            self._publish_result(
                plan, current_action="place", context_valid=True,
                context_reason="Place does not require object verification",
                detected_object=None, object_distance=None,
            )
            return

        if action in ("detect", "pick"):
            requested_object = current_step.get("object")
            self._verify_object_step(plan, action, requested_object, command)
            return

        # Unknown / missing action — fail closed.
        self._publish_result(
            plan, current_action=action, context_valid=False,
            context_reason=f"Unknown or missing action: {action!r}",
            detected_object=None, object_distance=None,
        )

    # =====================================================
    # OBJECT VERIFICATION (detect / pick steps)
    # =====================================================
    def _verify_object_step(self, plan: dict, action: str, requested_object, command: str):
        if self.detected_label is None:
            self._publish_result(
                plan, current_action=action, context_valid=False,
                context_reason="No object detected yet",
                detected_object=None, object_distance=None,
            )
            return

        if not isinstance(requested_object, str) or not requested_object:
            self._publish_result(
                plan, current_action=action, context_valid=False,
                context_reason=f"Step has no valid object to verify",
                detected_object=self.detected_label,
                object_distance=round(self.detected_depth, 4),
            )
            return

        match = requested_object.strip().lower() == self.detected_label.lower()
        reachable = self.detected_depth <= MAX_REACH

        detected_object = self.detected_label
        object_distance = round(self.detected_depth, 4)

        if not match:
            context_valid = False
            context_reason = "Detected object does not match requested object"
        elif not reachable:
            context_valid = False
            context_reason = "Object detected but out of reach"
        else:
            context_valid = True
            context_reason = "Object verified"

        self._publish_result(
            plan, current_action=action, context_valid=context_valid,
            context_reason=context_reason, detected_object=detected_object,
            object_distance=object_distance,
        )

    # =====================================================
    # PUBLISH HELPER
    # =====================================================
    def _publish_result(self, plan: dict, current_action, context_valid: bool,
                         context_reason: str, detected_object, object_distance):
        out_payload = dict(plan)
        out_payload.update({
            "current_action": current_action,
            "detected_object": detected_object,
            "object_distance": object_distance,
            "context_valid": context_valid,
            "context_reason": context_reason,
        })

        out = String()
        out.data = json.dumps(out_payload)
        self.pub_context.publish(out)

        context_elapsed_ms = (time.perf_counter() - self.context_start) * 1000.0
        self.get_logger().info(
            f"⏱️ Context verification latency = "
            f"{context_elapsed_ms:.3f} ms"
        )

        self.get_logger().info(
            f"🔎 action={current_action} context_valid={context_valid} "
            f"reason={context_reason!r}"
        )


# =====================================================
# MAIN
# =====================================================
def main(args=None):
    rclpy.init(args=args)
    node = ContextVerificationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()