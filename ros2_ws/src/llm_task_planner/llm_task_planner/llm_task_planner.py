#!/usr/bin/env python3
"""
llm_task_planner.py

LLM Task Planner node.

Pipeline position (CHANGED — was previously downstream of HDRE, now sits
right after the ML guardrail, before Context Verification and HDRE):

    Local ML Guardrail -> validated_input -> LLM Task Planner
        -> context_input -> Context Verification (step-aware)
        -> context_verified_input -> HDRE -> planned_task -> secure_pub_v2 -> ...

Responsibilities:
    - Subscribe to validated_input. Every message here has ALREADY been
      classified SAFE (or SAFE_AFTER_CLARIFICATION) by local_ml_guardrail.py
      — that node only publishes to validated_input for safe commands, so
      this node doesn't re-check intent_label/decision itself.
    - Turn the command into a MULTI-STEP task plan: an ordered list of
      steps, each with an "action" (navigate / detect / pick / place) and
      whatever parameters that action needs (location or object).
    - Enforce and validate a strict JSON schema on the model's output,
      including per-step action vocabulary and per-step required fields.
      A response that doesn't validate is retried; if all retries are
      exhausted, nothing is published (fail closed).
    - Publish the validated multi-step plan to context_input, where
      context_verification_node.py will check the CURRENT step (not the
      whole plan) against real perception data before HDRE ever sees it.

IMPORTANT — what this node does NOT do:
    - It does NOT make an authorization decision. There is no "decision"
      field in its output. Authorization now happens later, in HDRE,
      after Context Verification has checked the current step and after
      HDRE has checked the auth session's trust level. This node's output
      is a *proposed* plan, not an approved one.
    - It does NOT verify that objects exist, are reachable, or are
      correctly located — that's Context Verification's job, and it runs
      AFTER this node now (previously it ran before, extracting the
      object from raw command text; now the LLM already gives Context
      Verification an explicit object/location per step, which is more
      reliable than re-deriving it from text).

Subscribes:
    validated_input   (std_msgs/String, JSON)
        Published by local_ml_guardrail.py. Schema:
        {
            "command": str,
            "intent_label": str,
            "intent_confidence": float,
            "clarification_required": bool,
            "timestamp": int
        }

Publishes:
    context_input  (std_msgs/String, JSON)
        {
            "command": str,
            "intent_label": str,          # carried forward from guardrail
            "intent_confidence": float,   # carried forward from guardrail
            "task_type": "compound",
            "task_plan": [
                {"action": "navigate", "location": str},
                {"action": "detect",   "object": str},
                {"action": "pick",     "object": str},
                {"action": "navigate", "location": str},
                {"action": "place"}
            ],
            "reasoning": str,       # short, LLM-authored, informational only
            "planner": str,
            "timestamp": int
        }

Requires:
    - A running Ollama server (default http://localhost:11434) with the
      configured model pulled, e.g.:
          ollama pull qwen2.5:7b-instruct
    - The `requests` package in whatever Python environment runs this
      node (pip install requests --break-system-packages if needed).
"""

import json
import os
import re
import time

import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# =====================================================
# CONFIG
# =====================================================
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# NOTE: adjust this to whatever tag `ollama list` shows on your machine —
# Ollama model tags don't always match the display name used below.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

PLANNER_NAME = "qwen2.5-7b-instruct"

REQUEST_TIMEOUT_SECONDS = 25
MAX_RETRIES = 2  # total attempts = MAX_RETRIES + 1

MAX_REASONING_CHARS = 200
MAX_PLAN_STEPS = 8  # sanity cap so a runaway response can't produce a huge plan

# --- Schema vocabulary -------------------------------------------------
# Keep these in sync with context_verification_node.py and robot_controller.py.
ALLOWED_ACTIONS = ["navigate", "detect", "pick", "place"]

ALLOWED_OBJECTS = [
    "bottle",
    "cup",
    "apple",
    "medicine",
    "knife",
    "syringe",
]

ALLOWED_LOCATIONS = [
    "table",
    "tv room",
    "kitchen",
    "room 1",
    "room 2",
    "reception",
    "user",
]

SYSTEM_PROMPT = f"""You are a task planner for a mobile manipulator robot (TurtleBot3 Waffle + OpenMANIPULATOR-X).

You will be given a JSON context block describing a command that has
ALREADY been classified as safe by an upstream safety pipeline. Your only
job is to break it into an ordered sequence of steps. You do not make
authorization or safety decisions — do not include "decision", "allow",
or "approved" fields. Any such field will be ignored.

Respond with ONLY a single JSON object, no prose, no markdown fences, no
explanation. The JSON object MUST have exactly these fields:

{{
  "task_type": "compound",
  "task_plan": an ordered list of 1 to {MAX_PLAN_STEPS} step objects, where each
    step has an "action" field which is one of {ALLOWED_ACTIONS}, plus:
      - "navigate" steps also need: "location", one of {ALLOWED_LOCATIONS}
      - "detect" steps also need: "object", one of {ALLOWED_OBJECTS}
      - "pick" steps also need: "object", one of {ALLOWED_OBJECTS}
      - "place" steps need no extra field
  "reasoning": a single short sentence (under 25 words) explaining the plan in
    plain language, for logging/display only — it has no effect on execution
}}

Rules:
- A typical fetch command ("bring me the X") becomes:
  navigate(to X's location) -> detect(X) -> pick(X) -> navigate(to "user") -> place
- A pure navigation command ("go to the kitchen") becomes a single
  navigate step.
- Use "table" as the default location for an object if the command
  doesn't name a specific room.
- Output strictly valid JSON. Do not include comments or trailing commas.
- Do not include a "decision" field — authorization is not your job.
"""


class LLMTaskPlanner(Node):
    def __init__(self):
        super().__init__("llm_task_planner")

        self.get_logger().info("🧠 LLM Task Planner node started")
        self.get_logger().info(f"   Ollama host: {OLLAMA_HOST}")
        self.get_logger().info(f"   Ollama model: {OLLAMA_MODEL}")

        # --------------------------
        # ROS Interfaces
        # --------------------------
        # NOTE: this node now sits right after the guardrail, not after
        # HDRE — it subscribes to validated_input, not hdre_output.
        self.sub_validated = self.create_subscription(
            String, "validated_input", self.cb_validated_input, 10
        )

        # NOTE: publishes to context_input now (was "planned_task").
        # "planned_task" is now published by hdre_node.py instead, after
        # Context Verification and HDRE have both run.
        self.pub_plan = self.create_publisher(
            String, "context_input", 10
        )

    # =====================================================
    # VALIDATED INPUT CALLBACK
    # =====================================================
    def cb_validated_input(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(f"Malformed validated_input payload: {msg.data!r}")
            return

        command = payload.get("command", "")

        if not command:
            self.get_logger().warn("validated_input received but command is empty — skipping")
            return

        self.get_logger().info(f"🧠 Planning task for: '{command}'")
        llm_start = time.perf_counter()
        plan = self.plan_task(command)

        if plan is None:
            self.get_logger().error(
                f"❌ Failed to produce a valid plan after {MAX_RETRIES + 1} attempts "
                f"— nothing published for '{command}'"
            )
            return

        out_payload = {
            "command": command,
            "intent_label": payload.get("intent_label"),
            "intent_confidence": payload.get("intent_confidence"),
            "task_type": "compound",
            "task_plan": plan["task_plan"],
            "reasoning": plan["reasoning"],
            "planner": PLANNER_NAME,
            "timestamp": int(time.time()),
        }

        out = String()
        out.data = json.dumps(out_payload)
        self.pub_plan.publish(out)

        llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000.0

        self.get_logger().info(
            f"✅ Published task_plan ({len(plan['task_plan'])} steps): "
            f"{plan['task_plan']} reasoning={plan['reasoning']!r}"
        )

        self.get_logger().info(
            f"⏱️ LLM planning latency = {llm_elapsed_ms:.3f} ms"
        )

    # =====================================================
    # PLANNING
    # =====================================================
    def plan_task(self, command: str):
        """
        Calls the LLM (with retries) and returns a validated plan dict,
        or None if no valid plan could be produced.
        """
        last_error = None

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                raw_text = self.call_ollama(command)
            except requests.exceptions.Timeout:
                last_error = "Ollama request timed out"
                self.get_logger().warn(f"⏱️ Attempt {attempt}: {last_error}")
                continue
            except requests.exceptions.ConnectionError as e:
                last_error = f"Could not reach Ollama at {OLLAMA_HOST}: {e}"
                self.get_logger().error(f"🔌 Attempt {attempt}: {last_error}")
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"Ollama request failed: {e}"
                self.get_logger().warn(f"Attempt {attempt}: {last_error}")
                continue

            plan = self.parse_and_validate(raw_text)

            if plan is not None:
                if attempt > 1:
                    self.get_logger().info(f"✅ Valid plan on attempt {attempt}")
                return plan

            last_error = "Model response failed schema validation"
            self.get_logger().warn(
                f"Attempt {attempt}: {last_error} | raw response: {raw_text[:300]!r}"
            )

        self.get_logger().error(f"All planning attempts exhausted. Last error: {last_error}")
        return None

    def call_ollama(self, command: str) -> str:
        url = f"{OLLAMA_HOST}/api/generate"

        body = {
            "model": OLLAMA_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": f'Command: "{command}"\n\nRespond with the JSON plan now.',
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        response = requests.post(url, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "")

    # =====================================================
    # VALIDATION
    # =====================================================
    def parse_and_validate(self, raw_text: str):
        if not raw_text or not raw_text.strip():
            return None

        cleaned = self._strip_code_fences(raw_text)

        try:
            plan = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(plan, dict):
            return None

        if plan.get("task_type") != "compound":
            self.get_logger().warn(f"Unsupported task_type: {plan.get('task_type')!r}")
            return None

        steps = plan.get("task_plan")
        if not isinstance(steps, list) or not (1 <= len(steps) <= MAX_PLAN_STEPS):
            self.get_logger().warn(
                f"task_plan must be a list of 1-{MAX_PLAN_STEPS} steps, got: {steps!r}"
            )
            return None

        validated_steps = []
        for i, step in enumerate(steps):
            validated_step = self._validate_step(step, i)
            if validated_step is None:
                return None
            validated_steps.append(validated_step)

        reasoning = plan.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = ""
        reasoning = reasoning.strip()[:MAX_REASONING_CHARS]

        return {
            "task_plan": validated_steps,
            "reasoning": reasoning,
        }

    def _validate_step(self, step, index: int):
        if not isinstance(step, dict):
            self.get_logger().warn(f"Step {index} is not an object: {step!r}")
            return None

        action = step.get("action")
        if not isinstance(action, str):
            self.get_logger().warn(f"Step {index} missing/invalid 'action': {step!r}")
            return None
        action = action.strip().lower()

        if action not in ALLOWED_ACTIONS:
            self.get_logger().warn(f"Step {index} has unsupported action: {action!r}")
            return None

        if action == "navigate":
            location = step.get("location")
            if not isinstance(location, str):
                self.get_logger().warn(f"Step {index} (navigate) missing 'location'")
                return None
            location = location.strip().lower()
            if location not in ALLOWED_LOCATIONS:
                self.get_logger().warn(f"Step {index} (navigate) unsupported location: {location!r}")
                return None
            return {"action": "navigate", "location": location}

        if action in ("detect", "pick"):
            obj = step.get("object")
            if not isinstance(obj, str):
                self.get_logger().warn(f"Step {index} ({action}) missing 'object'")
                return None
            obj = obj.strip().lower()
            if obj not in ALLOWED_OBJECTS:
                self.get_logger().warn(f"Step {index} ({action}) unsupported object: {obj!r}")
                return None
            return {"action": action, "object": obj}

        # action == "place"
        return {"action": "place"}

    # =====================================================
    # HELPERS
    # =====================================================
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text)
            text = re.sub(r"```$", "", text)
        return text.strip()


# =====================================================
# MAIN
# =====================================================
def main(args=None):
    rclpy.init(args=args)
    node = LLMTaskPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()