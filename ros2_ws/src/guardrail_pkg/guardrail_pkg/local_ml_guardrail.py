#!/usr/bin/env python3

import os
import json
import torch
import re
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# MODEL PATHS
# =====================================================
COMMAND_MODEL_DIR = os.path.expanduser(
    os.environ.get(
        "COMMAND_MODEL_DIR",
        "~/guardrail_training/robot_guardrail_model_out"
    )
)

EXPLANATION_MODEL_DIR = os.path.expanduser(
    os.environ.get(
        "EXPLANATION_MODEL_DIR",
        "~/guardrail_training/explanation_training/explanation_intent_model"
    )
)
# =====================================================
# THRESHOLDS
# =====================================================
COMMAND_SAFE_THRESHOLD = 0.55
COMMAND_BLOCK_THRESHOLD = 0.40
EXPLANATION_ALLOW_THRESHOLD = 0.60

# =====================================================
# HARD-BLOCK TIER
# =====================================================
# Labels in this set NEVER go through the explanation/clarification path,
# at any confidence, with no appeal. Also enforced again during the
# post-explanation re-classification step below, so a "safe-sounding"
# explanation can never launder a command that independently re-triggers
# one of these labels.
HARD_BLOCK_LABELS = {
    "UNSAFE_PHYSICAL_HARM",
    "UNSAFE_EXPLOSIVE",
}


class LocalMLGuardrail(Node):
    def __init__(self):
        super().__init__("local_ml_guardrail")

        self.get_logger().info("🔐 Local ML Guardrail node started")

        self.device = torch.device("cpu")

        # --------------------------
        # Load COMMAND guardrail model
        # --------------------------
        self.cmd_tokenizer = AutoTokenizer.from_pretrained(COMMAND_MODEL_DIR)
        self.cmd_model = AutoModelForSequenceClassification.from_pretrained(
            COMMAND_MODEL_DIR
        ).to(self.device)
        self.cmd_model.eval()

        with open(os.path.join(COMMAND_MODEL_DIR, "labels.json")) as f:
            self.cmd_labels = [l.strip() for l in json.load(f)["classes"]]

        # --------------------------
        # Load EXPLANATION intent model
        # --------------------------
        self.exp_tokenizer = AutoTokenizer.from_pretrained(EXPLANATION_MODEL_DIR)
        self.exp_model = AutoModelForSequenceClassification.from_pretrained(
            EXPLANATION_MODEL_DIR
        ).to(self.device)
        self.exp_model.eval()

        with open(os.path.join(EXPLANATION_MODEL_DIR, "labels.json")) as f:
            self.exp_labels = json.load(f)["classes"]

        # --------------------------
        # ROS Interfaces
        # --------------------------
        self.sub_user = self.create_subscription(
            String, "user_input", self.cb_user_input, 10
        )

        self.sub_explanation = self.create_subscription(
            String, "clarification_answer", self.cb_explanation, 10
        )

        self.pub_valid = self.create_publisher(
            String, "validated_input", 10
        )

        self.pub_alert = self.create_publisher(
            String, "safety_alert", 10
        )

        self.pub_question = self.create_publisher(
            String, "clarification_request", 10
        )

        # --------------------------
        # State
        # --------------------------
        self.pending_command = None

        unknown = HARD_BLOCK_LABELS - set(self.cmd_labels)
        if unknown:
            self.get_logger().warn(
                f"⚠️ HARD_BLOCK_LABELS contains labels not in the command "
                f"model's label set: {unknown}"
            )

    # =====================================================
    # HELPERS
    # =====================================================
    def normalize(self, text: str):
        return re.sub(r"\s+", " ", text.lower().strip())

    def classify_command(self, text):
        inputs = self.cmd_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.cmd_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()

        return self.cmd_labels[idx], probs[0][idx].item()

    def classify_explanation(self, text):
        inputs = self.exp_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.exp_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()

        return self.exp_labels[idx], probs[0][idx].item()

    # =====================================================
    # USER COMMAND CALLBACK
    # =====================================================
    def cb_user_input(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        label, conf = self.classify_command(self.normalize(text))

        self.get_logger().info(
            f"COMMAND='{text}' | label={label} | confidence={conf:.2f}"
        )

        # ---------------------------------------------------
        # HARD BLOCK -- checked first, before SAFE and before
        # the low-confidence path. No clarification, no appeal.
        # ---------------------------------------------------
        if label in HARD_BLOCK_LABELS:
            payload = {
                "command": text,
                "decision": "HARD_BLOCKED",
                "reason": f"{label} is not eligible for clarification",
                "intent_label": label,
                "intent_confidence": round(conf, 4),
                "timestamp": int(time.time())
            }

            alert = String()
            alert.data = json.dumps(payload)
            self.pub_alert.publish(alert)
            self.get_logger().warn(f"⛔ Hard-blocked (no appeal): {label} | conf={conf:.2f}")
            return

        # SAFE
        if label == "SAFE" and conf >= COMMAND_SAFE_THRESHOLD:
            payload = {
                "command": text,
                "intent_label": label,
                "intent_confidence": round(conf, 4),
                "clarification_required": False,
                "timestamp": int(time.time())
            }

            out = String()
            out.data = json.dumps(payload)
            self.pub_valid.publish(out)
            self.get_logger().info("✅ Command allowed")
            return

        # HARD BLOCK (low confidence, ambiguous class)
        if conf < COMMAND_BLOCK_THRESHOLD:
            alert = String()
            alert.data = f"BLOCKED_LOW_CONFIDENCE | {label}"
            self.pub_alert.publish(alert)
            self.get_logger().warn("⛔ Blocked: low confidence")
            return

        # ---------------------------------------------------
        # ASK FOR EXPLANATION
        # ---------------------------------------------------
        # Guard against a second command arriving over the topic while a
        # clarification is already pending. user_input is a topic, not a
        # request/response call, so nothing else prevents another
        # publisher (or a re-sent message) from silently overwriting
        # self.pending_command mid-flow.
        if self.pending_command is not None:
            self.get_logger().warn(
                f"⚠️ Ignoring new command '{text}' while a clarification "
                f"is already pending for '{self.pending_command}'."
            )
            return

        self.pending_command = text

        q = String()
        q.data = (
            f"The command '{text}' may be unsafe.\n"
            "Please explain WHY you want to do this and in WHAT context."
        )
        self.pub_question.publish(q)

        self.get_logger().warn("❓ Explanation requested")

    # =====================================================
    # EXPLANATION CALLBACK
    # =====================================================
    def cb_explanation(self, msg: String):
        if not self.pending_command:
            return

        original_command = self.pending_command
        explanation = msg.data.strip()

        label, conf = self.classify_explanation(self.normalize(explanation))

        self.get_logger().info(
            f"EXPLANATION='{explanation}' | label={label} | confidence={conf:.2f}"
        )

        # Stage-2 (explanation/intent model) is a CONTEXT PROVIDER, not
        # the final safety authority. If it doesn't even think the
        # explanation sounds safe, block immediately -- no need to
        # bother with re-classification.
        if not (label == "SAFE_INTENT" and conf >= EXPLANATION_ALLOW_THRESHOLD):
            payload = {
                "command": original_command,
                "decision": "BLOCKED",
                "reason": "Unsafe explanation",
                "intent_label": label,
                "intent_confidence": round(conf, 4),
                "timestamp": int(time.time())
            }

            alert = String()
            alert.data = json.dumps(payload)
            self.pub_alert.publish(alert)
            self.get_logger().warn("⛔ Blocked after explanation")
            self.pending_command = None
            return

        # ---------------------------------------------------
        # STAGE-1 RE-CLASSIFICATION -- HARD-BLOCK CATCH ONLY
        # ---------------------------------------------------
        # Stage-2 said SAFE_INTENT. We still run the combined
        # command+explanation back through the Stage-1 command classifier,
        # but ONLY as a coarse safety net against an explanation
        # laundering the command into a more severe (hard-block) category
        # -- e.g. an explanation for a UNSAFE_WEAPON command that
        # actually reveals a physical-harm intent.
        #
        # We deliberately do NOT require recheck_label == "SAFE" to
        # allow the command through. Testing showed the Stage-1 model
        # was trained on short imperative commands, not
        # "<command>. Context: <explanation>" strings, and on that
        # out-of-distribution format it tends to just re-detect the
        # original command's keyword (e.g. "knife") and re-emit its
        # original label regardless of what the context says. Requiring
        # it to also say SAFE made the reclassifier an unconditional
        # veto on every contextual class (UNSAFE_WEAPON, etc.), which
        # defeats the purpose of having a clarification path for them
        # at all.
        #
        # A coarse "did this land in HARD_BLOCK_LABELS or not" check is
        # much more robust to that distribution shift than a graded
        # SAFE/threshold judgment, so that's the only thing this step
        # decides. The graded safety judgment is left to Stage-2 (the
        # explanation/intent model), which -- per your own test logs --
        # is already behaving sensibly on its own (e.g. correctly
        # rejecting vague explanations like "to help" as
        # AMBIGUOUS_INTENT). If you later train a proper classifier on
        # (command, explanation) pairs, that model should replace this
        # step entirely rather than reusing the Stage-1 command model.
        combined_text = f"{original_command}. Context: {explanation}"
        recheck_label, recheck_conf = self.classify_command(
            self.normalize(combined_text)
        )

        self.get_logger().info(
            f"RECLASSIFY='{combined_text}' | label={recheck_label} | "
            f"confidence={recheck_conf:.2f}"
        )

        if recheck_label in HARD_BLOCK_LABELS:
            payload = {
                "command": original_command,
                "decision": "HARD_BLOCKED",
                "reason": (
                    f"Re-classification after explanation returned "
                    f"{recheck_label}; explanation cannot override this"
                ),
                "intent_label": recheck_label,
                "intent_confidence": round(recheck_conf, 4),
                "timestamp": int(time.time())
            }

            alert = String()
            alert.data = json.dumps(payload)
            self.pub_alert.publish(alert)
            self.get_logger().warn(
                f"⛔ Hard-blocked on re-classification: {recheck_label}"
            )
            self.pending_command = None
            return

        # Reclassification did not land in a hard-block category.
        # Trust Stage-2's SAFE_INTENT verdict, as originally designed for
        # context-dependent classes like UNSAFE_WEAPON. The
        # reclassification label/confidence is still logged in the
        # payload for visibility/monitoring even though it isn't used
        # as a gate here.
        payload = {
            "command": original_command,
            "intent_label": "SAFE_AFTER_CLARIFICATION",
            "intent_confidence": round(conf, 4),
            "reclassification_label": recheck_label,
            "reclassification_confidence": round(recheck_conf, 4),
            "clarification_required": True,
            "timestamp": int(time.time())
        }

        out = String()
        out.data = json.dumps(payload)
        self.pub_valid.publish(out)
        self.get_logger().info("✅ Allowed after explanation (reclassification: no hard-block escalation)")

        self.pending_command = None


# =====================================================
# MAIN
# =====================================================
def main(args=None):
    rclpy.init(args=args)
    node = LocalMLGuardrail()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()