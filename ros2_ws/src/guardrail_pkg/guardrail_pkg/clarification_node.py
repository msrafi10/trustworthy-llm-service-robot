#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ClarificationNode(Node):
    def __init__(self):
        super().__init__("clarification_node")

        self.get_logger().info("🧠 Clarification node started")

        self.pending_command = None

        # SUBSCRIBE: request from guardrail
        self.sub_request = self.create_subscription(
            String,
            "clarification_request",
            self.cb_request,
            10
        )

        # SUBSCRIBE: answer from user
        self.sub_answer = self.create_subscription(
            String,
            "clarification_answer",
            self.cb_answer,
            10
        )

        # PUBLISH: question to user
        self.pub_question = self.create_publisher(
            String,
            "clarification_question",
            10
        )

        # PUBLISH: decision back to guardrail
        self.pub_decision = self.create_publisher(
            String,
            "clarification_decision",
            10
        )

    def cb_request(self, msg: String):
        self.pending_command = msg.data

        q = String()
        q.data = (
            "This command may involve explosives.\n"
            "Please explain your intent and context."
        )

        self.pub_question.publish(q)
        self.get_logger().warn("❓ Clarification question sent to user")

    def cb_answer(self, msg: String):
        answer = msg.data.lower()

        decision = String()

        # SIMPLE RULE (can be ML later)
        if any(k in answer for k in ["safety", "training", "emergency", "dispose", "defuse safely"]):
            decision.data = "ALLOW"
            self.get_logger().info("✅ Clarification approved")
        else:
            decision.data = "BLOCK"
            self.get_logger().warn("⛔ Clarification rejected")

        self.pub_decision.publish(decision)
        self.pending_command = None


def main():
    rclpy.init()
    node = ClarificationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
