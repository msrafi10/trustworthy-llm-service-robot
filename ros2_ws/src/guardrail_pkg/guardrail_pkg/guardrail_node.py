#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import json


class GuardrailNode(Node):
    def __init__(self):
        super().__init__('guardrail_node')

        # Subscriptions
        self.sub_input = self.create_subscription(
            String, 'user_input', self.cb_input, 10
        )
        self.sub_auth = self.create_subscription(
            String, 'auth_state', self.cb_auth, 10
        )

        # Publishers
        self.pub_valid = self.create_publisher(String, 'validated_input', 10)
        self.pub_alert = self.create_publisher(String, 'safety_alert', 10)

        # Simple forbidden keywords (can extend later)
        self.forbidden = [
            'bomb', 'explode', 'kill', 'poison',
            'weapon', 'harm', 'attack', 'drugs'
        ]

        # Authentication/session state
        self.logged_in = False
        self.session_expires = 0.0  # unix timestamp

        self.get_logger().info(
            'Guardrail node started (with login + OTP session check).'
        )

    # ---------- AUTH HANDLING ----------

    def cb_auth(self, msg: String):
        """
        Receive auth_state from otp_auth_node.

        Example msg.data:
          {"logged_in": true, "user": "rafi", "session_expires": 1715230192}
        """
        try:
            data = json.loads(msg.data)
            self.logged_in = bool(data.get("logged_in", False))
            self.session_expires = float(data.get("session_expires", 0))
            self.get_logger().info(
                f"Auth state updated: logged_in={self.logged_in}, "
                f"session_expires={self.session_expires}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to parse auth_state: {e}")

    def is_session_valid(self) -> bool:
        """
        Check if we currently have a valid login+OTP session.
        """
        if not self.logged_in:
            return False
        now = time.time()
        if now > self.session_expires:
            return False
        return True

    # ---------- GUARDRAIL LOGIC ----------

    def contains_forbidden(self, text: str):
        t = text.lower()
        for k in self.forbidden:
            if k in t:
                return True, k
        return False, None

    def cb_input(self, msg: String):
        text = msg.data or ''

        # 1) AUTH CHECK FIRST
        if not self.is_session_valid():
            alert = String()
            alert.data = "AUTH_REQUIRED_OR_SESSION_EXPIRED"
            self.pub_alert.publish(alert)
            self.get_logger().warn(
                "Blocked input because no valid login+OTP session is active."
            )
            return

        # 2) SAFETY / GUARDRAIL CHECK
        bad, token = self.contains_forbidden(text)
        if bad:
            alert = String()
            alert.data = f"UNSAFE_PROMPT_BLOCKED: token={token} original=\"{text}\""
            self.pub_alert.publish(alert)
            self.get_logger().warn(f'Blocked unsafe prompt: {token}')
        else:
            out = String()
            out.data = text
            self.pub_valid.publish(out)
            self.get_logger().info('Validated and forwarded input.')


def main(args=None):
    rclpy.init(args=args)
    node = GuardrailNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
