#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json
import time
import getpass


class RobotCli(Node):
    """
    Interactive CLI with:
      1) Username + password login
      2) Email OTP verification
      3) Command input
      4) Human-in-the-loop clarification
    """

    def __init__(self):
        super().__init__("robot_cli")

        # =========================
        # Publishers
        # =========================
        self.pub_login = self.create_publisher(String, "login_request", 10)
        self.pub_otp = self.create_publisher(String, "otp_input", 10)
        self.pub_cmd = self.create_publisher(String, "user_input", 10)

        self.pub_clarification_answer = self.create_publisher(
            String, "clarification_answer", 10
        )

        # =========================
        # Subscribers
        # =========================
        self.sub_auth = self.create_subscription(
            String, "auth_state", self.cb_auth, 10
        )

        self.sub_login_result = self.create_subscription(
            String, "login_result", self.cb_login_result, 10
        )

        self.sub_clarification_request = self.create_subscription(
            String, "clarification_request", self.cb_clarification_request, 10
        )

        self.sub_validated = self.create_subscription(
            String, "validated_input", self.cb_validated, 10
        )

        self.sub_alert = self.create_subscription(
            String, "safety_alert", self.cb_alert, 10
        )

        # =========================
        # State
        # =========================
        self.logged_in = False
        self.session_expires = 0.0
        self.waiting_for_clarification = False
        self.login_status = None
        self.login_attempts = 0
        self.max_login_attempts = 5

        # Only cb_validated / cb_alert set this now. It represents the
        # guardrail's FINAL verdict on a submitted command -- not the
        # fact that a clarification answer was merely submitted.
        self.last_guardrail_decision = None

        # Set True the instant a clarification answer has been published,
        # so the main loop knows to give the guardrail a *fresh* timeout
        # window for its follow-up (Stage-2 + reclassification) verdict,
        # instead of judging against a deadline that may have already
        # expired while the human was busy typing their explanation.
        self.clarification_just_submitted = False

        self.get_logger().info("🤖 Robot CLI started")

    # =========================
    # CALLBACKS
    # =========================

    def cb_auth(self, msg: String):
        try:
            data = json.loads(msg.data)

            authenticated = bool(data.get("authenticated", False))
            trust = float(data.get("trust_level", 0.0))

            self.logged_in = authenticated and trust >= 1.0
            self.session_expires = float(data.get("session_expires", 0))
            user = data.get("user", "unknown")

            self.get_logger().info(
                f"AUTH_STATE → authenticated={authenticated}, "
                f"trust={trust:.2f}, logged_in={self.logged_in}, user={user}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to parse auth_state: {e}")

    def cb_login_result(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.login_status = data.get("status")
            self.get_logger().info(f"LOGIN_RESULT → {self.login_status}")
        except Exception as e:
            self.get_logger().error(f"Failed to parse login_result: {e}")

    def cb_clarification_request(self, msg: String):
        """
        System asks clarification → pause CLI and ask user.

        IMPORTANT: this does NOT mark the command as finished. It only
        marks that an answer has been sent. The command is only truly
        finished once cb_validated or cb_alert fires with the guardrail's
        real verdict on the explanation (and any re-classification).
        """

        self.waiting_for_clarification = True

        print("\n======================================")
        print("❓ CLARIFICATION REQUIRED BY SYSTEM")
        print("======================================")
        print(msg.data)

        answer = input("\n🧑 Your explanation: ").strip()

        reply = String()
        reply.data = answer
        self.pub_clarification_answer.publish(reply)

        print("✅ Clarification sent to system. Waiting for final decision...\n")

        self.waiting_for_clarification = False
        self.clarification_just_submitted = True

    def cb_validated(self, msg: String):
        self.get_logger().info(f"VALIDATED → {msg.data}")
        self.last_guardrail_decision = "validated"

    def cb_alert(self, msg: String):
        self.get_logger().warn(f"ALERT → {msg.data}")
        self.last_guardrail_decision = "alert"

    # =========================
    # HELPERS
    # =========================

    def send_login(self, username: str, password: str):
        msg = String()
        msg.data = json.dumps({"username": username, "password": password})
        self.pub_login.publish(msg)

    def send_otp(self, otp: str):
        msg = String()
        msg.data = otp.strip()
        self.pub_otp.publish(msg)

    def send_command(self, text: str):
        msg = String()
        msg.data = text
        self.pub_cmd.publish(msg)

    def session_valid(self) -> bool:
        if not self.logged_in:
            return False
        return time.time() <= self.session_expires


# =========================
# MAIN LOOP
# =========================

def main(args=None):
    rclpy.init(args=args)
    node = RobotCli()

    INITIAL_TIMEOUT_SEC = 5.0
    POST_CLARIFICATION_TIMEOUT_SEC = 10.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            # -------------------------
            # LOGIN + OTP
            # -------------------------
            if not node.session_valid():
                if node.login_attempts >= node.max_login_attempts:
                    print("\n🚫 Too many failed login attempts. Exiting.\n")
                    break

                print("\n==============================")
                print("🔐 LOGIN REQUIRED")
                print("==============================")

                username = input("Username: ")
                password = getpass.getpass("Password: ")

                node.login_status = None
                node.send_login(username, password)

                for _ in range(3):
                    rclpy.spin_once(node, timeout_sec=0.05)

                print("\n⏳ Checking credentials...")
                start = time.time()
                while time.time() - start < 10:
                    rclpy.spin_once(node, timeout_sec=0.2)
                    if node.login_status is not None:
                        break

                if node.login_status == "LOGIN_FAILED":
                    print("❌ Invalid username or password.\n")
                    node.login_status = None
                    node.login_attempts += 1
                    continue

                if node.login_status != "OTP_REQUIRED":
                    print("❌ No response from authentication node. Try again.\n")
                    node.login_status = None
                    continue

                print("\n📧 OTP sent to your email")
                otp = input("Enter OTP: ")

                node.login_status = None
                node.send_otp(otp)
                for _ in range(3):
                    rclpy.spin_once(node, timeout_sec=0.05)

                print("\n⏳ Verifying...")
                start = time.time()
                while time.time() - start < 10:
                    rclpy.spin_once(node, timeout_sec=0.2)

                    if node.session_valid():
                        break

                    time.sleep(0.05)

                if not node.session_valid():
                    if time.time() - start >= 10:
                        print("⌛ Authentication timed out.\n")
                    else:
                        print("❌ Authentication failed.\n")

                    node.login_status = None
                    node.login_attempts += 1
                    continue

                print("✅ Login successful\n")
                node.login_status = None
                node.login_attempts = 0

            # -------------------------
            # WAIT DURING CLARIFICATION
            # -------------------------
            if node.waiting_for_clarification:
                continue

            # -------------------------
            # COMMAND INPUT
            # -------------------------
            rclpy.spin_once(node, timeout_sec=0.1)

            remaining = max(0, int(node.session_expires - time.time()))

            print("------------------------------------")
            print(f"🟢 Session active ({remaining}s remaining)")
            cmd = input("Command (or 'exit'): ").strip()

            if cmd.lower() in ("exit", "logout", "quit"):
                print("👋 Exiting CLI")
                break

            if not cmd:
                continue

            # -------------------------
            # WAIT FOR THE GUARDRAIL'S FINAL VERDICT
            # -------------------------
            # This loop is the fix for the original race condition. It
            # blocks the CLI from showing the next command prompt until
            # one of three things happens:
            #   1. The guardrail allows/blocks the command outright
            #      (validated_input / safety_alert) — done immediately.
            #   2. The guardrail asks for clarification. That callback
            #      (cb_clarification_request) runs synchronously inside
            #      one of the spin_once() calls below and blocks on
            #      input() until the human answers, so there is no gap
            #      in which a stray next command can be misread as the
            #      answer. Once the answer is sent, we do NOT treat the
            #      command as finished — we reset the deadline and keep
            #      waiting for the guardrail's real follow-up verdict.
            #   3. Nothing comes back within the timeout — we warn and
            #      move on rather than hanging forever.
            node.last_guardrail_decision = None
            node.clarification_just_submitted = False
            node.send_command(cmd)

            deadline = time.time() + INITIAL_TIMEOUT_SEC

            while True:
                rclpy.spin_once(node, timeout_sec=0.1)

                if node.last_guardrail_decision is not None:
                    break

                if node.clarification_just_submitted:
                    # Human has answered; the guardrail still needs to
                    # run Stage-2 inference (and, if implemented,
                    # re-classification) before validated_input or
                    # safety_alert arrives. Give it a fresh window
                    # rather than judging against the old deadline,
                    # which may have already elapsed while the human
                    # was typing.
                    node.clarification_just_submitted = False
                    deadline = time.time() + POST_CLARIFICATION_TIMEOUT_SEC
                    continue

                if time.time() > deadline:
                    print(
                        "⚠️  No final guardrail response received in time — "
                        "check that local_ml_guardrail is running.\n"
                    )
                    break

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()