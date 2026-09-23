#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import time
import json
import random
import os
import smtplib
import ssl
from email.message import EmailMessage


class OtpAuthNode(Node):
    """
    Username+Password + OTP-based 2-step authentication node.

    Flow:
    1) User sends login credentials to /login_request:
         {"username": "<AUTH_USER>", "password": "<AUTH_PASS>"}
    2) Node immediately publishes /login_result so the CLI knows whether
       to proceed to the OTP step at all:
         {"status": "LOGIN_FAILED"}   — bad username/password, or the
                                         OTP email could not be sent
         {"status": "OTP_REQUIRED"}   — credentials correct, OTP emailed
    3) If username+password match, node generates a 6-digit OTP and
       sends it by email to OTP_MAIL_TO.
    4) User reads OTP from email and sends it to /otp_input.
    5) If OTP is correct and not expired, node publishes /auth_state:
         {"authenticated": true, "user": "<AUTH_USER>", "auth_method": "OTP",
          "trust_level": 1.0, "timestamp": <ts>, "session_expires": <ts>}
    """

    def __init__(self):
        super().__init__('otp_auth_node')
        self.get_logger().info("OtpAuthNode (user+pass + email OTP) started.")

        # Credentials are supplied through environment variables.
        # Do not store credentials in source code.
        self.valid_username = os.environ.get("AUTH_USER")
        self.valid_password = os.environ.get("AUTH_PASS")

        if not self.valid_username or not self.valid_password:
            raise RuntimeError(
                "AUTH_USER and AUTH_PASS environment variables must be set."
            )

        # Email config (use Gmail SMTP)
        self.smtp_user = os.environ.get("SMTP_USER")   # e.g. your Gmail
        self.smtp_pass = os.environ.get("SMTP_PASS")   # Gmail App Password
        self.otp_recipient = os.environ.get("OTP_MAIL_TO", self.smtp_user)

        if not self.smtp_user or not self.smtp_pass:
            self.get_logger().warn(
                "SMTP_USER or SMTP_PASS not set. OTP email sending will fail. "
                "Set them via environment variables."
            )

        # Subscriptions
        self.sub_login = self.create_subscription(
            String, 'login_request', self.cb_login_request, 10
        )
        self.sub_otp = self.create_subscription(
            String, 'otp_input', self.cb_otp_input, 10
        )

        # Publisher
        self.pub_auth_state = self.create_publisher(
            String, 'auth_state', 10
        )

        self.pub_login_result = self.create_publisher(
            String, 'login_result', 10
        )

        # OTP & session state
        self.current_otp = None
        self.otp_expires = 0.0
        self.session_expires = 0.0

        # Timing configuration
        self.otp_valid_seconds = 180       # OTP valid for 3 minutes
        self.session_valid_seconds = 600   # Session valid for 10 minutes

    # ---------- Helper: send email ----------

    def send_otp_email(self, to_email: str, otp: str) -> bool:
        """
        Send the OTP via email using Gmail SMTP.
        """
        if not self.smtp_user or not self.smtp_pass:
            self.get_logger().error("SMTP_USER/SMTP_PASS not set. Cannot send OTP email.")
            return False

        try:
            msg = EmailMessage()
            msg["Subject"] = "Your Robot OTP Code"
            msg["From"] = self.smtp_user
            msg["To"] = to_email
            body = (
                f"Your OTP code is: {otp}\n\n"
                f"It is valid for {self.otp_valid_seconds} seconds.\n"
                "This code is for logging into the LLM-based robot controller."
            )
            msg.set_content(body)

            context = ssl.create_default_context()
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)

            self.get_logger().info(f"OTP email sent to {to_email}.")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to send OTP email: {e}")
            return False

    # ---------- Step 1: Username + Password ----------

    def cb_login_request(self, msg: String):
        """
        Handle login request containing username & password in JSON.
        Example:
          { "username": "<AUTH_USER>", "password": "<AUTH_PASS>" }
        """
        try:
            data = json.loads(msg.data)
            username = data.get("username", "")
            password = data.get("password", "")
        except Exception as e:
            self.get_logger().error(f"Failed to parse login_request: {e}")
            return

        if username == self.valid_username and password == self.valid_password:
            # Credentials correct → generate OTP
            otp = f"{random.randint(0, 999999):06d}"
            now = time.time()
            self.current_otp = otp
            self.otp_expires = now + self.otp_valid_seconds

            self.get_logger().info(
                f"Login OK for user '{username}'. Generated OTP (hidden). "
                f"Valid for {self.otp_valid_seconds} seconds."
            )

            # Tell the CLI to ask for the OTP right away — sending it
            # actually happens over Gmail SMTP below, which can take
            # several seconds (connect + TLS handshake + login + send).
            # Publishing first avoids the CLI's login_result wait timing
            # out before the (much slower) email send even starts.
            result = String()
            result.data = json.dumps({"status": "OTP_REQUIRED"})
            self.pub_login_result.publish(result)

            # Give rclpy a chance to actually flush this message out
            # before this callback blocks on the Gmail SMTP call.
            time.sleep(0.1)

            # Send OTP by email
            if not self.send_otp_email(self.otp_recipient, otp):
                self.get_logger().error(
                    "Could not send OTP email. The CLI has already been told "
                    "OTP_REQUIRED, but no email will arrive — the OTP entered "
                    "will simply fail to match since current_otp is cleared."
                )
                # Clear OTP on failure so any /otp_input received afterward
                # is correctly rejected as "no OTP generated".
                self.current_otp = None
                self.otp_expires = 0.0
        else:
            self.get_logger().warn(
                f"Login FAILED for user '{username}'. Invalid username or password."
            )

            result = String()
            result.data = json.dumps({"status": "LOGIN_FAILED"})
            self.pub_login_result.publish(result)

    # ---------- Step 2: OTP verification ----------

    def cb_otp_input(self, msg: String):
        """
        Handle OTP input from user via /otp_input.
        """
        if self.current_otp is None:
            self.get_logger().warn(
                "Received OTP input but no OTP has been generated. "
                "Send /login_request with correct credentials first."
            )
            return

        now = time.time()
        if now > self.otp_expires:
            self.get_logger().warn(
                "Received OTP input but OTP has expired. Please login again."
            )

            state = {
                "authenticated": False,
                "user": self.valid_username,
                "auth_method": "OTP",
                "trust_level": 0.0,
                "timestamp": int(time.time()),
                "session_expires": 0,
            }

            out = String()
            out.data = json.dumps(state)
            self.pub_auth_state.publish(out)

            self.current_otp = None
            self.otp_expires = 0.0
            return

        user_otp = msg.data.strip()
        if user_otp == self.current_otp:
            # OTP correct → issue session
            self.session_expires = now + self.session_valid_seconds
            self.get_logger().info(
                f"OTP correct. Session established for user '{self.valid_username}' "
                f"valid for {self.session_valid_seconds} seconds."
            )

            state = {
                "authenticated": True,
                "user": self.valid_username,
                "auth_method": "OTP",
                "trust_level": 1.0,
                "timestamp": int(time.time()),
                "session_expires": int(self.session_expires),
            }

            out = String()
            out.data = json.dumps(state)
            self.pub_auth_state.publish(out)

            # clear OTP (one-time use)
            self.current_otp = None
            self.otp_expires = 0.0
        else:
            self.get_logger().warn("Incorrect OTP entered. Login failed.")

            state = {
                "authenticated": False,
                "user": self.valid_username,
                "auth_method": "OTP",
                "trust_level": 0.0,
                "timestamp": int(time.time()),
                "session_expires": 0,
            }

            out = String()
            out.data = json.dumps(state)
            self.pub_auth_state.publish(out)

            self.current_otp = None
            self.otp_expires = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = OtpAuthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()