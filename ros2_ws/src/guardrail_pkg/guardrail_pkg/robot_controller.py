#!/usr/bin/env python3

# =============================================================
# robot_controller.py — v10.1 Real 3D Pick
#
# Changes from v10:
#   1. object_3d_callback saves "saved_3d" when close
#   2. _get_pick_coordinates uses saved_3d with drive adjust
#   3. Coast transition initializes coast_distance
#   4. handle_pick_and_place initializes coast_distance
# =============================================================

import copy
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from linkattacher_msgs.srv import AttachLink

from guardrail_pkg.moveit_pick_place import MoveItPickPlace

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


# ─── SYNONYMS ───────────────────────────────────────────────
OBJECT_SYNONYMS = {
    "bottle": [
        "bottle",
        "water bottle"
    ],

    "thermo flask": [
        "thermo flask",
        "thermo_flask",
        "flask",
        "thermos",
        "thermo"
    ],

    "apple": [
        "apple"
    ],

    "knife": [
        "knife",
        "blade"
    ],

    "cup": [
        "cup",
        "mug",
        "glass"
    ],

    "Medicine": [
        "medicine",
        "Medicine",
        "pill",
        "tablet",
        "drug",
        "medicine bottle"
    ],

    "syringe": [
        "syringe",
        "needle",
        "injection"
    ]
}

SKIP_WORDS = {
    "the", "a", "an", "this", "that", "some", "my",
    "go", "to", "and", "bring", "get", "fetch", "pick",
    "up", "from", "please", "can", "you", "room",
    "kitchen", "bedroom", "office", "in", "on", "at"
}

# ─── LOCATIONS ──────────────────────────────────────────────
ROOM_LOCATIONS = {
    "tv room":   ( 5.0510,  4.2169),
    "kitchen":   (-5.3457,  7.1929),
    "room 2":    (-3.0568,  1.8437),
    "room 1":    (-3.0194, -5.7493),
    "reception": ( 5.2180, -5.6051)
}

ROOM_ORIENTATIONS = {
    "tv room":   1.57,
    "kitchen":   1.57,
    "room 2":    0.0,
    "room 1":    -1.57,
    "reception": 0.0
}

ROOM_SEARCH_POINTS = {
    "kitchen": [
        (-5.3, 7.1),
        (-4.8, 6.5),
        (-5.8, 6.8),
        (-5.0, 7.5)
    ],
    "room 1": [
        (-3.0, -5.7),
        (-2.5, -5.2),
        (-3.5, -6.0)
    ],
    "room 2": [
        (-3.0,  1.8),
        (-2.5,  1.3),
        (-3.5,  2.3)
    ],
    "tv room": [
        (5.0, 4.2),
        (4.5, 3.8),
        (5.5, 4.6)
    ],
    "reception": [
        ( 5.2, -5.6),
        ( 4.8, -5.1),
        ( 5.6, -6.0)
    ]
}

# ─── VISION ─────────────────────────────────────────────────
IMAGE_CENTER_X    = 320
RGB_W, RGB_H      = 640, 480
EDGE_MARGIN       = 15

# ─── ALIGNMENT ──────────────────────────────────────────────
ALIGN_TOLERANCE    = 40
ALIGN_SPEED        = 0.25
REALIGN_THRESHOLD  = 50
ALIGN_LOST_TIMEOUT = 10.0

# ─── APPROACH ───────────────────────────────────────────────
APPROACH_SPEED     = 0.04
APPROACH_DISTANCE  = 0.15
APPROACH_TOL       = 0.03

BBOX_STOP_WIDTH    = 200
BBOX_PICK_WIDTH    = 250

# ─── 3D QUALITY ─────────────────────────────────────────────
STALE_3D_TIMEOUT   = 2.0
LOST_TIMEOUT       = 10.0
CONFIRM_COUNT      = 3
CONFIRM_SPREAD     = 0.08
CONFIRM_FRESH_SEC  = 3.0

# ─── PICK ───────────────────────────────────────────────────
FIXED_PICK_X       = 0.15
FIXED_PICK_Y       = 0.00
FIXED_PICK_Z       = 0.10

USE_MOVEIT         = True
MAX_PICK_RETRIES   = 3
NUDGE_FORWARD_DIST = 0.03
NUDGE_DURATION     = 1.5

# ─── COAST ──────────────────────────────────────────────────
COAST_TIME         = 4.0

# ─── REACTIVE ──────────────────────────────────────────────
REACTIVE_DEPTH_THRESHOLD = 1.5
REACTIVE_MIN_CONF        = 0.65
REACTIVE_CONFIRM_FRAMES  = 3

# ─── BACKUP ────────────────────────────────────────────────
BACKUP_DURATION          = 2.0


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # ─── Subscriptions ──────────────────────────────
        self.subscription = self.create_subscription(
            String, 'executor_cmd',
            self.command_callback, 10)

        self.sub_detect = self.create_subscription(
            String, 'detected_object',
            self.detect_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        self.sub_3d_pose = self.create_subscription(
            PoseStamped, '/object_3d_pose',
            self.object_3d_callback, 10)

        self.sub_3d_info = self.create_subscription(
            String, '/object_3d_info',
            self.object_3d_info_callback, 10)

        self.sub_quick_depth = self.create_subscription(
            String, '/object_quick_depth',
            self.quick_depth_callback, 10)

        # ─── Publishers ─────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        self.action_cb_group = ReentrantCallbackGroup()

        # ─── MoveIt ─────────────────────────────────────
        self.pick_place = MoveItPickPlace(
            self, cb_group=self.action_cb_group)

        # ─── IFRA Link Attacher ─────────────────────────
        self.attach_cli = self.create_client(
            AttachLink,
            '/ATTACHLINK'
        )


        while not self.attach_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                'Waiting for ATTACHLINK service...')



        # ─── State ──────────────────────────────────────
        self.current_pose    = None
        self.start_pose      = None
        self.current_task    = None
        self.search_index    = 0
        self.nav_busy        = False
        self.nav_goal_handle = None
        self.task_locked     = False

        # ─── Executor plan state ────────────────────────
        self.current_command  = ""
        self.task_plan        = []
        self.current_step     = 0
        self.plan_active      = False
        self.awaiting_plan_nav = False

        # ─── 3D / Depth ────────────────────────────────
        self.object_3d_pos   = None
        self.object_depth    = None
        self.quick_depth     = None
        self.confirm_buffer  = []

        # ─── Reactive ──────────────────────────────────
        self.reactive_hit_count = 0

        # ─── Timer ──────────────────────────────────────
        self.timer = self.create_timer(
            0.3, self.task_executor,
            callback_group=self.action_cb_group)

        self.get_logger().info("=" * 60)
        self.get_logger().info(
            "Robot Controller v10.1 — Real 3D Pick ✅")
        self.get_logger().info(
            f"  Approach speed   : "
            f"{APPROACH_SPEED}m/s")
        self.get_logger().info(
            f"  Bbox stop width  : "
            f"{BBOX_STOP_WIDTH}px")
        self.get_logger().info(
            f"  Fixed pick pos   : "
            f"({FIXED_PICK_X}, {FIXED_PICK_Y}, "
            f"{FIXED_PICK_Z})")
        self.get_logger().info(
            f"  Coast time       : {COAST_TIME}s")
        self.get_logger().info(
            f"  Nudge per retry  : "
            f"{NUDGE_FORWARD_DIST}m/s × "
            f"{NUDGE_DURATION}s")
        self.get_logger().info("=" * 60)

    # =========================================================
    # CALLBACKS
    # =========================================================
    def odom_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().info(
                f"✅ Odom online → "
                f"x={msg.pose.pose.position.x:.3f}")
        self.current_pose = msg.pose.pose

    # ═══ CHANGE 1: Save best 3D when object is close ════════
    def object_3d_callback(self, msg):
        pos = (msg.pose.position.x,
               msg.pose.position.y,
               msg.pose.position.z)
        self.object_3d_pos = pos

        if self.current_task is not None:
            now = time.time()
            self.current_task["object_3d"] = pos
            self.current_task["last_3d_time"] = now

            dist = math.sqrt(pos[0]**2 + pos[1]**2)
            self.current_task["last_known_dist"] = dist

            self.confirm_buffer.append(
                (pos[0], pos[1], pos[2], now))
            if len(self.confirm_buffer) > 10:
                self.confirm_buffer.pop(0)

            # ═══ NEW: Save 3D when within approach range
            # This survives even when YOLO loses the
            # object during close approach / coasting.
            if dist < 0.50:
                self.current_task["saved_3d"] = pos
                self.current_task["saved_3d_time"] = now
                self.current_task["saved_3d_dist"] = dist
                self.get_logger().info(
                    f"  💾 Saved 3D: "
                    f"({pos[0]:.3f}, {pos[1]:.3f}, "
                    f"{pos[2]:.3f}) "
                    f"dist={dist:.3f}m")

    def object_3d_info_callback(self, msg):
        parts = msg.data.split()
        if len(parts) >= 5:
            try:
                self.object_depth = float(parts[4])
            except ValueError:
                pass

    def quick_depth_callback(self, msg):
        parts = msg.data.split()
        if len(parts) >= 2:
            try:
                self.quick_depth = float(parts[1])
            except ValueError:
                pass

    def command_callback(self, msg: String):
        try:
            plan = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Invalid executor plan: {e}")
            return

        decision = plan.get("decision", "DENY")

        if decision != "ALLOW":
            self.get_logger().warn("Task rejected.")
            return

        self.current_command = plan.get("command", "")

        self.task_plan = plan.get("task_plan", [])

        self.current_step = 0

        self.get_logger().info(
            f"Received plan for: {self.current_command}"
        )

        self.execute_next_step()

    # =========================================================
    # EXECUTOR PLAN STEP DISPATCH
    # =========================================================
    def execute_next_step(self):

        if self.current_step >= len(self.task_plan):
            self.get_logger().info("Task completed.")
            return

        step = self.task_plan[self.current_step]

        if not isinstance(step, dict):
            self.get_logger().warn(
                f"Unexpected step format (not a dict): {step!r} "
                f"— skipping")
            self.current_step += 1
            self.execute_next_step()
            return

        self.get_logger().info(f"Executing: {step}")

        action = step.get("action", "").strip().lower()

        if action == "navigate":
            self.navigate_step(step)

        elif action == "detect":
            self.detect_step(step)

        elif action == "pick":
            self.pick_step(step)

        elif action == "place":
            self.place_step(step)

        else:
            self.get_logger().warn(f"Unknown action: {action!r} in step {step}")
            self.current_step += 1
            self.execute_next_step()

    # =========================================================
    # PLAN STEPS — Navigate / Detect / Place are absorbed into
    # the existing pick pipeline, so they just log and advance.
    # Pick is the only step that actually drives the robot, using
    # self.current_command (the full natural-language command,
    # which already carries both room and object context) — not
    # the individual step dict — so handle_pick_and_place()'s
    # room-substring matching keeps working exactly as before.
    # =========================================================
    def navigate_step(self, step):
        location = step.get("location", "").strip().lower()

        coords = ROOM_LOCATIONS.get(location)
        if coords is None:
            self.get_logger().warn(
                f"⚠ Unknown navigate location '{location}' "
                f"— skipping step")
            self.current_step += 1
            self.execute_next_step()
            return

        x, y = coords
        yaw = ROOM_ORIENTATIONS.get(location, 0.0)

        self.get_logger().info(
            f"🧭 Plan-driven NAVIGATE step -> '{location}' "
            f"({x}, {y})")

        # awaiting_plan_nav is a dedicated flag, separate from
        # plan_active. handle_pick_and_place()'s internal chain also
        # calls send_navigation_goal() repeatedly (initial room nav,
        # search waypoints, return-to-start) — all of those funnel
        # through the same _on_nav_result() callback. This flag is
        # only ever set True here, so Pick's internal navigation never
        # triggers a plan-step advance; only a standalone Navigate
        # step (like a bare "Go to kitchen" command with no Pick step
        # in its plan) does.
        self.awaiting_plan_nav = True
        self.send_navigation_goal(x, y, yaw=yaw)
        # current_step is NOT incremented here — _on_nav_result
        # advances it once this specific goal completes.

    def detect_step(self, step):
        action = step.get("action", "unknown")
        self.get_logger().info(
            f"⏭ '{action}' handled internally by Pick — skipping")
        self.current_step += 1
        self.execute_next_step()

    def pick_step(self, step):
        if self.current_task is not None:
            self.get_logger().warn(
                f"⚠ Pick step '{step}' ignored — "
                f"a task is already in progress")
            return

        self.get_logger().info(
            f"🔧 Plan-driven PICK step: {step} "
            f"(command='{self.current_command}')")

        self.plan_active = True
        self.handle_pick_and_place(
            self.current_command.lower())

    def place_step(self, step):
        action = step.get("action", "unknown")
        self.get_logger().info(
            f"⏭ '{action}' handled internally by Pick — skipping")
        self.current_step += 1
        self.execute_next_step()

    def _finish_plan_pick(self):
        """Called once the pick pipeline reaches a terminal
        state (placed, or not_found) while a plan-driven pick
        step is in progress, so the task_plan can advance."""
        if self.plan_active:
            self.plan_active = False
            self.current_step += 1
            self.execute_next_step()

    # =========================================================
    # DETECTION CALLBACK
    # =========================================================
    def detect_callback(self, msg):

        if self.current_task is None:
            return

        stage = self.current_task["stage"]

        ACTIVE_STAGES = [
            "navigating_to_object",
            "searching",
            "aligning",
            "confirming",
            "approaching",
            "coasting"
        ]
        if stage not in ACTIVE_STAGES:
            return

        data = msg.data.strip()
        parts = data.split()
        if len(parts) < 6:
            return

        if parts[0] == "none":
            return

        try:
            conf = float(parts[-1])
            y2   = int(parts[-2])
            x2   = int(parts[-3])
            y1   = int(parts[-4])
            x1   = int(parts[-5])
            detected = " ".join(
                parts[:-5]).lower().strip()
        except (ValueError, IndexError):
            return

        target = self.current_task[
            "object"].lower().strip()
        allowed = OBJECT_SYNONYMS.get(
            target, [target])

        if not any(detected in a or a in detected
                   for a in allowed):
            self.reactive_hit_count = 0
            return

        now = time.time()

        self.current_task["last_seen"] = now
        self.current_task["bbox_center"] = (
            (x1 + x2) / 2)
        bbox_w = x2 - x1
        self.current_task["bbox_width"] = bbox_w
        self.current_task["bbox"] = (
            x1, y1, x2, y2)

        is_clipped = (
            x1 < EDGE_MARGIN or
            y1 < EDGE_MARGIN or
            x2 > RGB_W - EDGE_MARGIN or
            y2 > RGB_H - EDGE_MARGIN)
        self.current_task["is_clipped"] = is_clipped

        self.get_logger().info(
            f"👁 {detected} c={conf:.2f} "
            f"cx={(x1+x2)/2:.0f} "
            f"w={bbox_w} "
            f"{'CLIP' if is_clipped else 'ok'} "
            f"stage={stage}")

        # ═══ REACTIVE — during navigation ═══════════════
        if stage == "navigating_to_object":
            if conf < REACTIVE_MIN_CONF:
                self.reactive_hit_count = 0
                return

            self.reactive_hit_count += 1

            self.get_logger().warn(
                f"🚨 REACTIVE: Bottle while navigating! "
                f"({self.reactive_hit_count}/"
                f"{REACTIVE_CONFIRM_FRAMES}) "
                f"conf={conf:.2f}")

            if (self.reactive_hit_count >=
                    REACTIVE_CONFIRM_FRAMES):

                current_depth = (
                    self.quick_depth or
                    self.object_depth)

                if current_depth is not None:
                    if (current_depth >
                            REACTIVE_DEPTH_THRESHOLD):
                        self.get_logger().info(
                            f"  Bottle at "
                            f"{current_depth:.2f}m "
                            f"— far, keep navigating")
                        return

                self.get_logger().warn(
                    "🛑 STOPPING — Bottle in path!")
                self.cancel_navigation()
                self.stop_robot()

                self.current_task["object_found"] = True
                self.current_task["stage"] = "aligning"
                self.reactive_hit_count = 0

                self.get_logger().info(
                    "✅ REACTIVE → aligning")
            return

        # ═══ Re-detected during coasting ════════════════
        if stage == "coasting":
            self.get_logger().info(
                "👁 Re-detected during coast "
                "→ approach")
            self.current_task["stage"] = "approaching"
            return

        # ═══ NORMAL detection ═══════════════════════════
        if not self.current_task.get(
                "object_found", False):
            self.get_logger().info(
                "✅ FOUND → aligning")
            self.cancel_navigation()
            self.stop_robot()
            self.current_task["object_found"] = True
            self.current_task["stage"] = "aligning"

    # =========================================================
    # HELPERS
    # =========================================================
    def extract_object(self, command):
        words = command.lower().split()
        triggers = ["bring", "fetch", "get",
                     "pick", "carry"]
        trigger_idx = -1
        for t in triggers:
            if t in words:
                trigger_idx = words.index(t)
                break
        if trigger_idx >= 0:
            for i in range(
                    trigger_idx + 1, len(words)):
                if words[i] not in SKIP_WORDS:
                    return words[i]
        for obj in OBJECT_SYNONYMS:
            if obj in command:
                return obj
        return "bottle"

    def handle_navigation(self, command):
        for room, coords in ROOM_LOCATIONS.items():
            if room in command:
                x, y = coords
                yaw = ROOM_ORIENTATIONS.get(room, 0.0)
                self.send_navigation_goal(
                    x, y, yaw=yaw)
                return

    # ═══ CHANGE 4: coast_distance initialized ═══════════════
    def handle_pick_and_place(self, command):
        target_object = self.extract_object(command)
        self.get_logger().info(
            f"🎯 Target: {target_object}")

        for room, coords in ROOM_LOCATIONS.items():
            if room in command:
                x, y = coords
                yaw = ROOM_ORIENTATIONS.get(
                    room, 0.0)
                self.get_logger().info(
                    f"Step 1 → {room}")

                self.current_task = {
                    "type":          "pick",
                    "room":          room,
                    "object":        target_object,
                    "stage":
                        "navigating_to_object",
                    "object_found":  False,
                    "pick_attempts": 0,
                    "last_known_dist": None,
                    "coast_distance": 0.0,
                }
                self.search_index = 0
                self.confirm_buffer = []
                self.reactive_hit_count = 0
                self.quick_depth = None
                self.send_navigation_goal(
                    x, y, yaw=yaw)
                return

        self.get_logger().warn("⚠ No room found")
        self._finish_plan_pick()

    def stop_robot(self):
        twist = Twist()
        for _ in range(3):
            self.cmd_vel_pub.publish(twist)


    # =========================================================
    # IFRA ATTACH
    # =========================================================
    def attach_object(self, object_model):

        req = AttachLink.Request()

        req.model1_name = (
            'turtlebot3_manipulation_system')
        req.link1_name = 'gripper_right_link'

        req.model2_name = object_model
        req.link2_name = 'bottle_link'

        future = self.attach_cli.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(
            f'✅ ATTACHED: {object_model}')




    def cancel_navigation(self):
        if self.nav_goal_handle is not None:
            self.get_logger().info(
                "🚫 Cancelling Nav2 goal")
            try:
                self.nav_goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(
                    f"Cancel error: {e}")
            self.nav_goal_handle = None
        self.nav_busy = False

    # =========================================================
    # 3D STABILITY CHECK
    # =========================================================
    def _is_3d_stable(self):
        now = time.time()
        fresh = [p for p in self.confirm_buffer
                 if now - p[3] < CONFIRM_FRESH_SEC]

        if len(fresh) < CONFIRM_COUNT:
            self.get_logger().info(
                f"  3D buffer: {len(fresh)}/"
                f"{CONFIRM_COUNT} fresh")
            return False

        recent = fresh[-CONFIRM_COUNT:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]

        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)

        stable = (spread_x < CONFIRM_SPREAD and
                  spread_y < CONFIRM_SPREAD)

        if stable:
            avg_x = sum(
                p[0] for p in recent) / len(recent)
            avg_y = sum(
                p[1] for p in recent) / len(recent)
            avg_z = sum(
                p[2] for p in recent) / len(recent)
            self.current_task["stable_3d"] = (
                avg_x, avg_y, avg_z)

        self.get_logger().info(
            f"  3D spread: "
            f"({spread_x:.3f},{spread_y:.3f})"
            f" fresh={len(fresh)}"
            f" {'✅' if stable else '⏳'}")

        return stable

    # =========================================================
    # ═══ CHANGE 2: Real 3D pick coordinates ═════════════════
    # Priority:
    #   1. Fresh 3D (< 2s, within arm reach)
    #   2. Saved 3D (captured when close, adjusted for driving)
    #   3. Stable 3D (from confirm, adjusted for driving)
    #   4. Last object_3d (adjusted for driving)
    #   5. FIXED fallback (emergency only)
    # =========================================================
    def _get_pick_coordinates(self):
        now = time.time()
        obj_3d = self.current_task.get("object_3d")
        last_3d_t = self.current_task.get(
            "last_3d_time", 0)
        coast_dist = self.current_task.get(
            "coast_distance", 0.0)

        # Calculate total drive distance since last 3D
        nudge_dist = self.current_task.get(
            "pick_attempts", 0) * (
            NUDGE_FORWARD_DIST * NUDGE_DURATION)
        total_driven = coast_dist + nudge_dist

        # ─── 1. FRESH 3D (< 2s old) ────────────────────
        if (obj_3d is not None and
                now - last_3d_t < 2.0):
            ox, oy, oz = obj_3d
            arm_x = ox + 0.092
            arm_h = math.sqrt(arm_x**2 + oy**2)

            if arm_h <= 0.28:
                self.get_logger().info(
                    f"  📍 FRESH 3D: "
                    f"({ox:.3f}, {oy:.3f}, {oz:.3f})"
                    f" arm_h={arm_h:.3f}m ✅")
                return ox, oy, oz
            else:
                self.get_logger().warn(
                    f"  ⚠ Fresh 3D too far: "
                    f"arm_h={arm_h:.3f}m")

        # ─── 2. SAVED 3D (from close range) ────────────
        saved = self.current_task.get("saved_3d")
        saved_t = self.current_task.get(
            "saved_3d_time", 0)

        if saved is not None:
            age = now - saved_t

            # Adjust for distance robot drove since save
            drive_since_save = 0.0
            if age > 0.5:
                drive_since_save = total_driven

            adj_x = saved[0] - drive_since_save
            adj_y = saved[1]
            adj_z = saved[2]

            arm_x = adj_x + 0.092
            arm_h = math.sqrt(arm_x**2 + adj_y**2)

            if 0.05 <= arm_h <= 0.28:
                self.get_logger().info(
                    f"  📍 SAVED 3D (age={age:.1f}s "
                    f"drove={drive_since_save:.3f}m): "
                    f"({adj_x:.3f}, {adj_y:.3f}, "
                    f"{adj_z:.3f}) "
                    f"arm_h={arm_h:.3f}m ✅")
                return adj_x, adj_y, adj_z
            else:
                self.get_logger().warn(
                    f"  ⚠ Saved 3D out of range: "
                    f"arm_h={arm_h:.3f}m "
                    f"(drove={drive_since_save:.3f}m)")

        # ─── 3. STABLE 3D — adjusted ───────────────────
        stable = self.current_task.get("stable_3d")
        if stable is not None:
            adj_x = stable[0] - total_driven
            adj_y = stable[1]
            adj_z = stable[2]
            arm_x = adj_x + 0.092
            arm_h = math.sqrt(arm_x**2 + adj_y**2)

            if 0.05 <= arm_h <= 0.28:
                self.get_logger().info(
                    f"  📍 ADJUSTED stable_3d: "
                    f"({adj_x:.3f}, {adj_y:.3f}, "
                    f"{adj_z:.3f})"
                    f" arm_h={arm_h:.3f}m "
                    f"(drove {total_driven:.3f}m) ✅")
                return adj_x, adj_y, adj_z

        # ─── 4. LAST object_3d — adjusted ──────────────
        if obj_3d is not None:
            adj_x = obj_3d[0] - total_driven
            adj_y = obj_3d[1]
            adj_z = obj_3d[2]
            arm_x = adj_x + 0.092
            arm_h = math.sqrt(arm_x**2 + adj_y**2)

            if 0.05 <= arm_h <= 0.28:
                self.get_logger().info(
                    f"  📍 ADJUSTED last_3d: "
                    f"({adj_x:.3f}, {adj_y:.3f}, "
                    f"{adj_z:.3f})"
                    f" arm_h={arm_h:.3f}m ✅")
                return adj_x, adj_y, adj_z

        # ─── 5. FIXED fallback (emergency) ─────────────
        self.get_logger().warn(
            f"  📍 FIXED fallback: "
            f"({FIXED_PICK_X}, {FIXED_PICK_Y}, "
            f"{FIXED_PICK_Z}) "
            f"⚠ No real 3D available!")
        return FIXED_PICK_X, FIXED_PICK_Y, FIXED_PICK_Z

    # =========================================================
    # TASK EXECUTOR
    # =========================================================
    def task_executor(self):

        if self.current_task is None:
            return
        if self.task_locked:
            return
        if self.nav_busy:
            return

        stage = self.current_task["stage"]
        self.get_logger().info(f"▶ {stage}")

        if stage == "navigating_to_object":
            if self.current_task.get("object_found"):
                self.get_logger().info(
                    "📍 Found reactively → skip")
                return
            self.get_logger().info(
                "📍 Arrived → searching...")
            self.current_task["stage"] = "searching"
            self.search_index = 0
            self._send_next_search_waypoint()

        elif stage == "searching":
            if self.current_task.get("object_found"):
                return
            self._send_next_search_waypoint()

        elif stage == "aligning":
            self._do_aligning()

        elif stage == "confirming":
            self._do_confirming()

        elif stage == "approaching":
            self._do_approaching()

        elif stage == "coasting":
            self._do_coasting()

        elif stage == "picking":
            self._do_picking()

        elif stage == "nudging":
            self._do_nudging()

        elif stage == "backup_before_realign":
            self._do_backup_before_realign()

        elif stage == "returning":
            self.get_logger().info(
                "✅ Arrived → placing")
            self.current_task["stage"] = "placing"

        elif stage == "placing":
            self._do_placing()

        elif stage == "not_found":
            self.get_logger().warn(
                "Object NOT found.")
            self.current_task = None
            self._finish_plan_pick()

    # =========================================================
    # STAGE: ALIGNING
    # =========================================================
    def _do_aligning(self):

        last_seen = self.current_task.get(
            "last_seen", 0)
        time_since_seen = time.time() - last_seen

        if time_since_seen > ALIGN_LOST_TIMEOUT:

            last_w = self.current_task.get(
                "bbox_width", 0)
            if last_w >= BBOX_STOP_WIDTH:
                self.get_logger().warn(
                    f"⚠ Lost but last bbox w={last_w}"
                    f" → try picking")
                self.stop_robot()
                self.current_task["stage"] = "picking"
                return

            self.get_logger().warn(
                "⚠ Lost too long → searching")
            self.stop_robot()
            self.current_task["object_found"] = False
            self.current_task["stage"] = "searching"
            self.reactive_hit_count = 0
            return

        if "bbox_center" not in self.current_task:
            self.stop_robot()
            return

        center_x = self.current_task["bbox_center"]
        error = center_x - IMAGE_CENTER_X

        self.get_logger().info(
            f"  Align: cx={center_x:.0f} "
            f"err={error:.0f}px "
            f"seen={time_since_seen:.1f}s ago")

        if abs(error) <= ALIGN_TOLERANCE:
            self.stop_robot()

            if time_since_seen > LOST_TIMEOUT:
                self.get_logger().warn(
                    f"  Centered but stale "
                    f"({time_since_seen:.1f}s) "
                    f"— waiting for fresh detection")
                return

            if not self.current_task.get(
                    "is_clipped", True):
                self.get_logger().info(
                    "✅ Centered + clean → confirming")

                now = time.time()
                self.confirm_buffer = [
                    p for p in self.confirm_buffer
                    if now - p[3] < CONFIRM_FRESH_SEC
                ]

                self.current_task["stage"] = (
                    "confirming")
                self.current_task["confirm_start"] = (
                    time.time())
            else:
                self.get_logger().info(
                    "Centered but clipped — nudging")
                twist = Twist()
                twist.angular.z = -0.08
                self.cmd_vel_pub.publish(twist)
            return

        if error < 0:
            self.current_task["last_align_dir"] = 1.0
        else:
            self.current_task["last_align_dir"] = -1.0

        twist = Twist()

        if time_since_seen < 2.0:
            speed = min(ALIGN_SPEED,
                        abs(error) * 0.0008 + 0.08)
            twist.angular.z = speed * (
                self.current_task["last_align_dir"])
            self.get_logger().info(
                f"  Rotating: {twist.angular.z:.2f}")
        else:
            direction = self.current_task.get(
                "last_align_dir", 1.0)
            twist.angular.z = 0.20 * direction
            self.get_logger().info(
                f"  Blind rotate: "
                f"{twist.angular.z:.2f}"
                f" (lost {time_since_seen:.1f}s)")

        self.cmd_vel_pub.publish(twist)

    # =========================================================
    # STAGE: CONFIRMING
    # =========================================================
    def _do_confirming(self):

        self.stop_robot()

        last_seen = self.current_task.get(
            "last_seen", 0)
        if time.time() - last_seen > LOST_TIMEOUT:

            last_w = self.current_task.get(
                "bbox_width", 0)
            if last_w >= BBOX_STOP_WIDTH:
                self.get_logger().warn(
                    "⚠ Lost during confirm but "
                    f"w={last_w} → picking")
                self.current_task["stage"] = "picking"
                return

            self.get_logger().warn(
                "⚠ Lost during confirm → aligning")
            self.current_task["stage"] = "aligning"
            return

        confirm_start = self.current_task.get(
            "confirm_start", time.time())
        if time.time() - confirm_start > 8.0:
            self.get_logger().warn(
                "⚠ Confirm timeout → approach")
            self.current_task["stage"] = "approaching"
            return

        if "bbox_center" in self.current_task:
            cx = self.current_task["bbox_center"]
            drift = abs(cx - IMAGE_CENTER_X)
            if drift > ALIGN_TOLERANCE + 20:
                self.get_logger().warn(
                    f"  Drifted {drift:.0f}px "
                    f"→ re-align")
                self.current_task["stage"] = "aligning"
                return

        if self._is_3d_stable():
            stable = self.current_task["stable_3d"]
            self.get_logger().info(
                f"✅ 3D CONFIRMED: "
                f"x={stable[0]:.3f} "
                f"y={stable[1]:.3f} "
                f"z={stable[2]:.3f}")
            self.current_task["stage"] = "approaching"

    # =========================================================
    # STAGE: APPROACHING
    # =========================================================
    def _do_approaching(self):

        last_seen = self.current_task.get(
            "last_seen", 0)
        time_since = time.time() - last_seen

        # ─── Object lost? ───────────────────────────────
        if time_since > LOST_TIMEOUT:

            last_w = self.current_task.get(
                "bbox_width", 0)

            if last_w >= BBOX_STOP_WIDTH:
                self.get_logger().warn(
                    f"⚠ Lost but w={last_w}px "
                    f"→ object is close → PICKING")
                self.stop_robot()
                self.current_task["stage"] = "picking"
                return

            # ═══ CHANGE 3: Coast with distance tracking ═
            if last_w >= 80:
                last_dist = self.current_task.get(
                    "last_known_dist", 0.35)
                target_dist = 0.15
                gap = max(
                    last_dist - target_dist, 0.05)
                coast_speed = 0.025
                coast_dur = min(
                    gap / coast_speed, 4.0)

                self.get_logger().warn(
                    f"⚠ Lost during approach "
                    f"(w={last_w}px) "
                    f"last_dist={last_dist:.2f}m "
                    f"gap={gap:.2f}m "
                    f"→ coast {coast_dur:.1f}s")
                self.stop_robot()
                self.current_task["coast_start"] = (
                    time.time())
                self.current_task["coast_duration"] = (
                    coast_dur)
                self.current_task["coast_distance"] = (
                    0.0)
                self.current_task["stage"] = "coasting"
                return

            self.get_logger().warn(
                "⚠ Lost during approach → aligning")
            self.stop_robot()
            self.current_task["stage"] = "aligning"
            return

        # ─── Check bbox width ───────────────────────────
        bbox_w = self.current_task.get("bbox_width", 0)

        if bbox_w >= BBOX_PICK_WIDTH:
            self.get_logger().info(
                f"✅ bbox w={bbox_w}px ≥ "
                f"{BBOX_PICK_WIDTH} → PICKING!")
            self.stop_robot()
            self.current_task["stage"] = "picking"
            return

        # ─── Check drift ────────────────────────────────
        if "bbox_center" in self.current_task:
            cx = self.current_task["bbox_center"]
            drift = abs(cx - IMAGE_CENTER_X)
            if drift > REALIGN_THRESHOLD:
                if bbox_w < BBOX_STOP_WIDTH:
                    self.get_logger().warn(
                        f"⚠ Drift {drift:.0f}px "
                        f"→ re-align")
                    self.stop_robot()
                    self.current_task["stage"] = (
                        "aligning")
                    return
                else:
                    self.get_logger().info(
                        f"  Drift {drift:.0f}px "
                        f"but close — continuing")

        # ─── Try 3D approach ────────────────────────────
        last_3d = self.current_task.get(
            "last_3d_time", 0)
        obj_3d = self.current_task.get("object_3d")
        use_3d = (obj_3d is not None and
                  time.time() - last_3d < STALE_3D_TIMEOUT)

        if use_3d:
            ox, oy, oz = obj_3d
            dist = math.sqrt(ox**2 + oy**2)

            self.get_logger().info(
                f"  Approach 3D: d={dist:.3f}m "
                f"w={bbox_w}px")

            twist = Twist()

            if dist > APPROACH_DISTANCE + APPROACH_TOL:
                speed = min(APPROACH_SPEED,
                            dist * 0.12 + 0.02)
                twist.linear.x = speed
                if abs(oy) > 0.02:
                    twist.angular.z = -oy * 0.4
                self.cmd_vel_pub.publish(twist)

            elif dist < (APPROACH_DISTANCE -
                         APPROACH_TOL):
                twist.linear.x = -0.03
                self.cmd_vel_pub.publish(twist)

            else:
                self.stop_robot()
                self.get_logger().info(
                    f"✅ 3D: IN POSITION "
                    f"d={dist:.3f}m → PICKING!")
                self.current_task["stage"] = "picking"
            return

        # ─── Fallback: bbox approach ────────────────────
        self.get_logger().info(
            f"  Approach bbox: w={bbox_w}px "
            f"target={BBOX_STOP_WIDTH}px")

        if bbox_w < BBOX_STOP_WIDTH:
            twist = Twist()
            twist.linear.x = APPROACH_SPEED
            self.cmd_vel_pub.publish(twist)
        else:
            self.stop_robot()
            self.get_logger().info(
                f"✅ bbox w={bbox_w}px ≥ "
                f"{BBOX_STOP_WIDTH} → PICKING!")
            self.current_task["stage"] = "picking"

    # =========================================================
    # STAGE: COASTING
    # =========================================================
    def _do_coasting(self):
        elapsed = time.time() - self.current_task.get(
            "coast_start", time.time())

        coast_duration = self.current_task.get(
            "coast_duration", COAST_TIME)

        coast_speed = 0.025
        distance_driven = coast_speed * elapsed
        self.current_task["coast_distance"] = (
            distance_driven)

        if elapsed >= coast_duration:
            self.stop_robot()
            self.get_logger().info(
                f"✅ Coast done ({elapsed:.1f}s, "
                f"drove {distance_driven:.3f}m) "
                f"→ PICKING")
            self.current_task["stage"] = "picking"
            return

        twist = Twist()
        twist.linear.x = coast_speed
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f"  Coasting: {elapsed:.1f}/"
            f"{coast_duration:.1f}s "
            f"driven={distance_driven:.3f}m")

    # =========================================================
    # STAGE: PICKING
    # =========================================================
    def _do_picking(self):

        self.task_locked = True

        try:
            attempts = self.current_task.get(
                "pick_attempts", 0)

            ox, oy, oz = self._get_pick_coordinates()

            self.get_logger().info(
                f"🤖 PICK attempt {attempts + 1}/"
                f"{MAX_PICK_RETRIES} "
                f"at ({ox:.3f},{oy:.3f},{oz:.3f})")

            reachable, reason, _ = (
                self.pick_place.is_reachable(
                    ox, oy, oz))

            if not reachable:
                self.get_logger().error(
                    f"❌ UNREACHABLE: {reason}")

                if attempts < MAX_PICK_RETRIES:
                    if "Too far" in reason:
                        self.get_logger().warn(
                            f"🔄 Nudging forward "
                            f"(attempt {attempts+1})")
                        self.task_locked = False
                        self.current_task[
                            "pick_attempts"] = (
                            attempts + 1)
                        self.current_task[
                            "nudge_start"] = (
                            time.time())
                        self.current_task["stage"] = (
                            "nudging")
                        return

                    elif "Too close" in reason:
                        self.get_logger().warn(
                            "🔄 Too close — backing up")
                        self.task_locked = False
                        self.current_task[
                            "pick_attempts"] = (
                            attempts + 1)
                        self.current_task[
                            "backup_start"] = (
                            time.time())
                        self.current_task["stage"] = (
                            "backup_before_realign")
                        return

                self.get_logger().error(
                    "❌ All retries failed "
                    "→ emergency pick")
                self._emergency_pick()
                self.current_task["stage"] = (
                    "returning")
                self._return_to_start()
                return

            # ─── REACHABLE — DO THE PICK ────────────────
            self.get_logger().info(
                f"✅ REACHABLE: {reason}")

            success = self.pick_place.pick_object(
                ox, oy, oz, use_moveit=USE_MOVEIT)

            if success:

                # ─── IMPORTANT ───────────────────────────
                # Wait briefly after gripper close
                # before virtual attachment.
                time.sleep(1.0)

                # ─── IFRA Attach ─────────────────────────
                self.attach_object(
                    'bottle_red_wine_pick')

                self.get_logger().info(
                    "✅ PICK SUCCESS! → returning")

                self.current_task["stage"] = (
                    "returning")

                self._return_to_start()
            else:
                attempts += 1
                self.current_task[
                    "pick_attempts"] = attempts

                if attempts < MAX_PICK_RETRIES:
                    self.get_logger().warn(
                        f"❌ Pick failed, retry "
                        f"{attempts}")
                    self.current_task[
                        "nudge_start"] = time.time()
                    self.current_task["stage"] = (
                        "nudging")
                else:
                    self.get_logger().error(
                        "❌ PICK FAILED → emergency")
                    self._emergency_pick()
                    self.current_task["stage"] = (
                        "returning")
                    self._return_to_start()

        finally:
            self.task_locked = False

    # =========================================================
    # STAGE: NUDGING
    # =========================================================
    def _do_nudging(self):

        elapsed = time.time() - self.current_task.get(
            "nudge_start", time.time())

        if elapsed >= NUDGE_DURATION:
            self.stop_robot()
            self.get_logger().info(
                f"✅ Nudge done → retry pick")
            self.current_task["stage"] = "picking"
            return

        twist = Twist()
        twist.linear.x = NUDGE_FORWARD_DIST
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f"  Nudging: {elapsed:.1f}/"
            f"{NUDGE_DURATION:.1f}s")

    # =========================================================
    # STAGE: BACKUP BEFORE REALIGN
    # =========================================================
    def _do_backup_before_realign(self):
        elapsed = time.time() - self.current_task.get(
            "backup_start", time.time())

        if elapsed >= BACKUP_DURATION:
            self.stop_robot()
            self.get_logger().info(
                "✅ Backup done → re-aligning")
            self.current_task["stage"] = "aligning"
            self.confirm_buffer = []
        else:
            twist = Twist()
            twist.linear.x = -0.05
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info(
                f"  Backing up: {elapsed:.1f}/"
                f"{BACKUP_DURATION:.1f}s")

    # =========================================================
    # STAGE: PLACING
    # =========================================================
    def _do_placing(self):

        self.task_locked = True

        try:
            self.get_logger().info("📦 PLACING...")
            success = self.pick_place.place_object(
                height=0.05, use_moveit=USE_MOVEIT)

            if success:

                # ─── Open gripper first ──────────────────
                self.pick_place.open_gripper()

                time.sleep(0.5)



                self.get_logger().info(
                    "✅ TASK COMPLETE! 🎉")
            else:
                self.get_logger().warn(
                    "⚠ Place failed — opening gripper")
                self.pick_place.open_gripper()
                self.pick_place.go_home()
        finally:
            self.task_locked = False

        self.current_task = None
        self._finish_plan_pick()

    # =========================================================
    # EMERGENCY PICK
    # =========================================================
    def _emergency_pick(self):
        self.get_logger().warn("🚨 Emergency pick")
        try:
            self.pick_place.move_joints(
                [0.0, -0.5, 0.3, 0.2])
            self.pick_place.close_gripper()
            self.pick_place.move_joints(
                [0.0, -1.05, 0.35, 0.70])
        except Exception as e:
            self.get_logger().error(
                f"Emergency error: {e}")

    # =========================================================
    # SEARCH WAYPOINTS
    # =========================================================
    def _send_next_search_waypoint(self):

        room = self.current_task["room"]
        points = ROOM_SEARCH_POINTS.get(room, [])

        if self.search_index < len(points):
            x, y = points[self.search_index]
            self.get_logger().info(
                f"Search {self.search_index + 1}/"
                f"{len(points)}: ({x}, {y})")
            self.send_navigation_goal(x, y)
            self.search_index += 1
        else:
            self.get_logger().warn(
                "❌ All waypoints — not found")
            self.current_task["stage"] = "not_found"
            self._return_to_start()

    # =========================================================
    # RETURN TO START
    # =========================================================
    def _return_to_start(self):
        if self.start_pose is not None:
            self.get_logger().info(
                f"🏠 Returning → "
                f"x={self.start_pose.position.x:.3f} "
                f"y={self.start_pose.position.y:.3f}")
            self.send_pose_goal(self.start_pose)
        else:
            x, y = ROOM_LOCATIONS["reception"]
            self.send_navigation_goal(x, y)

    # =========================================================
    # NAVIGATION
    # =========================================================
    def send_pose_goal(self, pose):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.pose = pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = ps

        self.nav_client.wait_for_server()
        self.nav_busy = True
        future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav_feedback_cb)
        future.add_done_callback(
            self._nav_goal_response_cb)

    def send_navigation_goal(self, x, y, yaw=None):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.pose.position.x = x
        ps.pose.position.y = y

        if yaw is not None:
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
        else:
            ps.pose.orientation.w = 1.0

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = ps

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "❌ NavigateToPose action server not available")
            self.nav_busy = False

            if self.awaiting_plan_nav:
                self.awaiting_plan_nav = False
                self.current_step += 1
                self.execute_next_step()

            return

        self.nav_busy = True
        future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav_feedback_cb)
        future.add_done_callback(
            self._nav_goal_response_cb)

        yaw_deg = (math.degrees(yaw)
                   if yaw is not None else 0)
        self.get_logger().info(
            f"Nav → ({x:.2f}, {y:.2f}) "
            f"yaw={yaw_deg:.0f}°")

    def _nav_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(
                "❌ Nav REJECTED")
            self.nav_busy = False

            if self.awaiting_plan_nav:
                self.awaiting_plan_nav = False
                self.current_step += 1
                self.execute_next_step()

            return
        self.nav_goal_handle = goal_handle
        result_future = (
            goal_handle.get_result_async())
        result_future.add_done_callback(
            self._on_nav_result)

    def _on_nav_result(self, future):
        self.nav_busy = False
        self.nav_goal_handle = None
        status = future.result().status
        if status == 4:
            self.get_logger().info("✅ Nav completed")
        else:
            self.get_logger().warn(
                f"⚠ Nav status={status}")

        if self.awaiting_plan_nav:
            self.awaiting_plan_nav = False
            self.current_step += 1
            self.execute_next_step()

    def _nav_feedback_cb(self, feedback_msg):
        dist = (
            feedback_msg.feedback.distance_remaining)
        if dist < 2.0:
            self.get_logger().info(
                f"  Nav: {dist:.2f}m remaining")


# =============================================================
# MAIN
# =============================================================
def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()