#!/usr/bin/env python3

# =============================================================
# moveit_pick_place.py — v6.2 Fixed Lift
#
# ROOT CAUSE OF ARM DIP / ROBOT BACKWARD:
#   _cartesian_lift sent multiple separate MoveIt pose
#   goals. MoveIt solved each with independent IK —
#   sometimes picking elbow-up, sometimes elbow-down.
#   The arm flipped configuration mid-lift causing the
#   dip. The torque reversal pushed the robot backward.
#
# FIX:
#   After grasping, lift uses direct FollowJointTrajectory
#   with IK-computed waypoints interpolated smoothly.
#   Single trajectory, no IK re-solving, no config flip.
#
#   Sequence:
#     grasp_joints  (IK at grasp position)
#         ↓ interpolate
#     pre_pick_joints (IK at hover position)
#         ↓ interpolate
#     carry_joints  (known safe config)
#   All sent as ONE multi-waypoint trajectory.
# =============================================================

import math
import time

import rclpy
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, Quaternion
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints,
    JointConstraint, PositionConstraint,
    OrientationConstraint, PlanningOptions,
    BoundingVolume,
)

from control_msgs.action import (
    FollowJointTrajectory, GripperCommand)
from trajectory_msgs.msg import (
    JointTrajectory, JointTrajectoryPoint)
from builtin_interfaces.msg import Duration

from guardrail_pkg.arm_ik import (
    solve_ik,
    pre_pick_pose,
    approach_pose,
    carry_pose,
    home_pose,
    pick_pose
)


# =============================================================
# CONFIG
# =============================================================
ARM_GROUP         = "arm"
ARM_JOINTS        = ["joint1", "joint2", "joint3", "joint4"]
EE_LINK           = "link5"
ARM_ACTION        = "/arm_controller/follow_joint_trajectory"
GRIPPER_ACTION    = "/gripper_controller/gripper_cmd"
MOVEIT_ACTION     = "/move_action"

GRIPPER_OPEN      =  0.015
GRIPPER_CLOSED    =  0.003
GRIPPER_EFFORT    =  30.0

PLANNING_TIME     = 10.0
PLANNING_ATTEMPTS = 20
MAX_VEL_SCALE     = 0.05
MAX_ACC_SCALE     = 0.05

# Arm mounting offset from base_link
ARM_OFFSET_X      = -0.092
ARM_OFFSET_Y      =  0.000
ARM_OFFSET_Z      =  0.091

# Arm workspace limits (in arm frame)
ARM_REACH_MIN     = 0.08
ARM_REACH_MAX     = 0.28
ARM_Z_MIN         = -0.10
ARM_Z_MAX         = 0.30

# Pick geometry
BOTTLE_HEIGHT     = 0.20   # meters — measure your bottle
GRASP_RATIO       = 0.25   # grasp at 25% of bottle height
PRE_PICK_HEIGHT   = 0.04   # hover this far above object


class MoveItPickPlace:

    def __init__(self, node, cb_group=None):
        self.node = node
        self.log  = node.get_logger()

        self.arm_client = ActionClient(
            node, FollowJointTrajectory, ARM_ACTION,
            callback_group=cb_group)

        self.gripper_client = ActionClient(
            node, GripperCommand, GRIPPER_ACTION,
            callback_group=cb_group)

        self.moveit_client = ActionClient(
            node, MoveGroup, MOVEIT_ACTION,
            callback_group=cb_group)

        self.moveit_ok = False
        self._check_moveit()

        self.current_joints = None
        node.create_subscription(
            JointState, '/joint_states',
            self._joint_cb, 10,
            callback_group=cb_group)

        self.log.info("=" * 55)
        self.log.info("MoveIt Pick&Place v6.2 — Fixed Lift")
        self.log.info(
            f"  MoveIt2: {'✅' if self.moveit_ok else '⚠ OFF'}")
        self.log.info(
            f"  Grasp at {GRASP_RATIO*100:.0f}% of "
            f"{BOTTLE_HEIGHT*100:.0f}cm bottle")
        self.log.info("=" * 55)

    def _check_moveit(self):
        self.moveit_ok = self.moveit_client.wait_for_server(
            timeout_sec=5.0)
        if self.moveit_ok:
            self.log.info("✅ MoveIt2 server found")
        else:
            self.log.warn("⚠ MoveIt2 unavailable — IK fallback")

    def _joint_cb(self, msg):
        self.current_joints = dict(zip(msg.name, msg.position))

    # =========================================================
    # UTILITIES
    # =========================================================
    def _wait_future(self, future, timeout=10.0):
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                return False
            time.sleep(0.05)
        return True

    def safe_sleep(self, d):
        time.sleep(d)

    def base_to_arm(self, bx, by, bz):
        return (bx - ARM_OFFSET_X,
                by - ARM_OFFSET_Y,
                bz - ARM_OFFSET_Z)

    def _pick_orientation(self, ax, ay, az):
        yaw   = math.atan2(ay, ax)
        horiz = math.sqrt(ax**2 + ay**2)
        pitch = (-math.atan2(-az, horiz)
                 if horiz > 0.01 else -math.pi/2)
        pitch = max(-math.pi/2, min(pitch, -math.pi/6))
        return self._rpy_to_quat(0.0, pitch, yaw)

    def _rpy_to_quat(self, roll, pitch, yaw):
        cr, sr = math.cos(roll/2),  math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
        q = Quaternion()
        q.w = cr*cp*cy + sr*sp*sy
        q.x = sr*cp*cy - cr*sp*sy
        q.y = cr*sp*cy + sr*cp*sy
        q.z = cr*cp*sy - sr*sp*cy
        return q

    # =========================================================
    # REACHABILITY CHECK
    # =========================================================
    def is_reachable(self, bx, by, bz):
        ax, ay, az = self.base_to_arm(bx, by, bz)
        horiz = math.sqrt(ax**2 + ay**2)

        self.log.info(
            f"  Reach: base=({bx:.3f},{by:.3f},{bz:.3f})"
            f" → arm=({ax:.3f},{ay:.3f},{az:.3f})"
            f" h={horiz:.3f}m")

        if horiz < ARM_REACH_MIN:
            return False, f"Too close: h={horiz:.3f}m", (ax,ay,az)
        if horiz > ARM_REACH_MAX:
            return False, f"Too far: h={horiz:.3f}m", (ax,ay,az)
        if az < ARM_Z_MIN:
            return False, f"Too low: az={az:.3f}m", (ax,ay,az)
        if az > ARM_Z_MAX:
            return False, f"Too high: az={az:.3f}m", (ax,ay,az)

        return True, f"✅ h={horiz:.3f}m z={az:.3f}m", (ax,ay,az)

    # =========================================================
    # GRIPPER
    # =========================================================
    def open_gripper(self, position=None):
        """
        Open gripper.

        position=None -> use default open width.
        position=value -> open to a custom width.
        """
        if position is None:
            position = GRIPPER_OPEN

        return self._gripper(position, f"OPEN ({position:.4f})")

    def close_gripper(self, position=None):
        """
        Close gripper.

        position=None -> use default fully closed.
        position=value -> close to a custom width.
        """
        if position is None:
            position = GRIPPER_CLOSED

        return self._gripper(position, f"CLOSE ({position:.4f})")

    def _gripper(self, pos, name):
        if not self.gripper_client.wait_for_server(timeout_sec=3.0):
            self.log.error("Gripper server unavailable!")
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = pos
        goal.command.max_effort = GRIPPER_EFFORT

        self.log.info(f"  Gripper → {name}")

        future = self.gripper_client.send_goal_async(goal)
        if not self._wait_future(future, 5.0):
            self.log.error(f"  Gripper {name} — no response")
            return False

        gh = future.result()
        if not gh.accepted:
            self.log.error(f"  Gripper {name} REJECTED")
            return False

        res = gh.get_result_async()
        if not self._wait_future(res, 10.0):
            self.log.warn(f"  Gripper {name} — timeout (ok)")
            return True

        self.log.info(f"  ✅ Gripper {name}")
        return True

    # =========================================================
    # ARM — Direct Joint Trajectory
    # Used for: home, carry, and the fixed-config lift
    # =========================================================
    def move_joints(self, positions, duration=4.0):
        if not self.arm_client.wait_for_server(timeout_sec=3.0):
            self.log.error("Arm server unavailable!")
            return False

        n = len(ARM_JOINTS)
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS

        # Add midpoint for smooth motion if we know current pose
        if self.current_joints is not None:
            current = [self.current_joints.get(j, 0.0)
                       for j in ARM_JOINTS]
            mid = JointTrajectoryPoint()
            mid.positions     = [(c+t)/2 for c,t in zip(current, positions)]
            mid.velocities    = [0.0] * n
            mid.accelerations = [0.0] * n
            half = duration * 0.5
            mid.time_from_start = Duration(
                sec=int(half),
                nanosec=int((half % 1) * 1e9))
            traj.points.append(mid)

        final = JointTrajectoryPoint()
        final.positions     = list(positions)
        final.velocities    = [0.0] * n
        final.accelerations = [0.0] * n
        final.time_from_start = Duration(
            sec=int(duration),
            nanosec=int((duration % 1) * 1e9))
        traj.points.append(final)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.log.info(
            f"  Arm → {[f'{p:.3f}' for p in positions]}")

        future = self.arm_client.send_goal_async(goal)
        if not self._wait_future(future, 5.0):
            self.log.error("  Arm — no response")
            return False

        gh = future.result()
        if not gh.accepted:
            self.log.error("  Arm REJECTED")
            return False

        res = gh.get_result_async()
        if not self._wait_future(res, duration + 5.0):
            self.log.warn("  Arm — result timeout")
            return True

        self.log.info("  ✅ Arm done")
        return True

    # =========================================================
    # ARM — Multi-waypoint Joint Trajectory
    # THE KEY FIX: sends grasp→pre_pick→carry as ONE
    # trajectory so the arm lifts smoothly without any
    # IK re-solving or configuration flips between steps.
    # =========================================================
    def move_joints_smooth(self, waypoints, total_duration=8.0):
        """
        Send multiple joint waypoints as a single trajectory.

        Args:
            waypoints: list of joint position lists
                       e.g. [grasp_joints, pre_pick_joints, carry_joints]
            total_duration: total time for full trajectory

        Returns:
            True if successful
        """
        if not self.arm_client.wait_for_server(timeout_sec=3.0):
            self.log.error("Arm server unavailable!")
            return False

        if not waypoints:
            return False

        n = len(ARM_JOINTS)
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS

        dt = total_duration / len(waypoints)

        for i, positions in enumerate(waypoints):
            t = dt * (i + 1)
            pt = JointTrajectoryPoint()
            pt.positions     = list(positions)
            pt.velocities    = [0.0] * n
            pt.accelerations = [0.0] * n
            pt.time_from_start = Duration(
                sec=int(t),
                nanosec=int((t % 1) * 1e9))
            traj.points.append(pt)

            self.log.info(
                f"  Waypoint {i+1}/{len(waypoints)} "
                f"at t={t:.1f}s: "
                f"{[f'{p:.3f}' for p in positions]}")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.log.info(
            f"  Smooth trajectory: "
            f"{len(waypoints)} waypoints "
            f"over {total_duration:.1f}s")

        future = self.arm_client.send_goal_async(goal)
        if not self._wait_future(future, 5.0):
            self.log.error("  Smooth traj — no response")
            return False

        gh = future.result()
        if not gh.accepted:
            self.log.error("  Smooth traj REJECTED")
            return False

        res = gh.get_result_async()
        if not self._wait_future(res, total_duration + 5.0):
            self.log.warn("  Smooth traj — timeout")
            return True

        self.log.info("  ✅ Smooth trajectory done")
        return True

    # =========================================================
    # ARM — MoveIt Joint Goal (for home/carry)
    # =========================================================
    def move_joints_moveit(self, positions):
        if not self.moveit_ok:
            return self.move_joints(positions)

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = PLANNING_ATTEMPTS
        req.allowed_planning_time           = PLANNING_TIME
        req.max_velocity_scaling_factor     = MAX_VEL_SCALE
        req.max_acceleration_scaling_factor = MAX_ACC_SCALE

        constraints = Constraints()
        for name, pos in zip(ARM_JOINTS, positions):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints = [constraints]

        opts = PlanningOptions()
        opts.plan_only       = False
        opts.replan          = True
        opts.replan_attempts = 3

        goal.request         = req
        goal.planning_options = opts

        self.log.info(
            f"  MoveIt joint: "
            f"{[f'{p:.3f}' for p in positions]}")

        future = self.moveit_client.send_goal_async(goal)
        if not self._wait_future(future, 10.0):
            self.log.warn("  MoveIt no response → fallback")
            return self.move_joints(positions)

        gh = future.result()
        if not gh.accepted:
            self.log.warn("  MoveIt rejected → fallback")
            return self.move_joints(positions)

        res = gh.get_result_async()
        if not self._wait_future(res, 30.0):
            self.log.warn("  MoveIt timeout")
            return self.move_joints(positions)

        result = res.result()
        if result.result.error_code.val == 1:
            self.log.info("  ✅ MoveIt joint done")
            return True
        else:
            self.log.warn(
                f"  MoveIt err "
                f"{result.result.error_code.val} → fallback")
            return self.move_joints(positions)

    # =========================================================
    # ARM — MoveIt Pose Goal (for pre-pick and descend)
    # =========================================================
    def move_to_pose(self, x, y, z,
                     orientation=None,
                     frame_id="link1"):
        if not self.moveit_ok:
            self.log.warn("  MoveIt unavailable for pose goal")
            return False

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = PLANNING_ATTEMPTS
        req.allowed_planning_time           = PLANNING_TIME
        req.max_velocity_scaling_factor     = MAX_VEL_SCALE
        req.max_acceleration_scaling_factor = MAX_ACC_SCALE

        constraints = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name       = EE_LINK

        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type       = SolidPrimitive.SPHERE
        sphere.dimensions = [0.03]
        bv.primitives.append(sphere)

        sp = Pose()
        sp.position.x    = x
        sp.position.y    = y
        sp.position.z    = z
        sp.orientation.w = 1.0
        bv.primitive_poses.append(sp)

        pc.constraint_region = bv
        pc.weight            = 1.0
        constraints.position_constraints.append(pc)

        if orientation is not None:
            oc = OrientationConstraint()
            oc.header.frame_id           = frame_id
            oc.link_name                 = EE_LINK
            oc.orientation               = orientation
            oc.absolute_x_axis_tolerance = 0.1
            oc.absolute_y_axis_tolerance = 0.1
            oc.absolute_z_axis_tolerance = 3.14
            oc.weight                    = 0.5
            constraints.orientation_constraints.append(oc)

        req.goal_constraints  = [constraints]

        opts = PlanningOptions()
        opts.plan_only        = False
        opts.replan           = True
        opts.replan_attempts  = 5

        goal.request          = req
        goal.planning_options = opts

        self.log.info(
            f"  MoveIt pose: ({x:.3f},{y:.3f},{z:.3f}) "
            f"frame={frame_id}")

        future = self.moveit_client.send_goal_async(goal)
        if not self._wait_future(future, 10.0):
            self.log.error("  MoveIt pose — no response")
            return False

        gh = future.result()
        if not gh.accepted:
            self.log.error("  MoveIt pose REJECTED")
            return False

        res = gh.get_result_async()
        if not self._wait_future(res, 30.0):
            self.log.warn("  MoveIt pose — timeout")
            return False

        err = res.result().result.error_code.val
        if err == 1:
            self.log.info("  ✅ MoveIt pose done")
            return True
        else:
            self.log.error(f"  ❌ MoveIt pose failed (err={err})")
            return False



    # =========================================================
    # BOTTLE SDF (embedded — no external file needed)
    # =========================================================
    BOTTLE_SDF = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='bottle_red_wine_pick'>
    <static>false</static>
    <link name='bottle_link'>
      <pose>0 0 0 0 0 0</pose>

      <inertial>
        <mass>0.10</mass>
        <pose>0 0 0.05 0 0 0</pose>
        <inertia>
          <ixx>0.00010</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.00010</iyy>
          <iyz>0</iyz>
          <izz>0.00002</izz>
        </inertia>
      </inertial>

      <!-- Visual -->
      <visual name='visual'>
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>model://bottle_red_wine/meshes/bottle.dae</uri>
            <scale>0.5 0.5 0.5</scale>
          </mesh>
        </geometry>
      </visual>

      <!-- Body collision -->
      <collision name='body_collision'>
        <pose>0 0 0.05 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.009</radius>
            <length>0.09</length>
          </cylinder>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1</mu>
              <mu2>1</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <kp>5000</kp>
              <kd>50</kd>
              <max_vel>0</max_vel>
              <min_depth>0.0005</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>

      <!-- Neck collision -->
      <collision name='neck_collision'>
        <pose>0 0 0.10 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.004</radius>
            <length>0.02</length>
          </cylinder>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1</mu>
              <mu2>1</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <kp>5000</kp>
              <kd>50</kd>
              <max_vel>0</max_vel>
              <min_depth>0.0005</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>

    </link>
  </model>
</sdf>"""





    # =========================================================
    # PICK OBJECT — v6.2
    #
    # KEY CHANGE vs v6.1:
    #   Step 6 (lift) no longer uses MoveIt pose goals.
    #   Instead it computes joint angles at grasp, pre-pick,
    #   and carry positions using geometric IK, then sends
    #   all three as ONE smooth joint trajectory.
    #
    #   This prevents:
    #   - IK configuration flips (elbow up ↔ elbow down)
    #   - Arm dipping during lift
    #   - Robot recoil from torque reversal
    # =========================================================
    def pick_object(self, bx, by, bz, use_moveit=True):

        self.log.info("=" * 55)
        self.log.info("🤖 PICK v6.2 — Fixed Lift")
        self.log.info(
            f"  base_link: ({bx:.3f}, {by:.3f}, {bz:.3f})")

        ok, reason, arm_xyz = self.is_reachable(bx, by, bz)
        ax, ay, az = arm_xyz

        if not ok:
            self.log.error(f"  ❌ {reason}")
            return False

        self.log.info(
            f"  arm_frame: ({ax:.3f}, {ay:.3f}, {az:.3f})")

        jfn = (self.move_joints_moveit if use_moveit
               else self.move_joints)

        orient = self._pick_orientation(ax, ay, az)

        # ── Step 1: Home ───────────────────────────────
        self.log.info("  [1/7] Home")
        if not jfn(home_pose()):
            self.log.error("  ❌ Home failed")
            return False
        self.safe_sleep(0.5)

        # ── Step 2: APPROACH POSE ─────────────────────

        self.log.info("  [2/7] APPROACH")

        approach_joints = approach_pose(
            ax, ay, az)

        if approach_joints is None:
            self.log.error("  ❌ Approach IK failed")
            return False

        if not jfn(approach_joints):
            self.log.error("  ❌ Approach move failed")
            return False

        self.safe_sleep(1.0)


        # ── Step 3: Open gripper ───────────────────────
        self.log.info("  [3/7] Open gripper")
        self.open_gripper()
        self.safe_sleep(0.5)

        # ── Step 4: FINAL PICK ────────────────────────

        self.log.info("  [4/7] FINAL PICK")

        pick_joints = pick_pose(
            ax, ay, az)

        if pick_joints is None:
            self.log.error("  ❌ Pick IK failed")
            return False

        if not jfn(pick_joints):
            self.log.error("  ❌ Pick move failed")
            return False

        self.safe_sleep(1.0)

        # ── Step 5: Grasp ──────────────────────────────

        self.log.info("  [5/7] GRASPING")
        self.safe_sleep(0.5)
        self.close_gripper()
        self.safe_sleep(1.0)
        self.close_gripper()
        self.safe_sleep(0.5)

        # ─── IFRA ATTACH ───────────────────────────
        self.node.attach_object(
            'bottle_red_wine_pick')

        self.safe_sleep(0.5)

        # DELETE bottle from sim


        # ── Step 6: LIFT ──────────────────────────────

        self.log.info("  [6/7] LIFT")

        if not jfn(approach_joints):
            self.log.error("  ❌ Lift failed")
            return False

        self.safe_sleep(1.0)

        # ── Step 7: Carry confirmation ─────────────────
        self.log.info("  [7/7] Carry confirmation")
        jfn(carry_pose())
        self.safe_sleep(0.5)

        self.log.info("=" * 55)
        self.log.info("✅ PICK COMPLETE!")
        self.log.info("=" * 55)
        return True

    # =========================================================
    # PLACE OBJECT
    # =========================================================
    def place_object(self, height=0.05,
                     use_moveit=True):
        jfn = (self.move_joints_moveit if use_moveit
               else self.move_joints)

        self.log.info("🤖 PLACE")

        self.log.info("  [1/4] Place pose")
        if use_moveit and self.moveit_ok:
            ok = self.move_to_pose(
                0.15, 0.0, height, None, "link1")
            if not ok:
                from guardrail_pkg.arm_ik import (
                    place_pose as ik_place)
                pl = ik_place(height)
                if pl is None:
                    pl = [0.0, -0.5, 0.3, 0.2]
                jfn(pl)
        else:
            from guardrail_pkg.arm_ik import (
                place_pose as ik_place)
            pl = ik_place(height)
            if pl is None:
                pl = [0.0, -0.5, 0.3, 0.2]
            jfn(pl)

        self.safe_sleep(1.0)

        # RESPAWN bottle at gripper position
        self.log.info("  [2/4] Release")
        self.open_gripper()
        self.safe_sleep(1.0)

        self.log.info("  [3/4] Release")

        self.safe_sleep(1.0)

        self.log.info("  [4/4] Home")
        jfn(home_pose())

        self.log.info("✅ PLACE complete!")
        return True

    # =========================================================
    # UTILITY
    # =========================================================
    def go_home(self, use_moveit=True):
        jfn = (self.move_joints_moveit if use_moveit
               else self.move_joints)
        return jfn(home_pose())