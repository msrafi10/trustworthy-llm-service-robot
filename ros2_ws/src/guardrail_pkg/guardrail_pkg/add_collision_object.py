#!/usr/bin/env python3
"""
add_collision_object.py

Reads a named object's pose from Gazebo's /model_states, converts it into
the robot's base_footprint frame using the robot's /odom pose (position +
yaw), and publishes it to the MoveIt Planning Scene as a CollisionObject
built from one or more primitive shapes matching the object's actual SDF
collision geometry.

Usage:
    ros2 run guardrail_pkg add_collision_object <object_name>

Example:
    ros2 run guardrail_pkg add_collision_object bottle_red_wine_pick_0
    ros2 run guardrail_pkg add_collision_object bottle_red_wine_pick
    ros2 run guardrail_pkg add_collision_object apple_pick
    ros2 run guardrail_pkg add_collision_object apple_pick_0
    ros2 run guardrail_pkg add_collision_object cup_pick
    ros2 run guardrail_pkg add_collision_object medicine_pick
    ros2 run guardrail_pkg add_collision_object syringe_pick
    ros2 run guardrail_pkg add_collision_object knife_pick
"""

import sys
from math import atan2, cos, sin

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry

from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


# ---------------------------------------------------------------------------
# OBJECT DATABASE
#
# Dimensions and offsets here are taken directly from each object's SDF
# collision geometry (body/neck for the bottle, blade/handle for the knife,
# and the collision-origin offsets for apple/cup/medicine/syringe).
#
# Supported "type" values: "sphere", "cylinder", "box", "two_cylinder", "two_box"
# ---------------------------------------------------------------------------
OBJECT_DB = {

    "bottle_red_wine_pick": {
        "type": "two_cylinder",

        "body_radius": 0.009,
        "body_height": 0.09,
        "body_offset": [0.0, 0.0, 0.05],

        "neck_radius": 0.004,
        "neck_height": 0.02,
        "neck_offset": [0.0, 0.0, 0.10],
    },

    "apple_pick": {
        "type": "sphere",
        "radius": 0.028,
        "offset": [0.0, 0.0, 0.031],
    },

    "cup_pick": {
        "type": "cylinder",
        "radius": 0.022,
        "height": 0.070,
        "offset": [0.0, 0.0, 0.035],
    },

    "medicine_pick": {
        "type": "cylinder",
        "radius": 0.012,
        "height": 0.034,
        "offset": [0.0, 0.0, 0.017],
    },

    "syringe_pick": {
        "type": "box",
        "dims": [0.013, 0.052, 0.008],
        "offset": [0.0, 0.0, 0.0],
    },

    "knife_pick": {
        "type": "two_box",

        "blade_dims": [0.100, 0.025, 0.003],
        "blade_offset": [-0.045, 0.0, 0.0],

        "handle_dims": [0.080, 0.020, 0.008],
        "handle_offset": [0.055, 0.0, 0.0],
    },
}


def build_primitives(spec):
    """
    Given an OBJECT_DB entry, return a list of (SolidPrimitive, local_offset_xyz)
    tuples. local_offset_xyz is a [x, y, z] translation relative to the
    object's own pose - taken from the SDF collision offsets so the
    Planning Scene matches the actual Gazebo collision geometry.

    NOTE: local orientation for each part is left as identity. If any part
    is rotated relative to the object's local frame in the SDF (not just
    translated), that rotation isn't captured here yet.
    """
    obj_type = spec["type"]
    primitives = []

    if obj_type == "sphere":
        p = SolidPrimitive()
        p.type = SolidPrimitive.SPHERE
        p.dimensions = [spec["radius"]]
        primitives.append((p, spec["offset"]))

    elif obj_type == "cylinder":
        p = SolidPrimitive()
        p.type = SolidPrimitive.CYLINDER
        # shape_msgs cylinder dimension order: [height, radius]
        p.dimensions = [spec["height"], spec["radius"]]
        primitives.append((p, spec["offset"]))

    elif obj_type == "box":
        p = SolidPrimitive()
        p.type = SolidPrimitive.BOX
        p.dimensions = list(spec["dims"])
        primitives.append((p, spec["offset"]))

    elif obj_type == "two_cylinder":
        body = SolidPrimitive()
        body.type = SolidPrimitive.CYLINDER
        body.dimensions = [spec["body_height"], spec["body_radius"]]
        primitives.append((body, spec["body_offset"]))

        neck = SolidPrimitive()
        neck.type = SolidPrimitive.CYLINDER
        neck.dimensions = [spec["neck_height"], spec["neck_radius"]]
        primitives.append((neck, spec["neck_offset"]))

    elif obj_type == "two_box":
        blade = SolidPrimitive()
        blade.type = SolidPrimitive.BOX
        blade.dimensions = list(spec["blade_dims"])
        primitives.append((blade, spec["blade_offset"]))

        handle = SolidPrimitive()
        handle.type = SolidPrimitive.BOX
        handle.dimensions = list(spec["handle_dims"])
        primitives.append((handle, spec["handle_offset"]))

    else:
        raise ValueError(f"Unknown primitive type: {obj_type}")

    return primitives


class AddCollisionObject(Node):

    def __init__(self):
        super().__init__("add_collision_object")

        if len(sys.argv) < 2:
            self.get_logger().error(
                "Usage:\n"
                "  ros2 run guardrail_pkg add_collision_object <object_name>\n"
                f"Known objects: {', '.join(OBJECT_DB.keys())}"
            )
            rclpy.shutdown()
            sys.exit(1)

        self.target = sys.argv[1]

        # Match by prefix so duplicated Gazebo instances (apple_pick_0,
        # apple_pick_1, bottle_red_wine_pick_2, ...) all resolve to the
        # same database entry without needing one entry per instance.
        matched_key = None
        for key in OBJECT_DB:
            if self.target.startswith(key):
                matched_key = key
                break

        if matched_key is None:
            self.get_logger().error(
                f"'{self.target}' does not match any known object prefix.\n"
                f"Known object prefixes: {', '.join(OBJECT_DB.keys())}"
            )
            rclpy.shutdown()
            sys.exit(1)

        self.spec = OBJECT_DB[matched_key]
        self.robot_pose = None
        self._done = False

        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.cli = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /apply_planning_scene...")

        self._model_states_sub = None

        # Step 1: remove any stale copy of this object from a previous run
        self.remove_previous_object()

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    # ------------------------------------------------------------------
    # STEP 1: remove previous object (if any) before adding the new one
    # ------------------------------------------------------------------
    def remove_previous_object(self):
        scene = PlanningScene()
        scene.is_diff = True

        remove_obj = CollisionObject()
        remove_obj.header.frame_id = "base_footprint"
        remove_obj.id = self.target
        remove_obj.operation = CollisionObject.REMOVE

        scene.world.collision_objects.append(remove_obj)

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = self.cli.call_async(request)
        future.add_done_callback(self.on_remove_done)

    def on_remove_done(self, future):
        try:
            future.result()
        except Exception as e:
            # Not fatal - the object may simply not have existed yet.
            self.get_logger().warn(f"Remove step reported: {e}")

        self.get_logger().info(
            f"Cleared any previous '{self.target}'. Waiting for /odom and /model_states..."
        )

        # Step 2: now start listening for the object's live pose
        self._model_states_sub = self.create_subscription(
            ModelStates,
            "/model_states",
            self.model_callback,
            10
        )

    # ------------------------------------------------------------------
    # STEP 2: read object pose, convert into base_footprint via /odom,
    # and publish as ADD
    # ------------------------------------------------------------------
    def model_callback(self, msg):
        if self._done:
            return

        if self.robot_pose is None:
            # Haven't received /odom yet.
            return

        obj_index = None
        for i, name in enumerate(msg.name):
            if name == self.target:
                obj_index = i
                break

        if obj_index is None:
            return

        obj_pose = msg.pose[obj_index]

        self.get_logger().info(f"FOUND {self.target}, building collision object...")

        pose_in_base = self.transform_to_base_footprint(obj_pose)
        self.publish_object(pose_in_base)

    def transform_to_base_footprint(self, obj_pose):
        """
        Convert obj_pose (in the world frame reported by /model_states)
        into the base_footprint frame, using the robot's /odom pose.
        This is the version confirmed to place objects correctly in RViz.
        """
        rx = self.robot_pose.position.x
        ry = self.robot_pose.position.y
        rz = self.robot_pose.position.z

        q = self.robot_pose.orientation
        yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        dx = obj_pose.position.x - rx
        dy = obj_pose.position.y - ry
        dz = obj_pose.position.z - rz

        x_robot = cos(yaw) * dx + sin(yaw) * dy
        y_robot = -sin(yaw) * dx + cos(yaw) * dy

        # Small 2mm lift to avoid the collision object intersecting the
        # table surface due to tiny simulation/measurement noise.
        Z_SAFETY_MARGIN = 0.002

        p = Pose()
        p.position.x = x_robot
        p.position.y = y_robot
        p.position.z = dz + Z_SAFETY_MARGIN
        p.orientation = obj_pose.orientation

        return p

    def publish_object(self, pose_in_base: Pose):
        scene = PlanningScene()
        scene.is_diff = True

        obj = CollisionObject()
        obj.header.frame_id = "base_footprint"
        obj.id = self.target
        obj.operation = CollisionObject.ADD

        # Overall object pose in base_footprint. Individual primitives are
        # offset from this pose using the SDF collision offsets.
        obj.pose = pose_in_base

        for primitive, offset in build_primitives(self.spec):
            local_pose = Pose()
            local_pose.position.x = offset[0]
            local_pose.position.y = offset[1]
            local_pose.position.z = offset[2]
            local_pose.orientation.w = 1.0  # identity - see docstring caveat

            obj.primitives.append(primitive)
            obj.primitive_poses.append(local_pose)

        scene.world.collision_objects.append(obj)

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = self.cli.call_async(request)
        future.add_done_callback(self.on_add_done)

    def on_add_done(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            result = None

        if result is not None and result.success:
            self.get_logger().info(f"'{self.target}' added successfully.")
        else:
            self.get_logger().error(f"Failed to add '{self.target}'.")

        self._done = True
        self.destroy_node()
        rclpy.shutdown()


def main():
    rclpy.init()
    node = AddCollisionObject()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()