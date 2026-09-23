#!/usr/bin/env python3
"""
get_room_orientations.py
Drive robot to each room, face desired direction,
then press Enter to record the orientation.

HOW TO USE:
  Terminal 1: Launch your robot (bringup)
  Terminal 2: Launch teleop (keyboard control)
  Terminal 3: Run this script

  Drive to each room → face the direction you want
  the robot to arrive facing → press Enter
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OrientationReader(Node):

    def __init__(self):
        super().__init__('orientation_reader')
        self.current_odom = None
        self.sub = self.create_subscription(
            Odometry, '/odom',
            self.odom_cb, 10)
        self.get_logger().info("Waiting for /odom...")

    def odom_cb(self, msg):
        self.current_odom = msg

    def quaternion_to_yaw(self, q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def read_current(self):
        if self.current_odom is None:
            print("  ❌ No odom data received!")
            return None

        p = self.current_odom.pose.pose
        x = p.position.x
        y = p.position.y
        yaw = self.quaternion_to_yaw(p.orientation)
        qz = p.orientation.z
        qw = p.orientation.w

        print(f"\n  📍 Position:    x={x:.4f}  y={y:.4f}")
        print(f"  🧭 Orientation: yaw={yaw:.4f} rad "
              f"({math.degrees(yaw):.1f}°)")
        print(f"  🔢 Quaternion:  z={qz:.4f}  w={qw:.4f}")

        return {
            'x': x, 'y': y,
            'yaw': yaw,
            'qz': qz, 'qw': qw
        }


def main():
    rclpy.init()
    node = OrientationReader()

    # Wait for first odom message
    print("\n⏳ Waiting for odometry data...")
    for i in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.current_odom is not None:
            print("✅ Odometry connected!\n")
            break
    else:
        print("❌ No odometry after 5 seconds!")
        print("   Make sure your robot is running")
        print("   Check: ros2 topic echo /odom --once")
        node.destroy_node()
        rclpy.shutdown()
        return

    rooms = [
        "kitchen",
        "tv room",
        "room 1",
        "room 2",
        "reception"
    ]

    results = {}

    print("=" * 55)
    print("  ROOM ORIENTATION RECORDER")
    print("=" * 55)
    print()
    print("  HOW IT WORKS:")
    print("  1. Use teleop to drive robot to a room")
    print("  2. Face the robot in the direction you")
    print("     want it to arrive facing")
    print("  3. Press ENTER to record")
    print("  4. Repeat for each room")
    print()
    print("  TIP: Face the robot toward the CENTER")
    print("  of the room so it can see objects")
    print()

    for room in rooms:
        print(f"\n{'─'*55}")
        print(f"  ROOM: {room.upper()}")
        print(f"{'─'*55}")
        print(f"  Drive to {room} and face the")
        print(f"  direction you want the robot to")
        print(f"  arrive facing.")
        print()

        input(f"  Press ENTER when ready...")

        # Get fresh odom data
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)

        data = node.read_current()
        if data:
            results[room] = data
            print(f"\n  ✅ {room} recorded!")
        else:
            print(f"\n  ❌ Failed to record {room}")

    # ─── Print results ──────────────────────────────────
    print("\n\n")
    print("=" * 55)
    print("  RESULTS — COPY INTO robot_controller.py")
    print("=" * 55)

    print("\nROOM_LOCATIONS = {")
    for room, data in results.items():
        print(f'    "{room}":'.ljust(20) +
              f'({data["x"]:8.4f}, {data["y"]:8.4f}),')
    print("}")

    print("\nROOM_ORIENTATIONS = {")
    for room, data in results.items():
        deg = math.degrees(data['yaw'])
        print(f'    "{room}":'.ljust(20) +
              f'{data["yaw"]:7.4f},  '
              f'# {deg:.1f}°')
    print("}")

    # Also save to a file for backup
    try:
        with open('/tmp/room_orientations.txt', 'w') as f:
            f.write("ROOM_LOCATIONS = {\n")
            for room, data in results.items():
                f.write(f'    "{room}":'.ljust(20) +
                        f'({data["x"]:8.4f}, '
                        f'{data["y"]:8.4f}),\n')
            f.write("}\n\n")
            f.write("ROOM_ORIENTATIONS = {\n")
            for room, data in results.items():
                deg = math.degrees(data['yaw'])
                f.write(f'    "{room}":'.ljust(20) +
                        f'{data["yaw"]:7.4f},  '
                        f'# {deg:.1f}°\n')
            f.write("}\n")
        print("\n💾 Also saved to /tmp/room_orientations.txt")
    except Exception:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
