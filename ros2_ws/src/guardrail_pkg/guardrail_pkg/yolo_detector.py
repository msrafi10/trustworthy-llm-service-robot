#!/usr/bin/env python3

# =============================================================
# yolo_detector.py — YOLO + Depth Camera Fusion (v5-reactive)
# Package: guardrail_pkg
#
# v5 changes over v4:
#   - FIXED get_depth_at() call with RGB resolution params
#   - Best-per-TARGET-class (not global best) — won't miss
#     bottle because a box scored higher
#   - Publishes "no_detection" heartbeat so controller knows
#     when object leaves frame (prevents stale bbox)
#   - Adaptive publish rate: faster when object detected
#   - Removed unused Float32 import
#   - Exception logging in _get_quick_depth
# =============================================================

import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import tf2_ros
import tf2_geometry_msgs

from guardrail_pkg.depth_processor import DepthProcessor


# ─── CONFIG ──────────────────────────────────────────────────
RGB_TOPIC         = '/pi_camera/image_raw'
DEPTH_TOPIC       = '/realsense_d435/depth/image_raw'
CAMERA_INFO_TOPIC = '/realsense_d435/depth/camera_info'
CAMERA_FRAME      = 'realsense_depth_optical_frame'
TARGET_FRAME      = 'base_link'
YOLO_MODEL        = os.environ.get(
    'YOLO_MODEL_PATH',
    os.path.expanduser('~/yolo_model/service_robot_7class/weights/best.pt')
)
YOLO_CONF         = 0.40
RGB_W, RGB_H      = 640, 480

MIN_DEPTH         = 0.15
MAX_DEPTH         = 3.0

# ─── PUBLISH RATE ───────────────────────────────────────────
# Normal rate when nothing detected
PUBLISH_RATE_NORMAL = 0.1     # 10 Hz
# Fast rate when object is actively detected
PUBLISH_RATE_ACTIVE = 0.05    # 20 Hz — faster during tracking

# ─── DEPTH ESTIMATION FROM BBOX ─────────────────────────────
KNOWN_BOTTLE_HEIGHT = 0.25    # meters

# ─── NO-DETECTION HEARTBEAT ─────────────────────────────────
# After this many consecutive empty frames, publish "none"
# so controller knows object left the view
NO_DETECT_FRAMES    = 3
# ─────────────────────────────────────────────────────────────


class YoloDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # ─── YOLO ───────────────────────────────────────
        self.model = YOLO(YOLO_MODEL)
        self.model.to('cuda')
        self.get_logger().info("🚀 YOLO running on GPU!")

        self.bridge = CvBridge()
        self.depth_proc = DepthProcessor(
            logger=self.get_logger())

        # ─── TF ─────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self)

        # ─── Subscriptions ──────────────────────────────
        self.sub_rgb = self.create_subscription(
            Image, RGB_TOPIC,
            self.image_callback, 10)

        self.sub_depth = self.create_subscription(
            Image, DEPTH_TOPIC,
            self.depth_callback, 10)

        self.sub_info = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC,
            self.cam_info_cb, 10)

        # ─── Publishers ─────────────────────────────────
        self.pub_det = self.create_publisher(
            String, 'detected_object', 10)

        self.pub_3d_pose = self.create_publisher(
            PoseStamped, '/object_3d_pose', 10)

        self.pub_3d_info = self.create_publisher(
            String, '/object_3d_info', 10)

        # v4: Quick depth for reactive controller
        self.pub_quick_depth = self.create_publisher(
            String, '/object_quick_depth', 10)

        # ─── State ──────────────────────────────────────
        self.last_pub = 0.0
        self.focal_length_y = None
        self.consecutive_empty = 0
        self.object_active = False   # tracking state

        self.get_logger().info("=" * 55)
        self.get_logger().info(
            "YOLO + Depth v5 — Reactive Edition")
        self.get_logger().info(f"  RGB:   {RGB_TOPIC}")
        self.get_logger().info(f"  Depth: {DEPTH_TOPIC}")
        self.get_logger().info(f"  Frame: {CAMERA_FRAME}")
        self.get_logger().info(
            f"  Rate normal: {PUBLISH_RATE_NORMAL}s "
            f"({1.0/PUBLISH_RATE_NORMAL:.0f} Hz)")
        self.get_logger().info(
            f"  Rate active: {PUBLISH_RATE_ACTIVE}s "
            f"({1.0/PUBLISH_RATE_ACTIVE:.0f} Hz)")
        self.get_logger().info("=" * 55)

    # =========================================================
    # CAMERA INFO
    # =========================================================
    def cam_info_cb(self, msg):
        if not self.depth_proc.intrinsics_ready:
            self.depth_proc.set_intrinsics_from_msg(msg)

        if self.focal_length_y is None and msg.k[4] > 0:
            self.focal_length_y = msg.k[4]
            self.get_logger().info(
                f"📷 Focal length fy="
                f"{self.focal_length_y:.1f}")

    # =========================================================
    # DEPTH IMAGE
    # =========================================================
    def depth_callback(self, msg):
        try:
            if msg.encoding == '16UC1':
                d = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='16UC1')
            elif msg.encoding == '32FC1':
                d = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='32FC1')
            else:
                d = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
            self.depth_proc.update_depth(
                d, msg.encoding)
        except Exception as e:
            self.get_logger().error(f"Depth: {e}")

    # =========================================================
    # RGB IMAGE — Main Detection Loop
    # =========================================================
    def image_callback(self, msg):

        now = time.time()

        # Adaptive rate: faster when actively tracking
        rate = (PUBLISH_RATE_ACTIVE if self.object_active
                else PUBLISH_RATE_NORMAL)

        if now - self.last_pub < rate:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(
                f"RGB convert: {e}")
            return

        results = self.model(frame, conf=YOLO_CONF)

        # ─── Collect ALL detections ─────────────────────
        detections = []
        for r in results:
            for box in r.boxes:
                c = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                coords = list(map(int, box.xyxy[0]))
                detections.append({
                    'label': label,
                    'conf': c,
                    'bbox': coords,
                    'cls_id': cls_id
                })

        # ─── No detections → heartbeat ──────────────────
        if len(detections) == 0:
            self.consecutive_empty += 1

            if (self.consecutive_empty >=
                    NO_DETECT_FRAMES and
                    self.object_active):
                # Tell controller: nothing in view
                none_msg = String()
                none_msg.data = "none 0 0 0 0 0.0"
                self.pub_det.publish(none_msg)
                self.object_active = False
                self.get_logger().info(
                    "👁 No detection — published none")

            self.last_pub = now
            return

        # ─── Reset empty counter ────────────────────────
        self.consecutive_empty = 0
        self.object_active = True

        # ─── Find best detection PER CLASS ──────────────
        # This ensures we always publish the best bottle
        # even if a "box" scores higher overall
        best_per_class = {}
        for det in detections:
            label = det['label']
            if (label not in best_per_class or
                    det['conf'] >
                    best_per_class[label]['conf']):
                best_per_class[label] = det

        # ─── Publish ALL best-per-class detections ──────
        # Controller will filter by target object
        for label, det in best_per_class.items():
            self._process_detection(det, now)

        self.last_pub = now

    # =========================================================
    # PROCESS SINGLE DETECTION
    # =========================================================
    def _process_detection(self, det, now):
        """Process and publish one detection."""

        label = det['label']
        conf  = det['conf']
        x1, y1, x2, y2 = det['bbox']

        clean_label = label.replace(" ", "_")

        # ─── Publish 2D detection ───────────────────────
        det_msg = String()
        det_msg.data = (
            f"{clean_label} {x1} {y1} {x2} {y2} "
            f"{conf:.2f}")
        self.pub_det.publish(det_msg)

        cx = (x1 + x2) / 2
        w  = x2 - x1
        h  = y2 - y1

        self.get_logger().info(
            f"Det: {clean_label} conf={conf:.2f} "
            f"bbox=[{x1},{y1},{x2},{y2}] "
            f"cx={cx:.0f} w={w}")

        # ─── Edge clipping check ────────────────────────
        edge_margin = 5
        is_clipped = (
            x1 < edge_margin or
            y1 < edge_margin or
            x2 > RGB_W - edge_margin or
            y2 > RGB_H - edge_margin)

        # ─── Quick depth (always attempt) ───────────────
        quick_depth = self._get_quick_depth(
            x1, y1, x2, y2, h)

        if quick_depth is not None:
            depth_msg = String()
            depth_msg.data = (
                f"{clean_label} {quick_depth:.3f}")
            self.pub_quick_depth.publish(depth_msg)

            self.get_logger().info(
                f"  Quick depth: {quick_depth:.2f}m"
                f" {'(CLIPPED)' if is_clipped else ''}")

        # ─── Full 3D only if NOT clipped ────────────────
        if is_clipped:
            self.get_logger().warn(
                "⚠ Edge-clipped — 2D + quick depth only")
            return

        if self.depth_proc.is_ready():
            self._pub_3d(
                clean_label, x1, y1, x2, y2)

    # =========================================================
    # QUICK DEPTH — Fast depth at bbox center
    # =========================================================
    def _get_quick_depth(self, x1, y1, x2, y2, bbox_h):
        """
        Get depth at bbox center from depth image.
        Falls back to bbox-height estimation.
        """
        depth_val = None

        # ── Method 1: Read from depth image ─────────────
        if self.depth_proc.is_ready():
            try:
                cx_int = int((x1 + x2) / 2)
                cy_int = int((y1 + y2) / 2)

                # v5 FIX: Pass RGB resolution so
                # depth_processor can scale to depth
                # image coordinates properly
                depth_val = self.depth_proc.get_depth_at(
                    cx_int, cy_int, RGB_W, RGB_H)

                if depth_val is not None:
                    if (depth_val < MIN_DEPTH or
                            depth_val > MAX_DEPTH):
                        depth_val = None

            except Exception as e:
                self.get_logger().warn(
                    f"Quick depth error: {e}")
                depth_val = None

        # ── Method 2: Estimate from bbox height ─────────
        if (depth_val is None and
                self.focal_length_y is not None and
                bbox_h > 20):

            depth_val = (
                KNOWN_BOTTLE_HEIGHT *
                self.focal_length_y / bbox_h)

            if (depth_val < MIN_DEPTH or
                    depth_val > MAX_DEPTH):
                depth_val = None
            else:
                self.get_logger().info(
                    f"  Depth est from bbox: "
                    f"{depth_val:.2f}m "
                    f"(h={bbox_h}px)")

        return depth_val

    # =========================================================
    # FULL 3D POSE PUBLISH
    # =========================================================
    def _pub_3d(self, label, x1, y1, x2, y2):

        result = self.depth_proc.bbox_to_3d(
            x1, y1, x2, y2, RGB_W, RGB_H)

        if result is None:
            return

        xc    = result['x']
        yc    = result['y']
        zc    = result['z']
        depth = result['depth']

        if depth < MIN_DEPTH or depth > MAX_DEPTH:
            self.get_logger().warn(
                f"⚠ Depth {depth:.2f}m out of range")
            return

        self.get_logger().info(
            f"3D cam: x={xc:.3f} y={yc:.3f} "
            f"z={zc:.3f} d={depth:.2f}m")

        # ─── Transform to base_link ─────────────────────
        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp = (
            self.get_clock().now().to_msg())
        pt.point.x = xc
        pt.point.y = yc
        pt.point.z = zc

        try:
            pt_b = self.tf_buffer.transform(
                pt, TARGET_FRAME,
                timeout=rclpy.duration.Duration(
                    seconds=0.5))

            bx = pt_b.point.x
            by = pt_b.point.y
            bz = pt_b.point.z

            pose = PoseStamped()
            pose.header.frame_id = TARGET_FRAME
            pose.header.stamp = (
                self.get_clock().now().to_msg())
            pose.pose.position.x = bx
            pose.pose.position.y = by
            pose.pose.position.z = bz
            pose.pose.orientation.w = 1.0
            self.pub_3d_pose.publish(pose)

            info = String()
            info.data = (
                f"{label} {bx:.3f} {by:.3f} "
                f"{bz:.3f} {depth:.3f}")
            self.pub_3d_info.publish(info)

            self.get_logger().info(
                f"🎯 {label} base: "
                f"x={bx:.3f} y={by:.3f} "
                f"z={bz:.3f} d={depth:.2f}m")

        except Exception as e:
            self.get_logger().warn(f"TF: {e}")

            # Fallback: camera-frame coords
            info = String()
            info.data = (
                f"{label} {xc:.3f} {yc:.3f} "
                f"{zc:.3f} {depth:.3f}")
            self.pub_3d_info.publish(info)


# =============================================================
# MAIN
# =============================================================
def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()