#!/usr/bin/env python3

# =============================================================
# depth_processor.py — Depth Processing Module (v2)
# Package: guardrail_pkg
#
# Pure computation — no ROS dependencies
# Converts 2D bbox + depth image → 3D position
#
# v2 changes:
#   - Added get_depth_at() alias for yolo_detector v4
#   - Added get_depth_at_rgb() with auto RGB→depth scaling
#   - Depth image staleness tracking
#   - Better edge-case handling for small/empty regions
# =============================================================

import time
import numpy as np


class DepthProcessor:

    def __init__(self, logger=None):
        self.logger = logger

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.intrinsics_ready = False

        # Depth image
        self.latest_depth = None
        self.depth_ready = False
        self.depth_timestamp = 0.0

        # Config
        self.min_depth = 0.1       # meters
        self.max_depth = 10.0      # meters
        self.sample_region = 5     # pixels radius for median
        self.stale_timeout = 2.0   # seconds

        # Resolution tracking
        self.depth_w = 0
        self.depth_h = 0

    def log(self, msg, level='info'):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    # =========================================================
    # SET INTRINSICS
    # =========================================================
    def set_intrinsics(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.intrinsics_ready = True
        self.log(
            f"📷 Intrinsics: fx={fx:.1f} fy={fy:.1f} "
            f"cx={cx:.1f} cy={cy:.1f}")

    def set_intrinsics_from_msg(self, camera_info_msg):
        k = camera_info_msg.k
        self.set_intrinsics(
            fx=k[0], fy=k[4], cx=k[2], cy=k[5])

    # =========================================================
    # UPDATE DEPTH IMAGE
    # =========================================================
    def update_depth(self, depth_np, encoding='32FC1'):
        if encoding == '16UC1':
            self.latest_depth = depth_np.astype(
                np.float32) / 1000.0
        elif encoding == '32FC1':
            self.latest_depth = depth_np.astype(
                np.float32)
        else:
            self.latest_depth = depth_np.astype(
                np.float32)
            if np.nanmean(self.latest_depth) > 100:
                self.latest_depth /= 1000.0

        self.depth_h, self.depth_w = (
            self.latest_depth.shape[:2])
        self.depth_timestamp = time.time()

        if not self.depth_ready:
            self.depth_ready = True
            self.log(
                f"✅ Depth ready: "
                f"{self.depth_w}x{self.depth_h}")

    # =========================================================
    # DEPTH STALENESS CHECK
    # =========================================================
    def is_depth_fresh(self):
        """Check if depth image is recent enough."""
        if not self.depth_ready:
            return False
        age = time.time() - self.depth_timestamp
        return age < self.stale_timeout

    # =========================================================
    # GET DEPTH AT PIXEL (depth image coordinates)
    # =========================================================
    def get_depth_at_pixel(self, px, py):
        """
        Get depth at a pixel in DEPTH IMAGE coordinates.
        Uses median of a small patch for robustness.

        Args:
            px: x pixel in depth image
            py: y pixel in depth image

        Returns:
            depth in meters, or None
        """
        if self.latest_depth is None:
            return None

        h, w = self.latest_depth.shape[:2]
        px = max(0, min(int(px), w - 1))
        py = max(0, min(int(py), h - 1))

        r = self.sample_region
        y_lo = max(0, py - r)
        y_hi = min(h, py + r + 1)
        x_lo = max(0, px - r)
        x_hi = min(w, px + r + 1)

        # Safety: ensure we have a region
        if y_hi <= y_lo or x_hi <= x_lo:
            return None

        region = self.latest_depth[y_lo:y_hi, x_lo:x_hi]

        valid = region[
            (region > self.min_depth) &
            (region < self.max_depth) &
            (~np.isnan(region)) &
            (~np.isinf(region))
        ]

        if len(valid) == 0:
            return None

        return float(np.median(valid))

    # =========================================================
    # GET DEPTH AT RGB PIXEL (auto-scales to depth resolution)
    # =========================================================
    def get_depth_at_rgb(self, rgb_x, rgb_y,
                         rgb_w=640, rgb_h=480):
        """
        Get depth at a pixel in RGB IMAGE coordinates.
        Automatically scales to depth image resolution.

        Args:
            rgb_x: x pixel in RGB image
            rgb_y: y pixel in RGB image
            rgb_w: RGB image width (default 640)
            rgb_h: RGB image height (default 480)

        Returns:
            depth in meters, or None
        """
        if self.latest_depth is None:
            return None

        # Scale RGB coords → depth coords
        dh, dw = self.latest_depth.shape[:2]
        dx = int(rgb_x * (dw / rgb_w))
        dy = int(rgb_y * (dh / rgb_h))

        return self.get_depth_at_pixel(dx, dy)

    # =========================================================
    # GET DEPTH AT — ALIAS (for yolo_detector v4 compatibility)
    # =========================================================
    def get_depth_at(self, cx, cy,
                     rgb_w=640, rgb_h=480):
        """
        Convenience alias used by yolo_detector v4.
        Accepts RGB pixel coordinates, auto-scales to depth.

        Args:
            cx: x pixel in RGB image
            cy: y pixel in RGB image
            rgb_w: RGB image width
            rgb_h: RGB image height

        Returns:
            depth in meters, or None
        """
        return self.get_depth_at_rgb(
            cx, cy, rgb_w, rgb_h)

    # =========================================================
    # BBOX → 3D IN CAMERA FRAME
    # =========================================================
    def bbox_to_3d(self, x1, y1, x2, y2,
                   rgb_w=640, rgb_h=480):
        """
        Convert YOLO bbox → 3D point in camera optical frame.

        Camera optical frame convention:
            x = right
            y = down
            z = forward (depth)

        Returns dict with x, y, z, depth or None.
        """
        if not self.intrinsics_ready:
            self.log("No intrinsics", 'warn')
            return None

        if self.latest_depth is None:
            self.log("No depth image", 'warn')
            return None

        # Bbox center in RGB coords
        cx_rgb = (x1 + x2) / 2.0
        cy_rgb = (y1 + y2) / 2.0

        # Scale to depth image resolution
        dh, dw = self.latest_depth.shape[:2]
        cx_d = int(cx_rgb * (dw / rgb_w))
        cy_d = int(cy_rgb * (dh / rgb_h))

        # Get robust depth
        depth_m = self.get_depth_at_pixel(cx_d, cy_d)
        if depth_m is None:
            self.log(
                f"No depth at ({cx_d},{cy_d})", 'warn')
            return None

        # Pinhole deprojection (using depth-image coords)
        z = depth_m
        x = (cx_d - self.cx) * z / self.fx
        y = (cy_d - self.cy) * z / self.fy

        self.log(
            f"3D camera: x={x:.3f} y={y:.3f} "
            f"z={z:.3f}m")

        return {
            'x': x, 'y': y, 'z': z,
            'depth': depth_m
        }

    # =========================================================
    # GET DEPTH STATS FOR BBOX (for debugging)
    # =========================================================
    def get_bbox_depth_stats(self, x1, y1, x2, y2,
                             rgb_w=640, rgb_h=480):
        """
        Get depth statistics for the entire bbox region.
        Useful for debugging depth quality.

        Returns dict with min, max, mean, median, valid_pct
        or None.
        """
        if self.latest_depth is None:
            return None

        dh, dw = self.latest_depth.shape[:2]

        # Scale to depth coords
        dx1 = int(x1 * (dw / rgb_w))
        dy1 = int(y1 * (dh / rgb_h))
        dx2 = int(x2 * (dw / rgb_w))
        dy2 = int(y2 * (dh / rgb_h))

        # Clamp
        dx1 = max(0, min(dx1, dw - 1))
        dy1 = max(0, min(dy1, dh - 1))
        dx2 = max(dx1 + 1, min(dx2, dw))
        dy2 = max(dy1 + 1, min(dy2, dh))

        region = self.latest_depth[dy1:dy2, dx1:dx2]
        total = region.size

        if total == 0:
            return None

        valid = region[
            (region > self.min_depth) &
            (region < self.max_depth) &
            (~np.isnan(region)) &
            (~np.isinf(region))
        ]

        if len(valid) == 0:
            return {
                'valid_pct': 0.0,
                'min': 0, 'max': 0,
                'mean': 0, 'median': 0
            }

        return {
            'min':       float(np.min(valid)),
            'max':       float(np.max(valid)),
            'mean':      float(np.mean(valid)),
            'median':    float(np.median(valid)),
            'valid_pct': len(valid) / total * 100.0
        }

    # =========================================================
    # STATUS CHECK
    # =========================================================
    def is_ready(self):
        return self.intrinsics_ready and self.depth_ready

    def get_status(self):
        age = 0.0
        if self.depth_timestamp > 0:
            age = time.time() - self.depth_timestamp

        return {
            'intrinsics': self.intrinsics_ready,
            'depth': self.depth_ready,
            'depth_age': age,
            'depth_fresh': self.is_depth_fresh(),
            'depth_resolution': (
                f"{self.depth_w}x{self.depth_h}"
                if self.depth_ready else "N/A"),
            'fx': self.fx,
            'fy': self.fy
        }