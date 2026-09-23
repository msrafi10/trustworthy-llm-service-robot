#!/usr/bin/env python3

# =============================================================
# arm_ik.py — Geometric IK for OpenManipulator-X (4-DOF) v2
# Package: guardrail_pkg
#
# v2 changes:
#   - Separate home vs carry poses
#   - Smarter pitch selection based on z-height
#   - Taller hover for pre-pick (bottle-safe)
#   - Added approach_from_above pose
#   - Joint limit validation on static poses
#   - Better error reporting
# =============================================================

import math


# ─── ARM LINK LENGTHS (meters) — OpenManipulator-X ──────────
L1 = 0.077    # base to shoulder
L2 = 0.130    # shoulder to elbow
L3 = 0.124    # elbow to wrist
L4 = 0.126    # wrist to gripper tip

MAX_REACH = L2 + L3 + L4   # 0.380m theoretical
PRACTICAL_REACH = 0.28      # safe working range

# ─── JOINT LIMITS (radians) ─────────────────────────────────
LIMITS = {
    'j1': (-2.827,  2.827),
    'j2': (-1.571,  1.571),
    'j3': (-1.571,  1.396),
    'j4': (-1.745,  1.222),
}


def clamp(val, lo, hi):
    return max(lo, min(val, hi))


def safe_acos(val):
    return math.acos(clamp(val, -1.0, 1.0))


def _check_limits(joints):
    """Validate all joints are within limits."""
    names = ['j1', 'j2', 'j3', 'j4']
    for i, (name, val) in enumerate(zip(names, joints)):
        lo, hi = LIMITS[name]
        if val < lo or val > hi:
            return False
    return True


def _solve_ik_single(x, y, z, pitch):
    """Single-pitch IK attempt. Returns [j1,j2,j3,j4] or None."""

    j1 = math.atan2(y, x)
    r = math.sqrt(x**2 + y**2)

    wr = r - L4 * math.cos(pitch)
    wz = z - L4 * math.sin(pitch) - L1

    D = math.sqrt(wr**2 + wz**2)

    if D > (L2 + L3):
        return None
    if D < abs(L2 - L3):
        return None

    cos3 = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
    j3 = math.pi - safe_acos(cos3)

    alpha = math.atan2(wz, wr)
    beta = safe_acos(
        (L2**2 + D**2 - L3**2) / (2 * L2 * D))
    j2 = -(alpha + beta)

    j4 = pitch - j2 - j3

    joints = [j1, j2, j3, j4]

    if not _check_limits(joints):
        return None

    # Clamp for floating-point edge cases
    joints = [
        clamp(j1, *LIMITS['j1']),
        clamp(j2, *LIMITS['j2']),
        clamp(j3, *LIMITS['j3']),
        clamp(j4, *LIMITS['j4']),
    ]

    return joints


def solve_ik(x, y, z, pitch=None):
    """
    Geometric IK — tries multiple pitch angles.
    If pitch is None, automatically selects pitches
    based on target z-height.

    IMPORTANT: x, y, z must be in ARM BASE frame.
    Returns [j1, j2, j3, j4] or None.
    """

    # ─── If explicit pitch given, try it first ──────────
    if pitch is not None:
        result = _solve_ik_single(x, y, z, pitch)
        if result is not None:
            return result

    # ─── Smart pitch selection based on height ──────────
    # Objects below arm base → need steeper downward pitch
    # Objects at arm height → near-horizontal pitch
    # Objects above → upward pitch
    if z < -0.05:
        # Floor level — steep down
        priority_pitches = [
            -math.pi / 3,
            -math.pi / 4,
            -math.pi / 2,
            -math.pi / 5,
            -math.pi / 6,
        ]
    elif z < 0.05:
        # Near arm base height
        priority_pitches = [
            -math.pi / 4,
            -math.pi / 6,
            -math.pi / 3,
            -math.pi / 8,
             0.0,
        ]
    else:
        # Above arm base
        priority_pitches = [
            -math.pi / 6,
             0.0,
            -math.pi / 8,
             math.pi / 8,
            -math.pi / 4,
        ]

    # ─── Also include the default pitch if not in list ──
    default = -math.pi / 4
    if pitch is not None and pitch not in priority_pitches:
        priority_pitches.insert(0, pitch)
    elif default not in priority_pitches:
        priority_pitches.append(default)

    for p in priority_pitches:
        result = _solve_ik_single(x, y, z, p)
        if result is not None:
            return result

    # ─── Last resort: sweep fine-grained ────────────────
    for deg in range(-80, 30, 5):
        p = math.radians(deg)
        result = _solve_ik_single(x, y, z, p)
        if result is not None:
            return result

    return None


def pre_pick_pose(x, y, z, hover=0.07):
    """
    Hover above object.
    v2: Increased default hover from 5cm to 7cm
    for tall bottle clearance.
    """
    return solve_ik(x, y, z + hover)


def pick_pose(x, y, z):
    """At object level for grasping."""
    return solve_ik(x, y, z)


def approach_pose(x, y, z, pullback=0.03):
    """
    v2 NEW: Slightly pulled-back pose before final descent.
    Useful to avoid knocking bottle over during approach.
    """
    r = math.sqrt(x**2 + y**2)
    if r < 0.01:
        return pre_pick_pose(x, y, z)

    # Pull back along the approach direction
    scale = (r - pullback) / r
    ax = x * scale
    ay = y * scale

    return solve_ik(ax, ay, z + 0.04)


def carry_pose():
    """
    Safe carry — arm tucked upward to avoid collisions
    during navigation. Slightly more upright than home.
    """
    joints = [0.0, -1.10, 0.40, 0.70]
    assert _check_limits(joints), \
        f"carry_pose {joints} violates limits!"
    return joints


def place_pose(height=0.05):
    """
    Place object 15cm forward at given height (arm frame).
    Returns IK solution or fallback static pose.
    """
    result = solve_ik(0.15, 0.0, height)
    if result is not None:
        return result

    # Try slightly higher
    result = solve_ik(0.15, 0.0, height + 0.03)
    if result is not None:
        return result

    # Static fallback
    fallback = [0.0, -0.5, 0.3, 0.2]
    return fallback


def home_pose():
    """
    Rest position — arm folded, gripper up.
    v2: Distinct from carry_pose.
    """
    joints = [0.0, -1.05, 0.35, 0.70]
    assert _check_limits(joints), \
        f"home_pose {joints} violates limits!"
    return joints


def get_workspace_info():
    """Return workspace limits for debugging."""
    return {
        'max_reach_horizontal': MAX_REACH,
        'practical_reach': PRACTICAL_REACH,
        'max_height': L1 + L2 + L3,
        'link_lengths': (L1, L2, L3, L4),
        'joint_limits': LIMITS,
    }


# ─── SELF TEST ───────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("ARM IK TEST v2 — OpenManipulator-X")
    print(f"  Links: L1={L1} L2={L2} L3={L3} L4={L4}")
    print(f"  Max reach: {MAX_REACH:.3f}m")
    print(f"  Practical: {PRACTICAL_REACH:.3f}m")
    print("=" * 55)

    # ── Verify static poses ──────────────────────────────
    print("\n── Static Pose Validation ──")
    for name, fn in [("home", home_pose),
                     ("carry", carry_pose)]:
        joints = fn()
        valid = _check_limits(joints)
        print(f"  {name:10s}: {joints} "
              f"{'✅' if valid else '❌ INVALID'}")

    pl = place_pose(0.05)
    print(f"  {'place':10s}: {pl} "
          f"{'✅ IK' if len(pl) == 4 else '❌'}")

    # ── Floor pick tests ─────────────────────────────────
    print("\n── Floor Pick Tests (arm frame) ──")
    tests = [
        (0.15, 0.00, -0.06, "15cm fwd, floor"),
        (0.18, 0.00, -0.06, "18cm fwd, floor"),
        (0.20, 0.00, -0.06, "20cm fwd, floor"),
        (0.22, 0.00, -0.06, "22cm fwd, floor"),
        (0.25, 0.00, -0.06, "25cm fwd, floor"),
        (0.28, 0.00, -0.06, "28cm fwd, floor (limit)"),
        (0.30, 0.00, -0.06, "30cm fwd — should fail"),
        (0.18, 0.05, -0.06, "18cm fwd, 5cm left"),
        (0.18, 0.00, -0.04, "18cm fwd, slightly up"),
    ]

    for x, y, z, desc in tests:
        r = solve_ik(x, y, z)
        if r:
            print(f"  ✅ {desc:35s} → "
                  f"[{r[0]:+.3f}, {r[1]:+.3f}, "
                  f"{r[2]:+.3f}, {r[3]:+.3f}]")
        else:
            print(f"  ❌ {desc:35s} → UNREACHABLE")

    # ── Pre-pick / approach tests ────────────────────────
    print("\n── Pre-pick & Approach Tests ──")
    ax, ay, az = 0.192, 0.0, -0.061
    print(f"  Target (arm): ({ax:.3f}, {ay:.3f}, {az:.3f})")

    for name, fn in [
        ("pre_pick", lambda: pre_pick_pose(ax, ay, az)),
        ("approach", lambda: approach_pose(ax, ay, az)),
        ("pick",     lambda: pick_pose(ax, ay, az)),
    ]:
        r = fn()
        if r:
            print(f"  ✅ {name:12s} → "
                  f"[{r[0]:+.3f}, {r[1]:+.3f}, "
                  f"{r[2]:+.3f}, {r[3]:+.3f}]")
        else:
            print(f"  ❌ {name:12s} → UNREACHABLE")

    # ── Real scenario ────────────────────────────────────
    print("\n── Real Scenario ──")
    print("  ARM offset: x=-0.092, z=0.091")
    print("  Bottle on floor at base_link (0.10, 0, 0.03)")
    bx, by, bz = 0.10, 0.0, 0.03
    ax = bx - (-0.092)
    ay = by - 0.0
    az = bz - 0.091
    print(f"  arm coords: ({ax:.3f}, {ay:.3f}, {az:.3f})")

    pre = pre_pick_pose(ax, ay, az)
    pk  = pick_pose(ax, ay, az)
    ap  = approach_pose(ax, ay, az)
    print(f"  Pre-pick:  {'✅' if pre else '❌'} {pre}")
    print(f"  Approach:  {'✅' if ap else '❌'} {ap}")
    print(f"  Pick:      {'✅' if pk else '❌'} {pk}")