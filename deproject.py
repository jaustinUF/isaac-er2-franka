"""
Stage 6A -- deproject.py  (NATIVE SIDE, pure numpy -- no Isaac, no ROS import)

Turns an ER 2 point  [y, x] in 0..1000  into a world point on a horizontal
plane z = z0, using the camera calibration written by stage6a_calib_dump.py.

This module is the runtime function for ladder item 3 (the orchestrator calls
deproject_er2()). Ladder item 2 tests it in isolation: tiers 0-2 in
stage6a_deproject_tiers.ipynb, then tier 3 (ER 2 in the chain).

THE PIPELINE (each stage is its own function so each can be tested alone)
    1. er2_to_pixel     [y, x] 0..1000       -> (u, v) pixels
    2. pixel_to_ray     (u, v)               -> unit ray in WORLD frame (via K^-1, R)
    3. ray_plane        camera center + ray  -> point where the ray meets z = z0
    project()           world point          -> (u, v)   (the forward model, for testing)

CONVENTIONS -- the whole point of this file is to never hand-convert axes
    * Pixel coordinates are EDGE-based, matching Isaac's K (cx = W/2, cy = H/2):
      u = 0 is the image's left edge; pixel column i has its CENTER at i + 0.5.
      ER 2's 0..1000 scale maps onto the same edge-based span (0 -> 0, 1000 -> W).
    * Camera frame is ROS optical: +X right, +Y down, +Z forward (optical axis).
      We take the rotation from Isaac's get_world_pose(camera_axes="ros"),
      read from the calibration JSON -- NOT built by hand from the USD quaternion.
      (Isaac's docstring calls "ros" "+Y up"; the code, U_R_TRANSFORM = diag(1,-1,-1),
      makes it +Y down. Verified against rendered pixels in ladder item 2.)
    * R_wc maps camera-frame vectors to world: d_world = R_wc @ d_cam.
      Its COLUMNS are the camera's right / down / forward axes in world coords.

KNOWN LIMITS (on the record, not bugs)
    * Pinhole only -- no lens distortion. True in this sim; false on hardware.
    * The plane height z0 is an ASSUMPTION the caller supplies. With the 6A
      camera (~34.5 deg depression) 1 cm of z0 error moves the answer ~1.46 cm
      along world x. For a top-face point on the 8 cm cube, z0 = 0.08.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def quat_wxyz_to_R(q) -> np.ndarray:
    """Scalar-first quaternion (w, x, y, z) -> 3x3 rotation matrix. Normalizes first."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray        # 3x3 intrinsics (fx, fy, cx, cy), edge-based pixels
    R_wc: np.ndarray     # 3x3, ROS-optical camera axes -> world
    C: np.ndarray        # (3,) camera center in world, meters
    width: int
    height: int

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)


def load_calibration(json_path: str | Path) -> CameraModel:
    """Build a CameraModel from calib_dump.json. Fails loudly if any input is missing."""
    J = json.loads(Path(json_path).read_text())
    c = J["calibration"]
    for key in ("K", "pose_ros", "resolution"):
        if not c[key]["ok"]:
            raise ValueError(f"calibration field '{key}' failed at dump time: {c[key]['error']}")
    if not c.get("square_pixel_check", {}).get("pass", False):
        raise ValueError(f"square-pixel check did not pass: {c.get('square_pixel_check')}")
    K = np.array(c["K"]["value"], dtype=float)
    pose = c["pose_ros"]["value"]
    W, H = c["resolution"]["value"]
    return CameraModel(K=K,
                       R_wc=quat_wxyz_to_R(pose["quat_wxyz"]),
                       C=np.array(pose["position"], dtype=float),
                       width=int(W), height=int(H))


# ----------------------------------------------------------------------------
# Stage 1: ER 2 normalized point -> pixel
# ----------------------------------------------------------------------------
def er2_to_pixel(cam: CameraModel, yx_1000) -> tuple[float, float]:
    """ER 2 returns [y, x] (ROWS FIRST) on a 0..1000 scale. -> (u, v) edge-based pixels."""
    y, x = float(yx_1000[0]), float(yx_1000[1])
    return x / 1000.0 * cam.width, y / 1000.0 * cam.height


# ----------------------------------------------------------------------------
# Forward model (used by the tests; also useful for drawing overlays)
# ----------------------------------------------------------------------------
def project(cam: CameraModel, P_world) -> np.ndarray:
    """World point(s) (n,3) -> (n,3) array of [u, v, z_depth]. z_depth > 0 means in front."""
    P = np.atleast_2d(np.asarray(P_world, dtype=float))
    p_cam = (P - cam.C) @ cam.R_wc          # row-vector form of R_wc.T @ (P - C)
    uvw = p_cam @ cam.K.T
    return np.column_stack([uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2], p_cam[:, 2]])


# ----------------------------------------------------------------------------
# Stage 2: pixel -> ray
# ----------------------------------------------------------------------------
def pixel_to_ray(cam: CameraModel, u: float, v: float) -> np.ndarray:
    """(u, v) -> unit direction in WORLD frame, from the camera center through that pixel."""
    d_cam = cam.K_inv @ np.array([u, v, 1.0])       # = [(u-cx)/fx, (v-cy)/fy, 1]
    d_world = cam.R_wc @ d_cam
    return d_world / np.linalg.norm(d_world)


# ----------------------------------------------------------------------------
# Stage 3: ray -> plane z = z0
# ----------------------------------------------------------------------------
def ray_plane(C, d, z0: float) -> np.ndarray:
    """Intersect ray C + t*d (t > 0) with the plane z = z0. Raises if it never gets there."""
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float)
    if abs(d[2]) < 1e-9:
        raise ValueError("ray is parallel to the plane (horizontal line of sight)")
    t = (z0 - C[2]) / d[2]
    if t <= 0:
        raise ValueError(f"plane z={z0} is behind the camera along this ray (t={t:.3f})")
    return C + t * d


# ----------------------------------------------------------------------------
# The composed functions the orchestrator calls
# ----------------------------------------------------------------------------
def deproject_pixel(cam: CameraModel, u: float, v: float, z0: float) -> np.ndarray:
    """(u, v) pixels -> world (x, y, z0)."""
    return ray_plane(cam.C, pixel_to_ray(cam, u, v), z0)


def deproject_er2(cam: CameraModel, yx_1000, z0: float) -> np.ndarray:
    """ER 2 [y, x] 0..1000 -> world (x, y, z0). The ladder-item-3 runtime call."""
    u, v = er2_to_pixel(cam, yx_1000)
    return deproject_pixel(cam, u, v, z0)


# ----------------------------------------------------------------------------
# Self-test: tier 0 (round trip) only -- proves the module loads and runs on the
# NATIVE interpreter with nothing but numpy. Tiers 1-2 live in the notebook.
# ----------------------------------------------------------------------------
def _selftest(json_path: str) -> int:
    cam = load_calibration(json_path)
    print(f"[selftest] calibration : {json_path}")
    print(f"[selftest] resolution  : {cam.width} x {cam.height}")
    print(f"[selftest] fx, fy      : {cam.K[0, 0]:.3f}, {cam.K[1, 1]:.3f}   cx, cy: {cam.K[0, 2]}, {cam.K[1, 2]}")
    print(f"[selftest] camera C    : {np.round(cam.C, 4).tolist()}")
    print(f"[selftest] forward axis: {np.round(cam.R_wc[:, 2], 4).tolist()}   "
          f"right axis: {np.round(cam.R_wc[:, 0], 4).tolist()}")

    J = json.loads(Path(json_path).read_text())
    worst = 0.0
    n = 0
    for p in J["positions"]:
        for name, P in p["world_points"].items():
            u, v, depth = project(cam, P)[0]
            back = deproject_pixel(cam, u, v, P[2])
            worst = max(worst, float(np.linalg.norm(back - np.asarray(P))))
            n += 1
    ok = worst < 1e-6
    print(f"[selftest] tier 0 round trip: {n} points, worst error {worst:.2e} m  -> "
          f"{'PASS' if ok else 'FAIL'} (limit 1e-6 m)")
    ex = deproject_er2(cam, (1000 * 554.0 / cam.height, 500.0), 0.08)
    print(f"[selftest] example: ER 2 [y,x]=[{1000*554/cam.height:.1f}, 500.0] at z0=0.08 -> "
          f"world ({ex[0]:.4f}, {ex[1]:.4f})   (default cube top face is at (0.5, 0.0))")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python3 deproject.py <path/to/calib_dump.json>")
        sys.exit(2)
    sys.exit(_selftest(sys.argv[1]))
