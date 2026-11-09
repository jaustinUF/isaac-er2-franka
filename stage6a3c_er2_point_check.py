"""
stage6a3c_er2_point_check.py -- Stage 6A.3c: ER 2 pointing recheck on the 5 cm cube.
PERCEPTION ONLY: no motion, no joint commands.

Why: 6A's tier 3 measured ER 2's pointing on the 8 cm cube. The cube is now 5 cm
(stage6a3c_camera_node.py --cube-size), so the top face is smaller in the image
(~22 x 7 px vs ~35 x 11), and the 6A pairing rule moves the deprojection plane:
    prompt "Point to the center of the top face of the red cube."  <->  z0 = top face
    = cube size (0.05). Change one without the other and the error returns.

What it does, per run (one cube position per node restart):
    1. Wait for a frame + gripper pose (arm at home), check the frame matches the
       calibration resolution.
    2. Encode the frame ONCE (PNG); save those exact bytes (er2_frame.png).
    3. Call ER 2 --calls times (default 3) on the SAME bytes: repeatability per image.
    4. For each answer: parse [y, x], deproject at z0, compare with the ORACLE top-face
       center (cube_x, cube_y, cube_size): pixel error, deprojection error (dx along
       the line of sight, dy lateral), and whether the point lies on the top face.

Code provenance (copied, not imported, from stage6a3b_er2_reach.py, unchanged
there): MODEL_ID, FORMAT_INSTRUCTION, PROMPT, extract_json_array(),
first_valid_point(), the ER 2 call, and the annotation style. deproject.py and
calib_dump.json are the 6A item-2 module and calibration (run
calib_20260925_161456), copied into this project unchanged.

Pass criteria (tier 3 reference, 8 cm: mean 3.5 mm, worst 16.9 mm, lateral ~1 mm):
    every point ON the top face (|dx|, |dy| < half the cube edge), which also rules
    out the "body center" choice: that point sits cube/2 below the plane and shows
    up as a large dx (~1.4 x 25 mm ~ 35 mm at this camera).

Artifacts: runs/er2check_<ts>_x<..>_y<..>/
    er2_frame.png, er2_annotated.png, er2_raw_<i>.txt, er2_check.json, er2_check.csv

RUN
    ./stage6a3c_camera_node.sh --cube-xy 0.5,0.0
    ./stage6a3c_er2_point_check.sh --cube-xy 0.5,0.0       (same --cube-xy)
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

import deproject   # 6A item-2 module, same directory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_TOPIC   = "/camera/image_raw"
GRIPPER_TOPIC = "/gripper_pose"
NODE_NAME     = "er2_point_check"

# --- ER 2 (VERBATIM from stage6a3b_er2_reach.py) ----------------------------
MODEL_ID = "gemini-robotics-er-2-preview"
FORMAT_INSTRUCTION = (
    ' The answer should follow the JSON format: '
    '[{"point": [y, x], "label": <label>}, ...]. '
    'The points are in [y, x] format normalized to 0-1000. '
    'Return only objects actually present in the image. Return only the JSON.'
)
PROMPT = "Point to the center of the top face of the red cube."

CUBE_SIZE_DEFAULT = 0.05
CALLS_DEFAULT = 3
PRE_SPIN_SEC = 1.0


class FrameGrabber(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.latest_frame = None
        self.latest_gripper = None
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self.create_subscription(PoseStamped, GRIPPER_TOPIC, self._on_gripper, 1)

    def _on_image(self, msg):
        if msg.encoding != "rgb8" or msg.step != msg.width * 3:
            return
        self.latest_frame = np.frombuffer(msg.data, np.uint8).reshape(
            msg.height, msg.width, 3).copy()

    def _on_gripper(self, msg):
        p = msg.pose.position
        self.latest_gripper = np.array([p.x, p.y, p.z], dtype=float)


def spin_for(node, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.05)


# ----------------------------------------------------------------------------
# ER 2 parser -- VERBATIM from stage6a3b_er2_reach.py (itself from stage5f)
# ----------------------------------------------------------------------------
def extract_json_array(text):
    if not text:
        return None, None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None, None
    sub = candidate[start:end + 1]
    try:
        parsed = json.loads(sub)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(parsed, list):
        return None, None
    return parsed, sub


def first_valid_point(parsed):
    valid = []
    for entry in parsed or []:
        if not isinstance(entry, dict) or "point" not in entry:
            continue
        pt = entry["point"]
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            continue
        try:
            y, x = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        if not (0.0 <= y <= 1000.0 and 0.0 <= x <= 1000.0):
            continue
        valid.append(([y, x], str(entry.get("label", "")).strip()))
    if not valid:
        return None, None, 0
    return valid[0][0], valid[0][1], len(valid)


def main():
    ap = argparse.ArgumentParser(description="6A.3c: ER 2 pointing recheck (perception only)")
    ap.add_argument("--cube-xy", dest="cube_xy", default="0.5,0.0",
                    help="ORACLE cube X,Y -- must match the node's --cube-xy")
    ap.add_argument("--cube-size", dest="cube_size", type=float, default=CUBE_SIZE_DEFAULT,
                    help="cube edge (m) -- must match the node's --cube-size")
    ap.add_argument("--z0", type=float, default=None,
                    help="deprojection plane (m); default = cube size (the top face), "
                         "PAIRED with the top-face prompt")
    ap.add_argument("--calls", type=int, default=CALLS_DEFAULT)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--calib", default=os.path.join(SCRIPT_DIR, "calib_dump.json"))
    ap.add_argument("--runs-dir", dest="runs_dir", default=os.path.join(SCRIPT_DIR, "runs"))
    args = ap.parse_args()
    cx, cy = (float(v) for v in args.cube_xy.split(","))
    z0 = args.cube_size if args.z0 is None else args.z0
    half = args.cube_size / 2.0
    oracle_top = np.array([cx, cy, args.cube_size])

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("[error] GEMINI_API_KEY not set.")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("[error] google-genai not importable on this interpreter.")

    cam = deproject.load_calibration(args.calib)
    ou, ov, _ = deproject.project(cam, oracle_top)[0]
    print(f"[info] calibration {args.calib}  {cam.width}x{cam.height}")
    print(f"[info] oracle top-face center {oracle_top.tolist()} -> pixel ({ou:.1f}, {ov:.1f}); "
          f"cube {args.cube_size * 1000:g} mm, z0 {z0:.3f} m, {args.calls} calls")

    tag = f"x{cx:+.2f}_y{cy:+.2f}".replace("+", "p").replace("-", "m")
    run_dir = os.path.join(args.runs_dir, datetime.now().strftime(f"er2check_%Y%m%d_%H%M%S_{tag}"))
    os.makedirs(run_dir, exist_ok=True)
    log = {"script": os.path.basename(__file__), "started": datetime.now().isoformat(timespec="seconds"),
           "cube_xy": [cx, cy], "cube_size_m": args.cube_size, "z0": z0, "prompt": args.prompt,
           "model": MODEL_ID, "calib": os.path.abspath(args.calib),
           "oracle_top_xyz": oracle_top.tolist(), "oracle_pixel_uv": [float(ou), float(ov)],
           "calls": []}

    rclpy.init()
    node = FrameGrabber()
    try:
        t0 = time.time()
        while node.latest_frame is None or node.latest_gripper is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - t0 > 30:
                sys.exit("[error] no frame/gripper pose in 30 s. Is stage6a3c_camera_node running?")
        spin_for(node, PRE_SPIN_SEC)
        frame = node.latest_frame
        log["gripper_xyz"] = node.latest_gripper.tolist()
        print(f"[info] gripper {np.round(node.latest_gripper, 4).tolist()} (home ~ [0.0844, 0, 0.9377])")
        H, W = frame.shape[:2]
        if (W, H) != (cam.width, cam.height):
            sys.exit(f"[error] frame {W}x{H} != calibration {cam.width}x{cam.height}")

        buf = io.BytesIO()
        PILImage.fromarray(frame).save(buf, format="PNG")
        png_bytes = buf.getvalue()
        with open(os.path.join(run_dir, "er2_frame.png"), "wb") as fh:
            fh.write(png_bytes)

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        prompt = args.prompt.rstrip() + FORMAT_INSTRUCTION
        for i in range(1, args.calls + 1):
            t1 = time.perf_counter()
            try:
                resp = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png"), prompt])
                raw, err = resp.text or "", None
            except Exception as e:
                raw, err = "", str(e)
            ms = (time.perf_counter() - t1) * 1000.0
            with open(os.path.join(run_dir, f"er2_raw_{i}.txt"), "w") as fh:
                fh.write(raw if not err else f"# CALL FAILED: {err}\n")
            rec = {"call": i, "api_ms": ms, "error": err}
            parsed, _ = extract_json_array(raw)
            yx, label, n_valid = first_valid_point(parsed)
            rec.update({"n_points_valid": n_valid, "label_log_only": label})
            if err or yx is None:
                rec["refused"] = err or "no parsable point"
                print(f"[call {i}] {ms:.0f} ms  REFUSED: {rec['refused']}")
                log["calls"].append(rec)
                continue
            u, v = deproject.er2_to_pixel(cam, yx)
            try:
                tgt = deproject.deproject_er2(cam, yx, z0)
            except ValueError as e:
                rec["refused"] = f"deprojection failed: {e}"
                print(f"[call {i}] REFUSED: {rec['refused']}")
                log["calls"].append(rec)
                continue
            d = np.asarray(tgt) - oracle_top
            on_top = bool(abs(d[0]) < half and abs(d[1]) < half)
            rec.update({"point_yx_1000": yx, "pixel_uv": [float(u), float(v)],
                        "pixel_error_px": float(np.hypot(u - ou, v - ov)),
                        "target_xyz": np.asarray(tgt).tolist(), "err_xyz_m": d.tolist(),
                        "err_mm": 1000 * float(np.linalg.norm(d[:2])), "on_top_face": on_top})
            print(f"[call {i}] {ms:.0f} ms  [y,x]=[{yx[0]:.0f},{yx[1]:.0f}]  pixel err "
                  f"{rec['pixel_error_px']:.1f} px  deproj err {rec['err_mm']:.1f} mm "
                  f"(dx {1000 * d[0]:+.1f}, dy {1000 * d[1]:+.1f})  "
                  f"{'on top face' if on_top else 'OFF THE TOP FACE'}  label '{label}' (log only)")
            log["calls"].append(rec)

        ok = [c for c in log["calls"] if "err_mm" in c]
        if ok:
            e = np.array([c["err_mm"] for c in ok])
            P = np.array([c["pixel_uv"] for c in ok])
            log["result"] = {"n_ok": len(ok), "n_calls": args.calls,
                             "mean_err_mm": float(e.mean()), "worst_err_mm": float(e.max()),
                             "repeat_spread_px": float(np.max(np.ptp(P, axis=0))),
                             "all_on_top_face": all(c["on_top_face"] for c in ok)}
            r = log["result"]
            print(f"\n[result] {r['n_ok']}/{args.calls} answers  mean {r['mean_err_mm']:.1f} mm  "
                  f"worst {r['worst_err_mm']:.1f} mm  repeat spread {r['repeat_spread_px']:.1f} px  "
                  f"{'ALL ON THE TOP FACE' if r['all_on_top_face'] else 'NOT ALL ON THE TOP FACE'}")
            print(f"[result] tier-3 reference (8 cm): mean 3.5 mm, worst 16.9 mm")
        else:
            log["result"] = {"n_ok": 0, "n_calls": args.calls}
            print("\n[result] no usable answers")

        # Annotated overlay: ER 2 points = red crosshairs (numbered), oracle = white square.
        img = PILImage.fromarray(frame)
        dr = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        r_ = 7
        dr.rectangle([ou - r_, ov - r_, ou + r_, ov + r_], outline=(255, 255, 255), width=2)
        dr.text((ou + r_ + 2, ov + r_), "oracle top", fill=(255, 255, 255), font=font)
        for c in ok:
            u, v = c["pixel_uv"]
            dr.line([(u - r_, v), (u + r_, v)], fill=(255, 0, 0), width=2)
            dr.line([(u, v - r_), (u, v + r_)], fill=(255, 0, 0), width=2)
            dr.text((u + r_ + 2, v - r_ - 10), str(c["call"]), fill=(255, 0, 0), font=font)
        img.save(os.path.join(run_dir, "er2_annotated.png"))
    finally:
        log["finished"] = datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(run_dir, "er2_check.json"), "w") as fh:
            json.dump(log, fh, indent=2)
        with open(os.path.join(run_dir, "er2_check.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["call", "api_ms", "er2_y", "er2_x", "pixel_u", "pixel_v", "pixel_err_px",
                        "target_x", "target_y", "target_z", "err_x_mm", "err_y_mm", "err_mm",
                        "on_top_face"])
            for c in log["calls"]:
                if "err_mm" not in c:
                    w.writerow([c["call"], f"{c['api_ms']:.4f}"] + [""] * 12)
                    continue
                w.writerow([c["call"], f"{c['api_ms']:.4f}",
                            *[f"{v:.4f}" for v in c["point_yx_1000"]],
                            *[f"{v:.4f}" for v in c["pixel_uv"]], f"{c['pixel_error_px']:.4f}",
                            *[f"{v:.4f}" for v in c["target_xyz"]],
                            f"{1000 * c['err_xyz_m'][0]:.4f}", f"{1000 * c['err_xyz_m'][1]:.4f}",
                            f"{c['err_mm']:.4f}", c["on_top_face"]])
        print(f"\n[done] artifacts in {run_dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
