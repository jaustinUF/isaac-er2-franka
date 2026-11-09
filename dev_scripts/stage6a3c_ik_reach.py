"""
stage6a3c_ik_reach.py -- Stage 6A.3c, sim validation: IK move_to a fingers-down hover
pose above an ORACLE cube position (no ER 2 yet).

Sequence (one target per sim-node restart):
    0. BASELINE   30 steps at home, no commands. Checks: arm really at home (restart
                  the node if not), and the MEASURED TCP (/gripper_pose + 0.1034 m
                  along the hand z) agrees with toolbox FK of the MEASURED joints.
                  That second check catches a quaternion-order mistake (~140 deg).
    1. SOLVE      IK for the hover, seeded from READY, using the SAME code and gates as
                  the offline step (imports stage6a3c_offline_predict, unchanged).
                  Refuse to move if any IK gate or path sweep fails.
    2. MOVE 1     home -> READY, min-jerk, streamed one waypoint per sim step.
    3. HOLD       READY_HOLD_STEPS at READY: measure how READY landed (droop).
    4. MOVE 2     READY -> hover, same profile.
    5. HOLD       HOVER_HOLD_STEPS at the hover; the last JUDGE_STEPS are averaged.
    6. JUDGE      tracking error = measured TCP vs the commanded target, in mm and
                  degrees, against 5 mm / 2 deg tilt / 2 deg heading.

PLANT-CLOCKED STREAMING (decided 2026-11-02)
    The node publishes /joint_states once per loop step, and each step advances
    physics 1/60 s of SIM time. Measured: ~82 steps per wall second (sim runs ~1.36x
    real time). So this script sends the next waypoint each time a /joint_states
    message arrives: the trajectory is sampled at 60 Hz of SIM time, durations and
    speeds mean what the offline prediction says, and exactly one command arrives
    per step (the node consumes one per step). All holds are counted in steps.
    EE analog: sampling synchronously off the plant's own conversion clock instead
    of a free-running timer.

The node is stage6a3b_camera_node.py, UNCHANGED: it already takes name-based
JointState on /joint_command and publishes /joint_states + /gripper_pose each step.

Artifacts: runs/ik_reach_<ts>/
    steps.csv     one row per sim step: phase, commanded q, actual q, hand pose, TCP
    summary.json  IK result, READY landing, final errors, PASS/FAIL, config
    final_frame.png

RUN (node restarted fresh before every run; same --cube-xy on both sides)
    ./stage6a3b_camera_node.sh --cube-xy 0.5,0.0
    ./stage6a3c_ik_reach.sh    --cube-xy 0.5,0.0
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped

import roboticstoolbox as rtb
from spatialmath import SE3

import stage6a3c_offline_predict as op   # same directory; imported, never modified

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
STATE_TOPIC   = "/joint_states"
GRIPPER_TOPIC = "/gripper_pose"
COMMAND_TOPIC = "/joint_command"
IMAGE_TOPIC   = "/camera/image_raw"
NODE_NAME     = "ik_reach_controller"
ARM_JOINTS    = [f"panda_joint{k}" for k in range(1, 8)]

SIM_HZ            = 60.0     # node's PHYSICS_DT = 1/60: one /joint_states per step
BASELINE_STEPS    = 30
READY_HOLD_STEPS  = 120      # 2 s sim
HOVER_HOLD_STEPS  = 180      # 3 s sim
JUDGE_STEPS       = 30       # averaged at the end of each hold

HOME_CHECK_RAD    = 0.01     # baseline: every arm joint within this of HOME_Q
FK_CHECK_MM       = 1.0      # baseline: measured TCP vs FK(measured q)
FK_CHECK_DEG      = 0.5
STATE_TIMEOUT_S   = 5.0      # wall seconds without /joint_states -> abort


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def quat_xyzw_to_R(x, y, z, w):
    """Rotation matrix from a ROS geometry_msgs quaternion, which is ordered
    x,y,z,w. (Isaac and spatialmath use w,x,y,z: reading one as the other is the
    silent ~140 deg error from exercise 2a.)"""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def mean_rotation(Rs):
    """Average of nearly identical rotations: mean the matrices, re-orthonormalize."""
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def steps_move(q_a, q_b):
    """Min-jerk joint move sampled once per SIM step. Duration from the offline
    rule (op.plan_move: peak-speed cap, rounded up), so T matches the prediction."""
    plan = op.plan_move(q_a, q_b)
    T = plan["T"]
    n = int(round(T * SIM_HZ))
    u = np.arange(1, n + 1) / n                      # first waypoint already moving
    Q = q_a + np.outer(op.min_jerk(u), q_b - q_a)
    return Q, T, plan["v_peak"]


# ----------------------------------------------------------------------------
# ROS node: plant-clocked streamer + logger
# ----------------------------------------------------------------------------
class IKReach(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.cmd_pub = self.create_publisher(JointState, COMMAND_TOPIC, 10)
        self.create_subscription(JointState, STATE_TOPIC, self._on_state, 10)
        self.create_subscription(PoseStamped, GRIPPER_TOPIC, self._on_gripper, 10)
        self.latest_hand = None            # (p (3,), R (3,3), quat xyzw (4,))
        self.n_states = 0
        self.last_state_wall = None
        self.program = []                  # list of (phase, Q or None, n_steps)
        self.pi = 0                        # program index
        self.k = 0                         # step within phase
        self.done = False
        self.last_cmd = None
        self.rows = []
        self.frame = None
        self.get_logger().info(f"sub {STATE_TOPIC} + {GRIPPER_TOPIC}, pub {COMMAND_TOPIC}")

    def _on_gripper(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.latest_hand = (np.array([p.x, p.y, p.z]),
                            quat_xyzw_to_R(o.x, o.y, o.z, o.w),
                            np.array([o.x, o.y, o.z, o.w]))

    def _on_image(self, msg):
        if msg.encoding == "rgb8" and msg.step == msg.width * 3:
            self.frame = np.frombuffer(msg.data, np.uint8).reshape(
                msg.height, msg.width, 3).copy()

    def send(self, q):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINTS
        msg.position = [float(v) for v in q]
        self.cmd_pub.publish(msg)
        self.last_cmd = np.asarray(q, dtype=float)

    def _on_state(self, msg):
        """One call per sim step: log, then advance the program by one step."""
        self.n_states += 1
        self.last_state_wall = time.time()
        if self.done or not self.program:
            return
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q_act = np.array([msg.position[idx[n]] for n in ARM_JOINTS])
        except KeyError:
            return
        phase, Q, n = self.program[self.pi]
        if Q is not None:                  # moving: send this step's waypoint
            self.send(Q[self.k])
        hand = self.latest_hand
        self.rows.append(dict(
            step=len(self.rows), phase=phase, k=self.k,
            cmd=None if self.last_cmd is None else self.last_cmd.copy(),
            act=q_act, hand=hand))
        self.k += 1
        if self.k >= n:
            self.pi += 1
            self.k = 0
            if self.pi >= len(self.program):
                self.done = True


def spin_until(node, cond, wall_timeout):
    t0 = time.time()
    while rclpy.ok() and not cond():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.last_state_wall and time.time() - node.last_state_wall > STATE_TIMEOUT_S:
            return "state timeout"
        if time.time() - t0 > wall_timeout:
            return "wall timeout"
    return None


# ----------------------------------------------------------------------------
# Measurement helpers on logged rows
# ----------------------------------------------------------------------------
def tcp_of(hand, tcp_off):
    p, R, _ = hand
    return p + R @ np.array([0.0, 0.0, tcp_off]), R


def phase_tail(rows, phase, n):
    sel = [r for r in rows if r["phase"] == phase and r["hand"] is not None]
    return sel[-n:]


def summarize_hold(rows, phase, tcp_off, panda):
    tail = phase_tail(rows, phase, JUDGE_STEPS)
    P = np.array([tcp_of(r["hand"], tcp_off)[0] for r in tail])
    R = mean_rotation(np.array([tcp_of(r["hand"], tcp_off)[1] for r in tail]))
    q_act = np.mean([r["act"] for r in tail], axis=0)
    q_cmd = tail[-1]["cmd"]
    T_fk = panda.fkine(q_act)
    return dict(tcp=P.mean(axis=0), tcp_sd_mm=1000 * float(np.max(P.std(axis=0))), R=R,
                q_act=q_act, q_cmd=q_cmd,
                droop=None if q_cmd is None else q_act - q_cmd,   # baseline: no command yet
                fk_vs_meas_mm=1000 * float(np.linalg.norm(T_fk.t - P.mean(axis=0))),
                fk_vs_meas_deg=op.angle_deg(T_fk.R, R))


def main():
    ap = argparse.ArgumentParser(description="6A.3c: IK move_to a hover pose (oracle target).")
    ap.add_argument("--cube-xy", dest="cube_xy", default="0.5,0.0",
                    help="cube X,Y -- must match the node's --cube-xy")
    ap.add_argument("--runs-dir", dest="runs_dir", default=os.path.join(SCRIPT_DIR, "runs"))
    args = ap.parse_args()
    cx, cy = (float(v) for v in args.cube_xy.split(","))

    panda = rtb.models.Panda()
    tcp_off = float(panda.grippers[0].tool.t[2])
    print(f"[info] roboticstoolbox {rtb.__version__}, numpy {np.__version__}, "
          f"TCP offset {tcp_off:.4f} m along panda_hand z")

    # ---- SOLVE (offline code, before anything moves) -------------------------
    psi_ready = op.heading_deg(panda.fkine(op.READY_Q).R)
    psi = min((0.0, 90.0, 180.0, -90.0), key=lambda c: abs(op.wrap180(c - psi_ready)))
    T_goal = SE3.Rt(op.fingers_down_R(psi), [cx, cy, op.CUBE_TOP_Z + op.HOVER_M])
    ik = op.solve(panda, T_goal, op.READY_Q)
    print(f"[solve] cube ({cx:.2f}, {cy:+.2f}) -> TCP target "
          f"{np.round(T_goal.t, 4).tolist()}, heading {psi:+.0f} deg")
    print(f"[solve] success {ik['success']}  iters {ik['iterations']}  searches "
          f"{ik['searches']}  FK err {ik['pos_mm']:.4f} mm  margin {ik['min_margin']:.4f} "
          f"(j{ik['limiting_joint']})  gates {ik['gates']}")
    Q1_plan = op.plan_move(op.HOME_Q, op.READY_Q)
    Q2_plan = op.plan_move(op.READY_Q, ik["q"])
    s1 = op.sweep(panda, Q1_plan["Q"], cx, cy)
    s2 = op.sweep(panda, Q2_plan["Q"], cx, cy)
    if not (ik["ik_ok"] and s1["ok"] and s2["ok"]):
        sys.exit(f"[solve] REFUSE: IK gates {ik['gates']}, sweep1 ok {s1['ok']}, "
                 f"sweep2 ok {s2['ok']} -- no motion.")

    run_dir = os.path.join(args.runs_dir, datetime.now().strftime("ik_reach_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    summary = {"script": os.path.basename(__file__),
               "started": datetime.now().isoformat(timespec="seconds"),
               "cube_xy": [cx, cy], "target_tcp": T_goal.t.tolist(), "heading_deg": psi,
               "ik": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in ik.items()},
               "config": {"sim_hz": SIM_HZ, "ready_hold_steps": READY_HOLD_STEPS,
                          "hover_hold_steps": HOVER_HOLD_STEPS, "judge_steps": JUDGE_STEPS,
                          "v_peak_cap": op.V_PEAK_CAP, "tol_pos_mm": op.TOL_POS_MM,
                          "tol_tilt_deg": op.TOL_TILT_DEG,
                          "tol_heading_deg": op.TOL_HEADING_DEG, "tcp_offset_m": tcp_off}}

    rclpy.init()
    node = IKReach()
    passed = False
    try:
        # ---- 0. BASELINE ----------------------------------------------------
        node.program = [("baseline", None, BASELINE_STEPS)]
        print("[info] waiting for the sim node...")
        err = spin_until(node, lambda: node.done, 60.0)
        if err:
            sys.exit(f"[error] baseline: {err}. Is stage6a3b_camera_node running?")
        base = summarize_hold(node.rows, "baseline", tcp_off, panda)
        home_dev = float(np.max(np.abs(base["q_act"] - op.HOME_Q)))
        print(f"[baseline] max |q - HOME_Q| {home_dev:.4f} rad   measured TCP "
              f"{np.round(base['tcp'], 4).tolist()}")
        print(f"[baseline] FK(measured q) vs measured TCP: {base['fk_vs_meas_mm']:.4f} mm, "
              f"{base['fk_vs_meas_deg']:.4f} deg   (quaternion-order / model check)")
        summary["baseline"] = {"max_home_dev_rad": home_dev, "tcp": base["tcp"].tolist(),
                               "fk_vs_meas_mm": base["fk_vs_meas_mm"],
                               "fk_vs_meas_deg": base["fk_vs_meas_deg"]}
        if home_dev > HOME_CHECK_RAD:
            sys.exit("[baseline] REFUSE: arm is not at home. Restart the sim node.")
        if base["fk_vs_meas_mm"] > FK_CHECK_MM or base["fk_vs_meas_deg"] > FK_CHECK_DEG:
            sys.exit("[baseline] REFUSE: measured TCP disagrees with FK -- check the "
                     "quaternion order / TCP offset before moving.")

        # ---- 1-5. MOVES, plant-clocked -------------------------------------
        Q1, T1, v1 = steps_move(base["q_act"], op.READY_Q)   # start where the arm IS
        Q2, T2, v2 = steps_move(op.READY_Q, ik["q"])
        print(f"[move] home->ready {T1:.1f} s sim ({len(Q1)} steps, peak "
              f"{v1.max():.4f} rad/s), ready->hover {T2:.1f} s sim ({len(Q2)} steps, "
              f"peak {v2.max():.4f} rad/s)")
        hold_ready = np.repeat(op.READY_Q[None, :], READY_HOLD_STEPS, axis=0)
        hold_hover = np.repeat(ik["q"][None, :], HOVER_HOLD_STEPS, axis=0)
        node.program = [("move_ready", Q1, len(Q1)), ("hold_ready", hold_ready, READY_HOLD_STEPS),
                        ("move_hover", Q2, len(Q2)), ("hold_hover", hold_hover, HOVER_HOLD_STEPS)]
        node.pi, node.k, node.done = 0, 0, False
        n0, w0 = node.n_states, time.time()
        err = spin_until(node, lambda: node.done, 120.0)
        wall = time.time() - w0
        steps = node.n_states - n0
        if err:
            print(f"[error] during motion: {err} -- partial log written.")
            summary["error"] = err
        print(f"[move] {steps} steps in {wall:.2f} s wall -> {steps / max(wall, 1e-9):.1f} "
              f"steps/s, sim/wall {steps / SIM_HZ / max(wall, 1e-9):.2f}x")
        summary["timing"] = {"steps": steps, "wall_s": wall}

        # ---- JUDGE ----------------------------------------------------------
        if not err:
            rdy = summarize_hold(node.rows, "hold_ready", tcp_off, panda)
            T_rdy = panda.fkine(op.READY_Q)
            rdy_mm = 1000 * float(np.linalg.norm(rdy["tcp"] - T_rdy.t))
            print(f"\n[ready] measured TCP {np.round(rdy['tcp'], 4).tolist()} vs FK(READY) "
                  f"{np.round(T_rdy.t, 4).tolist()}: {rdy_mm:.4f} mm")
            print(f"[ready] joint droop (actual - commanded, mrad): "
                  + " ".join(f"j{k + 1} {1000 * d:+.3f}" for k, d in enumerate(rdy["droop"])))

            hov = summarize_hold(node.rows, "hold_hover", tcp_off, panda)
            pos_err = hov["tcp"] - T_goal.t
            pos_mm = 1000 * float(np.linalg.norm(pos_err))
            tilt = op.tilt_deg(hov["R"])
            head = abs(op.wrap180(op.heading_deg(hov["R"]) - psi))
            passed = pos_mm <= op.TOL_POS_MM and tilt <= op.TOL_TILT_DEG and head <= op.TOL_HEADING_DEG
            print(f"\n[hover] target  {np.round(T_goal.t, 4).tolist()}")
            print(f"[hover] measured TCP {np.round(hov['tcp'], 4).tolist()}  "
                  f"(sd over last {JUDGE_STEPS} steps {hov['tcp_sd_mm']:.4f} mm)")
            print(f"[hover] TRACKING ERROR {pos_mm:.4f} mm (dx {1000 * pos_err[0]:+.4f}, "
                  f"dy {1000 * pos_err[1]:+.4f}, dz {1000 * pos_err[2]:+.4f})   tilt "
                  f"{tilt:.4f} deg   heading {head:.4f} deg")
            print(f"[hover] joint droop (mrad): "
                  + " ".join(f"j{k + 1} {1000 * d:+.3f}" for k, d in enumerate(hov["droop"])))
            print(f"[hover] FK(actual q) vs measured TCP {hov['fk_vs_meas_mm']:.4f} mm "
                  f"(model agreement at the hover)")
            print(f"\n[result] {'PASS' if passed else 'FAIL'}  (tolerance {op.TOL_POS_MM:g} mm / "
                  f"{op.TOL_TILT_DEG:g} deg tilt / {op.TOL_HEADING_DEG:g} deg heading)")
            summary["ready"] = {"tcp": rdy["tcp"].tolist(), "err_vs_fk_mm": rdy_mm,
                                "droop_rad": rdy["droop"].tolist()}
            summary["hover"] = {"tcp": hov["tcp"].tolist(), "tcp_sd_mm": hov["tcp_sd_mm"],
                                "err_xyz_m": pos_err.tolist(), "err_mm": pos_mm,
                                "tilt_deg": tilt, "heading_err_deg": head,
                                "droop_rad": hov["droop"].tolist(),
                                "q_act": hov["q_act"].tolist(),
                                "fk_vs_meas_mm": hov["fk_vs_meas_mm"]}
            summary["PASS"] = passed

        # ---- final frame (subscribe only now: 2.7 MB frames) -----------------
        node.create_subscription(Image, IMAGE_TOPIC, node._on_image, 1)
        spin_until(node, lambda: node.frame is not None, 10.0)
        if node.frame is not None:
            from PIL import Image as PILImage
            PILImage.fromarray(node.frame).save(os.path.join(run_dir, "final_frame.png"))
    finally:
        # ---- artifacts --------------------------------------------------------
        hdr = (["step", "phase", "k", "t_sim_s"] + [f"cmd_q{k}" for k in range(1, 8)]
               + [f"act_q{k}" for k in range(1, 8)] + ["hand_x", "hand_y", "hand_z",
               "hand_qx", "hand_qy", "hand_qz", "hand_qw", "tcp_x", "tcp_y", "tcp_z"])
        out = []
        for r in node.rows:
            cmd = r["cmd"] if r["cmd"] is not None else [float("nan")] * 7
            if r["hand"] is not None:
                tcp, _ = tcp_of(r["hand"], tcp_off)
                hand = [*r["hand"][0], *r["hand"][2], *tcp]
            else:
                hand = [float("nan")] * 10
            out.append([r["step"], r["phase"], r["k"], r["step"] / SIM_HZ,
                        *[float(v) for v in cmd], *[float(v) for v in r["act"]],
                        *[float(v) for v in hand]])
        op.write_csv(os.path.join(run_dir, "steps.csv"), hdr, out)
        summary["finished"] = datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(run_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=lambda o: o.tolist()
                      if isinstance(o, np.ndarray) else str(o))
        print(f"\n[done] artifacts in {run_dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
