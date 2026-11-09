"""
stage6a3c_offline_predict.py -- Stage 6A.3c, validation step 0: predict every move
BEFORE the sim sees it. Toolbox only: no Isaac, no ROS, no network.

Origin: the target/solve/verify pattern and the 4 dp CSV writer are copied from
IK_FK_scrach/ex2b_panda_ikine_lm.py (unchanged there). What changed, per the 3c
design decisions (2026-11-01):
  * HEADING is fixed in the world and aligned with the cube's faces (ex2b turned
    home's heading by the cube's bearing). Of the four face-aligned headings
    (the cube looks the same every 90 deg) we use the one closest to READY's.
  * SEED is the named READY pose (ex2b used toolbox qr / home).
  * tol = 1e-10 (ex2b used the 1e-6 default, ~1.4 mm). Acceptance is still OUR
    check: FK in mm and degrees, joint-limit margins, distance from the seed.
  * Adds the two interpolated moves (home -> ready -> hover) and a path sweep.

Terms used here:
  * Minimum-jerk profile: a smooth 0 -> 1 ramp s(t) = 10u^3 - 15u^4 + 6u^5, u = t/T.
    Velocity and acceleration are zero at both ends. EE analog: a setpoint ramp /
    rate limiter, shaped so the step never "kicks" the drives. Peak velocity is
    1.875 * delta_q / T, at mid-move.
  * Joint-space interpolation: every joint follows the same s(t), so all finish
    together. The hand path is curved; that's fine in free space.
  * Path sweep: FK at every 50 Hz waypoint; record how close the arm's sampled
    points come to the floor and to the cube. These are POINTS (link origins and
    the TCP), not link volumes, so the clearance margins below are deliberately
    generous. It's a screen, not a collision checker.

Steps:
  1. READY check: FK of READY. Fingers straight down? Heading? Limit margins?
  2. Pick the face-aligned heading nearest READY's.
  3. IK for the hover over all 9 calibration-grid cubes, seeded from READY.
  4. Moves: home -> ready, ready -> hover. Duration from a peak-velocity cap.
  5. Path sweep of both moves.
  6. Mirror check for the y = +/-0.30 pairs (compare the ARM'S SHAPE, not joint
     values: redundancy can split the same shape differently between j1/j3/j7).

Output: runs/predict_<ts>/ready_check.csv, grid_predict.csv, mirror_check.csv,
        traj_centerline.csv, predict.png
"""

import csv
import math
import os
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3

# ======================================================================
# Parameters
# ======================================================================
HOME_Q = np.array([   # /joint_states at home, 2026-10-30 (same as ex2a/ex2b)
    0.000515350722707808, -0.05362680181860924, 1.910451646836009e-05,
    -0.0697999894618988, -0.00025607398129068315, 0.08609218150377274,
    0.08695203810930252,
])
# Franka's conventional "ready" pose (bent elbow, fingers down). Verified in step 1.
READY_Q = np.array([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])

CUBE_SIZE = 0.08                     # m, cube sits on the floor (z = 0)
CUBE_TOP_Z = CUBE_SIZE               # top face
HOVER_M = 0.10                       # TCP height above the top face (decided 2026-11-01)
GRID_X = (0.35, 0.50, 0.65)          # 6A calibration grid (calib_20260925_161456)
GRID_Y = (-0.30, 0.0, 0.30)

IK_TOL = 1e-10                       # ikine_LM tol on E = 1/2|e|^2 -> ~14 um / urad
RNG_SEED = 0                         # makes any random restarts repeatable

# "Reached" tolerances (provisional, decided 2026-11-01): applied to the measured TCP
# in the sim. Here they gate the PREDICTION, which should beat them by orders.
TOL_POS_MM = 5.0
TOL_TILT_DEG = 2.0
TOL_HEADING_DEG = 2.0

# Gates on the IK answer
LIMIT_MARGIN_MIN = 0.05              # rad (~3 deg) from every joint limit
SEED_JUMP_MAX = 2.0                  # rad: no joint may move more than this from READY
                                     # (catches a random restart that wandered off;
                                     # legitimate grid answers move <= ~1.5 rad)

# Moves
V_PEAK_CAP = 0.6                     # rad/s peak joint speed (~25% of Franka's limits)
T_ROUND = 0.5                        # s: durations rounded UP to this step
STREAM_HZ = 50.0                     # setpoint rate the native side will stream at

# Path-sweep clearance screens (points, not volumes -> generous)
FLOOR_CLEAR_MIN = 0.10               # m: link origins (above the base) above the floor
TCP_FLOOR_CLEAR_MIN = 0.03           # m: TCP above the floor
CUBE_CLEAR_MIN = 0.03                # m: TCP outside the cube box by at least this
# Frames screened against the floor: everything above the shoulder (link0/link1 sit
# at the base by design), plus the hand and fingers.
SWEEP_FRAMES = ("panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7",
                "panda_link8", "panda_hand", "panda_leftfinger", "panda_rightfinger")


# ======================================================================
# Geometry helpers
# ======================================================================
def tilt_deg(R):
    """Angle between the TCP z-axis and straight down."""
    return math.degrees(math.acos(np.clip(-R[2, 2], -1.0, 1.0)))


def heading_deg(R):
    """Heading of the TCP x-axis projected onto the floor, degrees from world +x."""
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def fingers_down_R(psi_deg):
    """TCP z straight down, TCP x at heading psi."""
    p = math.radians(psi_deg)
    x = np.array([math.cos(p), math.sin(p), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def angle_deg(R1, R2):
    R = R1.T @ R2
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def limit_margins(panda, q):
    lo, hi = panda.qlim
    return np.minimum(q - lo, hi - q)


def cube_box_distance(p, cx, cy):
    """Distance from point p to the cube's box (0 if inside)."""
    h = CUBE_SIZE / 2
    lo = np.array([cx - h, cy - h, 0.0])
    hi = np.array([cx + h, cy + h, CUBE_SIZE])
    d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    return float(np.linalg.norm(d))


# ======================================================================
# Moves
# ======================================================================
def min_jerk(u):
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def plan_move(q_a, q_b):
    """Joint-space min-jerk move; duration set by the peak-velocity cap."""
    dq_max = float(np.max(np.abs(q_b - q_a)))
    T = max(T_ROUND, math.ceil((1.875 * dq_max / V_PEAK_CAP) / T_ROUND) * T_ROUND)
    n = int(round(T * STREAM_HZ))
    t = np.arange(n + 1) / STREAM_HZ
    s = min_jerk(t / T)
    Q = q_a + np.outer(s, q_b - q_a)
    v_peak = 1.875 * np.abs(q_b - q_a) / T
    return dict(T=T, t=t, Q=Q, v_peak=v_peak, dq_max=dq_max)


def sweep(panda, Q, cx, cy):
    """Point-sampled clearance over a move."""
    link_floor = np.inf
    link_floor_name = ""
    tcp_floor = np.inf
    tcp_cube = np.inf
    for q in Q:
        # Named frames, one FK call each (fkine_all returns one frame per elementary
        # transform, 13 for this model, which doesn't map 1:1 onto link names).
        for name in SWEEP_FRAMES:
            p = panda.fkine(q, end=name).t
            if p[2] < link_floor:
                link_floor, link_floor_name = float(p[2]), name
        tcp = panda.fkine(q).t
        tcp_floor = min(tcp_floor, float(tcp[2]))
        tcp_cube = min(tcp_cube, cube_box_distance(tcp, cx, cy))
    ok = (link_floor >= FLOOR_CLEAR_MIN and tcp_floor >= TCP_FLOOR_CLEAR_MIN
          and tcp_cube >= CUBE_CLEAR_MIN)
    return dict(link_floor=link_floor, link_floor_name=link_floor_name,
                tcp_floor=tcp_floor, tcp_cube=tcp_cube, ok=ok)


# ======================================================================
# Solve and verify
# ======================================================================
def solve(panda, T, q0):
    t0 = time.perf_counter()
    s = panda.ikine_LM(T, q0=q0, seed=RNG_SEED, tol=IK_TOL)
    ms = (time.perf_counter() - t0) * 1000.0
    q = np.array(s.q)
    Tq = panda.fkine(q)
    m = limit_margins(panda, q)
    jump = np.abs(q - q0)
    pos_mm = 1000 * float(np.linalg.norm(Tq.t - T.t))
    tilt = tilt_deg(Tq.R)
    head_err = abs(wrap180(heading_deg(Tq.R) - heading_deg(T.R)))
    gates = dict(
        solver=bool(s.success),
        pos=pos_mm <= TOL_POS_MM,
        tilt=tilt <= TOL_TILT_DEG,
        heading=head_err <= TOL_HEADING_DEG,
        limits=float(m.min()) >= LIMIT_MARGIN_MIN,
        seed_jump=float(jump.max()) <= SEED_JUMP_MAX,
    )
    return dict(
        success=bool(s.success), iterations=int(s.iterations), searches=int(s.searches),
        ms=ms, q=q, tcp=Tq.t.copy(), pos_mm=pos_mm, ori_deg=angle_deg(Tq.R, T.R),
        tilt_deg=tilt, heading_err_deg=head_err,
        min_margin=float(m.min()), limiting_joint=int(np.argmin(m)) + 1,
        max_jump=float(jump.max()), max_jump_joint=int(np.argmax(jump)) + 1,
        gates=gates, ik_ok=all(gates.values()),
    )


# ======================================================================
# Formatting (4 dp rule)
# ======================================================================
def f4(v):
    v = float(v)
    if v == 0.0:
        return "0.0000"
    if 1e-3 <= abs(v) < 1e5:
        return f"{v:.4f}"
    return f"{v:.4e}"


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f4(x) if isinstance(x, (float, np.floating)) else x for x in r])


# ======================================================================
# Plot
# ======================================================================
def arm_pts(panda, q):
    pts = panda.fkine_all(q).t
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > 1e-6:
            keep.append(p)
    keep.append(panda.fkine(q).t)
    return np.array(keep)


def make_plot(run_dir, panda, psi, results, center):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # (a) side view, centerline: home, ready, hover, and the TCP path
    ax = axs[0, 0]
    for q, col, lab in ((HOME_Q, "0.6", "home"), (READY_Q, "tab:orange", "ready"),
                        (center["ik"]["q"], "tab:blue", "hover (IK)")):
        a = arm_pts(panda, q)
        ax.plot(a[:, 0], a[:, 2], "-o", color=col, lw=2.5, ms=3, label=lab)
    for mv, col in ((center["m1"], "tab:orange"), (center["m2"], "tab:blue")):
        P = np.array([panda.fkine(q).t for q in mv["Q"]])
        ax.plot(P[:, 0], P[:, 2], ":", color=col, lw=1.5)
    ax.add_patch(plt.Rectangle((0.5 - CUBE_SIZE / 2, 0), CUBE_SIZE, CUBE_SIZE,
                               color="tab:red", alpha=0.35))
    ax.axhline(0, color="0.4", lw=1)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 0.9)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("(a) centerline cube, side view (x-z): home, ready, hover\n"
                 "dotted = TCP path of each interpolated move", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # (b) top view: grid, pass/fail, max joint move from READY
    ax = axs[0, 1]
    for r in results:
        ok = r["ik"]["ik_ok"] and r["s1"]["ok"] and r["s2"]["ok"]
        ax.add_patch(plt.Rectangle((r["cx"] - CUBE_SIZE / 2, r["cy"] - CUBE_SIZE / 2),
                                   CUBE_SIZE, CUBE_SIZE, color="tab:red", alpha=0.2))
        ax.plot(r["cx"], r["cy"], "o" if ok else "X", ms=10,
                color="tab:green" if ok else "tab:red", mfc="none" if ok else None, mew=2)
        ax.annotate(f"m {r['ik']['min_margin']:.2f} j{r['ik']['limiting_joint']}\n"
                    f"T {r['m2']['T']:.1f} s",
                    (r["cx"], r["cy"]), xytext=(0, -26), textcoords="offset points",
                    ha="center", fontsize=7.5)
    ax.set_aspect("equal")
    ax.set_xlim(0.25, 0.75)
    ax.set_ylim(-0.42, 0.42)
    ax.set_xlabel("cube x (m)")
    ax.set_ylabel("cube y (m)")
    ax.set_title(f"(b) 9-point grid, top view, heading {psi:+.0f} deg\n"
                 "m = smallest joint-limit margin (rad) and its joint; T = ready->hover time",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

    # (c) centerline joint trajectories, home -> ready -> hover
    ax = axs[1, 0]
    m1, m2 = center["m1"], center["m2"]
    t = np.concatenate([m1["t"], m1["T"] + m2["t"][1:]])
    Q = np.vstack([m1["Q"], m2["Q"][1:]])
    for k in range(7):
        ax.plot(t, Q[:, k], lw=1.8, label=f"joint{k + 1}")
    ax.axvline(m1["T"], color="0.5", ls="--", lw=1)
    ax.text(m1["T"] + 0.05, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0, "ready",
            fontsize=8, color="0.4")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint target (rad)")
    ax.set_title(f"(c) centerline setpoint stream at {STREAM_HZ:g} Hz, min-jerk profile\n"
                 f"home->ready {m1['T']:.1f} s, ready->hover {m2['T']:.1f} s "
                 f"(peak speed cap {V_PEAK_CAP:g} rad/s)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    ax.grid(True, alpha=0.3)

    # (d) per-point stress: limit margin and max joint move
    ax = axs[1, 1]
    lab = [f"({r['cx']:.2f},{r['cy']:+.2f})" for r in results]
    xx = np.arange(len(results))
    ax.bar(xx - 0.2, [r["ik"]["min_margin"] for r in results], 0.4,
           color="tab:purple", label="smallest limit margin (rad)")
    ax.bar(xx + 0.2, [r["ik"]["max_jump"] for r in results], 0.4,
           color="tab:cyan", label="largest joint move from ready (rad)")
    ax.axhline(LIMIT_MARGIN_MIN, color="tab:purple", ls=":", lw=1)
    ax.axhline(SEED_JUMP_MAX, color="tab:cyan", ls=":", lw=1)
    ax.set_xticks(xx)
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("rad")
    ax.set_title("(d) stress per grid point (dotted = gate)\n"
                 "use this to choose which points get sim runs", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"6A.3c offline prediction (roboticstoolbox {rtb.__version__}): "
                 f"TCP {HOVER_M:g} m above an {CUBE_SIZE * 100:g} cm cube, fingers down, "
                 f"heading {psi:+.0f} deg, seed READY", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(run_dir, "predict.png")
    fig.savefig(path, dpi=110)
    return path


# ======================================================================
# Run
# ======================================================================
def main():
    print(f"roboticstoolbox {rtb.__version__}, numpy {np.__version__}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"predict_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    panda = rtb.models.Panda()

    # ---- 1. READY check --------------------------------------------------
    Tr = panda.fkine(READY_Q)
    Th = panda.fkine(READY_Q, end="panda_hand")
    rm = limit_margins(panda, READY_Q)
    psi_ready = heading_deg(Tr.R)
    print("\n1. READY check")
    print(f"   TCP        ({Tr.t[0]:.4f}, {Tr.t[1]:.4f}, {Tr.t[2]:.4f}) m")
    print(f"   panda_hand ({Th.t[0]:.4f}, {Th.t[1]:.4f}, {Th.t[2]:.4f}) m")
    print(f"   tilt from straight down {tilt_deg(Tr.R):.4f} deg, heading {psi_ready:.4f} deg")
    print(f"   smallest limit margin {rm.min():.4f} rad (joint{int(np.argmin(rm)) + 1})")
    write_csv(os.path.join(run_dir, "ready_check.csv"),
              ["tcp_x", "tcp_y", "tcp_z", "hand_x", "hand_y", "hand_z", "tilt_deg",
               "heading_deg", "min_limit_margin_rad", "limiting_joint"]
              + [f"q{k}_rad" for k in range(1, 8)],
              [[*Tr.t, *Th.t, tilt_deg(Tr.R), psi_ready, float(rm.min()),
                int(np.argmin(rm)) + 1, *READY_Q]])

    # ---- 2. heading --------------------------------------------------------
    cands = (0.0, 90.0, 180.0, -90.0)
    psi = min(cands, key=lambda c: abs(wrap180(c - psi_ready)))
    R_hover = fingers_down_R(psi)
    print(f"\n2. Face-aligned heading nearest READY's: {psi:+.0f} deg "
          f"({abs(wrap180(psi - psi_ready)):.4f} deg of wrist turn from READY)")

    # ---- 3-5. per grid point -----------------------------------------------
    m1 = plan_move(HOME_Q, READY_Q)      # the same for every point
    results = []
    for cy in GRID_Y:
        for cx in GRID_X:
            T = SE3.Rt(R_hover, [cx, cy, CUBE_TOP_Z + HOVER_M])
            ik = solve(panda, T, READY_Q)
            m2 = plan_move(READY_Q, ik["q"])
            results.append(dict(cx=cx, cy=cy, target=T.t.copy(), ik=ik, m1=m1, m2=m2,
                                s1=sweep(panda, m1["Q"], cx, cy),
                                s2=sweep(panda, m2["Q"], cx, cy)))

    print(f"\n3-5. Grid (seed READY, tol {IK_TOL:g}); home->ready takes {m1['T']:.1f} s, "
          f"peak {m1['v_peak'].max():.4f} rad/s on joint{int(np.argmax(m1['v_peak'])) + 1}")
    print("   cube (x, y)     IK  it s  TCP err mm  margin(j)    max move(j)  T s  "
          "floor m (link)          tcp-cube m  PASS")
    for r in results:
        ik, s1, s2 = r["ik"], r["s1"], r["s2"]
        floor = min(s1["link_floor"], s2["link_floor"])
        fname = s1["link_floor_name"] if s1["link_floor"] <= s2["link_floor"] else s2["link_floor_name"]
        ok = ik["ik_ok"] and s1["ok"] and s2["ok"]
        bad = [k for k, v in ik["gates"].items() if not v]
        bad += ["sweep1"] if not s1["ok"] else []
        bad += ["sweep2"] if not s2["ok"] else []
        print(f"   ({r['cx']:.2f},{r['cy']:+.2f})  {'ok ' if ik['success'] else 'NO '}"
              f"{ik['iterations']:>3} {ik['searches']:>1}  {ik['pos_mm']:>10.4f}  "
              f"{ik['min_margin']:.4f}(j{ik['limiting_joint']})  "
              f"{ik['max_jump']:.4f}(j{ik['max_jump_joint']})  {r['m2']['T']:>3.1f}  "
              f"{floor:.4f} ({fname:<12})  {s2['tcp_cube']:.4f}      "
              f"{'PASS' if ok else 'FAIL ' + ','.join(bad)}")

    hdr = (["cube_x", "cube_y", "target_x", "target_y", "target_z", "ik_success",
            "iterations", "searches", "solve_ms", "pred_tcp_x", "pred_tcp_y", "pred_tcp_z",
            "tcp_pos_err_mm", "orient_err_deg", "tilt_deg", "heading_err_deg",
            "min_limit_margin_rad", "limiting_joint", "max_move_from_ready_rad",
            "max_move_joint", "ik_gates_ok", "home_ready_T_s", "ready_hover_T_s",
            "ready_hover_peak_v_rad_s", "sweep1_min_link_z_m", "sweep1_tcp_min_z_m",
            "sweep2_min_link_z_m", "sweep2_tcp_min_z_m", "sweep2_tcp_cube_clear_m",
            "sweeps_ok", "PASS"] + [f"q{k}_rad" for k in range(1, 8)])
    rows = []
    for r in results:
        ik = r["ik"]
        rows.append([r["cx"], r["cy"], *r["target"], ik["success"], ik["iterations"],
                     ik["searches"], ik["ms"], *ik["tcp"], ik["pos_mm"], ik["ori_deg"],
                     ik["tilt_deg"], ik["heading_err_deg"], ik["min_margin"],
                     ik["limiting_joint"], ik["max_jump"], ik["max_jump_joint"], ik["ik_ok"],
                     r["m1"]["T"], r["m2"]["T"], float(r["m2"]["v_peak"].max()),
                     r["s1"]["link_floor"], r["s1"]["tcp_floor"], r["s2"]["link_floor"],
                     r["s2"]["tcp_floor"], r["s2"]["tcp_cube"], r["s1"]["ok"] and r["s2"]["ok"],
                     ik["ik_ok"] and r["s1"]["ok"] and r["s2"]["ok"], *ik["q"]])
    write_csv(os.path.join(run_dir, "grid_predict.csv"), hdr, rows)

    # ---- 6. mirror check (shape, not joint values) --------------------------
    print("\n6. Mirror check, y = +/-0.30 pairs (mirror the -y arm across the x-z plane)")
    mrows = []
    for cx in GRID_X:
        rp = next(r for r in results if r["cx"] == cx and r["cy"] == 0.30)
        rn = next(r for r in results if r["cx"] == cx and r["cy"] == -0.30)
        Pp = panda.fkine_all(rp["ik"]["q"]).t
        Pn = panda.fkine_all(rn["ik"]["q"]).t * np.array([1.0, -1.0, 1.0])
        shape = float(np.max(np.linalg.norm(Pp - Pn, axis=1)))
        tp, tn = rp["ik"]["tcp"], rn["ik"]["tcp"] * np.array([1.0, -1.0, 1.0])
        tcp = 1000 * float(np.linalg.norm(tp - tn))
        j13 = (rp["ik"]["q"][0] + rp["ik"]["q"][2], rn["ik"]["q"][0] + rn["ik"]["q"][2])
        print(f"   x {cx:.2f}: link origins differ by up to {shape:.4f} m, TCPs by {tcp:.4f} mm; "
              f"joint1+joint3 {j13[0]:+.4f} vs {j13[1]:+.4f}")
        mrows.append([cx, shape, tcp, j13[0], j13[1], *rp["ik"]["q"], *rn["ik"]["q"]])
    write_csv(os.path.join(run_dir, "mirror_check.csv"),
              ["cube_x", "max_link_mismatch_m", "tcp_mismatch_mm", "j1_plus_j3_pos_y",
               "j1_plus_j3_neg_y"] + [f"pos_q{k}" for k in range(1, 8)]
              + [f"neg_q{k}" for k in range(1, 8)], mrows)

    # ---- centerline setpoint stream (what the sim will be sent) -------------
    center = next(r for r in results if r["cx"] == 0.50 and r["cy"] == 0.0)
    m2c = center["m2"]
    t = np.concatenate([m1["t"], m1["T"] + m2c["t"][1:]])
    Q = np.vstack([m1["Q"], m2c["Q"][1:]])
    seg = ["home_ready"] * len(m1["t"]) + ["ready_hover"] * (len(m2c["t"]) - 1)
    write_csv(os.path.join(run_dir, "traj_centerline.csv"),
              ["t_s", "segment"] + [f"q{k}_rad" for k in range(1, 8)],
              [[float(ti), sg, *q] for ti, sg, q in zip(t, seg, Q)])

    png = make_plot(run_dir, panda, psi, results, center)
    n_pass = sum(1 for r in results if r["ik"]["ik_ok"] and r["s1"]["ok"] and r["s2"]["ok"])
    print(f"\n{n_pass}/{len(results)} grid points PASS all gates")
    print(f"wrote {run_dir}/: ready_check.csv grid_predict.csv mirror_check.csv "
          f"traj_centerline.csv")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
