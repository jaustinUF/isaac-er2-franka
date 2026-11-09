# isaac-er2-franka

Gemini Robotics ER 2 directing a Franka Panda in NVIDIA Isaac Sim 5.1, over ROS 2 Jazzy.
ER 2 looks at a camera frame and points at a target. We deproject the point to world
coordinates, solve inverse kinematics (IK) for a fingers-down hover pose above it, and
move the arm there. Each milestone is tagged; this is **v0.1-ik-reach**.

**v0.1 — IK `move_to`, predicted offline and validated in sim.** Every move is checked
before the sim runs it: IK, joint limits and a path sweep. The arm goes home → a ready pose →
the hover pose on smooth, interpolated trajectories, one waypoint per sim step. Measured
in the sim, all the remaining position error was gravity droop: the joints sag slightly
under load, by 1–5 mm at the fingertips. A one-shot correction (measure, shift the target, re-solve)
trims that to under 0.06 mm. The cube was then cut from 8 cm to 5 cm so the gripper can
hold it, ER 2's pointing was rechecked (15/15 on the top face, ~1.8 mm mean error), and
finally ER 2 chose the target itself:

| Cube (x, y) m | ER 2 target error | Arm tracking error (corrected) | End-to-end |
|---|---|---|---|
| (0.50, 0.00) | 2.84 mm | 0.005 mm | 2.84 mm |
| (0.50, +0.30) | 1.22 mm | 0.027 mm | 1.23 mm |
| (0.50, −0.30) | 1.74 mm | 0.025 mm | 1.73 mm |

With motion error trimmed out, end-to-end accuracy equals perception accuracy, about
one step of ER 2's 0–1000 pointing grid at this camera.

**Layout.** Current scripts are at the root, and their launch wrappers are in `wrapper_scripts/`.
Earlier versions, each one deliberate change from the last, are in `dev_scripts/`. Run data and the
analysis output are in `results/`, and `stage6a3c_ik_reach_analysis.ipynb` reproduces the
analysis from that data. The offline prediction and the notebook need only Python and
roboticstoolbox-python, not Isaac Sim.

**Environment.** Two processes on one ROS 2 graph. The sim side runs Isaac Sim's bundled Python 3.11
(conda `env_isaaclab`). The native side runs system Python 3.12 with ROS 2 Jazzy, in a
`--system-site-packages` venv with roboticstoolbox-python 1.4.4. `constraints.txt`
pins numpy 1.26.4 and scipy 1.11.4 (the versions ROS Jazzy is built against) and
matplotlib-inline 0.1.7 (to match the system matplotlib 3.6.3). The ER 2 scripts read
`GEMINI_API_KEY` from the environment.
