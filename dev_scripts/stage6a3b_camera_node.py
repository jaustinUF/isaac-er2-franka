"""
Stage 6A.3b - Camera node with a PARAMETERIZED cube position.
    Forked from stage6a3_camera_node.py (6A.3). ONE change:

    --cube-xy X,Y   places the red cube at world (X, Y, 0.04). Default 0.5,0.0
                    (the 6A.3 centerline position, so a bare run reproduces 6A.3).
    Use one of the nine calibrated positions (x in {0.35,0.50,0.65},
    y in {-0.30,0,+0.30}) -- ER 2's pointing there was measured in tier 3.
    The reach script takes the SAME --cube-xy for its log-only oracle; its
    oracle pixel check flags a mismatch (see stage6a3b_er2_reach.py).

    Args are parsed with parse_known_args BEFORE SimulationApp is created, the
    pattern Isaac's standalone examples use, so any Kit args pass through.

    Everything below is the 6A.3 node's docstring, unchanged.

---------------------------------------------------------------------------
Stage 6A.3 - Camera node (whole-arm framing), forked from stage5e3_camera_node_cube.py.

6A.3 DELTA -- two changes vs. the 5E.3 node, nothing else:
    1. CAM_POSITION = (21.35, 0.0, 14.95), the whole-arm framing. This is the
       camera calib_dump.json (run calib_20260925_161456) describes. (Jim made
       this edit in his project copy; carried here.)
    2. HOLD TARGETS DURING RTX WARM-UP. The 5E.3 node wrote joint targets only in
       its main loop, so for the 60 warm-up steps the Franka had NO targets and
       folded over, then snapped back upright when the loop started (Jim's
       screencast, arm_loading.webm). Same defect class as calibration run 1's
       collapsed arm ("an articulation with no targets is not holding still").
       Fix: write the default pose as position targets before every warm-up
       world.step(), the stage6a_calib_dump.py hold_step() pattern. This also
       makes the node's startup IDENTICAL to the calibration scene (the dump
       held from step 1), removing the one known difference between them.

    Everything below this block is the 5E.3 node's original docstring.

---------------------------------------------------------------------------
Stage 5E.3 - Camera node + CUBE : the proven 5E.2 camera node, additively grown
    with a red perception target.

WHAT THIS IS
    A COPY of the proven Stage 5E.2 camera node (stage5e2_camera_node.py), with
    ONE additive change: a red 8 cm VisualCuboid on the ground ~0.5 m in front of
    the arm base -- the object OWL-ViT will be prompted to find ("red cube"), and
    the same cube the earlier RL arm could never see (the deployable-perception
    payoff). Everything else -- the JointState plant path, the camera, the RTX
    warm-up, the publish loop -- is carried VERBATIM from the proven 5E.2 node.

    Why VisualCuboid (grep-verified against installed 5.1 source):
      - VisualCuboid (isaacsim.core.api.objects.cuboid, line 28) is the PURE
        VISUAL base class -- no collision, no rigid body, no physics material.
        It cannot fall, cannot be knocked, needs no PhysicsMaterial. That is
        exactly right for a see-only perception target, and it sidesteps the 5B
        "a free object falls before it is caught" trap entirely.
      - FixedCuboid (line 157) is also static but drags in a PhysicsMaterial we
        have no use for; DynamicCuboid (line 258) adds mass and WOULD fall.
        VisualCuboid is the simplest reasonable change.
      - Constructor takes `size` (SCALAR edge length, float) and
        `color` (np.ndarray RGB in 0-1). So an 8 cm red cube is
        size=0.08, color=np.array([1.0, 0.0, 0.0]).

    Placement: position=(0.5, 0.0, 0.04). +X is "in front" of the Franka; ground
    is z=0; an 8 cm cube's CENTER sits at half-edge = 0.04, so it rests ON the
    ground rather than half-buried or floating.

WHAT THIS IS *NOT* YET
    Framing is still the deferred open item -- but now it has a CONCRETE target.
    The camera pose is unchanged from the proven 5E.2 node (the base-tight probe
    aim). Once this runs, EYEBALL the saved frame: the pass/fail test is "is the
    red cube clearly in frame and reasonably sized," NOT aesthetics. If the cube
    is out of frame or tiny, re-aim (pitch/yaw) or reposition against THIS target
    -- one reframe, judged against the requirement. No detector yet.

    A COPY of the proven Stage 5C.3 bidirectional plant node
    (stage5c3_bidirectional_node.py), with ONE additive change: the sim node now
    also carries a fixed world-frame camera and publishes its frames as raw
    sensor_msgs/Image on /camera/image_raw, once per step.

    Everything that made 5C.3 the proven plant is UNCHANGED and carried verbatim:
      - OUTPUT port: sensor_msgs/JointState on /joint_states every step (9 DOFs).
      - INPUT port : sensor_msgs/JointState on /joint_command, mapped by name.
      - passive-until-commanded; loop order; RELIABLE QoS; wall-clock stamp.
    The camera is PURELY ADDITIVE -- a new prim, a new publisher, and one extra
    read+publish at the tail of the existing loop. The JointState plant path is
    not touched, so 5C.3's proven behavior is preserved.

    This is 5E step 1: prove a RAW FRAME crosses the proven 5C seam INTACT.
    NEW PAYLOAD (Image), IDENTICAL ARCHITECTURE (same two-process / one-DDS-graph
    seam, same sim-side bundled-rclpy wrapper). No detector anywhere near it yet.

WHAT WE CONFIRMED AGAINST INSTALLED 5.1 SOURCE BEFORE WRITING THIS
    - Read path: Camera.get_rgb() reads self._custom_annotators["rgb"].get_data()
      -- the SAME 'rgb' annotator characterized in 5B -- and slices alpha for us
      (returns (H,W,3)). So 5B's characterization still describes this array
      (~229 Hz execution-bound cadence, noiseless flat regions, edge AA jitter,
      3-frame / ~13-17 ms latency). get_rgb() returns None before the renderer
      has produced a frame -> we GUARD and skip publishing on None.
    - Clipping: Camera.set_clipping_range(near, far) is a real method (camera.py).
      A CREATED camera gets the USD schema near-plane default of 1.0 m, which
      clips a close robot -> uniform grey frames (the 5B gotcha). We override to
      0.01 m up front so the robot is never clipped.
    - Constructor: Camera(prim_path, position=(3,), orientation=(4,) SCALAR-FIRST
      (w,x,y,z), resolution=(w,h), frequency/dt optional). No look-at param, so
      we build the orientation quaternion from an eye+target look-at helper.
    - Payload: sensor_msgs/Image imports and constructs on the sim side through
      the Option-1 wrapper (bundle rclpy), same as JointState in 5C.3.

CAMERA POSE (first guess -- eyeball the saved frame and adjust)
    A fixed "watch the robot work" vantage: out in front of the arm (+X), raised,
    angled down at the workspace where the arm and the eventual red cube sit.
    Position and look-at TARGET are the knobs; move the target to re-aim.

LOOP ORDER (each step) -- 1-6 UNCHANGED from 5C.3; 7 is the additive camera:
    1. spin_once            -- service ROS callbacks (may refresh latest command)
    2. apply latest command -- overwrite addressed joints' targets (by name)
    3. WRITE targets        -- set_joint_position_targets
    4. world.step           -- advance physics one tick
    5. READ all 9 DOFs      -- positions + velocities
    6. PUBLISH JointState   -- current state out on /joint_states
    7. PUBLISH Image        -- current camera frame out on /camera/image_raw (NEW)

RUN
    ./stage5e2_camera_node.sh                    # sim side (bundled rclpy)

    # Terminal 2 (native `rosjazzy`): confirm the image topic is publishing --
    ros2 topic hz /camera/image_raw
    ros2 topic info /camera/image_raw --verbose   # check QoS if no data
    # (The external stage5e2_image_subscriber pulls a frame and saves it.)
"""

# ----------------------------------------------------------------------------
# 1. LAUNCH THE APP FIRST. headless=False to watch the arm respond.
# ----------------------------------------------------------------------------
# 6A.3b: parse --cube-xy BEFORE the app launches (parse_known_args leaves any
# other args alone).
import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--cube-xy", dest="cube_xy", default="0.5,0.0",
                 help="cube center world X,Y in metres (default 0.5,0.0)")
_ARGS, _UNKNOWN = _ap.parse_known_args()
_CUBE_X, _CUBE_Y = (float(v) for v in _ARGS.cube_xy.split(","))
# Hand Kit only the args that aren't ours, so it never sees --cube-xy.
import sys
sys.argv = [sys.argv[0]] + _UNKNOWN

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ----------------------------------------------------------------------------
# 1a. ENABLE THE ROS 2 BRIDGE, then one app.update() -- probe-confirmed ordering.
# ----------------------------------------------------------------------------
from isaacsim.core.utils.extensions import enable_extension

_ok = enable_extension("isaacsim.ros2.bridge")
print(f"[info] enable_extension('isaacsim.ros2.bridge') -> {_ok}")
simulation_app.update()

# ----------------------------------------------------------------------------
# 1b. POINT ISAAC SIM AT THE LOCAL ASSET PACK.
# ----------------------------------------------------------------------------
import os
import carb.settings

LOCAL_ASSET_ROOT = os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1")

_settings = carb.settings.get_settings()
_settings.set("/persistent/isaac/asset_root/default", LOCAL_ASSET_ROOT)
print(f"[info] asset_root set to local: {LOCAL_ASSET_ROOT}")

# ----------------------------------------------------------------------------
# 2. NOW import sim/scene APIs AND rclpy + the message types (bundle rclpy).
# ----------------------------------------------------------------------------
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim  # RigidPrim NEW (5E.3 loop)
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.objects.cuboid import VisualCuboid  # NEW (5E.3) -- red cube
from isaacsim.sensors.camera import Camera          # NEW (5E.2)
import isaacsim.core.utils.numpy.rotations as rot_utils  # euler_angles_to_quats

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image                   # NEW (5E.2)
from geometry_msgs.msg import PoseStamped           # NEW (5E.3) -- gripper pose

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 60.0

STATE_TOPIC   = "/joint_states"        # OUTPUT: sim state -> world
COMMAND_TOPIC = "/joint_command"       # INPUT:  world -> sim targets
IMAGE_TOPIC   = "/camera/image_raw"    # OUTPUT: camera frames -> world (NEW)
GRIPPER_TOPIC = "/gripper_pose"        # OUTPUT: measured hand world pose (NEW 5E.3)
NODE_NAME     = "isaac_sim_plant"

FRANKA_PRIM_PATH = "/World/Franka"
HAND_PRIM_PATH   = "/World/Franka/panda_hand"   # end-effector link (5E.3 probe)
FRANKA_USD = os.path.join(
    LOCAL_ASSET_ROOT, "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)

# --- Red cube (NEW 5E.3) ---------------------------------------------------
# The perception target. size = SCALAR edge (m); color = RGB in 0-1.
# Center at z = half-edge so the cube rests ON the ground (z=0), +X in front.
CUBE_PRIM_PATH = "/World/red_cube"
CUBE_SIZE      = 0.08                              # 8 cm edge
CUBE_POSITION  = (_CUBE_X, _CUBE_Y, CUBE_SIZE / 2.0)   # 6A.3b: from --cube-xy
CUBE_COLOR     = (1.0, 0.0, 0.0)                   # red -- matches "red cube" prompt

# --- Camera (NEW 5E.2) -----------------------------------------------------
CAMERA_PRIM_PATH = "/World/workspace_cam"
CAM_RESOLUTION   = (1280, 720)         # (width, height) -- matches 5B
# AIMING VIA THE PROBE'S PROVEN PATH (5E.3 diagnostic). The aim-sweep probe
# rendered the robot cleanly using euler_angles_to_quats(0, 35, 180) applied via
# camera.set_world_pose() AFTER initialize(). The node had instead passed a
# quaternion to the Camera CONSTRUCTOR -- and went black with every quaternion we
# tried. Hypothesis: the constructor's orientation arg does not "take" (stored
# but not applied), so the camera stayed at its default aim regardless of the
# quaternion. This block mirrors the probe EXACTLY to test that: same euler
# angles, same set_world_pose() call, same fixed position the probe used.
#   If the subscriber now shows the arm base -> constructor-vs-set_world_pose was
#   the bug (a real defect + reusable lesson). If still black -> the difference
#   is elsewhere (ROS loop / scene build), and we have a narrow diagnostic.
# Exact framing is deferred; this is a "does the node aim AT ALL" test, using the
# probe's known-good pose (base-of-arm, close, pitch+35/yaw180).
CAM_POSITION = (21.35, 0.0, 14.95)        # far pull-back; whole arm + cube in frame
# CAM_POSITION = (10.3, 0.0, 7.0)          # close framing that gave the arm-wins result
CAM_QUAT_WXYZ    = (0.62576, 0.32927, 0.32927, 0.62576)  # from workspace_cam panel (W,X,Y,Z)
CAM_FOCAL_LENGTH = 181.5                  # telephoto, matches the framed shot
# CAM_EULER_DEG no longer used -- we set the panel quaternion directly, avoiding
# the euler-order (XYZ-vs-YXZ) convention gap that bit us in 5E.2.

CAM_NEAR, CAM_FAR = 0.01, 1.0e5        # override schema 1.0 m near (5B gotcha)


# NOTE (5E.2 aiming fix): the first attempt used a hand-rolled eye->target
# look-at matrix -> quaternion. It produced a quaternion that aimed the camera
# into empty space (frame came back BLACK -- camera looking at nothing, distinct
# from GREY which would be clipping). Replaced with Isaac's SHIPPED converter,
# rot_utils.euler_angles_to_quats(...), which the installed camera TESTS use to
# orient cameras -- the gold-standard working pattern (grep-verified in
# .../isaacsim.sensors.camera/.../tests/test_camera_sensor.py). We orient by
# explicit, reason-about-able Euler angles (roll, pitch, yaw about x, y, z) set
# in CONFIG below, and confirm the aim by eyeballing the saved frame.


class SimPlantNode(Node):
    """The sim-side plant node: BOTH JointState ports on one node, PLUS a camera
    image publisher (5E.2).
      - publisher   on /joint_states     (output)  -- 5C.3, unchanged
      - subscription on /joint_command    (input)  -- 5C.3, unchanged
      - publisher   on /camera/image_raw  (output) -- NEW (5E.2)
    The sim loop publishes state each step, reads latest_cmd each step, and (new)
    publishes the current camera frame each step. Single-threaded via spin_once,
    so latest_cmd needs no lock."""

    def __init__(self):
        super().__init__(NODE_NAME)
        # OUTPUT port -- JointState (5C.3, unchanged).
        self.state_pub = self.create_publisher(JointState, STATE_TOPIC, 10)
        # INPUT port -- JointState (5C.3, unchanged).
        self.latest_cmd = None          # {joint_name: target_pos} or None
        self._n_received = 0
        self.create_subscription(JointState, COMMAND_TOPIC, self._on_command, 10)
        # OUTPUT port -- Image (NEW 5E.2). Depth 1: only the freshest frame
        # matters for step-1 first light; a raw 720p frame is ~2.7 MB, so we do
        # not want a deep queue buffering stale frames.
        self.image_pub = self.create_publisher(Image, IMAGE_TOPIC, 1)
        self._n_frames = 0
        # OUTPUT port -- gripper (panda_hand) world pose (NEW 5E.3). Measured FK
        # from the sim, so the external controller can compute gripper-to-cube
        # distance without needing Isaac. Depth 1: freshest pose only.
        self.gripper_pub = self.create_publisher(PoseStamped, GRIPPER_TOPIC, 1)
        self._n_poses = 0
        self.get_logger().info(
            f"plant node up: publishing {STATE_TOPIC} + {IMAGE_TOPIC} + "
            f"{GRIPPER_TOPIC}, subscribing {COMMAND_TOPIC}"
        )

    def _on_command(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            self.get_logger().warn(
                f"ignoring command: {len(msg.name)} names vs "
                f"{len(msg.position)} positions"
            )
            return
        self.latest_cmd = dict(zip(msg.name, msg.position))
        self._n_received += 1
        if self._n_received == 1:
            self.get_logger().info(f"FIRST command received: {self.latest_cmd}")

    def publish_state(self, dof_names, pos, vel):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()   # WALL-CLOCK stamp
        msg.name = dof_names
        msg.position = [float(x) for x in pos]
        msg.velocity = [float(x) for x in vel]
        # effort left empty (position control); add via get_measured_joint_efforts
        # -- NOT get_applied_ (reads 0) -- if a later stage wants it.
        self.state_pub.publish(msg)

    def publish_image(self, rgb):
        """Publish an (H,W,3) uint8 RGB array as raw sensor_msgs/Image (rgb8).
        Caller MUST pass a non-None, correctly shaped array -- the None-guard
        lives in the loop so we never construct a message from an empty frame."""
        h, w = rgb.shape[0], rgb.shape[1]
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()   # WALL-CLOCK stamp
        msg.header.frame_id = "workspace_cam"
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(w * 3)                                # bytes per row
        # Contiguous uint8 buffer, exactly w*h*3 bytes, row-major -- bit-for-bit
        # what OWL-ViT will later see (raw, no encode/decode).
        msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        self.image_pub.publish(msg)
        self._n_frames += 1
        if self._n_frames == 1:
            self.get_logger().info(
                f"FIRST frame published: {w}x{h} rgb8, step={msg.step}, "
                f"{len(msg.data)} bytes"
            )

    def publish_gripper_pose(self, pos, quat_wxyz):
        """Publish panda_hand world pose (measured FK) as PoseStamped (NEW 5E.3).
        pos: (3,) xyz metres. quat_wxyz: (4,) scalar-first (w,x,y,z) from Isaac.
        PoseStamped.orientation is (x,y,z,w) order, so we re-order on the way out."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        # Isaac quats are (w,x,y,z); ROS geometry_msgs wants (x,y,z,w).
        msg.pose.orientation.w = float(quat_wxyz[0])
        msg.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.orientation.z = float(quat_wxyz[3])
        self.gripper_pub.publish(msg)
        self._n_poses += 1
        if self._n_poses == 1:
            self.get_logger().info(
                f"FIRST gripper pose published: xyz "
                f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
            )

def main():
    # ------------------------------------------------------------------
    # FAST-FAIL GUARD.
    # ------------------------------------------------------------------
    if not os.path.isfile(FRANKA_USD):
        raise FileNotFoundError(
            f"Franka USD not found at:\n  {FRANKA_USD}\n"
            f"Check LOCAL_ASSET_ROOT and the path under it."
        )
    print(f"[info] Franka USD found: {FRANKA_USD}")

    # ------------------------------------------------------------------
    # 3. BUILD THE WORLD. (5C.3, unchanged.)
    # ------------------------------------------------------------------
    world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=FRANKA_USD, prim_path=FRANKA_PRIM_PATH)
    franka = Articulation(prim_paths_expr=FRANKA_PRIM_PATH, name="franka")
    world.scene.add(franka)

    # --- Red cube (NEW 5E.3). PURELY ADDITIVE -- a pure-visual prim, added
    #     before reset() alongside the Franka. No collision / no rigid body, so
    #     it neither perturbs physics nor the articulation handle init below.
    #     This is the object OWL-ViT will look for.
    red_cube = VisualCuboid(
        prim_path=CUBE_PRIM_PATH,
        name="red_cube",
        position=np.array(CUBE_POSITION, dtype=np.float32),
        size=CUBE_SIZE,
        color=np.array(CUBE_COLOR, dtype=np.float32),
    )
    world.scene.add(red_cube)
    print(f"[info] red cube at {CUBE_POSITION} size {CUBE_SIZE} m "
          f"color(rgb) {CUBE_COLOR}")

    # --- Gripper pose handle (NEW 5E.3). RigidPrim view on panda_hand -- the
    #     grep-verified per-link pose handle (Articulation.get_world_poses gives
    #     the BASE, not the hand). Read via get_world_poses() -> (1,3),(1,4).
    #     Scene-added so its physics handle is valid after reset().
    hand = RigidPrim(prim_paths_expr=HAND_PRIM_PATH, name="panda_hand_view")
    world.scene.add(hand)

    world.reset()

    # ------------------------------------------------------------------
    # 3a. CAMERA (NEW 5E.2). Create fixed world-frame camera, aim via look-at,
    #     override the near-plane (5B grey-frame gotcha), then initialize so its
    #     'rgb' annotator is attached (that is what get_rgb() reads).
    # ------------------------------------------------------------------
    # MIRROR THE PROBE'S PROVEN AIMING PATH. Construct the camera (orientation
    # here is a placeholder -- identity), initialize (attaches rgb annotator),
    # set clipping/focal, THEN aim via set_world_pose() -- the exact call the
    # aim-sweep probe used to render the robot. This is the one change under test
    # (constructor orientation -> set_world_pose orientation).
    cam_quat = np.array(CAM_QUAT_WXYZ, dtype=np.float32)

    camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        position=np.array(CAM_POSITION, dtype=np.float32),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # placeholder
        resolution=CAM_RESOLUTION,
    )
    camera.initialize()                          # attaches the 'rgb' annotator
    camera.set_clipping_range(CAM_NEAR, CAM_FAR) # 0.01 m near -> no robot clip
    # FOCAL LENGTH: write the USD prim attribute DIRECTLY, in PANEL units (what
    # the Property panel's "Focal Length" field shows). This deliberately does
    # NOT use camera.set_focal_length() -- that setter multiplies by 10
    # (USD_CAMERA_TENTHS_TO_STAGE_UNIT, grep-verified) AND had an ordering/reset
    # interaction that left the prim at its 50.0 default. Writing focalLength
    # directly is convention-proof: it sets exactly the attribute the panel reads,
    # in the panel's own units, so CAM_FOCAL_LENGTH=181.5 -> panel shows 181.5.
    camera.prim.GetAttribute("focalLength").Set(float(CAM_FOCAL_LENGTH))

    # The Orient quaternion was harvested from the GUI panel = USD-prim space
    # (+Y up, -Z forward). set_world_pose defaults to camera_axes="world"
    # (+Z up, +X forward), which would RE-INTERPRET the quaternion in the wrong
    # frame (the component-shuffle we saw). Pass camera_axes="usd" so Isaac
    # applies the panel quaternion in the frame it actually came from.
    # (grep-verified: "usd" is an accepted camera_axes value, camera.py ~line 556.)
    camera.set_world_pose(
        position=np.array(CAM_POSITION, dtype=np.float32),
        orientation=cam_quat.astype(np.float32),
        camera_axes="usd",
    )

    print(f"[info] camera at {CAM_POSITION} "
          f"quat(wxyz) {cam_quat.tolist()} focal {CAM_FOCAL_LENGTH} "
          f"res {CAM_RESOLUTION} near/far {CAM_NEAR}/{CAM_FAR}")
    print("[info] orientation applied via set_world_pose() (probe-mirrored path)")

    # ------------------------------------------------------------------
    # 3b. ROS 2 NODE (both JointState ports + image publisher).
    # ------------------------------------------------------------------
    rclpy.init()
    node = SimPlantNode()

    world.reset()  # known starting pose

    # ------------------------------------------------------------------
    # DOF bookkeeping: ordered names for publishing, name->index for commands.
    # (5C.3, unchanged.)
    # ------------------------------------------------------------------
    dof_names = list(franka.dof_names)
    name_to_idx = {n: i for i, n in enumerate(dof_names)}
    print(f"[info] {len(dof_names)} DOFs: {dof_names}")

    # Start from (and hold) the default pose until commanded.
    default_pos = np.array(franka.get_joint_positions(), dtype=np.float32).reshape(-1)
    target_pos = default_pos.copy()

    unknown_warned = set()
    frame_warned = False   # so the "no frame yet" note prints at most once

    # ------------------------------------------------------------------
    # RENDER WARM-UP (5E.3 fix). RTX lighting (the ground plane's SphereLight)
    # takes several frames to CONVERGE -- the ray-traced illumination accumulates.
    # get_rgb() renders through the camera's RTX render product, which sees ONLY
    # real stage lights (NOT the viewport's implicit headlight -- that headlight
    # is why the GUI view looks lit while early published frames are dark). If we
    # publish on the first loop iterations, the scene is still dark (only the
    # arm's EMISSIVE base strips show, since they need no illumination). The
    # aim-sweep probe avoided this by stepping ~8 frames before grabbing; that
    # settle was dropped when the camera merged into this node. Restore it here:
    # step the renderer until lighting converges, THEN start publishing.
    # 6A.3: HOLD the default pose during warm-up (see docstring, change 2).
    # Without this the arm has no position targets for these 60 steps and
    # visibly folds over before the main loop's first target write recovers it.
    print("[info] warming up RTX renderer (letting lighting converge), "
          "holding default pose...")
    for _ in range(60):
        franka.set_joint_position_targets(target_pos.reshape(1, -1))
        world.step(render=True)
    print("[info] renderer warm; starting publish loop.")

    print("[info] plant passive -- holding default pose until commanded.")
    print("[info] close the window (or Ctrl+C) to stop.")

    # ------------------------------------------------------------------
    # 4. THE BIDIRECTIONAL LOOP. Steps 1-6 UNCHANGED from 5C.3; step 7 (camera
    #    publish) is the additive 5E.2 change.
    # ------------------------------------------------------------------
    while simulation_app.is_running():
        # 1. SERVICE ROS callbacks (may refresh node.latest_cmd).
        rclpy.spin_once(node, timeout_sec=0.0)

        # 2. APPLY latest command (by name) onto the target vector.
        if node.latest_cmd is not None:
            for jname, jpos in node.latest_cmd.items():
                idx = name_to_idx.get(jname)
                if idx is None:
                    if jname not in unknown_warned:
                        print(f"[warn] command names unknown joint '{jname}' -- ignoring")
                        unknown_warned.add(jname)
                    continue
                target_pos[idx] = float(jpos)

        # 3. WRITE targets.
        franka.set_joint_position_targets(target_pos.reshape(1, -1))

        # 4. STEP the sim one physics tick.
        world.step(render=True)

        # 5. READ all 9 DOFs AFTER the step (state reflects the tick just taken).
        pos = np.asarray(franka.get_joint_positions()).reshape(-1)
        vel = np.asarray(franka.get_joint_velocities()).reshape(-1)

        # 6. PUBLISH JointState out. (5C.3, unchanged.)
        node.publish_state(dof_names, pos, vel)

        # 7. PUBLISH the current camera frame out (NEW 5E.2). get_rgb() returns
        #    None until the renderer has produced a frame -> GUARD and skip.
        rgb = camera.get_rgb()
        if rgb is None:
            if not frame_warned:
                print("[info] camera warming up -- no frame yet, skipping publish.")
                frame_warned = True
        else:
            node.publish_image(rgb)

        # 8. PUBLISH gripper (panda_hand) world pose out (NEW 5E.3). Measured FK
        #    so the external controller computes gripper-to-cube distance without
        #    Isaac. RigidPrim.get_world_poses() -> (1,3),(1,4); take row 0.
        hpos, hquat = hand.get_world_poses()
        hpos = np.asarray(hpos, dtype=np.float32).reshape(-1)[:3]
        hquat = np.asarray(hquat, dtype=np.float32).reshape(-1)[:4]
        node.publish_gripper_pose(hpos, hquat)

    # ------------------------------------------------------------------
    # 5. CLEAN SHUTDOWN.
    # ------------------------------------------------------------------
    print("[info] shutting down ROS 2 node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
