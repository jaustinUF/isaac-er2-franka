#!/usr/bin/env bash
# Stage 6A.3c - native-side wrapper for stage6a3c_ik_reach_corr.py (IK move_to + droop correction).
#   ./stage6a3c_ik_reach_corr.sh --cube-xy 0.65,0.3 [--corr-iters 2]   (match the node's --cube-xy)
# Copy of stage6a3c_ik_reach.sh; only the script name changed.
#
# Conda block and ROS sourcing copied from stage6a3b_er2_reach.sh. Change: the
# interpreter is the project .venv (system Python 3.12 + --system-site-packages,
# roboticstoolbox 1.4.4, numpy pinned to the system 1.26.4 via constraints.txt),
# called explicitly. No GEMINI_API_KEY check: oracle runs make no ER 2 call.
set -e

# --- Kill conda shadowing (the 5C.3 gotcha) --------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    while [ -n "${CONDA_PREFIX:-}" ]; do
        conda deactivate
    done
else
    export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "$CONDA_BASE" | paste -sd ':' -)"
    unset CONDA_PREFIX
fi

# --- Source native ROS 2 Jazzy (system Python 3.12) ------------------------
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp   # match the sim side's RMW

# REPO LAYOUT: this wrapper lives in dev_scripts/wrapper_scripts/; the script is in dev_scripts/,
# the .venv at the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$SCRIPT_DIR/../.venv/bin/python"

echo "[wrapper] python -> $PY ($("$PY" --version 2>&1))"
echo "[wrapper] ROS_DISTRO=$ROS_DISTRO RMW=$RMW_IMPLEMENTATION"
echo "[wrapper] REMINDER: restart the sim node before each run (arm must start at home)."

cd "$SCRIPT_DIR"
exec "$PY" stage6a3c_ik_reach_corr.py "$@"
