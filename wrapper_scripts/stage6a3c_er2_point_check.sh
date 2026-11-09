#!/usr/bin/env bash
# Stage 6A.3c - native-side wrapper for stage6a3c_er2_point_check.py (ER 2 pointing
# recheck on the 5 cm cube; perception only, no motion).
#   ./stage6a3c_er2_point_check.sh --cube-xy 0.5,0.0 [--calls 3]   (match the node's --cube-xy)
#
# Copy of stage6a3c_ik_reach_v2.sh (project .venv, conda deactivated, native ROS
# sourced) plus the GEMINI_API_KEY fail-fast from stage6a3b_er2_reach.sh.
# google-genai lives in ~/.local, which the --system-site-packages venv sees.
# Needs deproject.py + calib_dump.json at the repo root.
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

# REPO LAYOUT: this wrapper lives in wrapper_scripts/; scripts, .venv and data are at the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"

echo "[wrapper] python -> $PY ($("$PY" --version 2>&1))"
echo "[wrapper] ROS_DISTRO=$ROS_DISTRO RMW=$RMW_IMPLEMENTATION"
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "[wrapper] ERROR: GEMINI_API_KEY is not set (export it in ~/.bashrc)." >&2
    exit 1
fi
echo "[wrapper] GEMINI_API_KEY is set"
for f in deproject.py calib_dump.json; do
    [ -f "$SCRIPT_DIR/$f" ] || { echo "[wrapper] ERROR: $f missing from the repo root" >&2; exit 1; }
done
echo "[wrapper] REMINDER: restart the sim node for each cube position (arm at home)."

cd "$SCRIPT_DIR"
exec "$PY" stage6a3c_er2_point_check.py "$@"
