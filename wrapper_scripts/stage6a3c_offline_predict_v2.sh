#!/usr/bin/env bash
# stage6a3c_offline_predict_v2.sh -- native side (--cube-size, default 0.05), toolbox only (no sim, no ROS needed).
# Runs on the project's .venv (system Python 3.12 + --system-site-packages), so it
# ignores whatever conda env is active in the terminal.
set -euo pipefail
# REPO LAYOUT: this wrapper lives in wrapper_scripts/; scripts, .venv and data are at the repo root.
cd "$(dirname "$0")/.."
exec ./.venv/bin/python stage6a3c_offline_predict_v2.py "$@"
