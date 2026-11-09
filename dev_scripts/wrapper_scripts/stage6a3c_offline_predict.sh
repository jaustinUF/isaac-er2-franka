#!/usr/bin/env bash
# stage6a3c_offline_predict.sh -- native side, toolbox only (no sim, no ROS needed).
# Runs on the project's .venv (system Python 3.12 + --system-site-packages), so it
# ignores whatever conda env is active in the terminal.
set -euo pipefail
# REPO LAYOUT: this wrapper lives in dev_scripts/wrapper_scripts/; the script is in dev_scripts/,
# the .venv at the repo root.
cd "$(dirname "$0")/.."
exec ../.venv/bin/python stage6a3c_offline_predict.py "$@"
