# dev_scripts

The steps that led to the current scripts, kept as a record. Each one is a copy of the
one before it with a single, named change; the change is described at the top of its docstring.

| Script | What it was for | Replaced by |
|---|---|---|
| `stage6a3b_camera_node.py` | Sim node with a fixed 8 cm cube (from the 6A.3b stage, unchanged) | `stage6a3c_camera_node.py` (`--cube-size`) |
| `stage6a3c_offline_predict.py` | Offline prediction for the 8 cm cube | `stage6a3c_offline_predict_v2.py` |
| `stage6a3c_ik_reach.py` | First sim runs: IK move to the hover pose, no correction (8 cm) | `stage6a3c_ik_reach_corr.py` |
| `stage6a3c_ik_reach_corr.py` | Adds the one-shot droop correction and finer CSV logging (8 cm) | `stage6a3c_ik_reach_v2.py` (`--cube-size`) |

They still run through `wrapper_scripts/` here, using the `.venv` at the repo root.
