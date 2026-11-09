# results

Data from the v0.1 runs. Each run folder holds `summary.json` (full-precision results)
and `steps.csv` (one row per sim step: commanded and actual joints, hand pose, fingertip point).
`../stage6a3c_ik_reach_analysis.ipynb` reads `runs/` and writes `analysis/`.

| Folder | What it holds | Headline |
|---|---|---|
| `offline_prediction/cube80mm/`, `cube50mm/` | IK and path predictions for the 9-point grid, before any sim run | 9/9 reachable; ±y pairs exact mirrors |
| `runs/1_oracle_8cm/` | Arm sent to the known cube position, no correction | 1.1–4.6 mm, all of it gravity droop |
| `runs/2_droop_correction_8cm/` | Same moves plus a one-shot droop correction | 4.61 → 0.055 mm worst case |
| `runs/3_oracle_5cm/` | 5 cm cube, with correction | matched prediction |
| `runs/4_er2_target_5cm/` | ER 2 picks the target; error split into perception and motion | end-to-end 1.2–2.8 mm = ER 2 pointing error |
| `er2_point_check/` | ER 2 pointing on the 5 cm cube, 3 calls at each of 5 positions, no motion | 15/15 on the top face, 1.8 mm mean |
| `analysis/` | `ik_reach_summary.csv` from the notebook | |
