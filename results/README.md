# Results

## `session4/` — best on-site session (2026-09-20, RGB)

| File | What |
|---|---|
| `nuron_session4_rgb_2026-09-20.mp4` | live pipeline output, 2250 frames (re-encoded to 1280×720 for size) |
| `trajectory_xy.png` | top-down XY track, GPS leg vs. dead-reckoned estimate |
| `trajectory_3d.png` | 3D track with the 13 reference windows marked |

2250 frames · **2784 m flown** · GPS for the first 450 frames, then **1800 frames with
none** · X −22…884 m, Y −1019…0 m, Z 0…80 m · start (0,0) → end (822,−877), 1202.1 m apart.

In the plots, **green** is the GPS-anchored calibration leg and **blue** is NURON's own
estimate with no GPS at all.

## `other_sessions/`

3D trajectory plots for the other three on-site sessions — sessions 1 and 2 RGB, session 3
thermal. Their videos are not included to keep the repository small.

## Reproducing the numbers

The end-to-end figures in the root README come from `server/e2e_analiz.py` run against a
local replica of the evaluation server. Competition datasets are TEKNOFEST's and are not
redistributed, so these plots and the video are the shareable record of those runs.
