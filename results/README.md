# Results

## Session recordings

Every on-site session is published **complete and unedited** — 2250 frames, 1920×1080,
original encoding, no trimming and no re-encoding — as assets on the
**[v1.0-results release](https://github.com/adembtr/nuron-teknofest-2026/releases/tag/v1.0-results)**.

| Asset | Session | Modality | Date | Size |
|---|---|---|---|---:|
| `nuron_session4_rgb.mp4` | 4 — best session | RGB | 2026-09-20 | 130 MB |
| `nuron_session1_rgb.mp4` | 1 | RGB | 2026-09-18 | 136 MB |
| `nuron_session2_rgb.mp4` | 2 | RGB | 2026-09-19 | 123 MB |
| `nuron_session3_thermal.mp4` | 3 | Thermal | 2026-09-19 | 60 MB |
| `nuron_qualification_replay.mp4` | Online qualification replay | RGB | — | 262 MB |

They live on the release rather than in the tree so that cloning this repository stays
cheap — release assets do not count toward repository size.

## `session4/` — best on-site session (2026-09-20, RGB)

| File | What |
|---|---|
| `trajectory_xy.png` | top-down XY track, GPS leg vs. dead-reckoned estimate |
| `trajectory_3d.png` | 3D track with the 13 reference windows marked |

2250 frames · **2784 m flown** · GPS for the first 450 frames, then **1800 frames with
none** · X −22…884 m, Y −1019…0 m, Z 0…80 m · start (0,0) → end (822,−877), 1202.1 m apart.

In the plots, **green** is the GPS-anchored calibration leg and **blue** is NURON's own
estimate with no GPS at all.

## `other_sessions/`

3D trajectory plots for the other three on-site sessions — sessions 1 and 2 RGB, session 3
thermal.

## Reproducing the numbers

The end-to-end figures in the root README come from `server/e2e_analiz.py` run against a
local replica of the evaluation server. Competition datasets are TEKNOFEST's and are not
redistributed, so these recordings and plots are the shareable record of those runs.
