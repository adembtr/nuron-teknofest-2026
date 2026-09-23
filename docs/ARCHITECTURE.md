# Architecture

Design decisions and the measurements behind them. The Turkish originals with full
experiment logs are preserved in the source tree docstrings.

---

## Contents

1. [Process topology](#1-process-topology)
2. [Task 1 — detection, motion, landing](#2-task-1--detection-motion-landing)
3. [Task 2 — GPS-denied positioning](#3-task-2--gps-denied-positioning)
4. [Task 3 — reference-object matching](#4-task-3--reference-object-matching)
5. [Sharing one 8 GB GPU](#5-sharing-one-8-gb-gpu)

---

## 1. Process topology

Two processes in two conda environments, because DPVO's CUDA extension (`cuda_corr`) will
only build against its own torch build.

```
┌─ conda: nuron ─────────────────────────┐   ┌─ conda: dpvo ──────────┐
│ connect.main                           │   │ scripts/gps_worker.py  │
│   ├─ Task 1  detector · motion · landing│   │   └─ DPVOTracker       │
│   ├─ Task 3  bank · candidates · track │   │                        │
│   └─ Task 2  bridge ──── file IPC ─────┼──►│   x y z d_med nx ny nz rms
└────────────────────────────────────────┘   └────────────────────────┘
```

The bridge is `src/client/gps_bridge.py`; the worker writes one line per frame.

### Run folders

Each run gets `kosular/folder<N>/`, and `runtime/AKTIF_KOSU` is a marker file naming the
live one so a worker started later attaches to the right run. If the marked directory has
been deleted, the marker is treated as **stale** and is not recreated — otherwise a worker
would resurrect a folder the user had just cleaned up.

### Atomic handoff

Frames cross the process boundary as files. A plain write let the reader open a partially
written JPEG: **9% of frames arrived corrupt**, producing a **322 m** 3D error. The fix was
write-to-temp then `os.replace` — an atomic rename on POSIX. Corrupt rate went to zero.

This is the single largest error reduction in the project, and no model was involved.

---

## 2. Task 1 — detection, motion, landing

### Detection

YOLO26-L, one checkpoint per modality, four classes:

```
0 = vehicle   1 = person   2 = UAP (landing zone)   3 = UAI (non-landing zone)
```

Class-based confidence thresholds, plus duplicate-box suppression at IoU > 0.8.

### Vehicle motion — rule v2

The camera is moving, so pixel motion alone says nothing. NURON estimates the **background
homography** between consecutive frames, then decomposes each vehicle's epipolar residual
into perpendicular and along-track components.

v2 changed five things:

- **Edge-safe anchor** — decide as the vehicle enters the frame; a box that merely grows
  as the vehicle comes into view is not motion.
- **Fast decision** — two strong direction-consistent residuals, or one very strong one.
- **YOLO gap protection** and **vote consistency** across frames.
- **Size gate** on matching.
- **Modality-specific measured thresholds.** The old absolute 10/4 px threshold caught
  *nothing* on thermal.
- **Fixed working resolution** — 960 px RGB, 640 px thermal.

| Session (hand-labelled GT) | v1 | v2 |
|---|---|---|
| RGB 2026 (1080p) | 0.946 · 25 FP · 5-frame latency | **0.985 · 5 FP · 1-frame latency** |
| Thermal 2026 | 0.985 · 0/24 moving found | **0.999 · 22/24** |
| Thermal 2025 s4 | 0.957 | **0.984** |
| RGB 4K 2025 s2 / s3 | 0.832 / 0.757 | **0.921 / 0.939** |

Leave-one-session-out showed no overfitting of the thresholds (≤ 1.25% difference).

### Landing suitability

Runs only on UAP/UAI boxes. Three signals: a **PaDiM** anomaly score (ResNet18 Mahalanobis
over patch embeddings, input 448×196), a **framing check**, and **obstacle overlap**.

---

## 3. Task 2 — GPS-denied positioning

```
frame → preprocess → DPVO (scale-free) → Umeyama alignment (first 450 vs GT)
      → anchor at GPS events → X, Y, Z
```

XY is the learned network. **Z uses no model at all** — ORB feature pair-distances and
ground sample distance, pure geometry.

### Calibration — the first 450 frames

Motion-gated smart-window Umeyama gives rotation, scale and translation `(R, s, t)`. ORB
scale-ratio gives `h0` at 960 px. Then a **geometric decision**: the angle between the
optical axis and the ground-plane normal, plus the plane-fit rms, classify the view.

| Geometry | Condition | Z source weights (A/B/C) |
|---|---|---|
| nadir + flat | angle < 15°, rms < 0.08 | 0.8 / 0.1 / 0.1 |
| oblique / relief | otherwise | 0.1 / 0.6 / 0.3 |

Getting this wrong is expensive: without a ground-plane normal the code assumes oblique,
which costs about 7 m of Z on a nadir session.

### XY — a 4-state EKF

State `[px, py, θ, λ]`, update `p += e^λ · R(θ) · Δ(aligned DPVO)`.

GPS events are measurements with shrinking noise — 2 m on the first, then 1 m, then 0.3 m.
A λ-drift prior of −2e-5 per frame captures DPVO's post-calibration scale drift, fitted
leave-one-out across three sessions. There is **no single-frame re-anchor**.

### Z — three sources

| Source | What | Kalman state |
|---|---|---|
| **A** | ORB scale ratio (`ObjectSizeZ`) | `[c, k]` — h0 scale |
| **B** | aligned DPVO z | `[c, a, b]` — with slope |
| **C** | −s · median patch depth (absolute) | `[c, a, b]` — with slope |

B and C carry a **slope** term because DPVO's z error is linear in horizontal displacement
(a 1–5° tilt of the gravity anchor).

**The Z-v2 fix.** End-to-end with anchors at 1200+1600 the RGB Z error came out at 3.93 m,
while the identical code with anchors at 1400+1900 gave 1.37 m. Root cause: at a GPS event
the Kalman was learning a **scale/slope coefficient from a single measurement** — a transient
6.6 m ORB error at frame 1200 was written into `k`, and when the altitude reversed it
generated 6–9 m of error. Fix: at an event the sources are **fully anchored** (`r_olay` 0.05),
the offset decays to zero with τ = 150 frames, and the coefficients are **frozen**
(`p_k = p_ab = 0`). Across 3 sessions × 8 anchor schedules: mean Z 1.56 → 1.47, worst
4.03 → 2.36, XY unchanged. Removing the config keys restores the old behaviour exactly.

### Input-size normalisation

`scale` was a fixed multiplier, so intrinsics ignored the incoming resolution. But K is in
pixels — `fx` means "pixels per radian" — so when image size changes, `fx, fy, cx, cy` must
change too.

- halve `fx` → every motion looks twice as large → the trajectory inflates
- wrong `cx` → the projection **skews**, not just scales; no single multiplier fixes it
- `undistort` breaks too — distortion coefficients are normalised, but K must agree

```
ratio  = W_TARGET / W_incoming
K_dpvo = K_calib × (W_TARGET / W_CALIB)      ← W_incoming never appears
```

Order: undistort at the **incoming** size → resize by a **single** ratio → DPVO with fixed K.

| | W_CALIB | W_TARGET |
|---|---|---|
| RGB | 1920 | **960** |
| Thermal | 640 | **640** |

`INTER_AREA` when shrinking (block averaging also suppresses thermal noise), `INTER_LINEAR`
when growing.

**No letterboxing.** DPVO is fully convolutional and crops to a multiple of 16; it does not
want a fixed canvas. Padding invents fake edges and corners and the patch extractor latches
onto them. YOLO needs padding because it expects a fixed tensor — DPVO does not. Aspect ratio
is never broken: separate ratios for `fx` and `fy` would invalidate K. The 16-crop takes from
right and bottom, so `cx, cy` stay valid.

Z is normalised too (`position.py::_zt_img`): `ObjectSizeZ` uses a fixed `f = 731.80`,
calibrated for 640 px. At 1280 the pixel velocities would double and the GSD altitude would
halve.

Verified: 320 / 640 / 1280 / 1920 / 3840 all land on target with identical K, and a 640×512
thermal frame produces **bit-identical** output to the pre-change code.

### RGB DPVO resolution — 960 over 640

Measured across 4 sessions with GPS on the first 450 frames only, the hardest scenario.
960 wins on land. **It loses badly on water** — see below.

### The calibration confidence gate

Over textureless sea the first 450 frames cannot constrain scale. One session: alignment
residual 30 m, scale 0.35, **Z error 982 m**.

After calibration, three checks:

| Check | Threshold | Catches |
|---|---|---|
| alignment residual | > 4 m | geometry that never fitted |
| scale `s` | outside [5, 400] | scale collapse / blow-up |
| calibration path | < 30 m | not enough motion to solve scale |

Fail one → **conservative mode**: XY = last GT + damped velocity (≤ 60 frames), Z = last GT
z, re-anchor at every GPS event. In all modes: per-frame clamp 3 m XY / 1.5 m Z, and Z may
not deviate more than 60 m from the last anchor. Parameters live under `koruma` in
`config/gps_rgb_final.json`.

| Session | Without gate | With gate |
|---|---|---|
| sample 1 | X 68 / Y 62 / **Z 982** | X 38 / Y 56 / **Z 6.4** |
| sample 2 | — | X 51 / Y 19 / Z 0.2 |
| 2025 s1 (first ~500 frames sea) | — | X 162 / Y 128 / Z 7.0 |

Healthy sessions sit at 0.2–0.6 m residual, so the gate never fires on them and the
good-session regression is bit-identical.

**Honest limit.** The gate stops the catastrophe; it does not rescue the trajectory. XY is
still tens of metres. DPVO's scale swings 2–6× on those sessions and re-calibrating from
anchors did not help either. With scale-free monocular VO over textureless water, better is
not available.

### Thermal specifics

The thermal recording **freezes for ~6 frames every ~61 s** (identical consecutive frames)
then jumps in a single frame, while GPS runs continuously. DPVO entered a seed-dependent
scale mode on that jump and collapsed to 150–544 m.

Fix, verified on 7 of 7 seeds: the worker does not feed repeated frames to DPVO
(`tekrar_esik` 1.5 on the preprocessed grayscale difference), calls DPVO with the **real**
frame index so the timestamp scales the motion-model gap, and sets `MOTION_DAMPING = 1.0`.

The **v4 (G″) estimator** then added: tail-window yaw refresh (calibration yaw updated over
the last 100 frames, XY only); a **sign-gated saturating scale ramp** that predicts drifting
scale from the in-calibration local-scale slope; a **vertical leakage correction** — DPVO
leaks horizontal motion into vertical (Σ|dz|/Σ|d3D| measured .36 against GT .12), undone
under a nadir gate; and early-GPS-event protection. `NURON_TERMAL_V4=0` reverts to v1
bit-for-bit (704/704 frames, difference 0.000).

---

## 4. Task 3 — reference-object matching

```
session start:  reference images → HQ-SAM crop → embedding bank
per frame:      FastSAM candidates → contact growth → embed → match → margin lock → track
```

Embedding backbone is chosen by the **reference image's** modality, not the session's:
**C-RADIOv3-H** for colour references, **WebSSL DINO-300M** for grayscale. Selecting the
line this way moved recall from 64.4% to 83.3%.

### The margin rule

A candidate is only locked if it beats the runner-up by a margin. Otherwise **no box is
emitted**.

This is why precision runs far ahead of recall:

| | recall | precision |
|---|---|---|
| RGB | 74.9% | **98.8%** |
| Thermal (hybrid) | 83.3% | 97.7% |

Wrong boxes are penalised; missing boxes merely score nothing. Staying quiet when uncertain
is the correct play, and it is a deliberate choice rather than a limitation.

Once locked, **DAM4SAM** (SAM 2.1) tracks the object — `hiera_large` on RGB,
`hiera_base_plus` on thermal, where it measured *better* than large.

### Where the loss is

Task 3 loss concentrates in individual windows rather than spreading out. On RGB, one window
(frames 2098–2248) returned 1 correct box in 49 frames and cost about **10 points of recall**
on its own. That is a tractable target, not diffuse weakness.

---

## 5. Sharing one 8 GB GPU

Thermal Task 3's hybrid path loads two embedding backbones and peaks at **~5.7 GB**. The
DPVO worker wants 1.9 GB (K=1) to 2.9 GB (K=2) on the same card.

They did not fit. First measured run: **607 of 980 frames** threw
`CUDA out of memory` — YOLO never ran at all — and 246 more hit OOM in Task 3.

Two switches fixed it **without touching any decision logic**:

| Switch | Effect | Cost |
|---|---|---|
| `NURON_PADIM_DEVICE=cpu` | landing PaDiM to CPU (`cov_inv` 196×448×448 ≈ 0.4 GB) | ~50 ms on frames with a UAP/UAI box; output identical |
| `NURON_TERMAL_K=1` | one DPVO instance in the worker | calibration residual identical to K=2 (0.25 m) |

Result: client ~4.6 GB + worker ~1.3 GB ≈ **5.9 GB**, no OOM.

A file-based GPU queue (`src/common/gpu_kuyruk.py`, `GPU_ISTEK` / `GPU_ONAY`) arbitrates
between the processes.

**Start order matters on thermal**: server → client → worker, so the client places its heavy
models first. On RGB the order is free.

---

## Frame budget

| | |
|---|---|
| Session | 2250 frames / 90 min |
| Budget | **2.4 s/frame** |
| Measured | RGB 0.83 s · thermal 0.48 s |
| Utilisation | **22–36%** |

The interface also enforces a minimum 0.25 s between frames (`MIN_FRAME_INTERVAL`), and the
server will not release frame *n+1* until frame *n* has been answered.
