# Competition protocol

TEKNOFEST 2026 — Artificial Intelligence in Aviation. What the evaluation server expects
and how NURON talks to it. No credentials appear in this repository; supply your own via
`connect/config/.env`.

---

## Session format

| | |
|---|---|
| Frames | **2250** per session |
| Duration | **90 minutes** → 2.4 s/frame budget |
| Frame rate | recorded at 30 fps, delivered at **7.5 fps** (every 4th frame) |
| Flight length | 2250 frames ≈ 300 s of flight |
| Modality | announced at session start — **RGB** or **thermal** |
| Resolution | RGB **1920×1080** · thermal **640×512** |
| Image format | jpg / png / webp — any of them may arrive |
| GPS ground truth | **first 450 frames only** (`health_status = 1`) |

After frame 450 the server still sends a `translation` field, but with `health_status = 0`
and a **plausible yet wrong** value. It is not NaN — reading it by accident corrupts the
estimate silently. NURON never copies it: in both end-to-end runs, the number of frames
where the server's wrong GPS was echoed back was **0**.

The server performs **no scale or coordinate alignment**. Aligning the estimated trajectory
to the competition frame using the first 450 GT frames is the competitor's job — NURON does
it with Umeyama similarity alignment.

---

## Endpoints

Implemented in `connect/iface/connection_handler.py`.

| Endpoint | Method | Purpose |
|---|---|---|
| `auth/` | POST | username + password → token |
| `progress/` | GET | session name, total frames, frames received, `completed` |
| `session/` | GET | session metadata |
| `frames/` | GET | the **next single** frame |
| `translation/` | GET | that frame's `health_status` + `translation_x/y/z` |
| `reference/` | GET | reference object list + `references.json` + images |
| `prediction/` | POST | the prediction packet |
| `media/<path>` | GET | frame and reference image bytes |

Auth header on every call after login:

```
Authorization: Token <token>
```

### Rules that shape the design

- **Frame gating.** Frame *n+1* is only released once a prediction for frame *n* has been
  posted. The whole pipeline is therefore strictly sequential — latency is the constraint,
  not throughput.
- **One prediction per frame.** Posting twice returns **HTTP 406**.
- **Rate limit: 300 requests/minute.**
- **Resume.** `progress/` reports how many frames the server has accepted, so a crashed run
  can pick up where it stopped rather than restarting the session.

Timeouts: auth 10 s · progress 30 s · frame and translation 60 s · prediction retried.

---

## Tasks and weights

| Task | Weight | NURON's approach |
|---|---|---|
| **1** — object detection | 25% | YOLO26-L per modality; vehicle motion from background-homography epipolar residuals; landing suitability from framing, obstacle overlap and PaDiM |
| **2** — GPS-denied positioning | 40% | DPVO + Umeyama alignment + 4-state XY EKF; Z from a separate three-source geometric fusion |
| **3** — reference-object matching | 25% | HQ-SAM crops → embedding bank; FastSAM candidates; margin-based single lock; SAM 2.1 tracking |

Final score = (Task1 × 0.25 + Task3 × 0.25 + Task2 × 0.40) ÷ 0.90.

### Class and status codes

From `config/competition.yaml` (specification tables 2–4):

```
cls            : 0 vehicle · 1 person · 2 UAP · 3 UAI
landing_status : 0 unsuitable · 1 suitable · -1 not a landing area   (UAP/UAI only)
motion_status  : 0 stationary · 1 moving    · -1 not a vehicle       (vehicles only)
```

### Prediction packet

One JSON per frame. Values are sent as **strings** — the interface's own format.

```jsonc
{
  "frame": "<frame url>",
  "detected_objects": [
    {
      "cls": "<class url>",
      "landing_status": "...",
      "moving_status": "...",
      // box corners
    }
  ],
  "detected_translations": [ { "translation_x": "...", "translation_y": "...", "translation_z": "..." } ],
  "reference_predictions": [ { "reference": "<reference url>", /* box */ } ]
}
```

The first 10 frames are not scored, so `null` is acceptable there.

---

## Camera calibration

Official 1080p RGB parameters, stored at `config/calibration/calib_rgb1080.txt`:

```
fx 1389.7   fy 1387.1   cx 954.007   cy 558.896   radial 0.1378  −0.2564
```

Thermal parameters are at `config/calibration/calib_termal.txt`, calibrated for 640 px.

These are calibration-resolution values. The pipeline rescales K to the network's target
width and **never** lets the incoming frame width enter the intrinsics —
see [ARCHITECTURE.md](ARCHITECTURE.md#input-size-normalisation).

---

## Configuration

```bash
cp connect/config/example.env connect/config/.env
```

```ini
TEAM_NAME=<your_team_name>
PASSWORD=<your_password>
EVALUATION_SERVER_URL="http://<host>:<port>/"
SESSION_NAME=<session_name>
```

- `connect/config/.env` is the **only** path that is read.
- `SESSION_NAME` is effectively unused — the real session name arrives from the server in
  the `progress/` response.
- On competition day the **only** line that changes is `EVALUATION_SERVER_URL`. The
  organisers may hand out a local address such as `http://<ip>:5000/`. No code changes.
- `.env` is git-ignored. Never commit it — it holds your team password.

---

## Local testing

The end-to-end figures in the README come from a local replica of the evaluation server
that reproduces the real one's awkward parts: frames released one at a time behind
prediction gating, `health=1` for the first 450 frames, and **plausible but wrong** GPS on
every other frame — a stricter test than NaN, because wrong-but-readable values fail
silently.

`server/e2e_analiz.py` scores a run: Task 1 counts, Task 2 per-axis error **plus a leakage
check** that counts frames where the server's wrong GPS was echoed back, and Task 3
per-window recall and precision.

Competition datasets, reference imagery and ground truth belong to TEKNOFEST and are not
redistributed here.
