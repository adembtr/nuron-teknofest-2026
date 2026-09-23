# Model weights

No weights live in this repository. `models/` is git-ignored; this file says what goes
into it and where each file comes from.

Naming: **`rgb_*`** = RGB session · **`termal_*`** = thermal session · **no prefix** = shared.
Paths are resolved in `src/task2_position/paths.py` (Task 2) and
`src/task3_reference/*/ayarlar.py` (Task 3).

---

## 1. Trained by Team NURON

**[`adembtr/nuron-teknofest-2026`](https://huggingface.co/adembtr/nuron-teknofest-2026)** on HuggingFace.

```bash
pip install huggingface_hub
hf download adembtr/nuron-teknofest-2026 --local-dir models/
```

| File | Task | Architecture | Size |
|---|---|---|---|
| `rgb_yolo.pt` | 1 | YOLO26-L, 4 classes | 151 MB |
| `termal_yolo.pt` | 1 | YOLO26-L, 4 classes, thermal | 51 MB |
| `rgb_padim_uap/{rgb_padim_uap,rgb_padim_uap2}.pth` | 1 | PaDiM ResNet18, 2-model ensemble | 2 × 151 MB |
| `rgb_padim_uai/{rgb_padim_uai,rgb_padim_uai2}.pth` | 1 | PaDiM ResNet18, 2-model ensemble | 2 × 151 MB |
| `termal_padim_uap.pth` | 1 | PaDiM ResNet18 | 151 MB |
| `termal_padim_uai.pth` | 1 | PaDiM ResNet18 | 151 MB |

**PaDiM loading rule** (`src/task1_detection/landing.py`): if
`models/{modality}_padim_{uap,uai}/` exists as a **folder**, every `.pth` inside it is
loaded in name order and ensembled. If not, the flat file
`models/{modality}_padim_{uap,uai}.pth` is used. RGB ships as folders, thermal as flat files.

PaDiM fits on CPU when VRAM is tight — `NURON_PADIM_DEVICE=cpu`. It only runs when a
UAP/UAI box exists, so it costs ~50 ms on those frames, and **the output is identical**.

---

## 2. Public checkpoints — fetch from the original authors

Unmodified third-party weights. NURON trained none of these.

| File | Source | Size |
|---|---|---|
| `rgb_dpvo.pth`, `termal_dpvo.pth` | [DPVO](https://github.com/princeton-vl/DPVO) | 14 MB each |
| `sam_hq_vit_l.pth` | [HQ-SAM](https://github.com/SysCV/sam-hq) | 1.25 GB |
| `rgb_fastsam_x.pt` | [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) | 139 MB |
| `rgb_radio_v3h.pth.tar` | [NVIDIA C-RADIOv3-H](https://github.com/NVlabs/RADIO) | 1.85 GB |
| `radio_repo/` | [NVIDIA RADIO](https://github.com/NVlabs/RADIO) repo clone | 26 MB |
| `webssl_dino300m_light2b/` | [Meta WebSSL DINO-300M](https://huggingface.co/facebook/webssl-dino300m-full2b-224) | 1.2 GB |
| `sam2.1_hiera_large.pt` | [SAM 2.1](https://github.com/facebookresearch/sam2) | 857 MB |
| `sam2.1_hiera_base_plus.pt` | [SAM 2.1](https://github.com/facebookresearch/sam2) | 309 MB |
| `dpvo_default.yaml` | DPVO config (shared; thermal tuning lives in `paths.py`) | 377 B |

**The two DPVO files are byte-identical** (md5 `a0f9fe5b…`). There is no thermal-specific
training — the difference between the two paths is preprocessing and config, nothing more.

`radio_repo/` must be present as a **local hub directory**: the competition runs with no
internet, so C-RADIOv3-H cannot be fetched at load time.

### SAM 2.1 choice

RGB uses `hiera_large`, thermal uses `hiera_base_plus`. That is not a compromise for
memory — on thermal, **base_plus measured better than large**. In the hybrid thermal path
both lines share the base_plus instance.

---

## 3. Expected layout

```
models/
├── rgb_yolo.pt
├── termal_yolo.pt
├── rgb_padim_uap/{rgb_padim_uap.pth, rgb_padim_uap2.pth}
├── rgb_padim_uai/{rgb_padim_uai.pth, rgb_padim_uai2.pth}
├── termal_padim_uap.pth
├── termal_padim_uai.pth
├── rgb_dpvo.pth
├── termal_dpvo.pth
├── dpvo_default.yaml
├── sam_hq_vit_l.pth
├── rgb_fastsam_x.pt
├── rgb_radio_v3h.pth.tar
├── radio_repo/
├── webssl_dino300m_light2b/{config.json, model.safetensors}
├── sam2.1_hiera_large.pt
└── sam2.1_hiera_base_plus.pt
```

Weights sit **directly** under `models/` — no nested `models/models/`.

Camera calibration is **not** in `models/`; it is version-controlled at
`config/calibration/calib_rgb1080.txt` (for 1920 px) and `calib_termal.txt` (for 640 px).

---

## 4. Third-party source repositories

`third_party/` is git-ignored. Clone what you need:

```bash
mkdir -p third_party
git clone https://github.com/princeton-vl/DPVO           third_party/dpvo_repo
git clone https://github.com/jovanavidenovic/DAM4SAM     third_party/DAM4SAM
```

### The DPVO patch

NURON changes **one line** in `third_party/dpvo_repo/dpvo/dpvo.py` (around line 427):
patch-depth initialisation `rand_like` → `ones_like`, for a deterministic start.

```python
# dpvo/dpvo.py — patch depth initialisation
- patches[...,2] = torch.rand_like(patches[...,2,0,0,None,None])
+ patches[...,2] = torch.ones_like(patches[...,2,0,0,None,None])
```

Without it, consecutive runs on identical input drift apart. Even with it, DPVO on GPU is
not bit-reproducible run to run — the same calibration can yield an alignment residual of
11.5 or 8.3, roughly 6 m of variability. Comparing two single runs against each other is
therefore misleading; the numbers in the README are from repeated runs.
