# `models/`

Weights go here. **Nothing in this folder is committed** — see the root
[`.gitignore`](../.gitignore).

## Trained by Team NURON — YOLO + PaDiM

```bash
pip install huggingface_hub
hf download adembtr/nuron-teknofest-2026 --local-dir models/
```

## Everything else

DPVO, HQ-SAM, FastSAM, C-RADIOv3-H, WebSSL DINO-300M and SAM 2.1 are unmodified public
checkpoints — download them from their original authors.

**Full table, expected layout and the one-line DPVO patch: [`../docs/MODELS.md`](../docs/MODELS.md).**
