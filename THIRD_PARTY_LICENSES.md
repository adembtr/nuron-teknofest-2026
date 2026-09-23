# Third-party licences

NURON's own code is Apache-2.0 (see [LICENSE](LICENSE)). It depends on the projects
below, each under its own terms. **Read this before redistributing or deploying
commercially.**

## ⚠️ AGPL-3.0 — Ultralytics YOLO

Task 1 detection is built on [Ultralytics](https://github.com/ultralytics/ultralytics),
licensed **AGPL-3.0**. AGPL-3.0 is copyleft and reaches network use: if you run a modified
version as a network service, you must offer the corresponding source to its users.

This governs the YOLO-derived components and the checkpoints trained with them. Ultralytics
also sells a commercial licence for projects that cannot comply with AGPL-3.0.

If you intend to use NURON commercially, resolve this first.

## Other dependencies

| Project | Licence | Used for |
|---|---|---|
| [DPVO](https://github.com/princeton-vl/DPVO) | MIT | Task 2 visual odometry |
| [SAM 2.1](https://github.com/facebookresearch/sam2) | Apache-2.0 | Task 3 tracking |
| [HQ-SAM](https://github.com/SysCV/sam-hq) | Apache-2.0 | Task 3 reference cropping |
| [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) | AGPL-3.0 | Task 3 candidate generation |
| [NVIDIA RADIO](https://github.com/NVlabs/RADIO) | NSCLv1 | Task 3 colour embeddings |
| [Meta WebSSL](https://huggingface.co/facebook/webssl-dino300m-full2b-224) | FAIR NC | Task 3 grayscale embeddings |
| [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) | Apache-2.0 | Task 3 distractor-aware tracking |
| PyTorch, OpenCV, NumPy, SciPy | BSD / Apache-2.0 | general |

> **FastSAM is also AGPL-3.0.** **NVIDIA RADIO** ships under the NVIDIA Source Code Licence
> and **WebSSL** under a FAIR non-commercial licence — both restrict commercial use.
> The permissive Apache/MIT components are the exception here, not the rule.

## Competition assets

Competition imagery, reference objects, ground truth and the evaluation server belong to
**TEKNOFEST** and are not redistributed in this repository. `connect/` is the organisers'
official team-interface, included so the pipeline is reproducible; its protocol is unmodified.
