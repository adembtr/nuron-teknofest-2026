#!/usr/bin/env python3
"""
rpg_trajectory_evaluation ile AYNI metrik: Sim3 hizalama (Umeyama, olcek dahil) → translation RMSE.
Yarisma Gorev 2 skoru budur. Mono icin 'sim3' (olcek serbest); 'se3' (olcek sabit) ve 'none' da var.

Girdi: dunya-cerceve traje [N,3] (x=Dogu, y=Kuzey, z=Asagi) + GT [N,3].
Cikti: 3B / XY / Z RMSE (tum + tahmin bolgesi).
"""
import numpy as np


def umeyama(src, dst, with_scale=True):
    """dst ≈ s·R·src + t. rpg 'sim3' = with_scale=True, 'se3' = False."""
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    C = D.T @ S / len(src)
    U, sv, Vt = np.linalg.svd(C)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (np.trace(np.diag(sv) @ E) / (S ** 2).sum() * len(src)) if with_scale else 1.0
    t = md - s * R @ ms
    return R, s, t


def align(est, gt, mode="sim3", idx=None):
    """est'i gt'ye hizala (idx alt-kumesiyle fit; hepsine uygula)."""
    est = np.asarray(est, float); gt = np.asarray(gt, float)
    I = np.arange(len(est)) if idx is None else np.asarray(idx)
    if mode == "none":
        return est.copy()
    R, s, t = umeyama(est[I], gt[I], with_scale=(mode == "sim3"))
    return (s * (R @ est.T)).T + t


def rmse(a, b, ax=None):
    a = np.asarray(a); b = np.asarray(b)
    if ax is not None:
        a, b = a[:, ax], b[:, ax]
    d = a - b
    if d.ndim == 1:
        return float(np.sqrt((d ** 2).mean()))
    return float(np.sqrt((d ** 2).sum(1).mean()))


def score(est, gt, mode="sim3", G=450, align_idx=None, label=""):
    """rpg-stili skor. align_idx: hizalama fit karesi (None=hepsi, rpg varsayilani)."""
    al = align(est, gt, mode, align_idx)
    out = dict(
        mode=mode,
        RMSE_3D=rmse(al, gt),
        RMSE_XY=rmse(al, gt, [0, 1]),
        RMSE_Z=rmse(al[:, 2], gt[:, 2]),
        RMSE_3D_pred=rmse(al[G:], gt[G:]),
        RMSE_XY_pred=rmse(al[G:], gt[G:], [0, 1]),
    )
    if label:
        print(f"[{label}] mode={mode}  3B={out['RMSE_3D']:.2f}  XY={out['RMSE_XY']:.2f}  "
              f"Z={out['RMSE_Z']:.2f}  (tahmin: 3B={out['RMSE_3D_pred']:.2f} XY={out['RMSE_XY_pred']:.2f})", flush=True)
    return out, al


def load_traj(path):
    """traj.txt: kare x y z  (dunya cercevesi)."""
    t = np.loadtxt(path)
    return t[:, 0].astype(int), t[:, 1:4]


if __name__ == "__main__":
    import sys, pandas as pd
    traj, gtcsv = sys.argv[1], sys.argv[2]
    kare, est = load_traj(traj)
    gt = pd.read_csv(gtcsv)[['translation_x', 'translation_y', 'translation_z']].values[kare]
    for md in ("sim3", "se3", "none"):
        score(est, gt, md, label=traj.split('/')[-2] if '/' in traj else traj)
