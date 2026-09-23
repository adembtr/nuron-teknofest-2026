#!/usr/bin/env python3
"""
Modalite (rgb / termal) belirleme.

Oturum başında modalite sunucudan/kullanıcıdan bildirilir (şartname: modalite oturum
başlamadan önce iletilir, oturum içinde DEĞİŞMEZ). Otomatik yedek olarak görüntüden de
tahmin edilebilir: termal görüntü 3 kanallı ama R=G=B (grayscale).
"""
import numpy as np


def detect_from_image(img: np.ndarray) -> str:
    """Görüntüden modalite tahmini. img: HxWx3 (BGR/RGB) veya HxW."""
    if img.ndim == 2:
        return "termal"
    if img.shape[2] == 1:
        return "termal"
    b, g, r = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    # kanallar birbirine çok yakınsa grayscale → termal
    diff = np.abs(b - g).mean() + np.abs(g - r).mean()
    return "termal" if diff < 2.0 else "rgb"


def resolve(explicit: str | None, sample_img: np.ndarray | None = None) -> str:
    """explicit ('rgb'/'termal') verildiyse onu kullan; yoksa görüntüden tahmin et."""
    if explicit in ("rgb", "termal"):
        return explicit
    if sample_img is not None:
        return detect_from_image(sample_img)
    raise ValueError("Modalite belirlenemedi: explicit ver ya da örnek görüntü sağla.")
