#!/usr/bin/env python3
"""DAM4SAM (SAM2.1 base+) AKIS takibi — kareler TEK TEK verilir.

Neden DAM4SAM: 7 takipci x 4 politika taramasinda base_plus varyantlari
large'i her ailede gecti (hosgorulu %79.9-80.1 vs %77.0-77.7). SAMURAI ile
skor ESIT cikti ama SAMURAI toplu kare klasoru istiyor; DAM4SAM'in
initialize/track arayuzu NEDENSEL — yarisma kosuluna uygun olan bu.
"""
import os, sys
import numpy as np
from src.task3_reference.rgb2 import ayarlar as A


def _kutu(m):
    ys, xs = np.nonzero(np.asarray(m) > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


class Takip:
    def __init__(self):
        # DAM4SAM KENDI sam2 catalini tasir (`return_all_masks` eklentisi var) ve
        # kurulu `sam2` paketiyle AYNI ISIMDE. RGB oturumunda cakisma yok
        # (kurulu sam2'yi yalnizca termal segmenters.py + eski ref_tracker.py
        # kullanir, ikisi de RGB akisinda yuklenmez). Yine de garantiye aliyoruz:
        # onbellekteki sam2 modulleri temizlenir, DAM4SAM yolu ONE alinir.
        for k in [m for m in list(sys.modules) if m == "sam2" or m.startswith("sam2.")]:
            del sys.modules[k]
        if A.DAM4SAM_REPO not in sys.path:
            sys.path.insert(0, A.DAM4SAM_REPO)
        ck = os.path.join(A.DAM4SAM_REPO, "checkpoints")
        os.makedirs(ck, exist_ok=True)
        for _ad, _yol in (("sam2.1_hiera_base_plus.pt", A.SAM2_BP_CKPT),
                          ("sam2.1_hiera_large.pt", getattr(A, "SAM2_L_CKPT", None))):
            if not _yol or not os.path.exists(_yol):
                continue
            hedef = os.path.join(ck, _ad)
            if not os.path.exists(hedef):
                os.symlink(_yol, hedef)
        from dam4sam_tracker import DAM4SAMTracker
        self.tr = DAM4SAMTracker(A.TAKIPCI)
        self.acik = False

    def baslat(self, pil, kutu, maske=None):
        """kutu: (x1,y1,x2,y2) ORIJINAL kare olceginde.
        maske verilirse SAM MASKE ile kurulur — kutu istemiyle SAM nesneyi kendi
        icinden buduyordu (olculdu: model kutusu IoU 0.92 iken SAM maskesi 0.28)."""
        import torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if maske is not None:
                o = self.tr.initialize(pil, np.asarray(maske).astype(np.uint8))
            else:
                o = self.tr.initialize(pil, None, bbox=[int(kutu[0]), int(kutu[1]),
                                                        int(kutu[2]-kutu[0]), int(kutu[3]-kutu[1])])
        self.acik = True
        m = o.get("pred_mask") if isinstance(o, dict) else None
        self._son_m = m
        return _kutu(m) if m is not None else None

    def adim(self, pil):
        return self.adim2(pil)[0]

    def adim2(self, pil):
        """(kutu, maske) dondurur — maske ucuz dogrulama ve kapsam uyumu icin gerekir."""
        import torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.tr.track(pil)
        m = o.get("pred_mask") if isinstance(o, dict) else None
        self._son_m = m
        k = _kutu(m) if m is not None else None
        if k is None:
            self.acik = False
        return k, m

    def son_maske(self, sekil):
        """son initialize/track maskesi, verilen sekle olceklenmis (bool)"""
        import cv2
        m = getattr(self, "_son_m", None)
        if m is None:
            return None
        m = np.asarray(m) > 0
        if m.shape != tuple(sekil):
            m = cv2.resize(m.astype(np.uint8), (sekil[1], sekil[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
        return m
