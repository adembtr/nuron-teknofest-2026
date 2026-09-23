#!/usr/bin/env python3
"""DAM4SAM (SAM2.1 base+) AKIS takibi — kareler TEK TEK verilir.

Neden DAM4SAM: 7 takipci x 4 politika taramasinda base_plus varyantlari
large'i her ailede gecti (hosgorulu %79.9-80.1 vs %77.0-77.7). SAMURAI ile
skor ESIT cikti ama SAMURAI toplu kare klasoru istiyor; DAM4SAM'in
initialize/track arayuzu NEDENSEL — yarisma kosuluna uygun olan bu.
"""
import os, sys
import numpy as np
from src.task3_reference.termal2 import ayarlar as A


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
        hedef = os.path.join(ck, "sam2.1_hiera_base_plus.pt")
        if not os.path.exists(hedef):
            os.symlink(A.SAM2_BP_CKPT, hedef)
        from dam4sam_tracker import DAM4SAMTracker
        self.tr = DAM4SAMTracker(A.TAKIPCI)
        self.acik = False

    def baslat(self, pil, kutu):
        """kutu: (x1,y1,x2,y2) ORIJINAL kare olceginde"""
        import torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.tr.initialize(pil, None, bbox=[int(kutu[0]), int(kutu[1]),
                                                    int(kutu[2]-kutu[0]), int(kutu[3]-kutu[1])])
        self.acik = True
        return _kutu(o.get("pred_mask")) if isinstance(o, dict) and o.get("pred_mask") is not None else None

    def adim(self, pil):
        import torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.tr.track(pil)
        k = _kutu(o.get("pred_mask")) if isinstance(o, dict) and o.get("pred_mask") is not None else None
        if k is None:
            self.acik = False
        return k
