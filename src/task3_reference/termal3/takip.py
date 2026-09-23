#!/usr/bin/env python3
"""DAM4SAM (SAM2.1 base+) AKIS takibi — kareler TEK TEK verilir.

Neden DAM4SAM: 7 takipci x 4 politika taramasinda base_plus varyantlari
large'i her ailede gecti (hosgorulu %79.9-80.1 vs %77.0-77.7). SAMURAI ile
skor ESIT cikti ama SAMURAI toplu kare klasoru istiyor; DAM4SAM'in
initialize/track arayuzu NEDENSEL — yarisma kosuluna uygun olan bu.
"""
import os, sys
import numpy as np
from src.task3_reference.termal3 import ayarlar as A


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

    def sam_sinyal(self):
        """SON initialize/track cagrisinda SAM2'nin ZATEN hesapladigi guven sayilari.
        EK GPU MALIYETI YOK: DAM4SAM `out_dict`'te yalnizca `pred_mask` donuyor, ama
        ayni sayilar `predictor` durumunda (compact cikti) duruyor — oradan OKUNUR:
          obj  = object_score_logits (SAM2 "nesne var mi" basi; >0 = var, sigmoid ~olasilik)
          iou  = secilen maskenin ONGORULEN IoU'su (all_pred_masks[1] maksimumu; DAM4SAM
                 bu sayiyi zaten DRM bellek kararinda kullaniyor, m_iou)
          npix = maskenin pozitif piksel sayisi (n_pixels_pos)
        NOT: initialize() karesinde maske GIRDI olarak verildigi icin obj/iou sabit
        (SAM2 `_use_mask_as_output`: ious=1, obj=out_scale+out_bias) — kilit karesinde
        bu iki alan AYIRT EDICI DEGILDIR, takip karelerinde anlamlidir.
        -> {"obj":float, "iou":float, "npix":int} ya da None (yalniz teshis; karara girmez)"""
        try:
            tr = self.tr
            st = getattr(tr, "inference_state", None)
            if not st:
                return None
            fi = int(getattr(tr, "frame_index", 0))
            od = st.get("output_dict") or {}
            cik = (od.get("non_cond_frame_outputs") or {}).get(fi) \
                or (od.get("cond_frame_outputs") or {}).get(fi)
            if cik is None:          # initialize(): cikti gecici per-obj sozlukte durur
                for d in (st.get("temp_output_dict_per_obj") or {}).values():
                    cik = (d.get("cond_frame_outputs") or {}).get(fi) \
                        or (d.get("non_cond_frame_outputs") or {}).get(fi)
                    if cik is not None:
                        break
            if cik is None:
                return None
            s = {}
            o = cik.get("object_score_logits")
            if o is not None:
                v = o.detach().float().cpu().numpy() if hasattr(o, "detach") else np.asarray(o)
                v = np.atleast_1d(v).ravel()
                if v.size:
                    s["obj"] = round(float(v[0]), 4)
            ap = cik.get("all_pred_masks")
            if isinstance(ap, (tuple, list)) and len(ap) > 1 and ap[1] is not None:
                v = np.atleast_1d(np.asarray(ap[1], dtype=np.float64)).ravel()
                if v.size:
                    s["iou"] = round(float(v.max()), 4)
            npx = cik.get("n_pixels_pos")
            if npx is not None:
                s["npix"] = int(npx)
            return s or None
        except Exception:
            return None

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
