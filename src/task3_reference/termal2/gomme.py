#!/usr/bin/env python3
"""WebSSL-300M-light2b gomme — crop'lar KENDI BOYUTUNDA gecer.

Model: facebook/webssl-dino300m-light2b-224 (304M, ViT-L/14, 1024 boyut).
Meta'nin Web-SSL ailesi: 2 milyar web goruntusu, ALTYAZILAR ATILMIS, saf gorsel
SSL. Model hayatinda tek kelime gormedi. 'light2b' = yazi iceren goruntuler
ayiklanmis havuz (full2b %55.6 -> light2b %77.8, olculdu).

224'e kucultme YOLU YOK: ayni nesne bir aralikta 8.9x-200.8x alan farkiyla
gorunuyor; sabit boyut bu farki yok sayiyor. Yalnizca HAM_TAVAN'i asanlar
kucultulur. Kovalama (patch'in katina yuvarlama) ile yigin gecis yapilir.
"""
import numpy as np, cv2, torch
import torch.nn.functional as Fn
from src.task3_reference.termal2 import ayarlar as A

_ORT = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class Gomme:
    def __init__(self, yari=None):
        self.m = None
        self.patch = 14
        self.tavan = A.HAM_TAVAN
        # bf16: gomme ileri gecisi sureyi %80'ini yiyor; yari hassasiyet olculecek
        self.yari = getattr(A, "GOMME_YARI", False) if yari is None else yari
        self.dtip = torch.bfloat16 if self.yari else torch.float32

    def yukle(self):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(A.GOMME_YOLU, dtype=self.dtip,
                                           trust_remote_code=True).to(A.DEVICE).eval()
        self.patch = getattr(self.m.config, "patch_size", 14)
        return self

    def _adim(self):
        return self.patch * max(1, int(round(64 / self.patch)))    # patch14 -> 70

    @torch.inference_mode()
    def __call__(self, crops):
        """crop listesi -> (N, D) L2-normal CLS vektorleri"""
        if not crops:
            return torch.zeros(0, 1024)
        A_ = self._adim()
        tavan = max(A_, (self.tavan // A_) * A_)
        hazir, kova = [], []
        for c in crops:
            h, w = c.shape[:2]
            S = max(h, w)
            if S > tavan:                                  # yalnizca DEVLER kucultulur
                f = tavan / S
                c = cv2.resize(c, (max(16, int(w*f)), max(16, int(h*f))),
                               interpolation=cv2.INTER_AREA)
                h, w = c.shape[:2]
            K = max(A_, (max(h, w) + A_ - 1) // A_ * A_)
            z = np.zeros((K, K, 3), c.dtype); z[:h, :w] = c
            r = cv2.cvtColor(z, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            hazir.append(torch.from_numpy((r - _ORT) / _STD).permute(2, 0, 1))
            kova.append(K)
        grup = {}
        for i, K in enumerate(kova):
            grup.setdefault(K, []).append(i)
        cik = [None] * len(hazir)
        for K, idx in grup.items():
            B = int(np.clip(3_000_000 // (K*K), 1, 64))
            for s0 in range(0, len(idx), B):
                par = idx[s0:s0+B]
                x = torch.stack([hazir[q] for q in par]).to(A.DEVICE, dtype=self.dtip)
                o = self.m(pixel_values=x).last_hidden_state
                v = Fn.normalize(o[:, 0].float(), dim=-1).cpu()
                for r2, q in enumerate(par):
                    cik[q] = v[r2]
        return torch.stack(cik)
