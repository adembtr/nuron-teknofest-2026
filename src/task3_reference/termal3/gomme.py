#!/usr/bin/env python3
"""Gomme omurgalari — termal v3.

  RadyoGomme : C-RADIOv3-H (NVIDIA), Z3'un secimi. Kirpim px x px kareye getirilir
               (tavan), ImageNet normalizasyonu YOK, bf16 autocast, OZET (summary) vektoru.
  RadyoHF    : C-RADIOv3-L / -B, HF onbelleginden (trust_remote_code) — daha hizli seceneği.
  Gomme      : WebSSL-300M-light2b (termal2 omurgasi) — kirpim KENDI boyutunda gecer,
               yalniz tavani asan kucultulur, CLS vektoru, bf16 secimli.
               Ayni sinif HF DINOv3 kimlikleriyle de calisir (d3_l, d3_sp, d3_b).
  gomme_yap(ad) : ada gore kurulu omurga.

Her omurga `tavan` ozniteligi tasir: Motor aday icin A.TUR0_PX, Referans banka icin
A.HAM_TAVAN atar. __call__(crops) -> (N, D) L2-normal float tensor (CPU).
"""
import os
import numpy as np, cv2, torch
import torch.nn.functional as Fn
from src.task3_reference.termal3 import ayarlar as A

_ORT = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_HF = {
    "webssl_300m_l": (A.GOMME_YOLU, 14),
    "d3_l":  ("facebook/dinov3-vitl16-pretrain-lvd1689m", 16),
    "d3_sp": ("facebook/dinov3-vits16plus-pretrain-lvd1689m", 16),
    "d3_b":  ("facebook/dinov3-vitb16-pretrain-lvd1689m", 16),
}
_RADIO_HF = {"radio_l": "nvidia/C-RADIOv3-L", "radio_b": "nvidia/C-RADIOv3-B",
             "radio_h_hf": "nvidia/C-RADIOv3-H"}


class Gomme:
    """HF ViT (WebSSL / DINOv3) — CLS, kendi boyutunda (yalniz devler kucultulur)."""

    def __init__(self, yol=None, patch=14, yari=None, tavan=None):
        self.m = None
        self.yol = yol or A.GOMME_YOLU
        self.patch = patch
        self.tavan = tavan or A.HAM_TAVAN
        self.yari = A.GOMME_YARI if yari is None else yari
        self.dtip = torch.bfloat16 if self.yari else torch.float32

    def yukle(self):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(self.yol, dtype=self.dtip,
                                           trust_remote_code=True).to(A.DEVICE).eval()
        self.patch = getattr(self.m.config, "patch_size", self.patch)
        return self

    def _adim(self):
        return self.patch * max(1, int(round(64 / self.patch)))    # patch14 -> 70, patch16 -> 64

    @torch.inference_mode()
    def __call__(self, crops):
        if not crops:
            return torch.zeros(0, 1024)
        A_ = self._adim()
        tavan = max(A_, (int(self.tavan) // A_) * A_)
        hazir, kova = [], []
        for c in crops:
            h, w = c.shape[:2]
            S = max(h, w)
            if S > tavan:
                f = tavan / S
                c = cv2.resize(c, (max(16, int(w*f)), max(16, int(h*f))), interpolation=cv2.INTER_AREA)
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


class RadyoGomme:
    """C-RADIOv3-H (NVIDIA) — INTERNETSIZ yerel yukleme (Z3 ile birebir).

    Cozunurluk 16'nin kati olmak ZORUNDA (modelin min_resolution_step'i).
    Ozet vektoru 3840 boyut; spatial ozellikler atilir.
    """

    def __init__(self, tavan=None):
        self.m = None
        self.tavan = tavan or A.TUR0_PX

    def yukle(self):
        import torch as _t
        self.m = _t.hub.load(A.RADIO_REPO, "radio_model", source="local",
                             version=A.RADIO_CKPT, progress=False).to(A.DEVICE).eval()
        return self

    def _ileri(self, x):
        s, _ = self.m(x)
        return s

    @torch.inference_mode()
    def __call__(self, crops):
        if not crops:
            return torch.zeros(0, 1)
        px = max(64, (int(self.tavan)//16)*16)
        xs = [torch.from_numpy(cv2.cvtColor(
                  cv2.resize(c, (px, px), interpolation=cv2.INTER_AREA),
                  cv2.COLOR_BGR2RGB)).permute(2, 0, 1) for c in crops]
        x = torch.stack(xs).float().div_(255).to(A.DEVICE)
        out = []
        with torch.autocast(A.DEVICE, dtype=torch.bfloat16):
            for z in range(0, len(x), 16):
                out.append(self._ileri(x[z:z+16]).float().cpu())
        v = torch.cat(out)
        return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


class RadyoHF(RadyoGomme):
    """C-RADIOv3 -L / -B: HF onbelleginden (trust_remote_code). Ayni __call__."""

    def __init__(self, hf_id, tavan=None):
        super().__init__(tavan)
        self.hf_id = hf_id

    def yukle(self):
        from transformers import AutoModel
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.m = AutoModel.from_pretrained(self.hf_id, trust_remote_code=True).to(A.DEVICE).eval()
        return self

    def _ileri(self, x):
        o = self.m(x)
        if isinstance(o, (tuple, list)):
            return o[0]
        return getattr(o, "summary", o[0])


class CiftGomme:
    """IKI omurga: vektorler yan yana (her yari L2-normal). Kosinus = iki kosinusun ortalamasi
    olur; Motor._skorla ise `bolum`/`olcek` ile yarilari ayri okuyup gz_min fuzyonu yapar
    (sonda kiyasi: gz_min ILK1 %67.6 ~ WebSSL tek %68.0; RADIO tek %58.3)."""

    def __init__(self, adlar):
        self.adlar = adlar
        self.g = [gomme_yap(a) for a in adlar]
        self.bolum = None
        self.olcek = [A.GOMME_OLCEK.get(a, (0.5, 0.1)) for a in adlar]
        self._tavan = A.HAM_TAVAN

    @property
    def tavan(self):
        return self._tavan

    @tavan.setter
    def tavan(self, v):
        self._tavan = v
        for g in self.g:
            g.tavan = v

    def yukle(self):
        return self

    def __call__(self, crops):
        vs = [g(crops) for g in self.g]
        self.bolum = [v.shape[1] for v in vs]
        return torch.cat(vs, dim=1) if crops else torch.zeros(0, sum(self.bolum or [1]))


def _esikler(ad):
    """Model olcegine gore SKOR_MIN / DEN_COS / CIPA_COS (RADIO z-esdegeri)."""
    ma, sa = A.GOMME_OLCEK["radio_h"]
    if ad in A.GOMME_OLCEK:
        m, sd = A.GOMME_OLCEK[ad]
        z = lambda e: round(m + sd*(e - ma)/sa, 3)
        return dict(skor_min=z(0.35), den_cos=z(0.30), cipa_cos=z(0.25))
    return dict(skor_min=0.15, den_cos=0.10, cipa_cos=0.10)


def gomme_yap(ad=None):
    """ad: tek model | 'a+b' (vektor fuzyonu, CiftGomme) — 'a|b' ikili KARAR icin oturum iki ayri gomme kurar."""
    ad = ad or A.GOMME_ADI
    if "|" in ad:
        ad = ad.split("|")[0]
    if "+" in ad:
        g = CiftGomme(ad.split("+")); g.ad = ad; g.esik = _esikler("radio_h"); return g
    if ad in ("radio_h", "radio"):
        g = RadyoGomme().yukle()
    elif ad in _RADIO_HF:
        g = RadyoHF(_RADIO_HF[ad]).yukle()
    elif ad in _HF:
        yol, patch = _HF[ad]
        g = Gomme(yol=yol, patch=patch).yukle()
    else:
        raise ValueError(f"bilinmeyen gomme: {ad}  (secenekler: radio_h, {', '.join(_RADIO_HF)}, {', '.join(_HF)})")
    g.ad = ad; g.esik = _esikler(ad)
    return g
