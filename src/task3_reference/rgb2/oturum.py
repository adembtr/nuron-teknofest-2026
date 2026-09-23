#!/usr/bin/env python3
"""GOREV 3 · RGB v2 OTURUMU — yarisma girisi (M16 motoru, 2026-09-01).

Kullanim (DEGISMEDI — cagiran kodlar aynen calisir):
    o = RgbOturum().yukle()                  # modeller + FastSAM
    o.banka_kur({4: crop_bgr_rgba, ...})     # oturum basi bir kez
    o.aktif(4)                               # aralik basinda
    kutu = o.kare(frame_bgr)                 # HER KARE — None = gonderme

Ic yapi tamamen degisti; karar mantigi motor.py'de. Ozet:
    ARAMA : FastSAM adaylari -> TEMAS BUYUMESI ile butunlestirilir -> C-RADIOv3-H
            ile referansa gore skorlanir. 1. aday, FARKLI bir nesneden yeterince
            ayrisiyorsa (marj) ve bu ust uste KILIT_N karede tekrarlaniyorsa SAM
            BIR KEZ o nesnenin MASKESI ile kurulur.
    TAKIP : DAM4SAM akisi kare kare. Model kutuyu ARTIK SECMEZ, sadece denetler:
            (a) her DEN_HER karede SAM maskesinin kendisi gomulup referansa
                benzerligi olculur — iki kez ust uste dusukse kilit ATILIR,
            (b) model AYNI YERDE belirgin farkli boyut gosterirse iki olcumle
                onaylanip SAM o kutuya yeniden kurulur (buyut/kucult).
    BITTI : nesne kadrajdan cikip SAM'i kaybettiyse arama KAPANIR (benzer
            nesneye siçrama olmaz).

Olculen (14 referans, 1163 etiketli kare): nesneyi bulma %88.5, dogru kutu %69.6,
kare basi 0.88 sn. Ayrinti: nuron_referance_rgb/SISTEM.md
"""
import numpy as np, cv2

from src.task3_reference.rgb2 import ayarlar as A
from src.task3_reference.rgb2.gomme import RadyoGomme
from src.task3_reference.rgb2.motor import Motor, Referans, donmeler   # noqa: F401 (donmeler: geriye uyum)
from src.task3_reference.rgb2.takip import Takip

_nrm = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class RgbOturum:
    def __init__(self, esik=None, ardisik=None):
        # esik/ardisik artik modaliteye gore ayarlar.KURAL'dan gelir; imza geriye uyum icin durur
        self.gomme = None
        self.tk = None
        self.motor = None
        self.banka = {}          # oid -> Referans
        self.oid = None

    # ── kurulum ────────────────────────────────────────────────
    def yukle(self):
        self.gomme = RadyoGomme().yukle()
        self.tk = Takip()
        self.motor = Motor(self.gomme, self.tk)
        return self

    @staticmethod
    def kirpim_oku(yol):
        """PNG oku -> (bgr, maske). Alfa varsa maske alfadan, yoksa siyah-olmayan
        piksellerden turetilir (yarismanin verdigi ref_crops 3 kanalli gelir)."""
        im = cv2.imread(yol, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(yol)
        if im.ndim == 3 and im.shape[2] == 4:
            return im[:, :, :3].copy(), im[:, :, 3] > 10
        m = im.max(2) > 10
        if m.mean() < 0.02:
            m = np.ones(im.shape[:2], bool)
        return im, m

    def banka_kur(self, referanslar):
        """referanslar: {oid: (crop_bgr, maske_bool|None)} — crop KESILMIS gelir.
        Her referans icin modalite (RGB/termal) OTOMATIK belirlenir ve o referansin
        kural kumesi secilir."""
        for oid, cm in referanslar.items():
            c, m = cm if isinstance(cm, (tuple, list)) else (cm, None)
            if m is None:
                m = c.max(2) > 10
                if m.mean() < 0.02:
                    m = np.ones(c.shape[:2], bool)
            self.banka[oid] = Referans(c, m, self.gomme)
        return self

    def aktif(self, oid):
        """Aralik basinda cagrilir — takip sifirlanir."""
        self.oid = oid
        if oid in self.banka:
            self.motor.aktif(self.banka[oid])
        elif self.motor is not None:
            self.motor.ref = None
            self.motor.sifirla()

    # ── kare kare ──────────────────────────────────────────────
    def kare(self, frame_bgr):
        """-> (x1,y1,x2,y2) ORIJINAL kare olceginde, ya da None (gonderme)"""
        if self.oid is None or self.oid not in self.banka:
            return None
        return self.motor.kare(frame_bgr)
