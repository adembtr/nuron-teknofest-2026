#!/usr/bin/env python3
"""GOREV 3 · TERMAL v3 OTURUMU — Z3 motorunun termal girisi.

Kullanim (termal2.TermalOturum ile AYNI arayuz):
    o = TermalOturum3().yukle()              # gomme + FastSAM + DAM4SAM
    o.banka_kur({4: (crop_bgr, maske), ...}) # oturum basi bir kez
    o.aktif(4)                               # aralik basinda (takip sifirlanir)
    kutu = o.kare(frame_bgr)                 # HER KARE -> (x1,y1,x2,y2) ya da None
    o.motor.sonda / o.motor.kaynak           # teshis: son karenin aday listesi / kutunun kaynagi

Karar mantigi motor.py'de (Z3'un birebir portu). Termal farklari ayarlar.py'de:
KIP_ZORLA="termal", BANKA (renkli/gri/ikisi), GOMME_ADI, TAKIPCI.
"""
import numpy as np, cv2

from src.task3_reference.termal3 import ayarlar as A
from src.task3_reference.termal3.gomme import gomme_yap
from src.task3_reference.termal3.motor import Motor, Referans, donmeler   # noqa: F401
from src.task3_reference.termal3.takip import Takip


class TermalOturum3:
    def __init__(self, kip=None, banka=None, gomme_adi=None):
        self.kip = kip or A.KIP_ZORLA or None
        self.banka_ad = banka or A.BANKA
        self.gomme_adi = gomme_adi or A.GOMME_ADI
        self.gomme = None
        self.tk = None
        self.motor = None
        self.banka = {}          # oid -> Referans
        self.banka2 = {}         # ikili karar: ikinci modelin bankasi
        self.gomme2 = None
        self.oid = None

    # ── kurulum ────────────────────────────────────────────────
    def yukle(self):
        self.gomme = gomme_yap(self.gomme_adi)
        self.gomme2 = gomme_yap(self.gomme_adi.split("|")[1]) if "|" in self.gomme_adi else None
        self.tk = Takip()
        self.motor = Motor(self.gomme, self.tk, gomme2=self.gomme2)
        return self

    @staticmethod
    def kirpim_oku(yol):
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
        for oid, cm in referanslar.items():
            c, m = cm if isinstance(cm, (tuple, list)) else (cm, None)
            if m is None:
                m = c.max(2) > 10
                if m.mean() < 0.02:
                    m = np.ones(c.shape[:2], bool)
            self.banka[oid] = Referans(c, m, self.gomme, kip=self.kip, banka=self.banka_ad)
            if self.gomme2 is not None:
                self.banka2[oid] = Referans(c, m, self.gomme2, kip=self.kip, banka=self.banka_ad)
        return self

    def aktif(self, oid):
        self.oid = oid
        if oid in self.banka:
            self.motor.aktif(self.banka[oid], self.banka2.get(oid))
        elif self.motor is not None:
            self.motor.ref = None
            self.motor.sifirla()

    # ── kare kare ──────────────────────────────────────────────
    def kare(self, frame_bgr):
        if self.oid is None or self.oid not in self.banka:
            return None
        return self.motor.kare(frame_bgr)

    @property
    def mod(self):
        return self.motor.mod if self.motor is not None else None


RgbOturum = TermalOturum3   # geriye uyum (kopyalanan kodun adi)
