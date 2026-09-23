#!/usr/bin/env python3
"""HİBRİT TERMAL OTURUM — referansın modalitesine göre iki kanıtlanmış hattan birini seçer.

  Referans GRİ (R=G=B, termal fotoğraf)  -> termal2.TermalOturum
        (WebSSL gri banka + turlu arama + 0.45/4 ardışık tutarlı + DAM4SAM SAM2.1-B, SADECE_TAKIP)
  Referans RENKLİ (RGB fotoğraf)         -> termal3.TermalOturum3
        (RADIO-H + FastSAM temas büyümesi + ALAN_MAKS + pencereli kilit + DAM4SAM SAM2.1-L)

Gerekçe (2026-09-05, 2026 oturumu 312 GERÇEK GT kutusu, çevrimdışı birleşim):
  gri referans aralıklarında termal2: 320-390 %91 · 1760-1840 %89 · 780-820 %57  (termal3: %84 / %50 / %23)
  renkli referans aralıklarında termal3: 400-460 %83 · 570-630 %84 · 880-920 %82  (termal2: %0 / %68 / %77)
  hibrit: GERÇEK-2026 BULMA %83.3, kesinlik %97.7, TAKİP-A %95.8  (taban %64.4 / %97.1 · termal3 tek %68.9 / %87.8)

Arayüz termal2/termal3 ile aynı: yukle() · banka_kur({oid: (crop, maske)}) · aktif(oid) · kare(frame) -> kutu|None
Teşhis için `motor` özelliği aktif hattın motorunu (termal2 için uyumlu bir kabuk) verir.
"""
import numpy as np, cv2

from src.task3_reference.termal3 import ayarlar as A
from src.task3_reference.termal3.oturum import TermalOturum3
from src.task3_reference.termal3.motor import termal_mi


class _Kabuk:
    """termal2 motoru için termal3 arayüz kabuğu (sonda / kaynak / kilit_kutu / birak_neden)."""
    def __init__(self):
        self.sonda = None; self.kaynak = None; self.kilit_kutu = None; self.birak_neden = None; self.mod = "ARAMA"


class TermalOturumHibrit:
    def __init__(self):
        from src.task3_reference.termal2.oturum import TermalOturum
        self.t2 = TermalOturum()
        self.t3 = TermalOturum3()
        self.kabuk = _Kabuk()
        self.yol = {}            # oid -> "t2" | "t3"
        self.banka = {}          # oid -> yol (arayüz uyumu)
        self.oid = None
        self.aktif_yol = "t3"

    def yukle(self):
        """Iki hatti yukler. HIBRIT_PAYLAS (varsayilan acik): FastSAM tek nesne, DAM4SAM takipcisi
        tek nesne (renkli hat da termal2'nin sam21pp-B'sini kullanir; E1 olcumu: renkli araliklarda
        L ile birebir ayni). Bellek ~6.6 GB -> ~4.5 GB. HIBRIT_PAYLAS=0 -> her hat kendi modelleri."""
        self.t2.yukle()          # WebSSL + FastSAM + SAM2.1-B
        if getattr(A, "HIBRIT_PAYLAS", True):
            from src.task3_reference.termal2 import bolucu as B2, ayarlar as A2
            from src.task3_reference.termal3 import bolucu as B3, takip as T3
            from src.task3_reference.termal3.gomme import gomme_yap
            from src.task3_reference.termal3.motor import Motor
            B3._MODEL = B2._MODEL                    # FastSAM: ayni checkpoint, tek yukleme (tembel)
            tk = T3.Takip.__new__(T3.Takip)          # termal3 sarmalayicisi, alttaki DAM4SAM termal2'ninki
            tk.tr = self.t2.tk.tr; tk.acik = False; tk._son_m = None
            A.TAKIPCI = A2.TAKIPCI                   # kayit/ozet tutarliligi
            self.t3.gomme = gomme_yap(self.t3.gomme_adi)
            self.t3.gomme2 = gomme_yap(self.t3.gomme_adi.split("|")[1]) if "|" in self.t3.gomme_adi else None
            self.t3.tk = tk
            self.t3.motor = Motor(self.t3.gomme, tk, gomme2=self.t3.gomme2)
        else:
            self.t3.yukle()      # RADIO-H + FastSAM + SAM2.1-L (ayri)
        return self

    kirpim_oku = staticmethod(TermalOturum3.kirpim_oku)

    def banka_kur(self, referanslar):
        for oid, cm in referanslar.items():
            c, m = cm if isinstance(cm, (tuple, list)) else (cm, None)
            if m is None:
                m = c.max(2) > 10
                if m.mean() < 0.02:
                    m = np.ones(c.shape[:2], bool)
            gri = termal_mi(c, m)
            zorla = getattr(A, "HIBRIT_ZORLA", "")
            yol = "t2" if gri else "t3"
            if zorla in ("t2", "t3"):
                yol = zorla
            self.yol[oid] = yol; self.banka[oid] = yol
            (self.t2 if yol == "t2" else self.t3).banka_kur({oid: (c, m)})
        return self

    def aktif(self, oid):
        self.oid = oid
        self.aktif_yol = self.yol.get(oid, "t3")
        if self.aktif_yol == "t2":
            self.t2.aktif(oid)
        else:
            self.t3.aktif(oid)
        self.kabuk = _Kabuk()

    def kare(self, frame_bgr):
        if self.aktif_yol == "t2":
            once = self.t2.mod
            k = self.t2.kare(frame_bgr)
            self.kabuk.sonda = None
            self.kabuk.kilit_kutu = list(k) if (k is not None and self.t2.mod == "TAKIP") else None
            self.kabuk.kaynak = None if k is None else ("kilit" if self.t2.mod == "TAKIP" else "model")
            self.kabuk.birak_neden = "t2_kilit_acildi" if (once == "TAKIP" and self.t2.mod == "ARAMA") else None
            self.kabuk.mod = self.t2.mod
            return k
        return self.t3.kare(frame_bgr)

    @property
    def mod(self):
        return self.t2.mod if self.aktif_yol == "t2" else self.t3.mod

    @property
    def motor(self):
        return self.kabuk if self.aktif_yol == "t2" else self.t3.motor
