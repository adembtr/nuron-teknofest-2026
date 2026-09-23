#!/usr/bin/env python3
"""GOREV 3 · TERMAL v2 OTURUMU — yarisma girisi.

Kullanim:
    o = TermalOturum().yukle()          # WebSSL + FastSAM + DAM4SAM
    o.banka_kur({4: crop_bgr, ...})     # oturum basi BIR kez
    o.aktif(4)                          # her aralik basinda (takip sifirlanir)
    kutu = o.kare(frame_bgr)            # HER KARE -> (x1,y1,x2,y2) ya da None

DURUM MAKINESI (olculdu, en iyi cikan):
  ARAMA : eslestirici HER karede calisir.
          skor >= CIPA VE onceki tespitle TUTARLI  -> cipa adayi
          ARDISIK (4) ardisik aday birikince       -> SAM CIPALANIR, KILITLENIR
          SADECE_TAKIP acikken bu modda HICBIR kutu GONDERILMEZ.
  TAKIP : DAM4SAM kare kare surer, eslestirici calismaz.
          her DOGRULA (5) karede eslestirici DENETIM icin calisir:
            kutusu SAM ile IoU < UYUS (0.30) ise uyusmazlik sayaci artar
            BOZ (2) kez ust uste uyusmazsa -> KILIT ACILIR
                matcher skoru CIPA ustundeyse oraya yeniden cipalanir
                degilse ARAMA'ya donulur
          SAM maskeyi kaybederse -> KILIT ACILIR

NEDEN SADECE_TAKIP (2026, 253 GT kutusu):
    kapali: 225 dogru / 22 YANLIS / 6 bos   -> dogruluk %88.9  kesinlik %91.1
    ACIK  : 201 dogru /  6 YANLIS / 46 bos  -> dogruluk %79.4  kesinlik %97.1
    24 dogru verip 16 yanlistan kurtulur. Yanlis kutu ceza yedigi icin secildi.
"""
import numpy as np, cv2
from PIL import Image
from src.task3_reference.termal2 import ayarlar as A, bolucu as B, arama as AR
from src.task3_reference.termal2.gomme import Gomme
from src.task3_reference.termal2.takip import Takip

_nrm = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _gri(c):
    return cv2.cvtColor(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _ters(c):
    return cv2.cvtColor(255 - cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _clahe(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.createCLAHE(3.0, (8, 8)).apply(g), cv2.COLOR_GRAY2BGR)


def fotograflar(c, m):
    """Referansin bankaya giren fotograflari (A.BANKA ile secilir)."""
    out = []
    for ad in A.BANKA.split("+"):
        if ad == "renkli":  out.append((c, m))
        elif ad == "gri":   out.append((_gri(c), m))
        elif ad == "ters":  out.append((_ters(c), m))
        elif ad == "clahe": out.append((_clahe(c), m))
    return out or [(_gri(c), m)]


def _donmeler(c, m):
    """4 x 90 derece donme. Baska varyant YOK (olculdu)."""
    out = []
    for d in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        cc = c if d is None else cv2.rotate(c, d)
        mm = m if d is None else cv2.rotate(m.astype(np.uint8), d).astype(bool)
        out.append((cc, mm))
    return out


def _kutu_iou(a, b):
    if a is None or b is None: return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    it = max(0, ix2-ix1) * max(0, iy2-iy1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/u if u > 0 else 0.0


def _merkez(b): return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)


def _yakin(a, b, oran):
    if a is None or b is None: return False
    k = np.hypot(a[2]-a[0], a[3]-a[1])
    return np.hypot(*(np.array(_merkez(a)) - np.array(_merkez(b)))) <= oran*max(k, 1.0)


class TermalOturum:
    def __init__(self, gonder=None, cipa=None, ardisik=None, sadece_takip=None):
        self.gonder = A.GONDER if gonder is None else gonder
        self.cipa = A.CIPA if cipa is None else cipa
        self.ardisik = A.ARDISIK if ardisik is None else ardisik
        self.sadece_takip = A.SADECE_TAKIP if sadece_takip is None else sadece_takip
        self.gomme = None
        self.tk = None
        self.banka = {}          # oid -> (K, D)
        self.oid = None
        self._sifirla()

    def _sifirla(self):
        self.mod, self.art, self.onceki = "ARAMA", 0, None
        self.kilit, self.uyusmaz = 0, 0

    # ── kurulum ────────────────────────────────────────────────
    def yukle(self):
        self.gomme = Gomme().yukle()
        self.tk = Takip()
        return self

    @staticmethod
    def kirpim_oku(yol):
        """PNG -> (bgr, maske). Alfa varsa ondan, yoksa siyah-olmayan piksellerden."""
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
        """referanslar: {oid: crop_bgr | (crop_bgr, maske)} -> oid basina (4, D)"""
        for oid, cm in referanslar.items():
            c, m = cm if isinstance(cm, (tuple, list)) else (cm, None)
            if m is None:
                m = c.max(2) > 10
                if m.mean() < 0.02:
                    m = np.ones(c.shape[:2], bool)
            girdiler = [B.girdi(cc, mm)
                        for foto, mk in fotograflar(c, m)
                        for cc, mm in _donmeler(foto, mk)]
            self.banka[oid] = _nrm(self.gomme(girdiler).numpy())
        return self

    def aktif(self, oid):
        """Aralik basinda cagrilir — takip ve sayaclar sifirlanir."""
        self.oid = oid
        self._sifirla()
        if self.tk:
            self.tk.acik = False

    # ── kare kare ──────────────────────────────────────────────
    def kare(self, frame_bgr):
        """-> (x1,y1,x2,y2) ORIJINAL kare olceginde, ya da None (gonderme)."""
        if self.oid is None or self.oid not in self.banka:
            return None
        H, W = frame_bgr.shape[:2]
        s = A.MAXSIDE/max(H, W) if max(H, W) > A.MAXSIDE else 1.0
        img = cv2.resize(frame_bgr, (int(W*s), int(H*s)),
                         interpolation=cv2.INTER_AREA) if s < 1.0 else frame_bgr
        pil = lambda: Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        if self.mod == "ARAMA":
            ku, sk = AR.ara(img, self.gomme, self.banka[self.oid])
            if ku is None or sk < self.gonder:
                self.art, self.onceki = 0, None
                return None
            tam = tuple(int(v/s) for v in ku)
            gonderilecek = None if self.sadece_takip else tam
            aday = sk >= self.cipa
            if aday and A.TUTARLI and self.onceki is not None:
                aday = _yakin(tam, self.onceki, A.TUTARLI)
            self.onceki = tam
            if not aday:
                self.art = 0
                return gonderilecek
            self.art += 1
            if self.art >= self.ardisik:
                b = self.tk.baslat(pil(), tam)
                if b is not None:
                    self.mod, self.art, self.onceki = "TAKIP", 0, None
                    self.kilit, self.uyusmaz = 0, 0
                    return tuple(int(v) for v in b)
            return gonderilecek

        # ── TAKIP ──
        b = self.tk.adim(pil())
        self.kilit += 1
        if b is None:                                   # SAM kaybetti -> KILIT ACILIR
            self.mod, self.art, self.onceki = "ARAMA", 0, None
            return None
        kutu = tuple(int(v) for v in b)
        if A.DOGRULA and self.kilit % A.DOGRULA == 0:   # periyodik denetim
            ku, sk = AR.ara(img, self.gomme, self.banka[self.oid])
            if ku is not None and sk >= self.cipa:
                mk = tuple(int(v/s) for v in ku)
                self.uyusmaz = self.uyusmaz + 1 if _kutu_iou(mk, kutu) < A.UYUS else 0
                if self.uyusmaz >= A.BOZ:               # KILIT ACILIR
                    self.uyusmaz = 0
                    b2 = self.tk.baslat(pil(), mk)
                    if b2 is not None:
                        self.kilit = 0
                        return tuple(int(v) for v in b2)
                    self.mod, self.art, self.onceki = "ARAMA", 0, None
                    return None
        return kutu
