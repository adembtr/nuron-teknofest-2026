#!/usr/bin/env python3
"""GOREV 3 · TERMAL v3 — Z3 MOTORU (RGB 2026-09-02 surumunun termal portu).

TERMAL EKLERI (2026-09-04): Referans(kip=, banka=) — kip zorlama + banka varyantlari;
Motor.sonda (aday dokumu) ve Motor.kaynak ("kilit"/"model"); A.SADECE_TAKIP secenegi.
Karar mantigi Z3 ile BIREBIR AYNI.

Bir referans araligi boyunca kare kare calisir: nesneyi ARAR, bir KURALLA kilitler,
SAM ile TAKIP eder, kilidi surekli dogrular ve kutuyu ayni yerde buyutup kucultur.

TASARIM KARARLARI (hepsi olcumle secildi, gerekce SISTEM.md'de):

 1) KARELER ARASI KONUM KARSILASTIRMASI YOK. Video 7.5 fps ve drone donuyor; iki kare
    arasinda "burasi eskiden ayni yer miydi" sorusu guvenilmez. Butun kararlar AYNI
    KARE icindeki olculere ve konumsuz sayaclara dayanir.

 2) TEK SAM. Once 3 SAM'li oylama denendi; hem 8 GB'a sigmiyor hem gereksiz. Model bir
    kuralla dogru nesneyi secer, SAM BIR KEZ kurulur.

 3) KILIT KURALI = skor + MARJ. Mutlak skor referanstan referansa kayiyor (bir referansta
    dogru nesne 0.638, bir baskasinda yanlis nesne 0.821). 1. adayin FARKLI bir nesneden
    ne kadar ayristigi (marj) modaliteden bagimsiz calisir. Ust uste KILIT_N arama
    karesinde saglanmali.

 4) BIRLESTIRME = TEMAS GRAFIGI. Parca ancak birlesime DEGIYORSA katilir; boylece
    farkli yerlerdeki benzer nesneler birlesemez. Her birlesimin ici doldurulur.

 5) TEK KILIT. Supheli kilit duzeltilmez, ATILIR ve aramaya donulur. Tek istisna:
    kutunun AYNI YERDE buyutulup kucultulmesi (kapsam uyumu) — o da iki olcumle onaylanir.

 7) RAKIPSIZ KARE. Karede FARKLI nesneden rakip yoksa marj OLCULEMEZ; bu "sonsuz
    marj" sayilip serbest gecis verilmez — yuksek MUTLAK skor sarti aranir (bosa
    gonderimin ana kaynagi buydu: bos pencerelerde 51 -> 1).

 8) COZUNURLUK KADEMESI. Ust uste ARTIR_N arama karesinde kilitlenilemezse FastSAM
    cozunurlugu 512 -> 768 yukselir ve orada kalir (kucuk nesne kurtarma). Kare basina
    hep TEK model kosusu vardir.

 9) KILIT SONRASI UC BEKCI (Z3): (a) DUZELTME PENCERESI — kilitten sonra DUZELT kare
    boyunca model SAM'i oylar, hic onay yoksa kilit atilir; (b) PERIYODIK DENETIM —
    her DENETIM_N karede model kosup SAM'le karsilastirilir, ust uste DEN_RED kararli
    red kilidi atar (SAM kendini maske benzerligiyle savunabilir); (c) KENAR-GIRIS —
    kenarda dogan kilit iceri girince kisa bir dogrulama penceresi kosar.
 6) MODALITE. Referans termalse (R=G=B) adaylar da griye cevrilerek gomulur ve tum
    marj/kazanc esikleri ~1/3'e iner (termalde kosinus dagilimi sikisik).
"""
import numpy as np, cv2
from collections import deque
from PIL import Image

from src.task3_reference.termal3 import ayarlar as A, bolucu as B
from src.task3_reference.termal3.takip import Takip

_nrm = lambda x: x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-8)


# ─────────────────────────── yardimcilar ───────────────────────────
def kutu_of(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    ke = (x2-x1)*(y2-y1)
    return ke/float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ke)


def alan(b):
    return max(1, (b[2]-b[0])*(b[3]-b[1])) if b else 1


def merkez_uyum(a, b):
    """AYNI NESNE mi (kutu tam olmasa da): merkez kapsanmasi ya da IoU>=0.15"""
    if not a or not b:
        return False
    ca = ((a[0]+a[2])/2, (a[1]+a[3])/2); cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    if b[0] <= ca[0] <= b[2] and b[1] <= ca[1] <= b[3]:
        return True
    if a[0] <= cb[0] <= a[2] and a[1] <= cb[1] <= a[3]:
        return True
    return iou(a, b) >= 0.15


def delik_kapa(m):
    """Nesnenin ICI ASLA DELIK KALMAZ: disariya acilmayan her bosluk nesnenindir."""
    u = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(u)
    if n <= 1:
        return m
    dis = set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) | \
          set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    return m | ~np.isin(lab, list(dis))


def _delik_doldur(m):
    u8 = m.astype(np.uint8); dol = u8.copy()
    ff = np.zeros((m.shape[0]+2, m.shape[1]+2), np.uint8)
    cv2.floodFill(dol, ff, (0, 0), 1)
    return u8.astype(bool) | ((dol == 0) & (u8 == 0))


def tamamla(m):
    """FastSAM parcali maske uretiyor: kutu icinde morfolojik kapama + delik doldurma."""
    if A.KAPAT <= 0:
        return m
    k = kutu_of(m)
    if k is None:
        return m
    cap = max(k[2]-k[0], k[3]-k[1]) + 1
    r = max(3, int(cap*A.KAPAT) | 1)
    cek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    t = _delik_doldur(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cek).astype(bool))
    kb = np.zeros_like(t, bool); kb[k[1]:k[3]+1, k[0]:k[2]+1] = True
    return t & kb


def en_boy(m):
    """maskenin (en/boy orani, uzun kenar) — donmeye dayanikli minAreaRect"""
    cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cn:
        return None, 0.0
    c = max(cn, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    if w < 1 or h < 1:
        return None, 0.0
    uzun, kisa = max(w, h), min(w, h)
    return float(uzun/max(1e-6, kisa)), float(uzun)


def _sekil64(m, k):
    p = m[k[1]:k[3], k[0]:k[2]].astype(np.uint8)
    h, w = p.shape
    S = max(h, w)
    kare = np.zeros((S, S), np.uint8)
    kare[(S-h)//2:(S-h)//2+h, (S-w)//2:(S-w)//2+w] = p
    return cv2.resize(kare, (64, 64), interpolation=cv2.INTER_NEAREST) > 0


def _hist(bgr, m):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], m.astype(np.uint8), [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)


def donmeler(c, m):
    """Referans bankasi = SADECE 4 x 90 derece donme (baska varyant olculdu, zarar verdi)."""
    out = []
    for d in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        cc = c if d is None else cv2.rotate(c, d)
        mm = m if d is None else cv2.rotate(m.astype(np.uint8), d).astype(bool)
        out.append((cc, mm))
    return out


def _gri3(im):
    """3 kanalli gri (R=G=B)."""
    return cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def termal_mi(crop_bgr, maske):
    """Referans TERMAL mi: termalde R=G=B (olculdu: kanal farki 0.0; en soluk renkli
    referansta bile 6.4). Elle isaretleme gerekmez."""
    kp = crop_bgr[:, :, :3].astype(np.int16)
    return float((kp.max(2)-kp.min(2))[maske].mean()) < 1.0


# ─────────────────────────── referans kaydi ───────────────────────────
class Referans:
    """Bir nesnenin bankasi + ondan turetilen olcutler + modalite kurali."""

    def __init__(self, crop_bgr, maske, gomme, kip=None, banka=None):
        self.gri = termal_mi(crop_bgr, maske)          # referans GORUNTUSU gri mi
        # KIP (karar kurallari): termal OTURUMDA kare hep gridir -> A.KIP_ZORLA="termal".
        # Bos ise Z3 davranisi: referans griyse "gri", degilse "rgb".
        self.kip = kip or getattr(A, "KIP_ZORLA", None) or ("gri" if self.gri else "rgb")
        self.kural = dict(A.KURAL[self.kip])
        # BANKA: referansin gomulen halleri ("renkli" | "gri" | "renkli+gri"), her hal x 4 donme
        self.banka_ad = banka or getattr(A, "BANKA", "renkli")
        gomme.tavan = A.HAM_TAVAN
        girdiler = [B.girdi(c, m) for foto, mk in self._fotograflar(crop_bgr, maske)
                    for c, m in donmeler(foto, mk)]
        self.banka = _nrm(gomme(girdiler).numpy())
        self.banka_etiket = [f"foto{i}/{d}" for i in range(len(girdiler) // 4) for d in (0, 90, 180, 270)]
        self.eb, _ = en_boy(maske)
        self.eb = self.eb or 1.0
        k = kutu_of(maske)
        self.wh = (k[2]-k[0]+1)/max(1, k[3]-k[1]+1) if k else 1.0
        nb, lab = cv2.connectedComponents(maske.astype(np.uint8))
        if nb > 1:
            en = max(range(1, nb), key=lambda z: int((lab == z).sum()))
            m2 = (lab == en); k2 = kutu_of(m2)
            self.dolu = float(m2[k2[1]:k2[3], k2[0]:k2[2]].mean())
            rs = _sekil64(m2, k2)
        else:
            self.dolu = 0.5
            rs = _sekil64(maske, k)
        self.sekil = [rs, np.rot90(rs, 1), np.rot90(rs, 2), np.rot90(rs, 3)]
        self.hist = _hist(crop_bgr, maske)

    def _foto_adlari(self):
        adlar = [a for a in self.banka_ad.split("+") if a in ("renkli", "gri", "yarim")]
        return adlar or ["renkli"]

    def _fotograflar(self, c, m):
        """Bankaya giren referans halleri. "yarim" (2026-09-05 denemesi): gri referansin sol/sag/ust/alt
        YARIMLARI — nesne kadraja kenardan girerken (1760-1840: 10 kare gecikme) kismi gorunumle eslesebilsin."""
        out = []
        for ad in self._foto_adlari():
            if ad == "renkli":
                out.append((c, m))
            elif ad == "gri":
                out.append((_gri3(c), m))
            elif ad == "yarim":
                g = _gri3(c); h, w = m.shape
                if h >= 32 and w >= 32:
                    for cc, mm in ((g[:, :w//2], m[:, :w//2]), (g[:, w//2:], m[:, w//2:]),
                                   (g[:h//2], m[:h//2]), (g[h//2:], m[h//2:])):
                        if mm.sum() >= 0.25 * m.sum():          # bos yarim (maske disi) eklenmez
                            out.append((cc, mm))
        return out

    def girdi(self, c, m):
        """modele giren ADAY goruntusu. Referans gri ya da kip gri/termal ise aday da
        griye cevrilir (termal karede zaten gri; RGB karede gri referans icin gerekli)."""
        g = B.girdi(c, m)
        if self.gri or self.kip in ("gri", "termal"):
            g = _gri3(g)
        return g


# ─────────────────────────── motor ───────────────────────────
class Motor:
    """Tek referans araligini kare kare surer. `kare(frame_bgr)` -> kutu ya da None."""

    def __init__(self, gomme, takip=None, gomme2=None):
        self.gomme = gomme
        self.gomme2 = gomme2          # IKILI KARAR: ikinci gomme (bagimsiz hakem), None = tek model
        self.tk = takip if takip is not None else Takip()
        self.ref = None
        self.ref2 = None
        self.sifirla()

    # ── durum ────────────────────────────────────────────────
    def sifirla(self):
        """Aralik basi: HER SEY sifirlanir (cozunurluk kademesi dahil)."""
        self.mod = "ARAMA"
        self.kilit_kutu = None
        self.pes_say = 0
        self.ara_atla = 0
        self.kademe = 0          # cozunurluk kademesi (A.ARTIR_PX indeksi)
        self.kademe_bos = 0      # ust uste kilitlenemeyen arama karesi
        self.kap_bek = None
        self.kap_zorla = 0
        self.cos_red = 0
        self.kilit_yas = 0
        self.onay_gordu = False
        self.den_say = 0
        self.den_red = 0
        self.den_bos = 0
        self.den_onceki = None
        self.denet = 0           # kenar-giris dogrulama penceresi sayaci
        self.uyum_say = 0
        self.duz = 0             # duzeltme penceresi kalan kare
        self.duz_uyum = 0
        self.duz_ornek = 0
        self.duz_onceki = None
        self.red_say = 0
        self.sup_bekle = False
        self.kilit_kenar = set()
        self.kenar_yapildi = False
        self.alan_onceki = None
        self.tepe_alan = 1
        self.kucul = 0
        self.cikis_izi = False
        self.tk.acik = False
        self.sonda = None        # son arama karesinin aday dokumu (teshis)
        self.kaynak = None       # son gonderilen kutunun kaynagi: "kilit" | "model" | None
        self.pen = deque(maxlen=max(1, getattr(A, "PEN_W", 3)))   # pencereli oylama tamponu
        self.pen2 = deque(maxlen=max(1, getattr(A, "PEN_W", 3)))  # ikinci modelin penceresi
        self.ates = [None, None]   # ikili: her modelin son atesi (arama_sayac, kutu, ms_idx, uyeler)
        self.ara_sayac = 0
        self.birak_neden = None  # bu karede kilit birakildiysa sebebi
        self.birak_say = 0       # bu aralikta kac kez kilit birakildi (KAPI_SONRA icin)
        # ── YANLIS-POZITIF KAPILARI (hepsi varsayilan KAPALI, bkz. ayarlar.FP_KAPI) ──
        self.pen_ozet = None     # son kilit denemesinin ozeti (teshis; sonda["kilit_ozet"])
        self.isin_kalan = 0      # ISINMA: kilit sonrasi izlenecek model karesi
        self.isin_uyum = 0       # ISINMA: goruldugu onay sayisi
        self.soguma = []         # SOGUMA: [(kutu_orijinal, biten_kare_no)] yasakli kutular
        self.kare_no = 0         # islenen kare sayaci (soguma yaslandirmasi)
        self.son_kutu = None     # bir onceki karenin kilit kutusu (soguma listesine girer)
        # ── KILIT SONRASI TANI (yalniz KAYIT; hicbir karara GIRMEZ, bkz. _tani_kur) ──
        self.takip_tani = None   # bu karenin takip sinyalleri (takip karesi degilse None)
        self._tani = {}          # bu karede hesaplanan ara olculer (den_cos / onay)
        self._tani_onceki = None # (onceki takip kutusu, onceki alan orani) — suregen olcutler

    def aktif(self, ref, ref2=None):
        self.ref = ref
        self.ref2 = ref2
        self.sifirla()

    def kilit_birak(self, neden="?"):
        """TEK KILIT: supheli kilit DUZELTILMEZ, ATILIR — aramaya donulur.
        (Cozunurluk kademesi KORUNUR: pencerenin nesne boyu degismedi.)
        neden: teshis icin (sam_kayip | cikis | ucuz_dogrulama | duzeltme_red | duzeltme_onaysiz |
               denetim_bos | denetim_red | kenar_giris | isinma)
        SOGUMA (varsayilan kapali): birakilan kutu SOGUMA_N kare boyunca yasak listesine
        girer; o kutuyla IoU>=SOGUMA_IOU olan kazanan kume yeniden kilit ALAMAZ."""
        self.birak_neden = neden
        self.birak_say = getattr(self, 'birak_say', 0) + 1
        if getattr(A, "SOGUMA_N", 0) > 0:
            k0 = self.kilit_kutu or self.son_kutu
            if k0:
                self.soguma.append((tuple(k0), self.kare_no + int(A.SOGUMA_N)))
        self.isin_kalan = 0
        self.isin_uyum = 0
        self.mod = "ARAMA"
        self.kilit_kutu = None
        self.pes_say = 0
        self.ara_atla = 0
        self.kap_bek = None
        self.kap_zorla = 0
        self.denet = 0
        self.duz = 0
        self.duz_uyum = 0
        self.duz_ornek = 0
        self.duz_onceki = None
        self.red_say = 0
        self.den_red = 0
        self.den_bos = 0
        self.den_onceki = None
        self.sup_bekle = False
        self.kucul = 0
        self.cikis_izi = False
        self.kenar_yapildi = False
        self._tani_onceki = None   # yeni kilit yeni zincir (teshis)
        self.pen.clear(); self.pen2.clear(); self.ates = [None, None]

    # ── pencereli oylama (TERMAL kilit kurali) ───────────────
    def _pencere_kilit(self, sira, sk, ku, pen=None, boyut=None, olcek=1.0):
        """Son PEN_W arama karesinde ilk PEN_K aday IoU>=PEN_IOU zinciriyle kumelenir.
        En iyi kume en az PEN_M karede goruldu VE puani 2. kumenin PEN_ORAN katiysa KILIT.
        Gerekce (sonda, 1100 kare): tek kare top-1 %45 dogru ama ilk-3 %78; GT kareler arasi
        az kayiyor (IoU med 0.66-0.99). -> bu karedeki uyenin ms indeksi ya da None

        YANLIS-POZITIF KAPILARI (hepsi varsayilan KAPALI = eski davranis birebir):
        PEN_S1_MIN/PEN_PUAN_MIN/PEN_MARJ_MIN/PEN_Z_MIN/PEN_TAM/PEN_KARARLI/PEN_KENAR
        ve SOGUMA_N. Kilit denenen her karede `self.pen_ozet` doldurulur (teshis).
        boyut: (W,H) — ku ile AYNI olcekte (PEN_KENAR icin; None = kenar kapisi atlanir).
        olcek: kare()'deki s_ (ku/olcek = orijinal olcek); SOGUMA kutulari orijinal olcekte."""
        pen = self.pen if pen is None else pen
        adaylar = [(float(sk[q]), tuple(ku[q]), int(q)) for q in sira[:A.PEN_K] if ku[q] is not None]
        pen.append(adaylar)
        self.pen_ozet = None
        if len(pen) < A.PEN_M:
            return None
        kume = []
        for j, kare in enumerate(pen):
            for r, (sc, kb, q) in enumerate(kare):
                en, eniyi = None, 0.0
                for c in kume:
                    v = max(iou(kb, u[2]) for u in c["u"][-6:])
                    if v >= A.PEN_IOU and v > eniyi:
                        en, eniyi = c, v
                if en is None:
                    kume.append(dict(u=[(j, sc, kb, q, r)], kareler={j}))
                else:
                    en["u"].append((j, sc, kb, q, r)); en["kareler"].add(j)
        if not kume:
            return None
        def puan(c):
            if getattr(A, "PEN_AG", "skor") == "sira":
                return sum(max(0, A.PEN_K - u[4]) for u in c["u"])
            return sum(u[1] for u in c["u"])
        kume.sort(key=lambda c: -puan(c))
        p1 = puan(kume[0]); p2 = puan(kume[1]) if len(kume) > 1 else 0.0
        son = len(pen) - 1
        uye = [u for u in kume[0]["u"] if u[0] == son]
        self.kilit_uyeler = [u[3] for u in uye]          # kazanan kumenin bu karedeki tum uyeleri (ms idx)
        kazanan = max(uye, key=lambda u: u[1]) if uye else None
        self.pen_ozet = self._kilit_ozet(kazanan, kume[0], p1, p2, pen, sira, sk, ku)
        if len(kume[0]["kareler"]) >= A.PEN_M and p1 >= A.PEN_ORAN * max(p2, 1e-6) and uye:
            red = self._fp_kapilar(kazanan, self.pen_ozet, boyut, olcek)
            if red:
                self.pen_ozet["ret"] = red      # hangi kapi kesti (teshis)
                return None
            return kazanan[3]
        return None

    # ── kilit ani teshisi + yanlis-pozitif kapilari ──────────
    def _kilit_ozet(self, kazanan, kume, p1, p2, pen, sira, sk, ku):
        """Kilit DENENEN her arama karesinde en iyi kumenin olculeri (JSON'a girer).
        {p1,p2,oran,kareler,tam,s1,marj,z,kararli} — kilit olmasa da doldurulur."""
        # marj: SONDA ile AYNI tanim (top-1 ile FARKLI nesneden ilk rakip arasi fark)
        marj = 9.0
        if sira:
            s_1 = float(sk[sira[0]]); s_2 = -9.0
            for q in sira[1:]:
                if ku[q] is not None and ku[sira[0]] is not None and not merkez_uyum(ku[q], ku[sira[0]]):
                    s_2 = float(sk[q]); break
            marj = s_1 - s_2 if s_2 > -8 else 9.0
        # z: kazanan skorun bu karedeki ilk-K aday skorlarina gore z-degeri.
        # TANIM = arastirma/fp_kural_tara.py kapi_gec() ile BIREBIR: ort/sd TUM ilk-K
        # skoru uzerinden (kazanan DAHIL), nufus sd'si; sd<=1e-6 -> olculemez (None).
        z = None
        if kazanan is not None and sira:
            K = max(2, int(getattr(A, "SONDA_K", 0) or 8))
            d = np.asarray([float(sk[q]) for q in sira[:K]], np.float64)
            if d.size > 1:
                sd = float(d.std())
                z = float((kazanan[1] - float(d.mean())) / sd) if sd > 1e-6 else -9.0
        # kararli (ic_iou): kazanan kume uye kutularinin ardisik ikili IoU ortalamasi.
        # TANIM = fp_kural_tara.py: uyeler kume["u"] SIRASIYLA duz listelenir (kare-major),
        # ardisik ciftlerin IoU ortalamasi; hic cift yoksa 1.0 (olcum yok = engelleme yok).
        kut = [u[2] for u in kume["u"]]
        ard = [iou(kut[i], kut[i+1]) for i in range(len(kut)-1)]
        kararli = float(sum(ard)/len(ard)) if ard else 1.0
        return dict(p1=round(float(p1), 4), p2=round(float(p2), 4),
                    oran=round(float(p1/max(p2, 1e-6)), 3),
                    kareler=int(len(kume["kareler"])), tam=bool(len(kume["kareler"]) == len(pen)),
                    s1=(round(float(kazanan[1]), 4) if kazanan is not None else None),
                    marj=round(float(marj), 4),
                    z=(round(z, 3) if z is not None else None),
                    kararli=(round(kararli, 3) if kararli is not None else None))

    def _fp_kapilar(self, kazanan, oz, boyut=None, olcek=1.0):
        """Kilit anindaki yanlis-pozitif kapilari. Hepsi kapaliyken HEP None doner
        (eski davranis birebir). -> kesen kapinin adi ya da None (kilit serbest)"""
        if kazanan is None or oz is None:
            return None
        if getattr(A, "KAPI_SONRA", False) and getattr(self, "birak_say", 0) == 0:
            return None         # KAPI_SONRA: ilk kilit taban kuraliyla; kapilar yalniz kilit BIRAKILDIKTAN sonraki yeniden kilitlerde
        if getattr(A, "PEN_S1_MIN", 0.0) > 0 and (oz["s1"] is None or oz["s1"] < A.PEN_S1_MIN):
            return "s1"
        if getattr(A, "PEN_PUAN_MIN", 0.0) > 0 and oz["p1"] < A.PEN_PUAN_MIN:
            return "puan"
        if getattr(A, "PEN_MARJ_MIN", 0.0) > 0 and oz["marj"] < A.PEN_MARJ_MIN:
            return "marj"
        if getattr(A, "PEN_Z_MIN", 0.0) > 0 and oz["z"] is not None and oz["z"] < A.PEN_Z_MIN:
            return "z"          # z=None (tek aday) ise simulatordeki gibi kapi calismaz
        if getattr(A, "PEN_TAM", False) and not oz["tam"]:
            return "tam"
        if getattr(A, "PEN_KARARLI", 0.0) > 0 and oz["kararli"] < A.PEN_KARARLI:
            return "kararli"
        if getattr(A, "PEN_KENAR", False) and boyut and self._kenarda(kazanan[2], boyut[0], boyut[1]):
            return "kenar"
        if getattr(A, "SOGUMA_N", 0) > 0 and self.soguma:
            ko = tuple(int(v/olcek) for v in kazanan[2]) if olcek != 1.0 else tuple(kazanan[2])
            if any(iou(ko, b) >= getattr(A, "SOGUMA_IOU", 0.5) for b, _ in self.soguma):
                return "soguma"   # yakin gecmiste birakilan kutuya yeniden kilitlenme YOK
        return None

    def _onay(self, kilit, sira, ku, s_):
        """`_onay_ham` + TANI KAYDI (davranis birebir ayni; yalniz sonucu not eder)."""
        o = self._onay_ham(kilit, sira, ku, s_)
        try:
            self._tani["onay"] = bool(o)
        except Exception:
            pass
        return o

    def _onay_ham(self, kilit, sira, ku, s_):
        """SAM kutusu modelin ilk ONAY_K adayindan biriyle uyusuyor mu (IoU ya da merkez+boyut).
        Termalde top-1 gurultulu (ILK1 %45, ILK3 %78) -> yalniz top-1'e bakmak iyi kilidi bozar."""
        if not sira or kilit is None:
            return False
        for q in sira[:max(1, getattr(A, "ONAY_K", 1))]:
            if ku[q] is None:
                continue
            mk = tuple(int(v/s_) for v in ku[q])
            if iou(mk, kilit) >= A.UYUM or (merkez_uyum(mk, kilit)
                    and alan(kilit) <= A.OY_ORAN*alan(mk) and alan(mk) <= A.OY_ORAN*alan(kilit)):
                return True
        return False

    # ── kilit sonrasi tani (YALNIZ KAYIT) ────────────────────
    _tani_hata = False        # ilk tani hatasi bir kez bildirilir (sessizce yutulmaz)

    def _tani_kur(self, msam, W, H):
        """Bu TAKIP karesinin sinyalleri -> self.takip_tani (JSON'a girer, kucuk ve duz).
        KARARA GIRMEZ: kilit sonrasi yanlis kilidi erken tanimak icin ONCE veri toplanir.
        msam: takip maskesi (img olceginde, bool) ya da None.  W,H: ORIJINAL kare boyu.
        Alanlar: yas · alan (maske/kare) · en_boy · kutu_iou_onceki · merkez_kayma(+_px) ·
                 alan_degisim · den_cos · den_red · cos_red · duz · uyum_say · onay ·
                 sam_skor (SAM2 object_score_logits) · sam_iou (ongorulen IoU) · sam_npix.
        `kaynak` / `kilit` / `birak` kayit sozlugunde ZATEN var — burada tekrarlanmaz."""
        try:
            t = dict(getattr(self, "_tani", {}) or {})
            t.setdefault("den_cos", None)
            t.setdefault("onay", None)
            k = self.kilit_kutu
            t["yas"] = int(self.kilit_yas)
            t["den_red"] = int(self.den_red)
            t["cos_red"] = int(self.cos_red)
            t["duz"] = int(self.duz)
            t["uyum_say"] = int(self.uyum_say)
            al = None
            if msam is not None:
                mm = np.asarray(msam) > 0
                if mm.size:
                    al = float(mm.sum()) / float(mm.size)   # maske/kare — olcekten bagimsiz
                eb, _u = en_boy(mm)
                t["en_boy"] = round(float(eb), 3) if eb else None
            elif k:
                al = alan(k) / float(max(1, W*H))
                t["en_boy"] = round((k[2]-k[0]+1) / max(1.0, float(k[3]-k[1]+1)), 3)
            else:
                t["en_boy"] = None
            t["alan"] = round(al, 6) if al is not None else None
            onc = self._tani_onceki
            if k and onc and onc[0]:
                t["kutu_iou_onceki"] = round(iou(tuple(k), onc[0]), 3)
                dx = (k[0]+k[2])/2.0 - (onc[0][0]+onc[0][2])/2.0
                dy = (k[1]+k[3])/2.0 - (onc[0][1]+onc[0][3])/2.0
                d = float(np.hypot(dx, dy))
                t["merkez_kayma_px"] = round(d, 1)
                t["merkez_kayma"] = round(d / max(1.0, float(np.hypot(W, H))), 4)
            else:
                t["kutu_iou_onceki"] = None
                t["merkez_kayma_px"] = None
                t["merkez_kayma"] = None
            t["alan_degisim"] = (round(al/onc[1], 3)
                                 if (al and onc and onc[1]) else None)
            s = None
            try:
                if hasattr(self.tk, "sam_sinyal"):
                    s = self.tk.sam_sinyal()
            except Exception:
                s = None
            t["sam_skor"] = (s or {}).get("obj")
            t["sam_iou"] = (s or {}).get("iou")
            t["sam_npix"] = (s or {}).get("npix")
            self.takip_tani = t
            self._tani_onceki = ((tuple(int(v) for v in k), al) if k else None)
        except Exception as e:
            self.takip_tani = None
            self._tani_onceki = None
            if not Motor._tani_hata:
                Motor._tani_hata = True
                print(f"[tani] HATA (bir kez bildirilir, kosu SURER): "
                      f"{type(e).__name__}: {e}", flush=True)

    # ── gomme yardimcilari ───────────────────────────────────
    def _gom(self, crops, gomme=None):
        G = gomme or self.gomme
        G.tavan = A.TUR0_PX
        V = np.concatenate([_nrm(G(crops[z:z+64]).numpy())
                            for z in range(0, len(crops), 64)])
        return V

    def _skorla(self, crops, gomme=None, ref=None):
        G = gomme or self.gomme; R = ref or self.ref
        V = self._gom(crops, G)
        bol = getattr(G, "bolum", None)
        if not bol or len(bol) < 2:
            return (V @ R.banka.T).max(1)
        # CIFT GOMME: yarilari ayri skorla, kuresel z'lerin MINIMUMU, RADIO olcegine geri haritala
        ma, sa = A.GOMME_OLCEK["radio_h"]
        zs = []; s0 = 0
        for (m, sd), d in zip(G.olcek, bol):
            Vk = V[:, s0:s0+d]; Bk = R.banka[:, s0:s0+d]
            nk = np.linalg.norm(Vk, axis=1, keepdims=True) + 1e-8
            nb = np.linalg.norm(Bk, axis=1, keepdims=True) + 1e-8
            zs.append((((Vk/nk) @ (Bk/nb).T).max(1) - m) / sd)
            s0 += d
        return ma + sa * np.minimum.reduce(zs)

    def maske_skor(self, img, m):
        """Bir maskenin referansa benzerligi — SAM'in kendini savunmasi icin."""
        if m is None:
            return -9.0
        k = kutu_of(m)
        if k is None or k[2]-k[0] < 8 or k[3]-k[1] < 8:
            return -9.0
        c2, m2 = B.kirp_gen(img, tamamla(m), k, A.GEN)
        return float(self._skorla([self.ref.girdi(c2, m2)])[0])

    def _renk_skor(self, img, m):
        d = cv2.compareHist(self.ref.hist, _hist(img, m), cv2.HISTCMP_BHATTACHARYYA)
        return 1.0 - float(d)

    def _sekil_skor(self, m, k):
        s = _sekil64(m, k)
        return max(float((s & r).sum()/max(1, (s | r).sum())) for r in self.ref.sekil)

    def _kenarda(self, b, W, H):
        z = set()
        if b[0] <= A.KENAR_PX: z.add("sol")
        if b[1] <= A.KENAR_PX: z.add("ust")
        if b[2] >= W-A.KENAR_PX: z.add("sag")
        if b[3] >= H-A.KENAR_PX: z.add("alt")
        return z

    # ── aday uretimi ─────────────────────────────────────────
    def model_kos(self, img, ucuz, px=None, gomme=None, ref=None):
        """FastSAM -> on suzgec -> gomme -> TEMAS BUYUMESI -> BOLME -> yeniden siralama.
        -> (sira, sk, ms, ku)  [sira: skor sirali aday indeksleri]
        px: FastSAM cozunurlugu (kademe); yoksa ucuz->A.ARA_PX, degilse tam (MAXSIDE).
        gomme/ref: ikili kararda ikinci model icin (varsayilan self.gomme/self.ref)."""
        G = gomme or self.gomme; R = ref or self.ref
        skor_min = getattr(G, "esik", {}).get("skor_min", A.SKOR_MIN) if G is not self.gomme else A.SKOR_MIN
        _px = px if px else (A.ARA_PX if ucuz else 0)
        if _px > 0:
            eski = A.MAXSIDE; A.MAXSIDE = _px
            try:
                ham = B.bol(img)
            finally:
                A.MAXSIDE = eski
        else:
            ham = B.bol(img)
        L_h, L_w = img.shape[:2]
        ms = [tamamla(m) for m in ham if int(m.sum()) >= A.ALAN_MIN]
        ku = [kutu_of(m) for m in ms]

        # ON SUZGEC (hiz): en/boy referansa uymayan aday GOMULMEZ — aday sayisi ~yariya iner
        cr, ok = [], []
        for q, m in enumerate(ms):
            k = ku[q]
            if k is None or k[2]-k[0] < 8 or k[3]-k[1] < 8:
                continue
            if getattr(A, "ALAN_MAKS", 0) and (k[2]-k[0])*(k[3]-k[1]) > A.ALAN_MAKS*L_h*L_w:
                continue      # TERMAL: kare boyutlu ZEMIN bolgesi aday olamaz (400-460: top-1 %94 kare)
            e0, _ = en_boy(m)
            if e0 is None or max(e0/R.eb, R.eb/e0) > A.EB_TOL:
                continue
            c2, m2 = B.kirp_gen(img, m, k, A.GEN)
            cr.append(R.girdi(c2, m2)); ok.append(q)
        sk = np.full(len(ms), -9.0)
        if cr:
            sk[np.asarray(ok)] = self._skorla(cr, G, R)

        # ── TEMAS BUYUMESI ──
        if len(ms):
            KA = L_h*L_w
            cek = np.ones((A.TEMAS_PX,)*2, np.uint8)
            gen = [cv2.dilate(m.astype(np.uint8), cek).astype(bool) for m in ms]
            kom = {q: set() for q in range(len(ms))}
            for i2 in range(len(ms)):
                ki = ku[i2]
                if ki is None:
                    continue
                for j2 in range(i2+1, len(ms)):
                    kj = ku[j2]
                    if kj is None:
                        continue
                    if (kj[2] < ki[0]-A.TEMAS_PX or kj[0] > ki[2]+A.TEMAS_PX
                            or kj[3] < ki[1]-A.TEMAS_PX or kj[1] > ki[3]+A.TEMAS_PX):
                        continue
                    if (gen[i2] & ms[j2]).any():
                        kom[i2].add(j2); kom[j2].add(i2)
            # TOHUM CESITLILIGI: en iyi 3 skor cogu zaman AYNI nesnenin parcalari
            sirali = [int(q) for q in np.argsort(-sk) if sk[q] > -8]
            tohumlar = []
            for q in sirali:
                if ku[q] is None or any(merkez_uyum(ku[q], ku[o]) for o in tohumlar):
                    continue
                tohumlar.append(q)
                if len(tohumlar) >= A.BIR_TOHUM:
                    break
            yeni, imza = [], set()
            for t in tohumlar:
                U = {t}; um = ms[t].copy(); us = float(sk[t])
                for _ in range(A.TEMAS_D):
                    kmsu = [c for q in U for c in kom[q] if c not in U]
                    if not kmsu:
                        break
                    kmsu = sorted(dict.fromkeys(kmsu), key=lambda c: -sk[c])[:A.TEMAS_KOM]
                    dene, dm = [], []
                    for c in kmsu:
                        bb = delik_kapa(um | ms[c])
                        kk = kutu_of(bb)
                        if kk is None or (kk[2]-kk[0])*(kk[3]-kk[1]) > A.BIR_MAKS*KA:
                            continue
                        e3, _ = en_boy(bb)
                        if e3 is None or max(e3/R.eb, R.eb/e3) > A.EB_TOL:
                            continue
                        dl = float(bb[kk[1]:kk[3], kk[0]:kk[2]].mean())
                        if max(dl/R.dolu, R.dolu/dl) > A.DOLU_TOL:
                            continue      # referanstan cok farkli DOLULUK = zemin birlesimi
                        dene.append(c); dm.append(bb)
                    if not dene:
                        break
                    ss = self._skorla([R.girdi(*B.kirp_gen(img, b2, kutu_of(b2), A.GEN))
                                       for b2 in dm], G, R)
                    en = int(np.argmax(ss))
                    if float(ss[en]) >= us + R.kural["TEMAS_ART"]:
                        U.add(dene[en]); um = dm[en]; us = float(ss[en])
                        kk = kutu_of(um)
                        if kk not in imza:
                            imza.add(kk); yeni.append((um.copy(), us))
                        continue
                    # IKILI SICRAMA: nesne ancak IKI parcayla tamamlaniyorsa tek adim gormez
                    if A.IKILI and len(dene) >= 2:
                        sir2 = list(np.argsort(-ss))[:A.IKILI_UST]
                        ci, cm = [], []
                        for a2 in range(len(sir2)):
                            for b3 in range(a2+1, len(sir2)):
                                bb = delik_kapa(um | ms[dene[sir2[a2]]] | ms[dene[sir2[b3]]])
                                kk3 = kutu_of(bb)
                                if kk3 is None or (kk3[2]-kk3[0])*(kk3[3]-kk3[1]) > A.BIR_MAKS*KA:
                                    continue
                                e4, _ = en_boy(bb)
                                if e4 is None or max(e4/R.eb, R.eb/e4) > A.EB_TOL:
                                    continue
                                ci.append(1); cm.append(bb)
                        if ci:
                            ss2 = self._skorla([R.girdi(*B.kirp_gen(img, b2, kutu_of(b2), A.GEN))
                                                for b2 in cm], G, R)
                            en2 = int(np.argmax(ss2))
                            if float(ss2[en2]) >= us + R.kural["TEMAS_ART"]:
                                um = cm[en2]; us = float(ss2[en2])
                                kk4 = kutu_of(um)
                                if kk4 not in imza:
                                    imza.add(kk4); yeni.append((um.copy(), us))
                                continue
                    break
            if yeni:
                ms = list(ms) + [m for m, _ in yeni]
                ku = list(ku) + [kutu_of(m) for m, _ in yeni]
                sk = np.concatenate([sk, np.array([v for _, v in yeni], np.float64)])

        # ── BOLME: FastSAM nesneyi cevresiyle KAYNASTIRDIYSA yarisi dogru olabilir ──
        bol_kume = set()
        if A.BOL and len(ms):
            KA2 = L_h*L_w
            yeni2, imza2 = [], set()
            for q in [int(z) for z in np.argsort(-sk) if sk[z] > -8][:A.BIR_TOHUM]:
                k = ku[q]
                if k is None or (k[2]-k[0])*(k[3]-k[1]) < A.BOL_MIN*KA2:
                    continue
                cx, cy = (k[0]+k[2])//2, (k[1]+k[3])//2
                sol = np.zeros((L_h, L_w), bool); sag = np.zeros((L_h, L_w), bool)
                if (k[2]-k[0]) >= (k[3]-k[1]):
                    sol[:, :cx] = True; sag[:, cx:] = True
                else:
                    sol[:cy, :] = True; sag[cy:, :] = True
                for p in (ms[q] & sol, ms[q] & sag):
                    if int(p.sum()) < A.ALAN_MIN:
                        continue
                    kk2 = kutu_of(p)
                    if kk2 is None or kk2 in imza2:
                        continue
                    # YUTAN-YARIM KAPISI: yarimin icinde kendinden cok kucuk, esik ustu
                    # AYRI bir aday varsa asil nesne muhtemelen ODUR.
                    yut = False
                    for q2 in range(len(ms)):
                        if q2 == q or sk[q2] < skor_min or ku[q2] is None:
                            continue
                        if alan(ku[q2]) <= 0.2*alan(kk2) and iou(ku[q2], kk2) > 0:
                            gx1, gy1 = max(ku[q2][0], kk2[0]), max(ku[q2][1], kk2[1])
                            gx2, gy2 = min(ku[q2][2], kk2[2]), min(ku[q2][3], kk2[3])
                            if (gx2-gx1)*(gy2-gy1) >= 0.8*alan(ku[q2]):
                                yut = True; break
                    if yut:
                        continue
                    imza2.add(kk2); yeni2.append(tamamla(p))
            if yeni2:
                ilk = len(ms)
                ss3 = self._skorla([R.girdi(*B.kirp_gen(img, m, kutu_of(m), A.GEN))
                                    for m in yeni2], G, R)
                ms = list(ms) + yeni2
                ku = list(ku) + [kutu_of(m) for m in yeni2]
                sk = np.concatenate([sk, ss3])
                bol_kume = set(range(ilk, len(ms)))

        # ── YENIDEN SIRALAMA: kosinus + RENK + SEKIL (bolme yarimlari bonus ALMAZ) ──
        aday = [int(q) for q in np.argsort(-sk) if sk[q] >= skor_min]
        if aday:
            n = min(len(aday), A.RER_K)
            puan = {}
            for q in aday[:n]:
                puan[q] = float(sk[q]) if q in bol_kume else (
                    float(sk[q]) + R.kural["W_RENK"]*self._renk_skor(img, ms[q])
                    + A.W_SEKIL*self._sekil_skor(ms[q], ku[q]))
            aday = sorted(aday[:n], key=lambda q: -puan[q]) + aday[n:]
        return aday[:A.UST_ADAY], sk, ms, ku

    # ── kare kare ────────────────────────────────────────────
    # Z3 AKISI (mimari.py'nin birebir portu). Onemli nuans: kilit birakilan karede
    # bile genel GONDERIM kapisi calisir — model tepe adayi esikleri geciyorsa o kare
    # bos gecmez (Z3 olcumleri bu davranisla alindi). Bu yuzden TAKIP bloklari erken
    # return ETMEZ; her yol sondaki tek gonderim kapisina dusur.
    def kare(self, frame_bgr):
        """-> (x1,y1,x2,y2) ORIJINAL kare olceginde, ya da None (gonderme)"""
        if self.ref is None or self.mod == "BITTI":
            self.takip_tani = None
            return None
        R = self.ref
        H, W = frame_bgr.shape[:2]
        s_ = A.MAXSIDE/max(H, W) if max(H, W) > A.MAXSIDE else 1.0
        img = cv2.resize(frame_bgr, (int(W*s_), int(H*s_)),
                         interpolation=cv2.INTER_AREA) if s_ < 1 else frame_bgr
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.sonda = None; self.kaynak = None; px_kul = None; self.birak_neden = None
        self.takip_tani = None; self._tani = {}; msam_tani = None   # TANI: kare basi sifir
        self.kare_no += 1
        self.son_kutu = self.kilit_kutu          # SOGUMA: birakilan kutuyu bulmak icin
        if self.soguma:                          # SOGUMA yaslandirmasi (kare tabanli)
            self.soguma = [(b, t) for b, t in self.soguma if t > self.kare_no]

        # ── bu karede model kosacak mi ──
        ara_kos = (self.mod == "ARAMA" and self.ara_atla <= 0)
        duz_kos = (self.duz > 0 and (A.DUZELT-self.duz) % A.DUZ_ADIM == 0)
        den_per = (A.DENETIM_N > 0 and (self.den_say+1) % A.DENETIM_N == 0)
        den_kos = (self.mod == "TAKIP" and self.duz == 0 and self.denet == 0
                   and (self.sup_bekle or den_per))
        if self.mod == "TAKIP" and self.duz == 0 and self.denet == 0 and A.DENETIM_N > 0:
            self.den_say += 1
        if self.mod == "ARAMA" and self.ara_atla > 0:
            self.ara_atla -= 1

        # ISINMA (varsayilan kapali): kilit sonrasi ilk ISINMA_N model karesi izlenir.
        # ISINMA_HER acikken model bu pencere boyunca HER karede kosar (ek GPU yuku:
        # kare basina bir tam cozunurluk model_kos; kapaliyken ek yuk YOK).
        isin_kos = (getattr(A, "ISINMA_N", 0) > 0 and getattr(A, "ISINMA_HER", False)
                    and self.mod == "TAKIP" and self.isin_kalan > 0)
        sira, sk, ms, ku = [], None, [], []
        model_kutu = None
        kos_bu_kare = bool(ara_kos or self.denet > 0 or duz_kos or den_kos
                           or self.kap_zorla or isin_kos)
        if kos_bu_kare:
            if ara_kos and self.mod == "ARAMA":       # arama: kademe cozunurlugu (ucuz)
                px = A.ARTIR_PX[min(self.kademe, len(A.ARTIR_PX)-1)]; px_kul = px
                sira, sk, ms, ku = self.model_kos(img, True, px)
            else:                                     # takip donemi kosulari: tam cozunurluk
                sira, sk, ms, ku = self.model_kos(img, False)
            if sira:
                model_kutu = tuple(int(v/s_) for v in ku[sira[0]])

        # ══ ARAMA: TEK SAM — skor + FARKLI nesneden marj, ust uste KILIT_N kare ══
        kilitlendi = False
        if self.mod == "ARAMA" and ara_kos:
            self.ara_atla = 0 if sira else (A.ARA_ADIM-1)
            s1 = float(sk[sira[0]]) if sira else -9.0
            s2 = -9.0
            for q in sira[1:]:
                if not merkez_uyum(ku[q], ku[sira[0]]):   # ayni nesnenin parcasi rakip degil
                    s2 = float(sk[q]); break
            marj = s1 - s2 if s2 > -8 else 9.0
            if getattr(A, "SONDA_K", 0) and sira:      # SONDA: kural taramasi icin ham dokum
                _sg = sk[sk > -8] if sk is not None else np.zeros(0)
                self.sonda = dict(px=px_kul, s1=s1, s2=s2, marj=marj, kademe=self.kademe,
                                  n_aday=int(_sg.size),
                                  s_ort=(round(float(_sg.mean()), 4) if _sg.size else None),
                                  s_sd=(round(float(_sg.std()), 4) if _sg.size else None),
                                  adaylar=[(round(float(sk[q]), 4), tuple(int(v/s_) for v in ku[q]))
                                           for q in sira[:A.SONDA_K]])
            secilen = None                               # kilitlenecek adayin ms indeksi
            _boyut = (img.shape[1], img.shape[0])        # FP kapilari: ku ile ayni olcek
            kilit_ozet = None
            if getattr(A, "KILIT_MOD", "z3") == "pencere" and self.gomme2 is not None and self.ref2 is not None:
                # IKILI KARAR (2026-09-05): iki bagimsiz gomme, iki pencere; ikisi de son IKILI_BEKLE arama
                # karesinde AYNI yerde (IoU>=0.5) ates ettiyse kilit. Sonda simulasyonu: 2026'da 7/7 dogru kilit.
                self.ara_sayac += 1
                f1 = self._pencere_kilit(sira, sk, ku, self.pen, _boyut, s_)
                kilit_ozet = self.pen_ozet
                if f1 is not None:
                    self.ates[0] = (self.ara_sayac, tuple(ku[f1]), int(f1), list(getattr(self, "kilit_uyeler", []) or []))
                sira2, sk2, ms2, ku2 = self.model_kos(img, True, px_kul, self.gomme2, self.ref2)
                f2 = self._pencere_kilit(sira2, sk2, ku2, self.pen2, _boyut, s_)
                if f2 is not None:
                    self.ates[1] = (self.ara_sayac, tuple(ku2[f2]), int(f2), [])
                a1, a2 = self.ates
                bekle = getattr(A, "IKILI_BEKLE", 3)
                if (a1 and a2 and self.ara_sayac - a1[0] <= bekle and self.ara_sayac - a2[0] <= bekle
                        and iou(a1[1], a2[1]) >= getattr(A, "IKILI_IOU", 0.5)):
                    if f1 is not None:
                        secilen = f1
                    else:
                        # model 1 bu karede ates etmedi: onun son atesine en yakin bu-kare adayi
                        en, eniyi = None, 0.0
                        for q in sira[:A.PEN_K]:
                            v = iou(ku[q], a1[1])
                            if v > eniyi: en, eniyi = int(q), v
                        if en is not None and eniyi >= 0.5:
                            secilen = en; self.kilit_uyeler = [en]
                        else:
                            secilen = None
                    if secilen is not None:
                        s1 = float(sk[secilen])
                        if self.sonda is not None:
                            self.sonda["ikili"] = dict(a1=a1[0], a2=a2[0], iou=round(iou(a1[1], a2[1]), 3))
            elif getattr(A, "KILIT_MOD", "z3") == "pencere":
                secilen = self._pencere_kilit(sira, sk, ku, None, _boyut, s_)
                kilit_ozet = self.pen_ozet
                if secilen is not None:
                    s1 = float(sk[secilen])
            else:
                # RAKIPSIZ KARE: marj olculemez -> yuksek MUTLAK skor sarti
                esik = R.kural["RAKIPSIZ"] if marj > 8 else R.kural["KILIT_SKOR"]
                uygun = bool(sira) and s1 >= esik and marj >= R.kural["KILIT_MARJ"]
                self.pes_say = self.pes_say+1 if uygun else 0
                if uygun and (self.pes_say >= R.kural["KILIT_N"] or s1 >= R.kural["KILIT_TEK"]):
                    secilen = sira[0]
            if self.sonda is not None and kilit_ozet is not None:
                self.sonda["kilit_ozet"] = kilit_ozet   # kilit OLMASA da: FP kurali taramasi icin
            if secilen is not None:
                kut, msk = ku[secilen], ms[secilen]
                # KILIT_BIRLESIM (termal, 2026-09-05): kazanan kumenin bu karedeki uyeleri ayni
                # nesnenin farkli kapsamlari (IoU>=PEN_IOU zinciri) -> maskeleri BIRLESTIR.
                # 780-820: yalniz traktor uyesiyle kurulan kilit IoU 0.59, romorklu uye ayni kumedeydi.
                uyeler = getattr(self, "kilit_uyeler", None) or []
                if getattr(A, "KILIT_BIRLESIM", False) and len(uyeler) > 1:
                    mb = np.zeros_like(msk, dtype=bool)
                    for q in uyeler:
                        if q < len(ms):
                            mb |= ms[q]
                    kb = kutu_of(mb)
                    if kb is not None and alan(kb) <= 2.5 * alan(kut):
                        kut, msk = kb, mb
                # CIFT COZUNURLUK: kilit aninda bir kez tam cozunurluk — ayni nesneyi
                # gosteren daha iyi kutu varsa SAM ONA kurulur (BICIM SARTIYLA; sart
                # olmadan takas nesnenin ICINDEKI parcaya geciyordu, olculdu).
                if A.CIFT_COZ:
                    s3, sk3, ms3, ku3 = self.model_kos(img, False)
                    for q in s3[:3]:
                        if (float(sk3[q]) > s1 and merkez_uyum(ku3[q], kut)
                                and iou(ku3[q], kut) >= A.CIFT_ES):
                            # TERMAL (2026-09-04): takasta MASKE kullanilir. Kutu istemiyle SAM
                            # bilesik nesnenin yalniz parlak parcasini aliyordu (780-820: traktor+
                            # romork -> yalniz traktor, IoU 0.95 -> 0.59, sonra suruklenme).
                            kut, msk = ku3[q], (ms3[q] if getattr(A, "CIFT_MASKE", True) else None)
                            break
                mtam = None
                if msk is not None:
                    mtam = cv2.resize(msk.astype(np.uint8), (W, H),
                                      interpolation=cv2.INTER_NEAREST) > 0
                b = self.tk.baslat(pil, tuple(int(v/s_) for v in kut), maske=mtam)
                if b is not None:
                    self.kilit_kutu = tuple(int(v) for v in b)
                    msam_tani = self.tk.son_maske(img.shape[:2])   # TANI: kilit karesi maskesi
                    cos = self.maske_skor(img, msam_tani)
                    self._tani["den_cos"] = round(float(cos), 4)   # kilit ani cipa kosinusu
                    if cos < A.CIPA_COS:      # SAM tuttugu sey referansa hic benzemiyor
                        self.kilit_kutu = None; self.pes_say = 0
                    else:
                        self.mod = "TAKIP"; kilitlendi = True
                        self.kilit_kenar = self._kenarda(self.kilit_kutu, W, H)
                        self.kenar_yapildi = False
                        self.duz = A.DUZELT; self.duz_uyum = 0; self.duz_ornek = 0
                        self.duz_onceki = None; self.red_say = 0
                        self.den_say = 0; self.den_red = 0; self.den_bos = 0
                        self.den_onceki = None; self.denet = 0
                        self.alan_onceki = alan(self.kilit_kutu)
                        self.tepe_alan = self.alan_onceki
                        self.kucul = 0; self.cikis_izi = False; self.sup_bekle = False
                        self.kilit_yas = 0; self.onay_gordu = False; self.cos_red = 0
                        self.kap_bek = None; self.kap_zorla = 0
                        self.isin_kalan = int(getattr(A, "ISINMA_N", 0))
                        self.isin_uyum = 0

        # ── COZUNURLUK KADEMESI: ust uste ARTIR_N arama karesinde kilit yoksa yukselt ──
        if self.mod == "ARAMA" and ara_kos:
            self.kademe_bos += 1
            if self.kademe_bos >= A.ARTIR_N and self.kademe < len(A.ARTIR_PX)-1:
                self.kademe += 1; self.kademe_bos = 0
                self.pes_say = 0        # skorlar kademeler arasi kiyaslanamaz
        elif self.mod == "TAKIP":
            self.kademe_bos = 0

        # ══ TAKIP ══ (kilit karesinde atlanir — o kare gonderimi kilit kutusudur)
        takip_kos = (self.mod == "TAKIP" and not kilitlendi)
        if takip_kos:
            b, mk = self.tk.adim2(pil)
            msam = None
            if mk is not None:
                msam = np.asarray(mk) > 0
                if msam.shape != img.shape[:2]:
                    msam = cv2.resize(msam.astype(np.uint8),
                                      (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST) > 0
                msam_tani = msam                       # TANI: bu karenin takip maskesi
            if b is None:
                # SAM kaybetti: cikis izi + yeterli yas + onay varsa aralik BITTI
                # (benzer nesneye sicrama kapanir); yoksa kilit birakilir.
                if (R.kural["BITTI"] and self.cikis_izi
                        and self.kilit_yas >= A.BITTI_YAS and self.onay_gordu):
                    self.mod = "BITTI"
                self.kilit_kutu = None
                if self.mod != "BITTI":
                    self.kilit_birak("sam_kayip")
            else:
                self.kilit_kutu = tuple(int(v) for v in b)
                self.kilit_yas += 1
                al = alan(self.kilit_kutu)
                self.tepe_alan = max(self.tepe_alan, al)
                kn = self._kenarda(self.kilit_kutu, W, H)
                # ── BEDAVA BEKCILER ──
                if (self.alan_onceki and self.duz == 0 and self.denet == 0 and not kn):
                    orn = max(al/self.alan_onceki, self.alan_onceki/al)
                    if orn > A.BEKCI_ORAN and not self.sup_bekle:
                        self.sup_bekle = True          # ani boyut sicramasi -> denetim iste
                if kn and self.alan_onceki and al < self.alan_onceki:
                    self.kucul += 1
                    if self.kucul >= A.KUCUL_N and not self.cikis_izi:
                        self.cikis_izi = True
                elif not kn:
                    self.kucul = 0
                    if self.cikis_izi and al > 0.5*self.tepe_alan:
                        self.cikis_izi = False         # kenardan geri girdi, iz sil
                if self.cikis_izi and al < A.TEPE_ORAN*self.tepe_alan and kn:
                    if (R.kural["BITTI"] and self.kilit_yas >= A.BITTI_YAS
                            and self.onay_gordu):
                        self.mod = "BITTI"
                    else:
                        self.kilit_birak("cikis")
                    self.kilit_kutu = None
                if self.kilit_kutu is not None:
                    self.alan_onceki = al

                # ── UCUZ DOGRULAMA: SAM maskesi hala referansa benziyor mu (tek gomme) ──
                if (self.mod == "TAKIP" and self.kilit_kutu is not None
                        and A.DEN_HER > 0 and msam is not None
                        and self.kilit_yas % A.DEN_HER == 0):
                    cs = self.maske_skor(img, msam)
                    self._tani["den_cos"] = round(float(cs), 4)      # TANI
                    if cs < A.DEN_COS:
                        self.cos_red += 1
                        if self.cos_red >= A.DEN_COS_RED:
                            self.kilit_birak("ucuz_dogrulama")
                    else:
                        self.cos_red = 0; self.onay_gordu = True

                # ── DUZELTME PENCERESI: kilit sonrasi model SAM'i oylar ──
                if self.mod == "TAKIP" and self.duz > 0:
                    hedef = max(1, A.DUZELT//A.DUZ_ADIM)
                    birakti = False
                    if duz_kos:
                        self.duz_ornek += 1
                        # onay = ilk ONAY_K adaydan biriyle IoU YA DA merkez uyumu + benzer boyut
                        onayli = self._onay(self.kilit_kutu, sira, ku, s_)
                        if onayli:
                            self.duz_uyum += 1; self.onay_gordu = True
                        kararli = (model_kutu is not None and self.duz_onceki is not None
                                   and max(alan(model_kutu)/alan(self.duz_onceki),
                                           alan(self.duz_onceki)/alan(model_kutu))
                                   <= A.KARAR_ORAN)
                        self.duz_onceki = model_kutu
                        self.red_say = 0 if onayli else (self.red_say+1 if kararli else 1)
                        # SAM SAVUNMASI: model SAM'in kendi benzerligini gecemiyorsa red sayilmaz
                        if (self.red_say >= A.ERKEN_N and model_kutu is not None
                                and msam is not None):
                            ss = self.maske_skor(img, msam)
                            if not (float(sk[sira[0]]) > ss + A.SAM_MARJ):
                                self.red_say = 0
                        # TEK KILIT: pencerede hic onay yok + ust uste red -> kilit atilir
                        if (A.ERKEN_N > 0 and self.red_say >= A.ERKEN_N
                                and self.duz_uyum == 0):
                            self.kilit_birak("duzeltme_red"); birakti = True
                    if not birakti:
                        self.duz -= 1
                        if duz_kos and self.duz_ornek >= hedef:
                            if self.duz_uyum > 0:      # SERBEST: pencere onayla kapandi
                                self.duz = 0; self.duz_ornek = 0
                                self.duz_onceki = None; self.red_say = 0
                            else:                      # pencere boyunca hic onay yok
                                self.kilit_birak("duzeltme_onaysiz")

                # ── PERIYODIK DENETIM (SERBEST donemde model tek seferlik) ──
                if self.mod == "TAKIP" and den_kos:
                    self.sup_bekle = False
                    if model_kutu is None:
                        self.den_bos += 1; self.den_red = 0; self.den_onceki = None
                        if self.den_bos >= A.DEN_BOS:
                            if (R.kural["BITTI"] and self.cikis_izi
                                    and self.kilit_yas >= A.BITTI_YAS and self.onay_gordu):
                                self.mod = "BITTI"; self.kilit_kutu = None
                                self.den_bos = 0; self.den_red = 0
                            else:
                                self.kilit_birak("denetim_bos")
                    else:
                        self.den_bos = 0
                        # merkez onayi BOYUT sartli (dev tarla dersi); ilk ONAY_K aday sayilir
                        if self._onay(self.kilit_kutu, sira, ku, s_):
                            self.den_red = 0; self.onay_gordu = True
                        else:
                            krl = (self.den_onceki is not None and
                                   max(alan(model_kutu)/alan(self.den_onceki),
                                       alan(self.den_onceki)/alan(model_kutu)) <= A.KARAR_ORAN)
                            s_sam = self.maske_skor(img, msam) if msam is not None else -9.0
                            s_mod = float(sk[sira[0]]) if sira else -9.0
                            gecti = s_mod > s_sam + A.SAM_MARJ    # SAM savunmasi
                            self.den_red = (self.den_red+1 if krl else 1) if gecti else 0
                            if self.den_red >= A.DEN_RED:   # TEK KILIT: denetim reddi
                                self.kilit_birak("denetim_red")
                        self.den_onceki = model_kutu

                # ── ISINMA (varsayilan kapali): kilit sonrasi ilk ISINMA_N model karesinde
                # model en az ISINMA_M kez kilidi ONAYLAMALI; yoksa kilit atilir. ──
                if (self.mod == "TAKIP" and getattr(A, "ISINMA_N", 0) > 0
                        and self.isin_kalan > 0 and kos_bu_kare):
                    self.isin_kalan -= 1
                    if self.kilit_kutu is not None and self._onay(self.kilit_kutu, sira, ku, s_):
                        self.isin_uyum += 1
                    if self.isin_kalan == 0 and self.isin_uyum < getattr(A, "ISINMA_M", 1):
                        self.kilit_birak("isinma")

                # ── KAPSAM UYUMU: AYNI YERDE boyut uyusmazligi -> IKI OLCUMLE onay ──
                if (self.mod == "TAKIP" and self.kilit_kutu is not None
                        and model_kutu is not None
                        and (duz_kos or den_kos or self.denet > 0 or self.kap_zorla)
                        and sira):
                    sm = float(sk[sira[0]])
                    am, ak = alan(model_kutu), alan(self.kilit_kutu)
                    orn = max(am/ak, ak/am)
                    yon = "buyu" if am > ak else "kucul"
                    uyusmaz = (merkez_uyum(model_kutu, self.kilit_kutu)
                               and orn >= A.KAP_ORAN and sm >= A.KAP_SKOR)
                    if uyusmaz and self.kap_bek is not None:
                        one, yon0 = self.kap_bek
                        tamam = (yon == yon0 and iou(model_kutu, one) >= A.KAP_ES)
                        # ASIMETRI: BUYUTME guvenli (parcadan butune), KUCULTME yikici —
                        # kucultmek icin model SAM'in KENDI benzerligini belirgin gecmeli.
                        if tamam and yon == "kucul":
                            ssam = self.maske_skor(img, msam) if msam is not None else -9.0
                            if sm < ssam + A.KUCUL_MARJ and ssam >= A.DEN_COS:
                                tamam = False
                        if tamam:
                            b3 = self.tk.baslat(pil, model_kutu)
                            if b3 is not None:
                                self.kilit_kutu = tuple(int(v) for v in b3)
                                self.alan_onceki = alan(self.kilit_kutu)
                                self.tepe_alan = max(self.tepe_alan, self.alan_onceki)
                                self.onay_gordu = True
                                self.red_say = 0; self.den_red = 0; self.cos_red = 0
                        self.kap_bek = None; self.kap_zorla = 0
                    elif uyusmaz:
                        self.kap_bek = (model_kutu, yon)   # ilk olcum: karar YOK
                        self.kap_zorla = 1                 # SONRAKI KARE modeli zorla kos
                    else:
                        self.kap_bek = None; self.kap_zorla = 0

                # ── KENAR-GIRIS KURALI: kenarda dogan kilit iceri girince dogrulanir ──
                if (self.mod == "TAKIP" and self.kilit_kutu is not None
                        and self.duz == 0 and self.denet == 0 and self.kilit_kenar
                        and not self.kenar_yapildi
                        and not (self._kenarda(self.kilit_kutu, W, H) & self.kilit_kenar)):
                    self.denet = A.YEN_KARE; self.uyum_say = 0
                    self.kenar_yapildi = True
                elif self.mod == "TAKIP" and self.denet > 0:
                    self.denet -= 1
                    uy = iou(model_kutu, self.kilit_kutu) if model_kutu else 0.0
                    # SAM modelin gosterdiginden BUYUKSE ve ayni nesnedeyse ONAY
                    if uy >= A.UYUM or (model_kutu
                                        and merkez_uyum(model_kutu, self.kilit_kutu)
                                        and alan(self.kilit_kutu) >= alan(model_kutu)):
                        self.uyum_say += 1
                    if self.denet == 0 and self.uyum_say < A.UYUM_MIN:
                        self.kilit_birak("kenar_giris")    # TEK KILIT: giris denetimi red

        # ── TANI (yalniz kayit; gonderim karari ETKILENMEZ) ──
        if kilitlendi or takip_kos:
            self._tani_kur(msam_tani, W, H)

        # ══ GONDERIM ══
        if self.mod == "TAKIP" and self.kilit_kutu:
            self.kaynak = "kilit"
            return self.kilit_kutu
        # kilitsiz (veya bu karede birakilmis): model tepe adayi kapilari gecerse gonder
        if getattr(A, "SADECE_TAKIP", False):          # termal2 politikasi: kilitsiz kutu gitmez
            return None
        if sira and sk is not None and model_kutu is not None:
            s1 = float(sk[sira[0]])
            if s1 >= R.kural["GONDER"]:
                g2 = -9.0
                for q in sira[1:]:
                    if not merkez_uyum(ku[q], ku[sira[0]]):
                        g2 = float(sk[q]); break
                if g2 <= -8:      # RAKIPSIZ KARE: yuksek mutlak skor sarti
                    if s1 >= R.kural["RAKIPSIZ"]:
                        self.kaynak = "model"; return model_kutu
                    return None
                if s1 - g2 >= R.kural["GONDER_MARJ"]:
                    self.kaynak = "model"; return model_kutu
        return None
