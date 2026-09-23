#!/usr/bin/env python3
"""GOREV 3 · RGB v2 — Z3 MOTORU (2026-09-02, M16 tabanli).

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
    hep TEK model kosusu vardir. M20 (yalniz RGB, kademe 0): 512'nin tepe skoru IKI_ESIK'in
    altindaysa AYNI KAREDE 768 de kosulur, tepe skoru yuksek olan kullanilir; 768'in tepesi
    512'nin tepesinin ICINDEKI kucuk parcaysa reddedilir (parca korumasi).

 9) KILIT SONRASI UC BEKCI (Z3): (a) DUZELTME PENCERESI — kilitten sonra DUZELT kare
    boyunca model SAM'i oylar, hic onay yoksa kilit atilir; (b) PERIYODIK DENETIM —
    her DENETIM_N karede model kosup SAM'le karsilastirilir, ust uste DEN_RED kararli
    red kilidi atar (SAM kendini maske benzerligiyle savunabilir); (c) KENAR-GIRIS —
    kenarda dogan kilit iceri girince kisa bir dogrulama penceresi kosar.
 6) MODALITE. Referans termalse (R=G=B) adaylar da griye cevrilerek gomulur ve tum
    marj/kazanc esikleri ~1/3'e iner (termalde kosinus dagilimi sikisik).
"""
import numpy as np, cv2
from PIL import Image

from src.task3_reference.rgb2 import ayarlar as A, bolucu as B
from src.task3_reference.rgb2.takip import Takip

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


def eb_red(e, R):
    """en/boy kapisi: True -> ELE. M19: RGB referansta ASIMETRIK (bkz. ayarlar.EB_ALT); termalde eski kural."""
    if e is None:
        return True
    alt = A.EB_ALT if (not R.gri or A.M19_GRI) else A.EB_TOL   # M19_GRI: gri'de de asimetrik
    return e > R.eb*A.EB_TOL or e < R.eb/alt


def termal_mi(crop_bgr, maske):
    """Referans TERMAL mi: termalde R=G=B (olculdu: kanal farki 0.0; en soluk renkli
    referansta bile 6.4). Elle isaretleme gerekmez."""
    kp = crop_bgr[:, :, :3].astype(np.int16)
    return float((kp.max(2)-kp.min(2))[maske].mean()) < 1.0


# ─────────────────────────── referans kaydi ───────────────────────────
class Referans:
    """Bir nesnenin bankasi + ondan turetilen olcutler + modalite kurali."""

    def __init__(self, crop_bgr, maske, gomme):
        self.gri = termal_mi(crop_bgr, maske)
        self.kural = dict(A.KURAL["gri" if self.gri else "rgb"])
        gomme.tavan = A.HAM_TAVAN
        self.banka = _nrm(gomme([self.girdi(c, m) for c, m in donmeler(crop_bgr, maske)]).numpy())
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

    def girdi(self, c, m):
        """modele giren goruntu; referans TERMAL ise aday da griye cevrilir.
        (FastSAM yine RGB karede calisir — sadece GOMMEYE giren goruntu termallesir.)"""
        g = B.girdi(c, m)
        if self.gri:
            g = cv2.cvtColor(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return g


# ─────────────────────────── motor ───────────────────────────
class Motor:
    """Tek referans araligini kare kare surer. `kare(frame_bgr)` -> kutu ya da None."""

    def __init__(self, gomme, takip=None):
        self.gomme = gomme
        self.tk = takip if takip is not None else Takip()
        self.ref = None
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

    def aktif(self, ref):
        self.ref = ref
        self.sifirla()

    def kilit_birak(self):
        """TEK KILIT: supheli kilit DUZELTILMEZ, ATILIR — aramaya donulur.
        (Cozunurluk kademesi KORUNUR: pencerenin nesne boyu degismedi.)"""
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

    # ── gomme yardimcilari ───────────────────────────────────
    def _gom(self, crops):
        self.gomme.tavan = A.TUR0_PX
        V = np.concatenate([_nrm(self.gomme(crops[z:z+64]).numpy())
                            for z in range(0, len(crops), 64)])
        return V

    def _skorla(self, crops):
        return (self._gom(crops) @ self.ref.banka.T).max(1)

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
    def model_kos(self, img, ucuz, px=None):
        """FastSAM -> on suzgec -> gomme -> TEMAS BUYUMESI -> BOLME -> yeniden siralama.
        -> (sira, sk, ms, ku)  [sira: skor sirali aday indeksleri]
        px: FastSAM cozunurlugu (kademe); yoksa ucuz->A.ARA_PX, degilse tam (MAXSIDE)."""
        R = self.ref
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
            e0, _ = en_boy(m)
            if eb_red(e0, R):
                continue
            c2, m2 = B.kirp_gen(img, m, k, A.GEN)
            cr.append(R.girdi(c2, m2)); ok.append(q)
        sk = np.full(len(ms), -9.0)
        if cr:
            sk[np.asarray(ok)] = self._skorla(cr)

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
                        if eb_red(e3, R):
                            continue
                        dl = float(bb[kk[1]:kk[3], kk[0]:kk[2]].mean())
                        if max(dl/R.dolu, R.dolu/dl) > A.DOLU_TOL:
                            continue      # referanstan cok farkli DOLULUK = zemin birlesimi
                        dene.append(c); dm.append(bb)
                    if not dene:
                        break
                    ss = self._skorla([R.girdi(*B.kirp_gen(img, b2, kutu_of(b2), A.GEN))
                                       for b2 in dm])
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
                                if eb_red(e4, R):
                                    continue
                                ci.append(1); cm.append(bb)
                        if ci:
                            ss2 = self._skorla([R.girdi(*B.kirp_gen(img, b2, kutu_of(b2), A.GEN))
                                                for b2 in cm])
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
                        if q2 == q or sk[q2] < A.SKOR_MIN or ku[q2] is None:
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
                                    for m in yeni2])
                ms = list(ms) + yeni2
                ku = list(ku) + [kutu_of(m) for m in yeni2]
                sk = np.concatenate([sk, ss3])
                bol_kume = set(range(ilk, len(ms)))

        # ── YENIDEN SIRALAMA: kosinus + RENK + SEKIL (bolme yarimlari bonus ALMAZ) ──
        aday = [int(q) for q in np.argsort(-sk) if sk[q] >= A.SKOR_MIN]
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
            return None
        R = self.ref
        H, W = frame_bgr.shape[:2]
        s_ = A.MAXSIDE/max(H, W) if max(H, W) > A.MAXSIDE else 1.0
        img = cv2.resize(frame_bgr, (int(W*s_), int(H*s_)),
                         interpolation=cv2.INTER_AREA) if s_ < 1 else frame_bgr
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

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

        sira, sk, ms, ku = [], None, [], []
        model_kutu = None
        if ara_kos or self.denet > 0 or duz_kos or den_kos or self.kap_zorla:
            if ara_kos and self.mod == "ARAMA":       # arama: kademe cozunurlugu (ucuz)
                px = A.ARTIR_PX[min(self.kademe, len(A.ARTIR_PX)-1)]
                sira, sk, ms, ku = self.model_kos(img, True, px)
                # M20 IKI KADEME AYNI KAREDE (yalniz RGB, kademe 0): 512 tepesi kilit esiginin altindaysa
                # ayni karede ARA_PX2; PARCA KORUMASI ile (bkz. ayarlar.IKI_KADEME_ARA).
                if A.IKI_KADEME_ARA and not R.gri and self.kademe == 0:
                    _s1a = float(sk[sira[0]]) if sira else -9.0
                    if _s1a < A.IKI_ESIK:
                        _r2 = self.model_kos(img, True, A.ARA_PX2)
                        _s1b = float(_r2[1][_r2[0][0]]) if _r2[0] else -9.0
                        _k1 = ku[sira[0]] if sira else None
                        _k2 = _r2[3][_r2[0][0]] if _r2[0] else None
                        _parca = (_k1 is not None and _k2 is not None and merkez_uyum(_k2, _k1)
                                  and alan(_k2) < A.IKI_PARCA*alan(_k1))
                        if _s1b > _s1a and not _parca:
                            sira, sk, ms, ku = _r2
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
            # RAKIPSIZ KARE: marj olculemez -> yuksek MUTLAK skor sarti
            esik = R.kural["RAKIPSIZ"] if marj > 8 else R.kural["KILIT_SKOR"]
            uygun = bool(sira) and s1 >= esik and marj >= R.kural["KILIT_MARJ"]
            self.pes_say = self.pes_say+1 if uygun else 0
            if uygun and (self.pes_say >= R.kural["KILIT_N"] or s1 >= R.kural["KILIT_TEK"]):
                kut, msk = ku[sira[0]], ms[sira[0]]
                # CIFT COZUNURLUK: kilit aninda bir kez tam cozunurluk — ayni nesneyi
                # gosteren daha iyi kutu varsa SAM ONA kurulur (BICIM SARTIYLA; sart
                # olmadan takas nesnenin ICINDEKI parcaya geciyordu, olculdu).
                if A.CIFT_COZ:
                    s3, sk3, ms3, ku3 = self.model_kos(img, False)
                    for q in s3[:3]:
                        if (float(sk3[q]) > s1 and merkez_uyum(ku3[q], kut)
                                and iou(ku3[q], kut) >= A.CIFT_ES):
                            kut, msk = ku3[q], None   # takasta maske YOK: SAM kutuyla kurulur
                            break
                mtam = None
                if msk is not None:
                    mtam = cv2.resize(msk.astype(np.uint8), (W, H),
                                      interpolation=cv2.INTER_NEAREST) > 0
                b = self.tk.baslat(pil, tuple(int(v/s_) for v in kut), maske=mtam)
                if b is not None:
                    self.kilit_kutu = tuple(int(v) for v in b)
                    cos = self.maske_skor(img, self.tk.son_maske(img.shape[:2]))
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

        # ── COZUNURLUK KADEMESI: ust uste ARTIR_N arama karesinde kilit yoksa yukselt ──
        if self.mod == "ARAMA" and ara_kos:
            self.kademe_bos += 1
            if self.kademe_bos >= A.ARTIR_N and self.kademe < len(A.ARTIR_PX)-1:
                self.kademe += 1; self.kademe_bos = 0
                self.pes_say = 0        # skorlar kademeler arasi kiyaslanamaz
        elif self.mod == "TAKIP":
            self.kademe_bos = 0

        # ══ TAKIP ══ (kilit karesinde atlanir — o kare gonderimi kilit kutusudur)
        if self.mod == "TAKIP" and not kilitlendi:
            b, mk = self.tk.adim2(pil)
            msam = None
            if mk is not None:
                msam = np.asarray(mk) > 0
                if msam.shape != img.shape[:2]:
                    msam = cv2.resize(msam.astype(np.uint8),
                                      (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST) > 0
            if b is None:
                # SAM kaybetti: cikis izi + yeterli yas + onay varsa aralik BITTI
                # (benzer nesneye sicrama kapanir); yoksa kilit birakilir.
                if (R.kural["BITTI"] and self.cikis_izi
                        and self.kilit_yas >= A.BITTI_YAS and self.onay_gordu):
                    self.mod = "BITTI"
                self.kilit_kutu = None
                if self.mod != "BITTI":
                    self.kilit_birak()
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
                        self.kilit_birak()
                    self.kilit_kutu = None
                if self.kilit_kutu is not None:
                    self.alan_onceki = al

                # ── UCUZ DOGRULAMA: SAM maskesi hala referansa benziyor mu (tek gomme) ──
                if (self.mod == "TAKIP" and self.kilit_kutu is not None
                        and A.DEN_HER > 0 and msam is not None
                        and self.kilit_yas % A.DEN_HER == 0):
                    cs = self.maske_skor(img, msam)
                    if cs < A.DEN_COS:
                        self.cos_red += 1
                        if self.cos_red >= A.DEN_COS_RED:
                            self.kilit_birak()
                    else:
                        self.cos_red = 0; self.onay_gordu = True

                # ── DUZELTME PENCERESI: kilit sonrasi model SAM'i oylar ──
                if self.mod == "TAKIP" and self.duz > 0:
                    hedef = max(1, A.DUZELT//A.DUZ_ADIM)
                    birakti = False
                    if duz_kos:
                        self.duz_ornek += 1
                        uy = iou(model_kutu, self.kilit_kutu) if model_kutu else 0.0
                        # onay = IoU YA DA merkez uyumu + benzer boyut (SAM tam kutu
                        # tutarken modelin parca tepesi yanlis red uretmesin)
                        onayli = (uy >= A.UYUM or (model_kutu is not None
                                  and merkez_uyum(model_kutu, self.kilit_kutu)
                                  and alan(self.kilit_kutu) <= A.OY_ORAN*alan(model_kutu)
                                  and alan(model_kutu) <= A.OY_ORAN*alan(self.kilit_kutu)))
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
                            self.kilit_birak(); birakti = True
                    if not birakti:
                        self.duz -= 1
                        if duz_kos and self.duz_ornek >= hedef:
                            if self.duz_uyum > 0:      # SERBEST: pencere onayla kapandi
                                self.duz = 0; self.duz_ornek = 0
                                self.duz_onceki = None; self.red_say = 0
                            else:                      # pencere boyunca hic onay yok
                                self.kilit_birak()

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
                                self.kilit_birak()
                    else:
                        self.den_bos = 0
                        uy2 = iou(model_kutu, self.kilit_kutu)
                        # merkez onayi BOYUT sartli: dev SAM kutusu icine dusen model
                        # tepesi onay SAYILMAZ (kadraja girmeden kilitlenen dev tarla dersi)
                        if uy2 >= A.UYUM or (merkez_uyum(model_kutu, self.kilit_kutu)
                                             and alan(self.kilit_kutu) <= A.OY_ORAN*alan(model_kutu)
                                             and alan(model_kutu) <= A.OY_ORAN*alan(self.kilit_kutu)):
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
                                self.kilit_birak()
                        self.den_onceki = model_kutu

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
                            _km = A.KUCUL_MARJ if (R.gri and not A.M19_GRI) else A.KUCUL_MARJ_RGB   # M19(+GRI): kucultme kapali
                            if sm < ssam + _km and ssam >= A.DEN_COS:
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
                        self.kilit_birak()    # TEK KILIT: giris denetimi red

        # ══ GONDERIM ══
        # M19 KARSIT KENAR BEKCISI (yalniz RGB): kutu karenin iki karsit kenarina birden degiyorsa
        # nesne degil zemin/cikis sizintisidir -> gonderme; takipteyse kilidi birak (aramaya don).
        def _karsit(b):
            return (A.KARSIT_KENAR and (not R.gri or A.M19_GRI) and b is not None and
                    ((b[0] <= A.KENAR_PX and b[2] >= W-A.KENAR_PX) or
                     (b[1] <= A.KENAR_PX and b[3] >= H-A.KENAR_PX)))
        if self.mod == "TAKIP" and self.kilit_kutu:
            if _karsit(self.kilit_kutu):
                self.kilit_birak()
                return None
            return self.kilit_kutu
        # kilitsiz (veya bu karede birakilmis): model tepe adayi kapilari gecerse gonder
        if sira and sk is not None and model_kutu is not None:
            s1 = float(sk[sira[0]])
            if s1 >= R.kural["GONDER"]:
                g2 = -9.0
                for q in sira[1:]:
                    if not merkez_uyum(ku[q], ku[sira[0]]):
                        g2 = float(sk[q]); break
                if g2 <= -8:      # RAKIPSIZ KARE: yuksek mutlak skor sarti
                    return (model_kutu if (s1 >= R.kural["RAKIPSIZ"] and not _karsit(model_kutu))
                            else None)
                if s1 - g2 >= R.kural["GONDER_MARJ"] and not _karsit(model_kutu):
                    return model_kutu
        return None
