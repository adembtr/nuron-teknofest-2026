#!/usr/bin/env python3
"""GOREV 3 · TERMAL v3 — yollar ve sabitler (RGB Z3 motorunun termal portu).

Kaynak: NURON_DRONE/src/task3_reference/rgb2/ayarlar.py (Z3, 2026-09-02) — birebir kopya,
ustune TERMAL eklemeleri:
  * KIP_ZORLA = "termal": termal oturumda KARE hep gridir; karar kurallari referansin
    renkli/gri olusuna degil, oturuma gore secilir. (Z3'te kip referanstan cikiyordu.)
  * BANKA: referansin bankaya giren halleri — "renkli" | "gri" | "renkli+gri" (ikili arama).
  * GOMME_ADI: gomme omurgasi — radio_h (C-RADIOv3-H, Z3'un secimi) | radio_l | radio_b |
    webssl_300m_l (termal2) | d3_l | d3_sp
  * Tarama icin her kritik deger ORTAM DEGISKENIYLE ezilebilir (K_<AD>=deger).

K3b (2026-09-18): KAPI_SONRA=True + PEN_S1_MIN=0.60 varsayilan — bir kilit birakildiktan sonraki yeniden kilitlerde
  kazanan skoru < 0.60 ise kilit yok (bos pencerede gonderim %79 -> %38.5; gercek araliklarda hibrit A ile birebir).
  Hibrit A'ya donmek icin KAPI_SONRA=0.

Boru hatti: FastSAM-x -> ayrik bolme -> TEMAS BUYUMESI -> gomme -> yeniden siralama
         -> TEK SAM kilidi (skor+marj, ust uste KILIT_N) -> DAM4SAM takip + bekciler.
"""
import os
from src.common.config import ROOT

_M = os.path.join(ROOT, "models")


def _env(ad, varsayilan):
    v = os.environ.get(ad)
    if v is None:
        return varsayilan
    if isinstance(varsayilan, bool):
        return v.strip().lower() in ("1", "true", "evet", "acik")
    if isinstance(varsayilan, int):
        return int(float(v))
    if isinstance(varsayilan, float):
        return float(v)
    return v


FASTSAM_CKPT = os.path.join(_M, "rgb_fastsam_x.pt")          # modaliteden bagimsiz
GOMME_YOLU   = os.path.join(_M, "webssl_dino300m_light2b")    # WebSSL (termal2 omurgasi)
RADIO_REPO   = os.path.join(_M, "radio_repo")                 # C-RADIOv3-H yerel hub reposu
RADIO_CKPT   = os.path.join(_M, "rgb_radio_v3h.pth.tar")
SAM2_BP_CKPT = os.path.join(_M, "sam2.1_hiera_base_plus.pt")
SAM2_L_CKPT  = os.path.join(_M, "sam2.1_hiera_large.pt")
DAM4SAM_REPO = os.path.join(ROOT, "third_party", "DAM4SAM")
HQSAM_CKPT   = os.path.join(_M, "sam_hq_vit_l.pth")
HQSAM_TYPE   = "vit_l"

# ── TERMAL v3 secimleri (env ile ezilebilir) ───────────────────
GOMME_ADI  = _env("GOMME_ADI", "radio_h")       # radio_h | radio_l | radio_b | webssl_300m_l | d3_l | d3_sp
BANKA      = _env("BANKA", "renkli+gri")        # renkli | gri | renkli+gri
KIP_ZORLA  = _env("KIP", "termal")              # termal | gri | rgb | "" (Z3: referansa gore)
TAKIPCI    = _env("TAKIPCI", "sam21pp-L")       # sam21pp-L (Z3) | sam21pp-B (termal2)
ARTIR_PX   = [int(x) for x in _env("ARTIR_PX", "512,768").split(",")]
GOMME_YARI = _env("GOMME_YARI", True)           # WebSSL/DINOv3 icin bf16
SONDA_K    = _env("SONDA_K", 8)                 # her arama karesinde kaydedilen aday sayisi (0 = kapali)
# ── TERMAL KILIT (sonda_radio_rg olcumuyle secildi, 2026-09-04) ─────────────
KILIT_MOD  = _env("KILIT_MOD", "pencere")       # pencere (termal) | z3 (skor+marj, N ardisik)
PEN_W      = _env("PEN_W", 3)                   # pencere: son kac arama karesi
PEN_K      = _env("PEN_K", 3)                   # her kareden ilk kac aday oylanir (ILK3 %78 vs ILK1 %45)
PEN_M      = _env("PEN_M", 3)                   # kazanan kume en az kac karede gorulmeli
PEN_IOU    = _env("PEN_IOU", 0.5)               # kareler arasi kume zinciri IoU'su
PEN_ORAN   = _env("PEN_ORAN", 1.6)              # kazanan kume puani / 2. kume puani
PEN_AG     = _env("PEN_AG", "skor")             # kume puani: skor toplami | sira agirligi
ALAN_MAKS  = _env("ALAN_MAKS", 0.65)            # kare alaninin bu oranindan buyuk aday elenir (GT maks %62)
ONAY_K     = _env("ONAY_K", 3)                  # kilit sonrasi onaylarda ilk kac aday sayilir
KILIT_BIRLESIM = _env("KILIT_BIRLESIM", False)  # kilit maskesi = kazanan kumenin bu karedeki uyelerinin birlesimi
IKILI_BEKLE = _env("IKILI_BEKLE", 3)            # ikili karar: iki atesin en fazla kac arama karesi arayla gelmesi
IKILI_IOU   = _env("IKILI_IOU", 0.5)            # ikili karar: iki modelin ates kutularinin uyusma IoU'su
HIBRIT_ZORLA = _env("HIBRIT_ZORLA", "")        # hibrit: "" (referansa gore) | t2 | t3 (zorla)
HIBRIT_PAYLAS = _env("HIBRIT_PAYLAS", True)     # hibrit: FastSAM + DAM4SAM (sam21pp-B) iki hat arasinda paylasilir (bellek)
SADECE_TAKIP = _env("SADECE_TAKIP", False)      # True: kilit yokken model kutusu HIC gonderilmez

# ── YANLIS-POZITIF KAPILARI (2026-09-17) ───────────────────────────────────
# Nesne kadrajda YOKKEN de kilit kuruluyor (bos araliklarda ~%79 kare kutu gitti).
# Asagidaki kapilarin HEPSI VARSAYILAN KAPALI: kapali degerlerle motor BIREBIR
# eski davranisi kosar. Her biri tek basina env ile acilir (bkz. FP_KAPILAR.md).
# (a) KILIT ANI — pencereli oylama kazananina ek sartlar:
PEN_S1_MIN   = _env("PEN_S1_MIN", 0.60)    # K3b varsayilani (2026-09-18); 0.0 = kapali     # kazanan uyenin BU KAREDEKI skoru >= (0=kapali)
PEN_PUAN_MIN = _env("PEN_PUAN_MIN", 0.0)   # kazanan kumenin puan toplami >= (0=kapali)
PEN_MARJ_MIN = _env("PEN_MARJ_MIN", 0.0)   # kilit karesinde s1-s2 (sonda marj tanimi) >= (0=kapali)
PEN_Z_MIN    = _env("PEN_Z_MIN", 0.0)      # kazanan skorun ilk-K aday icindeki z-degeri >= (0=kapali)
PEN_TAM      = _env("PEN_TAM", False)      # kazanan kume PEN_W karenin HEPSINDE gorulmus olmali
PEN_KARARLI  = _env("PEN_KARARLI", 0.0)    # kume uyelerinin ardisik kareler arasi IoU ort >= (0=kapali)
KAPI_SONRA   = _env("KAPI_SONRA", True)    # K3b varsayilani (2026-09-18): kapilar yalniz YENIDEN kilitlerde; False = hibrit A   # True: kilit-ani kapilari yalniz aralikta en az bir kilit birakildiktan sonra uygulanir
PEN_KENAR    = _env("PEN_KENAR", False)    # kazanan kutu kare kenarina degiyorsa kilit YOK
# (b) KILIT SONRASI — isinma dogrulamasi:
ISINMA_N     = _env("ISINMA_N", 0)         # kilitten sonraki ilk N model karesi izlenir (0=kapali)
ISINMA_M     = _env("ISINMA_M", 1)         # bu N karede en az M kez _onay uyusmasi sart
ISINMA_HER   = _env("ISINMA_HER", False)   # isinma boyunca modeli HER karede kostur (yavas ama hizli karar)
# (c) SOGUMA — birakilan kutuya yeniden kilitlenme yasagi:
SOGUMA_N     = _env("SOGUMA_N", 0)         # birakilan kutu kac KARE boyunca yasakli (0=kapali)
SOGUMA_IOU   = _env("SOGUMA_IOU", 0.5)     # yasak kutuyla bu IoU'yu gecen kazanan kilit alamaz

# ── kare / bolme ───────────────────────────────────────────────
MAXSIDE   = 1024
FASTSAM_CONF = _env("FASTSAM_CONF", 0.10)
FASTSAM_IOU  = 0.85
MIN_ORAN  = _env("MIN_ORAN", 0.00025)
AYRIK     = True

# ── temas grafigi (yalniz arama.py / eski yol) ─────────────────
IC_ESIK   = 0.75
KOM_UST   = 5

# ── gomme cozunurlukleri ───────────────────────────────────────
TUR0_TAVAN  = 224
HAM_TAVAN   = _env("HAM_TAVAN", 336)    # referans BANKASI gomme cozunurlugu (Z3: 336)
TUR0_PX     = _env("TUR0_PX", 224)      # aday gomme cozunurlugu
GEN         = _env("GEN", 8)
BUYUME      = 0.03
SABIR       = 2
MAKS_TUR    = 6

DEVICE  = "cuda"

# ══ KURALLAR ═══════════════════════════════════════════════════
# rgb / gri: Z3 degerleri (RGB oturumu, 1342 etiketli kare ile secildi) — DOKUNULMADI.
# termal  : TERMAL OTURUM icin baslangic = gri kurali; sonda verisiyle yeniden ayarlanir.
KURAL = {
    "rgb": dict(GONDER=0.66, GONDER_MARJ=0.05, KILIT_SKOR=0.66, KILIT_MARJ=0.06,
                KILIT_TEK=0.80, KILIT_N=2, RAKIPSIZ=0.74, BITTI=True,
                TEMAS_ART=0.020, W_RENK=0.10),
    "gri": dict(GONDER=0.78, GONDER_MARJ=0.05, KILIT_SKOR=0.62, KILIT_MARJ=0.06,
                KILIT_TEK=0.90, KILIT_N=3, RAKIPSIZ=0.78, BITTI=False,
                TEMAS_ART=0.005, W_RENK=0.00),
    "termal": dict(GONDER=0.78, GONDER_MARJ=0.05, KILIT_SKOR=0.62, KILIT_MARJ=0.06,
                   KILIT_TEK=0.90, KILIT_N=3, RAKIPSIZ=0.78, BITTI=False,
                   TEMAS_ART=0.005, W_RENK=0.00),
}
for _k, _v in list(KURAL["termal"].items()):          # K_KILIT_SKOR=0.55 gibi ezmeler
    KURAL["termal"][_k] = _env("K_" + _k, _v)

KILIT_N   = 2
# ── GOMME SKOR OLCEGI (sonda olcumleri, 2026-09-04): aday kosinuslerinin ort / sd ──
# Z3 esikleri (SKOR_MIN 0.35, DEN_COS 0.30, CIPA_COS 0.25) RADIO-H olcegine gore ayarlandi.
# Baska gomme secilirse ayni z-degerine denk gelen esik turetilir (env verilmezse).
# Olcegi bilinmeyen model: SONDA modunda zararsiz dusuk esikler (once sonda kos, olcegi buraya yaz).
GOMME_OLCEK = {
    "radio_h":       (0.614, 0.121),
    "webssl_300m_l": (0.465, 0.088),
    "d3_l":          (0.330, 0.113),   # sonda_d3l_gri (SKOR_MIN 0.15 ile)
}
def _z_esik(radio_esik, varsayilan_bilinmeyen):
    ma, sa = GOMME_OLCEK["radio_h"]
    adlar = [a for a in GOMME_ADI.split("+")]
    if all(a in GOMME_OLCEK for a in adlar):
        # fuzyonda skorlar RADIO olcegine geri haritalanir -> RADIO esigi gecerli
        m, sd = GOMME_OLCEK[adlar[0]] if len(adlar) == 1 else (ma, sa)
        return round(m + sd * (radio_esik - ma) / sa, 3)
    return varsayilan_bilinmeyen
CIPA_COS  = _env("CIPA_COS", _z_esik(0.25, 0.10))
SKOR_MIN  = _env("SKOR_MIN", _z_esik(0.35, 0.15))
EB_TOL    = _env("EB_TOL", 2.5)
ALAN_MIN  = _env("ALAN_MIN", 500)
UST_ADAY  = 10
ARA_PX    = 512
ARTIR_N   = _env("ARTIR_N", 60)
ARA_ADIM  = _env("ARA_ADIM", 2)
# temas buyumesi
TEMAS_D   = 4
TEMAS_KOM = 6
TEMAS_PX  = 5
BIR_TOHUM = 3
BIR_MAKS  = 0.35
IKILI     = True
IKILI_UST = 4
DOLU_TOL  = _env("DOLU_TOL", 3.0)
# bolme
BOL       = True
BOL_MIN   = 0.02
# yeniden siralama
RER_K     = 7
W_SEKIL   = _env("W_SEKIL", 0.10)
# kilit sonrasi
DEN_HER     = 2
DENETIM_N   = _env("DENETIM_N", 12)
DUZELT      = _env("DUZELT", 10)
DUZ_ADIM    = 2
UYUM        = 0.50
OY_ORAN     = 4.0
KARAR_ORAN  = 1.5
ERKEN_N     = _env("ERKEN_N", 2)
SAM_MARJ    = _env("SAM_MARJ", 0.05)
DEN_RED     = 2
DEN_BOS     = 3
YEN_KARE    = 3
UYUM_MIN    = 2
DEN_COS     = _env("DEN_COS", _z_esik(0.30, 0.10))
DEN_COS_RED = 2
KAP_ORAN    = 1.30
KAP_SKOR    = _env("KAP_SKOR", 0.65)
KAP_ES      = 0.50
KUCUL_MARJ  = 0.05
CIFT_COZ    = _env("CIFT_COZ", True)
CIFT_ES     = 0.50
CIFT_MASKE  = _env("CIFT_MASKE", True)   # takasta SAM maskeyle kurulur (kutu istemi parcayi aliyordu)
# bekciler
BEKCI_ORAN = 1.8
KUCUL_N    = 3
TEPE_ORAN  = 0.05
BITTI_YAS  = 15
KAPAT      = 0.15
KENAR_PX   = 3
# eski uyumluluk
ESIK    = 0.60
ARDISIK = 2


def ozet():
    """Kosunun ayarlarini tek sozlukte dondur (kayit icin)."""
    return dict(GOMME_ADI=GOMME_ADI, BANKA=BANKA, KIP=KIP_ZORLA, TAKIPCI=TAKIPCI,
                ARTIR_PX=ARTIR_PX, ARTIR_N=ARTIR_N, HAM_TAVAN=HAM_TAVAN, TUR0_PX=TUR0_PX,
                KURAL=KURAL.get(KIP_ZORLA or "termal"), CIPA_COS=CIPA_COS, SKOR_MIN=SKOR_MIN,
                EB_TOL=EB_TOL, ALAN_MIN=ALAN_MIN, DOLU_TOL=DOLU_TOL, W_SEKIL=W_SEKIL,
                DENETIM_N=DENETIM_N, DEN_COS=DEN_COS, SADECE_TAKIP=SADECE_TAKIP,
                FASTSAM_CONF=FASTSAM_CONF, MIN_ORAN=MIN_ORAN, GEN=GEN, SONDA_K=SONDA_K,
                KILIT_MOD=KILIT_MOD, PEN=(PEN_W, PEN_K, PEN_M, PEN_IOU, PEN_ORAN, PEN_AG),
                ALAN_MAKS=ALAN_MAKS, ONAY_K=ONAY_K, ARA_ADIM=ARA_ADIM,
                KILIT_BIRLESIM=KILIT_BIRLESIM, CIFT_MASKE=CIFT_MASKE,
                FP_KAPI=dict(PEN_S1_MIN=PEN_S1_MIN, PEN_PUAN_MIN=PEN_PUAN_MIN,
                             PEN_MARJ_MIN=PEN_MARJ_MIN, PEN_Z_MIN=PEN_Z_MIN,
                             PEN_TAM=PEN_TAM, PEN_KARARLI=PEN_KARARLI, PEN_KENAR=PEN_KENAR, KAPI_SONRA=KAPI_SONRA,
                             ISINMA_N=ISINMA_N, ISINMA_M=ISINMA_M, ISINMA_HER=ISINMA_HER,
                             SOGUMA_N=SOGUMA_N, SOGUMA_IOU=SOGUMA_IOU))
