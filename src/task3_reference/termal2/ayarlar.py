#!/usr/bin/env python3
"""GOREV 3 · TERMAL v2 — yollar ve sabitler.

SECILEN COZUM (2026-08-25, 2026 oturumunda 253 GT kutusu uzerinde olculdu):
    HQ-SAM referans kirpma
    -> referans bankasi: YALNIZ GRI x 4x90 derece donme  (4 vektor)
    -> FastSAM-x + AYRIK bolme + delik kapatma
    -> temas grafigi (en yakin 5, butun-govde yakinligi) + turlu birlesim
    -> WebSSL-300M-light2b CLS gomme (bf16)
    -> gonder 0.24 / cipa 0.45 / 4 ardisik tutarli tespit
    -> DAM4SAM sam21pp-B akis takibi
    -> SADECE TAKIP: SAM cipalanmadan HICBIR kutu gonderilmez

Olculen: 201 dogru / 6 YANLIS / 46 gonderilmedi
         dogruluk %79.4  kesinlik %97.1  ortalama IoU 0.871

Ayrintili gerekce: nuron_referance_termal/YONTEM.md ve ELENENLER.md
"""
import os
from src.common.config import ROOT

_M = os.path.join(ROOT, "models")

FASTSAM_CKPT = os.path.join(_M, "rgb_fastsam_x.pt")     # modaliteden bagimsiz
GOMME_YOLU   = os.path.join(_M, "webssl_dino300m_light2b")
GOMME_PATCH  = 14
SAM2_BP_CKPT = os.path.join(_M, "sam2.1_hiera_base_plus.pt")
DAM4SAM_REPO = os.path.join(ROOT, "third_party", "DAM4SAM")
HQSAM_CKPT   = os.path.join(_M, "sam_hq_vit_l.pth")
HQSAM_TYPE   = "vit_l"

# ── kare / bolme ───────────────────────────────────────────────
MAXSIDE      = 1024
FASTSAM_CONF = 0.10
FASTSAM_IOU  = 0.85
MIN_ORAN     = 0.00025
AYRIK        = True          # bir piksel = TEK nesne

# ── temas grafigi ──────────────────────────────────────────────
IC_ESIK      = 0.75
KOM_UST      = 5
# Komsuluk olcutu: komsunun EN UZAK pikselinin adaya uzakligi / adanin kutu
# kosegeni. Merkez uzakligi YANLISTI: uzun bir serit bir ucuyla degip
# "yakin" cikiyor, birlesince kutu bambaska yere uzuyordu.
KOM_UZAK     = 1.5
KOM_SIMETRIK = False         # acik olursa elenen serit kendini geri sokuyor

# ── turlu arama ────────────────────────────────────────────────
UST        = 5
HAM_TAVAN  = 512
TUR0_TAVAN = HAM_TAVAN       # termalde tur 0'da kucultme YOK (olculdu, gereksiz)
GEN        = 8
BUYUME     = 0.03
SABIR      = 2
MAKS_TUR   = 6

# ── referans bankasi ───────────────────────────────────────────
# YALNIZ GRI kazandi (%88.9 vs renkli+gri %88.5 vs yalniz renkli %84.6).
# Termal kare zaten gri; renkli referans bilgi katmiyor, karistiriyor.
# CLAHE ZARARLI (%82.6). Ters-gri berabere ama katki saglamadi.
BANKA = "gri"                # "gri" | "renkli" | "renkli+gri" | "+ters" | "+clahe"

# ── politika ───────────────────────────────────────────────────
GONDER  = 0.24    # kutu gonderme esigi (SADECE_TAKIP acikken kullanilmaz)
CIPA    = 0.45    # SAM'i cipalamak icin gereken skor  (0.42 -> 26 yanlis, 0.30 -> 37)
ARDISIK = 4       # kac ardisik TUTARLI tespit         (3 -> 36 yanlis, 2 -> 43)
TUTARLI = 1.0     # ardisik tespitlerin merkez mesafesi / kutu kosegeni siniri
DOGRULA = 5       # TAKIP sirasinda her N karede eslestirici denetim yapar
BOZ     = 2       # kac kez ust uste uyusmazsa kilit acilir
UYUS    = 0.30    # denetimde SAM ile matcher kutusunun en az IoU'su
SADECE_TAKIP = True   # SAM cipalanmadan HICBIR kutu gonderilmez
                      # (acik: 201 dogru/6 yanlis · kapali: 225 dogru/22 yanlis)

TAKIPCI = "sam21pp-B"   # base_plus; large'i gecti, small/tiny 13 puan geride
DEVICE  = "cuda"
GOMME_YARI = bool(int(os.environ.get("GOMME_YARI", "1")))   # bf16: 3x hizli, sonuc AYNI
