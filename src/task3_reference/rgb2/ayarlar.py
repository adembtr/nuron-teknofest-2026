#!/usr/bin/env python3
"""GOREV 3 · RGB v2 — yollar ve sabitler.
Boru hatti (M16, 2026-09-01): FastSAM-x -> ayrik bolme -> TEMAS BUYUMESI ->
C-RADIOv3-H gomme -> yeniden siralama -> TEK SAM kilidi (skor+marj kurali) ->
DAM4SAM (SAM2.1 large) takibi + ucuz dogrulama + kapsam uyumu.
Ayrintili gerekce: SISTEM.md ve NEDEN_ELENDI.md (nuron_referance_rgb/).
"""
import os
from src.common.config import ROOT

_M = os.path.join(ROOT, "models")

FASTSAM_CKPT = os.path.join(_M, "rgb_fastsam_x.pt")
GOMME_YOLU   = os.path.join(_M, "webssl_dino300m_light2b")   # (eski omurga, artik kullanilmiyor)
# C-RADIOv3-H (NVIDIA) — INTERNETSIZ yerel yukleme: hub reposu + checkpoint diskte.
RADIO_REPO   = os.path.join(_M, "radio_repo")
RADIO_CKPT   = os.path.join(_M, "rgb_radio_v3h.pth.tar")
SAM2_BP_CKPT = os.path.join(_M, "sam2.1_hiera_base_plus.pt")
SAM2_L_CKPT  = os.path.join(_M, "sam2.1_hiera_large.pt")
DAM4SAM_REPO = os.path.join(ROOT, "third_party", "DAM4SAM")
HQSAM_CKPT   = os.path.join(_M, "sam_hq_vit_l.pth")

# ── kare / bolme ───────────────────────────────────────────────
MAXSIDE   = 1024      # kare uzun kenari (FastSAM zaten 1024'te calisiyor)
FASTSAM_CONF = 0.10
FASTSAM_IOU  = 0.85
MIN_ORAN  = 0.00025   # en kucuk maske = kare alaninin %0.025'i
AYRIK     = True      # bir piksel = TEK nesne (FastSAM ust uste boyuyor)

# ── temas grafigi ──────────────────────────────────────────────
IC_ESIK   = 0.75      # ic-ice maskeler komsu sayilmaz
KOM_UST   = 5         # dugum basina EN YAKIN komsu sayisi

# ── turlu arama ────────────────────────────────────────────────
UST         = 5       # her turda genisletilen aday sayisi (CESITLI secim)
TUR0_TAVAN  = 224     # tur 0 = ucuz tarama (yalnizca siralama)
HAM_TAVAN   = 336     # referans BANKASI gomme cozunurlugu (Z3 bununla olculdu;
                      #  512 denenmedi degil — dev motor HAM_PX=336 kullaniyor)
GEN         = 8       # kutu/maske genisletme (px)
BUYUME      = 0.03    # birlesim maskeyi <%3 buyutuyorsa gomme
SABIR       = 2       # kazanan 2 tur degismezse dur
MAKS_TUR    = 6

# ── politika ───────────────────────────────────────────────────
DEVICE  = "cuda"
TAKIPCI = "sam21pp-L"   # SAM2.1 large (RADIO omurgasiyla base+'tan iyi)

# ══ M16 KURALLARI ══════════════════════════════════════════════
# Referans TERMAL ise (R=G=B, kanal farki ~0) adaylar da griye cevrilerek gomulur;
# termalde kosinus dagilimi SIKISIK oldugu icin TUM marj/kazanc esikleri ~1/3'e iner.
# Kanit: RGB icin dogru olan TEMAS_ART=0.020 termalde birlesmeyi durdurdu (%78 -> %0).
# ══ Z3 KURALLARI (2026-09-02) — iki veri setinin 1342 etiketli karesiyle secildi ══
# Nesneye OZEL hicbir deger yok; her esik 16 pencerenin COGUNLUGUNDA en iyilendi.
# RAKIPSIZ: karede FARKLI nesneden rakip yoksa marj OLCULEMEZ — eskiden "sonsuz marj"
#   serbest gecisti ve bosa gonderimin ana kaynagiydi (bos pencerelerde 51 -> 1).
#   Olculdu: rakipsiz karede RGB 0.74 -> dogru %57 gecer, yanlis %0; GRI 0.78 -> %71/%9.
# TERMAL (gri): kosinus dagilimi sikisik (dogru medyan 0.750 / YANLIS 0.740) — mutlak
#   skor AYIRMAZ. Skoru dusur (0.62), FARKLI nesneye karsi net marj (0.06) ve UST USTE
#   3 arama karesi iste. N=2: 1 dogru/4 yanlis · N=3: 4 dogru/0 yanlis (olculdu).
#   KILIT_TEK 0.90 = fiilen kapali (en yuksek DOGRU termal skor 0.869; karari N versin).
#   BITTI kapali: yanlis kilit "aralik bitti" ilan edip Nesne_12'nin 116 karesini yakti.
KURAL = {
    "rgb": dict(GONDER=0.66, GONDER_MARJ=0.05, KILIT_SKOR=0.66, KILIT_MARJ=0.06,
                KILIT_TEK=0.80, KILIT_N=2, RAKIPSIZ=0.74, BITTI=True,
                TEMAS_ART=0.020, W_RENK=0.10),
    "gri": dict(GONDER=0.78, GONDER_MARJ=0.05, KILIT_SKOR=0.62, KILIT_MARJ=0.06,
                KILIT_TEK=0.90, KILIT_N=3, RAKIPSIZ=0.78, BITTI=False,
                TEMAS_ART=0.005, W_RENK=0.00),
}
KILIT_N   = 2         # (geriye uyum — asil deger artik KURAL[kip]["KILIT_N"])
CIPA_COS  = 0.25      # kilit aninda SAM maskesinin referansa benzerlik tabani
SKOR_MIN  = 0.35      # aday alt siniri
EB_TOL    = 2.5       # en/boy toleransi (referansa gore)
# ══ M19 KURALLARI (2026-09-18) — folder0/folder1 (916 etiketli kare) + 2026 ile secildi; YALNIZ RGB REFERANS.
# Termal (gri) dali DEGISMEDI. Uc kural, tek tek ve birlesik olculdu; 2026'da 10/10 pencere birebir ayni.
#  EB_ALT: en/boy kapisi ASIMETRIK. Referans yerden/egik cekildiyse gorunen en/boy GERCEK orandan yalniz
#   BUYUK olabilir (perspektif bir ekseni sikistirir); tepeden bakan drone en az sikismis bakistir. Aday
#   referanstan EB_TOL kat UZUN olamaz (eski), ama EB_ALT kata kadar KISA olabilir. Kompakt referansta
#   (eb < EB_ALT) alt sinir <1 -> etkisiz. Olculdu: f1 ref2 tespit %19->%100, kutu %0->%100; diger 26 pencere ayni.
#  KUCUL_MARJ_RGB: kilit sonrasi kapsam KUCULTME adimi fiilen kapali (0.99). "Dogru bulup kesip birakma"nin
#   kaynagi bu adimdi (4 sette 147 kayip kare). Olculdu: f1 ref8 kutu %15->%88, ref5 %82->%94, f0 ref4 %67->%97.
#  KARSIT_KENAR: gonderilecek kutu karenin IKI KARSIT kenarina birden degiyorsa (sol+sag ya da ust+alt)
#   nesne degil zemin/cikis sizintisidir -> gonderme; takipteyse kilidi birak. Olculdu: f0 ref6 etiketsiz
#   kuyrukta bosa gonderim 30->3, kutu %67->%84; bedeli tespit %100->%96.
EB_ALT          = 8.0
KUCUL_MARJ_RGB  = 0.99
KARSIT_KENAR    = True
ALAN_MIN  = 500       # en kucuk aday alani (px, 1024 olceginde)
UST_ADAY  = 10
ARA_PX    = 512       # arama karesinde FastSAM cozunurlugu (hiz)
# COZUNURLUK KADEMESI — nesnenin karedeki PIKSEL boyu FastSAM'in onu ayirabilmesini
# belirler: 512'de buyukler kusursuz, kucukler kaybolur (Nesne_12 %0); 768'de tersi.
# Cozunurluk PENCERENIN ozelligidir: ust uste ARTIR_N arama karesinde kilitlenilemezse
# kademe YUKSELIR ve orada kalir; kare basina hep TEK model kosusu vardir.
# ARTIR_N=60 OLCULDU: dogru kilit gecikmesi ort 10.9 / medyan 6 / EN BUYUK 56 kare —
# esik dagilimin USTUNDE olmali. 10 denendi ve Nesne_10'u tam kilit karesinde bozdu
# (768'e gecis ham maskeyi 74->119 firlatti, skor 0.70->0.65 dustu, kilit hic olusmadi).
ARTIR_PX  = [512, 768]
ARTIR_N   = 60
# M20 IKI KADEME AYNI KAREDE (2026-09-19, yalniz RGB ve kademe 0): arama karesinde 512'nin tepe skoru
# IKI_ESIK'in altindaysa AYNI KAREDE ARA_PX2 kosulur, tepe skoru yuksek olan kullanilir. PARCA KORUMASI:
# 768'in tepesi 512'nin tepesinin ICINDE ve alani onun IKI_PARCA katindan kucukse "butun yerine parca"dir
# -> 768 sonucu REDDEDILIR. Olculdu: sabit 768 f0 ref4 kutu %97->%27 bozdu (parca .81 > butun .60); bu
# bicim f1 ref9 tespit 3->27 kutu 0->21; f0 ref4/ref7, f1 ref2/3/4, 2026 N04 degismedi, bosa 0.
# Maliyet yalniz 512'nin kaybettigi arama karelerinde (ikinci FastSAM kosusu).
IKI_KADEME_ARA = True
ARA_PX2        = 768
IKI_ESIK       = 0.70
IKI_PARCA      = 0.60
# M19_GRI (2026-09-19): M19 uclusu (EB_ALT asimetrik kapi, kucultme kapali, karsit-kenar bekcisi) GRI
# referansta da. Olculdu: gri FOTO f1 ref8 kutu 7->42, ref5 14->16; termal (f3 ref7/ref10, 2026 N04) ayni.
M19_GRI        = True
TUR0_PX   = 224       # gomme cozunurlugu
ARA_ADIM  = 2         # aday bulunamayan karede arama atlama
# temas buyumesi
TEMAS_D   = 4         # en fazla kac parca eklenir
TEMAS_KOM = 6         # adim basina denenecek en fazla komsu
TEMAS_PX  = 5         # degme toleransi
BIR_TOHUM = 3         # mekansal olarak AYRIK tohum sayisi
BIR_MAKS  = 0.35      # birlesim kare alaninin bu oranini asamaz
IKILI     = True      # tek parca yetmezse ikili sicrama
IKILI_UST = 4
DOLU_TOL  = 3.0       # referans dolulugundan sapma siniri (zemin birlesimi kapisi)
# bolme (uzun tohumu ikiye ayir)
BOL       = True
BOL_MIN   = 0.02
# yeniden siralama
RER_K     = 7
W_SEKIL   = 0.10
# kilit sonrasi
DEN_HER     = 2       # kac karede bir SAM maskesi gomulup dogrulanir (TEK gomme, ucuz)
DENETIM_N   = 12      # SERBEST donemde kac karede bir model kosup denetlenir (pahali)
# duzeltme penceresi (kilitten hemen sonra model DUZELT kare boyunca SAM'i oylar)
DUZELT      = 10      # pencere uzunlugu (kare)
DUZ_ADIM    = 2       # pencerede kac karede bir model kosar (10/2 = 5 ornek)
UYUM        = 0.50    # model-SAM IoU onay esigi
OY_ORAN     = 4.0     # merkez onayinda SAM/model alan orani siniri (dev kutu istisnasi)
KARAR_ORAN  = 1.5     # ardisik model kutulari bu kat icindeyse olcum KARARLI sayilir
ERKEN_N     = 2       # ust uste (kararli + SAM savunmasini asan) red -> kilit atilir
SAM_MARJ    = 0.05    # modelin SAM'in kendi benzerligini gecmesi gereken fark
DEN_RED     = 2       # periyodik denetimde ust uste red siniri
DEN_BOS     = 3       # denetimde model hic aday bulamazsa (bos) sinir
# kenar-giris kurali: kilit KENARDA dogdu ve nesne iceri girdiyse kisa dogrulama
YEN_KARE    = 3       # denetim penceresi (kare)
UYUM_MIN    = 2       # pencerede en az kac onay
DEN_COS     = 0.30    # bu benzerligin altinda supheli
DEN_COS_RED = 2       # ust uste bu kadar supheli olcum -> kilit birak
KAP_ORAN    = 1.30    # kapsam uyusmazligi orani
KAP_SKOR    = 0.65    # modelin o kutuya guveni
KAP_ES      = 0.50    # iki dogrulama olcumunun birbirine IoU uyumu
KUCUL_MARJ  = 0.05    # KUCULTMEK icin modelin SAM'i gecmesi gereken fark
CIFT_COZ    = True    # kilit aninda tek seferlik tam cozunurluk kontrolu
CIFT_ES     = 0.50    # takas icin kapsam benzerligi sarti
# bekciler
BEKCI_ORAN = 1.8
KUCUL_N    = 3
TEPE_ORAN  = 0.05
BITTI_YAS  = 15
KAPAT      = 0.15     # maske tamamlama (morfolojik kapama) orani
KENAR_PX   = 3
# eski v1 uyumlulugu (disaridan import edenler icin)
ESIK    = 0.60
ARDISIK = 2
